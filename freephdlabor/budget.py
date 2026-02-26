import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


class BudgetExceededError(RuntimeError):
    """Raised when the configured USD budget has been exhausted."""


class BudgetManager:
    """
    Tracks cumulative spend and enforces a hard USD limit.

    Notes:
    - Uses model-reported token_usage (prompt_tokens/completion_tokens) when available.
    - Requires per-model pricing to be provided by the user (input/output per 1K tokens).
    - Writes a lock file when the limit is reached to prevent further calls.
    """

    def __init__(
        self,
        usd_limit: float,
        pricing: Dict[str, Dict[str, float]],
        state_path: str,
        ledger_path: str,
        lock_path: str,
        hard_stop: bool = True,
        fail_closed: bool = True,
    ) -> None:
        self.usd_limit = float(usd_limit)
        self.pricing = pricing or {}
        self.state_path = state_path
        self.ledger_path = ledger_path
        self.lock_path = lock_path
        self.hard_stop = hard_stop
        self.fail_closed = fail_closed
        self.total_usd = 0.0
        self.by_model: Dict[str, float] = {}
        self._load_state()

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.total_usd = float(data.get("total_usd", 0.0))
            self.by_model = data.get("by_model", {}) or {}
        except Exception:
            # If state is corrupted, reset to avoid blocking usage
            self.total_usd = 0.0
            self.by_model = {}

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        data = {
            "usd_limit": self.usd_limit,
            "total_usd": round(self.total_usd, 6),
            "by_model": self.by_model,
            "last_updated": datetime.utcnow().isoformat() + "Z",
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _write_ledger(self, entry: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _normalize_model_id(self, model_id: str) -> str:
        if not model_id:
            return ""
        # Prefer provider-qualified exact match, but allow fallback to bare model name
        if model_id in self.pricing:
            return model_id
        if "/" in model_id:
            return model_id.split("/")[-1]
        return model_id

    def _get_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        if not model_id:
            return None
        if model_id in self.pricing:
            return self.pricing[model_id]
        normalized = self._normalize_model_id(model_id)
        return self.pricing.get(normalized)

    def _compute_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._get_pricing(model_id)
        if not pricing:
            if self.fail_closed:
                raise BudgetExceededError(
                    f"Budget enforcement: no pricing configured for model '{model_id}'. "
                    f"Add pricing to .llm_config.yaml under budget.pricing."
                )
            return 0.0
        input_per_1k = float(pricing.get("input_per_1k", 0.0))
        output_per_1k = float(pricing.get("output_per_1k", 0.0))
        return (prompt_tokens / 1000.0) * input_per_1k + (completion_tokens / 1000.0) * output_per_1k

    def check_budget(self) -> None:
        if not self.hard_stop:
            return
        if os.path.exists(self.lock_path):
            raise BudgetExceededError(
                f"Budget lock file present: {self.lock_path}. "
                f"USD limit {self.usd_limit} reached."
            )
        if self.total_usd >= self.usd_limit:
            raise BudgetExceededError(
                f"USD budget limit reached: {self.total_usd:.4f} / {self.usd_limit:.2f}."
            )

    def record_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        call_id: Optional[str] = None,
    ) -> float:
        cost = self._compute_cost(model_id, prompt_tokens, completion_tokens)
        self.total_usd += cost
        self.by_model[model_id] = round(self.by_model.get(model_id, 0.0) + cost, 6)
        self._save_state()

        entry = {
            "call_id": call_id or str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_id": model_id,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": round(cost, 6),
            "total_usd": round(self.total_usd, 6),
            "usd_limit": self.usd_limit,
        }
        self._write_ledger(entry)

        # Also track run-scoped cumulative tokens used by terminal step summaries.
        try:
            from .token_usage_tracker import record_token_usage

            record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source="budgeted_model",
                model_id=model_id,
            )
        except Exception:
            # Never let token-tracker errors break budget accounting.
            pass

        if self.hard_stop and self.total_usd >= self.usd_limit:
            # Create lock file to prevent further calls
            os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
            with open(self.lock_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Budget exceeded: {self.total_usd:.4f} / {self.usd_limit:.2f} USD\n"
                )
        return cost


class BudgetedLiteLLMModel:
    """
    Wrapper that enforces a BudgetManager on all model calls.
    """

    def __init__(self, model, budget_manager: BudgetManager):
        self.model = model
        self.budget_manager = budget_manager

    def _get_model_id(self) -> str:
        if hasattr(self.model, "model_id"):
            return self.model.model_id
        if hasattr(self.model, "model"):
            return self.model.model
        return "unknown_model"

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except Exception:
            return 0

    @classmethod
    def _extract_usage_from_obj(cls, usage_obj: Any) -> Optional[Dict[str, int]]:
        if not usage_obj:
            return None

        def _get(*keys: str) -> int:
            for key in keys:
                # dict-style usage
                if isinstance(usage_obj, dict) and key in usage_obj:
                    value = cls._safe_int(usage_obj.get(key))
                    if value:
                        return value
                # object-style usage
                if hasattr(usage_obj, key):
                    value = cls._safe_int(getattr(usage_obj, key))
                    if value:
                        return value
            return 0

        # Support both conventions:
        # - prompt/completion (OpenAI-style usage)
        # - input/output (smolagents TokenUsage wrapper)
        prompt_tokens = _get("prompt_tokens", "input_tokens")
        completion_tokens = _get("completion_tokens", "output_tokens")
        total_tokens = _get("total_tokens")
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        # Cost accounting requires an input/output split. If unavailable, fail closed upstream.
        if prompt_tokens == 0 and completion_tokens == 0:
            return None

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _extract_token_usage(self, response) -> Optional[Dict[str, int]]:
        # 1) smolagents ChatMessage.token_usage
        usage = self._extract_usage_from_obj(getattr(response, "token_usage", None))
        if usage:
            return usage

        # 2) direct response.usage
        usage = self._extract_usage_from_obj(getattr(response, "usage", None))
        if usage:
            return usage

        # 3) nested raw provider response (e.g., ChatMessage.raw.usage)
        raw = getattr(response, "raw", None)
        usage = self._extract_usage_from_obj(getattr(raw, "usage", None))
        if usage:
            return usage

        return None

    def generate(self, messages, **kwargs):
        self.budget_manager.check_budget()
        response = self.model.generate(messages, **kwargs)
        usage = self._extract_token_usage(response)
        if not usage:
            # Fail closed after the call if token usage isn't reported
            if self.budget_manager.fail_closed:
                raise BudgetExceededError(
                    "Budget enforcement: model response did not include token_usage. "
                    "Cannot account for cost; refusing further calls."
                )
            return response
        model_id = self._get_model_id()
        self.budget_manager.record_usage(
            model_id=model_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        return response

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)
