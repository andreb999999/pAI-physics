import logging
import os
import yaml
import functools

logger = logging.getLogger(__name__)

# Custom parameter filtering function for model-specific requirements
def filter_model_params(original_func):
    """Decorator to filter unsupported parameters for different models."""
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        model = kwargs.get('model', args[0] if args else '')

        # GPT-5/Codex specific filtering
        if isinstance(model, str) and (model.startswith('gpt-5') or "codex" in model):
            # Remove unsupported GPT-5/Codex parameters
            unsupported_params = {
                'stop', 'temperature', 'top_p', 'presence_penalty',
                'frequency_penalty', 'logprobs', 'top_logprobs',
                'logit_bias', 'max_tokens'
            }

            # Filter out unsupported parameters
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in unsupported_params}

            # Replace max_tokens with max_completion_tokens if present
            if 'max_tokens' in kwargs:
                filtered_kwargs['max_completion_tokens'] = kwargs['max_tokens']

            return original_func(*args, **filtered_kwargs)

        # ---- Anthropic / Claude branch ---------------------------------------
        elif isinstance(model, str) and ("claude" in model or "anthropic" in model):
            fk = kwargs.copy()

            # Translate legacy "effort" param to litellm-supported "reasoning_effort"
            if "effort" in fk and "reasoning_effort" not in fk:
                fk["reasoning_effort"] = fk.pop("effort")
            elif "effort" in fk:
                fk.pop("effort")

            # Don't send both temperature + top_p to Claude
            if "temperature" in fk and "top_p" in fk:
                fk.pop("top_p")

            # Normalize extended thinking:
            # Accept user-friendly `budget_tokens` and convert to Anthropic `thinking`
            budget = fk.pop("budget_tokens", None)
            thinking = fk.get("thinking")

            if budget is not None:
                if budget <= 0:
                    # Explicitly disable thinking if budget_tokens <= 0
                    fk["thinking"] = {"type": "disabled"}
                else:
                    fk["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
            elif thinking is None:
                # If neither budget_tokens nor thinking provided, do nothing.
                pass

            # Enforce invariant: max_tokens > thinking.budget_tokens
            # (If thinking is enabled, make sure max_tokens is large enough.)
            if isinstance(fk.get("thinking"), dict) and fk["thinking"].get("type") == "enabled":
                budget_tokens = int(fk["thinking"].get("budget_tokens", 0))
                # margin gives the model room to write the final answer after reasoning
                margin = 2048
                mt = fk.get("max_tokens")
                if mt is None or int(mt) <= budget_tokens:
                    fk["max_tokens"] = int(budget_tokens + margin)

            return original_func(*args, **fk)

        else:
            # Other models use original parameters
            return original_func(*args, **kwargs)
    return wrapper

def _validate_config(config):
    """Warn about missing or malformed config sections. Does not block execution."""
    if not isinstance(config, dict):
        logger.warning("LLM config is not a dict — ignoring")
        return
    if "main_agents" in config:
        ma = config["main_agents"]
        if not isinstance(ma, dict):
            logger.warning("main_agents must be a dict in .llm_config.yaml")
        elif "model" not in ma:
            logger.warning("main_agents.model is not set in .llm_config.yaml")
    if "budget" in config:
        b = config["budget"]
        if not isinstance(b, dict):
            logger.warning("budget must be a dict in .llm_config.yaml")
        elif "usd_limit" not in b:
            logger.warning("budget.usd_limit is not set — cost enforcement will be disabled")
        elif not b.get("pricing"):
            logger.warning("budget.pricing is empty — cost enforcement will fail for unknown models")


def load_llm_config():
    """Load LLM configuration from .llm_config.yaml if it exists."""
    config_path = ".llm_config.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Loaded LLM config from %s", config_path)
            _validate_config(config)
            return config
        except yaml.YAMLError as e:
            logger.warning("Error loading %s: %s", config_path, e)
            return None
    else:
        logger.info("No %s found, using CLI arguments", config_path)
        return None
