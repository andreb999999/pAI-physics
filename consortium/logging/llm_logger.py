"""
LLM Logging Infrastructure for the LangGraph research pipeline.

Provides a LangChain BaseCallbackHandler that logs all LLM calls made by any
agent node to a workspace-scoped JSONL file for debugging and prompt analysis.

Usage:
    handler = create_agent_logging_handler(workspace_dir=workspace_dir)
    # Pass via graph config: graph.invoke(state, config={"callbacks": [handler]})
    # Or per-agent: model.invoke(messages, config={"callbacks": [handler]})
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class ResearchLLMLogger(BaseCallbackHandler):
    """
    LangChain callback handler that logs every LLM call to a workspace JSONL file.

    Captures:
    - Input messages (serialized)
    - Response content + token usage
    - Timing information
    - Rate-limit retry events (via on_retry)
    """

    def __init__(self, log_file_path: str, agent_name: str = "unknown"):
        super().__init__()
        self.log_file_path = log_file_path
        self.agent_name = agent_name
        self._call_start: Dict[str, float] = {}
        self._call_inputs: Dict[str, Any] = {}
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id)
        self._call_start[call_id] = time.time()
        self._call_inputs[call_id] = {
            "model": serialized.get("name", "unknown"),
            "messages": [
                [{"role": m.__class__.__name__, "content": str(m.content)} for m in batch]
                for batch in messages
            ],
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id)
        duration_ms = int((time.time() - self._call_start.pop(call_id, time.time())) * 1000)
        inputs = self._call_inputs.pop(call_id, {})

        generations = []
        for batch in response.generations:
            for gen in batch:
                generations.append(getattr(gen, "text", str(gen)))

        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]

        entry = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "agent_name": self.agent_name,
            "status": "success",
            "input": inputs,
            "output": {
                "generations": generations,
                "token_usage": token_usage,
                "duration_ms": duration_ms,
            },
        }
        self._write(entry)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        call_id = str(run_id)
        duration_ms = int((time.time() - self._call_start.pop(call_id, time.time())) * 1000)
        inputs = self._call_inputs.pop(call_id, {})

        entry = {
            "call_id": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "agent_name": self.agent_name,
            "status": "error",
            "input": inputs,
            "output": {"error": str(error), "duration_ms": duration_ms},
        }
        self._write(entry)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"Warning: Failed to write LLM log entry: {exc}")


# ---------------------------------------------------------------------------
# Factory helper (called by create_specialist_agent and manager node)
# ---------------------------------------------------------------------------

def create_agent_logging_handler(
    workspace_dir: str,
    agent_name: str = "unknown",
) -> ResearchLLMLogger:
    """
    Return a callback handler that logs to <workspace_dir>/agent_llm_calls.jsonl.
    Pass it in the LangChain/LangGraph config dict:
        config={"callbacks": [handler]}
    """
    log_path = os.path.abspath(os.path.join(workspace_dir, "agent_llm_calls.jsonl"))
    return ResearchLLMLogger(log_file_path=log_path, agent_name=agent_name)
