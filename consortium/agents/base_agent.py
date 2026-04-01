"""
Base agent factory — LangGraph migration.

Agent factory that creates LangGraph create_react_agent nodes.

Each specialist agent module exposes:
  - get_tools(workspace_dir, model_id)  -> list[BaseTool]
  - build_node(model, workspace_dir, authorized_imports, **cfg) -> Callable
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from ..models import get_context_limit  # noqa: F401 — re-exported for backward compat

logger = logging.getLogger(__name__)


def _unwrap_model(model: Any) -> Any:
    """Unwrap BudgetedLiteLLMModel to get the underlying BaseChatModel."""
    from ..budget import BudgetedLiteLLMModel
    if isinstance(model, BudgetedLiteLLMModel):
        return model.model
    return model


def _extract_budget_callback(model: Any) -> Any:
    """Return a BudgetTrackingCallback if model is budget-wrapped, else None."""
    from ..budget import BudgetedLiteLLMModel, BudgetTrackingCallback
    if isinstance(model, BudgetedLiteLLMModel):
        return BudgetTrackingCallback(
            model.budget_manager,
            model_id=model._get_model_id(),
        )
    return None


def _extract_text(content: Any) -> str:
    """Extract text from LLM message content, handling both string and Anthropic content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def create_specialist_agent(
    model: Any,
    tools: List[BaseTool],
    system_prompt: str,
    agent_name: str,
    workspace_dir: Optional[str] = None,
) -> Callable[[dict], dict]:
    """
    Build a LangGraph ReAct agent node for a specialist.

    The returned callable accepts a ResearchState dict and returns a state
    update dict suitable for use as a LangGraph node.
    """
    if workspace_dir:
        os.makedirs(os.path.join(workspace_dir, agent_name), exist_ok=True)

    # Budget is now recorded automatically by the monkey-patched litellm.completion()
    # in config.py — no need for BudgetTrackingCallback on invoke.

    react_agent = create_react_agent(
        model=_unwrap_model(model),
        tools=tools,
        prompt=system_prompt,
    )

    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")

        from ..logging.training_data_logger import set_current_agent_name
        set_current_agent_name(agent_name)
        try:
            result = react_agent.invoke(
                {"messages": [HumanMessage(content=task)]},
            )
        except Exception as e:
            logger.error("Agent '%s' failed during invoke: %s", agent_name, e, exc_info=True)
            output = f"[AGENT_ERROR: {agent_name}: {type(e).__name__}: {e}]"
            return {
                "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
                "agent_task": None,
            }
        finally:
            set_current_agent_name(None)

        last_msg = result["messages"][-1] if result.get("messages") else None
        output = _extract_text(getattr(last_msg, "content", None)) if last_msg else ""

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    node_fn.__name__ = agent_name
    return node_fn
