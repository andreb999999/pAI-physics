"""
Base agent factory — LangGraph migration.

Replaces BaseResearchAgent (smolagents CodeAgent subclass) with a factory
that creates a LangGraph create_react_agent node.

Each specialist agent module exposes:
  - get_tools(workspace_dir, model_id)  -> list[BaseTool]
  - build_node(model, workspace_dir, authorized_imports, **cfg) -> Callable
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from ..models import get_context_limit  # noqa: F401 — re-exported for backward compat


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


def create_specialist_agent(
    model: Any,
    tools: List[BaseTool],
    system_prompt: str,
    agent_name: str,
    workspace_dir: Optional[str] = None,
) -> Callable:
    """
    Build a LangGraph ReAct agent node for a specialist.

    The returned callable accepts a ResearchState dict and returns a state
    update dict suitable for use as a LangGraph node.
    """
    if workspace_dir:
        os.makedirs(os.path.join(workspace_dir, agent_name), exist_ok=True)

    budget_callback = _extract_budget_callback(model)

    react_agent = create_react_agent(
        model=_unwrap_model(model),
        tools=tools,
        prompt=system_prompt,
    )

    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")

        invoke_config = {"callbacks": [budget_callback]} if budget_callback else None
        result = react_agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config=invoke_config,
        )

        last_msg = result["messages"][-1] if result.get("messages") else None
        output = last_msg.content if last_msg and hasattr(last_msg, "content") else str(last_msg)

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    node_fn.__name__ = agent_name
    return node_fn
