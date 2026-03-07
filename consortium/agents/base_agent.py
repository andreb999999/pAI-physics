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


MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "anthropic/claude-opus-4-6": 200_000,
    "anthropic/claude-sonnet-4-6": 200_000,
    "gpt-5": 256_000,
    "gpt-5-mini": 256_000,
    "gpt-5-nano": 256_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.3-codex": 200_000,
    "gpt-4o": 128_000,
    "o3-2025-04-16": 200_000,
    "o4-mini-2025-04-16": 128_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-3.0-pro": 2_000_000,
    "gpt-5.2": 256_000,
    "deepseek-chat": 64_000,
    "deepseek-coder": 64_000,
    "grok-4-0709": 128_000,
}


def get_context_limit(model_id: str) -> int:
    return MODEL_CONTEXT_LIMITS.get(model_id, 128_000)


def _unwrap_model(model: Any) -> Any:
    """Unwrap BudgetedLiteLLMModel to get the underlying BaseChatModel."""
    from ..budget import BudgetedLiteLLMModel
    if isinstance(model, BudgetedLiteLLMModel):
        return model.model
    return model


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

    react_agent = create_react_agent(
        model=_unwrap_model(model),
        tools=tools,
        prompt=system_prompt,
    )

    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")

        result = react_agent.invoke({
            "messages": [HumanMessage(content=task)],
        })

        last_msg = result["messages"][-1] if result.get("messages") else None
        output = last_msg.content if last_msg and hasattr(last_msg, "content") else str(last_msg)

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
        }

    node_fn.__name__ = agent_name
    return node_fn
