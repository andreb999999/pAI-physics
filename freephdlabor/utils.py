import json
import os
import re
from typing import Optional
from langchain_community.chat_models import ChatLiteLLM

# Available models
AVAILABLE_MODELS = [
    # OpenAI GPT-5 models (latest)
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.3-codex",
    # OpenAI models
    "gpt-4o",
    "gpt-4.1-mini-2025-04-14",
    # OpenAI reasoning models
    "o4-mini-2025-04-16",
    "o3-2025-04-16",
    "o3-pro-2025-06-10",
    # Claude models (Anthropic)
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    # grok models
    "grok-4-0709",
    # Google Gemini models
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


def extract_content_between_markers(response: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Extract content between specified start and end markers from a response string."""
    try:
        start_escaped = re.escape(start_marker)
        end_escaped = re.escape(end_marker)
        pattern = f"{start_escaped}(.*?){end_escaped}"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None
    except Exception:
        return None


def create_model(
    model_name,
    reasoning_effort="medium",
    verbosity="medium",
    budget_tokens=None,
    budget_config=None,
    budget_dir=None,
    effort=None,
):
    """Create a ChatLiteLLM model from the given model name using API keys from environment.

    Args:
        model_name:       Name of the model to create
        reasoning_effort: GPT-5 reasoning effort level (minimal, low, medium, high)
        verbosity:        GPT-5 verbosity level (low, medium, high)
        budget_tokens:    Claude Extended Thinking token budget (min: 1024)
        effort:           Claude 4.x effort level (low, medium, high, max)
    """
    model_kwargs: dict = {}

    if "claude" in model_name:
        if budget_tokens is not None:
            model_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        if effort is not None and model_name in {"claude-opus-4-6", "claude-sonnet-4-6"}:
            model_kwargs["effort"] = effort

        base_model = ChatLiteLLM(
            model=f"anthropic/{model_name}",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_kwargs=model_kwargs if model_kwargs else None,
        )

    elif "codex" in model_name:
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            model_kwargs={"reasoning_effort": reasoning_effort},
        )

    elif model_name.startswith("gpt-5"):
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            model_kwargs={"reasoning_effort": reasoning_effort, "verbosity": verbosity},
        )

    elif "gpt" in model_name or model_name.startswith(("o1-", "o3-", "o4-")):
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    elif "deepseek" in model_name:
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            api_base="https://api.deepseek.com",
        )

    elif "llama" in model_name:
        base_model = ChatLiteLLM(
            model=f"openrouter/{model_name}",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    elif "gemini" in model_name:
        if "gemini-2.5-pro" in model_name:
            model_kwargs["thinking_budget"] = 32768
        base_model = ChatLiteLLM(
            model=f"gemini/{model_name}",
            api_key=os.environ["GOOGLE_API_KEY"],
            model_kwargs=model_kwargs if model_kwargs else None,
        )

    elif "grok" in model_name:
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ.get("XAI_API_KEY", ""),
        )

    else:
        base_model = ChatLiteLLM(
            model=model_name,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )

    # Optional hard budget enforcement wrapper
    if budget_config and budget_config.get("usd_limit"):
        from .budget import BudgetManager, BudgetedLiteLLMModel
        budget_dir = budget_dir or os.getcwd()
        state_path = os.path.join(budget_dir, "budget_state.json")
        ledger_path = os.path.join(budget_dir, "budget_ledger.jsonl")
        lock_path = os.path.join(budget_dir, "budget.lock")

        manager = BudgetManager(
            usd_limit=float(budget_config["usd_limit"]),
            pricing=budget_config.get("pricing", {}),
            state_path=state_path,
            ledger_path=ledger_path,
            lock_path=lock_path,
            hard_stop=bool(budget_config.get("hard_stop", True)),
            fail_closed=bool(budget_config.get("fail_closed", True)),
        )
        return BudgetedLiteLLMModel(base_model, manager)

    return base_model


def build_research_graph(
    model,
    workspace_dir,
    essential_imports=None,
    enable_planning=False,
    planning_interval=3,
    interrupt_callback=None,
    require_pdf=False,
    enforce_paper_artifacts=False,
    require_experiment_plan=False,
    enable_math_agents=False,
    enforce_editorial_artifacts=False,
    min_review_score=8,
    pipeline_mode="default",
    followup_max_iterations=3,
    manager_max_steps=50,
):
    """
    Build and return the compiled LangGraph research pipeline.

    This is the LangGraph equivalent of the old initialize_agent_system().

    Args:
        model:                    ChatLiteLLM instance
        workspace_dir:            Absolute path to the run workspace
        essential_imports:        Authorized Python imports for code tools
        enable_planning:          Unused (kept for call-site compat)
        planning_interval:        Unused (kept for call-site compat)
        interrupt_callback:       Unused (interrupt handled via state)
        require_pdf:              Require final_paper.pdf at finish
        enforce_paper_artifacts:  Run artifact gate before FINISH
        require_experiment_plan:  Require experiments_to_run_later.md at finish
        enable_math_agents:       Include math pipeline agents
        enforce_editorial_artifacts: Run full editorial gates
        min_review_score:         Reviewer score threshold
        pipeline_mode:            "default" | "full_research" | "quick"
        followup_max_iterations:  Max follow-up loops in full_research mode
        manager_max_steps:        Max total manager iterations

    Returns:
        (compiled_graph, checkpointer) tuple.
    """
    print("Building LangGraph research pipeline...")

    from .graph import build_research_graph as _build_graph, get_default_checkpointer

    checkpointer = get_default_checkpointer(workspace_dir)

    compiled = _build_graph(
        model=model,
        workspace_dir=workspace_dir,
        pipeline_mode=pipeline_mode,
        enable_math_agents=enable_math_agents,
        enforce_paper_artifacts=enforce_paper_artifacts,
        enforce_editorial_artifacts=enforce_editorial_artifacts,
        require_pdf=require_pdf,
        require_experiment_plan=require_experiment_plan,
        min_review_score=min_review_score,
        followup_max_iterations=followup_max_iterations,
        manager_max_steps=manager_max_steps,
        authorized_imports=essential_imports,
        checkpointer=checkpointer,
    )
    print("LangGraph pipeline ready.")
    return compiled, checkpointer


# Keep old name as alias for any remaining call sites
def initialize_agent_system(*args, **kwargs):
    """Alias for build_research_graph — kept for backwards compatibility."""
    return build_research_graph(*args, **kwargs)


def save_agent_memory(manager):
    """No-op after LangGraph migration — SqliteSaver checkpointer handles persistence."""
    pass
