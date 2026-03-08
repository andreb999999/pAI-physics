"""
Model counsel — multi-model debate and synthesis for each pipeline stage.

For each stage, spawns one independent ReAct agent per counsel model, each
working in its own sandbox copy of the workspace. Their outputs feed a
text-based debate across N rounds, then the strongest model (Opus 4.6)
synthesizes a final consensus. Sandbox artifacts are merged back into the
main workspace after synthesis.

Sandbox agents and debate critiques run in parallel via ThreadPoolExecutor,
reducing wall-clock time from O(N) to roughly O(1) per phase.

Usage (via graph.py/build_node):
    from .counsel import create_counsel_node
    node = create_counsel_node(system_prompt, tools, "ideation_agent",
                               workspace_dir, counsel_models, max_debate_rounds=3)
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import litellm
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent


# Default model specs matching the 4 top-tier frontier models.
# Used for debate-phase litellm.completion calls (where we need the model name string).
DEFAULT_COUNSEL_MODEL_SPECS = [
    {"model": "claude-opus-4-6",  "effort": "max"},
    {"model": "claude-sonnet-4-6", "effort": "max"},
    {"model": "gpt-5.2",          "reasoning_effort": "high", "verbosity": "high"},
    {"model": "gemini-3.0-pro",   "thinking_budget": 65536},
]

SYNTHESIS_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_counsel_models(
    budget_config: Optional[dict] = None,
    budget_dir: Optional[str] = None,
    model_specs: Optional[List[dict]] = None,
) -> List[Any]:
    """Instantiate all counsel ChatLiteLLM models from specs."""
    from .utils import create_model
    specs = model_specs or DEFAULT_COUNSEL_MODEL_SPECS
    models = []
    for spec in specs:
        m = create_model(
            model_name=spec["model"],
            reasoning_effort=spec.get("reasoning_effort", "high"),
            verbosity=spec.get("verbosity", "medium"),
            budget_tokens=spec.get("thinking_budget"),
            effort=spec.get("effort"),
            budget_config=budget_config or {},
            budget_dir=budget_dir or os.getcwd(),
        )
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def _populate_sandbox(workspace_dir: str, sandbox_dir: str) -> None:
    """Copy current workspace into a fresh sandbox, skipping sandbox subtrees and DBs."""
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            shutil.copytree(
                workspace_dir,
                sandbox_dir,
                ignore=shutil.ignore_patterns("counsel_sandboxes", "*.db", "*.lock"),
            )
            return
        except InterruptedError:
            if attempt == max_attempts - 1:
                raise
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            time.sleep(0.2 * (2**attempt))


def _sandbox_tools(original_tools: List[BaseTool], sandbox_dir: str) -> List[BaseTool]:
    """Re-instantiate each tool pointing to sandbox_dir instead of workspace_dir."""
    result = []
    for tool in original_tools:
        try:
            cls = type(tool)
            kwargs: dict = {}
            if hasattr(tool, "working_dir"):
                kwargs["working_dir"] = sandbox_dir
            if hasattr(tool, "workspace_dir"):
                kwargs["workspace_dir"] = sandbox_dir
            if hasattr(tool, "model"):
                kwargs["model"] = tool.model
            if hasattr(tool, "model_name"):
                kwargs["model_name"] = tool.model_name
            if hasattr(tool, "authorized_imports"):
                kwargs["authorized_imports"] = tool.authorized_imports
            if hasattr(tool, "allow_accepted_transition"):
                kwargs["allow_accepted_transition"] = tool.allow_accepted_transition
            result.append(cls(**kwargs) if kwargs else tool)
        except Exception:
            result.append(tool)
    return result


def _merge_sandbox(sandbox_dir: str, workspace_dir: str) -> None:
    """Copy all files from sandbox into workspace (last sandbox wins on conflicts)."""
    for root, dirs, files in os.walk(sandbox_dir):
        dirs[:] = [d for d in dirs if d != "counsel_sandboxes"]
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, sandbox_dir)
            dst = os.path.join(workspace_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                print(f"[counsel] merge warning: failed to copy {rel}: {exc}")


# ---------------------------------------------------------------------------
# Core counsel stage runner
# ---------------------------------------------------------------------------

def run_counsel_stage(
    task: str,
    system_prompt: str,
    tools: List[BaseTool],
    workspace_dir: str,
    counsel_models: List[Any],
    agent_name: str,
    max_debate_rounds: int = 3,
    model_specs: Optional[List[dict]] = None,
) -> str:
    """
    Run multi-model counsel for one pipeline stage.

    1. Sandbox phase  — each model runs independently in its own workspace copy
                        (all N agents run in parallel via ThreadPoolExecutor).
    2. Debate phase   — models critique each other's solutions (litellm.completion);
                        all N critiques per round run in parallel.
    3. Synthesis      — Opus 4.6 produces the final consensus output.
    4. Promotion      — sandbox artifacts are merged back to the main workspace.

    Returns the consensus output string.
    """
    specs = model_specs or DEFAULT_COUNSEL_MODEL_SPECS
    sandbox_base = os.path.join(workspace_dir, "counsel_sandboxes", agent_name)
    os.makedirs(sandbox_base, exist_ok=True)

    # Pre-build sandbox dir paths (needed for merge even if agent fails)
    sandbox_dirs: List[str] = []
    for i, _ in enumerate(counsel_models):
        label = specs[i]["model"] if i < len(specs) else f"model_{i}"
        safe_label = label.replace("/", "_").replace(":", "_")
        sandbox_dirs.append(os.path.join(sandbox_base, f"model_{i}_{safe_label}"))

    # ------------------------------------------------------------------
    # 1. Sandbox phase — all agents run in parallel
    # ------------------------------------------------------------------

    def _run_one_sandbox(idx: int) -> tuple[int, str]:
        model = counsel_models[idx]
        label = specs[idx]["model"] if idx < len(specs) else f"model_{idx}"
        sandbox_dir = sandbox_dirs[idx]
        _populate_sandbox(workspace_dir, sandbox_dir)
        s_tools = _sandbox_tools(tools, sandbox_dir)
        try:
            from .agents.base_agent import _unwrap_model
            agent = create_react_agent(model=_unwrap_model(model), tools=s_tools, prompt=system_prompt)
            result = agent.invoke({"messages": [HumanMessage(content=task)]})
            last = result["messages"][-1] if result.get("messages") else None
            output = last.content if last and hasattr(last, "content") else ""
        except Exception as e:
            output = f"[{label} failed: {e}]"
        print(f"[counsel:{agent_name}] model_{idx} ({label}) complete.")
        return idx, output

    sandbox_outputs: List[str] = [""] * len(counsel_models)
    with ThreadPoolExecutor(max_workers=len(counsel_models)) as pool:
        futures = {pool.submit(_run_one_sandbox, i): i for i in range(len(counsel_models))}
        for future in as_completed(futures):
            idx, output = future.result()
            sandbox_outputs[idx] = output

    # ------------------------------------------------------------------
    # 2. Debate phase — all critiques per round run in parallel
    # ------------------------------------------------------------------
    formatted = "\n\n".join(
        f"=== Solution {i} ({specs[i]['model'] if i < len(specs) else i}) ===\n{out}"
        for i, out in enumerate(sandbox_outputs)
    )
    debate_history: List[str] = []

    for rnd in range(max_debate_rounds):
        base_prompt = (
            f"Task: {task}\n\n"
            f"Here are {len(sandbox_outputs)} independent solutions:\n\n{formatted}\n\n"
        )
        if debate_history:
            base_prompt += "Prior debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        base_prompt += (
            "Identify: (1) strongest elements of each solution, "
            "(2) weaknesses or errors, (3) a synthesized approach capturing the best of all. "
            "Be specific and concise."
        )

        def _one_critique(i: int) -> tuple[int, str]:
            spec = specs[i]
            try:
                resp = litellm.completion(
                    model=spec["model"],
                    messages=[{"role": "user", "content": base_prompt}],
                    max_tokens=2048,
                )
                critique = resp.choices[0].message.content or ""
            except Exception as e:
                critique = f"[{spec['model']} debate error: {e}]"
            return i, f"Model {i} ({spec['model']}):\n{critique}"

        critiques: List[str] = [""] * len(specs)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_one_critique, i): i for i in range(len(specs))}
            for future in as_completed(futures):
                i, text = future.result()
                critiques[i] = text

        debate_history.append(f"[Round {rnd + 1}]\n" + "\n\n".join(critiques))
        print(f"[counsel:{agent_name}] debate round {rnd + 1} complete.")

    # ------------------------------------------------------------------
    # 3. Synthesis
    # ------------------------------------------------------------------
    synthesis_prompt = (
        f"Task: {task}\n\n"
        f"You are the synthesis model. Review {len(sandbox_outputs)} independent solutions "
        f"and {len(debate_history)} rounds of debate, then produce the final authoritative "
        f"output for this pipeline stage.\n\n"
        f"Solutions:\n{formatted}\n\n"
        f"Debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        "Incorporate the strongest elements from all solutions. Be comprehensive and precise."
    )

    try:
        resp = litellm.completion(
            model=SYNTHESIS_MODEL,
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=8192,
        )
        final_output = resp.choices[0].message.content or sandbox_outputs[0]
    except Exception as e:
        print(f"[counsel:{agent_name}] synthesis failed ({e}), using first sandbox output.")
        final_output = sandbox_outputs[0] if sandbox_outputs else ""

    # ------------------------------------------------------------------
    # 4. Artifact promotion (earlier sandboxes first, later sandboxes win conflicts)
    # ------------------------------------------------------------------
    for sandbox_dir in sandbox_dirs:
        _merge_sandbox(sandbox_dir, workspace_dir)

    print(f"[counsel:{agent_name}] counsel complete.")
    return final_output


# ---------------------------------------------------------------------------
# LangGraph node factory
# ---------------------------------------------------------------------------

def create_counsel_node(
    system_prompt: str,
    tools: List[BaseTool],
    agent_name: str,
    workspace_dir: str,
    counsel_models: List[Any],
    max_debate_rounds: int = 3,
    model_specs: Optional[List[dict]] = None,
) -> Any:
    """
    Return a LangGraph node callable that wraps run_counsel_stage.

    Drop-in replacement for create_specialist_agent when counsel mode is enabled.
    Accepts the same state dict and returns the same state-update dict shape.
    """
    def counsel_node(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")
        output = run_counsel_stage(
            task=task,
            system_prompt=system_prompt,
            tools=tools,
            workspace_dir=workspace_dir,
            counsel_models=counsel_models,
            agent_name=agent_name,
            max_debate_rounds=max_debate_rounds,
            model_specs=model_specs,
        )
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    counsel_node.__name__ = agent_name
    return counsel_node
