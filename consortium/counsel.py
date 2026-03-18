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

from .utils import normalize_model_for_litellm


# Default model specs matching the 4 top-tier frontier models.
# Used for debate-phase litellm.completion calls (where we need the model name string).
DEFAULT_COUNSEL_MODEL_SPECS = [
    {"model": "claude-opus-4-6",  "reasoning_effort": "high"},
    {"model": "gpt-5.4",          "reasoning_effort": "high", "verbosity": "high"},
    {"model": "gemini-3-pro-preview",   "thinking_budget": 65536},
]

SYNTHESIS_MODEL = "claude-opus-4-6"

# Per-model timeout for sandbox and debate phases.
# Set via set_counsel_timeout() at pipeline init, or COUNSEL_MODEL_TIMEOUT_SECONDS env var.
DEFAULT_MODEL_TIMEOUT_SECONDS: int = int(os.environ.get("COUNSEL_MODEL_TIMEOUT_SECONDS", "3600"))


def set_counsel_timeout(seconds: int) -> None:
    """Set the global per-model counsel timeout. Called at pipeline init from campaign config."""
    global DEFAULT_MODEL_TIMEOUT_SECONDS
    DEFAULT_MODEL_TIMEOUT_SECONDS = seconds


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
        # Support both "effort" (legacy) and "reasoning_effort" spec keys
        claude_effort = spec.get("effort") or spec.get("reasoning_effort")
        m = create_model(
            model_name=spec["model"],
            reasoning_effort=spec.get("reasoning_effort", "high"),
            verbosity=spec.get("verbosity", "medium"),
            budget_tokens=spec.get("thinking_budget"),
            effort=claude_effort,
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
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            shutil.copytree(
                workspace_dir,
                sandbox_dir,
                ignore=shutil.ignore_patterns(
                    "counsel_sandboxes", "_test_sandboxes", "*_sandboxes",
                    "*.db", "*.lock", "__pycache__",
                ),
            )
            return
        except (InterruptedError, OSError) as exc:
            if attempt == max_attempts - 1:
                print(
                    f"[counsel] sandbox copy failed after {max_attempts} attempts: {exc}"
                )
                raise
            print(
                f"[counsel] sandbox copy attempt {attempt + 1} failed ({exc}), "
                f"retrying in {1 << attempt}s..."
            )
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            time.sleep(1 << attempt)  # 1s, 2s, 4s, 8s


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
        except Exception as exc:
            # WARNING: falling back to the original tool means sandbox writes
            # go to the main workspace, bypassing isolation. Log prominently.
            print(
                f"[counsel] WARNING: could not re-instantiate tool '{tool.name}' "
                f"for sandbox ({exc}). Excluding from sandbox tools to preserve isolation."
            )
            # Skip the tool rather than break sandbox isolation
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
                # Log merge failures prominently — critical files failing to merge
                # can cause downstream pipeline errors.
                print(f"[counsel] merge WARNING: failed to copy {rel}: {exc}")


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
    model_timeout_seconds: int = 600,
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

    # Extract BudgetManager for direct litellm calls (debate + synthesis)
    _budget_mgr = None
    from .budget import BudgetedLiteLLMModel
    for _m in counsel_models:
        if isinstance(_m, BudgetedLiteLLMModel):
            _budget_mgr = _m.budget_manager
            break

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
        import time as _time
        model = counsel_models[idx]
        label = specs[idx]["model"] if idx < len(specs) else f"model_{idx}"
        sandbox_dir = sandbox_dirs[idx]
        _populate_sandbox(workspace_dir, sandbox_dir)
        s_tools = _sandbox_tools(tools, sandbox_dir)
        try:
            from .agents.base_agent import _unwrap_model
            # Budget is now recorded automatically by the monkey-patched litellm.completion()
            agent = create_react_agent(model=_unwrap_model(model), tools=s_tools, prompt=system_prompt)
            invoke_cfg = None

            # Retry on transient API errors (auth, service unavailable, rate limit)
            _RETRYABLE = (
                "AuthenticationError", "ServiceUnavailableError",
                "RateLimitError", "APIConnectionError", "APIError",
                "InternalServerError", "Timeout", "ClientError",
                "BadRequestError", "HTTPStatusError",
            )
            _RETRYABLE_SUBSTRINGS = ("429", "500", "502", "503", "rate limit", "overloaded")
            _MAX_RETRIES = 3
            result = None
            for _attempt in range(_MAX_RETRIES):
                try:
                    result = agent.invoke({"messages": [HumanMessage(content=task)]}, config=invoke_cfg)
                    break
                except Exception as _api_err:
                    err_type = type(_api_err).__name__
                    err_str = str(_api_err).lower()
                    is_retryable = (
                        any(r in err_type for r in _RETRYABLE)
                        or any(r in str(type(_api_err)) for r in _RETRYABLE)
                        or any(s in err_str for s in _RETRYABLE_SUBSTRINGS)
                    )
                    if is_retryable:
                        wait = 5 * (2 ** _attempt)  # 5s, 10s, 20s
                        print(f"[counsel:{agent_name}] model_{idx} ({label}) transient error (attempt {_attempt+1}/{_MAX_RETRIES}): {err_type}. Retrying in {wait}s...")
                        _time.sleep(wait)
                        if _attempt == _MAX_RETRIES - 1:
                            raise  # final attempt, propagate
                    else:
                        raise  # non-retryable, propagate immediately
            # Extract the last AIMessage with non-empty text content.
            # The final message may be a ToolMessage or an AIMessage with only
            # tool_calls and no text — walk backwards to find actual content.
            output = ""
            for msg in reversed(result.get("messages", [])):
                # Skip ToolMessages — they contain tool output, not the agent's answer
                if type(msg).__name__ == "ToolMessage":
                    continue
                raw = getattr(msg, "content", "")
                # Anthropic models return content as a list of blocks
                if isinstance(raw, list):
                    text_parts = []
                    for block in raw:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    text = "\n".join(text_parts)
                else:
                    text = str(raw) if raw else ""
                if text.strip():
                    output = text
                    break
            # DEBUG: log what we found
            if not output:
                msg_types = [type(m).__name__ for m in result.get("messages", [])[-5:]]
                last_contents = []
                for m in result.get("messages", [])[-3:]:
                    c = getattr(m, "content", None)
                    last_contents.append(f"{type(m).__name__}:type={type(c).__name__}:len={len(str(c)[:100])}")
                print(f"[counsel:DEBUG] model_{idx} empty output. Last 5 msg types: {msg_types}. Last 3: {last_contents}")
            # Fallback: if no text found, summarize the workspace diff
            if not output:
                import os
                new_files = []
                for root, dirs, files in os.walk(sandbox_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        if not any(x in fp for x in ["__pycache__", "token_usage", "budget_", "checkpoint"]):
                            new_files.append(os.path.relpath(fp, sandbox_dir))
                if new_files:
                    output = f"Agent completed. Files created/modified in sandbox: {len(new_files)} files."
            print(f"[counsel:DEBUG] model_{idx} output len={len(output)} starts_with_bracket={output[:1]=='['} first80={repr(output[:80])}")
        except Exception as e:
            output = f"[{label} failed: {e}]"
        print(f"[counsel:{agent_name}] model_{idx} ({label}) complete.")
        return idx, output

    sandbox_outputs: List[str] = [""] * len(counsel_models)
    with ThreadPoolExecutor(max_workers=len(counsel_models)) as pool:
        futures = {pool.submit(_run_one_sandbox, i): i for i in range(len(counsel_models))}
        try:
            for future in as_completed(futures, timeout=model_timeout_seconds + 60):
                try:
                    idx, output = future.result(timeout=model_timeout_seconds)
                    sandbox_outputs[idx] = output
                except TimeoutError:
                    # Identify which model timed out
                    for f, i in futures.items():
                        if f is future:
                            label = specs[i]["model"] if i < len(specs) else f"model_{i}"
                            sandbox_outputs[i] = f"[{label} timed out after {model_timeout_seconds}s]"
                            print(f"[counsel:{agent_name}] model_{i} ({label}) TIMED OUT after {model_timeout_seconds}s")
                            break
                except Exception as e:
                    for f, i in futures.items():
                        if f is future:
                            label = specs[i]["model"] if i < len(specs) else f"model_{i}"
                            sandbox_outputs[i] = f"[{label} error: {e}]"
                            print(f"[counsel:{agent_name}] model_{i} ({label}) error: {e}")
                            break
        except TimeoutError:
            # as_completed() outer timeout — some futures didn't finish.
            # Completed futures already populated sandbox_outputs; downstream quorum logic handles gaps.
            for f, i in futures.items():
                if not f.done():
                    label = specs[i]["model"] if i < len(specs) else f"model_{i}"
                    sandbox_outputs[i] = f"[{label} timed out after {model_timeout_seconds}s]"
                    print(f"[counsel:{agent_name}] model_{i} ({label}) TIMED OUT (outer as_completed timeout)")
                    f.cancel()

    # ------------------------------------------------------------------
    # 1b. Quorum check — skip debate if too few models succeeded
    # ------------------------------------------------------------------
    _MIN_QUORUM = 2  # need at least 2 valid outputs for meaningful debate
    valid_outputs = [
        i for i, out in enumerate(sandbox_outputs)
        if out and not out.startswith("[") and not out.endswith("timed out]") and not out.endswith("failed]")
    ]
    # DEBUG: show what each sandbox produced
    for i, out in enumerate(sandbox_outputs):
        is_valid = i in valid_outputs
        print(f"[counsel:QUORUM_DEBUG] model_{i}: valid={is_valid} len={len(out)} truthy={bool(out)} first60={repr(out[:60])}")
    if len(valid_outputs) < _MIN_QUORUM and len(counsel_models) >= _MIN_QUORUM:
        print(
            f"[counsel:{agent_name}] Only {len(valid_outputs)}/{len(counsel_models)} models "
            f"produced valid output (quorum={_MIN_QUORUM}). Skipping debate, "
            f"using best available sandbox output for synthesis."
        )
        max_debate_rounds = 0  # skip debate, go straight to synthesis

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
            # Extract provider-specific params (everything except "model")
            extra_params = {k: v for k, v in spec.items() if k != "model"}
            # Normalize legacy "effort" key to "reasoning_effort" for litellm
            if "effort" in extra_params:
                extra_params.setdefault("reasoning_effort", extra_params.pop("effort"))
            try:
                resp = litellm.completion(
                    model=normalize_model_for_litellm(spec["model"]),
                    messages=[{"role": "user", "content": base_prompt}],
                    max_tokens=2048,
                    **extra_params,
                )
                critique = resp.choices[0].message.content or ""
                # Budget recorded automatically via litellm.completion monkey-patch
            except Exception as e:
                critique = f"[{spec['model']} debate error: {e}]"
            return i, f"Model {i} ({spec['model']}):\n{critique}"

        debate_timeout = model_timeout_seconds  # same timeout for debate
        critiques: List[str] = [""] * len(specs)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_one_critique, i): i for i in range(len(specs))}
            try:
                for future in as_completed(futures, timeout=debate_timeout + 60):
                    try:
                        i, text = future.result(timeout=debate_timeout)
                        critiques[i] = text
                    except TimeoutError:
                        for f, idx in futures.items():
                            if f is future:
                                label = specs[idx]["model"] if idx < len(specs) else f"model_{idx}"
                                critiques[idx] = f"Model {idx} ({label}):\n[debate timed out after {debate_timeout}s]"
                                print(f"[counsel:{agent_name}] debate model_{idx} ({label}) TIMED OUT")
                                break
                    except Exception as e:
                        for f, idx in futures.items():
                            if f is future:
                                critiques[idx] = f"Model {idx}:\n[debate error: {e}]"
                                break
            except TimeoutError:
                # as_completed() outer timeout — some debate critiques didn't finish.
                # Partial debate results are better than crashing the entire pipeline.
                print(f"[counsel:{agent_name}] debate round timeout — some critiques did not complete within {debate_timeout + 60}s")
                for f, idx in futures.items():
                    if not f.done():
                        label = specs[idx]["model"] if idx < len(specs) else f"model_{idx}"
                        critiques[idx] = f"Model {idx} ({label}):\n[debate timed out after {debate_timeout}s]"
                        f.cancel()

        debate_history.append(f"[Round {rnd + 1}]\n" + "\n\n".join(critiques))
        # Circuit breaker: if >50% of debate models failed, skip remaining
        # rounds and warn — synthesis from mostly-failed inputs is unreliable
        failed_count = sum(
            1 for c in critiques if "error:" in c or "timed out" in c
        )
        if failed_count > len(specs) / 2:
            print(
                f"[counsel:{agent_name}] CIRCUIT BREAKER: {failed_count}/{len(specs)} "
                f"debate models failed in round {rnd + 1}. Skipping remaining rounds."
            )
            break
        print(f"[counsel:{agent_name}] debate round {rnd + 1} complete.")

    # ------------------------------------------------------------------
    # 3. Pre-synthesis circuit breaker
    # ------------------------------------------------------------------
    sandbox_failures = sum(1 for o in sandbox_outputs if "error:" in o)
    if sandbox_failures == len(sandbox_outputs):
        # All sandbox runs failed — synthesis would be meaningless
        print(
            f"[counsel:{agent_name}] ALL {len(sandbox_outputs)} sandbox models failed. "
            f"Returning first sandbox error instead of synthesizing garbage."
        )
        return sandbox_outputs[0]

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

    # Find synthesis model params from specs (default to reasoning_effort=high)
    _synth_params = {}
    for _sp in specs:
        if _sp["model"] == SYNTHESIS_MODEL:
            _synth_params = {k: v for k, v in _sp.items() if k != "model"}
            if "effort" in _synth_params:
                _synth_params.setdefault("reasoning_effort", _synth_params.pop("effort"))
            break

    try:
        resp = litellm.completion(
            model=normalize_model_for_litellm(SYNTHESIS_MODEL),
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=8192,
            **_synth_params,
        )
        final_output = resp.choices[0].message.content or sandbox_outputs[0]
        # Budget recorded automatically via litellm.completion monkey-patch
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
    model_timeout_seconds: Optional[int] = None,
) -> Any:
    """
    Return a LangGraph node callable that wraps run_counsel_stage.

    Drop-in replacement for create_specialist_agent when counsel mode is enabled.
    Accepts the same state dict and returns the same state-update dict shape.
    """
    _timeout = model_timeout_seconds if model_timeout_seconds is not None else DEFAULT_MODEL_TIMEOUT_SECONDS

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
            model_timeout_seconds=_timeout,
        )
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    counsel_node.__name__ = agent_name
    return counsel_node
