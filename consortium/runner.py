"""
Core run logic for the consortium multi-agent system.

Separated from launch_multiagent.py so the entrypoint stays thin and
this module can be imported/tested independently.
"""

import importlib.util
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import litellm
from dotenv import load_dotenv

from .args import parse_arguments
from .config import load_llm_config, filter_model_params
from .interaction.callback_tools import setup_user_input_socket
from .interaction.http_steering import add_http_steering
from .prereqs import check_latex_prereqs
from .supervision import sanitize_result_payload
from .token_usage_tracker import initialize_run_token_tracker
from .counsel import create_counsel_models, DEFAULT_COUNSEL_MODEL_SPECS, set_counsel_timeout
from .graph import build_pipeline_stages_v2, get_default_checkpointer
from .utils import create_model, create_model_registry, save_agent_memory

logger = logging.getLogger(__name__)

load_dotenv(override=True)

litellm.drop_params = True
litellm.completion = filter_model_params(litellm.completion)


class _VertexErrorFilter:
    """Suppress repetitive Vertex AI credential errors from litellm logs."""
    _SUPPRESSED = frozenset([
        "Failed to load vertex credentials",
        "Google Cloud SDK not found",
        "vertex_llm_base",
    ])

    def filter(self, record):
        msg = str(getattr(record, "msg", ""))
        return not any(s in msg for s in self._SUPPRESSED)

_DEFAULT_TASK = (
    "Investigate whether training small language models with multiple paraphrased responses "
    "reduces hallucinations and improves response quality. Generate research ideas exploring: "
    "(1) Fine-tuning GPT-2 Small (124M) on instruction-following datasets with single vs multiple "
    "response variants, using small subsets (5K samples) for rapid experimentation, "
    "(2) Comparing response diversity and factual accuracy between single-response and multi-response "
    "training regimes, (3) Testing whether exposure to diverse correct answers during training acts "
    "as implicit regularization against repetitive or factually incorrect outputs, "
    "(4) Measuring trade-offs between response quality, diversity, and training efficiency with "
    "automated metrics. Use small-scale datasets like Alpaca-5K with automated paraphrase generation. "
    "Focus on fast automated evaluation including response diversity via Self-BLEU, factual consistency "
    "via rule-based checks, and response quality via ROUGE scores. Target achieving measurable "
    "improvements in response diversity while maintaining quality, with each experimental run "
    "completing in under 1 hour."
)

_CONTINUATION_TASK = (
    "Continue working on a previous research task. Please meticulously analyze the existing "
    "files and then plan how to call the relevant agents to further progress the research task "
    "and deliver better research outputs."
)

_STAGE_ALIASES = {
    "literature": "literature_review_agent",
    "litreview": "literature_review_agent",
    "literature_review": "literature_review_agent",
    "literature_review_agent": "literature_review_agent",
    "experiment": "experimentation_agent",
    "experimentation": "experimentation_agent",
    "experimentation_agent": "experimentation_agent",
    "analysis": "results_analysis_agent",
    "results_analysis": "results_analysis_agent",
    "results_analysis_agent": "results_analysis_agent",
    "math_literature": "math_literature_agent",
    "math_literature_agent": "math_literature_agent",
    "math_proposer": "math_proposer_agent",
    "math_proposer_agent": "math_proposer_agent",
    "math_prover": "math_prover_agent",
    "math_prover_agent": "math_prover_agent",
    "math_rigorous_verifier": "math_rigorous_verifier_agent",
    "math_rigorous_verifier_agent": "math_rigorous_verifier_agent",
    "math_empirical_verifier": "math_empirical_verifier_agent",
    "math_empirical_verifier_agent": "math_empirical_verifier_agent",
    "proof_transcription": "proof_transcription_agent",
    "proof_transcription_agent": "proof_transcription_agent",
    "resources": "resource_preparation_agent",
    "resource_preparation": "resource_preparation_agent",
    "resource_preparation_agent": "resource_preparation_agent",
    "writeup": "writeup_agent",
    "writeup_agent": "writeup_agent",
    "proofread": "proofreading_agent",
    "proofreading": "proofreading_agent",
    "proofreading_agent": "proofreading_agent",
    "review": "reviewer_agent",
    "reviewer": "reviewer_agent",
    "reviewer_agent": "reviewer_agent",
    "persona_council": "persona_council",
    "council": "persona_council",
    "brainstorm": "brainstorm_agent",
    "brainstorm_agent": "brainstorm_agent",
    "formalize_goals": "formalize_goals_agent",
    "formalize_goals_agent": "formalize_goals_agent",
    "goals": "formalize_goals_agent",
    "research_plan_writeup": "research_plan_writeup_agent",
    "research_plan_writeup_agent": "research_plan_writeup_agent",
    "plan_writeup": "research_plan_writeup_agent",
    "formalize_results": "formalize_results_agent",
    "formalize_results_agent": "formalize_results_agent",
}


def _setup_optional_tracing():
    enabled = os.getenv("CONSORTIUM_ENABLE_TRACING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        logger.debug("Tracing disabled (set CONSORTIUM_ENABLE_TRACING=1 to enable).")
        return
    # LangSmith auto-tracing is enabled via LANGCHAIN_TRACING_V2 env var.
    # Just confirm it's configured.
    if os.getenv("LANGCHAIN_TRACING_V2"):
        logger.info("LangSmith tracing enabled.")
    else:
        logger.info("Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable LangSmith tracing.")


def _install_litellm_token_callback():
    """Register a litellm success callback that writes every API call to the
    private token ledger.  This is a belt-and-suspenders layer: even if
    BudgetTrackingCallback or BudgetedLiteLLMModel is bypassed, every litellm
    call gets recorded for cost accounting.

    Also logs full input/output pairs for training data when the
    TrainingDataLogger is active.
    """
    from .token_usage_tracker import record_token_usage

    def _on_litellm_success(kwargs, response_obj, start_time, end_time):
        prompt_tokens = 0
        completion_tokens = 0
        model = "unknown"
        try:
            usage = getattr(response_obj, "usage", None)
            if usage is None:
                return
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            if not prompt_tokens and not completion_tokens:
                return
            model = kwargs.get("model") or getattr(response_obj, "model", "unknown")
            record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source="litellm_callback",
                model_id=model,
            )
        except Exception:
            # Log but don't break the LLM call chain.
            # Use a counter to avoid flooding logs — warn after 10 failures.
            _on_litellm_success._fail_count = getattr(_on_litellm_success, "_fail_count", 0) + 1
            if _on_litellm_success._fail_count <= 3:
                logger.debug(
                    "Token recording failed (occurrence #%d)", _on_litellm_success._fail_count,
                    exc_info=True,
                )
            elif _on_litellm_success._fail_count == 10:
                logger.warning(
                    "Token recording has failed %d times — token tracking may be inaccurate. "
                    "Check disk space and permissions.",
                    _on_litellm_success._fail_count,
                )

        # --- Training data logging (best-effort) ---
        try:
            from .logging.training_data_logger import get_training_data_logger
            tdl = get_training_data_logger()
            if tdl is None or not tdl.enabled:
                return

            messages = kwargs.get("messages", [])
            response_content = ""
            tool_calls = None
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                msg = getattr(choice, "message", None)
                if msg:
                    response_content = getattr(msg, "content", "") or ""
                    tool_calls = getattr(msg, "tool_calls", None)

            duration_ms = None
            if start_time and end_time:
                try:
                    duration_ms = int((end_time - start_time).total_seconds() * 1000)
                except Exception:
                    pass

            tdl.log_completion(
                messages=messages,
                response_content=response_content,
                model_id=model,
                source="litellm_callback",
                token_usage={
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                },
                call_id=kwargs.get("litellm_call_id"),
                duration_ms=duration_ms,
                tool_calls=tool_calls,
            )
        except Exception:
            pass  # Never break LLM calls for training data logging

    litellm.success_callback.append(_on_litellm_success)


def _filter_installed_imports(import_names):
    available, missing = [], []
    for name in import_names:
        root = name.split(".")[0]
        if importlib.util.find_spec(root) is not None:
            available.append(name)
        else:
            missing.append(name)
    if missing:
        logger.info("Skipping unavailable authorized imports: " + ", ".join(sorted(set(missing))))
    return available


def _resolve_model_settings(args, llm_config):
    """Return (model_name, reasoning_effort, verbosity, budget_tokens, effort)."""
    model_name = "claude-sonnet-4-6"
    reasoning_effort = "high"
    verbosity = "medium"
    budget_tokens = None
    effort = None

    if llm_config and "main_agents" in llm_config:
        cfg = llm_config["main_agents"]
        model_name = cfg.get("model", model_name)
        reasoning_effort = cfg.get("reasoning_effort", reasoning_effort)
        verbosity = cfg.get("verbosity", verbosity)
        budget_tokens = cfg.get("budget_tokens", budget_tokens)
        # Support both "effort" (legacy) and "reasoning_effort" config keys
        effort = cfg.get("effort") or cfg.get("reasoning_effort", effort)
        logger.info("Loaded config file settings for main agents")

    if args.model is not None:
        model_name = args.model
        logger.info("CLI override: using model %s", model_name)
    if args.reasoning_effort is not None:
        reasoning_effort = args.reasoning_effort
        logger.info("CLI override: reasoning_effort=%s", reasoning_effort)
    if args.verbosity is not None:
        verbosity = args.verbosity
        logger.info("CLI override: verbosity=%s", verbosity)

    return model_name, reasoning_effort, verbosity, budget_tokens, effort


def _build_required_artifacts(args, enforce_paper_artifacts, enforce_editorial_artifacts,
                               require_experiment_plan):
    artifacts = []
    if enforce_paper_artifacts:
        artifacts.append("final_paper.tex")
        if require_experiment_plan:
            artifacts.append("experiments_to_run_later.md")
        if args.require_pdf:
            artifacts.append("final_paper.pdf")
        if enforce_editorial_artifacts:
            artifacts.extend([
                "paper_workspace/author_style_guide.md",
                "paper_workspace/intro_skeleton.tex",
                "paper_workspace/style_macros.tex",
                "paper_workspace/reader_contract.json",
                "paper_workspace/editorial_contract.md",
                "paper_workspace/theorem_map.json",
                "paper_workspace/revision_log.md",
                "paper_workspace/copyedit_report.tex",
                "paper_workspace/review_report.tex",
                "paper_workspace/review_verdict.json",
            ])
            if args.enable_math_agents:
                artifacts.append("paper_workspace/claim_traceability.json")
    return artifacts


def _canonical_stage_name(stage_name: str) -> str:
    normalized = stage_name.strip().lower().replace("-", "_")
    return _STAGE_ALIASES.get(normalized, normalized)


def _resolve_start_stage_index(stage_name: str, pipeline_stages: list[str]) -> int:
    canonical = _canonical_stage_name(stage_name)
    if canonical not in pipeline_stages:
        valid = ", ".join(pipeline_stages)
        raise ValueError(
            f"Unknown --start-from-stage '{stage_name}' (resolved: '{canonical}'). "
            f"Valid stages for this run: {valid}"
        )
    return pipeline_stages.index(canonical)


_MODEL_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "o4": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok": "XAI_API_KEY",
    "xai": "XAI_API_KEY",
}


def _validate_api_keys(model_name: str) -> list[str]:
    """Return a list of error messages for missing API keys given a model name.

    Returns an empty list if all required keys are present.
    """
    errors = []
    model_lower = model_name.lower()
    required_key = None
    for prefix, env_var in _MODEL_KEY_MAP.items():
        if prefix in model_lower:
            required_key = env_var
            break
    if required_key and not os.getenv(required_key):
        errors.append(
            f"Model '{model_name}' requires {required_key} but it is not set.\n"
            f"  Add it to your .env file:  {required_key}=your_key_here"
        )
    # Counsel mode needs three keys; check is done lazily if counsel is enabled.
    return errors


def _list_runs(results_dir: str = "results") -> None:
    """Print a summary table of past runs in the results directory."""
    if not os.path.isdir(results_dir):
        print(f"No results directory found at '{results_dir}'.")
        return

    entries = sorted(
        (d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))),
        reverse=True,
    )
    if not entries:
        print("No past runs found in results/.")
        return

    header = f"{'Workspace':<45} {'Task (truncated)':<45} {'Cost':>8}  Status"
    print(header)
    print("-" * len(header))

    for name in entries:
        ws = os.path.join(results_dir, name)
        # Cost
        cost_str = "?"
        budget_path = os.path.join(ws, "budget_state.json")
        if os.path.exists(budget_path):
            try:
                with open(budget_path) as f:
                    bd = json.load(f)
                total = bd.get("total_usd", bd.get("total_cost_usd", None))
                if total is not None:
                    cost_str = f"${float(total):.2f}"
            except (json.JSONDecodeError, OSError):
                logger.debug("Failed to read budget_state.json for %s", name, exc_info=True)
        # Status
        status_str = "unknown"
        status_path = os.path.join(ws, "STATUS.txt")
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status_str = f.read().strip()[:20]
            except OSError:
                logger.debug("Failed to read STATUS.txt for %s", name, exc_info=True)
        # Task
        task_str = ""
        summary_path = os.path.join(ws, "run_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    sm = json.load(f)
                task_str = sm.get("task", "")[:43]
            except (json.JSONDecodeError, OSError):
                logger.debug("Failed to read run_summary.json for %s", name, exc_info=True)
        print(f"{name:<45} {task_str:<45} {cost_str:>8}  {status_str}")


def _write_experiment_metadata(workspace_dir: str, args, model_name: str, task: str) -> None:
    """Write experiment_metadata.json to the workspace at run start."""
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except OSError:
        logger.debug("Could not determine git commit hash", exc_info=True)

    git_dirty = False
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        git_dirty = bool(result.strip())
    except OSError:
        logger.debug("Could not determine git dirty status", exc_info=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "model": model_name,
        "task_preview": task[:200],
        "cli_args": {
            "enable_math_agents": getattr(args, "enable_math_agents", False),
            "enable_counsel": getattr(args, "enable_counsel", False),
            "output_format": getattr(args, "output_format", "latex"),
            "enforce_paper_artifacts": getattr(args, "enforce_paper_artifacts", False),
            "min_review_score": getattr(args, "min_review_score", 8),
        },
    }
    out_path = os.path.join(workspace_dir, "experiment_metadata.json")
    try:
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except OSError:
        logger.debug("Failed to write experiment_metadata.json", exc_info=True)


def _write_run_summary(workspace_dir: str, task: str, model_name: str,
                        start_time: datetime, stages_completed: list[str]) -> None:
    """Write run_summary.json to the workspace at run end."""
    duration_s = (datetime.now() - start_time).total_seconds()

    total_cost = None
    budget_path = os.path.join(workspace_dir, "budget_state.json")
    if os.path.exists(budget_path):
        try:
            with open(budget_path) as f:
                bd = json.load(f)
            total_cost = bd.get("total_usd", bd.get("total_cost_usd"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read budget_state.json in run summary", exc_info=True)

    total_tokens = None
    token_path = os.path.join(workspace_dir, "run_token_usage.json")
    if os.path.exists(token_path):
        try:
            with open(token_path) as f:
                tok = json.load(f)
            total_tokens = tok.get("total_tokens")
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read run_token_usage.json in run summary", exc_info=True)

    # Detect final paper path
    paper_path = None
    for candidate in ["final_paper.pdf", "final_paper.tex", "final_paper.md"]:
        p = os.path.join(workspace_dir, candidate)
        if os.path.exists(p):
            paper_path = candidate
            break

    summary = {
        "task": task,
        "model": model_name,
        "started_at": start_time.isoformat(),
        "duration_seconds": round(duration_s, 1),
        "stages_completed": stages_completed,
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "final_paper": paper_path,
        "workspace": workspace_dir,
    }
    out_path = os.path.join(workspace_dir, "run_summary.json")
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Run summary written: %s", out_path)
    except OSError:
        logger.debug("Failed to write run_summary.json", exc_info=True)


def main():
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start_time = datetime.now()
    effective_pipeline_mode = "full_research"

    # --list-runs: print past workspaces and exit
    if getattr(args, "list_runs", False):
        _list_runs()
        return 0

    # --- Logging setup ---
    env_log_default = os.getenv("CONSORTIUM_LOG_TO_FILES", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    enable_file_logs = env_log_default if args.log_to_files is None else args.log_to_files
    if enable_file_logs and not getattr(args, "dry_run", False):
        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/consortium_{timestamp}.out"
        err_path = f"logs/consortium_{timestamp}.err"
        logger.info("Redirecting stdout/stderr to: %s, %s", out_path, err_path)
        sys.stdout = open(out_path, "w", buffering=1)
        sys.stderr = open(err_path, "w", buffering=1)

    if args.debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        os.environ.setdefault("LITELLM_LOG", "WARNING")

    # P2-9: Suppress non-fatal Vertex AI credential errors that flood stderr.
    # These fire every time litellm tries (and fails) to use Vertex AI as a
    # fallback provider. They are harmless but produce thousands of log lines.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    _vertex_logger = logging.getLogger("LiteLLM Router")
    _vertex_logger.setLevel(logging.ERROR)
    for _handler in logging.root.handlers[:]:
        _handler.addFilter(_VertexErrorFilter())

    _setup_optional_tracing()
    _install_litellm_token_callback()

    # --- Deployment mode resolution ---
    from .mode import resolve_mode, load_mode_config, apply_mode_defaults
    mode = resolve_mode(args)
    mode_config = load_mode_config(mode)
    apply_mode_defaults(args, mode_config)
    logger.info("[PoggioAI] Running in %s mode — %s", mode, mode_config.get('description', ''))

    llm_config = load_llm_config()

    if args.pipeline_mode is not None:
        logger.warning(
            "--pipeline-mode is deprecated and ignored. "
            "Running fixed-stage full pipeline mode."
        )

    if args.start_from_stage and not args.resume:
        logger.error("--start-from-stage requires --resume <workspace_dir>.")
        return 1

    model_name, reasoning_effort, verbosity, budget_tokens, effort = _resolve_model_settings(
        args, llm_config
    )

    # --- API key validation (always run; also serves as the dry-run check) ---
    key_errors = _validate_api_keys(model_name)
    # Eagerly validate counsel API keys so we don't fail hours into an expensive run
    if getattr(args, "enable_counsel", False):
        for spec in DEFAULT_COUNSEL_MODEL_SPECS:
            key_errors.extend(_validate_api_keys(spec["model"]))
    if key_errors:
        for err in key_errors:
            logger.error(err)
        logger.error("Fix the above before running. See .env.example for the full list of keys.")
        return 1

    # --- Dry-run mode: validate config and exit without touching the pipeline ---
    if getattr(args, "dry_run", False):
        budget_config = llm_config.get("budget", {}) if llm_config else {}
        usd_limit = budget_config.get("usd_limit", "not set")
        logger.info("[dry-run] Environment checks passed:")
        logger.info("  model           : %s", model_name)
        logger.info("  reasoning_effort: %s", reasoning_effort)
        logger.info("  budget cap      : $%s", usd_limit)
        logger.info("  math agents     : %s", getattr(args, 'enable_math_agents', False))
        logger.info("  counsel mode    : %s", getattr(args, 'enable_counsel', False))
        logger.info("  output format   : %s", getattr(args, 'output_format', 'latex'))
        logger.info("[dry-run] All checks passed. Remove --dry-run to start the real run.")
        return 0

    # Set up interrupt socket (used by live-steering via state injection)
    input_queue = None
    if not getattr(args, "no_steering", False):
        input_queue = setup_user_input_socket(args.callback_host, args.callback_port)
        logger.info("Interruption port available at %s:%s", args.callback_host, args.callback_port)
        # Also start HTTP steering server on port+1 for programmatic clients (e.g. OpenClaw)
        add_http_steering(input_queue, host=args.callback_host, port=args.callback_port + 1)
    else:
        from queue import Queue
        input_queue = Queue()
        logger.info("[PoggioAI] Steering sockets disabled (--no-steering)")

    logger.info("LangGraph Research System Initialized — model: %s", model_name)

    # --- Experiment tool env config ---
    if llm_config and "run_experiment_tool" in llm_config:
        exp = llm_config["run_experiment_tool"]
        os.environ["RUN_EXPERIMENT_CODE_MODEL"] = exp.get("code_model", "gpt-5")
        os.environ["RUN_EXPERIMENT_FEEDBACK_MODEL"] = exp.get("feedback_model", "gpt-5")
        os.environ["RUN_EXPERIMENT_VLM_MODEL"] = exp.get("vlm_model", "gpt-5")
        os.environ["RUN_EXPERIMENT_REPORT_MODEL"] = exp.get("report_model", "gpt-5")
        os.environ["RUN_EXPERIMENT_REASONING_EFFORT"] = exp.get("reasoning_effort", "high")

    # --- Workspace setup ---
    if args.resume:
        results_base_dir = os.path.abspath(args.resume)
        if not os.path.exists(results_base_dir):
            logger.error("Workspace directory does not exist: %s", results_base_dir)
            return 1
        if not os.path.isdir(results_base_dir):
            logger.error("Path is not a directory: %s", results_base_dir)
            return 1
        task = args.task or _CONTINUATION_TASK
        logger.info("Resuming from: %s", results_base_dir)
    elif getattr(args, "iterate", None):
        # --- Iterate mode: revision from prior paper + feedback ---
        if args.start_from_stage:
            logger.error("--iterate and --start-from-stage cannot be used together. "
                         "Use --iterate-start-stage instead.")
            return 1
        from .iterate import validate_iterate_dir, build_iterate_state_seed
        try:
            validate_iterate_dir(args.iterate)
        except ValueError as e:
            logger.error("Iterate validation failed: %s", e)
            return 1
        results_base_dir = os.path.join("results", f"consortium_{timestamp}_iterate")
        os.makedirs(results_base_dir, exist_ok=True)
        task = args.task or "Revise and improve the paper based on reviewer feedback."
        iterate_seed = build_iterate_state_seed(args.iterate, results_base_dir)
        logger.info("Created iterate workspace: %s", results_base_dir)
        logger.info("Prior paper: %s", iterate_seed.get('iterate_prior_paper_path', 'N/A'))
        logger.info("Feedback: %s", iterate_seed.get('iterate_feedback_summary', 'N/A'))
        _write_experiment_metadata(results_base_dir, args, model_name, task)
    else:
        results_base_dir = os.path.join("results", f"consortium_{timestamp}")
        os.makedirs(results_base_dir, exist_ok=True)
        task = args.task or _DEFAULT_TASK
        logger.info("Created workspace: %s", results_base_dir)
        _write_experiment_metadata(results_base_dir, args, model_name, task)

    os.environ["RESULTS_BASE_DIR"] = results_base_dir
    os.environ["CONSORTIUM_OUTPUT_FORMAT"] = getattr(args, "output_format", "latex")

    # --- Token tracking ---
    run_id = f"{timestamp}_{os.getpid()}"
    token_file = initialize_run_token_tracker(
        workspace_dir=results_base_dir, run_id=run_id, reset=True,
    )
    logger.info("Token tracker initialized: %s", token_file)

    # --- Training data logging ---
    log_training = getattr(args, "log_training_data", False) or \
        os.getenv("CONSORTIUM_LOG_TRAINING_DATA", "").strip().lower() in {"1", "true", "yes"}
    if log_training:
        from .logging.training_data_logger import initialize_training_data_logger
        include_tool_calls = not getattr(args, "no_tool_calls_in_training_data", False)
        initialize_training_data_logger(
            workspace_dir=results_base_dir,
            run_id=run_id,
            enabled=True,
            include_tool_calls=include_tool_calls,
        )
        logger.info("Training data logging enabled: %s/training_data.jsonl", results_base_dir)

    logger.info("Task: %s%s", task[:100], '...' if len(task) > 100 else '')
    logger.info("Pipeline mode: full_research (fixed-stage)")
    if args.enable_math_agents:
        logger.info("Math agent workflow enabled.")

    pipeline_stages = build_pipeline_stages_v2(args.enable_math_agents)
    logger.info("Pipeline version: v2 (persona-council-driven)")
    try:
        start_stage_index = (
            _resolve_start_stage_index(args.start_from_stage, pipeline_stages)
            if args.start_from_stage
            else 0
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    if args.start_from_stage:
        resolved_stage = pipeline_stages[start_stage_index]
        logger.info(
            "Stage-based resume requested: start from '%s' (index %d).",
            resolved_stage, start_stage_index,
        )

    # --- Artifact gate setup ---
    auto_enforce = "final_paper" in task.lower() or "experiments_to_run_later" in task.lower()
    enforce_paper_artifacts = args.enforce_paper_artifacts or auto_enforce
    enforce_editorial_artifacts = args.enforce_editorial_artifacts and enforce_paper_artifacts
    require_experiment_plan = args.require_experiment_plan or (
        "experiments_to_run_later" in task.lower()
    )
    required_paper_artifacts = _build_required_artifacts(
        args, enforce_paper_artifacts, enforce_editorial_artifacts, require_experiment_plan
    )

    os.environ["CONSORTIUM_REQUIRE_PDF"] = "1" if args.require_pdf else "0"
    os.environ["CONSORTIUM_ENFORCE_PAPER_ARTIFACTS"] = "1" if enforce_paper_artifacts else "0"
    os.environ["CONSORTIUM_REQUIRE_EXPERIMENT_PLAN"] = "1" if require_experiment_plan else "0"
    os.environ["CONSORTIUM_ENFORCE_EDITORIAL_ARTIFACTS"] = (
        "1" if enforce_editorial_artifacts else "0"
    )

    if enforce_paper_artifacts:
        logger.info("Paper artifact gate: %s", ", ".join(required_paper_artifacts))

    # --- LaTeX prereq check ---
    require_latex = enforce_paper_artifacts or args.require_pdf or enforce_editorial_artifacts
    if require_latex:
        pdflatex_path, bibtex_path, latex_error = check_latex_prereqs()
        if latex_error:
            logger.error("Missing LaTeX prerequisites.\n%s", latex_error)
            return 1
        os.environ["CONSORTIUM_PDFLATEX_PATH"] = pdflatex_path
        os.environ["CONSORTIUM_BIBTEX_PATH"] = bibtex_path
        logger.info("LaTeX toolchain: pdflatex=%s, bibtex=%s", pdflatex_path, bibtex_path)

    # --- Model setup ---
    budget_config = llm_config.get("budget", {}) if llm_config else {}
    model = create_model(
        model_name, reasoning_effort, verbosity, budget_tokens,
        effort=effort,
        budget_config=budget_config, budget_dir=results_base_dir,
    )
    logger.info("Created model: %s", getattr(model, 'model', model_name))

    # --- Per-agent model tiering ---
    model_registry = create_model_registry(
        llm_config, model, budget_config=budget_config, budget_dir=results_base_dir,
    )
    if model_registry._agent_models:
        tier_counts: dict[str, int] = {}
        for _agent, _m in model_registry._agent_models.items():
            mid = getattr(_m, "model", "unknown")
            tier_counts[mid] = tier_counts.get(mid, 0) + 1
        logger.info("Per-agent model tiers: %s", tier_counts)

    essential_imports = _filter_installed_imports([
        "json", "os", "posixpath", "ntpath", "sys", "datetime", "uuid", "typing",
        "pathlib", "shutil", "textwrap", "functools", "copy", "pickle", "logging",
        "warnings", "gc", "argparse", "configparser", "yaml", "toml", "requests",
        "urllib", "datasets", "transformers", "huggingface_hub", "tokenizers",
        "wandb", "tensorboard", "tqdm", "zipfile", "tarfile",
    ])

    # --- Counsel model setup ---
    counsel_cfg = llm_config.get("counsel", {}) if llm_config else {}
    # Priority: --enable-counsel flag > --no-counsel flag > config file
    counsel_enabled = getattr(args, "enable_counsel", False)
    counsel_disabled = getattr(args, "no_counsel", False)
    if counsel_disabled:
        counsel_enabled = False
    elif not counsel_enabled:
        counsel_enabled = bool(counsel_cfg.get("enabled", False))

    counsel_models_list = None
    if counsel_enabled:
        max_debate_rounds = getattr(args, "counsel_max_debate_rounds", None)
        if max_debate_rounds is None:
            max_debate_rounds = int(counsel_cfg.get("max_debate_rounds", 3))

        # Build model specs from config if provided, otherwise use defaults
        cfg_model_specs = counsel_cfg.get("models")
        model_specs = cfg_model_specs if cfg_model_specs else None

        logger.info("Counsel mode enabled — %d models, %d debate rounds per stage.",
                    len(model_specs or DEFAULT_COUNSEL_MODEL_SPECS), max_debate_rounds)
        counsel_models_list = create_counsel_models(
            budget_config=budget_config,
            budget_dir=results_base_dir,
            model_specs=model_specs,
        )
        os.environ["CONSORTIUM_COUNSEL_MAX_DEBATE_ROUNDS"] = str(max_debate_rounds)

        # Set per-model counsel timeout from env or default (600s)
        counsel_timeout = int(os.environ.get("COUNSEL_MODEL_TIMEOUT_SECONDS", "600"))
        set_counsel_timeout(counsel_timeout)

    # Build tree search config if enabled
    tree_search_config = None
    if getattr(args, "enable_tree_search", False):
        from consortium.tree_search.tree_state import CounselMode, TreeSearchConfig
        tree_search_config = TreeSearchConfig(
            enabled=True,
            max_breadth=getattr(args, "tree_max_breadth", 3),
            max_depth=getattr(args, "tree_max_depth", 4),
            max_parallel=getattr(args, "tree_max_parallel", 6),
            pruning_threshold=getattr(args, "tree_pruning_threshold", 0.2),
            counsel_mode=CounselMode(getattr(args, "tree_counsel_mode", "all_nodes")),
        )

    try:
        autonomous_mode = getattr(args, "autonomous_mode", True)
        enable_milestone_gates = getattr(args, "enable_milestone_gates", False)
        # In autonomous mode, force milestone gates off regardless of other flags
        if autonomous_mode and enable_milestone_gates:
            logger.info("Autonomous mode: milestone gates disabled.")
            enable_milestone_gates = False
        milestone_timeout = getattr(args, "milestone_timeout", 3600)
        adversarial_verification = getattr(args, "adversarial_verification", False)

        # --- Persona-council-driven pipeline ---
        pc_cfg = llm_config.get("persona_council", {}) if llm_config else {}
        dc_cfg = llm_config.get("duality_check", {}) if llm_config else {}

        persona_debate_rounds = getattr(args, "persona_debate_rounds", None)
        if persona_debate_rounds is None:
            persona_debate_rounds = int(pc_cfg.get("max_debate_rounds", 3))

        persona_council_specs = pc_cfg.get("personas") or None
        persona_synthesis_model = pc_cfg.get("synthesis_model", "claude-opus-4-6")
        duality_check_model = dc_cfg.get("model", "claude-opus-4-6")
        enable_duality_check = not getattr(args, "no_duality_check", False)

        # Extract budget manager if available
        budget_manager = None
        if hasattr(model, "budget_manager"):
            budget_manager = model.budget_manager

        from .graph import build_research_graph_v2
        from .graph_config import (
            ResearchGraphConfig,
            PersonaCouncilConfig,
            DualityCheckConfig,
            ArtifactEnforcementConfig,
        )
        checkpointer = get_default_checkpointer(results_base_dir)
        graph_config = ResearchGraphConfig(
            model=model,
            workspace_dir=results_base_dir,
            pipeline_mode=effective_pipeline_mode,
            enable_math_agents=args.enable_math_agents,
            artifacts=ArtifactEnforcementConfig(
                enforce_paper_artifacts=enforce_paper_artifacts,
                enforce_editorial_artifacts=enforce_editorial_artifacts,
                require_pdf=args.require_pdf,
                require_experiment_plan=require_experiment_plan,
            ),
            min_review_score=args.min_review_score,
            followup_max_iterations=args.followup_max_iterations,
            manager_max_steps=args.manager_max_steps or 50,
            authorized_imports=essential_imports,
            checkpointer=checkpointer,
            counsel_models=counsel_models_list,
            tree_search=tree_search_config,
            enable_milestone_gates=enable_milestone_gates,
            adversarial_verification=adversarial_verification,
            iterate_mode=bool(getattr(args, "iterate", None)),
            persona_council=PersonaCouncilConfig(
                specs=persona_council_specs,
                debate_rounds=persona_debate_rounds,
                synthesis_model=persona_synthesis_model,
            ),
            duality_check=DualityCheckConfig(
                enabled=enable_duality_check,
                model=duality_check_model,
            ),
            budget_manager=budget_manager,
            model_registry=model_registry,
        )
        graph = build_research_graph_v2(graph_config)
        logger.info("Pipeline: %d persona debate rounds, duality_check=%s",
                    persona_debate_rounds, 'enabled' if enable_duality_check else 'disabled')

        if adversarial_verification:
            logger.info("Adversarial verification enabled — red-team verifiers will "
                        "challenge proofs and experiments after cooperative verification passes.")

        if enable_milestone_gates:
            logger.info("Milestone gates enabled (timeout=%ds). "
                        "POST /milestone_response to approve/modify/abort at each gate.",
                        milestone_timeout)

        # Build initial state
        initial_state = {
            "messages": [],
            "task": task,
            "workspace_dir": results_base_dir,
            "pipeline_mode": effective_pipeline_mode,
            "math_enabled": args.enable_math_agents,
            "enforce_paper_artifacts": enforce_paper_artifacts,
            "enforce_editorial_artifacts": enforce_editorial_artifacts,
            "require_pdf": args.require_pdf,
            "require_experiment_plan": require_experiment_plan,
            "min_review_score": args.min_review_score,
            "followup_max_iterations": args.followup_max_iterations,
            "manager_max_steps": args.manager_max_steps if args.manager_max_steps else 50,
            "pipeline_stages": pipeline_stages,
            "pipeline_stage_index": start_stage_index,
            "current_agent": None,
            "agent_task": None,
            "agent_outputs": {},
            "artifacts": {},
            "iteration_count": 0,
            "followup_iteration": 0,
            "research_cycle": 0,
            "max_research_cycles": args.followup_max_iterations,
            "novelty_check_attempts": 0,
            "rebuttal_iteration": 0,
            "max_rebuttal_iterations": args.max_rebuttal_iterations,
            "validation_results": {},
            "interrupt_instruction": None,
            "theory_track_status": None,
            "experiment_track_status": None,
            "track_decomposition": None,
            "tree_search_enabled": getattr(args, "enable_tree_search", False),
            "tree_state_path": None,
            "active_branch_id": None,
            "milestone_reports": [],
            "human_feedback_history": [],
            "enable_milestone_gates": enable_milestone_gates,
            "milestone_timeout": milestone_timeout,
            "intermediate_validation_log": [],
            "finished": False,
            # Pipeline fields
            "autonomous_mode": autonomous_mode,
            "research_proposal": None,
            "brainstorm_output": None,
            "brainstorm_history": [],
            "research_goals": None,
            "formalized_results": None,
            "lit_review_feasibility": None,
            "verify_completion_result": None,
            "verify_completion_history": [],
            "duality_check_result": None,
            "lit_review_attempts": 0,
            "brainstorm_cycle": 0,
            "verify_rework_attempts": 0,
            "duality_rework_attempts": 0,
            # Iterate mode fields
            "iterate_mode": False,
            "iterate_prior_paper_path": None,
            "iterate_feedback_path": None,
            "iterate_feedback_summary": None,
        }

        # --- Iterate mode: merge state seed and override entry stage ---
        if getattr(args, "iterate", None):
            initial_state.update(iterate_seed)  # noqa: F821 (defined in iterate branch above)
            iterate_entry_stage = getattr(args, "iterate_start_stage", "resource_preparation_agent")
            canonical = _canonical_stage_name(iterate_entry_stage)
            try:
                initial_state["pipeline_stage_index"] = pipeline_stages.index(canonical)
            except ValueError:
                logger.warning("iterate-start-stage '%s' not in pipeline stages, "
                              "defaulting to resource_preparation_agent", iterate_entry_stage)
                initial_state["pipeline_stage_index"] = pipeline_stages.index("resource_preparation_agent")

        # Use workspace_dir as thread_id for default resumability.
        # For stage-based resume, create a fresh thread in the same workspace so
        # previous checkpoint state does not override the requested stage index.
        if args.start_from_stage:
            canonical_stage = _canonical_stage_name(args.start_from_stage)
            thread_id = f"{results_base_dir}::stage_resume::{canonical_stage}::{timestamp}"
        else:
            thread_id = results_base_dir
        run_config = {"configurable": {"thread_id": thread_id}}

        logger.info("=" * 50)
        logger.info("Running LangGraph research pipeline...")
        logger.info("Task: %s", task)
        logger.info("=" * 50)

        # --- Progress heartbeat watchdog (Tier 1.1) ---
        # Writes a .progress_heartbeat file every 2 minutes so the campaign
        # heartbeat / OpenClaw overseer can detect hung stages (process alive
        # but not making progress for >30 min).
        progress_file = os.path.join(results_base_dir, ".progress_heartbeat")
        _graph_done = threading.Event()

        def _watchdog_writer():
            while not _graph_done.is_set():
                try:
                    tmp = progress_file + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"ts": time.time(), "pid": os.getpid()}, f)
                    os.replace(tmp, progress_file)
                except OSError:
                    logger.debug("Heartbeat write failed (disk full / permissions)", exc_info=True)
                _graph_done.wait(120)

        watchdog_thread = threading.Thread(target=_watchdog_writer, daemon=True)
        watchdog_thread.start()

        # --- Hard timeout via SIGALRM (Tier 1.1) ---
        max_run_seconds = getattr(args, "max_run_seconds", None)
        _old_alarm_handler = None
        if max_run_seconds and hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Pipeline exceeded --max-run-seconds={max_run_seconds}. "
                    f"Killing run to prevent infinite hang."
                )
            _old_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max_run_seconds)
            logger.info("Hard timeout set: %ds", max_run_seconds)

        # Validate initial state before graph invocation
        from .state import validate_initial_state
        try:
            state_warnings = validate_initial_state(initial_state)
            for w in state_warnings:
                logger.warning("[state] %s", w)
        except ValueError as e:
            logger.error("[state] FATAL: %s", e)
            raise

        try:
            final_state = graph.invoke(initial_state, config=run_config)
        finally:
            _graph_done.set()
            if max_run_seconds and hasattr(signal, "SIGALRM"):
                signal.alarm(0)  # cancel alarm
                if _old_alarm_handler is not None:
                    signal.signal(signal.SIGALRM, _old_alarm_handler)
            # Clean up progress heartbeat file on normal completion
            try:
                os.remove(progress_file)
            except OSError:
                logger.debug("Could not remove progress heartbeat file", exc_info=True)

        result = final_state.get("agent_outputs", {})
        result = sanitize_result_payload(
            result=str(result),
            workspace_dir=results_base_dir,
            required_artifacts=required_paper_artifacts,
        )
        if isinstance(result, dict) and result.get("status") == "incomplete":
            logger.warning(
                "Run marked incomplete — missing: %s",
                ", ".join(result.get("missing_required_artifacts", [])),
            )

        # Determine which stages completed (read from state if available)
        stages_done = []
        if isinstance(final_state, dict):
            idx = final_state.get("pipeline_stage_index", 0)
            stages_done = pipeline_stages[:idx]

        _write_run_summary(
            workspace_dir=results_base_dir,
            task=task,
            model_name=model_name,
            start_time=run_start_time,
            stages_completed=stages_done,
        )

        logger.info("=" * 50)
        logger.info("Task finished.")
        logger.info("=" * 50)

    except Exception as e:
        logger.error("Error during pipeline execution: %s", e, exc_info=True)
        return 1

    return 0
