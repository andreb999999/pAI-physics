"""
Core run logic for the freephdlabor multi-agent system.

Separated from launch_multiagent.py so the entrypoint stays thin and
this module can be imported/tested independently.
"""

import importlib.util
import os
import sys
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
from .counsel import create_counsel_models, DEFAULT_COUNSEL_MODEL_SPECS
from .graph import build_pipeline_stages
from .utils import create_model, build_research_graph, save_agent_memory

load_dotenv(override=True)

litellm.drop_params = True
litellm.completion = filter_model_params(litellm.completion)

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
    "ideation": "ideation_agent",
    "ideation_agent": "ideation_agent",
    "literature": "literature_review_agent",
    "litreview": "literature_review_agent",
    "literature_review": "literature_review_agent",
    "literature_review_agent": "literature_review_agent",
    "planning": "research_planner_agent",
    "plan": "research_planner_agent",
    "research_planner": "research_planner_agent",
    "research_planner_agent": "research_planner_agent",
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
}


def _setup_optional_tracing():
    enabled = os.getenv("FREEPHDLABOR_ENABLE_TRACING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        print("Tracing disabled (set FREEPHDLABOR_ENABLE_TRACING=1 to enable).")
        return
    # LangSmith auto-tracing is enabled via LANGCHAIN_TRACING_V2 env var.
    # Just confirm it's configured.
    if os.getenv("LANGCHAIN_TRACING_V2"):
        print("LangSmith tracing enabled.")
    else:
        print("Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable LangSmith tracing.")


def _filter_installed_imports(import_names):
    available, missing = [], []
    for name in import_names:
        root = name.split(".")[0]
        if importlib.util.find_spec(root) is not None:
            available.append(name)
        else:
            missing.append(name)
    if missing:
        print("Skipping unavailable authorized imports: " + ", ".join(sorted(set(missing))))
    return available


def _resolve_model_settings(args, llm_config):
    """Return (model_name, reasoning_effort, verbosity, budget_tokens, effort)."""
    model_name = "gpt-5"
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
        effort = cfg.get("effort", effort)
        print("Loaded config file settings for main agents")

    if args.model is not None:
        model_name = args.model
        print(f"CLI override: using model {model_name}")
    if args.reasoning_effort is not None:
        reasoning_effort = args.reasoning_effort
        print(f"CLI override: reasoning_effort={reasoning_effort}")
    if args.verbosity is not None:
        verbosity = args.verbosity
        print(f"CLI override: verbosity={verbosity}")

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
                "paper_workspace/copyedit_report.md",
                "paper_workspace/review_report.md",
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


def main():
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_pipeline_mode = "full_research"

    # --- Logging setup ---
    env_log_default = os.getenv("FREEPHDLABOR_LOG_TO_FILES", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    enable_file_logs = env_log_default if args.log_to_files is None else args.log_to_files
    if enable_file_logs:
        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/freephdlabor_{timestamp}.out"
        err_path = f"logs/freephdlabor_{timestamp}.err"
        print(f"Redirecting stdout/stderr to: {out_path}, {err_path}")
        sys.stdout = open(out_path, "w", buffering=1)
        sys.stderr = open(err_path, "w", buffering=1)

    if args.debug:
        os.environ["LITELLM_LOG"] = "DEBUG"
    else:
        os.environ.setdefault("LITELLM_LOG", "WARNING")

    _setup_optional_tracing()

    llm_config = load_llm_config()

    if args.pipeline_mode is not None:
        print(
            "Warning: --pipeline-mode is deprecated and ignored. "
            "Running deterministic full pipeline mode."
        )

    if args.start_from_stage and not args.resume:
        print("Error: --start-from-stage requires --resume <workspace_dir>.")
        return 1

    # Set up interrupt socket (used by live-steering via state injection)
    input_queue = setup_user_input_socket(args.callback_host, args.callback_port)
    print(f"Interruption port available at {args.callback_host}:{args.callback_port}")
    # Also start HTTP steering server on port+1 for programmatic clients (e.g. OpenClaw)
    add_http_steering(input_queue, host=args.callback_host, port=args.callback_port + 1)

    model_name, reasoning_effort, verbosity, budget_tokens, effort = _resolve_model_settings(
        args, llm_config
    )

    print(f"\nLangGraph Research System Initialized — model: {model_name}")

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
            print(f"Workspace directory does not exist: {results_base_dir}")
            return 1
        if not os.path.isdir(results_base_dir):
            print(f"Path is not a directory: {results_base_dir}")
            return 1
        task = args.task or _CONTINUATION_TASK
        print(f"Resuming from: {results_base_dir}")
    else:
        results_base_dir = os.path.join("results", f"freephdlabor_{timestamp}")
        os.makedirs(results_base_dir, exist_ok=True)
        task = args.task or _DEFAULT_TASK
        print(f"Created workspace: {results_base_dir}")

    os.environ["RESULTS_BASE_DIR"] = results_base_dir

    # --- Token tracking ---
    run_id = f"{timestamp}_{os.getpid()}"
    token_file = initialize_run_token_tracker(
        workspace_dir=results_base_dir, run_id=run_id, reset=True,
    )
    print(f"Token tracker initialized: {token_file}")

    print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
    print("Pipeline mode: full_research (deterministic)")
    if args.enable_math_agents:
        print("Math agent workflow enabled.")

    pipeline_stages = build_pipeline_stages(args.enable_math_agents)
    try:
        start_stage_index = (
            _resolve_start_stage_index(args.start_from_stage, pipeline_stages)
            if args.start_from_stage
            else 0
        )
    except ValueError as e:
        print(str(e))
        return 1

    if args.start_from_stage:
        resolved_stage = pipeline_stages[start_stage_index]
        print(
            f"Stage-based resume requested: start from '{resolved_stage}' "
            f"(index {start_stage_index})."
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

    os.environ["FREEPHDLABOR_REQUIRE_PDF"] = "1" if args.require_pdf else "0"
    os.environ["FREEPHDLABOR_ENFORCE_PAPER_ARTIFACTS"] = "1" if enforce_paper_artifacts else "0"
    os.environ["FREEPHDLABOR_REQUIRE_EXPERIMENT_PLAN"] = "1" if require_experiment_plan else "0"
    os.environ["FREEPHDLABOR_ENFORCE_EDITORIAL_ARTIFACTS"] = (
        "1" if enforce_editorial_artifacts else "0"
    )

    if enforce_paper_artifacts:
        print("Paper artifact gate: " + ", ".join(required_paper_artifacts))

    # --- LaTeX prereq check ---
    require_latex = enforce_paper_artifacts or args.require_pdf or enforce_editorial_artifacts
    if require_latex:
        pdflatex_path, bibtex_path, latex_error = check_latex_prereqs()
        if latex_error:
            print(f"Missing LaTeX prerequisites.\n{latex_error}")
            return 1
        os.environ["FREEPHDLABOR_PDFLATEX_PATH"] = pdflatex_path
        os.environ["FREEPHDLABOR_BIBTEX_PATH"] = bibtex_path
        print(f"LaTeX toolchain: pdflatex={pdflatex_path}, bibtex={bibtex_path}")

    # --- Model setup ---
    budget_config = llm_config.get("budget", {}) if llm_config else {}
    model = create_model(
        model_name, reasoning_effort, verbosity, budget_tokens,
        effort=effort,
        budget_config=budget_config, budget_dir=results_base_dir,
    )
    print(f"Created model: {getattr(model, 'model', model_name)}")

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

        print(f"Counsel mode enabled — {len(model_specs or DEFAULT_COUNSEL_MODEL_SPECS)} models, "
              f"{max_debate_rounds} debate rounds per stage.")
        counsel_models_list = create_counsel_models(
            budget_config=budget_config,
            budget_dir=results_base_dir,
            model_specs=model_specs,
        )
        os.environ["FREEPHDLABOR_COUNSEL_MAX_DEBATE_ROUNDS"] = str(max_debate_rounds)

    try:
        graph, checkpointer = build_research_graph(
            model=model,
            workspace_dir=results_base_dir,
            essential_imports=essential_imports,
            require_pdf=args.require_pdf,
            enforce_paper_artifacts=enforce_paper_artifacts,
            require_experiment_plan=require_experiment_plan,
            enable_math_agents=args.enable_math_agents,
            enforce_editorial_artifacts=enforce_editorial_artifacts,
            min_review_score=args.min_review_score,
            pipeline_mode=effective_pipeline_mode,
            followup_max_iterations=args.followup_max_iterations,
            manager_max_steps=args.manager_max_steps if args.manager_max_steps else 50,
            counsel_models=counsel_models_list,
        )

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
            "validation_results": {},
            "interrupt_instruction": None,
            "theory_track_status": None,
            "experiment_track_status": None,
            "track_decomposition": None,
            "finished": False,
        }

        # Use workspace_dir as thread_id for default resumability.
        # For stage-based resume, create a fresh thread in the same workspace so
        # previous checkpoint state does not override the requested stage index.
        if args.start_from_stage:
            canonical_stage = _canonical_stage_name(args.start_from_stage)
            thread_id = f"{results_base_dir}::stage_resume::{canonical_stage}::{timestamp}"
        else:
            thread_id = results_base_dir
        run_config = {"configurable": {"thread_id": thread_id}}

        print(f"\n{'='*50}\nRunning LangGraph research pipeline...\nTask: {task}\n{'='*50}")

        final_state = graph.invoke(initial_state, config=run_config)

        result = final_state.get("agent_outputs", {})
        result = sanitize_result_payload(
            result=str(result),
            workspace_dir=results_base_dir,
            required_artifacts=required_paper_artifacts,
        )
        if isinstance(result, dict) and result.get("status") == "incomplete":
            print(
                "Run marked incomplete — missing: "
                + ", ".join(result.get("missing_required_artifacts", []))
            )

        print(f"\n{'='*50}\nTask finished.\n{'='*50}")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0
