"""Translate presets and CLI options into consortium argv flags."""

from __future__ import annotations

from consortium.cli.core.presets import Preset, PRESETS


def preset_to_argv(preset: Preset, task: str, **overrides: object) -> list[str]:
    """Convert a preset + task + overrides into a consortium CLI argv list.

    Returns a list like ["consortium", "--task", "...", "--model", "...", ...].
    """
    argv = ["consortium"]

    # Task
    argv.extend(["--task", task])

    # Model (overridable)
    model = str(overrides.pop("model", preset.model))
    argv.extend(["--model", model])

    # Budget is set via .llm_config.yaml (written by run.py), not CLI flag.
    overrides.pop("budget_usd", None)  # consume but don't emit

    # Output format (overridable)
    output_format = str(overrides.pop("output_format", preset.output_format))
    argv.extend(["--output-format", output_format])

    # Boolean flags
    if overrides.pop("enable_counsel", preset.enable_counsel):
        argv.append("--enable-counsel")
    if overrides.pop("no_counsel", preset.no_counsel):
        argv.append("--no-counsel")
    if overrides.pop("enable_math_agents", preset.enable_math_agents):
        argv.append("--enable-math-agents")
    if overrides.pop("enable_tree_search", preset.enable_tree_search):
        argv.append("--enable-tree-search")
    if overrides.pop("adversarial_verification", preset.adversarial_verification):
        argv.append("--adversarial-verification")
    if overrides.pop("enable_planning", preset.enable_planning):
        argv.append("--enable-planning")
    if overrides.pop("enforce_paper_artifacts", preset.enforce_paper_artifacts):
        argv.append("--enforce-paper-artifacts")
    if overrides.pop("enforce_editorial_artifacts", preset.enforce_editorial_artifacts):
        argv.append("--enforce-editorial-artifacts")
    if overrides.pop("autonomous_mode", preset.autonomous_mode):
        argv.append("--autonomous-mode")

    # Don't log to files — we handle output display
    argv.append("--no-log-to-files")

    # Dry run
    if overrides.pop("dry_run", False):
        argv.append("--dry-run")

    # Resume
    resume = overrides.pop("resume", None)
    if resume:
        argv.extend(["--resume", str(resume)])

    # Start from stage
    start_from = overrides.pop("start_from_stage", None)
    if start_from:
        argv.extend(["--start-from-stage", str(start_from)])

    # Mode override
    mode = overrides.pop("mode", None)
    if mode:
        argv.extend(["--mode", str(mode)])

    # Task file (alternative to inline task)
    task_file = overrides.pop("task_file", None)
    if task_file:
        # Replace the inline task with file contents
        try:
            with open(str(task_file), "r") as f:
                file_task = f.read().strip()
            # Find and replace the task in argv
            idx = argv.index("--task")
            argv[idx + 1] = file_task
        except (OSError, ValueError):
            pass

    # Max run seconds
    max_run = overrides.pop("max_run_seconds", None)
    if max_run:
        argv.extend(["--max-run-seconds", str(max_run)])

    return argv


def build_argv(
    task: str,
    preset_name: str = "standard",
    **overrides: object,
) -> list[str]:
    """Build consortium argv from a preset name + overrides.

    This is the main entry point for the run command.
    """
    preset = PRESETS.get(preset_name)
    if not preset:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from: {', '.join(PRESETS)}")
    return preset_to_argv(preset, task, **overrides)
