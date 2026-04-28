"""openpi setup — interactive first-time setup wizard."""

from __future__ import annotations

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openpi.core.config_manager import get_config_dir, save_config
from openpi.core.env_manager import (
    API_KEYS,
    NOTIFICATION_KEYS,
    load_env_vars,
    save_env_file,
)
from openpi.core.platform_detect import detect, detect_consortium

console = Console()


# Model choices for the wizard
MODEL_OPTIONS = [
    {"name": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6  (balanced, recommended)", "cost": "$$"},
    {"name": "claude-opus-4-6", "label": "Claude Opus 4.6    (highest quality)", "cost": "$$$"},
    {"name": "gpt-5", "label": "GPT-5              (strong reasoning)", "cost": "$$$"},
    {"name": "gpt-5-mini", "label": "GPT-5 Mini         (cost-effective)", "cost": "$"},
    {"name": "gemini-3-pro-preview", "label": "Gemini 3 Pro       (2M context window)", "cost": "$$"},
    {"name": "deepseek-chat", "label": "DeepSeek Chat      (budget-friendly)", "cost": "$"},
]


@click.command()
@click.option("--non-interactive", is_flag=True, help="Skip prompts, use defaults.")
@click.pass_context
def setup(ctx: click.Context, non_interactive: bool) -> None:
    """Interactive first-time setup wizard.

    Walks you through:
      - Platform detection
      - API key configuration
      - Default model selection
      - Notification setup (optional)
    """
    config_dir_override = ctx.obj.get("config_dir")
    config_dir = get_config_dir(config_dir_override)

    console.print(Panel(
        "[bold]Welcome to OpenPI Setup[/]\n\n"
        "This wizard will configure your environment for running\n"
        "AI-powered multi-agent research pipelines.",
        title="openpi setup",
        border_style="blue",
    ))

    # ── Step 1: Platform Detection ──────────────────────────────────
    console.print("\n[bold cyan]Step 1:[/] Detecting your platform...\n")
    info = detect()
    has_consortium, consortium_path = detect_consortium()

    table = Table(show_header=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("OS", f"{info.os_name} {info.arch}" + (" (WSL)" if info.wsl else ""))
    table.add_row("Python", f"{info.python_version} ({info.python_path})")
    table.add_row("Environment", _env_label(info))
    table.add_row("Consortium", f"[green]installed[/] ({consortium_path})" if has_consortium else "[yellow]not installed[/]")
    table.add_row("SLURM", "[green]available[/]" if info.has_slurm else "[dim]not detected[/]")
    table.add_row("LaTeX", "[green]available[/]" if info.has_pdflatex else "[dim]not installed[/] (needed for PDF output)")
    console.print(table)

    if info.missing:
        for m in info.missing:
            console.print(f"  [red]![/] {m}")
        raise SystemExit(1)

    if not has_consortium:
        console.print(
            "\n[yellow]Consortium engine not installed.[/] "
            "Install it with:\n"
            "  [bold]pip install -e /path/to/OpenPI[/]\n"
        )

    # ── Step 2: API Keys ────────────────────────────────────────────
    console.print("\n[bold cyan]Step 2:[/] Configure API keys\n")
    console.print("  OpenPI needs at least one LLM API key. Counsel mode requires 3 providers.\n")

    existing = load_env_vars(config_dir_override)
    env_vars = dict(existing)

    if non_interactive:
        console.print("  [dim]Skipping prompts (--non-interactive). Using existing keys.[/]")
    else:
        for key_info in API_KEYS:
            _prompt_api_key(key_info, env_vars)

    # ── Step 3: Model Selection ─────────────────────────────────────
    console.print("\n[bold cyan]Step 3:[/] Choose your default model\n")

    if non_interactive:
        default_model = "claude-sonnet-4-6"
    else:
        choices = [
            questionary.Choice(title=m["label"], value=m["name"])
            for m in MODEL_OPTIONS
        ]
        default_model = questionary.select(
            "Default model:",
            choices=choices,
            default="claude-sonnet-4-6",
        ).ask()
        if default_model is None:
            raise SystemExit(1)

    # ── Step 4: Notifications (optional) ────────────────────────────
    console.print("\n[bold cyan]Step 4:[/] Notifications (optional)\n")

    if not non_interactive:
        setup_notif = questionary.confirm(
            "Set up Telegram/Slack notifications?",
            default=False,
        ).ask()
        if setup_notif:
            for key_info in NOTIFICATION_KEYS:
                _prompt_env_var(key_info, env_vars)

    # ── Step 5: Save Configuration ──────────────────────────────────
    console.print("\n[bold cyan]Step 5:[/] Saving configuration...\n")

    # Save .env
    env_path = save_env_file(env_vars, config_dir_override)
    console.print(f"  API keys saved to [bold]{env_path}[/] (chmod 600)")

    # Save config.yaml
    config_data = {
        "model": default_model,
        "preset": "standard",
        "output_format": "markdown",
        "budget_usd": 25,
        "autonomous_mode": True,
        "notifications": {
            "telegram": {
                "enabled": bool(env_vars.get("TELEGRAM_BOT_TOKEN")),
            },
            "slack": {
                "enabled": bool(env_vars.get("SLACK_WEBHOOK_URL")),
            },
        },
    }
    cfg_path = save_config(config_data, config_dir_override)
    console.print(f"  Config saved to [bold]{cfg_path}[/]")

    # ── Done ────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold green]Setup complete![/]\n\n"
        "Next steps:\n"
        f"  [bold]openpi doctor[/]                     Verify your environment\n"
        f"  [bold]openpi run \"your question\"[/]         Start a research run\n"
        f"  [bold]openpi run --preset quick --dry-run[/] Test without cost",
        border_style="green",
    ))


def _env_label(info) -> str:
    """Human-readable environment label."""
    if info.active_conda_env:
        return f"conda ({info.active_conda_env})"
    if info.has_venv:
        return "venv (active)"
    if info.has_conda:
        return "conda (available, no env active)"
    return "system Python"


def _prompt_api_key(key_info: dict, env_vars: dict) -> None:
    """Prompt for a single API key."""
    name = key_info["name"]
    env_var = key_info["env_var"]
    level = key_info["level"]
    desc = key_info["description"]

    existing = env_vars.get(env_var) or __import__("os").environ.get(env_var, "")
    masked = _mask_key(existing) if existing else ""

    level_tag = {
        "required": "[red](required)[/]",
        "recommended": "[yellow](recommended)[/]",
        "optional": "[dim](optional)[/]",
    }.get(level, "")

    if existing:
        console.print(f"  {name} {level_tag}: [green]configured[/] ({masked})")
        change = questionary.confirm(f"  Change {name} key?", default=False).ask()
        if not change:
            return

    console.print(f"  {name} {level_tag}: {desc}")
    value = questionary.password(f"  Enter {env_var}:").ask()
    if value and value.strip():
        env_vars[env_var] = value.strip()
    elif level == "required":
        console.print(f"  [yellow]Skipped[/] — you'll need at least one LLM key to run.")


def _prompt_env_var(key_info: dict, env_vars: dict) -> None:
    """Prompt for a generic env var (notifications etc)."""
    name = key_info["name"]
    env_var = key_info["env_var"]
    desc = key_info["description"]

    existing = env_vars.get(env_var, "")
    console.print(f"  {name}: {desc}")

    value = questionary.text(
        f"  {env_var}:",
        default=existing,
    ).ask()
    if value and value.strip():
        env_vars[env_var] = value.strip()


def _mask_key(key: str) -> str:
    """Mask an API key, showing only first 4 and last 4 chars."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
