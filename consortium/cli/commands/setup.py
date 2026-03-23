"""msc setup — interactive first-time setup wizard."""

from __future__ import annotations

import os
from pathlib import Path

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from consortium.cli.core.config_manager import get_config_dir, save_config
from consortium.cli.core.env_manager import (
    API_KEYS,
    NOTIFICATION_KEYS,
    load_env_vars,
    save_env_file,
)
from consortium.cli.core.platform_detect import detect, detect_consortium

console = Console()


# Model choices with quality/speed/cost ratings
MODEL_OPTIONS = [
    {
        "name": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
        "quality": 4, "speed": 4, "cost": "$10-25",
        "desc": "Balanced, recommended",
    },
    {
        "name": "claude-opus-4-6",
        "label": "Claude Opus 4.6",
        "quality": 5, "speed": 2, "cost": "$40-100",
        "desc": "Highest quality",
    },
    {
        "name": "gpt-5",
        "label": "GPT-5",
        "quality": 5, "speed": 3, "cost": "$30-80",
        "desc": "Strong reasoning",
    },
    {
        "name": "gpt-5-mini",
        "label": "GPT-5 Mini",
        "quality": 3, "speed": 5, "cost": "$2-5",
        "desc": "Cost-effective",
    },
    {
        "name": "gemini-3-pro-preview",
        "label": "Gemini 3 Pro",
        "quality": 4, "speed": 3, "cost": "$15-40",
        "desc": "2M context window",
    },
    {
        "name": "deepseek-chat",
        "label": "DeepSeek Chat",
        "quality": 3, "speed": 4, "cost": "$1-3",
        "desc": "Budget-friendly",
    },
]


def _rating_bar(n: int, max_n: int = 5) -> str:
    """Render a rating as filled/empty blocks."""
    return "\u2588" * n + "\u2591" * (max_n - n)


def _print_model_table() -> None:
    """Display a rich table of available models with ratings."""
    table = Table(
        title="Available Models",
        border_style="bright_black",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Model", style="bold white", min_width=20)
    table.add_column("Quality", style="blue", justify="center")
    table.add_column("Speed", style="blue", justify="center")
    table.add_column("Cost/run", style="dim white", justify="right")
    table.add_column("", style="dim white")

    for m in MODEL_OPTIONS:
        table.add_row(
            m["label"],
            _rating_bar(m["quality"]),
            _rating_bar(m["speed"]),
            m["cost"],
            m["desc"],
        )
    console.print(table)


def _test_api_connectivity(env_vars: dict) -> None:
    """Test connectivity to configured API providers."""
    import urllib.request

    console.print("\n  [bold blue]Testing API connectivity...[/]\n")

    endpoints = {
        "ANTHROPIC_API_KEY": ("Anthropic", "https://api.anthropic.com/v1/messages"),
        "OPENAI_API_KEY": ("OpenAI", "https://api.openai.com/v1/models"),
        "GOOGLE_API_KEY": ("Google AI", "https://generativelanguage.googleapis.com/v1beta/models"),
    }

    for env_var, (name, url) in endpoints.items():
        key = env_vars.get(env_var) or os.environ.get(env_var, "")
        if not key:
            console.print(f"    {name:12s} [dim]skipped (no key)[/]")
            continue
        try:
            req = urllib.request.Request(url, method="GET")
            if "anthropic" in url:
                req.add_header("x-api-key", key)
                req.add_header("anthropic-version", "2023-06-01")
            elif "openai" in url:
                req.add_header("Authorization", f"Bearer {key}")
            elif "google" in url:
                req = urllib.request.Request(f"{url}?key={key}", method="GET")
            urllib.request.urlopen(req, timeout=5)
            console.print(f"    {name:12s} [blue]\u2713 connected[/]")
        except Exception:
            console.print(f"    {name:12s} [dim]\u2717 unreachable (key may still be valid)[/]")


def _migrate_from_openpi() -> bool:
    """Check for and offer migration from ~/.openpi to ~/.msc."""
    old_dir = Path.home() / ".openpi"
    new_dir = Path.home() / ".msc"

    if not old_dir.exists() or new_dir.exists():
        return False

    console.print(Panel(
        "[bold white]Existing OpenPI configuration detected[/]\n\n"
        f"Found config at [blue]{old_dir}[/]\n"
        "Would you like to migrate it to the new MSc format?",
        border_style="blue",
        title="Migration",
    ))

    migrate = questionary.confirm("Migrate existing configuration?", default=True).ask()
    if not migrate:
        return False

    import shutil
    new_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("config.yaml", ".env"):
        src = old_dir / fname
        if src.exists():
            shutil.copy2(src, new_dir / fname)
            console.print(f"    [blue]\u2713[/] Migrated {fname}")

    console.print(f"  [blue]Migration complete.[/] Config now at {new_dir}\n")
    return True


@click.command()
@click.option("--non-interactive", is_flag=True, help="Skip prompts, use defaults.")
@click.option("--migrate", is_flag=True, help="Migrate from ~/.openpi to ~/.msc.")
@click.pass_context
def setup(ctx: click.Context, non_interactive: bool, migrate: bool) -> None:
    """Interactive first-time setup wizard.

    Walks you through:
      - Platform detection
      - API key configuration
      - Default model selection
      - Notification setup (optional)
    """
    config_dir_override = ctx.obj.get("config_dir")

    # Banner
    try:
        from consortium.cli.banner import print_banner
        print_banner()
    except ImportError:
        pass

    console.print(Panel(
        "[bold white]Welcome to PoggioAI/MSc[/]\n\n"
        "[dim white]This wizard will configure your environment for running\n"
        "AI-powered multi-agent research pipelines.\n"
        "Built by the Poggio Lab at MIT.[/]",
        title="[bold blue]msc setup[/]",
        border_style="blue",
    ))

    # Check for migration
    if migrate or not (Path.home() / ".msc").exists():
        _migrate_from_openpi()

    config_dir = get_config_dir(config_dir_override)

    # ── Step 1: Platform Detection ──────────────────────────────────
    console.print("\n[bold blue]Step 1:[/] [white]Detecting your platform...[/]\n")
    info = detect()
    has_consortium, consortium_path = detect_consortium()

    table = Table(show_header=False, padding=(0, 2), border_style="bright_black")
    table.add_column("", style="bold white")
    table.add_column("")
    table.add_row("OS", f"{info.os_name} {info.arch}" + (" (WSL)" if info.wsl else ""))
    table.add_row("Python", f"{info.python_version} ({info.python_path})")
    table.add_row("Environment", _env_label(info))
    table.add_row("Engine", f"[blue]installed[/] ({consortium_path})" if has_consortium else "[dim]not installed[/]")
    table.add_row("SLURM", "[blue]available[/]" if info.has_slurm else "[dim]not detected[/]")
    table.add_row("LaTeX", "[blue]available[/]" if info.has_pdflatex else "[dim]not installed[/] (needed for PDF output)")
    console.print(table)

    if info.missing:
        for m in info.missing:
            console.print(f"  [bold white on red] ! [/] {m}")
        raise SystemExit(1)

    if not has_consortium:
        console.print(
            "\n[dim]Engine not installed.[/] "
            "Install it with:\n"
            "  [bold white]pip install -e .[/]\n"
        )

    # ── Step 2: API Keys ────────────────────────────────────────────
    console.print("\n[bold blue]Step 2:[/] [white]Configure API keys[/]\n")
    console.print("  [dim]MSc needs at least one LLM API key. Counsel mode requires 3 providers.[/]\n")

    existing = load_env_vars(config_dir_override)
    env_vars = dict(existing)

    if non_interactive:
        console.print("  [dim]Skipping prompts (--non-interactive). Using existing keys.[/]")
    else:
        for key_info in API_KEYS:
            _prompt_api_key(key_info, env_vars)

    # Connectivity test
    if not non_interactive:
        _test_api_connectivity(env_vars)

    # ── Step 3: Model Selection ─────────────────────────────────────
    console.print("\n[bold blue]Step 3:[/] [white]Choose your default model[/]\n")
    _print_model_table()
    console.print()

    if non_interactive:
        default_model = "claude-sonnet-4-6"
    else:
        choices = [
            questionary.Choice(title=f"{m['label']:20s}  {m['desc']}", value=m["name"])
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
    console.print("\n[bold blue]Step 4:[/] [white]Notifications (optional)[/]\n")

    if not non_interactive:
        setup_notif = questionary.confirm(
            "Set up Telegram/Slack notifications?",
            default=False,
        ).ask()
        if setup_notif:
            for key_info in NOTIFICATION_KEYS:
                _prompt_env_var(key_info, env_vars)

    # ── Step 5: Save Configuration ──────────────────────────────────
    console.print("\n[bold blue]Step 5:[/] [white]Saving configuration...[/]\n")

    # Save .env
    env_path = save_env_file(env_vars, config_dir_override)
    console.print(f"  [blue]\u2713[/] API keys saved to [bold white]{env_path}[/] (chmod 600)")

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
    console.print(f"  [blue]\u2713[/] Config saved to [bold white]{cfg_path}[/]")

    # ── Done ────────────────────────────────────────────────────────
    console.print(Panel(
        "[bold blue]\u2713 Setup complete![/]\n\n"
        "[white]Next steps:[/]\n"
        f"  [bold white]msc doctor[/]                     [dim]Verify your environment[/]\n"
        f"  [bold white]msc run \"your question\"[/]         [dim]Start a research run[/]\n"
        f"  [bold white]msc run --preset quick --dry-run[/] [dim]Test without cost[/]",
        border_style="blue",
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
        "required": "[bold white](required)[/]",
        "recommended": "[dim](recommended)[/]",
        "optional": "[dim](optional)[/]",
    }.get(level, "")

    if existing:
        console.print(f"  {name} {level_tag}: [blue]\u2713 configured[/] ({masked})")
        change = questionary.confirm(f"  Change {name} key?", default=False).ask()
        if not change:
            return

    console.print(f"  {name} {level_tag}: {desc}")
    value = questionary.password(f"  Enter {env_var}:").ask()
    if value and value.strip():
        env_vars[env_var] = value.strip()
    elif level == "required":
        console.print(f"  [dim]Skipped — you'll need at least one LLM key to run.[/]")


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
