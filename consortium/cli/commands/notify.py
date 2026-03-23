"""msc notify — configure and test notifications."""

from __future__ import annotations

import click
import questionary
from rich.console import Console

from consortium.cli.core.config_manager import get_config_dir, set_value
from consortium.cli.core.env_manager import (
    NOTIFICATION_KEYS,
    load_env_vars,
    save_env_file,
)

console = Console()


@click.group()
def notify() -> None:
    """Configure pipeline notifications (Telegram, Slack, etc.).

    \b
    Examples:
      msc notify setup    # Interactive notification setup
      msc notify test     # Send a test message
    """


@notify.command("setup")
@click.pass_context
def notify_setup(ctx: click.Context) -> None:
    """Configure notification channels interactively."""
    config_dir = ctx.obj.get("config_dir")
    env_vars = load_env_vars(config_dir)

    console.print("[bold]Notification Setup[/]\n")

    for key_info in NOTIFICATION_KEYS:
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

    env_path = save_env_file(env_vars, config_dir)
    console.print(f"\n[blue]Saved to {env_path}[/]")

    # Update config
    if env_vars.get("TELEGRAM_BOT_TOKEN"):
        set_value("notifications.telegram.enabled", "true", config_dir)
    if env_vars.get("SLACK_WEBHOOK_URL"):
        set_value("notifications.slack.enabled", "true", config_dir)


@notify.command("test")
@click.option("--channel", type=click.Choice(["telegram", "slack"]), default="telegram")
@click.pass_context
def notify_test(ctx: click.Context, channel: str) -> None:
    """Send a test notification."""
    import os
    from consortium.cli.core.env_manager import inject_env

    config_dir = ctx.obj.get("config_dir")
    inject_env(config_dir)

    if channel == "telegram":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            console.print("[red]Telegram not configured.[/] Run [bold]msc notify setup[/]")
            return

        import urllib.request
        import json

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": "PoggioAI/MSc test notification"}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    console.print("[blue]Test message sent to Telegram![/]")
                else:
                    console.print(f"[red]Telegram API returned {resp.status}[/]")
        except Exception as e:
            console.print(f"[bold white on red] Failed [/] {e}")

    elif channel == "slack":
        webhook = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook:
            console.print("[red]Slack not configured.[/] Run [bold]msc notify setup[/]")
            return

        import urllib.request
        import json

        data = json.dumps({"text": "PoggioAI/MSc test notification"}).encode()
        req = urllib.request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    console.print("[blue]Test message sent to Slack![/]")
                else:
                    console.print(f"[red]Slack webhook returned {resp.status}[/]")
        except Exception as e:
            console.print(f"[bold white on red] Failed [/] {e}")
