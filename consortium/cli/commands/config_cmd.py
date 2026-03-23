"""msc config — view and manage configuration."""

from __future__ import annotations

import os
import subprocess

import click
from rich.console import Console
from rich.syntax import Syntax

import yaml

from consortium.cli.core.config_manager import (
    get_config_dir,
    get_value,
    load_config,
    set_value,
)

console = Console()


@click.group()
def config() -> None:
    """View and manage PoggioAI/MSc configuration.

    \b
    Examples:
      msc config list                  # Show all settings
      msc config get model             # Get a specific value
      msc config set model gpt-5       # Change a setting
      msc config edit                  # Open in $EDITOR
    """


@config.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """Show all configuration values."""
    config_dir = ctx.obj.get("config_dir")
    cfg = load_config(config_dir)
    rendered = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    console.print(Syntax(rendered, "yaml", theme="monokai", line_numbers=False))


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
    """Get a configuration value by key.

    Use dot notation for nested keys: notifications.telegram.enabled
    """
    config_dir = ctx.obj.get("config_dir")
    value = get_value(key, config_dir)
    if value is None:
        console.print(f"[yellow]Key '{key}' not found.[/]")
        raise SystemExit(1)
    console.print(f"{key} = {value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value.

    \b
    Examples:
      msc config set model claude-opus-4-6
      msc config set budget_usd 50
      msc config set notifications.telegram.enabled true
    """
    config_dir = ctx.obj.get("config_dir")
    set_value(key, value, config_dir)
    console.print(f"[blue]Set[/] {key} = {value}")


@config.command("edit")
@click.pass_context
def config_edit(ctx: click.Context) -> None:
    """Open config file in $EDITOR."""
    config_dir = ctx.obj.get("config_dir")
    cfg_dir = get_config_dir(config_dir)
    cfg_path = cfg_dir / "config.yaml"

    if not cfg_path.exists():
        console.print("[yellow]No config file yet.[/] Run [bold]msc setup[/] first.")
        raise SystemExit(1)

    editor = os.environ.get("EDITOR", "vi")
    subprocess.run([editor, str(cfg_path)])


@config.command("path")
@click.pass_context
def config_path(ctx: click.Context) -> None:
    """Show the config directory path."""
    config_dir = ctx.obj.get("config_dir")
    cfg_dir = get_config_dir(config_dir)
    console.print(str(cfg_dir))
