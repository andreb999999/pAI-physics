"""Top-level click group for the openpi CLI."""

import click

from openpi import __version__


class OpenPIGroup(click.Group):
    """Custom group that shows a helpful message if consortium is not installed."""

    def invoke(self, ctx: click.Context) -> None:
        super().invoke(ctx)


@click.group(cls=OpenPIGroup)
@click.version_option(__version__, prog_name="openpi")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output.")
@click.option(
    "--config-dir",
    type=click.Path(),
    default=None,
    help="Override config directory (default: ~/.openpi).",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config_dir: str | None) -> None:
    """OpenPI — AI-powered multi-agent research pipeline.

    Run autonomous research from question to paper with one command.

    \b
    Quick start:
      openpi setup                        # First-time setup
      openpi run "your research question" # Run a research pipeline
      openpi doctor                       # Check your environment
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_dir"] = config_dir


# Register subcommands (lazy imports to keep startup fast)
def _register_commands() -> None:
    from openpi.commands.run import run
    from openpi.commands.setup import setup
    from openpi.commands.doctor import doctor
    from openpi.commands.config_cmd import config
    from openpi.commands.runs import runs
    from openpi.commands.resume import resume
    from openpi.commands.campaign import campaign
    from openpi.commands.budget import budget
    from openpi.commands.notify import notify
    from openpi.commands.openclaw import openclaw
    from openpi.commands.extras import install

    cli.add_command(run)
    cli.add_command(setup)
    cli.add_command(doctor)
    cli.add_command(config)
    cli.add_command(runs)
    cli.add_command(resume)
    cli.add_command(campaign)
    cli.add_command(budget)
    cli.add_command(notify)
    cli.add_command(openclaw)
    cli.add_command(install)


_register_commands()
