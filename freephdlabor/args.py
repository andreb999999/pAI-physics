"""
Args for system initialization
"""
import argparse
from .utils import AVAILABLE_MODELS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smolagents Research System - Multi-Agent AI Research Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new workspace (uses default: gpt-5 with high reasoning effort)
  python launch_multiagent.py --task "Research transformer attention mechanisms"


  # Resume from existing workspace
  python launch_multiagent.py --resume results/freephdlabor_20250929_143022/ --model claude-sonnet-4-5
  python launch_multiagent.py --resume results/freephdlabor_20250929_143022/ --task "Continue writing the conclusion section"
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=AVAILABLE_MODELS,
        default=None,
        help="LLM model to use for all agents (default: gpt-5, overrides .llm_config.yaml)"
    )

    parser.add_argument(
        "--interpreter",
        type=str,
        default="python",
        help="Python interpreter path for experiment execution"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--log-to-files",
        dest="log_to_files",
        action="store_true",
        default=None,
        help="Redirect stdout/stderr to logs/freephdlabor_<timestamp>.{out,err} (default: on)"
    )

    parser.add_argument(
        "--no-log-to-files",
        dest="log_to_files",
        action="store_false",
        help="Do not redirect stdout/stderr to log files"
    )

    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="GPT-5 reasoning effort level (default: high, overrides .llm_config.yaml)"
    )

    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="GPT-5 verbosity level (default: medium, overrides .llm_config.yaml)"
    )

    parser.add_argument(
        "--callback_host",
        type=str,
        default="127.0.0.1",
        help="Host for the callback server"
    )

    parser.add_argument(
        "--callback_port",
        type=int,
        default=5001,
        help="Port for the callback server"
    )

    parser.add_argument(
        "--enable-planning",
        action="store_true",
        help="Enable planning feature for research agents (creates systematic step-by-step plans)"
    )

    parser.add_argument(
        "--planning-interval",
        type=int,
        default=3,
        help="Interval for planning steps (e.g., 3 = replan every 3 steps). Only used if --enable-planning is set."
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from an existing workspace directory (e.g., results/20250128_143022_task_name/). "
             "If specified, will continue from the existing workspace instead of creating a new one."
    )

    parser.add_argument(
        "--task",
        type=str,
        help="The research task description. Can be used with --resume to continue a previous task."
    )

    parser.add_argument(
        "--require-pdf",
        action="store_true",
        help="Require final_paper.pdf before the manager can terminate successfully."
    )

    parser.add_argument(
        "--enforce-paper-artifacts",
        action="store_true",
        help="Enforce paper artifact checks (final_paper.tex; optionally final_paper.pdf and experiments_to_run_later.md)."
    )

    parser.add_argument(
        "--manager-max-steps",
        type=int,
        default=None,
        help="Override manager agent max steps for this run."
    )

    parser.add_argument(
        "--enable-math-agents",
        action="store_true",
        help="Enable theorem-oriented math agents (proposer, prover, rigorous verifier, empirical verifier)."
    )

    parser.add_argument(
        "--require-experiment-plan",
        action="store_true",
        help="When paper artifact enforcement is enabled, also require experiments_to_run_later.md."
    )

    parser.add_argument(
        "--enforce-editorial-artifacts",
        action="store_true",
        help="Require editorial workflow artifacts (style guide, intro skeleton, review verdict, etc.).",
    )

    parser.add_argument(
        "--min-review-score",
        type=int,
        default=8,
        help="Minimum reviewer overall_score required by the strict review gate.",
    )

    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["default", "full_research", "quick"],
        default="default",
        help="Workflow mode. Use 'full_research' for the full 8-step literature/plan/execution/writeup pipeline.",
    )

    parser.add_argument(
        "--followup-max-iterations",
        type=int,
        default=3,
        help="Maximum Step 6 <-> 6.2 follow-up loops in full_research mode.",
    )

    return parser.parse_args()
