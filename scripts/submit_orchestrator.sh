#!/bin/bash
# =============================================================================
# Submit the Consortium orchestrator to SLURM on MIT Engaging.
#
# Usage:
#   ./scripts/submit_orchestrator.sh [sbatch flags and/or consortium flags]
#
# Examples:
#   # Default task (from .llm_config.yaml)
#   ./scripts/submit_orchestrator.sh --no-counsel
#
#   # Custom task
#   RESEARCH_TASK="Investigate X..." ./scripts/submit_orchestrator.sh --no-counsel
#
#   # Task from file
#   RESEARCH_TASK="$(cat my_task.txt)" ./scripts/submit_orchestrator.sh
#
#   # Override partition (e.g., longer time limit)
#   ./scripts/submit_orchestrator.sh --partition=mit_preemptable --time=2-00:00:00
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Ensure output directory exists (SLURM needs it before job starts)
mkdir -p slurm_outputs

echo "Submitting consortium orchestrator to SLURM..."
echo "  Repo: $REPO_DIR"

# Export RESEARCH_TASK so the SLURM script can pick it up
export RESEARCH_TASK="${RESEARCH_TASK:-}"

sbatch "$SCRIPT_DIR/launch_orchestrator_engaging.sh" "$@"
