#!/bin/bash
# =============================================================================
# MIT Engaging — Orchestrator SLURM Job (CPU-only, Tier 1)
#
# Usage:
#   ./scripts/submit_orchestrator.sh [--task "..." ] [extra flags...]
#   # or directly:
#   cd phdlabor-1 && mkdir -p slurm_outputs && sbatch scripts/launch_orchestrator_engaging.sh
#
# The orchestrator makes LLM API calls over the network. It does NOT need GPU.
# When the pipeline reaches the experimentation stage, GPU experiments are
# automatically submitted as separate SLURM jobs (Tier 2) to the GPU partition
# defined in engaging_config.yaml (default: pi_tpoggio, A100x8).
# =============================================================================

#SBATCH --job-name=consortium_orch
#SBATCH --partition=sched_mit_hill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_outputs/orch_%j.out
#SBATCH --error=slurm_outputs/orch_%j.err

set -euo pipefail

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

mkdir -p slurm_outputs

echo "========================================"
echo "Consortium Orchestrator — MIT Engaging"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Repo:      $REPO_DIR"
echo "========================================"

# --- Environment ---
module load miniforge/25.11.0-0
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate /home/mabdel03/conda_envs/consortium

echo "Python:    $(which python) ($(python --version 2>&1))"

# --- Enable SLURM experiment submission ---
export CONSORTIUM_SLURM_ENABLED=1
export ENGAGING_CONFIG="$REPO_DIR/engaging_config.yaml"

# --- Task ---
# RESEARCH_TASK can be set as an env var before sbatch, or passed via --task flag
if [[ -z "${RESEARCH_TASK:-}" ]]; then
  echo "No RESEARCH_TASK env var set. Using --task from CLI args or default task."
fi

# --- Process wrapper (avoids AI-Scientist-v2 cleanup killing the orchestrator) ---
PYTHON_PATH=$(which python)
RUNNER_SCRIPT=$(mktemp /tmp/orch_runner_XXXXXX)
cat > "$RUNNER_SCRIPT" << EOF
#!/bin/bash
exec $PYTHON_PATH launch_multiagent.py "\$@"
EOF
chmod +x "$RUNNER_SCRIPT"
trap "rm -f $RUNNER_SCRIPT" EXIT

# --- Launch ---
echo "=== Pipeline started at: $(date) ==="

if [[ -n "${RESEARCH_TASK:-}" ]]; then
  "$RUNNER_SCRIPT" --task "$RESEARCH_TASK" "$@"
else
  "$RUNNER_SCRIPT" "$@"
fi
exit_code=$?

echo "=== Pipeline finished with exit code $exit_code at: $(date) ==="
exit $exit_code
