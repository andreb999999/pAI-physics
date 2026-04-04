#!/bin/bash

# =============================================================================
# SLURM CONFIGURATION — Orchestrator (CPU-only, Tier 1)
# The orchestrator makes LLM API calls over the network; it does NOT need GPU.
# GPU experiments are submitted as separate SLURM jobs from within the pipeline.
# =============================================================================
#SBATCH --job-name=consortium_orch       # Job name
#SBATCH --partition=batch                 # Override via: sbatch --partition=YOUR_PARTITION
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH --cpus-per-task=4                # CPUs per task (API calls + local processing)
#SBATCH --time=12:00:00                  # Time limit (partition max)
#SBATCH --mem=32G                        # Memory allocation
#SBATCH --output=slurm_outputs/orch_%j.out   # Standard output log
#SBATCH --error=slurm_outputs/orch_%j.err    # Error log

# =============================================================================
# JOB INFORMATION
# =============================================================================
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $PWD"

# Forward any extra arguments passed to this script to the Python program
if [[ $# -gt 0 ]]; then
	echo "Extra args to Python: $*"
fi

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Determine repo root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
echo "Repo directory: $REPO_DIR"

# Create slurm output directory (must exist for SLURM logs)
mkdir -p "$REPO_DIR/slurm_outputs"

# Load conda module and activate environment
module load miniforge/25.11.0-0 2>/dev/null || true

echo "Activating conda environment..."
# Read conda paths from engaging_config.yaml if available
_ENGAGING_CFG="$REPO_DIR/engaging_config.yaml"
_CONDA_INIT="${CONDA_INIT_SCRIPT:-}"
_CONDA_PREFIX="${CONDA_ENV_PREFIX:-}"
if [ -z "$_CONDA_INIT" ] && [ -f "$_ENGAGING_CFG" ]; then
    _CONDA_INIT=$(grep 'conda_init_script:' "$_ENGAGING_CFG" 2>/dev/null | head -1 | sed 's/.*conda_init_script:\s*//' | sed 's/\s*#.*//' | tr -d '[:space:]' | sed 's/\${[^}]*:-\(.*\)}/\1/')
fi
if [ -z "$_CONDA_PREFIX" ] && [ -f "$_ENGAGING_CFG" ]; then
    _CONDA_PREFIX=$(grep 'conda_env_prefix:' "$_ENGAGING_CFG" 2>/dev/null | head -1 | sed 's/.*conda_env_prefix:\s*//' | sed 's/\s*#.*//' | tr -d '[:space:]' | sed 's/\${[^}]*:-\(.*\)}/\1/')
fi
if [ -n "$_CONDA_INIT" ] && [ -f "$_CONDA_INIT" ]; then
    source "$_CONDA_INIT"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda deactivate 2>/dev/null || true
if [ -n "$_CONDA_PREFIX" ]; then
    conda activate "$_CONDA_PREFIX"
else
    conda activate consortium 2>/dev/null || conda activate base
fi

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# =============================================================================
# CONSORTIUM CONFIGURATION
# =============================================================================
# Enable SLURM-based experiment execution (Tier 2 GPU jobs)
export CONSORTIUM_SLURM_ENABLED=1
export ENGAGING_CONFIG="$REPO_DIR/engaging_config.yaml"

# =============================================================================
# RESEARCH TASK DEFINITION
# =============================================================================
# Define your research task here - describe what you want the system to do
RESEARCH_TASK="${RESEARCH_TASK:-Complete a full research project on [YOUR RESEARCH TOPIC].

RESEARCH OBJECTIVES: (1) [Objective 1], (2) [Objective 2], (3) [Objective 3], ...

WORKFLOW AUTONOMY: You have full autonomy to iterate between agents if any stage reveals limitations. ...}"

# =============================================================================
# EXECUTION - Advanced wrapper to prevent process name conflicts
# =============================================================================
# Create a temporary wrapper to hide "python" from process command line
# This prevents AI-Scientist-v2's cleanup routine from accidentally killing the main process
echo "Starting multiagent system..."

# Get the active Python interpreter from conda environment
PYTHON_PATH=$(which python)

# Create temporary runner script in a private directory (avoids race on multi-user systems)
RUNNER_DIR=$(mktemp -d /tmp/multiagent_runner_XXXXXX)
RUNNER_SCRIPT="$RUNNER_DIR/run.sh"
cat > "$RUNNER_SCRIPT" << EOF
#!/bin/bash
exec $PYTHON_PATH launch_multiagent.py "\$@"
EOF
chmod +x "$RUNNER_SCRIPT"

# Ensure cleanup of temporary directory on exit
cleanup() { rm -rf "$RUNNER_DIR"; }
trap cleanup EXIT

# =============================================================================
# LAUNCH MULTIAGENT SYSTEM
# =============================================================================
echo "=== Multiagent system started at: $(date) ==="

# Execute via wrapper
"$RUNNER_SCRIPT" --task "$RESEARCH_TASK" "$@"
exit_code=$?

echo "=== Multiagent system completed with exit code: $exit_code at: $(date) ==="

# =============================================================================
# CLEANUP AND REPORTING
# =============================================================================
echo "Job completed at: $(date)"
echo "Final exit code: $exit_code"
exit $exit_code
