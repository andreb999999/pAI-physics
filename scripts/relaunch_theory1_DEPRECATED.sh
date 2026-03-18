#!/bin/bash
# =============================================================================
# Relaunch theory1 via campaign_cli after OOM kill.
# Previous SLURM job 10502959 killed (signal 9) after exceeding 8GB memory.
# This job requests 32GB to prevent recurrence.
# =============================================================================

#SBATCH --job-name=theory1_relaunch
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1
#SBATCH --output=slurm_outputs/theory1_relaunch_%j.out
#SBATCH --error=slurm_outputs/theory1_relaunch_%j.err

REPO_DIR=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1
cd "$REPO_DIR"

echo "========================================"
echo "theory1 Relaunch — Campaign v4"
echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Memory:    32GB (increased from 8GB after OOM)"
echo "========================================"

# --- Environment (no module load — can hang on some nodes) ---
export PS1="${PS1:-}"
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate /home/mabdel03/conda_envs/consortium

echo "Python:    $(which python) ($(python --version 2>&1))"

# --- Enable SLURM experiment submission ---
export CONSORTIUM_SLURM_ENABLED=1
export ENGAGING_CONFIG="$REPO_DIR/engaging_config.yaml"

# --- Launch theory1 ---
echo "=== Launching theory1 at: $(date) ==="
python scripts/campaign_cli.py --campaign campaign_v4.yaml launch theory1
exit_code=$?

echo "=== theory1 finished with exit code $exit_code at: $(date) ==="
exit $exit_code
