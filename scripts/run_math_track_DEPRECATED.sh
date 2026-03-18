#!/bin/bash
#SBATCH --job-name=math-track-theory1
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_outputs/math_track_%j.out
#SBATCH --error=slurm_outputs/math_track_%j.err

# Math track runner — hardcoded repo path to avoid SLURM spool issues
REPO_DIR="/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1"
cd "$REPO_DIR"
mkdir -p slurm_outputs

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Repo: $REPO_DIR"

# Environment setup
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate /home/mabdel03/conda_envs/consortium

echo "Python: $(which python) ($(python --version))"

export CONSORTIUM_SLURM_ENABLED=1
export ENGAGING_CONFIG="$REPO_DIR/engaging_config.yaml"

TASK_FILE="$REPO_DIR/automation_tasks/run1_theory_task_stable.txt"
RESEARCH_TASK="$(cat "$TASK_FILE")"
WORKSPACE="/home/mabdel03/orcd/scratch/AI_Researcher/phdlabor-1/results/muon_campaign/theory1"

echo "=== Math track started at: $(date) ==="
python launch_multiagent.py \
  --task "$RESEARCH_TASK" \
  --resume "$WORKSPACE" \
  --start-from-stage math_literature_agent \
  --enable-math-agents \
  --pipeline-mode full_research
exit_code=$?

echo "=== Math track completed with exit code: $exit_code at: $(date) ==="
exit $exit_code
