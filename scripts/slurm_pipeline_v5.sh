#!/bin/bash
#SBATCH --job-name=muon_v5_pipeline
#SBATCH --partition=pi_tpoggio
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/logs/pipeline_v5_%j.out
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/logs/pipeline_v5_%j.err

echo "========================================"
echo "Muon v5 Pipeline Orchestrator"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

cd /orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal

export PYTHONDONTWRITEBYTECODE=1
export COUNSEL_MODEL_TIMEOUT_SECONDS=3600
export CONSORTIUM_PDFLATEX_PATH="/orcd/software/community/001/pkg/tex-live/20251104/bin/x86_64-linux/pdflatex"
export PATH="/orcd/software/community/001/pkg/tex-live/20251104/bin/x86_64-linux:/orcd/home/002/mabdel03/conda_envs/consortium/bin:$PATH"
export CONSORTIUM_SLURM_ENABLED=1

# Source .env for API keys
set -a; source .env 2>/dev/null; set +a

PYTHON=/orcd/home/002/mabdel03/conda_envs/consortium/bin/python
TASK=$($PYTHON -c "print(open('automation_tasks/muon_v5_iterate_task.txt').read())")

$PYTHON -B launch_multiagent.py \
  --model claude-opus-4-6 \
  --iterate /orcd/scratch/orcd/012/mabdel03/AI_Researcher/muon_v5_iterate_seed \
  --iterate-start-stage literature_review_agent \
  --enable-math-agents \
  --enable-counsel \
  --enforce-paper-artifacts \
  --enforce-editorial-artifacts \
  --require-pdf \
  --min-review-score 8 \
  --max-rebuttal-iterations 4 \
  --persona-debate-rounds 3 \
  --callback_port 5020 \
  --resume results/muon_v5_iterate/iterate_v5 \
  --task "$TASK"

echo "Pipeline exited with code $?"
echo "Finished: $(date)"
