#!/bin/bash
#SBATCH --job-name=ib_sweep
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-89
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_%a.out
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_%a.err

echo "========================================"
echo "Implicit Bias Identification Sweep"
echo "========================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Array Task:  $SLURM_ARRAY_TASK_ID"
echo "Node:        $(hostname)"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Started:     $(date)"
echo "========================================"

PYTHON=/orcd/home/002/mabdel03/conda_envs/consortium/bin/python
SCRIPT_DIR=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification
RESULTS_DIR=$SCRIPT_DIR/results

# Install cvxpy if needed
$PYTHON -c "import cvxpy" 2>/dev/null || $PYTHON -m pip install cvxpy --quiet

$PYTHON $SCRIPT_DIR/run_sweep.py \
    --instance-id $SLURM_ARRAY_TASK_ID \
    --total-instances 90 \
    --output-dir $RESULTS_DIR \
    --device cuda

echo "Finished: $(date)"
