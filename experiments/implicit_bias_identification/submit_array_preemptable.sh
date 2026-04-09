#!/bin/bash
#SBATCH --job-name=ib_sweep_p
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-89
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_p_%a.out
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_p_%a.err

PYTHON=/orcd/home/002/mabdel03/conda_envs/consortium/bin/python
SCRIPT_DIR=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification
RESULTS_DIR=$SCRIPT_DIR/results

$PYTHON $SCRIPT_DIR/run_sweep.py \
    --instance-id $SLURM_ARRAY_TASK_ID \
    --total-instances 90 \
    --output-dir $RESULTS_DIR \
    --device cuda
