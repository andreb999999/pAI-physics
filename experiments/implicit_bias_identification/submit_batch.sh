#!/bin/bash
#SBATCH --job-name=ib_sweep
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_%j.out
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification/logs/sweep_%j.err

echo "Implicit Bias Identification — Batch runner"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES, Started: $(date)"

module load miniforge/25.11.0-0
module load cuda/12.4.0
module load cudnn/9.8.0.87-cuda12
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda activate /home/mabdel03/conda_envs/nanogpt_env

PYTHON=/home/mabdel03/conda_envs/nanogpt_env/bin/python
SCRIPT_DIR=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/experiments/implicit_bias_identification
RESULTS_DIR=$SCRIPT_DIR/results

# Install cvxpy if needed
$PYTHON -c "import cvxpy" 2>/dev/null || $PYTHON -m pip install cvxpy --quiet

# Run all 90 instances sequentially on one GPU
for i in $(seq 0 119); do
    if [ -f "$RESULTS_DIR/result_$(printf '%04d' $i).json" ]; then
        echo "Skipping instance $i (already done)"
        continue
    fi
    echo "=== Instance $i ==="
    $PYTHON $SCRIPT_DIR/run_sweep.py \
        --instance-id $i \
        --total-instances 90 \
        --output-dir $RESULTS_DIR \
        --device cuda
done

echo "All done: $(date)"
