#!/bin/bash
#SBATCH --job-name=openclaw_gw
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1/results/muon_campaign_v2/logs/openclaw_gw_%j.log
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1/results/muon_campaign_v2/logs/openclaw_gw_%j.log

# OpenClaw Gateway SLURM launcher — runs the gateway as a long-running service.
# Self-resubmits before wall time expires.

set -uo pipefail

REPO_ROOT="/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1"
SCRIPT_PATH="$REPO_ROOT/scripts/launch_openclaw_gateway.sh"

# --- Environment ---
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda activate /home/mabdel03/conda_envs/consortium

# Ensure .openclaw symlink exists
if [ ! -L "$HOME/.openclaw" ] && [ ! -d "$HOME/.openclaw" ]; then
    ln -sf /orcd/scratch/orcd/012/mabdel03/AI_Researcher/.openclaw "$HOME/.openclaw"
fi

cd "$REPO_ROOT"

echo "[$(date)] OpenClaw Gateway starting on $(hostname), SLURM job $SLURM_JOB_ID"

# --- Self-resubmit handler ---
# Trap SIGTERM (sent by SLURM before killing) to resubmit
resubmit() {
    echo "[$(date)] Wall time approaching or SIGTERM received. Resubmitting..."
    sbatch "$SCRIPT_PATH"
    echo "[$(date)] Resubmission complete. Shutting down gracefully."
    kill $GATEWAY_PID 2>/dev/null
    exit 0
}
trap resubmit SIGTERM SIGUSR1

# --- Launch Gateway ---
# Run in background so we can trap signals
openclaw gateway --port 18789 --verbose &
GATEWAY_PID=$!
echo "[$(date)] Gateway started, PID=$GATEWAY_PID"

# Wait for gateway to exit (or signal)
wait $GATEWAY_PID
EXIT_CODE=$?

echo "[$(date)] Gateway exited with code $EXIT_CODE"

# If gateway crashed (not a graceful shutdown from our trap), resubmit
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date)] Unexpected exit. Resubmitting gateway..."
    sbatch "$SCRIPT_PATH"
fi

exit $EXIT_CODE
