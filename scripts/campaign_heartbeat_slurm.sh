#!/bin/bash
#SBATCH --job-name=openclaw-heartbeat
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/muon_campaign/logs/heartbeat_%j.log
#SBATCH --error=results/muon_campaign/logs/heartbeat_%j.log

# OpenClaw Campaign Heartbeat — SLURM loop
# Ticks every 30 minutes, checks/launches stages, sends ntfy notifications.
# Resubmits itself before the 12h wall-time expires.

set -uo pipefail
cd /orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1

# Activate conda environment
source activate /home/mabdel03/conda_envs/consortium 2>/dev/null || true

CAMPAIGN="campaign.yaml"
INTERVAL=1800  # 30 minutes in seconds
MAX_TICKS=22   # 22 * 30min = 11h, leaves 1h buffer before wall-time

echo "[$(date)] Heartbeat job started (SLURM job $SLURM_JOB_ID)"

for tick in $(seq 1 $MAX_TICKS); do
    echo ""
    echo "=== Tick $tick/$MAX_TICKS — $(date) ==="

    # Note: campaign_heartbeat.py uses non-zero exit codes as signals
    # (1=in-progress, 3=just-launched), so we must not use set -e here.
    python scripts/campaign_heartbeat.py --campaign "$CAMPAIGN"
    CODE=$?

    if [ $CODE -eq 0 ]; then
        echo "[$(date)] Campaign complete. Exiting."
        exit 0
    elif [ $CODE -eq 2 ]; then
        echo "[$(date)] Campaign failed. Exiting."
        exit 1
    fi
    # CODE=1 (in-progress) or CODE=3 (just-launched): keep ticking

    # Don't sleep after the last tick
    if [ $tick -lt $MAX_TICKS ]; then
        echo "[$(date)] Sleeping ${INTERVAL}s until next tick..."
        sleep $INTERVAL
    fi
done

# Wall-time approaching — resubmit ourselves
echo "[$(date)] Approaching wall-time limit. Resubmitting heartbeat job..."
sbatch scripts/campaign_heartbeat_slurm.sh
echo "[$(date)] Resubmitted. This job exiting."
