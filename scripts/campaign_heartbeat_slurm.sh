#!/bin/bash
#SBATCH --job-name=openclaw-heartbeat
#SBATCH --partition=pi_tpoggio
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/muon_campaign/logs/heartbeat_%j.log
#SBATCH --error=results/muon_campaign/logs/heartbeat_%j.log

# OpenClaw Campaign Heartbeat — SLURM loop
# Ticks every 30 minutes, checks/launches stages, sends ntfy notifications.
# 7-day wall-time on pi_tpoggio — no self-resubmission needed.

set -uo pipefail
cd /orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1

# Activate conda environment
source activate /home/mabdel03/conda_envs/consortium 2>/dev/null || true

# Make tex-live available for paper/editorial stages
export PATH="/orcd/software/community/001/pkg/tex-live/20251104/bin/x86_64-linux:$PATH"

CAMPAIGN="campaign.yaml"
INTERVAL=1800  # 30 minutes in seconds
MAX_TICKS=332  # 332 * 30min ≈ 6.9 days, leaves ~2h buffer before 7-day wall-time
FAIL_COUNT=0   # consecutive exit-code-2 (failed) ticks — tolerates repair attempts
MAX_FAILS=3    # give repair agents up to 3 ticks before giving up

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
        # Stage failed — but a repair agent may have been submitted.
        # Keep ticking for a few more rounds to let repair finish.
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ $FAIL_COUNT -ge $MAX_FAILS ]; then
            echo "[$(date)] Campaign failed after $FAIL_COUNT consecutive failure ticks. Exiting."
            exit 1
        fi
        echo "[$(date)] Failure detected (tick $FAIL_COUNT/$MAX_FAILS). Repair may be in progress..."
    else
        FAIL_COUNT=0  # reset on any non-failure code (0, 1, 3)
    fi
    # CODE=1 (in-progress) or CODE=3 (just-launched): keep ticking

    # Don't sleep after the last tick
    if [ $tick -lt $MAX_TICKS ]; then
        echo "[$(date)] Sleeping ${INTERVAL}s until next tick..."
        sleep $INTERVAL
    fi
done

# All ticks exhausted — wall-time approaching
echo "[$(date)] All $MAX_TICKS ticks exhausted (7-day wall-time limit). Exiting."
