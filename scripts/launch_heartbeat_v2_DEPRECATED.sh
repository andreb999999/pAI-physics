#!/bin/bash
#SBATCH --job-name=heartbeat_v2
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1/results/muon_campaign_v2/logs/heartbeat_%j.log
#SBATCH --error=/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1/results/muon_campaign_v2/logs/heartbeat_%j.log

# ============================================================
# Heartbeat loop for muon_campaign_v2
# Ticks every 15 minutes. Runs for up to 12 hours (48 ticks).
# Resubmits itself before exiting if the campaign is not done.
# ============================================================

set -uo pipefail
# NOTE: no set -e — the heartbeat script uses non-zero exit codes for normal
# operation (1=in_progress, 2=failed, 3=stage_launched). We handle them explicitly.

REPO_ROOT="/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1"
CAMPAIGN_YAML="campaign_v4.yaml"
INTERVAL_SECONDS=900   # 15 minutes
MAX_TICKS=46           # ~11.5 hours (leave 30 min buffer for resubmit)

# --- Environment setup ---
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda activate /home/mabdel03/conda_envs/consortium

cd "$REPO_ROOT"

echo "[$(date)] Heartbeat job started (SLURM job $SLURM_JOB_ID)"

for tick in $(seq 1 $MAX_TICKS); do
    echo ""
    echo "=== Tick $tick/$MAX_TICKS — $(date) ==="

    # Capture exit code without set -e killing the loop
    exit_code=0
    python scripts/campaign_heartbeat.py --campaign "$CAMPAIGN_YAML" || exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] Campaign fully complete. Exiting heartbeat loop."
        exit 0
    elif [ $exit_code -eq 1 ]; then
        echo "[$(date)] Campaign in progress. Nothing to do this tick."
    elif [ $exit_code -eq 2 ]; then
        echo "[$(date)] Campaign has a failed stage. Continuing heartbeat for repair attempts."
    elif [ $exit_code -eq 3 ]; then
        echo "[$(date)] New stage launched this tick."
    else
        echo "[$(date)] Unexpected exit code $exit_code. Continuing."
    fi

    # Sleep until next tick (unless this was the last tick)
    if [ $tick -lt $MAX_TICKS ]; then
        echo "[$(date)] Sleeping ${INTERVAL_SECONDS}s until next tick..."
        sleep $INTERVAL_SECONDS
    fi
done

# --- Self-resubmit if campaign is not complete ---
echo ""
echo "[$(date)] Reached max ticks ($MAX_TICKS). Checking if campaign needs continuation..."

status_exit=0
python scripts/campaign_heartbeat.py --campaign "$CAMPAIGN_YAML" --status || status_exit=$?

if [ $status_exit -ne 0 ]; then
    echo "[$(date)] Campaign not yet complete. Resubmitting heartbeat job..."
    next_job=$(sbatch "$REPO_ROOT/scripts/launch_heartbeat_v2.sh" 2>&1)
    echo "[$(date)] Resubmitted: $next_job"
else
    echo "[$(date)] Campaign complete. No resubmit needed."
fi
