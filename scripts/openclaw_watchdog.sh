#!/bin/bash
# OpenClaw Gateway watchdog — checks if the gateway SLURM job is running,
# resubmits if not. Run this periodically from within the heartbeat loop
# or as a separate lightweight SLURM job.
#
# Usage: bash scripts/openclaw_watchdog.sh

set -uo pipefail

REPO_ROOT="/orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1"
GATEWAY_SCRIPT="$REPO_ROOT/scripts/launch_openclaw_gateway.sh"
NTFY_TOPIC="OpenClaw-Engaging"

# Check if openclaw_gw job is running
GW_RUNNING=$(squeue -u mabdel03 --name=openclaw_gw --noheader 2>/dev/null | wc -l)

if [ "$GW_RUNNING" -gt 0 ]; then
    echo "[$(date)] OpenClaw Gateway is running."
    exit 0
fi

echo "[$(date)] OpenClaw Gateway NOT running! Resubmitting..."

# Resubmit
JOB_ID=$(sbatch --parsable "$GATEWAY_SCRIPT" 2>&1)

if [ $? -eq 0 ]; then
    echo "[$(date)] Resubmitted gateway as SLURM job $JOB_ID"
    # Notify via ntfy
    curl -s -d "OpenClaw Gateway was down. Resubmitted as job $JOB_ID on $(hostname)" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    exit 0
else
    echo "[$(date)] ERROR: Failed to resubmit gateway: $JOB_ID"
    curl -s -d "CRITICAL: OpenClaw Gateway down and resubmit FAILED on $(hostname)" \
        "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    exit 1
fi
