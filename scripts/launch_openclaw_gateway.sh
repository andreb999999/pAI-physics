#!/bin/bash
#SBATCH --job-name=openclaw_gw
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_outputs/openclaw_gw_%j.log
#SBATCH --error=slurm_outputs/openclaw_gw_%j.log

# OpenClaw Gateway SLURM launcher — runs the gateway as a long-running service.
# Self-resubmits before wall time expires.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/scripts/launch_openclaw_gateway.sh"

# --- Environment ---
# Source conda from engaging_config.yaml if available, else try standard conda
_ENGAGING_CFG="$REPO_ROOT/engaging_config.yaml"
_CONDA_INIT=""
_CONDA_PREFIX=""
if [ -f "$_ENGAGING_CFG" ]; then
    _CONDA_INIT=$(grep 'conda_init_script:' "$_ENGAGING_CFG" 2>/dev/null | head -1 | sed 's/.*conda_init_script:\s*//' | sed 's/\s*#.*//' | tr -d '[:space:]')
    _CONDA_PREFIX=$(grep 'conda_env_prefix:' "$_ENGAGING_CFG" 2>/dev/null | head -1 | sed 's/.*conda_env_prefix:\s*//' | sed 's/\s*#.*//' | tr -d '[:space:]')
fi
# Resolve env vars in the values (strip ${...:-} wrappers)
_CONDA_INIT=$(echo "$_CONDA_INIT" | sed 's/\${[^}]*:-\(.*\)}/\1/')
_CONDA_PREFIX=$(echo "$_CONDA_PREFIX" | sed 's/\${[^}]*:-\(.*\)}/\1/')

# Override with env vars if set
_CONDA_INIT="${CONDA_INIT_SCRIPT:-$_CONDA_INIT}"
_CONDA_PREFIX="${CONDA_ENV_PREFIX:-$_CONDA_PREFIX}"

if [ -n "$_CONDA_INIT" ] && [ -f "$_CONDA_INIT" ]; then
    source "$_CONDA_INIT"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found. Set CONDA_INIT_SCRIPT or ensure conda is on PATH."
    exit 1
fi

if [ -n "$_CONDA_PREFIX" ]; then
    conda activate "$_CONDA_PREFIX"
else
    conda activate consortium 2>/dev/null || conda activate base
fi

# Ensure .openclaw symlink exists (user-configurable via OPENCLAW_CONFIG_DIR)
_OPENCLAW_SRC="${OPENCLAW_CONFIG_DIR:-}"
if [ -n "$_OPENCLAW_SRC" ] && [ ! -L "$HOME/.openclaw" ] && [ ! -d "$HOME/.openclaw" ]; then
    ln -sf "$_OPENCLAW_SRC" "$HOME/.openclaw"
fi

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/slurm_outputs"

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
