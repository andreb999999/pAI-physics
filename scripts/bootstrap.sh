#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-freephdlabor}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-core.txt"
python -m pip install -r "$REPO_ROOT/external_tools/run_experiment_tool/requirements.txt"

# Re-assert compatibility between smolagents and experiment-tool stack.
python -m pip install "transformers==4.44.2" "datasets==2.21.0" "huggingface-hub<1.0.0"

# Required by crawl4ai at runtime.
python -m playwright install chromium

python -m pip check
python "$REPO_ROOT/scripts/preflight_check.py"

echo "Setup complete."
echo "Next: add API key(s) in $REPO_ROOT/.env and run launch_multiagent.py"
