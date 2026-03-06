#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-consortium}"
PROFILE_RAW="${2:-full}"
PROFILE="${PROFILE_RAW// /}"

if [[ -z "$PROFILE" ]]; then
  PROFILE="full"
fi

IFS=',' read -r -a PROFILE_ITEMS <<< "$PROFILE"

has_capability() {
  local cap="$1"
  local item=""
  for item in "${PROFILE_ITEMS[@]}"; do
    if [[ "$item" == "full" || "$item" == "$cap" ]]; then
      return 0
    fi
  done
  return 1
}

if ! has_capability minimal && ! has_capability docs && ! has_capability web && ! has_capability experiment && ! has_capability latex; then
  echo "Error: profile '$PROFILE_RAW' is invalid."
  echo "Use one of: minimal, docs, web, experiment, latex, full"
  echo "You can combine capabilities with commas (example: minimal,web)."
  exit 1
fi

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
python -m pip install -r "$REPO_ROOT/requirements-minimal.txt"
python -m pip install -r "$REPO_ROOT/requirements-observability.txt"

if has_capability docs; then
  python -m pip install -r "$REPO_ROOT/requirements-docs.txt"
fi

if has_capability web; then
  python -m pip install -r "$REPO_ROOT/requirements-web.txt"
  # Required by crawl4ai at runtime.
  python -m playwright install chromium
fi

if has_capability experiment; then
  python -m pip install -r "$REPO_ROOT/requirements-experiment.txt"
fi

if has_capability latex; then
  # Install TeX toolchain in conda env to support pdflatex/bibtex compilation.
  conda install -n "$ENV_NAME" -c conda-forge texlive-core latexmk -y
  # Best-effort format generation to avoid "can't find pdflatex.fmt".
  if command -v fmtutil-user >/dev/null 2>&1; then
    fmtutil-user --byfmt pdflatex >/dev/null 2>&1 || true
  fi
fi

python -m pip check

PREFLIGHT_ARGS=()
if has_capability docs; then
  PREFLIGHT_ARGS+=("--with-docs")
fi
if has_capability web; then
  PREFLIGHT_ARGS+=("--with-web")
fi
if has_capability experiment; then
  PREFLIGHT_ARGS+=("--with-experiment")
fi
if has_capability latex; then
  PREFLIGHT_ARGS+=("--with-latex")
fi
python "$REPO_ROOT/scripts/preflight_check.py" "${PREFLIGHT_ARGS[@]}"

echo "Setup complete."
echo "Installed profile: $PROFILE_RAW"
echo "Next: add API key(s) in $REPO_ROOT/.env and run launch_multiagent.py"
