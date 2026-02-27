#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-freephdlabor}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Error: conda env '$ENV_NAME' does not exist."
  exit 1
fi

conda activate "$ENV_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Installing LaTeX toolchain in env '$ENV_NAME'..."
conda install -n "$ENV_NAME" -c conda-forge texlive-core latexmk -y

# Best-effort: generate pdflatex format files for current user.
if [ -x "$CONDA_PREFIX/bin/fmtutil-user" ]; then
  echo "Running fmtutil-user --byfmt pdflatex..."
  "$CONDA_PREFIX/bin/fmtutil-user" --byfmt pdflatex >/tmp/fmtutil_user.log 2>&1 || true
fi
if [ -x "$CONDA_PREFIX/bin/mktexfmt" ]; then
  echo "Running mktexfmt pdflatex.fmt..."
  "$CONDA_PREFIX/bin/mktexfmt" pdflatex.fmt >/tmp/mktexfmt.log 2>&1 || true
fi

echo "Verifying pdflatex smoke compile..."
tmpdir="$(mktemp -d)"
cat > "${tmpdir}/pdflatex_smoke.tex" <<'EOF'
\documentclass{article}
\begin{document}
pdflatex smoke test
\end{document}
EOF

if "$CONDA_PREFIX/bin/pdflatex" -interaction=nonstopmode -halt-on-error -output-directory "$tmpdir" "${tmpdir}/pdflatex_smoke.tex" >/tmp/pdflatex_smoke.log 2>&1; then
  echo "✅ pdflatex is working in env '$ENV_NAME'."
  echo "PDF: ${tmpdir}/pdflatex_smoke.pdf"
else
  echo "❌ pdflatex still failing in env '$ENV_NAME'."
  echo "---- pdflatex output ----"
  cat /tmp/pdflatex_smoke.log
  echo "-------------------------"
  exit 1
fi
