# consortium Dockerfile
# Multi-stage build: installs system deps, then Python deps, then the package.
#
# Build:
#   docker build -t consortium .
#   docker build --target minimal -t consortium:minimal .
#
# Run:
#   docker run --env-file .env -v $(pwd)/results:/app/results consortium \
#     python launch_multiagent.py --task "Your research task" --no-counsel --no-log-to-files

# ── Stage 1: system dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    # LaTeX toolchain
    texlive-latex-base \
    texlive-latex-extra \
    texlive-bibtex-extra \
    bibtex2html \
    latexmk \
    # Git (for git-commit metadata)
    git \
    # Build tools for some Python packages
    build-essential \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: minimal install (no web crawl, no experiment tool) ──────────────
FROM base AS minimal

COPY pyproject.toml .
COPY consortium/ consortium/
COPY launch_multiagent.py .
COPY .env.example .

RUN pip install --no-cache-dir -e ".[docs]"

ENV CONSORTIUM_LOG_TO_FILES=0
ENTRYPOINT ["python", "launch_multiagent.py"]

# ── Stage 3: full install (all optional extras) ───────────────────────────────
FROM base AS full

COPY pyproject.toml .
COPY consortium/ consortium/
COPY launch_multiagent.py .
COPY scripts/ scripts/
COPY .env.example .
COPY automation_tasks/ automation_tasks/
COPY examples/ examples/

# Install all extras
RUN pip install --no-cache-dir -e ".[docs,web,observability]"

# Install Playwright browsers
RUN python -m playwright install chromium --with-deps 2>/dev/null || true

ENV CONSORTIUM_LOG_TO_FILES=0
ENTRYPOINT ["python", "launch_multiagent.py"]

# Default target is full
FROM full
