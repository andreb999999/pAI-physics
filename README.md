# Internal Setup Guide

This repository runs a multi-agent research workflow from a local workspace.

## 1) Prerequisites

- macOS or Linux
- Conda installed
- Python 3.11 (managed by bootstrap)
- At least one LLM API key

## 2) Quick Install (Recommended)

From repo root:

```bash
./scripts/bootstrap.sh researchlab full
```

Install profiles:

- `minimal`: core runtime only
- `docs`: document/audio parsing extras
- `web`: web crawling extras (includes Playwright)
- `experiment`: experiment-tool stack
- `full`: all capabilities

You can combine profiles:

```bash
./scripts/bootstrap.sh researchlab minimal,web
```

## 3) Configure API Keys

Create `.env` in repo root:

```bash
OPENAI_API_KEY=your_key_here
# Optional providers
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
DEEPSEEK_API_KEY=...
```

## 4) Verify Environment

```bash
conda activate researchlab
python scripts/preflight_check.py --with-docs --with-web --with-experiment
```

If you installed fewer capabilities, remove the related flags.

## 5) Run a Task

```bash
python launch_multiagent.py \
  --reasoning-effort none \
  --verbosity low \
  --task "Summarize inputs and propose next research steps."
```

Optional: log stdout/stderr to files:

```bash
python launch_multiagent.py --log-to-files --task "..."
```

## 6) Resume a Workspace

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/my_project_001 \
  --reasoning-effort none \
  --verbosity low \
  --task "Continue from existing outputs."
```

## 7) Provide Context Files (PDF/TXT/MD)

1. Put files into workspace `inputs/`:

```bash
mkdir -p /absolute/path/to/results/my_project_001/inputs
```

2. Run with an explicit instruction:

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/my_project_001 \
  --reasoning-effort none \
  --verbosity low \
  --task "Read inputs/*.pdf and inputs/*.md and inputs/*.txt. Create context_summary.md. Do NOT run experiments."
```

## 8) Stop a Running Job

In the same terminal where it is running:

- `Ctrl + C`

If needed from another terminal:

```bash
pkill -f launch_multiagent.py
```

## 9) Common Issues

### Missing optional dependency (example: `crawl4ai`)

Install web profile:

```bash
./scripts/bootstrap.sh researchlab web
```

### Playwright Chromium missing

```bash
python -m playwright install chromium
```

### Audio warning about ffmpeg

Install ffmpeg:

```bash
brew install ffmpeg
```

### No API key detected

Check `.env` exists and contains a valid key. Then rerun preflight.

## 10) Handoff Checklist (for a coworker)

Before sharing, confirm:

- `./scripts/bootstrap.sh researchlab full` completes
- `python scripts/preflight_check.py --with-docs --with-web --with-experiment` passes
- A small test task runs successfully
- `.env` is not committed

Check with:

```bash
git status -sb
```

If `.env` appears tracked, untrack it before pushing.
