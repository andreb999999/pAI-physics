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

By default, stdout/stderr are written to `logs/freephdlabor_<timestamp>.out/.err`.

Disable file logging if you want output only in terminal:

```bash
python launch_multiagent.py --no-log-to-files --task "..."
```

## 6) Resume a Workspace

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/my_project_001 \
  --reasoning-effort none \
  --verbosity low \
  --task "Continue from existing outputs."

# Optional reliability controls for paper tasks:
# - enforce truthful artifact reporting and required deliverables
# - require final_paper.pdf before successful termination
# - optionally require experiments_to_run_later.md as an explicit planning deliverable
python launch_multiagent.py \
  --resume /absolute/path/to/results/my_project_001 \
  --task "Write and refine the paper, then produce final_paper.tex and final_paper.pdf." \
  --enforce-paper-artifacts \
  --require-pdf \
  --require-experiment-plan \
  --manager-max-steps 30
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

## 8) Pause/Steer a Running Job (No Restart)

The callback server listens on `127.0.0.1:5001` by default.

From another terminal, connect and send:

```bash
nc 127.0.0.1 5001
```

Then type:

1. `interrupt` (first line)
2. Your instruction text
3. Two empty lines (press Enter twice) to submit
4. `m` for modification or `n` for new task

Important behavior:

- `interrupt` does **not** restart the run
- it pauses at the next step boundary, appends your instruction to memory, and resumes from current state
- this is steering, not process termination

## 9) Stop (Kill) a Running Job

In the same terminal where it is running:

- `Ctrl + C`

If needed from another terminal:

```bash
pkill -f launch_multiagent.py
```

If it does not stop, target the PID directly:

```bash
pgrep -f launch_multiagent.py
kill <PID>
```

Last resort:

```bash
kill -9 <PID>
```

## 10) Common Issues

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

### Reduce citation retries / token burn (optional)

You can tune Semantic Scholar retry behavior:

```bash
export FREEPHDLABOR_SS_MAX_RETRIES=2
export FREEPHDLABOR_SS_BASE_DELAY_SEC=2
export FREEPHDLABOR_SS_COOLDOWN_SEC=60
```

### Limit large tool outputs in context (optional)

To prevent huge file dumps from inflating context/tokens:

```bash
export FREEPHDLABOR_SEE_FILE_MAX_CHARS=12000
export FREEPHDLABOR_SEARCH_MAX_CHARS=12000
export FREEPHDLABOR_SEARCH_MAX_MATCHES=200
```

## 11) Handoff Checklist (for a coworker)

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
