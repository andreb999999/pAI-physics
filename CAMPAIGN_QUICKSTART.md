# Campaign Quick-Start Guide

How to go from a research idea to an autonomous campaign producing a conference-grade paper, overseen via Telegram.

## Prerequisites

| Requirement | How to verify |
|------------|---------------|
| Consortium conda env | `conda activate /home/mabdel03/conda_envs/consortium && python -c "import consortium"` |
| API keys set in `.env` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` |
| Telegram bot configured | `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| OpenClaw installed | `which openclaw` (npm install -g openclaw) |
| Claude CLI installed | `which claude` (for repair agent) — run `claude auth` once for headless access |
| TeX Live on PATH | `/orcd/software/community/001/pkg/tex-live/20251104/bin/x86_64-linux` |

## Step 1: Write Your Research Proposal

Create a task file describing the research question. This is the seed for the entire campaign.

```bash
cat > automation_tasks/my_discovery_task.txt << 'EOF'
Investigate whether [your research question here].

Key aspects to explore:
1. ...
2. ...
3. ...

Target venue: NeurIPS 2026
EOF
```

See `automation_tasks/v5_discovery_task.txt` for a concrete example.

## Step 2: Create the Campaign YAML

Copy the template and customize:

```bash
cp campaign_v5.yaml campaign_NEW.yaml
```

Edit `campaign_NEW.yaml` — the critical fields to change:

```yaml
name: "Your Campaign Name v1"
workspace_root: "results/your_campaign_v1"      # MUST be unique per campaign

planning:
  enabled: true
  base_task_file: automation_tasks/my_discovery_task.txt   # your task from Step 1
  max_stages: 6
  max_parallel: 2
  human_review: true          # set false for fully autonomous (auto-approve plan)

stages: []                    # empty = dynamic planning generates stages automatically

notification:
  telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"
  on_stage_complete: true
  on_failure: true
  on_heartbeat: true
```

### Campaign Independence Rules

- **Every campaign MUST have a unique `workspace_root`** — never reuse a directory from a prior campaign.
- **Never reference prior campaign results** in `context_from` or `--resume` flags.
- **Rename old campaign YAMLs** with `_DEPRECATED.yaml` suffix when superseded.
- All stage artifacts, logs, and status files live under `workspace_root/`.

## Step 3: Initialize the Campaign

```bash
cd /orcd/scratch/orcd/012/mabdel03/AI_Researcher/phdlabor-1

source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
conda activate /home/mabdel03/conda_envs/consortium

python scripts/campaign_heartbeat.py --campaign campaign_NEW.yaml --init
```

This creates:
- `results/your_campaign_v1/campaign_status.json` (tracks stage states)
- `results/your_campaign_v1/campaign_status.lock` (concurrency lock)

## Step 4: Launch the First Stage

```bash
python scripts/campaign_cli.py --campaign campaign_NEW.yaml launch discovery_plan
```

The heartbeat will auto-advance subsequent stages as dependencies are satisfied.

## Step 5: Start the OpenClaw Gateway

If the gateway is not already running:

```bash
# Update the SLURM log path in launch_openclaw_gateway.sh to point to your campaign:
#   --output=.../results/your_campaign_v1/logs/openclaw_gw_%j.log
#   --error=.../results/your_campaign_v1/logs/openclaw_gw_%j.log

sbatch scripts/launch_openclaw_gateway.sh
```

The gateway:
- Runs on `mit_normal` partition (12-hour wall time)
- Self-resubmits before wall time expires
- Hosts the OpenClaw agent on port 18789
- Only one instance needed (shared across campaigns)

## Step 6: Set Up Cron Monitoring

Via OpenClaw CLI or Telegram, register the heartbeat cron:

```
Campaign heartbeat: every 15 minutes
  python scripts/campaign_heartbeat.py --campaign campaign_NEW.yaml

Log monitor: every 5 minutes (lightweight liveness checks)
```

**Important**: OpenClaw cron has TWO timeouts that must both be set:
- `--timeout <ms>` (gateway-level)
- `--timeout-seconds <n>` (agent payload)

Recommended: 900s for heartbeat, 180s for log monitor.

## Step 7: Monitor via Telegram

Once running, the overseer sends:
- **Every 15 min**: Heartbeat status (stages in progress, budget spent, artifacts found)
- **On stage complete**: Summary + artifacts
- **On failure**: Error diagnosis + repair attempt status

### Useful CLI Commands

```bash
# Full status
python scripts/campaign_cli.py --campaign campaign_NEW.yaml status

# Stage logs (last 100 lines)
python scripts/campaign_cli.py --campaign campaign_NEW.yaml stage-logs <stage_id> --tail 100

# Budget summary
python scripts/campaign_cli.py --campaign campaign_NEW.yaml budget

# Check API credits before launch
python scripts/campaign_cli.py --campaign campaign_NEW.yaml check-credits

# Approve the dynamic plan (if human_review: true)
python scripts/campaign_cli.py --campaign campaign_NEW.yaml approve-plan

# List stages ready to launch
python scripts/campaign_cli.py --campaign campaign_NEW.yaml launchable

# Override a stage status (emergency)
python scripts/campaign_cli.py --campaign campaign_NEW.yaml set-stage-status <stage_id> <status>
```

## Campaign Lifecycle

```
1. discovery_plan     — Literature review + research plan (auto-launched)
2. planning_counsel   — 4-model debate generates stage DAG (auto-launched)
   [HUMAN REVIEW]     — Approve/reject plan via CLI or Telegram
3. theory1/theory2    — Parallel theory tracks (auto-launched after approval)
4. experiment1        — Empirical validation (auto-launched when theory deps complete)
5. paper1             — Publication-quality paper (auto-launched when experiments done)
```

Exit codes from heartbeat:
- `0`: Campaign complete (all stages done)
- `1`: In progress (normal tick)
- `2`: Failed stage (human attention needed)
- `3`: New stage just launched

## Budget

Default: `$2,000` total across all stages. Configurable in `.llm_config.yaml`.

Monitor spend:
```bash
python scripts/campaign_cli.py --campaign campaign_NEW.yaml budget
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Heartbeat shows $0 budget | Normal during early ticks — budget updates on stage completion |
| API rate limit hit | Gateway auto-resubmits; add fallback models in config |
| Stage died silently | Heartbeat detects + triggers repair agent (2 attempts max) |
| Plan not approved | Run `campaign_cli.py approve-plan` or set `human_review: false` |
| Hollow experiment artifacts | Agents lack GPU access — see `engaging_config.yaml` for SLURM experiment setup |
| Telegram not receiving updates | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| Gateway Telegram 409 conflict | Only one gateway instance should run — check with `squeue -u $USER` |

## Files Reference

| File | Purpose |
|------|---------|
| `campaign_NEW.yaml` | Campaign configuration (stages, repair, notifications) |
| `engaging_config.yaml` | Cluster-specific settings (partitions, conda, modules) |
| `.llm_config.yaml` | Model configuration (main model, counsel models, budget) |
| `.env` | API keys and notification credentials |
| `scripts/campaign_heartbeat.py` | Heartbeat orchestrator (called by cron) |
| `scripts/campaign_cli.py` | CLI for status, launch, repair, approve-plan |
| `scripts/launch_openclaw_gateway.sh` | SLURM launcher for OpenClaw gateway |
| `results/your_campaign_v1/campaign_status.json` | Campaign state (auto-managed) |
