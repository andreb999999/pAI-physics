# OpenClaw Guide for PoggioAI/MSc

This guide documents the current supported OpenClaw and campaign-automation flow for MSc.

## What OpenClaw Controls

OpenClaw interacts with MSc through two automation-facing surfaces:

- `scripts/campaign_heartbeat.py`: campaign state machine and stage launcher
- `scripts/campaign_cli.py`: machine-readable status, log, repair, and admin operations

For normal interactive use, prefer the `msc campaign ...` commands. Use the direct scripts when you need cron jobs, OpenClaw, or SLURM-native orchestration.

## Credentials and Environment

The main engine requires `OPENROUTER_API_KEY`.

Supported credential precedence for both the CLI and direct scripts:

1. Existing shell environment variables
2. `--config-dir` or `~/.msc/.env`
3. repo-root `.env` only when launched from the checkout root or when explicitly enabled with `CONSORTIUM_USE_REPO_ENV=1`

Optional credentials:

- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
- `SLACK_WEBHOOK_URL`
- provider-specific keys for optional tooling or debugging

## Maintained Campaign Example

Use [`examples/quickstart/campaign.yaml`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/examples/quickstart/campaign.yaml) as the maintained example campaign.

## Preferred Operator Workflow

```bash
msc setup
msc doctor
msc campaign init --name "my_campaign" --task "Investigate optimizer stability" --budget 150
msc campaign start my_campaign_campaign.yaml
msc campaign status my_campaign_campaign.yaml
```

The generated campaign YAML uses the supported dynamic-planning contract:

```yaml
budget_usd: 150
planning:
  enabled: true
  base_task_file: automation_tasks/generated/my_campaign_discovery_task.txt
```

## Direct-Script Workflow

Initialize, validate, or tick a campaign directly:

```bash
python scripts/campaign_heartbeat.py --campaign my_campaign_campaign.yaml --init
python scripts/campaign_heartbeat.py --campaign my_campaign_campaign.yaml --validate
python scripts/campaign_heartbeat.py --campaign my_campaign_campaign.yaml
```

Query or operate on the campaign through the structured CLI bridge:

```bash
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml status
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml launchable
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml check-credits
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml repair <stage_id>
```

When you use the direct scripts from the checkout root, repo-root `.env` may participate in credential resolution. For installed CLI workflows outside the checkout root, prefer `msc setup` or an explicit `--config-dir`.

## Campaign YAML Fields That Matter Most

Recommended fields for dynamic planning:

```yaml
name: "My Research Campaign"
workspace_root: "results/my_campaign"
budget_usd: 150
heartbeat_interval_minutes: 15
max_idle_ticks: 6
max_campaign_hours: 96

planning:
  enabled: true
  base_task_file: automation_tasks/generated/my_campaign_discovery_task.txt
  max_stages: 6
  max_parallel: 2
  human_review: false
  planning_budget_usd: 5.0

repair:
  enabled: true
  max_attempts: 2
  launcher: local
  two_phase: true

notification:
  telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"
  on_stage_complete: true
  on_failure: true
  on_heartbeat: false
```

Notes:

- `budget_usd` is the campaign-wide budget cap
- `planning.base_task_file` is the supported discovery-task input
- legacy inline planning-task configs are still tolerated for backward compatibility, but they are deprecated

## Scheduling Heartbeats

Typical approaches:

- cron on a login node
- OpenClaw-managed recurring execution
- a lightweight SLURM wrapper if your site requires scheduler-managed control loops

Example cron entry:

```bash
*/15 * * * * cd /path/to/MSc_Internal && /path/to/python scripts/campaign_heartbeat.py --campaign my_campaign_campaign.yaml >> logs/heartbeat.log 2>&1
```

## OpenClaw on SLURM

To launch the gateway on SLURM:

```bash
sbatch scripts/launch_openclaw_gateway.sh
```

Recommended split:

- OpenClaw or cron triggers `campaign_heartbeat.py`
- the heartbeat launches or advances campaign stages
- stage workspaces write artifacts under the campaign `workspace_root`
- `campaign_cli.py` provides machine-readable inspection and recovery hooks

## Budget and Failure Handling

- Campaign-wide budget is enforced from `budget_usd` in the campaign YAML
- Per-run model selection comes from the selected run tier and any explicit overrides
- Each stage run writes `effective_models.json`, `experiment_metadata.json`, `run_status.json`, and `budget_state.json`
- `check-credits` validates OpenRouter access using a minimal routed request
- repair flow is configured under the `repair` section of the campaign spec

## Troubleshooting

### Authentication or credit failures

Run:

```bash
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml check-credits
```

Then verify `OPENROUTER_API_KEY` is available through your shell, `~/.msc/.env`, or repo-root `.env`.
If needed, compare `msc doctor` with and without `CONSORTIUM_USE_REPO_ENV=0` to confirm whether repo-root `.env` is participating.

### Campaign will not load

Regenerate the campaign with `msc campaign init`, or update old YAMLs to use `planning.base_task_file`.

### Stages do not advance

Use:

```bash
python scripts/campaign_heartbeat.py --campaign my_campaign_campaign.yaml --status
python scripts/campaign_cli.py --campaign my_campaign_campaign.yaml status
```

Check failures in this order:

1. Read the stage stderr log.
2. Inspect canonical artifacts under `paper_workspace/`.
3. Only then look at `stage_summaries/`, which are non-canonical summaries and do not prove stage completion.

Then check dependencies, missing artifacts, and launcher failures in the relevant stage workspace.

### PDF or LaTeX artifacts are missing

Install LaTeX and ensure the run mode/output settings actually request PDF generation.
