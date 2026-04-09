# MSc on SLURM/HPC Clusters

This guide covers the supported setup for running MSc on a SLURM-backed cluster.

## Recommended Setup

```bash
git clone https://github.com/PoggioAI/PoggioAI_MSc.git
cd PoggioAI_MSc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
msc setup
msc doctor
```

The preferred operational flow is:

- use `msc setup` to store user credentials in `~/.msc/.env`
- use `msc run` and `msc campaign ...` as the canonical interfaces
- use direct scripts only for scheduler-native automation tied to this checkout
- run from repo root only when you need relative example files or direct-script workflows

## Credential Resolution

For installed CLI usage on clusters, the default precedence is:

1. shell environment variables
2. `--config-dir` or `~/.msc/.env`
3. repo-root `.env` only when launched from the checkout root or when explicitly enabled

Useful overrides:

```bash
# Ignore repo-root .env even if you are standing in the checkout
CONSORTIUM_USE_REPO_ENV=0 msc doctor

# Force repo-root .env to participate outside the checkout root
CONSORTIUM_USE_REPO_ENV=1 msc doctor
```

For login-node and batch workflows, the recommended pattern is to rely on `~/.msc/.env` or an explicit `--config-dir`.

## Validating Cluster Readiness

```bash
msc doctor
msc run --mode hpc --tier budget --dry-run "Analyze optimizer stability"
```

`msc doctor` shows where each credential was sourced and whether optional tools such as LaTeX and SLURM are available.

For installed CLI usage, real runs are supported from outside the repo root. Results go to the current working directory's `results/`, and the run-local `.llm_config.yaml` is emitted there as part of startup.

## Running Single Jobs

```bash
msc run --mode hpc --tier budget "Analyze optimizer stability"
msc run --mode hpc --tier ultra --budget 100 --max-run-seconds 5400 "Max-rigor test task"
```

You can launch from the repo root or another working directory after installation. Results are written to the current working directory's `results/`.

## Running Campaigns

```bash
msc campaign init --name "cluster_demo" --task "Analyze optimizer stability" --budget 100
msc campaign start cluster_demo_campaign.yaml
msc campaign status cluster_demo_campaign.yaml
```

Maintained example campaign:

- [campaign.yaml](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal_audit_20260409_134959/MSc_Internal/examples/quickstart/campaign.yaml)

## Direct Scripts

Direct scripts are still available for cron, heartbeat loops, and SLURM wrappers, but they are not the primary onboarding path.

Typical advanced entrypoints:

```bash
python -m consortium.runner --dry-run --task "Test task"
python scripts/campaign_heartbeat.py --campaign examples/quickstart/campaign.yaml --validate
python scripts/campaign_cli.py --campaign examples/quickstart/campaign.yaml status
```

When you run direct scripts from the checkout root, repo-root `.env` may participate in credential resolution. That is useful for local automation, but less predictable for new-user flows than `msc setup`.

## Cluster Configuration

Cluster-specific launcher settings live in [engaging_config.yaml](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal_audit_20260409_134959/MSc_Internal/engaging_config.yaml).

Common values to verify:

- `CONDA_INIT_SCRIPT`
- `CONDA_ENV_PREFIX`
- `REPO_ROOT`
- `SLURM_OUTPUT_DIR`
- partition names and time limits for orchestrator vs experiment jobs

## Monitoring

```bash
msc status
msc logs -f
squeue -u "$USER"
```

For direct-script workflows, also inspect the SLURM output files and campaign heartbeat status.

## Troubleshooting

### Authentication surprises

If the wrong credentials are being picked up, run:

```bash
msc doctor
CONSORTIUM_USE_REPO_ENV=0 msc doctor
```

That usually makes it obvious whether a repo-local `.env` is participating unexpectedly.

### LaTeX/PDF failures

Install a TeX toolchain and verify `pdflatex` is on `PATH`, or configure `CONSORTIUM_PDFLATEX_PATH`.

### HPC stages never launch

Check `engaging_config.yaml`, SLURM availability, and the scheduler wrapper scripts under `scripts/`.
