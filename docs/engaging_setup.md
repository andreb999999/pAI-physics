# Consortium on MIT Engaging — Setup Guide

## Prerequisites

- Access to MIT Engaging cluster
- Conda installed at `/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/`
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, etc.)

## Quick Start

```bash
# 1. Source conda
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh

# 2. Bootstrap the environment (creates conda env + installs deps)
cd /orcd/scratch/orcd/012/mabdel03/AI_Researcher/OpenPI
./scripts/bootstrap.sh consortium full

# 3. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 4. Test configuration
conda activate /home/mabdel03/conda_envs/consortium
python launch_multiagent.py --dry-run --model claude-opus-4-6

# 5. Submit the orchestrator to SLURM
RESEARCH_TASK="Your research prompt here..." ./scripts/submit_orchestrator.sh --no-counsel
```

## Architecture: Two-Tier SLURM Model

Consortium uses a **two-tier execution model** on Engaging:

### Tier 1: Orchestrator (CPU)
- Runs on `sched_mit_hill` (CPU partition, 12hr limit)
- Makes outbound HTTPS calls to LLM APIs (Claude, GPT, Gemini)
- Coordinates 23+ specialist agents via LangGraph
- Does NOT need GPU

### Tier 2: Experiment Jobs (GPU)
- Submitted by the orchestrator via `sbatch` when experiments need GPU
- Default partition: `pi_tpoggio` (A100x8, 7-day limit)
- Fallback: `mit_normal_gpu` (6hr) or `mit_preemptable` (2-day)
- Runs AI-Scientist-v2 experiment execution

Set `CONSORTIUM_SLURM_ENABLED=1` to enable automatic GPU job submission.

## Configuration

### engaging_config.yaml
All Engaging-specific settings are centralized here:
- Partition names (GPU and CPU)
- Conda paths and module names
- Resource limits (CPUs, memory, time)

### .llm_config.yaml
LLM model selection, budget limits, counsel mode settings.

### .env
API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

## Available Partitions

| Partition | GPUs | Time Limit | Use Case |
|-----------|------|------------|----------|
| `sched_mit_hill` | None | 12hr | Orchestrator (Tier 1) |
| `pi_tpoggio` | A100x8 | 7 days | Default experiment GPU |
| `mit_normal_gpu` | H100/H200/L40S | 6hr | Short experiments |
| `mit_preemptable` | A100/H100/H200/L40S | 2 days | Long experiments (preemptible) |
| `pi_lhtsai` | RTX Pro 6000x2 | 2 days | PI-dedicated GPU |

## Running a Campaign (Multi-Stage)

```bash
# Initialize campaign
python scripts/campaign_heartbeat.py --campaign campaign.yaml --init

# Run heartbeat (advances stages)
python scripts/campaign_heartbeat.py --campaign campaign.yaml

# Or submit heartbeat as a SLURM job that advances stages via sbatch
```

Campaign stages can be launched as SLURM jobs using `launch_stage_slurm()` from the campaign runner.

## Monitoring

```bash
# Check SLURM jobs
squeue -u $USER

# Check job output
cat slurm_outputs/orch_<job_id>.out

# List past runs
python launch_multiagent.py --list-runs

# Check experiment GPU job logs
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.out
```

## Troubleshooting

### "conda not found"
```bash
source /orcd/data/lhtsai/001/om2/mabdel03/miniforge3/etc/profile.d/conda.sh
```

### "module not found"
```bash
module load miniforge/25.11.0-0
module load cuda/12.4.0
module load cudnn/9.8.0.87-cuda12
```

### GPU experiment job fails
Check the SLURM logs in the experiment run directory:
```bash
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.out
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.err
```

### API calls fail from compute node
The orchestrator needs outbound internet access. If compute nodes don't have it, run the orchestrator on a login node instead:
```bash
conda activate /home/mabdel03/conda_envs/consortium
nohup python launch_multiagent.py --task "..." --no-counsel &
```

### LaTeX/PDF compilation fails
```bash
# Install TeX toolchain in conda env
./scripts/fix_pdflatex_conda.sh consortium
# Or load system module
module load tex-live/20251104
```
