# Quickstart Example: Batch Normalization & Spectral Regularization

This is the recommended first real run for a new user. It exercises the installed `msc` CLI on a focused task that stays inside the supported `budget` tier contract.

## What This Example Covers

The task asks MSc to:

1. gather literature on batch normalization and spectral regularization
2. synthesize the relevant background
3. sketch a small research plan
4. propose a minimal experiment design
5. draft a Markdown paper

The goal is a cheap, self-contained validation run. It does not require LaTeX, GPUs, or an HPC setup.

## Prerequisites

- `python -m pip install -e ".[dev]"` completed in this checkout
- `msc setup` completed successfully
- `msc doctor` passes
- `OPENROUTER_API_KEY` available from your shell, `--config-dir`, or `~/.msc/.env`

## Running It

From the repo root:

```bash
# Free validation first
msc run --tier budget --dry-run "test"

# Real quickstart run
msc run --tier budget --task-file examples/quickstart/task.txt
```

From outside the repo root, use an absolute path to the task file:

```bash
msc run --tier budget --task-file /absolute/path/to/examples/quickstart/task.txt
```

Results are written to the current working directory's `results/`.

## Budget-Tier Runtime Contract

This quickstart is intended to stay inside the `budget` tier policy:

- primary model surface: `gpt-5-mini`
- persona council and helper surfaces: `gpt-5-mini`
- no counsel by default
- Markdown-first outputs

Inspect `effective_models.json` after startup to confirm the resolved model surfaces for the run.

## Expected Cost and Time

Typical behavior for this quickstart:

- cost: within the `budget` tier cap, with many runs finishing well below that ceiling
- time: usually tens of minutes, depending on model latency and tool activity

The important contract is tier fidelity, not an exact wall-clock promise. For a first validation run, start with `--dry-run`, then launch the full task once the config looks right.

## Expected Outputs

After launch, inspect the new run under `results/consortium_<timestamp>/`.

Common artifacts include:

- `experiment_metadata.json`
- `effective_models.json`
- `run_status.json`
- `budget_state.json`
- `budget_ledger.jsonl`
- `logs/`
- `paper_workspace/`
- a final paper artifact such as `final_paper.md` when the run completes

For this budget-tier example, `effective_models.json` should resolve to `gpt-5-mini` surfaces rather than premium frontier models.

## Reading the Results

Good signs:

- `run_status.json` shows ongoing progress or a clean terminal state
- `budget_state.json` and `budget_ledger.jsonl` show spend increasing in a controlled way
- `logs/` contains stage-level output
- the final paper artifact contains a coherent literature-grounded draft with references and a concrete proposed experiment

Common issues:

- no progress for a long time: check `msc status` and `msc logs -f`
- incomplete paper artifact: inspect `paper_workspace/`, then resume if appropriate with `msc resume results/consortium_<timestamp>/`
- unexpected model choice: inspect `effective_models.json` and `experiment_metadata.json`

## Next Steps

Once this quickstart looks healthy:

1. Run your own task with `msc run --tier budget "..."`
2. Move to `msc run --tier medium ...` if you want LaTeX and stronger default reasoning
3. Use `msc campaign ...` for multi-stage projects
4. Use `docs/engaging_setup.md` for HPC/SLURM setups
5. Use `OpenClaw_Use_Guide.md` only if you need cron, gateway, or automation-native campaign control
