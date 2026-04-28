# Architecture Overview

This document describes the current PoggioAI/MSc runtime architecture.

## Entry Points

Preferred user-facing entrypoints:

- `msc run ...` for a single research run
- `msc campaign ...` for campaign orchestration
- `msc doctor`, `msc config`, `msc notify`, and related operational commands

Supported direct-script entrypoints remain available for automation and HPC:

- `python launch_multiagent.py ...`
- `python scripts/campaign_heartbeat.py --campaign ...`
- `python scripts/campaign_cli.py --campaign ... <subcommand>`

Runtime configuration is resolved from three layers:

1. Existing shell environment variables
2. `~/.msc/.env`
3. Repo-root `.env`

Model/runtime settings are read from the project-root `.llm_config.yaml`, which `msc run` auto-generates from the selected tier unless `custom_llm_config: true` is set in `~/.msc/config.yaml`.

For an implementation-aligned graph diagram plus a plain-English agent summary, see [LangGraph Network Overview](langgraph_network.md).

## Pipeline Shape

The LangGraph workflow is built in [`consortium/graph.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/graph.py).

Base pipeline: 16 stages

1. `persona_council`
2. `literature_review_agent`
3. `brainstorm_agent`
4. `formalize_goals_entry`
5. `formalize_goals_agent`
6. `research_plan_writeup_agent`
7. `experiment_literature_agent`
8. `experiment_design_agent`
9. `experimentation_agent`
10. `experiment_verification_agent`
11. `experiment_transcription_agent`
12. `formalize_results_agent`
13. `resource_preparation_agent`
14. `writeup_agent`
15. `proofreading_agent`
16. `reviewer_agent`

Math-enabled pipeline: 22 stages

The six math stages are inserted between `research_plan_writeup_agent` and `experiment_literature_agent`:

1. `math_literature_agent`
2. `math_proposer_agent`
3. `math_prover_agent`
4. `math_rigorous_verifier_agent`
5. `math_empirical_verifier_agent`
6. `proof_transcription_agent`

## Routing and Validation

The stage roster above is only part of the full graph. The runtime also includes routers and gates that are not counted as user-facing pipeline stages:

- Track decomposition validation before execution fans out
- `track_router` fan-out into theory and empirical work
- `track_merge` fan-in before synthesis and writeup
- Intermediate artifact validation and strict review gates
- Follow-up loops when review or validation requires additional work

This is why the implementation includes more graph nodes than the 16/22 visible pipeline stages.

## Core Modules

- [`consortium/runner.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/runner.py): CLI/direct-script execution, workspace setup, env/bootstrap, resume logic
- [`consortium/config.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/config.py): `.llm_config.yaml` loading and model-param filtering
- [`consortium/graph.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/graph.py): Stage roster, routing, and graph construction
- [`consortium/state.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/state.py): `ResearchState` schema
- [`consortium/counsel.py`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/counsel.py): Multi-model debate/synthesis
- [`consortium/supervision/`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/supervision): Artifact, review, paper-quality, and traceability validators
- [`consortium/campaign/`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/consortium/campaign): Campaign spec loading, heartbeat state machine, repair flow, budget aggregation, notifications

## Campaign Architecture

Campaigns are defined by YAML specs such as [`campaign_template.yaml`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/campaign_template.yaml) or [`examples/quickstart/campaign.yaml`](/home/mabdel03/orcd/scratch/AI_Researcher/MSc_Internal/examples/quickstart/campaign.yaml).

High-level flow:

1. `msc campaign init` generates a campaign YAML plus a real `planning.base_task_file`
2. `msc campaign start` shells out to `scripts/campaign_heartbeat.py --init`
3. The heartbeat script validates state, launches stages, and advances the DAG
4. Each stage runs the main pipeline or a specialized launcher script
5. Stage outputs are distilled into campaign memory and notifications are emitted

Direct-script campaign automation remains supported. Those scripts now honor both repo-root `.env` and `~/.msc/.env`, matching the `msc` CLI behavior.

## Workspace and Outputs

Single runs write into `results/consortium_<timestamp>/` and include:

- `run_summary.json`
- `experiment_metadata.json`
- `budget_state.json`
- `paper_workspace/` and other stage artifacts
- checkpoint/state data for resume support

Campaigns write `campaign_status.json` and per-stage workspaces under the campaign `workspace_root`.

## Invariants

- OpenRouter is the required LLM backend for the main engine
- The selected tier is the baseline profile, but persisted config overrides can change model, budget, output format, mode, and feature toggles
- `messages` in `ResearchState` remain append-only through LangGraph reducers
- Budget enforcement is fail-closed when pricing data is available
- Project-root `.llm_config.yaml` is the runner-consumed model/budget config file
