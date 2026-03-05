# freephdlabor: Multi-Agent Research-to-Paper Pipeline

`freephdlabor` is a local multi-agent research system that turns a research prompt into literature-grounded, experiment-backed, and (optionally) theorem-verified paper artifacts.

- License: MIT (`LICENSE`)
- Runtime: Python 3.11 (recommended via conda)
- Entry point: `launch_multiagent.py`
- Core package: `freephdlabor/`

## How It Works

You provide a research task. A ManagerAgent orchestrates specialist agents through a multi-phase pipeline that produces literature-grounded, evidence-backed paper artifacts. When math mode is enabled, a parallel theory pipeline builds and verifies a formal claim graph.

![freephdlabor pipeline flowchart](docs/pipeline_flowchart.png)

The diagram above shows the `full_research` 8-step pipeline. In `default` mode, the manager has flexibility to pick agents adaptively. In `quick` mode, loops are shallower but truthfulness gates still apply.

## Quick Start

From repository root:

```bash
./scripts/bootstrap.sh researchlab full
conda activate researchlab
cp .env.example .env
# Edit .env and add at least one API key
python scripts/preflight_check.py --with-docs --with-web --with-experiment --with-latex
python launch_multiagent.py \
  --task "Investigate this topic and produce a paper draft with evidence-backed claims." \
  --pipeline-mode default \
  --no-log-to-files
```

Artifacts are written to `results/freephdlabor_<timestamp>/`.

## Installation (Detailed)

### Prerequisites

- macOS or Linux
- Conda (Miniconda or Anaconda)
- At least one LLM API key

### Bootstrap Profiles

Use:

```bash
./scripts/bootstrap.sh <env_name> <profile>
```

Supported profiles:

- `minimal`: core runtime
- `docs`: document/audio parsing stack
- `web`: web crawling stack + Playwright Chromium install
- `experiment`: experiment tool dependencies
- `latex`: TeX toolchain (`pdflatex`, `bibtex`)
- `full`: all capabilities

Profiles can be combined:

```bash
./scripts/bootstrap.sh researchlab minimal,web
```

### API Keys

Copy `.env.example` to `.env` and fill what you use:

```bash
OPENAI_API_KEY=your_openai_api_key_here
# Optional providers
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...
# OPENROUTER_API_KEY=...
# DEEPSEEK_API_KEY=...
```

### Preflight Validation

```bash
python scripts/preflight_check.py --with-docs --with-web --with-experiment --with-latex
```

Remove flags for capabilities you did not install.

## Configuration

### Model Selection and Precedence

Model settings are resolved in this order:

1. Built-in defaults in `freephdlabor/runner.py` (`gpt-5`, `reasoning_effort=high`, `verbosity=medium`)
2. `.llm_config.yaml`
3. CLI overrides (`--model`, `--reasoning-effort`, `--verbosity`)

### `.llm_config.yaml`

This file controls:

- `main_agents` model + reasoning settings (`reasoning_effort`, `verbosity`, and Claude 4.6 `effort`)
- `run_experiment_tool` model settings for experiment subprocesses
- `budget` hard USD cap and pricing map

Current repository defaults:

- `main_agents.model`: `claude-opus-4-6`
- `main_agents.effort`: `max` (this is "Opus 4.6 Max")
- `run_experiment_tool.code_model`: `gpt-5.3-codex`
- `run_experiment_tool.feedback_model`: `claude-sonnet-4-6`
- `run_experiment_tool.vlm_model`: `claude-sonnet-4-6`
- `run_experiment_tool.report_model`: `claude-opus-4-6`

### Supported `--model` Values

From `freephdlabor/utils.py`:

- OpenAI: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.3-codex`, `gpt-4o`, `gpt-4.1-mini-2025-04-14`, `o4-mini-2025-04-16`, `o3-2025-04-16`, `o3-pro-2025-06-10`
- Anthropic: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-20250514`, `claude-sonnet-4-20250514`, `claude-sonnet-4-5`, `claude-sonnet-4-5-20250929`
- Google: `gemini-2.5-pro`, `gemini-2.5-flash`
- DeepSeek: `deepseek-chat`, `deepseek-coder`
- xAI: `grok-4-0709`

Notes:

- "Opus 4.6 Max" is not a separate model ID. Use `claude-opus-4-6` with `effort: max` in `.llm_config.yaml`.

### Budget and Cost Controls

Budget enforcement is implemented in `freephdlabor/budget.py`.

- `budget.usd_limit`: hard spend cap
- `hard_stop: true`: blocks further calls after cap is reached
- `fail_closed: true`: blocks calls when usage cannot be priced
- `pricing`: per-model `input_per_1k` and `output_per_1k` token prices

Per-run budget files are written in workspace root:

- `budget_state.json`
- `budget_ledger.jsonl`
- `budget.lock` (when limit is reached)

### Token Tracking

Per-run token totals:

- `run_token_usage.json`

Private append-only local ledger:

- `.local/private_token_usage/api_token_calls.jsonl`
- `.local/private_token_usage/api_token_calls.txt`

Export a readable report:

```bash
python scripts/export_private_token_report.py
```

### Useful Environment Variables

- LaTeX overrides:
  - `FREEPHDLABOR_PDFLATEX_PATH`
  - `FREEPHDLABOR_BIBTEX_PATH`
- Logging/tracing:
  - `FREEPHDLABOR_LOG_TO_FILES` (default on)
  - `FREEPHDLABOR_ENABLE_TRACING=1` (optional Phoenix tracing)
- Citation retry controls:
  - `FREEPHDLABOR_SS_MAX_RETRIES`
  - `FREEPHDLABOR_SS_BASE_DELAY_SEC`
  - `FREEPHDLABOR_SS_COOLDOWN_SEC`
  - `FREEPHDLABOR_SS_TIMEOUT_SEC`
- Citation cache controls:
  - `FREEPHDLABOR_CITATION_CACHE_TTL_SEC`
  - `FREEPHDLABOR_CITATION_CACHE_MAX_ENTRIES`
- Tool output bounds:
  - `FREEPHDLABOR_SEE_FILE_MAX_CHARS`
  - `FREEPHDLABOR_SEARCH_MAX_CHARS`
  - `FREEPHDLABOR_SEARCH_MAX_MATCHES`

## Usage

### Pipeline Modes

| Mode | What it does | When to use |
|---|---|---|
| `default` | Baseline manager workflow (`Ideation -> Experimentation -> ResourcePreparation -> Writeup -> Proofreading -> Reviewer`) | Most day-to-day runs |
| `full_research` | Mandatory 8-step flow: decomposition, literature review, planning, execution, follow-up loop, outline, full writeup | Best for serious paper generation |
| `quick` | Reduced-depth loops with core truthfulness gates | Fast exploratory passes |

### Common Run Commands

#### Default Mode

```bash
python launch_multiagent.py \
  --task "Investigate this direction and produce a draft with supporting evidence." \
  --pipeline-mode default
```

#### Full Research Mode (Strict)

```bash
python launch_multiagent.py \
  --task "Run the complete literature-plan-execution-writeup loop for this prompt." \
  --pipeline-mode full_research \
  --followup-max-iterations 3 \
  --enable-math-agents \
  --enforce-paper-artifacts \
  --enforce-editorial-artifacts \
  --min-review-score 8 \
  --require-pdf
```

#### Quick Mode

```bash
python launch_multiagent.py \
  --task "Produce a fast exploratory pass of this topic." \
  --pipeline-mode quick
```

### Run the Provided Stable Task Templates

The repository includes staged tasks in `automation_tasks/`.

#### Step 1: Theory Task

```bash
python launch_multiagent.py \
  --pipeline-mode full_research \
  --enable-math-agents \
  --task "$(cat automation_tasks/run1_theory_task_stable.txt)"
```

#### Step 2: Experiment Task (resume same workspace)

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/freephdlabor_<timestamp> \
  --pipeline-mode full_research \
  --task "$(cat automation_tasks/run2_experiment_task_stable.txt)"
```

#### Step 3: Paper Synthesis Task (resume same workspace)

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/freephdlabor_<timestamp> \
  --pipeline-mode full_research \
  --task "$(cat automation_tasks/run3_paper_task_stable.txt)"
```

### Resume an Existing Workspace

```bash
python launch_multiagent.py \
  --resume /absolute/path/to/results/freephdlabor_<timestamp> \
  --task "Continue from current artifacts and improve the final deliverable."
```

### Provide Context Files (`.pdf`, `.md`, `.txt`)

```bash
mkdir -p /absolute/path/to/results/freephdlabor_<timestamp>/inputs
```

Put your files in that `inputs/` directory, then resume with a task that tells agents how to use them.

### Live Steering (Interrupt Without Restart)

Launcher opens a socket at `127.0.0.1:5001` by default (`--callback_host`, `--callback_port`).

From another terminal:

```bash
nc 127.0.0.1 5001
```

Then send:

1. `interrupt` (or `stop` / `pause`)
2. Your instruction
3. Empty line, empty line
4. `m` for modification or `n` for new task

### SLURM / HPC

Use `scripts/launch_multiagent_slurm.sh` as a template.

- Update `#SBATCH` values for your cluster
- Update conda env name if not `freephdlabor`
- Then submit with `sbatch scripts/launch_multiagent_slurm.sh [optional_launch_args]`

## CLI Reference

`python launch_multiagent.py --help`

| Flag | Default | Description |
|---|---|---|
| `--model` | `None` | Model for all agents (overrides `.llm_config.yaml`) |
| `--interpreter` | `python` | Python interpreter path for experiment execution |
| `--debug` | `false` | Enable debug logging |
| `--log-to-files` | env-driven | Force redirect stdout/stderr to `logs/freephdlabor_<timestamp>.{out,err}` |
| `--no-log-to-files` | env-driven | Disable file redirection |
| `--reasoning-effort` | `None` | GPT-5 reasoning level (`none|minimal|low|medium|high|xhigh`) |
| `--verbosity` | `None` | GPT-5 verbosity (`low|medium|high`) |
| `--callback_host` | `127.0.0.1` | Interruption socket host |
| `--callback_port` | `5001` | Interruption socket port |
| `--enable-planning` | `false` | Enable periodic plan/regroup behavior in agents |
| `--planning-interval` | `3` | Replanning interval (steps) |
| `--resume` | `None` | Resume existing workspace directory |
| `--task` | built-in default | Task string (required in practice for controlled runs) |
| `--manager-max-steps` | `None` | Override manager max step budget |
| `--pipeline-mode` | `default` | `default`, `full_research`, or `quick` |
| `--followup-max-iterations` | `3` | Max Step 6 <-> 6.2 follow-up loops in `full_research` |
| `--enable-math-agents` | `false` | Enable theorem/proof pipeline agents |
| `--enforce-paper-artifacts` | `false` | Enforce required paper artifacts before success |
| `--require-pdf` | `false` | Require `final_paper.pdf` |
| `--require-experiment-plan` | `false` | Also require `experiments_to_run_later.md` when artifact checks are on |
| `--enforce-editorial-artifacts` | `false` | Enforce editorial workflow artifacts and verdict gates |
| `--min-review-score` | `8` | Minimum reviewer `overall_score` for strict editorial gate |

Notes:

- `--enforce-paper-artifacts` can be auto-enabled if your `--task` text includes `final_paper` or `experiments_to_run_later`.
- LaTeX prerequisite checks are fail-fast when paper/editorial artifacts are required.

## Writing Better `--task` Prompts

Good prompts include:

1. The core intuition/hypothesis
2. Desired output type (bound, theorem, ablation, benchmark, etc.)
3. Scope boundaries (what to include/exclude)
4. Evidence expectations (theory, experiments, both)

For math-heavy work, read `MATH_RESEARCH_PRIMER.md`.

## Understanding Results

Each run creates:

```text
results/freephdlabor_YYYYMMDD_HHMMSS/
  final_paper.tex
  final_paper.pdf                    # if generated/required
  paper_workspace/
    literature_review.pdf            # full_research
    research_plan.pdf                # full_research
    results_assessment.pdf           # full_research
    followup_decision.json           # full_research
    paper_outline.md                 # full_research
    references.bib
  math_workspace/                    # when --enable-math-agents
    claim_graph.json
    proofs/
    checks/
    lemma_library.md
  inter_agent_messages/
  run_token_usage.json
  budget_state.json                  # when budget.usd_limit is configured
  budget_ledger.jsonl                # when budget.usd_limit is configured
  <agent_name>/
```

### What to Inspect First

1. `final_paper.tex` / `final_paper.pdf`
2. `paper_workspace/followup_decision.json` (in `full_research`)
3. `math_workspace/claim_graph.json` (if math agents are enabled)
4. `math_workspace/checks/*.jsonl` for symbolic/numeric verification evidence
5. `run_token_usage.json` and `budget_ledger.jsonl` for cost/usage accounting

## Quality Gates and Artifact Contracts

### Required Artifacts (manager-side enforcement)

When enabled, manager checks for:

- Base paper gate: `final_paper.tex`
- Optional strict additions:
  - `final_paper.pdf` (with `--require-pdf`)
  - `experiments_to_run_later.md` (with `--require-experiment-plan`)
- In `full_research`, additional required artifacts:
  - `paper_workspace/literature_review.pdf`
  - `paper_workspace/research_plan.pdf`
  - `paper_workspace/results_assessment.pdf`
  - `paper_workspace/followup_decision.json`
- With `--enforce-editorial-artifacts`, additional outputs:
  - `paper_workspace/author_style_guide.md`
  - `paper_workspace/intro_skeleton.tex`
  - `paper_workspace/style_macros.tex`
  - `paper_workspace/reader_contract.json`
  - `paper_workspace/editorial_contract.md`
  - `paper_workspace/theorem_map.json`
  - `paper_workspace/revision_log.md`
  - `paper_workspace/copyedit_report.md`
  - `paper_workspace/review_report.md`
  - `paper_workspace/review_verdict.json`
  - `paper_workspace/claim_traceability.json` (if math agents also enabled)

### Additional Strict Validations

- Review verdict gate (`overall_score >= --min-review-score`, no hard blockers)
- Paper quality validation
- Math acceptance and dependency consistency
- Claim traceability audit (when editorial + math mode are active)

## Architecture (Contributor View)

### Agent Orchestration (Data Flow)

```mermaid
flowchart TD
    userTask["--task prompt"] --> manager["ManagerAgent"]

    manager -->|"delegate"| ideation["IdeationAgent"]
    ideation -->|"working_idea.json"| litAgent["LiteratureReviewAgent"]
    litAgent -->|"literature_review.pdf\nreferences.bib\nsources.json"| planner["ResearchPlannerAgent"]
    planner -->|"research_plan.pdf\ntasks.json"| experiment["ExperimentationAgent"]
    experiment -->|"experiment results"| analysis["ResultsAnalysisAgent"]
    analysis -->|"results_assessment.pdf\nfollowup_decision.json"| manager

    manager -->|"outline task"| writeup["WriteupAgent"]
    manager -->|"organize evidence"| resourcePrep["ResourcePreparationAgent"]
    resourcePrep -->|"paper_workspace/"| writeup
    writeup -->|"final_paper.tex"| proofread["ProofreadingAgent"]
    proofread -->|"copy-edited .tex"| reviewer["ReviewerAgent"]
    reviewer -->|"review_verdict.json\nscore + diagnostics"| manager

    manager -.->|"if --enable-math-agents"| mathLit["MathLiteratureAgent"]
    mathLit -->|"lemma_library.md"| mathProp["MathProposerAgent"]
    mathProp -->|"claim_graph.json"| mathProv["MathProverAgent"]
    mathProv -->|"proofs/*.md"| mathRig["MathRigorousVerifierAgent"]
    mathRig -->|"checks/*.jsonl"| mathEmp["MathEmpiricalVerifierAgent"]
    mathEmp -->|"verified claims"| proofTx["ProofTranscriptionAgent"]
    proofTx -->|"theory_sections.tex\nappendix_proofs.tex"| writeup
```

### Claim Status Progression (Theory Pipeline)

```
proposed --> proved_draft --> verified_symbolic --> verified_numeric --> accepted
                  ^                                       |
                  |              (demoted on failure)     |
                  +---------------------------------------+
```

Claims reaching `accepted` appear as derived results in the paper. Non-accepted claims are labeled as conjectures.

### Module Map

| Module | Purpose |
|---|---|
| `launch_multiagent.py` | Thin entry point (`freephdlabor.runner.main`) |
| `freephdlabor/runner.py` | Run lifecycle: setup, config, model, artifacts, execution |
| `freephdlabor/args.py` | CLI argument definitions |
| `freephdlabor/config.py` | `.llm_config.yaml` loading and provider parameter filtering |
| `freephdlabor/prereqs.py` | LaTeX binary resolution and guidance |
| `freephdlabor/budget.py` | Budget enforcement wrappers and ledgers |
| `freephdlabor/token_usage_tracker.py` | Run-scoped and private token usage tracking |
| `freephdlabor/supervision/` | Validation gates (artifacts, reviews, traceability, math acceptance) |
| `freephdlabor/agents/` | Manager + specialist agent implementations |
| `freephdlabor/toolkits/` | Tool implementations used by agents |

### Toolkit Groups

| Directory | Contains |
|---|---|
| `toolkits/search/` | arXiv tools, OpenDeepSearch, browser/text inspection, VQA |
| `toolkits/filesystem/` | file editing, knowledge-base/repo tools |
| `toolkits/ideation/` | idea generation, novelty check, paper search |
| `toolkits/experimentation/` | experiment execution and idea standardization |
| `toolkits/writeup/` | LaTeX generation/compilation, citation, plotting |
| `toolkits/math/` | claim graph, proof workspace, symbolic rigor, numeric verification |
| `toolkits/communication/` | `talk_to_user` tool |

## Math Research Workflow

For deep usage guidance, see `MATH_RESEARCH_PRIMER.md`.

Common math artifacts:

- `math_workspace/claim_graph.json`
- `math_workspace/proofs/<claim_id>.md`
- `math_workspace/checks/<claim_id>.jsonl`
- `math_workspace/lemma_library.md`

Lemma library CLI:

```bash
python scripts/lemma_library_cli.py --workspace /absolute/path/to/results/freephdlabor_<timestamp>/math_workspace list
python scripts/lemma_library_cli.py --workspace /absolute/path/to/results/freephdlabor_<timestamp>/math_workspace get --lemma-id L_smooth_descent_standard
python scripts/lemma_library_cli.py --workspace /absolute/path/to/results/freephdlabor_<timestamp>/math_workspace touch --lemma-id L_smooth_descent_standard
```

## Runtime and Cost Expectations

Runtime and cost depend heavily on task scope, enabled tools, model choice, and how many revision loops are needed.

- `quick` mode is usually the fastest
- `default` mode often runs longer because of iterative reviewer loops
- `full_research` + math + PDF compilation can be substantially longer and more expensive

Control spend explicitly with `.llm_config.yaml` budget settings and monitor:

- `run_token_usage.json`
- `budget_ledger.jsonl`

## Troubleshooting

### `ModuleNotFoundError` (`yaml`, `smolagents`, `litellm`, etc.)

```bash
./scripts/bootstrap.sh researchlab minimal
conda activate researchlab
python scripts/preflight_check.py
```

### Missing web dependency (`crawl4ai`, etc.)

```bash
./scripts/bootstrap.sh researchlab web
```

### Playwright Chromium missing

```bash
python -m playwright install chromium
```

### LaTeX tools missing (`pdflatex` / `bibtex`)

```bash
./scripts/bootstrap.sh researchlab latex
python scripts/preflight_check.py --with-latex
```

If conda TeX formats are broken:

```bash
./scripts/fix_pdflatex_conda.sh researchlab
```

### `pydub` warning about `ffmpeg`

```bash
brew install ffmpeg
```

### No API key detected

Check `.env` and shell environment variables (`OPENAI_API_KEY`, etc.).

### Reduce citation retries / token burn

```bash
export FREEPHDLABOR_SS_MAX_RETRIES=2
export FREEPHDLABOR_SS_BASE_DELAY_SEC=2
export FREEPHDLABOR_SS_COOLDOWN_SEC=60
```

### Limit oversized tool outputs

```bash
export FREEPHDLABOR_SEE_FILE_MAX_CHARS=12000
export FREEPHDLABOR_SEARCH_MAX_CHARS=12000
export FREEPHDLABOR_SEARCH_MAX_MATCHES=200
```

## Running Tests

```bash
pytest tests/
```

Current deterministic test modules:

- `tests/test_validation.py`
- `tests/test_config.py`
- `tests/test_prereqs.py`

## Repository Hygiene

- Keep generated outputs and local runtime state out of version control (`results/`, `logs/`, `.env`, caches)
- Before pushing changes:

```bash
git status -sb
```

## License

MIT. See `LICENSE`.
