"""
LangGraph research pipeline — directly wired multi-phase workflow.

This graph replaces the manager hub-and-spoke loop with:
1. Discovery: ideation -> literature review -> research planner
2. Parallel execution: theory and experiment tracks in parallel
3. Synthesis loop: merge -> synthesis literature review -> results analysis
4. Paper production: resource preparation -> writeup -> proofreading -> reviewer
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents import (
    build_experiment_design_node,
    build_experiment_literature_node,
    build_experiment_transcription_node,
    build_experiment_verification_node,
    build_experimentation_node,
    build_literature_review_node,
    build_math_empirical_verifier_node,
    build_math_literature_node,
    build_math_proposer_node,
    build_math_prover_node,
    build_math_rigorous_verifier_node,
    build_proof_transcription_node,
    build_proofreading_node,
    build_resource_preparation_node,
    build_results_analysis_node,
    build_reviewer_node,
    build_track_merge_node,
    build_writeup_node,
    build_brainstorm_node,
    build_formalize_goals_node,
    build_formalize_results_node,
)
from .milestone_report import (
    generate_milestone_report,
    wait_for_human_response,
)
from .pdf_summary import with_pdf_summary
from .state import ResearchState
from .workflow_utils import (
    followup_decision_requires_loop,
    run_intermediate_validation,
    run_validation_gates,
    safe_int,
)

MATH_PIPELINE_STAGES = [
    "math_literature_agent",
    "math_proposer_agent",
    "math_prover_agent",
    "math_rigorous_verifier_agent",
    "math_empirical_verifier_agent",
    "proof_transcription_agent",
]

EXPERIMENT_PIPELINE_STAGES = [
    "experiment_literature_agent",
    "experiment_design_agent",
    "experimentation_agent",
    "experiment_verification_agent",
    "experiment_transcription_agent",
]

# ---------------------------------------------------------------------------
# V2 pipeline stage rosters
# ---------------------------------------------------------------------------

V2_PRE_TRACK_STAGES = [
    "persona_council",
    "literature_review_agent",
    "brainstorm_agent",
    "formalize_goals_agent",
]

V2_POST_TRACK_STAGES = [
    "formalize_results_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent",
]


def build_pipeline_stages_v2(enable_math_agents: bool) -> list[str]:
    stages = list(V2_PRE_TRACK_STAGES)
    if enable_math_agents:
        stages.extend(MATH_PIPELINE_STAGES)
    stages.extend(EXPERIMENT_PIPELINE_STAGES)
    stages.extend(V2_POST_TRACK_STAGES)
    return stages


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _format_track_task(state: dict, track_name: str, questions: list[str]) -> str:
    cycle = safe_int(state.get("research_cycle", 0), 0)
    if not questions:
        return f"No {track_name} questions were identified for this cycle."

    question_lines = "\n".join(f"- {question}" for question in questions)
    return (
        f"Research cycle: {cycle}\n"
        f"Execute the {track_name} track for the current research plan.\n"
        f"Questions assigned to this track:\n{question_lines}\n\n"
        "Read the latest planning artifacts from `paper_workspace/`, "
        "produce the mandatory artifacts for your track, and ground all work "
        "in workspace evidence and cited literature."
    )


def track_router(state: ResearchState) -> list[Send]:
    """Fan out to the theory and/or experiment tracks based on track decomposition."""
    track_decomposition = state.get("track_decomposition") or {}
    theory_questions = list(track_decomposition.get("theory_questions") or [])
    empirical_questions = list(track_decomposition.get("empirical_questions") or [])
    recommended_track = str(track_decomposition.get("recommended_track", "")).strip().lower()

    sends: list[Send] = []
    theory_allowed = state.get("math_enabled", False) and recommended_track in {"", "both", "theory"}
    experiment_allowed = recommended_track in {"", "both", "empirical"}

    if theory_allowed and theory_questions:
        sends.append(
            Send(
                "theory_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "theory", theory_questions),
                    "theory_track_status": "in_progress",
                },
            )
        )

    if experiment_allowed and empirical_questions:
        sends.append(
            Send(
                "experiment_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "experiment", empirical_questions),
                    "experiment_track_status": "in_progress",
                },
            )
        )

    if not sends:
        sends.append(
            Send(
                "track_merge",
                {
                    **state,
                    "agent_task": (
                        "No theory or empirical execution track was selected. "
                        "Proceed directly to synthesis and results analysis."
                    ),
                },
            )
        )
    return sends


def followup_router(state: ResearchState) -> str:
    # Routing decision was already made by followup_gate_node which sets current_agent
    # to "research_planner_agent" (loop) or "resource_preparation_agent" (continue).
    return state.get("current_agent") or "resource_preparation_agent"


def validation_router(state: ResearchState) -> str:
    if state.get("finished"):
        return END
    return "writeup_agent"


def build_followup_gate_node(workspace_dir: str) -> Any:
    def followup_gate_node(state: dict) -> dict:
        required, reason = followup_decision_requires_loop(workspace_dir)
        research_cycle = safe_int(state.get("research_cycle", 0), 0)
        max_cycles = max(0, safe_int(state.get("max_research_cycles", 3), 3))

        if required and research_cycle < max_cycles:
            return {
                "current_agent": "research_planner_agent",
                "research_cycle": research_cycle + 1,
                "followup_iteration": safe_int(state.get("followup_iteration", 0), 0) + 1,
                "finished": False,
                "agent_task": (
                    "Prepare a focused follow-up research plan based on "
                    f"results analysis. Reason: {reason}"
                ),
            }

        return {
            "current_agent": "resource_preparation_agent",
            "finished": False,
            "agent_task": None,
        }

    followup_gate_node.__name__ = "followup_gate"
    return followup_gate_node


def build_validation_gate_node() -> Any:
    def validation_gate_node(state: dict) -> dict:
        validation = run_validation_gates(state)
        if validation["gate_passed"]:
            return {
                "validation_results": validation["validation_results"],
                "finished": True,
                "agent_task": None,
            }

        error_lines = [
            f"- {gate}: {'; '.join(result['errors'])}"
            for gate, result in validation["validation_results"].items()
            if not result.get("is_valid")
        ]
        return {
            "validation_results": validation["validation_results"],
            "finished": False,
            "agent_task": (
                "Revise the paper to satisfy validation gates before finalization.\n"
                "Validation failures:\n" + "\n".join(error_lines)
            ),
        }

    validation_gate_node.__name__ = "validation_gate"
    return validation_gate_node


def build_novelty_gate_node(workspace_dir: str) -> Any:
    """Gate between ideation and literature review that checks novelty assessment."""

    def novelty_gate_node(state: dict) -> dict:
        assessment_path = os.path.join(
            workspace_dir, "paper_workspace", "novelty_assessment.json"
        )
        max_attempts = 3
        attempts = safe_int(state.get("novelty_check_attempts", 0), 0)

        if not os.path.exists(assessment_path):
            # No assessment file -- pass through (backward compat)
            return {"current_agent": "literature_review_agent", "agent_task": None}

        try:
            with open(assessment_path) as f:
                assessment = json.load(f)
        except Exception:
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if assessment.get("novel", True):
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if attempts >= max_attempts:
            # After max attempts, proceed anyway with a warning
            print(
                f"Warning: novelty gate failed after {max_attempts} attempts, "
                "proceeding to literature review."
            )
            return {"current_agent": "literature_review_agent", "agent_task": None}

        justification = assessment.get("novelty_justification", "N/A")
        closest = assessment.get("closest_existing_work", "N/A")
        return {
            "current_agent": "ideation_agent",
            "novelty_check_attempts": attempts + 1,
            "agent_task": (
                f"NOVELTY GATE REJECTION (attempt {attempts + 1}/{max_attempts}): "
                f"Your previous idea was assessed as NOT NOVEL.\n"
                f"Justification: {justification}\n"
                f"Closest existing work: {closest}\n\n"
                "Generate a substantially different idea that addresses a genuine "
                "gap in the literature. Delete the old novelty_assessment.json and "
                "run CheckIdeaNoveltyTool again on your new idea."
            ),
        }

    novelty_gate_node.__name__ = "novelty_gate"
    return novelty_gate_node


def build_milestone_gate_node(phase_name: str, workspace_dir: str) -> Any:
    """Build a milestone gate node that generates a report and optionally pauses.

    When ``enable_milestone_gates`` is True in state, the node blocks until a
    human responds via ``POST /milestone_response`` or the timeout expires.
    When disabled, it generates the PDF report but does not pause.
    """

    # Map milestone phase names to intermediate validation checkpoint names
    _PHASE_TO_CHECKPOINT = {
        "research_plan": None,           # no validation yet at planning stage
        "track_results": "post_merge",   # cross-track consistency
        "analysis": "post_analysis",     # early claim traceability
        "review": None,                  # final validation handles this
    }

    def milestone_gate_node(state: dict) -> dict:
        # Run intermediate validation if applicable for this phase
        checkpoint = _PHASE_TO_CHECKPOINT.get(phase_name)
        validation_update = {}
        if checkpoint:
            validation_update = run_intermediate_validation(state, checkpoint)

        budget_status = None
        budget_path = os.path.join(workspace_dir, "budget_state.json")
        if os.path.exists(budget_path):
            try:
                with open(budget_path) as f:
                    import json as _json
                    bd = _json.load(f)
                total = bd.get("total_usd", bd.get("total_cost_usd"))
                limit = bd.get("usd_limit")
                if total is not None and limit is not None:
                    budget_status = f"Budget: ${total:.2f} / ${limit:.2f} ({total / limit * 100:.0f}%)"
                elif total is not None:
                    budget_status = f"Budget spent: ${total:.2f}"
            except Exception:
                pass

        # Merge validation log into state before generating report
        # so the milestone report can display intermediate results
        merged_state = {**state, **validation_update}

        report_path = generate_milestone_report(
            phase=phase_name,
            state=merged_state,
            workspace_dir=workspace_dir,
            budget_status=budget_status,
        )

        prev_reports = list(state.get("milestone_reports") or [])
        if report_path:
            prev_reports.append(report_path)

        update: dict = {"milestone_reports": prev_reports}
        update.update(validation_update)

        if state.get("enable_milestone_gates"):
            timeout = state.get("milestone_timeout", 3600)
            feedback = wait_for_human_response(timeout=timeout)
            if feedback:
                update["human_feedback"] = feedback
                action = feedback.get("action", "approve")
                if action == "modify":
                    update["agent_task"] = feedback.get("feedback", "")
                elif action == "abort":
                    update["finished"] = True

        return update

    milestone_gate_node.__name__ = f"milestone_{phase_name}"
    return milestone_gate_node


def novelty_router(state: ResearchState) -> str:
    """Route based on novelty gate decision."""
    return state.get("current_agent") or "literature_review_agent"


def build_theory_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    model_registry: Optional[Any] = None,
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _m(agent_name: str) -> Any:
        if model_registry is not None:
            return model_registry.get(agent_name)
        return model

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "math_literature_agent",
        _wrap(build_math_literature_node(_m("math_literature_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_literature_agent"),
    )
    graph.add_node(
        "math_proposer_agent",
        _wrap(build_math_proposer_node(_m("math_proposer_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_proposer_agent"),
    )
    graph.add_node(
        "math_prover_agent",
        _wrap(build_math_prover_node(_m("math_prover_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_prover_agent"),
    )
    graph.add_node(
        "math_rigorous_verifier_agent",
        _wrap(build_math_rigorous_verifier_node(_m("math_rigorous_verifier_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_rigorous_verifier_agent"),
    )
    graph.add_node(
        "math_empirical_verifier_agent",
        _wrap(build_math_empirical_verifier_node(_m("math_empirical_verifier_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "math_empirical_verifier_agent"),
    )
    graph.add_node(
        "proof_transcription_agent",
        _wrap(build_proof_transcription_node(_m("proof_transcription_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "proof_transcription_agent"),
    )
    graph.set_entry_point("math_literature_agent")
    graph.add_edge("math_literature_agent", "math_proposer_agent")
    graph.add_edge("math_proposer_agent", "math_prover_agent")
    graph.add_edge("math_prover_agent", "math_rigorous_verifier_agent")
    graph.add_edge("math_rigorous_verifier_agent", "math_empirical_verifier_agent")
    graph.add_edge("math_empirical_verifier_agent", "proof_transcription_agent")
    graph.add_edge("proof_transcription_agent", END)
    return graph.compile()


def build_experiment_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    model_registry: Optional[Any] = None,
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _m(agent_name: str) -> Any:
        if model_registry is not None:
            return model_registry.get(agent_name)
        return model

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "experiment_literature_agent",
        _wrap(build_experiment_literature_node(_m("experiment_literature_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_literature_agent"),
    )
    graph.add_node(
        "experiment_design_agent",
        _wrap(build_experiment_design_node(_m("experiment_design_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_design_agent"),
    )
    graph.add_node(
        "experimentation_agent",
        _wrap(build_experimentation_node(_m("experimentation_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experimentation_agent"),
    )
    graph.add_node(
        "experiment_verification_agent",
        _wrap(build_experiment_verification_node(_m("experiment_verification_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_verification_agent"),
    )
    graph.add_node(
        "experiment_transcription_agent",
        _wrap(build_experiment_transcription_node(_m("experiment_transcription_agent"), workspace_dir, authorized_imports, **counsel_kwargs), "experiment_transcription_agent"),
    )
    graph.set_entry_point("experiment_literature_agent")
    graph.add_edge("experiment_literature_agent", "experiment_design_agent")
    graph.add_edge("experiment_design_agent", "experimentation_agent")
    graph.add_edge("experimentation_agent", "experiment_verification_agent")
    graph.add_edge("experiment_verification_agent", "experiment_transcription_agent")
    graph.add_edge("experiment_transcription_agent", END)
    return graph.compile()


def build_track_subgraph_node(
    subgraph: Any,
    status_field: str,
    status_value: str = "completed",
) -> Any:
    def node(state: dict) -> dict:
        final_state = subgraph.invoke(state)
        return {
            "agent_outputs": final_state.get("agent_outputs", {}),
            status_field: status_value,
            "agent_task": None,
        }

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_noop_track_node(status_field: str) -> Any:
    def node(state: dict) -> dict:
        return {
            status_field: state.get(status_field),
            "agent_task": None,
        }

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_synthesis_literature_node(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
) -> Any:
    base_node = build_literature_review_node(
        model,
        workspace_dir,
        authorized_imports,
        **({"counsel_models": counsel_models} if counsel_models else {}),
    )

    def synthesis_node(state: dict) -> dict:
        previous_output = state.get("agent_outputs", {}).get("literature_review_agent")
        result = base_node(state)
        outputs = dict(result.get("agent_outputs", {}))
        new_output = outputs.pop("literature_review_agent", previous_output)
        if previous_output is not None:
            outputs["literature_review_agent"] = previous_output
        outputs["synthesis_literature_review_agent"] = new_output
        return {
            **result,
            "agent_outputs": outputs,
        }

    synthesis_node.__name__ = "synthesis_literature_review_agent"
    return synthesis_node


# ---------------------------------------------------------------------------
# Pipeline — gate nodes and routers
# ---------------------------------------------------------------------------


def _read_file_safe(path: str, max_chars: int = 20000) -> str:
    """Read a file, returning empty string on failure."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def build_lit_review_gate_node(workspace_dir: str, max_attempts: int = 2) -> Any:
    """Gate after lit review: checks novelty flags then feasibility.

    Routes to:
      - ``persona_council`` if novelty-blocked or infeasible (retries remain)
      - ``brainstorm_agent`` if feasible or retries exhausted
    """

    def lit_review_gate_node(state: dict) -> dict:
        import re as _re
        attempts = safe_int(state.get("lit_review_attempts", 0), 0)

        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        lit_text = _read_file_safe(os.path.join(paper_ws, "literature_review.tex"))
        sources_text = _read_file_safe(
            os.path.join(paper_ws, "literature_review_sources.json"), 10000
        )

        if not lit_text:
            # No lit review output — pass through
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": "no lit review found"},
                "agent_task": None,
            }

        # ------------------------------------------------------------------
        # Novelty check — read novelty_flags.json (produced by Step 3b)
        # ------------------------------------------------------------------
        novelty_path = os.path.join(paper_ws, "novelty_flags.json")
        novelty_text = _read_file_safe(novelty_path, 10000)
        novelty_data = None
        blocking_claims = []

        if novelty_text:
            try:
                novelty_data = json.loads(novelty_text)
                blocking_claims = [
                    c for c in novelty_data.get("claims", [])
                    if c.get("blocking", False)
                ]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[lit_review_gate] Failed to parse novelty_flags.json: {e}")

        # Gate on blocking novelty claims BEFORE the LLM feasibility call
        if blocking_claims:
            blocking_summary = "; ".join(
                f"[{c.get('claim_id', '?')}] "
                f"{c.get('claim_text', '')[:150]}... "
                f"(status: {c.get('status', '?')}, "
                f"evidence: {len(c.get('evidence', []))} sources)"
                for c in blocking_claims
            )

            if attempts >= max_attempts:
                print(
                    f"Warning: lit_review_gate — blocking novelty claims after "
                    f"{max_attempts} attempts, proceeding to brainstorm_agent."
                )
                return {
                    "current_agent": "brainstorm_agent",
                    "lit_review_feasibility": {
                        "feasible": False,
                        "reason": f"NOVELTY BLOCKED: {blocking_summary}",
                    },
                    "agent_task": (
                        f"NOVELTY WARNING (max retries reached): Some proposed claims "
                        f"may not be novel. Proceed but explicitly acknowledge and "
                        f"work around: {blocking_summary}"
                    ),
                }

            return {
                "current_agent": "persona_council",
                "lit_review_attempts": attempts + 1,
                "lit_review_feasibility": {
                    "feasible": False,
                    "reason": f"NOVELTY BLOCKED: {blocking_summary}",
                },
                "agent_task": (
                    f"NOVELTY GATE REJECTION (attempt {attempts + 1}/{max_attempts}):\n"
                    f"The literature review found that one or more core claims in your "
                    f"research proposal are NOT novel — they have already been proven or "
                    f"established in the existing literature.\n\n"
                    f"Blocking claims:\n{blocking_summary}\n\n"
                    f"You MUST substantially reformulate or replace the blocking claims. "
                    f"Options:\n"
                    f"1. Strengthen the claim (generalize, relax assumptions, extend to "
                    f"new settings)\n"
                    f"2. Replace the claim with a genuinely open related problem\n"
                    f"3. Reframe the contribution as a new proof technique for a known "
                    f"result (only if the technique itself is novel)\n"
                    f"4. Pivot the research direction entirely\n\n"
                    f"Read `paper_workspace/novelty_flags.json` for full evidence."
                ),
            }

        # ------------------------------------------------------------------
        # LLM feasibility assessment
        # ------------------------------------------------------------------
        import litellm as _litellm

        # Build novelty context for the feasibility prompt
        novelty_context = ""
        if novelty_data:
            open_count = sum(
                1 for c in novelty_data.get("claims", [])
                if c.get("status") == "OPEN"
            )
            partial_count = sum(
                1 for c in novelty_data.get("claims", [])
                if c.get("status") == "PARTIAL"
            )
            novelty_context = (
                f"\n\nNOVELTY ASSESSMENT SUMMARY:\n"
                f"- Fully open claims: {open_count}\n"
                f"- Partially resolved claims: {partial_count}\n"
                f"- Blocking (known/equivalent) claims: {len(blocking_claims)}\n"
                f"Overall: {novelty_data.get('overall_novelty_assessment', 'N/A')}"
            )

        prompt = (
            "You are a research feasibility assessor. Given the literature review below, "
            "determine whether the proposed research direction is FEASIBLE.\n\n"
            "A direction is INFEASIBLE if:\n"
            "1. The core idea has already been thoroughly explored (no room for contribution)\n"
            "2. The required methods/data are fundamentally unavailable\n"
            "3. The theoretical foundations have known fatal flaws\n"
            "4. The scope is impossibly broad for a single research effort\n"
            "5. The core proposed claims are already proven results with no novel angle "
            "(check the NOVELTY ASSESSMENT SUMMARY below if available)\n\n"
            f"LITERATURE REVIEW:\n{lit_text[:15000]}\n\n"
            f"SOURCES:\n{sources_text[:5000]}"
            f"{novelty_context}\n\n"
            'Respond in JSON (no markdown fences): {"feasible": true/false, "reason": "one paragraph explanation"}'
        )
        try:
            resp = _litellm.completion(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
            raw = _re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = _re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
        except Exception as e:
            print(f"[lit_review_gate] LLM assessment failed: {e}, passing through")
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": f"assessment failed: {e}"},
                "agent_task": None,
            }

        feasible = result.get("feasible", True)
        reason = result.get("reason", "N/A")

        if feasible:
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": True, "reason": reason},
                "agent_task": (
                    f"Novelty summary from literature review: "
                    f"{novelty_data.get('overall_novelty_assessment', 'N/A')}"
                ) if novelty_data else None,
            }

        # Infeasible
        if attempts >= max_attempts:
            print(
                f"Warning: lit_review_gate failed after {max_attempts} attempts, "
                "proceeding to brainstorm_agent anyway."
            )
            return {
                "current_agent": "brainstorm_agent",
                "lit_review_feasibility": {"feasible": False, "reason": reason},
                "agent_task": None,
            }

        return {
            "current_agent": "persona_council",
            "lit_review_attempts": attempts + 1,
            "lit_review_feasibility": {"feasible": False, "reason": reason},
            "agent_task": (
                f"LIT REVIEW FEASIBILITY REJECTION (attempt {attempts + 1}/{max_attempts}): "
                f"The literature review found the research direction INFEASIBLE.\n"
                f"Reason: {reason}\n\n"
                "Re-evaluate the research direction. Consider pivoting the angle, "
                "narrowing scope, or identifying an unexplored niche."
            ),
        }

    lit_review_gate_node.__name__ = "lit_review_gate"
    return lit_review_gate_node


def build_verify_completion_node(workspace_dir: str) -> Any:
    """Verify whether formalized research goals have been met by track execution.

    Three-way routing with progress vetting on re-entry:
      - ``formalize_results_agent``: all/nearly all goals met (ratio >= 0.8)
      - ``formalize_goals_agent``: some goals not met (incomplete, ratio 0.5-0.8)
      - ``brainstorm_agent``: fundamental rethink needed (ratio < 0.5)

    On 2nd+ cycle, if goals_met delta <= 0 since last check, forces forward.
    """

    def verify_completion_node(state: dict) -> dict:
        import re as _re

        research_goals = state.get("research_goals") or {}
        goals = research_goals.get("goals", [])
        total_goals = len(goals)

        if total_goals == 0:
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": {
                    "goals_met": 0, "goals_total": 0, "ratio": 1.0,
                    "verdict": "complete", "goal_verdicts": [],
                },
                "agent_task": None,
            }

        # Collect evidence from workspace
        evidence_parts = []
        math_ws = os.path.join(workspace_dir, "math_workspace")
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        exp_ws = os.path.join(workspace_dir, "experiment_workspace")

        for name in ["claim_graph.json"]:
            content = _read_file_safe(os.path.join(math_ws, name), 8000)
            if content:
                evidence_parts.append(f"--- THEORY: {name} ---\n{content}")
        for name in ["experiment_results.json", "experiment_design.json"]:
            content = _read_file_safe(os.path.join(paper_ws, name), 8000)
            if content:
                evidence_parts.append(f"--- EXPERIMENT: {name} ---\n{content}")
        for name in ["results_summary.json", "experiment_report.md"]:
            content = _read_file_safe(os.path.join(exp_ws, name), 5000)
            if content:
                evidence_parts.append(f"--- EXPERIMENT: {name} ---\n{content}")

        agent_outputs = state.get("agent_outputs", {})
        for agent_name in ["proof_transcription_agent", "experiment_transcription_agent",
                           "experiment_verification_agent", "track_merge"]:
            output = str(agent_outputs.get(agent_name, "")).strip()
            if output:
                evidence_parts.append(f"--- AGENT: {agent_name} ---\n{output[:3000]}")

        evidence = "\n\n".join(evidence_parts) if evidence_parts else "No execution evidence found."

        goal_descriptions = "\n".join(
            f"Goal {i+1} (ID: {g.get('id', i)}):\n"
            f"  Description: {g.get('description', 'N/A')}\n"
            f"  Success Criteria: {g.get('success_criteria', 'N/A')}\n"
            f"  Track: {g.get('track', 'both')}"
            for i, g in enumerate(goals)
        )

        import litellm as _litellm

        prompt = (
            "You are a rigorous research goal completion assessor.\n\n"
            f"RESEARCH GOALS:\n{goal_descriptions}\n\n"
            f"EXECUTION EVIDENCE:\n{evidence}\n\n"
            "For EACH goal, assess whether it has been MET based on the evidence.\n"
            "A goal is MET if concrete evidence (data, proofs, results) supports completion.\n"
            "A goal is NOT MET if no evidence exists or evidence contradicts it.\n\n"
            "Respond in JSON (no markdown fences):\n"
            '{"goal_verdicts": [{"goal_id": "...", "met": true/false, "evidence_summary": "..."}], '
            '"overall_assessment": "one paragraph"}'
        )

        try:
            resp = _litellm.completion(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content or ""
            raw = _re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = _re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
        except Exception as e:
            print(f"[verify_completion] LLM assessment failed: {e}, passing through")
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": {
                    "goals_met": total_goals, "goals_total": total_goals,
                    "ratio": 1.0, "verdict": "complete",
                    "goal_verdicts": [], "error": str(e),
                },
                "agent_task": None,
            }

        goal_verdicts = result.get("goal_verdicts", [])
        goals_met = sum(1 for v in goal_verdicts if v.get("met", False))
        ratio = goals_met / total_goals if total_goals > 0 else 1.0

        verify_rework = safe_int(state.get("verify_rework_attempts", 0), 0)
        brainstorm_cyc = safe_int(state.get("brainstorm_cycle", 0), 0)

        # Progress vetting: compare against previous result
        prev_result = state.get("verify_completion_result")
        prev_history = list(state.get("verify_completion_history") or [])
        if prev_result is not None:
            prev_history.append(prev_result)
            prev_goals_met = prev_result.get("goals_met", 0)
            delta = goals_met - prev_goals_met
            if delta <= 0:
                print(
                    f"[verify_completion] Progress stalled: {goals_met} goals met "
                    f"(prev {prev_goals_met}, delta {delta}). Forcing forward."
                )
                return {
                    "current_agent": "formalize_results_agent",
                    "verify_completion_result": {
                        "goals_met": goals_met, "goals_total": total_goals,
                        "ratio": ratio, "verdict": "complete_forced_stalled",
                        "goal_verdicts": goal_verdicts,
                    },
                    "verify_completion_history": prev_history,
                    "agent_task": None,
                }

        new_result = {
            "goals_met": goals_met, "goals_total": total_goals,
            "ratio": ratio, "goal_verdicts": goal_verdicts,
        }

        # Three-way decision
        if ratio >= 0.8:
            new_result["verdict"] = "complete"
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": None,
            }

        if ratio >= 0.5:
            # Incomplete — rework goals
            if verify_rework >= 3:
                print(
                    f"Warning: verify_completion 'incomplete' after {verify_rework} "
                    "rework attempts, forcing forward."
                )
                new_result["verdict"] = "complete_forced"
                return {
                    "current_agent": "formalize_results_agent",
                    "verify_completion_result": new_result,
                    "verify_completion_history": prev_history,
                    "agent_task": None,
                }

            new_result["verdict"] = "incomplete"
            failed_goals = [v for v in goal_verdicts if not v.get("met", False)]
            failed_summary = "\n".join(
                f"  - {v.get('goal_id', '?')}: {v.get('evidence_summary', 'no evidence')}"
                for v in failed_goals
            )
            return {
                "current_agent": "formalize_goals_agent",
                "verify_rework_attempts": verify_rework + 1,
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": (
                    f"VERIFY COMPLETION: INCOMPLETE (attempt {verify_rework + 1}/3)\n"
                    f"{goals_met}/{total_goals} goals met ({ratio:.0%}).\n"
                    f"Failed goals:\n{failed_summary}\n\n"
                    "Revise the research goals to address the unmet criteria."
                ),
            }

        # < 50% — fundamental rethink
        if brainstorm_cyc >= 3:
            print(
                f"Warning: verify_completion 'no_half' after {brainstorm_cyc} "
                "brainstorm cycles, forcing forward."
            )
            new_result["verdict"] = "complete_forced"
            return {
                "current_agent": "formalize_results_agent",
                "verify_completion_result": new_result,
                "verify_completion_history": prev_history,
                "agent_task": None,
            }

        new_result["verdict"] = "no_half"
        return {
            "current_agent": "brainstorm_agent",
            "brainstorm_cycle": brainstorm_cyc + 1,
            "verify_completion_result": new_result,
            "verify_completion_history": prev_history,
            "agent_task": (
                f"VERIFY COMPLETION: FUNDAMENTAL RETHINK NEEDED (cycle {brainstorm_cyc + 1}/3)\n"
                f"Only {goals_met}/{total_goals} goals met ({ratio:.0%}).\n"
                f"Assessment: {result.get('overall_assessment', 'N/A')}\n\n"
                "The current approach is not working. Brainstorm substantially "
                "different approaches to the research question."
            ),
        }

    verify_completion_node.__name__ = "verify_completion"
    return verify_completion_node


def build_duality_gate_node() -> Any:
    """Gate after duality check: routes based on Check A + B results.

    Routes to:
      - ``resource_preparation_agent`` if both checks pass
      - ``followup_lit_review`` if either check fails (retries remaining)
    """

    def duality_gate_node(state: dict) -> dict:
        duality_result = state.get("duality_check_result") or {}
        both_passed = duality_result.get("both_passed", True)
        duality_rework = safe_int(state.get("duality_rework_attempts", 0), 0)
        max_attempts = 2

        if both_passed:
            return {
                "current_agent": "resource_preparation_agent",
                "agent_task": None,
            }

        if duality_rework >= max_attempts:
            print(
                f"Warning: duality_gate failed after {max_attempts} attempts, "
                "proceeding to resource_preparation anyway."
            )
            return {
                "current_agent": "resource_preparation_agent",
                "agent_task": None,
            }

        check_a = duality_result.get("check_a", {})
        check_b = duality_result.get("check_b", {})
        failures = []
        suggestions = []
        if not check_a.get("passed", True):
            failures.append(f"Check A (Practice): {check_a.get('reasoning', 'Failed')}")
            suggestions.extend(check_a.get("suggestions", []))
        if not check_b.get("passed", True):
            failures.append(f"Check B (Rigor): {check_b.get('reasoning', 'Failed')}")
            suggestions.extend(check_b.get("suggestions", []))

        return {
            "current_agent": "followup_lit_review",
            "duality_rework_attempts": duality_rework + 1,
            "agent_task": (
                f"DUALITY CHECK FAILURE (attempt {duality_rework + 1}/{max_attempts}):\n"
                + "\n".join(f"- {f}" for f in failures)
                + "\n\nSuggestions:\n"
                + "\n".join(f"- {s}" for s in suggestions)
                + "\n\nConduct a targeted follow-up literature review addressing "
                "these gaps, then re-enter the brainstorm phase."
            ),
        }

    duality_gate_node.__name__ = "duality_gate"
    return duality_gate_node


# V2 router functions

def lit_review_gate_router(state: ResearchState) -> str:
    return state.get("current_agent") or "brainstorm_agent"


def verify_completion_router(state: ResearchState) -> str:
    return state.get("current_agent") or "formalize_results_agent"


def duality_gate_router(state: ResearchState) -> str:
    return state.get("current_agent") or "resource_preparation_agent"


def build_followup_lit_review_node(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
) -> Any:
    """Followup lit review: reuses literature_review_agent under a different output key."""
    base_node = build_literature_review_node(
        model,
        workspace_dir,
        authorized_imports,
        **({"counsel_models": counsel_models} if counsel_models else {}),
    )

    def followup_node(state: dict) -> dict:
        previous_output = state.get("agent_outputs", {}).get("literature_review_agent")
        result = base_node(state)
        outputs = dict(result.get("agent_outputs", {}))
        new_output = outputs.pop("literature_review_agent", previous_output)
        if previous_output is not None:
            outputs["literature_review_agent"] = previous_output
        outputs["followup_lit_review_agent"] = new_output
        return {
            **result,
            "agent_outputs": outputs,
        }

    followup_node.__name__ = "followup_lit_review"
    return followup_node


# ---------------------------------------------------------------------------
# V2 graph builder
# ---------------------------------------------------------------------------

def build_research_graph_v2(
    model: Any,
    workspace_dir: str,
    pipeline_mode: str = "default",
    enable_math_agents: bool = False,
    enforce_paper_artifacts: bool = False,
    enforce_editorial_artifacts: bool = False,
    require_pdf: bool = False,
    require_experiment_plan: bool = False,
    min_review_score: int = 8,
    followup_max_iterations: int = 3,
    manager_max_steps: int = 50,
    authorized_imports: Optional[List[str]] = None,
    checkpointer=None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    tree_search_config: Optional[Any] = None,
    enable_milestone_gates: bool = False,
    adversarial_verification: bool = False,
    persona_council_specs: Optional[List[dict]] = None,
    persona_debate_rounds: int = 3,
    persona_synthesis_model: str = "claude-opus-4-6",
    persona_max_post_vote_retries: int = 1,
    lit_review_max_attempts: int = 2,
    duality_check_model: str = "claude-opus-4-6",
    enable_duality_check: bool = True,
    budget_manager: Optional[Any] = None,
    model_registry: Optional[Any] = None,
):
    """
    Build the V2 persona-council-driven research pipeline.

    Flow:
      persona_council → lit_review → lit_review_gate → brainstorm → formalize_goals
      → {theory | experiment} → track_merge → verify_completion → formalize_results
      → duality_check → duality_gate → resource_prep → writeup → proofreading
      → reviewer → validation_gate

    With feedback loops:
      - lit_review_gate → persona_council (infeasible)
      - verify_completion → formalize_goals (incomplete)
      - verify_completion → brainstorm (fundamental rethink)
      - duality_gate → followup_lit_review → brainstorm (quality failure)
    """
    from .persona_council import create_persona_council_node, create_duality_check_node

    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _m(agent_name: str) -> Any:
        """Resolve the model for *agent_name* from the registry or fallback."""
        if model_registry is not None:
            return model_registry.get(agent_name)
        return model

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    # Build track subgraphs (same as v1)
    theory_track_node = build_noop_track_node("theory_track_status")
    if enable_math_agents:
        if tree_search_config and getattr(tree_search_config, "enabled", False):
            from consortium.tree_search.graph_integration import (
                build_tree_search_theory_track,
            )
            theory_subgraph = build_tree_search_theory_track(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=counsel_models,
                summary_model_id=summary_model_id,
                tree_config=tree_search_config,
                adversarial_verification=adversarial_verification,
            )
        else:
            theory_subgraph = build_theory_track_subgraph(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=counsel_models,
                summary_model_id=summary_model_id,
                model_registry=model_registry,
            )
        theory_track_node = build_track_subgraph_node(theory_subgraph, "theory_track_status")

    if tree_search_config and getattr(tree_search_config, "enabled", False):
        from consortium.tree_search.experiment_tree_integration import (
            build_tree_search_experiment_track,
        )
        experiment_subgraph = build_tree_search_experiment_track(
            model=model,
            workspace_dir=workspace_dir,
            authorized_imports=authorized_imports,
            counsel_models=counsel_models,
            summary_model_id=summary_model_id,
            tree_config=tree_search_config,
            adversarial_verification=adversarial_verification,
        )
    else:
        experiment_subgraph = build_experiment_track_subgraph(
            model=model,
            workspace_dir=workspace_dir,
            authorized_imports=authorized_imports,
            counsel_models=counsel_models,
            summary_model_id=summary_model_id,
            model_registry=model_registry,
        )

    # Build all nodes
    nodes: dict[str, Any] = {
        # Pre-track (new v2 flow)
        "persona_council": create_persona_council_node(
            workspace_dir=workspace_dir,
            persona_specs=persona_council_specs,
            max_debate_rounds=persona_debate_rounds,
            synthesis_model=persona_synthesis_model,
            max_post_vote_retries=persona_max_post_vote_retries,
            budget_manager=budget_manager,
        ),
        "literature_review_agent": _wrap(
            build_literature_review_node(_m("literature_review_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "literature_review_agent",
        ),
        "lit_review_gate": build_lit_review_gate_node(workspace_dir, max_attempts=lit_review_max_attempts),
        "brainstorm_agent": _wrap(
            build_brainstorm_node(_m("brainstorm_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "brainstorm_agent",
        ),
        "formalize_goals_agent": _wrap(
            build_formalize_goals_node(_m("formalize_goals_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "formalize_goals_agent",
        ),
        "milestone_goals": build_milestone_gate_node("research_plan", workspace_dir),
        # Execution tracks (reused from v1)
        "theory_track": theory_track_node,
        "experiment_track": build_track_subgraph_node(experiment_subgraph, "experiment_track_status"),
        "track_merge": build_track_merge_node(workspace_dir=workspace_dir),
        # Post-track verification (new v2 gates)
        "verify_completion": build_verify_completion_node(workspace_dir),
        "formalize_results_agent": _wrap(
            build_formalize_results_node(_m("formalize_results_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "formalize_results_agent",
        ),
        "followup_lit_review": _wrap(
            build_followup_lit_review_node(_m("followup_lit_review"), workspace_dir, authorized_imports, counsel_models),
            "followup_lit_review",
        ),
        # Paper production chain (reused from v1)
        "resource_preparation_agent": _wrap(
            build_resource_preparation_node(_m("resource_preparation_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "resource_preparation_agent",
        ),
        "writeup_agent": _wrap(
            build_writeup_node(_m("writeup_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "writeup_agent",
        ),
        "proofreading_agent": _wrap(
            build_proofreading_node(_m("proofreading_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "proofreading_agent",
        ),
        "reviewer_agent": _wrap(
            build_reviewer_node(_m("reviewer_agent"), workspace_dir, authorized_imports, **counsel_kwargs),
            "reviewer_agent",
        ),
        "validation_gate": build_validation_gate_node(),
        "milestone_review": build_milestone_gate_node("review", workspace_dir),
    }

    # Add duality check nodes if enabled
    if enable_duality_check:
        nodes["duality_check"] = create_duality_check_node(
            workspace_dir=workspace_dir,
            check_model=duality_check_model,
            budget_manager=budget_manager,
        )
        nodes["duality_gate"] = build_duality_gate_node()

    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    # --- Edge wiring ---

    # Entry: persona council
    graph.set_entry_point("persona_council")
    graph.add_edge("persona_council", "literature_review_agent")

    # Lit review → gate
    graph.add_edge("literature_review_agent", "lit_review_gate")
    graph.add_conditional_edges(
        "lit_review_gate",
        lit_review_gate_router,
        {
            "persona_council": "persona_council",
            "brainstorm_agent": "brainstorm_agent",
        },
    )

    # Brainstorm → formalize goals → milestone → track execution
    graph.add_edge("brainstorm_agent", "formalize_goals_agent")
    graph.add_edge("formalize_goals_agent", "milestone_goals")
    graph.add_conditional_edges("milestone_goals", track_router)

    # Track execution → merge → verify
    graph.add_edge("theory_track", "track_merge")
    graph.add_edge("experiment_track", "track_merge")
    graph.add_edge("track_merge", "verify_completion")

    # Verify completion three-way routing
    graph.add_conditional_edges(
        "verify_completion",
        verify_completion_router,
        {
            "formalize_results_agent": "formalize_results_agent",
            "formalize_goals_agent": "formalize_goals_agent",
            "brainstorm_agent": "brainstorm_agent",
        },
    )

    if enable_duality_check:
        # Formalize results → duality check → duality gate
        graph.add_edge("formalize_results_agent", "duality_check")
        graph.add_edge("duality_check", "duality_gate")
        graph.add_conditional_edges(
            "duality_gate",
            duality_gate_router,
            {
                "resource_preparation_agent": "resource_preparation_agent",
                "followup_lit_review": "followup_lit_review",
            },
        )
        # Followup lit review → brainstorm (duality failure path)
        graph.add_edge("followup_lit_review", "brainstorm_agent")
    else:
        # No duality check — go straight to paper production
        graph.add_edge("formalize_results_agent", "resource_preparation_agent")

    # Paper production chain (same as v1)
    graph.add_edge("resource_preparation_agent", "writeup_agent")
    graph.add_edge("writeup_agent", "proofreading_agent")
    graph.add_edge("proofreading_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", "milestone_review")
    graph.add_edge("milestone_review", "validation_gate")
    graph.add_conditional_edges(
        "validation_gate",
        validation_router,
        {
            END: END,
            "writeup_agent": "writeup_agent",
        },
    )

    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def get_default_checkpointer(workspace_dir: str):
    """Return a SqliteSaver checkpointer scoped to the workspace directory."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        db_path = os.path.join(workspace_dir, "checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except (ImportError, Exception) as e:
        print(f"Checkpointer unavailable ({e}); resumability disabled.")
        return None
