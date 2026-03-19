"""Smoke test for the full counsel pipeline (sandbox → debate → synthesis)."""
import os
import sys
import time
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Load API keys from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Use cheaper/faster models for testing — 1 debate round, short timeout
TEST_SPECS = [
    {"model": "claude-sonnet-4-6", "reasoning_effort": "low"},
    {"model": "gpt-5.4",           "reasoning_effort": "low", "verbosity": "low"},
    {"model": "gemini-3-pro-preview"},
]

TASK = (
    "Explain in 2-3 sentences why the harmonic series (sum of 1/n for n=1 to infinity) "
    "diverges even though its terms approach zero. Be precise and concise."
)
SYSTEM_PROMPT = "You are a mathematics expert. Give clear, concise answers."


def main():
    from consortium.counsel import run_counsel_stage, create_counsel_models

    workspace = tempfile.mkdtemp(prefix="counsel_test_")
    print(f"Workspace: {workspace}")

    # Seed file so sandbox copy has content
    with open(os.path.join(workspace, "context.txt"), "w") as f:
        f.write("Counsel smoke test.\n")

    # Create models
    print(f"Models: {[s['model'] for s in TEST_SPECS]}")
    models = create_counsel_models(
        budget_config={}, budget_dir=workspace, model_specs=TEST_SPECS
    )
    print(f"Created {len(models)} models\n")

    t0 = time.time()
    try:
        result = run_counsel_stage(
            task=TASK,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            workspace_dir=workspace,
            counsel_models=models,
            agent_name="test_counsel",
            max_debate_rounds=1,
            model_specs=TEST_SPECS,
            model_timeout_seconds=120,
        )
        elapsed = time.time() - t0

        print(f"\n{'='*60}")
        print(f"COUNSEL COMPLETE in {elapsed:.1f}s")
        print(f"{'='*60}")
        print(f"\n--- Synthesis output (len={len(result)}) ---")
        print(result[:2000])
        if len(result) > 2000:
            print(f"\n... ({len(result) - 2000} chars truncated)")

        # Sanity checks
        if not result or len(result) < 20:
            print("\nFAIL: Output too short")
            sys.exit(1)
        if result.startswith("[") and "error" in result.lower():
            print(f"\nFAIL: Output is an error: {result[:200]}")
            sys.exit(1)

        print("\nPASS")

    except Exception as e:
        print(f"\nFAIL after {time.time() - t0:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
