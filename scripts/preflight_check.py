#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def check_import(module_name: str) -> str | None:
    try:
        importlib.import_module(module_name)
        return None
    except Exception as exc:
        return f"{module_name}: {exc}"


def check_playwright_chromium() -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return f"playwright import failed: {exc}"

    try:
        with sync_playwright() as p:
            exe = Path(p.chromium.executable_path)
            if not exe.exists():
                return (
                    "Chromium browser binary is missing. "
                    "Run: python -m playwright install chromium"
                )
    except Exception as exc:
        return f"playwright chromium check failed: {exc}"
    return None


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    # Load repo-local .env so API key checks reflect real runtime behavior.
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=env_path, override=False)
        except Exception as exc:
            warnings.append(f"Could not load .env: {exc}")

    required_modules = [
        "smolagents",
        "litellm",
        "yaml",
        "dotenv",
        "fitz",
        "pdfminer",
        "crawl4ai",
        "mammoth",
        "markdownify",
        "pptx",
        "puremagic",
        "pydub",
        "speech_recognition",
        "youtube_transcript_api",
    ]

    for mod in required_modules:
        err = check_import(mod)
        if err:
            errors.append(err)

    err = check_playwright_chromium()
    if err:
        errors.append(err)

    api_key_names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    if not any(os.getenv(name) for name in api_key_names):
        warnings.append(
            "No API key detected in environment. Add at least one key in .env or shell env vars."
        )

    llm_cfg = REPO_ROOT / ".llm_config.yaml"
    if not llm_cfg.exists():
        warnings.append(".llm_config.yaml not found at repo root.")

    print("=== freephdlabor preflight ===")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("No dependency errors found.")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nPreflight failed.")
        return 1

    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
