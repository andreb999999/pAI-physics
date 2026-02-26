#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate freephdlabor runtime dependencies and optional capabilities."
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Treat document parser stack as required.",
    )
    parser.add_argument(
        "--with-web",
        action="store_true",
        help="Treat web crawling stack as required.",
    )
    parser.add_argument(
        "--with-experiment",
        action="store_true",
        help="Treat experiment tool stack as required.",
    )
    return parser.parse_args()


def _check_modules(modules: list[str], errors: list[str], warnings: list[str], required: bool) -> None:
    for mod in modules:
        err = check_import(mod)
        if err:
            if required:
                errors.append(err)
            else:
                warnings.append(err)


def main() -> int:
    args = parse_args()
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
        "requests",
        "bs4",
        "pypdf",
    ]
    docs_modules = [
        "mammoth",
        "markdownify",
        "pptx",
        "puremagic",
        "pydub",
        "speech_recognition",
        "youtube_transcript_api",
    ]
    web_modules = [
        "crawl4ai",
        "playwright",
    ]
    experiment_modules = [
        "transformers",
        "datasets",
    ]

    _check_modules(required_modules, errors, warnings, required=True)
    _check_modules(docs_modules, errors, warnings, required=args.with_docs)
    _check_modules(web_modules, errors, warnings, required=args.with_web)
    _check_modules(experiment_modules, errors, warnings, required=args.with_experiment)

    # Validate chromium only if web capability is requested/installed.
    if args.with_web or check_import("playwright") is None:
        err = check_playwright_chromium()
        if err:
            if args.with_web:
                errors.append(err)
            else:
                warnings.append(err)

    # Warn if pydub is installed but ffmpeg is not available.
    if check_import("pydub") is None and shutil.which("ffmpeg") is None:
        warnings.append(
            "pydub is installed but ffmpeg is not on PATH. "
            "Install ffmpeg for audio transcription features."
        )

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
