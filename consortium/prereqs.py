"""
LaTeX prerequisite detection and help text for paper-writing runs.
"""

import os
import shutil


def resolve_executable(tool_name, env_var, extra_candidates=None):
    """
    Resolve an executable from env override, common absolute paths, or PATH.
    Returns: (resolved_path|None, error|None)
    """
    override = os.getenv(env_var, "").strip()
    if override:
        candidate = os.path.expanduser(override)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, None
        return None, f"{env_var} points to a missing/non-executable file: {override}"

    for candidate in extra_candidates or []:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate, None

    found = shutil.which(tool_name)
    if found:
        return found, None
    return None, f"{tool_name} not found on PATH"


def latex_prereq_help(env_name, pdflatex_error, bibtex_error):
    issues = []
    if pdflatex_error:
        issues.append(f"- pdflatex: {pdflatex_error}")
    if bibtex_error:
        issues.append(f"- bibtex: {bibtex_error}")
    issue_text = "\n".join(issues) if issues else "- unknown LaTeX toolchain issue"
    return (
        "LaTeX prerequisites are required for paper/editorial runs.\n"
        f"{issue_text}\n\n"
        "Fix options:\n"
        "1) Conda toolchain route:\n"
        f"   ./scripts/bootstrap.sh {env_name} latex\n"
        f"   # or: ./scripts/fix_pdflatex_conda.sh {env_name}\n"
        "2) MacTeX route (macOS):\n"
        "   brew install --cask mactex\n"
        "   eval \"$(/usr/libexec/path_helper)\"\n"
        f"   conda env config vars set -n {env_name} "
        "CONSORTIUM_PDFLATEX_PATH=/Library/TeX/texbin/pdflatex "
        "CONSORTIUM_BIBTEX_PATH=/Library/TeX/texbin/bibtex\n"
        f"   conda deactivate && conda activate {env_name}\n"
        "3) Verify:\n"
        "   python scripts/preflight_check.py --with-latex"
    )


def check_latex_prereqs():
    """
    Detect pdflatex and bibtex. Returns (pdflatex_path, bibtex_path, error_message|None).
    error_message is None if both binaries are found.
    """
    env_name = os.getenv("CONDA_DEFAULT_ENV", "consortium")
    pdflatex_path, pdflatex_err = resolve_executable(
        tool_name="pdflatex",
        env_var="CONSORTIUM_PDFLATEX_PATH",
        extra_candidates=["/Library/TeX/texbin/pdflatex"],
    )
    bibtex_path, bibtex_err = resolve_executable(
        tool_name="bibtex",
        env_var="CONSORTIUM_BIBTEX_PATH",
        extra_candidates=["/Library/TeX/texbin/bibtex"],
    )
    if pdflatex_err or bibtex_err:
        return None, None, latex_prereq_help(env_name, pdflatex_err, bibtex_err)
    return pdflatex_path, bibtex_path, None
