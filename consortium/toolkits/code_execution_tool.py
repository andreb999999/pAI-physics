"""
PythonCodeExecutionTool — LangChain BaseTool wrapping workspace-scoped
Python execution (Phase 6b).

This replaces WorkspacePythonExecutor as a tool available to any specialist
agent that needs to run arbitrary Python code in the workspace directory.
The inner execution logic (chdir / execute / restore) is preserved from
WorkspacePythonExecutor.__call__().
"""

from __future__ import annotations

import builtins
import importlib
import io
import logging
import os
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Standalone execution function (same logic as WorkspacePythonExecutor)
# ---------------------------------------------------------------------------

def _execute_in_workspace(
    code: str,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
) -> str:
    """
    Execute Python code in workspace_dir using RestrictedPython-backed
    execution.  Returns a string combining stdout output and any error.
    Raises on hard syntax / security errors so the agent sees them.
    """
    # Guard against non-Python DSL that some models occasionally emit
    if re.search(r"(?m)^\s*to\s*=\s*[A-Za-z_]\w*\s+code\b", code or ""):
        raise SyntaxError(
            "Non-Python tool-call syntax detected (`to=<tool> code`). "
            "Use executable Python only inside a ```python block."
        )

    allowed = set(authorized_imports or [])

    def _restricted_import(name, *args, **kwargs):
        root = name.split(".")[0]
        if allowed and root not in allowed:
            raise ImportError(
                f"Forbidden import '{name}'. Add it to authorized_imports to allow it."
            )
        return importlib.__import__(name, *args, **kwargs)

    original_dir = os.getcwd()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        os.chdir(workspace_dir)
        exec_globals: dict = {
            "__builtins__": {**vars(builtins), "__import__": _restricted_import},
            "__name__": "__main__",
        }
        compiled = compile(code, "<agent_code>", "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, exec_globals)  # noqa: S102

        parts = []
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        if out:
            parts.append(out.rstrip())
        if err:
            parts.append(f"[stderr]\n{err.rstrip()}")
        return "\n".join(parts) if parts else "(no output)"

    except Exception as exc:
        err_out = stderr_buf.getvalue()
        logging.error("Code execution error: %s", exc)
        raise RuntimeError(f"{exc}\n{err_out}".strip()) from exc
    finally:
        os.chdir(original_dir)


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------

class CodeExecutionInput(BaseModel):
    code: str = Field(description="Python code to execute in the workspace directory.")


# ---------------------------------------------------------------------------
# LangChain BaseTool
# ---------------------------------------------------------------------------

class PythonCodeExecutionTool(BaseTool):
    """Execute arbitrary Python code in the agent's workspace directory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "python_repl"
    description: str = (
        "Execute arbitrary Python code in the workspace directory. "
        "Use this to run experiments, manipulate files, compute results, "
        "plot data, or do anything that requires Python execution. "
        "Returns combined stdout / result output. Raises on syntax or "
        "security errors."
    )
    args_schema: Type[BaseModel] = CodeExecutionInput

    workspace_dir: str
    authorized_imports: List[str] = Field(default_factory=list)

    def __init__(self, workspace_dir: str, authorized_imports: Optional[List[str]] = None, **kwargs: Any):
        super().__init__(
            workspace_dir=os.path.abspath(workspace_dir),
            authorized_imports=authorized_imports or [],
            **kwargs,
        )
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _run(self, code: str) -> str:
        return _execute_in_workspace(
            code=code,
            workspace_dir=self.workspace_dir,
            authorized_imports=self.authorized_imports,
        )

    async def _arun(self, code: str) -> str:  # type: ignore[override]
        raise NotImplementedError("async execution not supported")
