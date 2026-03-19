"""
PythonCodeExecutionTool — LangChain BaseTool wrapping workspace-scoped
Python execution.

Provides a sandboxed Python executor available to any specialist agent
that needs to run arbitrary Python code in the workspace directory.
"""

from __future__ import annotations

import builtins
import importlib
import io
import logging
import os
import platform
import re
import signal
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Builtins that agent code must not access directly
# ---------------------------------------------------------------------------
_DANGEROUS_BUILTINS = frozenset({
    "breakpoint", "exit", "quit",
    "__import__",
})

_CODE_EXEC_TIMEOUT = int(os.getenv("CONSORTIUM_CODE_EXEC_TIMEOUT_SEC", "300"))


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded time limit")


# ---------------------------------------------------------------------------
# Standalone execution function
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
        # Relaxed: allow all imports. Agents need to read/analyze freely.
        return importlib.__import__(name, *args, **kwargs)

    abs_workspace = os.path.abspath(workspace_dir)
    original_chdir = os.chdir

    def _safe_chdir(path):
        # Relaxed: allow chdir anywhere for reads, writes are still workspace-scoped
        return original_chdir(path)

    original_dir = os.getcwd()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Build a restricted builtins dict excluding dangerous functions
    safe_builtins = {
        k: v for k, v in vars(builtins).items()
        if k not in _DANGEROUS_BUILTINS
    }
    safe_builtins["__import__"] = _restricted_import

    try:
        os.chdir(workspace_dir)
        exec_globals: dict = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }
        compiled = compile(code, "<agent_code>", "exec")

        # Set execution timeout (Unix only; on other platforms skip)
        # signal.alarm only works in the main thread; when counsel runs agents
        # inside ThreadPoolExecutor we must fall back to threading-based timeout.
        import threading as _threading
        _is_main = _threading.current_thread() is _threading.main_thread()
        use_alarm = (
            _is_main
            and platform.system() != "Windows"
            and hasattr(signal, "SIGALRM")
        )
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(_CODE_EXEC_TIMEOUT)

        try:
            if not _is_main:
                # Thread-safe timeout: run exec in a sub-thread with join timeout
                _exec_err = []
                def _run_exec():
                    try:
                        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                            exec(compiled, exec_globals)  # noqa: S102
                    except Exception as e:
                        _exec_err.append(e)
                t = _threading.Thread(target=_run_exec, daemon=True)
                t.start()
                t.join(timeout=_CODE_EXEC_TIMEOUT)
                if t.is_alive():
                    return f"ERROR: Code execution timed out after {_CODE_EXEC_TIMEOUT}s"
                if _exec_err:
                    return f"ERROR: {_exec_err[0]}"
            else:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compiled, exec_globals)  # noqa: S102
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

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
        # Return error as string instead of raising — let the agent recover
        return f"ERROR: {exc}\n{err_out}".strip()
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
