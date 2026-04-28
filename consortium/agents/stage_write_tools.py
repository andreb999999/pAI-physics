"""Restricted file-editing tools for stage-specific write scopes."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Type

from pydantic import ConfigDict

from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent,
    CreateFileWithContentInput,
    ModifyFile,
    ModifyFileInput,
)


class _RestrictedWriteMixin:
    model_config = ConfigDict(arbitrary_types_allowed=True)
    allowed_write_prefixes: tuple[str, ...] = ()

    def __init__(
        self,
        working_dir: str,
        allowed_write_prefixes: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ):
        prefixes = tuple(
            os.path.normpath(prefix)
            for prefix in (allowed_write_prefixes or ())
            if prefix
        )
        super().__init__(working_dir=working_dir, allowed_write_prefixes=prefixes, **kwargs)

    def _safe_path(self, path: str, write: bool = False) -> str:
        abs_path = super()._safe_path(path, write=write)
        if write and self.allowed_write_prefixes:
            rel_path = os.path.normpath(os.path.relpath(abs_path, os.path.abspath(self.working_dir)))
            if rel_path == ".":
                rel_path = ""
            allowed = any(
                rel_path == prefix or rel_path.startswith(prefix.rstrip(os.sep) + os.sep)
                for prefix in self.allowed_write_prefixes
            )
            if not allowed:
                raise PermissionError(
                    f"Write denied: '{path}' is outside this stage's allowed write scope. "
                    f"Allowed prefixes: {list(self.allowed_write_prefixes)}"
                )
        return abs_path


class RestrictedCreateFileWithContent(_RestrictedWriteMixin, CreateFileWithContent):
    args_schema: Type[CreateFileWithContentInput] = CreateFileWithContentInput


class RestrictedModifyFile(_RestrictedWriteMixin, ModifyFile):
    args_schema: Type[ModifyFileInput] = ModifyFileInput
