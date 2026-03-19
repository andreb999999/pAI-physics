"""
Knowledge Base Repo Addition Tools
This module contains tools for adding new content to a structured knowledge base.
Supports writing new files, copying files or folders from the working directory,
and appending content to existing files. All updates are automatically indexed
for semantic search.
"""

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any, Type
import shutil
from pathlib import Path
from consortium.toolkits.filesystem.kb_repo_management.repo_indexer import (
    RepoIndexer,
)


class WriteToKnowledgeBaseInput(BaseModel):
    content: str = Field(description="Text or code to write into the file.")
    destination_path: str = Field(description="Relative path of the new file in the knowledge base.")
    overwrite: bool = Field(description="Whether to overwrite if the file already exists.")


class WriteToKnowledgeBase(BaseTool):
    name: str = "write_to_knowledge_base"
    description: str = (
        "Create a new file in the knowledge base and write the given content into it. "
        "If overwrite=True, replaces any existing file. If overwrite=False, adds a numeric suffix to avoid conflict. "
        "Updates the semantic index automatically."
    )
    args_schema: Type[BaseModel] = WriteToKnowledgeBaseInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_indexer: Any = None
    root: Any = None

    def __init__(self, repo_indexer: RepoIndexer, **kwargs):
        super().__init__(
            repo_indexer=repo_indexer,
            root=Path(repo_indexer.root),
            **kwargs,
        )

    def _get_unique_path(self, base_path: Path) -> Path:
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(
                f"{base_path.stem}_{counter}{base_path.suffix}"
            )
            counter += 1
        return new_path

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def _run(self, content: str, destination_path: str, overwrite: bool) -> str:
        try:
            dst = self._safe_kb_path(destination_path)
        except PermissionError as e:
            return str(e)

        if dst.exists() and not overwrite:
            dst = self._get_unique_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)

        self.repo_indexer.update_file(dst)

        return f"Wrote content to '{dst.relative_to(self.root)}'. File has been indexed for semantic search."


class CopyToKnowledgeBaseInput(BaseModel):
    source_path: str = Field(description="Path in the working directory.")
    destination_path: str = Field(description="Target path in the knowledge base.")
    overwrite: bool = Field(description="Whether to overwrite existing files or folders.")


class CopyToKnowledgeBase(BaseTool):
    name: str = "copy_to_knowledge_base"
    description: str = (
        "Copy a file or folder from the working directory to the knowledge base. "
        "If overwrite=True, merges folders or replaces files. If overwrite=False, adds suffix to avoid conflict. "
        "All new or updated files are indexed for semantic search."
    )
    args_schema: Type[BaseModel] = CopyToKnowledgeBaseInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_indexer: Any = None
    working_dir: Any = None
    root: Any = None

    def __init__(self, repo_indexer: RepoIndexer, working_dir: str, **kwargs):
        super().__init__(
            repo_indexer=repo_indexer,
            working_dir=Path(working_dir),
            root=Path(repo_indexer.root),
            **kwargs,
        )

    def _get_unique_path(self, base_path: Path) -> Path:
        counter = 1
        new_path = base_path
        while new_path.exists():
            new_path = base_path.with_name(
                f"{base_path.stem}_{counter}{base_path.suffix}"
            )
            counter += 1
        return new_path

    def _safe_working_path(self, path: str) -> Path:
        abs_root = self.working_dir.resolve()
        abs_path = (self.working_dir / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the working directory is not allowed."
            )
        return abs_path

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def _run(self, source_path: str, destination_path: str, overwrite: bool) -> str:
        try:
            src = self._safe_working_path(source_path)
            dst = self._safe_kb_path(destination_path)
        except PermissionError as e:
            return str(e)

        if not src.exists():
            return f"Error: source '{source_path}' does not exist in the working directory."

        if dst.exists() and not overwrite:
            dst = self._get_unique_path(dst)

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            if dst.exists() and overwrite:
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.rglob("*"):
                    target = dst / item.relative_to(src)
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
            else:
                shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

        updated_files = []
        if dst.is_dir():
            for file in dst.rglob("*.*"):
                self.repo_indexer.update_file(file)
                updated_files.append(str(file.relative_to(self.root)))
        else:
            self.repo_indexer.update_file(dst)
            updated_files.append(str(dst.relative_to(self.root)))

        return (
            f"Copied '{source_path}' to '{dst.relative_to(self.root)}'. "
            f"Indexed {len(updated_files)} file(s): {', '.join(updated_files)}."
        )


class AppendToKnowledgeBaseFileInput(BaseModel):
    target_file: str = Field(description="Relative path of the file in the knowledge base.")
    new_content: str = Field(description="Content to insert into the file.")
    insert_mode: Optional[str] = Field(
        default=None,
        description=(
            "Content insertion position (default: 'end'):\n"
            "* 'end': Append content to the end of the file\n"
            "* 'before': Insert content before the line containing match_string\n"
            "* 'after': Insert content after the line containing match_string\n"
            "Note: 'before' and 'after' modes require match_string parameter"
        ),
    )
    match_string: Optional[str] = Field(
        default=None,
        description="String to locate insertion point for 'before' or 'after' modes.",
    )


class AppendToKnowledgeBaseFile(BaseTool):
    name: str = "append_to_knowledge_base_file"
    description: str = (
        "Append new content to a plain text file in the knowledge base. "
        "You can insert at the end, or before/after a specific line using match_string. "
        "If match_string is not found, the content is added to the end. "
        "Automatically reindexes the file for semantic search."
    )
    args_schema: Type[BaseModel] = AppendToKnowledgeBaseFileInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Any = None
    repo_indexer: Any = None

    def __init__(self, repo_indexer: RepoIndexer, **kwargs):
        super().__init__(
            root=Path(repo_indexer.root),
            repo_indexer=repo_indexer,
            **kwargs,
        )

    def _safe_kb_path(self, path: str) -> Path:
        abs_root = self.root.resolve()
        abs_path = (self.root / path).resolve()
        if not str(abs_path).startswith(str(abs_root)):
            raise PermissionError(
                "Access outside the knowledge base root is not allowed."
            )
        return abs_path

    def _run(
        self,
        target_file: str,
        new_content: str,
        insert_mode: str | None = None,
        match_string: str | None = None,
    ) -> str:
        try:
            filepath = self._safe_kb_path(target_file)
        except PermissionError as e:
            return str(e)

        if not filepath.exists():
            return f"Error: file '{target_file}' does not exist in the knowledge base."

        if not filepath.is_file():
            return f"Error: '{target_file}' is not a file."

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: Cannot read '{target_file}' — it may be a binary or non-text file."

        inserted = False
        new_lines = []

        if insert_mode is None:
            insert_mode = "end"

        if insert_mode == "end" or not match_string:
            lines.append(
                new_content if new_content.endswith("\n") else new_content + "\n"
            )
            inserted = True

        elif insert_mode in {"before", "after"}:
            for i, line in enumerate(lines):
                if match_string in line:
                    if insert_mode == "before":
                        new_lines = lines[:i] + [new_content + "\n"] + lines[i:]
                    else:  # after
                        new_lines = (
                            lines[: i + 1] + [new_content + "\n"] + lines[i + 1 :]
                        )
                    inserted = True
                    break

            if not inserted:
                lines.append(
                    new_content if new_content.endswith("\n") else new_content + "\n"
                )
                inserted = True
            else:
                lines = new_lines

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        self.repo_indexer.update_file(filepath)

        if insert_mode == "end" or not match_string:
            return f"Appended content to the end of '{target_file}'. File has been reindexed."
        elif inserted:
            return f"Inserted content {insert_mode} line matching '{match_string}' in '{target_file}'. File has been reindexed."
        else:
            return f"Match string '{match_string}' not found. Content appended to end of '{target_file}'. File has been reindexed."
