from __future__ import annotations
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import os
import importlib.util


def _safe_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value >= 0 else default
    except Exception:
        return default


def _truncate_text(content: str, max_chars: int, tool_name: str) -> str:
    if max_chars <= 0 or len(content) <= max_chars:
        return content

    # Keep beginning + middle + end to reduce blind spots.
    head_chars = max(1, int(max_chars * 0.5))
    mid_chars = max(1, int(max_chars * 0.2))
    tail_chars = max(1, max_chars - head_chars - mid_chars)

    head = content[:head_chars]
    mid_start = max(0, (len(content) // 2) - (mid_chars // 2))
    mid = content[mid_start: mid_start + mid_chars]
    tail = content[-tail_chars:]
    omitted = len(content) - len(head) - len(mid) - len(tail)
    return (
        f"[{tool_name}] Output truncated to {max_chars} characters "
        f"(omitted {omitted} chars). "
        "Use search_keyword for targeted extraction.\n\n"
        f"--- Begin excerpt ---\n{head}\n\n"
        f"--- Middle excerpt ---\n{mid}\n\n"
        f"--- End excerpt ---\n{tail}"
    )


class ListDirInput(BaseModel):
    directory: str = Field(description="The directory to check.")


class ListDir(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "list_dir"
    description: str = (
        "List files in the chosen directory. Use this to explore the directory structure. "
        "Note: only files under the allowed working directory are accessible."
    )
    args_schema: Type[BaseModel] = ListDirInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, directory: str) -> str:
        try:
            chosen_dir = self._safe_path(directory)
        except PermissionError as e:
            return str(e)
        except FileNotFoundError as e:
            return str(e)
        if not os.path.exists(chosen_dir):
            return f"The directory {directory} does not exist. Please start checking from the root directory."
        files = os.listdir(chosen_dir)
        if files == []:
            return f"The directory {directory} is empty."
        else:
            return '\n'.join(files)

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # For directory listing, directories should exist
            if not os.path.exists(abs_path):
                # Provide helpful error for agent
                parent_dir = os.path.dirname(abs_path)
                if os.path.exists(parent_dir):
                    raise FileNotFoundError(
                        f"Directory not found: '{path}' does not exist in the workspace. "
                        f"The parent directory exists. Please check the directory name."
                    )
                else:
                    raise FileNotFoundError(
                        f"Directory not found: '{path}' does not exist in the workspace. "
                        f"The parent directory '{os.path.dirname(path)}' was not found."
                    )

            return abs_path


class SeeFileInput(BaseModel):
    filename: str = Field(description="Name of the file to check.")


class SeeFile(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "see_file"
    description: str = (
        "Read workspace files quickly. Use for code files, configs, logs, and simple text files in your workspace. "
        "Returns clean file content without line numbers. "
        "For PDFs or complex documents, use inspect_file_as_text instead."
    )
    args_schema: Type[BaseModel] = SeeFileInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, filename: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        except FileNotFoundError as e:
            return str(e)
        if not os.path.exists(filepath):
            return f"The file {filename} does not exist."
        with open(filepath, "r", encoding="utf-8", errors="replace") as file:
            content = file.read()
        max_chars = _safe_int_env("CONSORTIUM_SEE_FILE_MAX_CHARS", 12000)
        return _truncate_text(content, max_chars=max_chars, tool_name="see_file")

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # Return path (existence checks handled by individual tools as needed)
            return abs_path


class ModifyFileInput(BaseModel):
    filename: str = Field(description="Name of the file to modify.")
    start_line: int = Field(description="Start line number to replace.")
    end_line: int = Field(description="End line number to replace.")
    new_content: str = Field(description="New content to insert (with proper indentation).")


class ModifyFile(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "modify_file"
    description: str = (
        "Modify a plain text file by replacing specific lines with new content. "
        "Only works with plain text files (e.g., .txt, .py, .md). Ensure correct indentation. "
        "Not applicable for binary files such as .pdf, .docx, or spreadsheets."
    )
    args_schema: Type[BaseModel] = ModifyFileInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, filename: str, start_line: int, end_line: int, new_content: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        if not os.path.exists(filepath):
            return f"The file {filename} does not exist."
        with open(filepath, "r+") as file:
            lines = file.readlines()
            lines[start_line - 1:end_line] = [new_content + "\n"]
            file.seek(0)
            file.truncate()
            file.write("".join(lines))
        return "Content modified."

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # Return path (existence checks handled by individual tools as needed)
            return abs_path


class CreateFileWithContentInput(BaseModel):
    filename: str = Field(description="Name of the file to create.")
    content: str = Field(description="Content to write into the file.")


class CreateFileWithContent(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "create_file_with_content"
    description: str = (
        "Create a new plain text file (e.g., .txt, .py, .md) and write content into it. "
        "This tool does not support creating binary files such as .pdf, .docx, or images."
    )
    args_schema: Type[BaseModel] = CreateFileWithContentInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, filename: str, content: str) -> str:
        try:
            filepath = self._safe_path(filename)
        except PermissionError as e:
            return str(e)
        with open(filepath, "w") as file:
            file.write(content)
        return "File created successfully."

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # Return path (existence checks handled by individual tools as needed)
            return abs_path


class SearchKeywordInput(BaseModel):
    path: str = Field(description="Path to the file or folder to search in.")
    keyword: str = Field(description="Keyword to search for.")
    context_lines: int = Field(description="Number of lines to include before and after each match.")


class SearchKeyword(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "search_keyword"
    description: str = (
        "Search for a keyword in a plain text file or recursively in all plain text files within a folder. "
        "Returns matching lines with file names, line numbers and context lines before and after each match. "
        "Only supports plain text files (e.g., .txt, .py, .md). Not suitable for binary formats like .pdf, .docx, .xlsx."
    )
    args_schema: Type[BaseModel] = SearchKeywordInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, path: str, keyword: str, context_lines: int) -> str:
        try:
            target_path = self._safe_path(path)
        except PermissionError as e:
            return str(e)
        if not os.path.exists(target_path):
            return f"The path '{path}' does not exist."

        if os.path.isfile(target_path):
            return self._search_in_file(target_path, keyword, context_lines, display_path=path)
        elif os.path.isdir(target_path):
            results = []
            for root, _, files in os.walk(target_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.relpath(fpath, self.working_dir)
                    try:
                        result = self._search_in_file(fpath, keyword, context_lines, display_path=rel_path)
                        if "No matches found" not in result:
                            results.append(result)
                    except Exception as e:
                        results.append(f"[{rel_path}]: Error reading file ({e})")
            if not results:
                return f"No matches found for '{keyword}' in folder '{path}'."
            body = "\n\n".join(results)
            max_chars = _safe_int_env("CONSORTIUM_SEARCH_MAX_CHARS", 12000)
            return _truncate_text(body, max_chars=max_chars, tool_name="search_keyword")
        else:
            return f"The path '{path}' is neither a file nor a directory."

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _search_in_file(self, filepath: str, keyword: str, context_lines: int, display_path: str) -> str:
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            return f"[{display_path}]: Cannot read binary or non-text file."

        num_lines = len(lines)
        match_indices = [i for i, line in enumerate(lines) if keyword in line]
        max_matches = _safe_int_env("CONSORTIUM_SEARCH_MAX_MATCHES", 200)
        truncated_matches = False
        if max_matches > 0 and len(match_indices) > max_matches:
            match_indices = match_indices[:max_matches]
            truncated_matches = True

        if not match_indices:
            return f"[{display_path}]: No matches found for '{keyword}'."

        output_lines = set()
        for idx in match_indices:
            start = max(0, idx - context_lines)
            end = min(num_lines, idx + context_lines + 1)
            output_lines.update(range(start, end))

        sorted_output = sorted(output_lines)
        formatted_output = [f"{i+1}: {lines[i].rstrip()}" for i in sorted_output]

        body = f"--- Matches in [{display_path}] ---\n" + "\n".join(formatted_output)
        if truncated_matches:
            body += (
                f"\n\n[search_keyword] Match list truncated to first {max_matches} matches. "
                "Narrow the keyword or search within a smaller file scope."
            )

        max_chars = _safe_int_env("CONSORTIUM_SEARCH_MAX_CHARS", 12000)
        return _truncate_text(body, max_chars=max_chars, tool_name="search_keyword")

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # Return path (existence checks handled by individual tools as needed)
            return abs_path


class DeleteFileOrFolderInput(BaseModel):
    filename: str = Field(description="Name of the file or folder to delete. Use empty string only for explicit workspace wipe.")
    confirmation_token: Optional[str] = Field(default=None, description="Required only when filename is empty. Must match CONSORTIUM_WIPE_CONFIRM_TOKEN.")


class DeleteFileOrFolder(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "delete_file_or_folder"
    description: str = (
        "Delete a specified file or folder. This action is irreversible."
        "If no filename is provided, a valid confirmation token is required before deleting everything in the working directory."
        "Only files under the allowed working directory are accessible."
    )
    args_schema: Type[BaseModel] = DeleteFileOrFolderInput
    working_dir: Optional[str] = None

    def __init__(self, working_dir, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir), **kwargs)

    def _run(self, filename: str, confirmation_token: str = None) -> str:
        if filename == "":
            required_token = os.getenv("CONSORTIUM_WIPE_CONFIRM_TOKEN", "").strip()
            provided_token = (confirmation_token or "").strip()
            if not required_token or provided_token != required_token:
                return (
                    "Refusing workspace wipe. To wipe all files, set CONSORTIUM_WIPE_CONFIRM_TOKEN "
                    "and pass matching confirmation_token."
                )
            abs_working_dir = os.path.abspath(self.working_dir)
            # Only delete inside the working directory
            for root, dirs, files in os.walk(abs_working_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            return "All files and folders in the working directory have been deleted."
        else:
            try:
                filepath = self._safe_path(filename)
            except PermissionError as e:
                return str(e)
            if os.path.exists(filepath):
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    return f"The file {filename} has been deleted."
                elif os.path.isdir(filepath):
                    os.rmdir(filepath)
                    return f"The folder {filename} has been deleted."
                else:
                    return f"The path {filename} is neither a file nor a folder."
            else:
                return f"The file or folder {filename} does not exist."

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute workspace path with clear error messages for agents."""
        if not self.working_dir:
            return path

        abs_working_dir = os.path.abspath(self.working_dir)

        # Check if input path is absolute or relative
        if os.path.isabs(path):
            # Absolute path handling
            abs_path = os.path.abspath(path)

            # Check if within workspace
            if abs_path.startswith(abs_working_dir):
                return abs_path
            else:
                # Provide actionable error for agent
                raise PermissionError(
                    f"Access denied: The absolute path '{path}' is outside the workspace. "
                    f"Please use a relative path or an absolute path within '{abs_working_dir}'. "
                    f"Example: Use 'subdirectory/filename.txt' instead of the full path."
                )
        else:
            # Relative path - join with workspace
            abs_path = os.path.abspath(os.path.join(abs_working_dir, path))

            # Return path (existence checks handled by individual tools as needed)
            return abs_path


# class LoadObjectFromPythonFile(BaseTool):
#     name = "load_object_from_python_file"
#     description = "Load a class or method from a Python file so it can be used by the agent."
#     inputs = {
#         "filename": {"type": "string", "description": "The Python file to load from."},
#         "object_name": {"type": "string", "description": "The name of the class or method to load."}
#     }
#     output_type = "object"  # We return an actual callable Python object

#     def __init__(self, working_dir: str):
#         super().__init__()
#         self.working_dir = working_dir

#     def forward(self, filename: str, object_name: str) -> Any:
#         try:
#             file_path = self._safe_path(filename)
#         except PermissionError as e:
#             raise FileNotFoundError(str(e))
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {filename} does not exist.")

#         # Create a module spec
#         module_name = os.path.splitext(os.path.basename(file_path))[0]
#         spec = importlib.util.spec_from_file_location(module_name, file_path)

#         if spec is None or spec.loader is None:
#             raise ImportError(f"Could not load spec for file {filename}")

#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)

#         if not hasattr(module, object_name):
#             raise AttributeError(f"The object {object_name} was not found in {filename}")

#         return getattr(module, object_name)

#     def _safe_path(self, path: str) -> str:
#         # Prevent absolute paths and directory traversal
#         abs_working_dir = os.path.abspath(self.working_dir)
#         abs_path = os.path.abspath(os.path.join(self.working_dir, path))
#         if not abs_path.startswith(abs_working_dir):
#             raise PermissionError("Access outside the working directory is not allowed.")
#         return abs_path
