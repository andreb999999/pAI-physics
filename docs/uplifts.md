# Platform Uplifts

This page records compatibility patches and platform-specific adaptations applied on top of the upstream `PoggioAI/PoggioAI_MSc` codebase.

---

## Linux → Windows (April 2026)

**Branch:** `MSc_Prod`  
**Motivation:** The upstream codebase targets Linux/macOS. Running on Windows 11 caused 6 test failures and one runtime crash in the `.env` loader.

### Changes

#### `consortium/campaign/status.py` — `is_pid_alive()`
On Windows, `os.kill(pid, 0)` raises `OSError` (WinError 11) for invalid or inaccessible PIDs instead of `ProcessLookupError`. Added an `except OSError: return False` branch so liveness checks do not propagate the exception.

#### `consortium/paper_contract.py` — artifact path constants
All path constants (`PAPER_CONTRACT_PATH`, `FINAL_PAPER_TEX`, etc.) were built with `os.path.join`, which produces backslash-separated strings on Windows. Replaced with explicit forward-slash strings (e.g. `"paper_workspace/paper_contract.json"`). These are relative artifact identifiers, not OS filesystem paths, so forward slashes are correct on all platforms. Also updated `canonical_section_paths()` to use an f-string instead of `os.path.join`.

#### `consortium/cli/core/env_manager.py` — `.env` file reader
`_load_env_file()` opened files without an explicit encoding. On Windows, the `~/.msc/.env` file written by `msc setup` contains an em-dash (`—`, byte `0x97` in Windows-1252) in the generated comment header, causing a `UnicodeDecodeError` when Python tried to decode it as UTF-8. Added `encoding="utf-8", errors="replace"` to the `open()` call.

#### `tests/test_campaign_recovery.py` — YAML path embedding
Two tests (`test_stage_env_is_loaded_from_campaign_spec`, `test_manual_failed_status_preserves_existing_reason_and_supports_override`) embedded `tmp_path` and `task_file` directly into YAML double-quoted strings via f-strings. On Windows, paths like `C:\Users\...` contain `\U`, which is a YAML Unicode escape sequence, causing a `yaml.ScannerError`. Fixed by calling `.as_posix()` on all `pathlib.Path` objects before embedding them in YAML.

#### `tests/test_campaign_recovery.py` — `CONSORTIUM_LIT_RATE_STATE_DIR` assertion
`test_campaign_launch_detaches_stage_and_keeps_attempt_logs` asserted `.endswith("iterate_v4/.lit_rate_state")`. On Windows the value uses backslashes. Fixed with `.replace("\\", "/")` before the assertion.

#### `tests/test_cli_contracts.py` — `.env` isolation
`test_runtime_env_ignores_repo_env_outside_repo_root` deleted `OPENROUTER_API_KEY` from the process environment but `build_runtime_env` still loaded the key from the user's real `~/.msc/.env`. Fixed by passing an empty temp directory as `config_dir_override` to both `build_runtime_env` and `get_runtime_env_sources`, preventing the test from reading any real credentials.

### Result
234/234 tests pass on Windows after these changes.
