"""
Tests for freephdlabor/prereqs.py — LaTeX binary resolution.
"""

import os
import stat
import pytest

from freephdlabor.prereqs import resolve_executable, check_latex_prereqs


class TestResolveExecutable:
    def test_finds_via_env_var(self, tmp_path, monkeypatch):
        fake_bin = tmp_path / "pdflatex"
        fake_bin.write_text("#!/bin/sh\necho hi")
        fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("FREEPHDLABOR_PDFLATEX_PATH", str(fake_bin))
        path, err = resolve_executable("pdflatex", "FREEPHDLABOR_PDFLATEX_PATH")
        assert path == str(fake_bin)
        assert err is None

    def test_env_var_points_to_missing_file(self, monkeypatch):
        monkeypatch.setenv("FREEPHDLABOR_PDFLATEX_PATH", "/nonexistent/pdflatex")
        path, err = resolve_executable("pdflatex", "FREEPHDLABOR_PDFLATEX_PATH")
        assert path is None
        assert "missing" in err.lower() or "non-executable" in err.lower()

    def test_finds_via_extra_candidates(self, tmp_path):
        fake_bin = tmp_path / "bibtex"
        fake_bin.write_text("#!/bin/sh\necho hi")
        fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC)
        path, err = resolve_executable(
            "bibtex_nonexistent_on_path",
            "FREEPHDLABOR_BIBTEX_MISSING_VAR",
            extra_candidates=[str(fake_bin)],
        )
        assert path == str(fake_bin)
        assert err is None

    def test_not_found_returns_error(self, monkeypatch):
        monkeypatch.delenv("FREEPHDLABOR_PDFLATEX_PATH", raising=False)
        path, err = resolve_executable(
            "definitely_not_a_real_binary_xyz123",
            "FREEPHDLABOR_PDFLATEX_PATH",
        )
        assert path is None
        assert err is not None
