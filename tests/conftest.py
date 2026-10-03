"""Shared fixtures for the VideOCR test suite.

Import path setup happens at module import time so that every test (and every
test module collected afterwards) can import both packages:

- ``videocr_gui``  — from the repo root
- ``videocr``      — from ``CLI/``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DIR = REPO_ROOT / "CLI"

for _p in (str(REPO_ROOT), str(CLI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must be set before PySide6 creates a platform window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Local test media lives in tests/data/ (gitignored). See AGENTS.md.
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_VIDEO = TEST_DATA_DIR / "test_video_2.mp4"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def test_video() -> Path:
    """Path to the local test video; skipped when absent (it is gitignored)."""
    if not TEST_VIDEO.exists():
        pytest.skip(f"test video not present: {TEST_VIDEO}")
    return TEST_VIDEO


class ErrorLogRecorder(list):
    """List of logged messages that also carries the original log_error callables."""

    originals: dict


@pytest.fixture(autouse=True)
def _error_log(monkeypatch, tmp_path):
    """Redirect all log_error() writes into a per-test recorder.

    Several modules bind ``log_error`` by value at import, so each binding is
    patched separately. Tests can assert on the returned recorder list.
    """
    records = ErrorLogRecorder()

    def _record(message: str, log_name: str = "error_log.txt") -> str:
        records.append(message)
        return str(tmp_path / log_name)

    import videocr_gui.config as config
    import videocr_gui.i18n as i18n
    import videocr_gui.workers as workers

    # Keep the originals reachable for tests that exercise real file logging.
    records.originals = {
        "config": config.log_error,
        "i18n": i18n.log_error,
        "workers": workers.log_error,
    }
    monkeypatch.setattr(config, "log_error", _record)
    monkeypatch.setattr(i18n, "log_error", _record)
    monkeypatch.setattr(workers, "log_error", _record)
    return records


@pytest.fixture(autouse=True)
def _tmp_config_file(monkeypatch, tmp_path):
    """Point config load/save at a throwaway file so the real config is never touched."""
    import videocr_gui.config as config

    target = tmp_path / "videocr_gui_config.ini"
    monkeypatch.setattr(config, "CONFIG_FILE", str(target))
    return target


@pytest.fixture(autouse=True)
def _fresh_i18n():
    """Reset i18n module state before and after every test."""
    import videocr_gui.i18n as i18n

    saved = (i18n.LANG, i18n._current_code, i18n._en_cache)
    i18n.LANG = {}
    i18n._current_code = "en"
    i18n._en_cache = None
    yield
    i18n.LANG, i18n._current_code, i18n._en_cache = saved


@pytest.fixture(autouse=True)
def _fresh_progress():
    """Reset the shared progress-rate-limit state before and after every test."""
    from videocr_gui import progress

    saved = progress._STATE
    progress._STATE = progress.ProgressState()
    yield
    progress._STATE = saved
