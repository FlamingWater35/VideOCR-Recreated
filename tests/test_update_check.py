"""Tests for the GitHub release update check (no network access)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from videocr_gui import update_check
from videocr_gui.update_check import fetch_latest_version, is_newer


def test_boot_delay_is_3_seconds():
    assert update_check.BOOT_CHECK_DELAY_MS == 3000


class TestIsNewer:
    @pytest.mark.parametrize(
        ("latest", "current", "expected"),
        [
            ("1.7.0", "1.6.3", True),
            ("v1.7.0", "1.6.3", True),
            ("1.10.0", "1.9.0", True),
            ("2.0", "1.6.3", True),
            ("1.6.3", "1.6.3", False),
            ("1.6.2", "1.6.3", False),
            ("1.6.3", "1.6.3.1", False),
            ("", "1.6.3", False),
            ("not-a-version", "1.6.3", False),
        ],
    )
    def test_is_newer(self, latest: str, current: str, expected: bool) -> None:
        assert is_newer(latest, current) is expected


class TestFetchLatestVersion:
    """Calls the real fetch_latest_version with a faked urlopen (no network)."""

    class _Response:
        def __init__(self, payload: object) -> None:
            self._raw = json.dumps(payload).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

    def test_returns_tag_without_v_prefix(self, monkeypatch):
        seen: dict[str, object] = {}

        def _open(request, timeout=None):
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return self._Response({"tag_name": "v1.7.0"})

        monkeypatch.setattr(update_check.urllib.request, "urlopen", _open)
        assert fetch_latest_version() == "1.7.0"
        assert seen["url"] == update_check._RELEASES_API_URL
        assert isinstance(seen["timeout"], float)

    def test_network_error_returns_none(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(update_check.urllib.request, "urlopen", _raise)
        assert fetch_latest_version() is None

    def test_missing_tag_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            update_check.urllib.request,
            "urlopen",
            lambda *args, **kwargs: self._Response({"name": "untagged"}),
        )
        assert fetch_latest_version() is None

    def test_non_dict_payload_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            update_check.urllib.request,
            "urlopen",
            lambda *args, **kwargs: self._Response([1, 2, 3]),
        )
        assert fetch_latest_version() is None
