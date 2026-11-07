"""Tests for the managed FFmpeg: resolution order, versioning and in-place updates."""

from __future__ import annotations

import os
import sys
import zipfile

import pytest

from videocr import ffmpeg_tools


def _write_fake_binary(path: str, payload: bytes = b"FAKE") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
    return path


class TestResolutionOrder:
    def test_bundled_copy_wins_over_path(self, tmp_path, monkeypatch):
        bundled = _write_fake_binary(
            os.path.join(str(tmp_path), "ffmpeg", ffmpeg_tools.executable_name())
        )
        monkeypatch.setattr(
            ffmpeg_tools.shutil,
            "which",
            lambda _name: os.path.join("C:", "system", "ffmpeg.exe"),
        )
        assert ffmpeg_tools.find_ffmpeg(str(tmp_path)) == bundled
        assert ffmpeg_tools.resolve_ffmpeg(str(tmp_path)) == (bundled, "bundled")

    def test_falls_back_to_path_when_no_bundled_copy(self, tmp_path, monkeypatch):
        system = os.path.join(str(tmp_path), "sys", "ffmpeg.exe")
        monkeypatch.setattr(ffmpeg_tools.shutil, "which", lambda _name: system)
        assert ffmpeg_tools.find_managed(str(tmp_path)) is None
        assert ffmpeg_tools.find_ffmpeg(str(tmp_path)) == system
        assert ffmpeg_tools.resolve_ffmpeg(str(tmp_path)) == (system, "system")

    def test_returns_none_when_neither_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ffmpeg_tools.shutil, "which", lambda _name: None)
        assert ffmpeg_tools.resolve_ffmpeg(str(tmp_path)) == (None, "")

    def test_managed_path_is_under_the_root(self, tmp_path):
        path = ffmpeg_tools.managed_path(str(tmp_path))
        assert os.path.dirname(path) == os.path.join(str(tmp_path), "ffmpeg")
        assert os.path.basename(path) == ffmpeg_tools.executable_name()

    def test_candidate_roots_include_parent_of_cli(self, tmp_path, monkeypatch):
        cli_dir = tmp_path / "CLI"
        cli_dir.mkdir()
        monkeypatch.setattr(sys, "argv", [str(cli_dir / "videocr_cli.py")])
        monkeypatch.chdir(str(tmp_path))
        roots = ffmpeg_tools.candidate_roots()
        assert str(tmp_path) in roots


class TestVersionParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("9.0.2-essentials_build-www.gyan.dev", (9, 0, 2)),
            ("9.0.2", (9, 0, 2)),
            ("v7.1.1", (7, 1, 1)),
            # A git snapshot reports a date, not a release: not comparable.
            ("2026-07-20-git-c23123630e-essentials_build-www.gyan.dev", ()),
            ("nonsense", ()),
            ("", ()),
        ],
    )
    def test_version_key(self, raw, expected):
        assert ffmpeg_tools._version_key(raw) == expected

    @pytest.mark.parametrize(
        ("latest", "current", "expected"),
        [
            ("9.1", "9.0.2", True),
            ("9.0.2", "9.0.1", True),
            ("9.0.2", "9.0.2", False),
            ("9.0.1", "9.0.2", False),
            ("9.0.2", "2026-07-20-git-abc", False),
            ("2026-07-20-git-abc", "9.0.2", False),
            ("", "9.0.2", False),
            ("9.0.2", "", False),
        ],
    )
    def test_is_newer(self, latest, current, expected):
        assert ffmpeg_tools.is_newer(latest, current) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("9.0.2-essentials_build-www.gyan.dev", "9.0.2"),
            ("2026-07-20-git-c23123630e-essentials_build", "2026-07-20 (git)"),
            ("7.1", "7.1"),
            ("", ""),
        ],
    )
    def test_short_version(self, raw, expected):
        assert ffmpeg_tools.short_version(raw) == expected

    def test_ffmpeg_version_reads_the_banner(self, monkeypatch):
        class Result:
            stdout = b"ffmpeg version 9.0.2-essentials_build Copyright (c) 2000-2025\n"
            stderr = b""

        monkeypatch.setattr(
            ffmpeg_tools.subprocess, "run", lambda *a, **k: Result()
        )
        assert ffmpeg_tools.ffmpeg_version("ffmpeg.exe") == "9.0.2-essentials_build"

    def test_ffmpeg_version_is_none_without_a_path(self):
        assert ffmpeg_tools.ffmpeg_version(None) is None
        assert ffmpeg_tools.ffmpeg_version("") is None

    def test_ffmpeg_version_survives_a_broken_binary(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("not executable")

        monkeypatch.setattr(ffmpeg_tools.subprocess, "run", boom)
        assert ffmpeg_tools.ffmpeg_version("missing.exe") is None


class TestInstall:
    def _make_archive(self, path: str, members: dict[str, bytes]) -> str:
        with zipfile.ZipFile(path, "w") as bundle:
            for name, payload in members.items():
                bundle.writestr(name, payload)
        return path

    def test_install_extracts_ffmpeg_and_ffprobe_only(self, tmp_path):
        archive = self._make_archive(
            str(tmp_path / "ff.zip"),
            {
                "ffmpeg-9.9.9-essentials_build/bin/ffmpeg.exe": b"FFMPEG",
                "ffmpeg-9.9.9-essentials_build/bin/ffprobe.exe": b"FFPROBE",
                "ffmpeg-9.9.9-essentials_build/bin/ffplay.exe": b"FFPLAY",
                "ffmpeg-9.9.9-essentials_build/doc/index.html": b"<html>",
            },
        )
        root = str(tmp_path / "app")
        installed = ffmpeg_tools.install_from_zip(archive, root=root)

        assert os.path.basename(installed) == ffmpeg_tools.executable_name()
        assert os.path.dirname(installed) == os.path.join(root, "ffmpeg")
        with open(installed, "rb") as handle:
            assert handle.read() == b"FFMPEG"
        assert os.path.exists(os.path.join(root, "ffmpeg", "ffprobe.exe"))
        # ffplay is never needed and must not be shipped.
        assert not os.path.exists(os.path.join(root, "ffmpeg", "ffplay.exe"))
        assert not os.path.exists(os.path.join(root, "ffmpeg", "doc"))

    def test_install_replaces_an_existing_binary(self, tmp_path):
        root = str(tmp_path / "app")
        old = _write_fake_binary(
            os.path.join(root, "ffmpeg", ffmpeg_tools.executable_name()), b"OLD"
        )
        archive = self._make_archive(
            str(tmp_path / "ff.zip"),
            {"pkg/bin/ffmpeg.exe": b"NEW"},
        )
        ffmpeg_tools.install_from_zip(archive, root=root)
        with open(old, "rb") as handle:
            assert handle.read() == b"NEW"

    def test_install_leaves_no_temp_files_behind(self, tmp_path):
        root = str(tmp_path / "app")
        archive = self._make_archive(
            str(tmp_path / "ff.zip"), {"pkg/bin/ffmpeg.exe": b"NEW"}
        )
        ffmpeg_tools.install_from_zip(archive, root=root)
        leftovers = [n for n in os.listdir(os.path.join(root, "ffmpeg")) if n.endswith(".new")]
        assert leftovers == []

    def test_install_rejects_an_archive_without_ffmpeg(self, tmp_path):
        archive = self._make_archive(
            str(tmp_path / "ff.zip"), {"pkg/bin/readme.txt": b"nothing"}
        )
        with pytest.raises(ValueError, match="does not contain an ffmpeg binary"):
            ffmpeg_tools.install_from_zip(archive, root=str(tmp_path / "app"))

    def test_install_tolerates_a_missing_ffprobe(self, tmp_path):
        archive = self._make_archive(
            str(tmp_path / "ff.zip"), {"pkg/bin/ffmpeg.exe": b"ONLY-FFMPEG"}
        )
        installed = ffmpeg_tools.install_from_zip(archive, root=str(tmp_path / "app"))
        assert os.path.exists(installed)

    def test_install_without_keep_probe_skips_ffprobe(self, tmp_path):
        archive = self._make_archive(
            str(tmp_path / "ff.zip"),
            {"pkg/bin/ffmpeg.exe": b"F", "pkg/bin/ffprobe.exe": b"P"},
        )
        root = str(tmp_path / "app")
        ffmpeg_tools.install_from_zip(archive, root=root, keep_probe=False)
        assert not os.path.exists(os.path.join(root, "ffmpeg", "ffprobe.exe"))


class TestManagedStatus:
    def test_status_reports_the_bundled_copy(self, tmp_path, monkeypatch):
        _write_fake_binary(
            os.path.join(str(tmp_path), "ffmpeg", ffmpeg_tools.executable_name())
        )
        monkeypatch.setattr(ffmpeg_tools, "ffmpeg_version", lambda _p: "9.0.2")
        status = ffmpeg_tools.managed_status(str(tmp_path))
        assert status["source"] == "bundled"
        assert status["version"] == "9.0.2"
        assert status["managed_path"].endswith(ffmpeg_tools.executable_name())

    def test_status_reports_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ffmpeg_tools.shutil, "which", lambda _n: None)
        status = ffmpeg_tools.managed_status(str(tmp_path))
        assert status["source"] == ""
        assert status["path"] == ""

    def test_describe_mentions_the_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ffmpeg_tools, "ffmpeg_version", lambda _p: "9.0.2")
        assert "bundled FFmpeg 9.0.2" in ffmpeg_tools.describe("x", "bundled")
        assert "system FFmpeg 9.0.2" in ffmpeg_tools.describe("x", "system")

    def test_describe_handles_no_binary(self):
        assert "no FFmpeg" in ffmpeg_tools.describe(None, "")


class TestNetworkBoundary:
    def test_latest_version_is_none_when_offline(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("no network")

        monkeypatch.setattr(ffmpeg_tools.urllib.request, "urlopen", boom)
        assert ffmpeg_tools.latest_release_version() is None

    def test_latest_version_strips_surrounding_whitespace(self, monkeypatch):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self):
                return b"  9.0.2\r\n"

        monkeypatch.setattr(
            ffmpeg_tools.urllib.request, "urlopen", lambda *a, **k: Response()
        )
        assert ffmpeg_tools.latest_release_version() == "9.0.2"

    def test_download_reports_progress(self, tmp_path, monkeypatch):
        payload = b"x" * 1000

        class Response:
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def read(self, size: int = -1):
                nonlocal payload
                chunk, payload = payload, b""
                return chunk

        monkeypatch.setattr(
            ffmpeg_tools.urllib.request, "urlopen", lambda *a, **k: Response()
        )
        seen: list[tuple[int, int]] = []
        destination = str(tmp_path / "ff.zip")
        ffmpeg_tools.download_zip(
            destination, progress=lambda done, total: seen.append((done, total))
        )
        assert os.path.getsize(destination) == 1000
        assert seen == [(1000, 1000)]
