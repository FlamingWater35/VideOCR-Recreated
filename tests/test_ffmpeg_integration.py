"""Tests for the FFmpeg fallback chain and the Advanced Settings row."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PY = REPO_ROOT / "CLI" / "videocr" / "video.py"


def _source() -> str:
    return VIDEO_PY.read_text(encoding="utf-8")


class TestFallbackChain:
    """The frame-scan mode must degrade, never abort, and say why."""

    def test_ffmpeg_call_site_is_guarded(self):
        """A runtime FFmpeg failure must not escape the call site.

        Before this guard, a non-zero FFmpeg exit raised RuntimeError straight
        into api.save_subtitles_to_file, which called sys.exit(1): the whole run
        aborted instead of using a slower path.
        """
        source = _source()
        tree = ast.parse(source)
        guarded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            and any(
                isinstance(inner, ast.Call)
                and "_run_easyocr_directml_ffmpeg_hw_scan" in ast.unparse(inner)
                for inner in ast.walk(node)
            )
        ]
        assert guarded, "the ffmpeg scan call is not wrapped in try/except"
        assert any(node.handlers for node in guarded), "no except handler"

    def test_fallback_order_names_directml_before_cpu(self):
        """The message must describe ffmpeg -> DirectML SSIM -> CPU SSIM."""
        source = _source()
        assert "falling back to " in source
        assert "AMD DirectML SSIM" in source
        assert "CPU SSIM" in source
        fallback_block = source.split("FFmpeg D3D11VA frame scan failed", 1)[1][:400]
        assert "AMD DirectML SSIM" in fallback_block
        assert "CPU SSIM" in fallback_block

    def test_fallback_switches_to_directml_ssim_mode(self):
        source = _source()
        marker = source.split("FFmpeg D3D11VA frame scan failed", 1)[1]
        # The assignment follows the explanatory comment and the timestamp reset.
        assert 'directml_frame_scan_mode = "directml_ssim"' in marker[:2000]

    def test_fallback_clears_stale_frame_timestamps(self):
        """Timestamps from the aborted scan would misalign every subtitle."""
        source = _source()
        fallback_block = source.split("FFmpeg D3D11VA frame scan failed", 1)[1][:900]
        assert "self.frame_timestamps.clear()" in fallback_block

    def test_preflight_declines_also_continue_on_directml(self):
        """A declined scan (no binary, bad fps, ...) also degrades gracefully."""
        source = _source()
        after_call = source.split("if used_ffmpeg_hw_path:", 1)[1][:700]
        assert 'directml_frame_scan_mode = "directml_ssim"' in after_call

    def test_every_ffmpeg_decline_logs_a_reason(self):
        """No silent fallback: each ``return False`` path prints first."""
        source = _source()
        body = source.split("def _run_easyocr_directml_ffmpeg_hw_scan", 1)[1]
        body = body.split("\n    def ", 1)[0]
        for line in body.splitlines():
            if line.strip() == "return False":
                preceding = body.split(line, 1)[0]
                tail = preceding[-400:]
                assert "print(" in tail, (
                    "a return False path declines the scan without logging why"
                )

    def test_uses_ffmpeg_tools_not_bare_which(self):
        source = _source()
        assert "ffmpeg_tools.find_ffmpeg()" in source
        assert 'shutil.which("ffmpeg")' not in source


class TestManagedPreference:
    def test_resolution_prefers_the_managed_copy(self, tmp_path, monkeypatch):
        from videocr import ffmpeg_tools

        bundled = tmp_path / "ffmpeg" / ffmpeg_tools.executable_name()
        bundled.parent.mkdir(parents=True)
        bundled.write_bytes(b"x")
        monkeypatch.setattr(
            ffmpeg_tools.shutil, "which", lambda _n: "/usr/bin/ffmpeg"
        )
        path, source = ffmpeg_tools.resolve_ffmpeg(str(tmp_path))
        assert source == "bundled"
        assert path == str(bundled)


class TestFFmpegRow:
    @pytest.fixture
    def row(self, qtbot):
        from videocr_gui.ffmpeg_settings import FFmpegRow

        widget = FFmpegRow()
        qtbot.addWidget(widget)
        return widget

    def test_reports_a_status_sentence(self, row):
        assert row.status_label.text()

    def test_missing_ffmpeg_is_described_as_such(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "",
                "source": "",
                "version": "",
                "managed_path": "/nowhere/ffmpeg",
            },
        )
        row.refresh()
        assert "No FFmpeg" in row.status_label.text()

    def test_bundled_copy_is_named_bundled(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "/app/ffmpeg/ffmpeg.exe",
                "source": "bundled",
                "version": "9.0.2-essentials_build",
                "managed_path": "/app/ffmpeg/ffmpeg.exe",
            },
        )
        row.refresh()
        assert "9.0.2" in row.status_label.text()

    def test_check_offers_the_newer_version(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(tools, "latest_release_version", lambda: "9.9.9")
        monkeypatch.setattr(tools, "is_newer", lambda latest, cur: True)
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "",
                "source": "",
                "version": "",
                "managed_path": "",
            },
        )
        row._on_check_finished("9.9.9")
        assert "9.9.9" in row.status_label.text()
        assert "available" in row.status_label.text().lower()

    def test_check_reports_up_to_date(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(tools, "is_newer", lambda latest, cur: False)
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "/x/ffmpeg",
                "source": "bundled",
                "version": "9.0.2",
                "managed_path": "/x/ffmpeg",
            },
        )
        row._on_check_finished("9.0.2")
        assert "up to date" in row.status_label.text().lower()

    def test_offline_check_says_nothing_changed(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "",
                "source": "",
                "version": "",
                "managed_path": "",
            },
        )
        row._on_check_finished("")
        assert "unchanged" in row.status_label.text().lower()

    def test_install_failure_is_reported_not_raised(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "",
                "source": "",
                "version": "",
                "managed_path": "",
            },
        )
        shown: list[tuple[str, str]] = []
        monkeypatch.setattr(
            ffmpeg_settings.dialogs,
            "info",
            lambda _parent, title, message: shown.append((title, message)),
        )
        row._on_install_finished("disk full")
        assert "disk full" in row.status_label.text()
        assert shown and "disk full" in shown[0][1]

    def test_install_success_mentions_no_restart(self, row, monkeypatch):
        from videocr_gui import ffmpeg_settings

        tools = ffmpeg_settings._tools()
        monkeypatch.setattr(
            tools,
            "managed_status",
            lambda _root: {
                "path": "/app/ffmpeg/ffmpeg.exe",
                "source": "bundled",
                "version": "9.9.9",
                "managed_path": "/app/ffmpeg/ffmpeg.exe",
            },
        )
        shown: list[tuple[str, str]] = []
        monkeypatch.setattr(
            ffmpeg_settings.dialogs,
            "info",
            lambda _parent, title, message: shown.append((title, message)),
        )
        row._on_install_finished("")
        assert shown
        assert "restart" in shown[0][1].lower()

    def test_worker_never_touches_widgets_directly(self):
        """Workers must emit signals; touching Qt widgets off-thread crashes."""
        source = (
            REPO_ROOT / "videocr_gui" / "ffmpeg_settings.py"
        ).read_text(encoding="utf-8")
        # Nested functions are indented 8 spaces; their bodies end at the next
        # line indented 8 or less.
        for name in ("def worker", "def on_progress"):
            start = source.index(name)
            lines = source[start:].splitlines()[1:]
            body: list[str] = []
            for line in lines:
                if line.strip() and not line.startswith(" " * 12):
                    break
                body.append(line)
            text = "\n".join(body)
            assert "self.status_label" not in text
            assert "self.progress.setValue" not in text
            assert "self.check_btn" not in text
            assert "_finished.emit" in text or "_progress.emit" in text


class TestTranslations:
    @pytest.mark.parametrize("code", sorted(p.stem for p in (REPO_ROOT / "languages").glob("*.json")))
    def test_every_language_has_the_ffmpeg_strings(self, code):
        import json

        data = json.loads(
            (REPO_ROOT / "languages" / f"{code}.json").read_text(encoding="utf-8")
        )
        for key in (
            "lbl_ffmpeg",
            "tip_ffmpeg",
            "btn_ffmpeg_check",
            "btn_ffmpeg_update",
            "ffmpeg_status_bundled",
            "ffmpeg_status_system",
            "ffmpeg_status_missing",
            "ffmpeg_update_available",
            "ffmpeg_install_done",
        ):
            assert key in data, f"{code}.json is missing {key}"
            assert data[key].strip(), f"{code}.json has an empty {key}"

    def test_placeholders_survive_in_every_language(self):
        import json

        for path in (REPO_ROOT / "languages").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "{version}" in data["ffmpeg_status_bundled"], path.name
            assert "{version}" in data["ffmpeg_status_system"], path.name
            assert "{version}" in data["ffmpeg_update_available"], path.name
            assert "{current}" in data["ffmpeg_update_available"], path.name
            assert "{version}" in data["ffmpeg_install_done"], path.name


class TestBuildBundling:
    def test_directml_build_bundles_ffmpeg(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "def fetch_ffmpeg(" in source
        assert "fetch_ffmpeg(Path(temp_cli_dist), build_target)" in source

    def test_only_the_directml_variant_ships_ffmpeg(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        body = source.split("def fetch_ffmpeg(", 1)[1].split("\ndef ", 1)[0]
        assert 'if variant != "gpu-directml":' in body

    def test_bundled_archive_has_a_d3d11va_capable_source(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "FFMPEG_BUILD_URL" in source
        assert "gyan.dev" in source

    def test_download_never_calls_sys_exit(self):
        """``download_file`` calls sys.exit(1), and SystemExit is a BaseException.

        Reusing it here would abort the whole build on a network blip, instead
        of shipping a PyAV-only package.
        """
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        body = source.split("def fetch_ffmpeg(", 1)[1].split("\ndef ", 1)[0]
        # Strip comments so the explanatory note above the download is not
        # mistaken for a call.
        code = "\n".join(
            line for line in body.splitlines() if not line.strip().startswith("#")
        )
        assert "download_file(" not in code
        assert "sys.exit" not in code

    def test_archive_path_matches_the_downloaded_filename(self, tmp_path, monkeypatch):
        """The helper must read the archive under the name the URL produces."""
        import build

        payload = _ffmpeg_zip()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=8192):
                yield payload

        monkeypatch.setattr(build.requests, "get", lambda *a, **k: Response())
        build.fetch_ffmpeg(tmp_path, "gpu-directml")

        assert (tmp_path / "ffmpeg" / "ffmpeg.exe").exists()
        assert (tmp_path / "ffmpeg" / "ffprobe.exe").exists()
        # ffplay is never needed.
        assert not (tmp_path / "ffmpeg" / "ffplay.exe").exists()
        # The archive must not linger in the package.
        assert list(tmp_path.glob("*.zip")) == []

    def test_download_failure_degrades_instead_of_aborting(self, tmp_path, monkeypatch):
        import build

        def boom(*_a, **_k):
            raise RuntimeError("network down")

        monkeypatch.setattr(build.requests, "get", boom)
        build.fetch_ffmpeg(tmp_path, "gpu-directml")  # must not raise
        assert not (tmp_path / "ffmpeg" / "ffmpeg.exe").exists()

    def test_corrupt_archive_degrades_instead_of_aborting(self, tmp_path, monkeypatch):
        import build

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=8192):
                yield b"not a zip"

        monkeypatch.setattr(build.requests, "get", lambda *a, **k: Response())
        build.fetch_ffmpeg(tmp_path, "gpu-directml")  # must not raise
        assert not (tmp_path / "ffmpeg" / "ffmpeg.exe").exists()

    @pytest.mark.parametrize("variant", ["cpu", "gpu-cuda11.8", "gpu-cuda12.9"])
    def test_non_directml_variants_download_nothing(self, tmp_path, monkeypatch, variant):
        import build

        calls: list[str] = []
        monkeypatch.setattr(
            build.requests, "get", lambda url, **k: calls.append(url) or None
        )
        build.fetch_ffmpeg(tmp_path, variant)
        assert calls == []
        assert not (tmp_path / "ffmpeg").exists()


def _ffmpeg_zip() -> bytes:
    """A minimal archive shaped like the real gyan.dev build."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("ffmpeg-9.9.9-essentials_build/bin/ffmpeg.exe", b"FFMPEG")
        bundle.writestr("ffmpeg-9.9.9-essentials_build/bin/ffprobe.exe", b"FFPROBE")
        bundle.writestr("ffmpeg-9.9.9-essentials_build/bin/ffplay.exe", b"FFPLAY")
        bundle.writestr("ffmpeg-9.9.9-essentials_build/doc/index.html", b"<html>")
    return buffer.getvalue()


class TestBuildVariantGating:
    """The row belongs only to packages that can use the D3D11VA scan."""

    @pytest.fixture
    def tab_for(self, monkeypatch):
        from videocr_gui import config as gui_config
        from videocr_gui.settings_tab import SettingsTab

        def build(variant: str) -> SettingsTab:
            monkeypatch.setattr(
                gui_config, "get_build_variant", lambda v=variant: v
            )
            tab = SettingsTab()
            tab.populate(gui_config.get_default_settings(), ["English"])
            tab._update_dependent_states()
            return tab

        return build

    @pytest.mark.parametrize("variant", ["cpu", "gpu-cuda11.8", "gpu-cuda12.9"])
    def test_hidden_in_non_directml_builds(self, tab_for, variant):
        tab = tab_for(variant)
        assert tab.ffmpeg_row.isHidden()
        assert tab.ffmpeg_label.isHidden()

    @pytest.mark.parametrize("variant", ["gpu-directml", "unknown"])
    def test_visible_in_directml_and_from_source(self, tab_for, variant):
        """``unknown`` is the from-source/debug case: everything stays available."""
        tab = tab_for(variant)
        assert not tab.ffmpeg_row.isHidden()
        assert not tab.ffmpeg_label.isHidden()

    def test_gating_follows_supports_directml(self):
        from videocr_gui import config as gui_config
        from videocr_gui.settings_tab import SettingsTab

        source = (
            REPO_ROOT / "videocr_gui" / "settings_tab.py"
        ).read_text(encoding="utf-8")
        block = source.split("ffmpeg_ok = ", 1)[1][:40]
        assert "dml_ok" in block
        # And dml_ok is what decides the backend gate.
        assert "dml_ok = config.supports_directml()" in source
        assert gui_config.supports_directml() is True  # from source == unknown
        assert hasattr(SettingsTab, "_update_dependent_states")
