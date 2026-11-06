"""Parity tests: every CLI flag must be reachable from the GUI.

The GUI spawns ``videocr-cli`` as a subprocess, so a CLI flag with no GUI
control is invisible to GUI users, and a GUI setting that never reaches the
command line silently does nothing. These tests lock the mapping down.
"""

from __future__ import annotations

import argparse
import sys

import pytest
from PySide6.QtWidgets import QApplication

from videocr_gui import args as gui_args
from videocr_gui import config as gui_config
from videocr_gui import i18n
from videocr_gui.settings_tab import SettingsTab


@pytest.fixture(scope="module")
def cli_parser() -> argparse.ArgumentParser:
    """The parser ``videocr_cli.main()`` builds.

    ``main()`` creates its parser inline and immediately parses ``sys.argv``,
    so capture the instance instead of duplicating the flag list here.
    """
    import videocr_cli

    captured: list[argparse.ArgumentParser] = []
    real_init = argparse.ArgumentParser.__init__

    def _capture(self, *a, **k):
        real_init(self, *a, **k)
        captured.append(self)

    argv = sys.argv
    argparse.ArgumentParser.__init__ = _capture
    try:
        sys.argv = ["videocr-cli", "-h"]
        with pytest.raises(SystemExit):
            videocr_cli.main()
    finally:
        argparse.ArgumentParser.__init__ = real_init
        sys.argv = argv
    assert captured, "videocr_cli.main() did not build an ArgumentParser"
    return captured[0]


@pytest.fixture(scope="module")
def gui_surface() -> tuple[set[str], set[str]]:
    """(settings-tab widget keys, keys build_args can forward)."""
    app = QApplication.instance() or QApplication([])
    del app
    i18n.load_language("en")
    defaults = gui_config.get_default_settings()
    tab = SettingsTab()
    tab.populate(defaults, ["English"])
    widgets = {k.lstrip("-") for k in tab._widgets}

    # Forward everything: booleans on, text fields given plausible values.
    maxed = dict(defaults)
    for key in list(maxed):
        if isinstance(maxed[key], bool):
            maxed[key] = True
    maxed["ocr_engine"] = "ONNX Runtime DirectML (AMD GPU Experimental)"
    maxed["enable_subtitle_alignment"] = True
    maxed["--use_dual_zone"] = True
    # Empty text fields are skipped by build_args, so give the ones that have
    # no default a value to prove they are forwarded at all.
    maxed["--label_lang"] = "japan"
    maxed["--time_start"] = "0:05"
    maxed["--time_end"] = "1:00"
    maxed["--label_time_start"] = "0:05"
    maxed["--label_time_end"] = "0:50"
    maxed["--brightness_threshold"] = "20"
    box = {"coords": {"crop_x": 0, "crop_y": 0, "crop_width": 10, "crop_height": 10}}
    forwarded, errors = gui_args.build_args(
        "v.mp4", maxed, [box, box], output_path="o.ass"
    )
    assert errors == []
    assert forwarded is not None
    return widgets, set(forwarded)


# Options with no settings-tab widget because they are owned elsewhere, or are
# structural rather than user-facing.
_NOT_A_WIDGET = {
    # rendered on the Process Video tab
    "ocr_engine",
    "lang",
    "subtitle_position",
    "output",
    "directml_device_index",
    "video_path",
    # driven by the crop canvas
    "crop_x",
    "crop_y",
    "crop_width",
    "crop_height",
    "crop_x2",
    "crop_y2",
    "crop_width2",
    "crop_height2",
    # deliberately CLI-only (see videocr_gui/DESIGN.md "Decisions")
    "benchmark_compare_engine",
    "benchmark_compare_sample_grids",
    "allow_system_sleep",  # GUI owns the wake lock and always forwards True
}


def test_every_cli_option_has_a_gui_control(cli_parser, gui_surface):
    """No CLI flag may be invisible to GUI users."""
    widgets, forwarded = gui_surface
    reachable = widgets | forwarded | _NOT_A_WIDGET
    orphans = sorted(
        opt.lstrip("-")
        for action in cli_parser._actions
        for opt in action.option_strings
        if opt.startswith("--") and opt != "--help" and opt.lstrip("-") not in reachable
    )
    assert orphans == [], f"CLI options with no GUI control: {orphans}"


def test_every_forwarded_key_is_a_real_cli_option(cli_parser, gui_surface):
    """The GUI must not pass flags the CLI does not define."""
    _, forwarded = gui_surface
    known = {
        opt.lstrip("-")
        for action in cli_parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }
    # GUI-only bookkeeping, dropped by workers.build_command().
    gui_only = {"send_notification"}
    unknown = sorted(forwarded - known - gui_only)
    assert unknown == [], f"GUI forwards flags the CLI does not define: {unknown}"


def test_label_lang_has_a_widget_and_reaches_the_cli(gui_surface):
    """--label_lang used to be config-file-only; keep the widget wired up."""
    widgets, forwarded = gui_surface
    # gui_surface normalizes widget keys by stripping the leading dashes.
    assert "label_lang" in widgets
    assert "label_lang" in forwarded


def test_label_lang_greys_out_without_label_detection():
    from videocr_gui.settings_tab import LABEL_DEPENDENT_WIDGETS

    assert "--label_lang" in LABEL_DEPENDENT_WIDGETS
