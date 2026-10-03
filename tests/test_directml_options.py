"""Tests for videocr_gui.directml — option/index conversion and adapter listing.

The adapter detection helpers shell out to powershell/wmic/torch, so those are
monkeypatched; only the pure conversion functions and the option-list assembly
logic are exercised.
"""

from __future__ import annotations

import pytest

from videocr_gui import constants as C
from videocr_gui import directml


class TestOptionToIndex:
    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("GPU 0: AMD Radeon", "0"),
            ("GPU 1: NVIDIA", "1"),
            ("GPU 10: Integrated", "10"),
            ("GPU  1:  Extra spaces", "1"),
            ("", ""),
            (None, ""),
            (C.DIRECTML_AUTO_OPTION, ""),
            ("Random text", ""),
            ("3", "3"),
            ("  7  ", "7"),
        ],
    )
    def test_conversion(self, option, expected):
        assert directml.option_to_index(option) == expected


class TestIndexToOption:
    OPTIONS = [
        C.DIRECTML_AUTO_OPTION,
        "GPU 0: AMD Radeon RX 7800 XT",
        "GPU 1: Intel UHD Graphics",
    ]

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            ("0", "GPU 0: AMD Radeon RX 7800 XT"),
            ("1", "GPU 1: Intel UHD Graphics"),
            ("", C.DIRECTML_AUTO_OPTION),
            (None, C.DIRECTML_AUTO_OPTION),
            ("2", "GPU 2: Unknown DirectML adapter"),
        ],
    )
    def test_lookup(self, index, expected):
        assert directml.index_to_option(self.OPTIONS, index) == expected

    def test_string_zero_is_not_treated_as_empty(self):
        # The config stores the index as a string; "0" must resolve to GPU 0.
        assert directml.index_to_option(self.OPTIONS, "0") == "GPU 0: AMD Radeon RX 7800 XT"

    def test_round_trip(self):
        for option in self.OPTIONS[1:]:
            idx = directml.option_to_index(option)
            assert directml.index_to_option(self.OPTIONS, idx) == option


class TestGetAdapterOptions:
    def test_uses_torch_count_and_names(self, monkeypatch):
        monkeypatch.setattr(directml, "_powershell_video_controller_names", lambda: [])
        monkeypatch.setattr(directml, "_torch_directml_adapter_count", lambda: 2)
        monkeypatch.setattr(
            directml, "_torch_directml_adapter_name",
            lambda i: f"AMD Radeon {i}" if i == 0 else None,
        )
        options = directml.get_adapter_options()
        assert options[0] == C.DIRECTML_AUTO_OPTION
        assert options[1] == "GPU 0: AMD Radeon 0"
        assert options[2] == "GPU 1: Unknown DirectML adapter"  # no name anywhere

    def test_falls_back_to_powershell_names(self, monkeypatch):
        monkeypatch.setattr(
            directml, "_powershell_video_controller_names",
            lambda: ["NVIDIA GeForce RTX 4060"],
        )
        monkeypatch.setattr(directml, "_torch_directml_adapter_count", lambda: None)
        monkeypatch.setattr(directml, "_torch_directml_adapter_name", lambda i: None)
        options = directml.get_adapter_options()
        assert options == [C.DIRECTML_AUTO_OPTION, "GPU 0: NVIDIA GeForce RTX 4060"]

    def test_powershell_names_used_when_torch_count_zero(self, monkeypatch):
        monkeypatch.setattr(
            directml, "_powershell_video_controller_names", lambda: ["A", "B", "C"]
        )
        monkeypatch.setattr(directml, "_torch_directml_adapter_count", lambda: 0)
        monkeypatch.setattr(directml, "_torch_directml_adapter_name", lambda i: None)
        assert len(directml.get_adapter_options()) == 4  # Auto + 3 GPUs

    def test_no_detection_on_non_windows(self, monkeypatch):
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "linux")
        monkeypatch.setattr(directml, "_powershell_video_controller_names", lambda: [])
        monkeypatch.setattr(directml, "_torch_directml_adapter_count", lambda: None)
        options = directml.get_adapter_options()
        assert options == [C.DIRECTML_AUTO_OPTION]

    def test_windows_without_detection_falls_back_to_two(self, monkeypatch):
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")
        monkeypatch.setattr(directml, "_powershell_video_controller_names", lambda: [])
        monkeypatch.setattr(directml, "_torch_directml_adapter_count", lambda: None)
        monkeypatch.setattr(directml, "_torch_directml_adapter_name", lambda i: None)
        options = directml.get_adapter_options()
        assert len(options) == 3  # Auto + 2 unknown GPUs
        assert options[1] == "GPU 0: Unknown DirectML adapter"


def test_gpu_option_pattern():
    match = directml.DIRECTML_GPU_OPTION_PATTERN.match("GPU 3: AMD Radeon RX 7900 XTX")
    assert match is not None
    assert match.group(1) == "3"
    assert match.group(2) == "AMD Radeon RX 7900 XTX"
    assert directml.DIRECTML_GPU_OPTION_PATTERN.match("not a gpu option") is None
