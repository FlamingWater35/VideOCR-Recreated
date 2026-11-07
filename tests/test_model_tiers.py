"""Tests for the three-way PP-OCRv6 model tier setting.

One user-facing setting (``--model_tier``) has to reach every OCR engine: the
PaddleOCR engines resolve it to a local support-files directory, the ONNX engine
resolves it to a rapidocr ``ModelType``. These tests pin both mappings plus the
GUI plumbing, so a new engine cannot quietly ignore the setting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from videocr import utils
from videocr.lang_dictionaries import PADDLEOCR_LANGS

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = sorted((REPO_ROOT / "languages").glob("*.json"))
TIER_KEYS = ("lbl_model_tier", "tip_model_tier", "tier_tiny", "tier_small", "tier_medium")


class TestTierConstants:
    def test_tiers_are_ordered_smallest_first(self):
        assert utils.MODEL_TIERS == ("tiny", "small", "medium")

    def test_default_is_small(self):
        assert utils.DEFAULT_MODEL_TIER == "small"

    def test_onnx_tiers_match_paddle_tiers(self):
        # The two engines must offer the same three names, or the dropdown would
        # describe something one backend cannot honor.
        from videocr import onnx_directml_ocr

        assert onnx_directml_ocr.PP_OCRV6_TIERS == utils.MODEL_TIERS
        assert onnx_directml_ocr.DEFAULT_PP_OCRV6_TIER == utils.DEFAULT_MODEL_TIER


class TestPaddleTierResolution:
    @pytest.mark.parametrize("tier", utils.MODEL_TIERS)
    def test_det_and_rec_follow_the_tier(self, tier):
        det, rec, _ = utils.resolve_model_dirs("en", model_tier=tier)
        assert det.endswith(f"PP-OCRv6_{tier}_det")
        assert rec.endswith(f"PP-OCRv6_{tier}_rec")

    @pytest.mark.parametrize("tier", utils.MODEL_TIERS)
    def test_tier_applies_to_language_specific_rec(self, tier):
        # Korean has no v6 recognition model; the tier must still change det.
        det, rec, _ = utils.resolve_model_dirs("korean", model_tier=tier)
        assert det.endswith(f"PP-OCRv6_{tier}_det")
        assert rec.endswith("korean_PP-OCRv5_mobile_rec")

    def test_georgian_ignores_the_tier(self):
        # Georgian has no v6 model at all, so both halves stay on v3.
        det, rec, _ = utils.resolve_model_dirs("ka", model_tier="tiny")
        assert det.endswith("PP-OCRv3_mobile_det")
        assert rec.endswith("ka_PP-OCRv3_mobile_rec")

    def test_positional_call_still_works(self):
        # resolve_model_dirs used to take a bool here; a bare string must work.
        det, _, _ = utils.resolve_model_dirs("en", "medium")
        assert det.endswith("PP-OCRv6_medium_det")

    def test_empty_tier_falls_back(self):
        det, rec, _ = utils.resolve_model_dirs("en", model_tier="")
        assert det.endswith("PP-OCRv6_small_det")
        assert rec.endswith("PP-OCRv6_small_rec")


class TestOnnxTierResolution:
    def test_tier_maps_to_model_type_enum(self):
        pytest.importorskip("rapidocr")
        from rapidocr.utils.typings import ModelType

        from videocr import onnx_directml_ocr

        for tier in utils.MODEL_TIERS:
            assert onnx_directml_ocr._tier_model_type(tier) is ModelType(tier)

    def test_unknown_tier_maps_to_default(self):
        pytest.importorskip("rapidocr")
        from rapidocr.utils.typings import ModelType

        from videocr import onnx_directml_ocr

        assert (
            onnx_directml_ocr._tier_model_type("gigantic") is ModelType.SMALL
        )

    def test_engine_params_use_model_type_not_model_name(self):
        """``model_name`` is not a rapidocr key and is silently ignored.

        Passing it made the engine always load the bundled small tier no matter
        what was requested, so the tier is wired through ``model_type``.
        """
        source = (
            REPO_ROOT / "CLI" / "videocr" / "onnx_directml_ocr.py"
        ).read_text(encoding="utf-8")
        assert '"Det.model_type"' in source
        assert '"Rec.model_type"' in source
        # The inert keys must not be passed to RapidOCR again.
        assert '"Det.model_name"' not in source
        assert '"Rec.model_name"' not in source


class TestCliFlag:
    def test_default_is_small(self):
        import videocr_cli

        captured: list = []
        real_init = __import__("argparse").ArgumentParser.__init__

        def _capture(self, *a, **k):
            real_init(self, *a, **k)
            captured.append(self)

        import argparse

        argv = sys.argv
        argparse.ArgumentParser.__init__ = _capture
        sys.argv = ["videocr-cli", "--help"]
        try:
            with pytest.raises(SystemExit):
                videocr_cli.main()
        finally:
            argparse.ArgumentParser.__init__ = real_init
            sys.argv = argv

        parser = captured[0]
        action = next(a for a in parser._actions if "--model_tier" in a.option_strings)
        assert action.default == "small"
        assert action.choices is None  # validated by type=, not choices=

    @pytest.mark.parametrize("tier", ["tiny", "small", "medium"])
    def test_validator_accepts_each_tier(self, tier):
        import videocr_cli

        assert videocr_cli.valid_model_tier(tier) == tier

    @pytest.mark.parametrize("bad", ["server", "large", "", "SMALL", "medium "])
    def test_validator_rejects_unknown_tier(self, bad):
        import argparse

        import videocr_cli

        with pytest.raises(argparse.ArgumentTypeError):
            videocr_cli.valid_model_tier(bad)

    def test_use_server_model_flag_is_gone(self, monkeypatch):
        """The boolean flag was replaced; the old spelling must not linger."""
        import argparse

        import videocr_cli

        captured: list = []
        real_init = argparse.ArgumentParser.__init__

        def _capture(self, *a, **k):
            real_init(self, *a, **k)
            captured.append(self)

        argv = sys.argv
        argparse.ArgumentParser.__init__ = _capture
        sys.argv = ["videocr-cli", "--help"]
        try:
            with pytest.raises(SystemExit):
                videocr_cli.main()
        finally:
            argparse.ArgumentParser.__init__ = real_init
            sys.argv = argv

        options = {
            opt for a in captured[0]._actions for opt in a.option_strings
        }
        assert "--model_tier" in options
        assert "--use_server_model" not in options


class TestGuiPlumbing:
    @pytest.fixture(autouse=True)
    def _qt(self):
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        yield app
        del app

    def test_default_setting_is_small(self):
        from videocr_gui import config

        assert config.get_default_settings()["--model_tier"] == "small"

    def test_not_treated_as_bool(self):
        from videocr_gui import config

        assert "--model_tier" not in config._BOOL_KEYS

    def test_widget_is_a_combo_with_three_internal_values(self):
        from videocr_gui import config, i18n
        from videocr_gui.settings_tab import SettingsTab

        i18n.load_language("en")
        tab = SettingsTab()
        tab.populate(config.get_default_settings(), ["English"])
        combo = tab._widgets["--model_tier"]
        assert combo.count() == 3
        assert tab._combo_internal["--model_tier"] == ["tiny", "small", "medium"]

    def test_round_trip_through_read_settings(self):
        from videocr_gui import config, i18n
        from videocr_gui.settings_tab import SettingsTab

        i18n.load_language("en")
        defaults = config.get_default_settings()
        defaults["--model_tier"] = "medium"
        tab = SettingsTab()
        tab.populate(defaults, ["English"])
        assert tab.read_settings()["--model_tier"] == "medium"

    def test_reaches_the_cli(self):
        from videocr_gui import config, i18n
        from videocr_gui.args import build_args

        i18n.load_language("en")
        settings = config.get_default_settings()
        settings["--model_tier"] = "tiny"
        box = {
            "coords": {
                "crop_x": 0,
                "crop_y": 0,
                "crop_width": 10,
                "crop_height": 10,
            }
        }
        args, errors = build_args("v.mp4", settings, [box], output_path="o.ass")
        assert errors == []
        assert args["model_tier"] == "tiny"

    def test_unknown_stored_tier_falls_back(self):
        from videocr_gui import config, i18n
        from videocr_gui.args import build_args

        i18n.load_language("en")
        settings = config.get_default_settings()
        settings["--model_tier"] = "gigantic"
        box = {
            "coords": {
                "crop_x": 0,
                "crop_y": 0,
                "crop_width": 10,
                "crop_height": 10,
            }
        }
        args, _ = build_args("v.mp4", settings, [box], output_path="o.ass")
        assert args["model_tier"] == "small"


class TestLegacyConfigMigration:
    def test_use_server_model_true_becomes_medium(self, tmp_path, monkeypatch):
        from videocr_gui import config

        cfg = tmp_path / "videocr_gui_config.ini"
        cfg.write_text(
            "[Settings]\n--use_server_model = True\n", encoding="utf-8"
        )
        monkeypatch.setattr(config, "CONFIG_FILE", str(cfg))
        assert config.load_settings()["--model_tier"] == "medium"

    def test_use_server_model_false_stays_default(self, tmp_path, monkeypatch):
        from videocr_gui import config

        cfg = tmp_path / "videocr_gui_config.ini"
        cfg.write_text(
            "[Settings]\n--use_server_model = False\n", encoding="utf-8"
        )
        monkeypatch.setattr(config, "CONFIG_FILE", str(cfg))
        assert config.load_settings()["--model_tier"] == "small"

    def test_explicit_model_tier_wins(self, tmp_path, monkeypatch):
        from videocr_gui import config

        cfg = tmp_path / "videocr_gui_config.ini"
        cfg.write_text(
            "[Settings]\n--use_server_model = True\n--model_tier = tiny\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "CONFIG_FILE", str(cfg))
        assert config.load_settings()["--model_tier"] == "tiny"


class TestTranslations:
    @pytest.mark.parametrize("path", LANGUAGES, ids=lambda p: p.stem)
    def test_has_every_tier_key(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in TIER_KEYS if key not in data]
        assert missing == [], f"{path.name} missing {missing}"

    @pytest.mark.parametrize("path", LANGUAGES, ids=lambda p: p.stem)
    def test_old_server_model_keys_are_gone(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "chk_server_model" not in data
        assert "tip_server_model" not in data

    def test_english_option_labels_are_non_empty(self):
        data = json.loads(
            (REPO_ROOT / "languages" / "en.json").read_text(encoding="utf-8")
        )
        for key in ("tier_tiny", "tier_small", "tier_medium"):
            assert data[key].strip()

    @pytest.mark.parametrize("path", LANGUAGES, ids=lambda p: p.stem)
    def test_tooltip_does_not_overclaim(self, path):
        """EasyOCR has no tiers, so "every engine" would be a lie."""
        data = json.loads(path.read_text(encoding="utf-8"))
        tip = data["tip_model_tier"].lower()
        assert "every ocr engine" not in tip
        assert "every engine" not in tip

    def test_english_tooltip_names_easyocr_as_the_exception(self):
        data = json.loads(
            (REPO_ROOT / "languages" / "en.json").read_text(encoding="utf-8")
        )
        tip = data["tip_model_tier"]
        assert "EasyOCR" in tip
        assert "PP-OCRv6" in tip


class TestBuildBundling:
    def test_build_fetches_tiny_models(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "def fetch_tiny_models" in source
        assert "PP-OCRv6_tiny_det" in source
        assert "PP-OCRv6_tiny_rec" in source

    def test_build_fetches_all_onnx_tiers(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "def fetch_onnx_models" in source
        assert "PP-OCRv6_{task}_{tier}.onnx" in source

    def test_build_prunes_unreachable_v5_det_models(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "def prune_unused_det_models" in source
        for name in ("PP-OCRv5_mobile_det", "PP-OCRv5_server_det"):
            assert name in source
        # And the resolver really cannot reach them.
        for lang in ("en", "japan", "korean", "ar", "ka"):
            for tier in utils.MODEL_TIERS:
                det, _, _ = utils.resolve_model_dirs(lang, tier)
                assert "PP-OCRv5" not in os.path.basename(det)

    def test_pruning_runs_for_every_target(self):
        """The prune call must not sit inside a single-variant branch."""
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        body = source.split("def package_target(", 1)[1]
        assert "prune_unused_det_models(support_root)" in body
        # It runs right after the tiny-model fetch, unconditionally on variant.
        tail = body.split("prune_unused_det_models(support_root)", 1)[0]
        assert "fetch_tiny_models(support_root)" in tail

    def test_unreachable_models_are_really_unreachable(self):
        """Exhaustive check over every language the resolver knows about."""
        reachable = set()
        languages = set()
        for group in PADDLEOCR_LANGS.values():
            if isinstance(group, str):
                languages.add(group)
            else:
                languages.update(group)
        for lang in languages:
            for tier in utils.MODEL_TIERS:
                det, rec, _ = utils.resolve_model_dirs(lang, tier)
                reachable.add(os.path.basename(det))
                reachable.add(os.path.basename(rec))
        for unused in ("PP-OCRv5_mobile_det", "PP-OCRv5_server_det"):
            assert unused not in reachable

    def test_every_reachable_model_is_shipped(self):
        """Guards the opposite mistake: pruning something the resolver needs."""
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        pruned = set()
        block = source.split("UNUSED_DET_MODELS = (", 1)[1].split(")", 1)[0]
        for token in block.split(","):
            token = token.strip().strip('"').strip("'")
            if token:
                pruned.add(token)
        reachable = set()
        for lang in ("en", "japan", "korean", "ar", "ka", "ch", "th"):
            for tier in utils.MODEL_TIERS:
                det, rec, _ = utils.resolve_model_dirs(lang, tier)
                reachable.add(os.path.basename(det))
                reachable.add(os.path.basename(rec))
        assert not (pruned & reachable)


class TestDuplicateTorchDlls:
    """torch ships the same DLLs at the package root and in ``torch/lib``."""

    def test_build_prunes_duplicate_root_dlls(self):
        source = (REPO_ROOT / "build.py").read_text(encoding="utf-8")
        assert "def prune_duplicate_torch_dlls" in source
        assert "prune_duplicate_torch_dlls(Path(temp_cli_dist))" in source

    def test_prune_keeps_the_canonical_copy(self, tmp_path):
        import build

        lib = tmp_path / "torch" / "lib"
        lib.mkdir(parents=True)
        for name in build.DUPLICATE_ROOT_DLLS:
            (lib / name).write_bytes(b"x" * 16)
            (tmp_path / name).write_bytes(b"x" * 16)
        build.prune_duplicate_torch_dlls(tmp_path)
        for name in build.DUPLICATE_ROOT_DLLS:
            assert not (tmp_path / name).exists(), f"{name} root copy survived"
            assert (lib / name).exists(), f"{name} canonical copy was removed"

    def test_prune_skips_when_canonical_copy_is_absent(self, tmp_path):
        """A layout change must not delete a file torch still needs."""
        import build

        (tmp_path / "torch").mkdir()
        target = tmp_path / "torch_cpu.dll"
        target.write_bytes(b"only copy")
        build.prune_duplicate_torch_dlls(tmp_path)
        assert target.exists()

    def test_prune_ignores_differing_sizes(self, tmp_path):
        """Only remove a byte-count match; a real difference means not a dup."""
        import build

        lib = tmp_path / "torch" / "lib"
        lib.mkdir(parents=True)
        (lib / "torch_cpu.dll").write_bytes(b"x" * 8)
        root_copy = tmp_path / "torch_cpu.dll"
        root_copy.write_bytes(b"x" * 16)
        build.prune_duplicate_torch_dlls(tmp_path)
        assert root_copy.exists()

    def test_prune_is_idempotent(self, tmp_path):
        import build

        lib = tmp_path / "torch" / "lib"
        lib.mkdir(parents=True)
        for name in build.DUPLICATE_ROOT_DLLS:
            (lib / name).write_bytes(b"x" * 16)
            (tmp_path / name).write_bytes(b"x" * 16)
        build.prune_duplicate_torch_dlls(tmp_path)
        build.prune_duplicate_torch_dlls(tmp_path)  # must not raise
        for name in build.DUPLICATE_ROOT_DLLS:
            assert (lib / name).exists()


class TestTierReachesTheRightEngines:
    """Which engines honor the tier, and which cannot.

    The tier is a PP-OCRv6 concept. PaddleOCR and ONNX Runtime DirectML both
    select a v6 model with it; the Google Lens detection pass shares PaddleOCR's
    detection model. EasyOCR is the internal fallback backend with one fixed
    recognition network per language, so it takes the parameter for signature
    parity and ignores it. These tests pin that split so neither the code nor
    the docs drift back into claiming universal support.
    """

    @pytest.mark.parametrize(
        ("module_name", "function_name"),
        [
            ("videocr.onnx_directml_ocr", "run_onnx_directml_on_stitched_images"),
            ("videocr.easyocr_directml", "run_easyocr_on_stitched_images"),
        ],
    )
    def test_runner_signature_has_model_tier(self, module_name, function_name):
        import importlib
        import inspect

        module = importlib.import_module(module_name)
        signature = inspect.signature(getattr(module, function_name))
        assert "model_tier" in signature.parameters

    def test_video_passes_the_tier_to_the_runner(self):
        source = (REPO_ROOT / "CLI" / "videocr" / "video.py").read_text(
            encoding="utf-8"
        )
        # Every directml runner call must forward the stored tier.
        assert source.count("self._model_tier,") >= 3

    def test_paddle_receives_the_tier_from_the_api(self):
        source = (REPO_ROOT / "CLI" / "videocr" / "api.py").read_text(
            encoding="utf-8"
        )
        assert "model_tier: str = utils.DEFAULT_MODEL_TIER" in source
        assert source.count("model_tier") >= 4

    def test_easyocr_documents_that_it_ignores_the_tier(self):
        source = (
            REPO_ROOT / "CLI" / "videocr" / "easyocr_directml.py"
        ).read_text(encoding="utf-8")
        assert "interface parity" in source
        assert "no tier to select" in source

    def test_cli_help_names_the_easyocr_exception(self):
        source = (REPO_ROOT / "CLI" / "videocr_cli.py").read_text(encoding="utf-8")
        assert "easyocr_directml has no tiers" in source

    def test_easyocr_reader_is_not_tier_parameterized(self):
        """EasyOCR's Reader takes no tier; the fallback must not fake one."""
        import inspect

        easyocr = pytest.importorskip("easyocr")
        params = set(inspect.signature(easyocr.Reader.__init__).parameters)
        assert "model_tier" not in params
        assert "model_type" not in params
