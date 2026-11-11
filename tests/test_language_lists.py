"""Guards the per-engine subtitle-language lists.

Each OCR engine can only recognise what its models were trained on, and the
GUI's language combo must not offer anything the selected engine cannot do.
These tests pin each list to the engine's own source of truth:

* PaddleOCR -> ``CLI/videocr/lang_dictionaries.PADDLEOCR_LANGS``
* Google Lens -> ``CLI/videocr/lang_dictionaries.GOOGLE_LENS_LANGS``
* EasyOCR -> ``CLI/videocr/easyocr_directml.EASYOCR_LANG_MAP``
* ONNX Runtime DirectML -> ``rapidocr.utils.model_resolver.PP_OCRV6_LANGS``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DIR = REPO_ROOT / "CLI"


@pytest.fixture(scope="module", autouse=True)
def _cli_on_path():
    for entry in (str(REPO_ROOT), str(CLI_DIR)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    yield


def _constants():
    from videocr_gui import constants

    return constants


class TestOnnxLanguageList:
    def test_list_is_not_empty(self):
        assert len(_constants().ONNX_LANGUAGES_LIST) > 40

    def test_every_code_is_accepted_by_ppocrv6(self):
        """The list must be exactly what RapidOCR's unified model accepts."""
        rapidocr_resolver = pytest.importorskip("rapidocr.utils.model_resolver")
        from rapidocr.utils.typings import ModelType, OCRVersion, TaskType

        supported = rapidocr_resolver.PP_OCRV6_LANGS
        for name, code in _constants().ONNX_LANGUAGES_LIST:
            assert code in supported, f"{name} ({code}) is not a PP-OCRv6 language"
            # And the resolver must actually resolve it (catches alias drift).
            rapidocr_resolver.resolve_model_key(
                TaskType.REC, OCRVersion.PPOCRV6, code, ModelType.SMALL
            )

    def test_list_covers_every_ppocrv6_language_we_can_name(self):
        """Nothing PP-OCRv6 supports may be silently missing from the combo."""
        rapidocr_resolver = pytest.importorskip("rapidocr.utils.model_resolver")

        constants = _constants()
        paddle_codes = {code for _n, code in constants.PADDLEOCR_LANGUAGES_LIST}
        offered = {code for _n, code in constants.ONNX_LANGUAGES_LIST}
        namable = {
            code for code in rapidocr_resolver.PP_OCRV6_LANGS if code in paddle_codes
        }
        assert namable == offered

    def test_codes_are_unique(self):
        codes = [code for _name, code in _constants().ONNX_LANGUAGES_LIST]
        assert len(codes) == len(set(codes))

    def test_names_are_unique(self):
        names = [name for name, _code in _constants().ONNX_LANGUAGES_LIST]
        assert len(names) == len(set(names))

    def test_arabic_korean_russian_thai_are_absent(self):
        """PP-OCRv6 has no recognition model for these; offering them lies."""
        codes = {code for _name, code in _constants().ONNX_LANGUAGES_LIST}
        for code in ("ar", "korean", "ru", "th"):
            assert code not in codes

    def test_lookup_maps_agree_with_the_list(self):
        constants = _constants()
        assert constants.onnx_display_names == [
            name for name, _code in constants.ONNX_LANGUAGES_LIST
        ]
        assert constants.onnx_abbr_lookup == dict(constants.ONNX_LANGUAGES_LIST)


class TestEngineListsMatchTheirEngines:
    def test_google_lens_matches_the_cli_table(self):
        from videocr.lang_dictionaries import GOOGLE_LENS_LANGS

        codes = {code for _n, code in _constants().GOOGLE_LENS_LANGUAGES_LIST}
        assert codes == set(GOOGLE_LENS_LANGS)

    def test_easyocr_codes_exist_in_the_cli_map(self):
        from videocr.easyocr_directml import EASYOCR_LANG_MAP

        codes = {code for _n, code in _constants().EASYOCR_LANGUAGES_LIST}
        assert codes <= set(EASYOCR_LANG_MAP)

    def test_paddle_codes_exist_in_the_cli_table(self):
        from videocr.lang_dictionaries import PADDLEOCR_LANGS

        flattened = set()
        for group in PADDLEOCR_LANGS.values():
            flattened |= set(group)
        codes = {code for _n, code in _constants().PADDLEOCR_LANGUAGES_LIST}
        assert codes <= flattened


class TestEngineRoutesToItsOwnList:
    def test_onnx_engine_gets_the_onnx_list(self, qtbot, monkeypatch):
        """_update_lang_list must switch to the PP-OCRv6 coverage for ONNX."""
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        constants = _constants()
        from videocr_gui.app import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        try:
            window._update_lang_list("ONNX Runtime DirectML (AMD GPU Experimental)")
            texts = [
                window.lang_combo.itemText(i)
                for i in range(window.lang_combo.count())
            ]
            assert texts == constants.onnx_display_names
        finally:
            window.close()

    def test_easyocr_keeps_its_own_list(self, qtbot, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        constants = _constants()
        from videocr_gui.app import MainWindow

        window = MainWindow()
        qtbot.addWidget(window)
        try:
            window._update_lang_list("EasyOCR DirectML (AMD GPU)")
            texts = [
                window.lang_combo.itemText(i)
                for i in range(window.lang_combo.count())
            ]
            assert texts == constants.easyocr_display_names
        finally:
            window.close()


class TestOnnxLanguageReachesTheCli:
    def test_build_args_forwards_the_onnx_code(self):
        from videocr_gui import config
        from videocr_gui.args import build_args

        settings = config.get_default_settings()
        settings["ocr_engine"] = "ONNX Runtime DirectML (AMD GPU Experimental)"
        settings["subtitle_language"] = "Polish"
        args, _warnings = build_args("C:/v/movie.mp4", settings, [], "C:/v/out.srt")
        assert args is not None
        assert args["ocr_engine"] == "onnx_directml"
        assert args["lang"] == "pl"

    def test_a_language_easyocr_had_but_onnx_lacks_is_rejected(self):
        """'Korean + English' is gone from the ONNX list; it must not leak."""
        from videocr_gui import config
        from videocr_gui.args import build_args

        settings = config.get_default_settings()
        settings["ocr_engine"] = "ONNX Runtime DirectML (AMD GPU Experimental)"
        settings["subtitle_language"] = "Korean + English"
        args, errors = build_args("C:/v/movie.mp4", settings, [], "C:/v/out.srt")
        # RapidOCR's PP-OCRv6 resolver rejects 'korean' outright, so a silent
        # fallback to English would be a lie about what the run will do.
        assert args is None
        assert errors
        assert "Korean + English" in errors[0]
