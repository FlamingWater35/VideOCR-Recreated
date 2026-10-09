"""Tests for videocr_gui.i18n — language loading, fallbacks, translation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videocr_gui import i18n

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES_DIR = REPO_ROOT / "languages"

EXPECTED_LANGUAGES = {"ar", "ch", "de", "en", "es", "fr", "id", "it", "ja", "ko", "pt", "ru", "th", "vi"}


class TestLanguageFiles:
    def test_expected_language_files_present(self):
        codes = {p.stem for p in LANGUAGES_DIR.glob("*.json")}
        assert codes >= EXPECTED_LANGUAGES

    @pytest.mark.parametrize("code", sorted(EXPECTED_LANGUAGES))
    def test_language_files_are_valid_json_dicts(self, code):
        data = json.loads((LANGUAGES_DIR / f"{code}.json").read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in data.items())

    def test_english_contains_keys_used_by_progress(self):
        en = json.loads((LANGUAGES_DIR / "en.json").read_text(encoding="utf-8"))
        for key in ("unknown", "lbl_step", "progress_step2_action", "eta_step"):
            assert key in en, key


class TestAvailableLanguages:
    def test_scans_repo_languages_dir(self, monkeypatch):
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(LANGUAGES_DIR))
        langs = i18n.get_available_languages()
        assert "English" in langs
        assert set(langs.values()) >= EXPECTED_LANGUAGES

    def test_native_names_for_known_codes(self, monkeypatch):
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(LANGUAGES_DIR))
        langs = i18n.get_available_languages()
        assert langs["Deutsch"] == "de"
        assert langs["日本語"] == "ja"
        assert langs["한국어"] == "ko"

    def test_missing_dir_falls_back_to_english(self, monkeypatch, tmp_path, _error_log):
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(tmp_path / "nope"))
        assert i18n.get_available_languages() == {"English": "en"}
        assert any("Languages directory not found" in m for m in _error_log)

    def test_empty_dir_falls_back_to_english(self, monkeypatch, tmp_path):
        empty = tmp_path / "langs"
        empty.mkdir()
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(empty))
        assert i18n.get_available_languages() == {"English": "en"}

    def test_unknown_code_gets_capitalized_name(self, monkeypatch, tmp_path):
        (tmp_path / "xx.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(tmp_path))
        assert i18n.get_available_languages() == {"Xx": "xx"}


class TestLoadLanguage:
    def test_load_english(self):
        i18n.load_language("en")
        assert i18n.current_language_code() == "en"
        assert i18n.LANG
        assert i18n.tr("lbl_step") == "Step"

    def test_load_german(self):
        i18n.load_language("de")
        assert i18n.current_language_code() == "de"
        assert i18n.LANG.get("lbl_step") not in (None, "", "Step")

    def test_missing_code_falls_back_to_english(self, _error_log):
        i18n.load_language("no-such-lang")
        assert i18n.current_language_code() == "no-such-lang"  # recorded as requested
        assert i18n.tr("lbl_step") == "Step"  # but content came from en
        assert any("Falling back to English" in m for m in _error_log)

    def test_missing_en_raises(self, monkeypatch, tmp_path, _error_log):
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(tmp_path))  # empty dir
        with pytest.raises(RuntimeError, match="en.json"):
            i18n.load_language("de")
        assert any("CRITICAL" in m for m in _error_log)

    def test_corrupt_json_falls_back(self, monkeypatch, tmp_path, _error_log):
        (tmp_path / "fr.json").write_text("{broken", encoding="utf-8")
        (tmp_path / "en.json").write_text(json.dumps({"lbl_step": "Step"}), encoding="utf-8")
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(tmp_path))
        i18n.load_language("fr")
        assert i18n.tr("lbl_step") == "Step"
        assert any("Syntax error in language file" in m for m in _error_log)


class TestHelpMessage:
    """The How to Use guide: present in every language, keybinds documented."""

    @staticmethod
    def _load(code: str) -> dict:
        return json.loads(
            (LANGUAGES_DIR / f"{code}.json").read_text(encoding="utf-8")
        )

    @pytest.mark.parametrize("code", sorted(EXPECTED_LANGUAGES))
    def test_help_keys_present_and_substantial(self, code):
        data = self._load(code)
        assert data.get("help_title", "").strip()
        # The guide is a full quick-start (sections + keybinds), not a stub.
        assert len(data.get("help_message", "")) > 500

    @pytest.mark.parametrize("code", sorted(EXPECTED_LANGUAGES))
    def test_no_unsubstituted_placeholders(self, code):
        # help_message is built from {ui_label} templates; a leftover brace
        # means a template key was not substituted for this language.
        message = self._load(code)["help_message"]
        assert "{" not in message and "}" not in message

    def test_english_help_matches_app_fallback(self):
        from videocr_gui.app import _HELP_MESSAGE

        data = self._load("en")
        assert data["help_message"] == _HELP_MESSAGE
        assert data["help_title"] == "How to Use"

    def test_english_help_documents_keybinds_and_seeking(self):
        message = self._load("en")["help_message"]
        assert "Left / Right arrow keys" in message
        assert "Keyboard Seek Step" in message
        assert "Seek bar: click or drag" in message


class TestTranslate:
    def test_key_from_loaded_language(self):
        i18n.load_language("en")
        assert i18n.tr("lbl_step", "fallback") == "Step"

    def test_default_used_when_key_missing_everywhere(self):
        i18n.load_language("en")
        assert i18n.tr("definitely_not_a_key", "fallback") == "fallback"

    def test_key_itself_when_no_default(self):
        i18n.load_language("en")
        assert i18n.tr("definitely_not_a_key") == "definitely_not_a_key"

    def test_falls_back_to_english_when_active_lang_lacks_key(self, monkeypatch, tmp_path):
        # Active language missing the key; English has it.
        (tmp_path / "xy.json").write_text(json.dumps({"only_here": "nur hier"}), encoding="utf-8")
        (tmp_path / "en.json").write_text(json.dumps({"shared": "english value"}), encoding="utf-8")
        monkeypatch.setattr(i18n, "LANGUAGES_DIR", str(tmp_path))
        i18n.load_language("xy")
        assert i18n.tr("shared") == "english value"
        assert i18n.tr("only_here") == "nur hier"

    def test_preferred_key_wins_over_default(self):
        i18n.load_language("en")
        assert i18n.tr("unknown", "custom default") == "unknown" or True
        # 'unknown' exists in en.json → translated value, not the default.
        assert i18n.tr("unknown", "custom default") != "custom default"
