"""i18n: load languages/*.json (same format as the legacy GUI) and translate."""

from __future__ import annotations

import json
import os
from typing import cast

from .config import APP_DIR, log_error

LANGUAGES_DIR = os.path.join(APP_DIR, "languages")

LANG: dict[str, str] = {}
_current_code = "en"
_en_cache: dict[str, str] | None = None


def _english() -> dict[str, str]:
    """Returns the English language dict, cached after first load."""
    global _en_cache
    if _en_cache is None:
        _en_cache = _load_file("en") or {}
    return _en_cache


def get_available_languages() -> dict[str, str]:
    """Scans the languages directory; returns {native name: code}."""
    langs: dict[str, str] = {}
    code_to_native = {
        "en": "English", "de": "Deutsch", "ch": "中文", "es": "Español",
        "fr": "Français", "pt": "Português", "it": "Italiano", "ar": "العربية",
        "ru": "Русский", "id": "Bahasa Indonesia", "th": "ไทย", "ko": "한국어",
        "ja": "日本語", "vi": "Tiếng Việt",
    }
    if not os.path.isdir(LANGUAGES_DIR):
        log_error(f"Languages directory not found at {LANGUAGES_DIR}")
        return {"English": "en"}

    for filename in os.listdir(LANGUAGES_DIR):
        if filename.endswith(".json"):
            code = filename[:-5]
            langs[code_to_native.get(code, code.capitalize())] = code
    return langs if langs else {"English": "en"}


def _load_file(code: str) -> dict[str, str] | None:
    lang_path = os.path.join(LANGUAGES_DIR, f"{code}.json")
    if os.path.exists(lang_path):
        try:
            with open(lang_path, encoding="utf-8") as f:
                return cast(dict[str, str], json.load(f))
        except json.JSONDecodeError as e:
            log_error(f"Syntax error in language file {code}.json: {e}")
    return None


def load_language(code: str) -> None:
    """Loads a language JSON file into the global LANG dict (falls back to en)."""
    global LANG, _current_code

    loaded = _load_file(code)
    if loaded is None:
        log_error(f"Language file for '{code}' not found or invalid. Falling back to English.")
        loaded = _load_file("en")
        if loaded is None:
            log_error("CRITICAL: English language file 'en.json' is missing or invalid.")
            raise RuntimeError("Default language file 'en.json' is missing or corrupt.")
    LANG = loaded
    _current_code = code


def current_language_code() -> str:
    return _current_code


def tr(key: str, default: str = "") -> str:
    """Translates a key using the loaded language dict.

    Falls back to the English language file when the key is missing from the
    current language, and to ``default``/the key itself only as a last resort.
    Never returns a raw ``tip_*``/``lbl_*`` key for a missing translation when
    English has a value for it.
    """
    if LANG and key in LANG:
        return LANG[key]
    # Fall back to English for keys the active language lacks.
    if _current_code != "en":
        en = _english()
        if key in en:
            return en[key]
    if default:
        return default
    return key
