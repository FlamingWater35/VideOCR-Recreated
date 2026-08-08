"""Configuration handling: config file location, defaults, load/save.

The config file is ``videocr_gui_config.ini`` (INI format via ``configparser``).
It lives next to the app when ``portable_mode.txt`` exists, otherwise in the
platform config dir (``%APPDATA%/VideOCR`` on Windows, XDG on Linux).

This is a port of the legacy ``VideOCR.py`` settings handling, minus the
update-check and benchmarking keys, with one behavior change: the DirectML
device index defaults to ``0`` and is persisted/restored reliably.
"""

from __future__ import annotations

import ast
import configparser
import os
import pathlib
import sys
from typing import Any

from . import constants as C

CONFIG_SECTION = "Settings"


# --- Paths -------------------------------------------------------------------
def get_app_dir() -> str:
    """Directory of the application (repo root when run from source, or exe dir when compiled)."""
    # Nuitka sets __compiled__ in compiled modules
    if "__compiled__" in globals():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


APP_DIR = get_app_dir()


def log_error(message: str, log_name: str = "error_log.txt") -> str:
    """Logs error messages to a platform-appropriate log file location."""
    portable_flag = os.path.join(APP_DIR, "portable_mode.txt")
    if os.path.exists(portable_flag):
        log_dir = APP_DIR
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            str(pathlib.Path.home()), "AppData", "Local"
        )
        log_dir = os.path.join(base, "VideOCR")
    else:
        xdg_state = os.environ.get("XDG_STATE_HOME")
        if xdg_state:
            log_dir = os.path.join(xdg_state, "VideOCR")
        else:
            log_dir = os.path.join(
                str(pathlib.Path.home()), ".local", "state", "VideOCR"
            )

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)
    import datetime

    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
    return log_path


def get_config_file_path() -> str:
    """Determines the correct path for the config file depending on installation mode."""
    portable_flag = os.path.join(APP_DIR, "portable_mode.txt")
    if os.path.exists(portable_flag):
        config_dir = APP_DIR
    elif os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(
            str(pathlib.Path.home()), "AppData", "Roaming"
        )
        config_dir = os.path.join(base, "VideOCR")
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            config_dir = os.path.join(xdg_config, "VideOCR")
        else:
            config_dir = os.path.join(str(pathlib.Path.home()), ".config", "VideOCR")

    os.makedirs(config_dir, exist_ok=True)

    return os.path.join(config_dir, "videocr_gui_config.ini")


CONFIG_FILE = get_config_file_path()

try:
    DEFAULT_DOCUMENTS_DIR = str(pathlib.Path.home() / "Documents")
except Exception:
    DEFAULT_DOCUMENTS_DIR = ""


# --- Defaults -----------------------------------------------------------------
def get_default_settings() -> dict[str, Any]:
    """Returns a dictionary of default settings (config keys -> default values).

    ``--directml_device_index`` defaults to ``0`` (first GPU) and is persisted.
    Update-check and benchmarking keys are intentionally absent.
    """
    return {
        "--language": "en",
        "ocr_engine": C.DEFAULT_OCR_ENGINE,
        "subtitle_language": C.DEFAULT_SUBTITLE_LANGUAGE,
        "subtitle_position": C.DEFAULT_INTERNAL_SUBTITLE_POSITION,
        "post_action": 0,
        "--time_start": C.DEFAULT_TIME_START,
        "--time_end": "",
        "--conf_threshold": str(C.DEFAULT_CONF_THRESHOLD),
        "--sim_threshold": str(C.DEFAULT_SIM_THRESHOLD),
        "--max_merge_gap": str(C.DEFAULT_MAX_MERGE_GAP),
        "--brightness_threshold": "",
        "--ssim_threshold": str(C.DEFAULT_SSIM_THRESHOLD),
        "--ocr_image_max_width": str(C.DEFAULT_OCR_IMAGE_MAX_WIDTH),
        "--directml_device_index": "0",
        "--directml_grid_max_width": "2400",
        "--directml_grid_max_height": "2400",
        "--directml_performance_preset": "Balanced (recommended)",
        "--directml_recognition_mode": "Stable Hybrid (recommended)",
        "--directml_frame_scan_mode": "CPU SSIM (compatible)",
        "--onnx_directml_tuning": "Balanced ONNX (recommended)",
        "--frames_to_skip": str(C.DEFAULT_FRAMES_TO_SKIP),
        "--use_fullframe": False,
        "--use_gpu": True,
        "--use_angle_cls": False,
        "--post_processing": False,
        "--min_subtitle_duration": str(C.DEFAULT_MIN_SUBTITLE_DURATION),
        "--use_server_model": False,
        "--use_dual_zone": False,
        "enable_subtitle_alignment": False,
        "--subtitle_alignment": C.DEFAULT_SUBTITLE_ALIGNMENT,
        "--subtitle_alignment2": C.DEFAULT_SUBTITLE_ALIGNMENT,
        "--keyboard_seek_step": str(C.KEY_SEEK_STEP),
        "--default_output_dir": DEFAULT_DOCUMENTS_DIR,
        "--save_in_video_dir": True,
        "--send_notification": True,
        "--save_crop_box": True,
        "--saved_crop_boxes": "[]",
        "prevent_system_sleep": True,
        "--normalize_to_simplified_chinese": True,
        "gui_scaling": C.DEFAULT_GUI_SCALING,
        # Label detection (text outside the subtitle crop, e.g. names)
        "enable_label_detection": False,
        "--label_ocr_image_max_width": "720",
        "--label_min_display_duration": "1.0",
        "--label_min_confirmation_frames": "2",
        "--label_reappear_merge_gap": "2.0",
        "--label_filter_single_char": True,
    }


# Keys that are stored as booleans
_BOOL_KEYS = {
    "--use_fullframe",
    "--use_gpu",
    "--use_angle_cls",
    "--post_processing",
    "--use_server_model",
    "--use_dual_zone",
    "enable_subtitle_alignment",
    "enable_label_detection",
    "--label_filter_single_char",
    "--save_in_video_dir",
    "--send_notification",
    "--save_crop_box",
    "prevent_system_sleep",
    "--normalize_to_simplified_chinese",
}


def _normalize_value(key: str, raw: str) -> Any:
    if key in _BOOL_KEYS:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw


def load_settings() -> dict[str, Any]:
    """Loads settings from the config file. Returns merged defaults on any error.

    This is GUI-independent (used by tests and by the app); GUI widgets are
    populated from the returned dict by the caller.
    """
    settings = get_default_settings()
    if not os.path.exists(CONFIG_FILE):
        return settings

    parser = configparser.ConfigParser()
    try:
        # Config files are written as UTF-8 (see save_settings); read them the
        # same way so CJK values (paths, output names) survive on Windows where
        # the locale encoding is cp1252/charmap.
        parser.read(CONFIG_FILE, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError) as e:
        log_error(f"Error parsing config file {CONFIG_FILE}: {e}. Using defaults.")
        return settings

    if not parser.has_section(CONFIG_SECTION):
        return settings

    for key, default in get_default_settings().items():
        if parser.has_option(CONFIG_SECTION, key):
            try:
                settings[key] = _normalize_value(key, parser.get(CONFIG_SECTION, key))
            except Exception as e:
                log_error(f"Error loading setting '{key}': {e}. Using default.")
                settings[key] = default
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Writes the settings dict to the config file."""
    config = configparser.ConfigParser()
    config.add_section(CONFIG_SECTION)
    for key, value in settings.items():
        config.set(CONFIG_SECTION, key, str(value))
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as configfile:
            config.write(configfile)
    except Exception as e:
        log_error(f"Error saving settings to {CONFIG_FILE}: {e}")


def parse_saved_crop_boxes(raw: str) -> list[dict[str, Any]]:
    """Parses the stored relative crop-box list (a repr'd Python literal)."""
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return value
    except (ValueError, SyntaxError):
        log_error(f"Could not parse saved_crop_boxes: {raw}")
    return []
