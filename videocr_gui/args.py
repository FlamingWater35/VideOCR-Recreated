"""Arg building/validation: maps GUI state to CLI args (ported from legacy GUI)."""

from __future__ import annotations

import pathlib
from typing import Any

from . import constants as C
from . import i18n
from .config import DEFAULT_DOCUMENTS_DIR


def _is_valid_time(time_str: str | None) -> bool:
    if not time_str:
        return True
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return m >= 0 and s >= 0 and s < 60
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h >= 0 and m >= 0 and s >= 0 and m < 60 and s < 60
    except ValueError:
        return False
    return False


def _time_to_seconds(time_str: str | None) -> int | None:
    if not time_str:
        return None
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def generate_output_path(
    video_path: str, settings: dict[str, Any], default_dir: str = DEFAULT_DOCUMENTS_DIR,
) -> pathlib.Path:
    """Generates a unique output subtitle path based on video path, settings and language.

    When label detection is enabled the output format is ASS (needed for the
    positioned Label events), so the extension becomes ``.ass``.
    """
    video_file = pathlib.Path(video_path)
    stem = video_file.stem

    save_in_video_dir = settings.get("--save_in_video_dir", True)
    if save_in_video_dir:
        output_dir = video_file.parent
    else:
        output_dir_str = str(settings.get("--default_output_dir", default_dir)).strip()
        output_dir = pathlib.Path(output_dir_str) if output_dir_str else pathlib.Path(default_dir)

    lang_code = _language_iso_code(settings)
    ext = ".ass" if settings.get("enable_label_detection") else ".srt"
    output_path = output_dir / f"{stem}.{lang_code}{ext}"

    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{stem}({counter}).{lang_code}{ext}"
        counter += 1
    return output_path


def _language_iso_code(settings: dict[str, Any]) -> str:
    engine = settings.get("ocr_engine", C.DEFAULT_OCR_ENGINE)
    lang_name = settings.get("subtitle_language", C.DEFAULT_SUBTITLE_LANGUAGE)
    if "Google Lens" in engine:
        code = C.lens_abbr_lookup.get(lang_name, "en")
    elif "EasyOCR DirectML" in engine or "ONNX Runtime DirectML" in engine:
        easy = C.easyocr_abbr_lookup.get(lang_name, "en")
        code = C.EASY_TO_ISO_MAP.get(easy, easy)
    else:
        paddle = C.paddle_abbr_lookup.get(lang_name, "en")
        code = C.PADDLE_TO_ISO_MAP.get(paddle, paddle)
    return code


def build_args(
    video_path: str,
    settings: dict[str, Any],
    crop_boxes: list[dict[str, Any]],
    output_path: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validates settings + crop boxes and builds the CLI args dict.

    Returns (args_dict, []) on success or (None, errors) on validation failure.
    """
    errors: list[str] = []

    time_start = str(settings.get("--time_start", "")).strip()
    time_end = str(settings.get("--time_end", "")).strip()

    if not _is_valid_time(time_start):
        errors.append(i18n.tr("val_err_start_time", "Invalid Start Time format."))
    if not _is_valid_time(time_end):
        errors.append(i18n.tr("val_err_end_time", "Invalid End Time format."))

    duration_s = float(settings.get("_video_duration_ms", 0)) / 1000.0
    start_s = _time_to_seconds(time_start)
    end_s = _time_to_seconds(time_end)

    if start_s is not None and duration_s > 0 and start_s > duration_s:
        errors.append(i18n.tr("val_err_start_exceeds", "Start Time ({}) exceeds video duration ({}).").format(
            _fmt_time(start_s), _fmt_time(duration_s)))
    if end_s is not None and duration_s > 0 and end_s > duration_s:
        errors.append(i18n.tr("val_err_end_exceeds", "End Time ({}) exceeds video duration ({}).").format(
            _fmt_time(end_s), _fmt_time(duration_s)))
    if start_s is not None and end_s is not None and start_s > end_s:
        errors.append(i18n.tr("val_err_start_after_end", "Start Time cannot be after End Time."))

    use_dual_zone = settings.get("--use_dual_zone", False)
    if use_dual_zone and len(crop_boxes) != 2:
        errors.append(i18n.tr("val_err_dual_zone", "Dual Zone OCR is enabled, but 2 crop boxes have not been selected."))

    numeric_params = {
        "--conf_threshold": (int, 0, 100, "Confidence Threshold"),
        "--sim_threshold": (int, 0, 100, "Similarity Threshold"),
        "--brightness_threshold": (int, 0, 255, "Brightness Threshold"),
        "--ssim_threshold": (int, 0, 100, "SSIM Threshold"),
        "--ocr_image_max_width": (int, 0, None, "Max OCR Image Width"),
        "--directml_device_index": (int, 0, None, "DirectML Adapter Index"),
        "--directml_grid_max_width": (int, 512, None, "DirectML Grid Max Width"),
        "--directml_grid_max_height": (int, 512, None, "DirectML Grid Max Height"),
        "--frames_to_skip": (int, 0, None, "Frames to Skip"),
        "--max_merge_gap": (float, 0.0, None, "Max Merge Gap"),
        "--min_subtitle_duration": (float, 0.0, None, "Minimum Subtitle Duration"),
    }
    for key, (cast_type, min_val, max_val, name) in numeric_params.items():
        value_str = str(settings.get(key, "")).strip()
        if not value_str:
            continue
        try:
            value = cast_type(value_str)
            if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                raise ValueError
        except ValueError:
            range_parts = []
            if min_val is not None:
                range_parts.append(f">={min_val}")
            if max_val is not None:
                range_parts.append(f"<={max_val}")
            errors.append(i18n.tr(
                "val_err_numeric",
                "Invalid value for {}. Must be {} {} {}.",
            ).format(name, "an" if cast_type.__name__.startswith(("i", "I")) else "a", cast_type.__name__, " and ".join(range_parts)))

    if errors:
        return None, errors

    args: dict[str, Any] = {}
    args["video_path"] = video_path

    engine_display = settings.get("ocr_engine", C.DEFAULT_OCR_ENGINE)
    lang_name = settings.get("subtitle_language", C.DEFAULT_SUBTITLE_LANGUAGE)
    if "Google Lens" in engine_display:
        args["ocr_engine"] = "google_lens"
        lang_abbr = C.lens_abbr_lookup.get(lang_name)
    elif "ONNX Runtime DirectML" in engine_display:
        args["ocr_engine"] = "onnx_directml"
        lang_abbr = C.easyocr_abbr_lookup.get(lang_name)
    elif "EasyOCR DirectML" in engine_display:
        args["ocr_engine"] = "easyocr_directml"
        lang_abbr = C.easyocr_abbr_lookup.get(lang_name)
    else:
        args["ocr_engine"] = "paddleocr"
        lang_abbr = C.paddle_abbr_lookup.get(lang_name)
    if lang_abbr:
        args["lang"] = lang_abbr

    # subtitle position
    # settings store the internal value (e.g. "left"); also accept the
    # localized display text for compatibility with older configs.
    pos_internal_values = [internal for _, internal in C.SUBTITLE_POSITIONS_LIST]
    pos_display_to_internal = {i18n.tr(key, key): internal for key, internal in C.SUBTITLE_POSITIONS_LIST}
    pos_raw = str(settings.get("subtitle_position", ""))
    pos_value = pos_raw if pos_raw in pos_internal_values else pos_display_to_internal.get(pos_raw, C.DEFAULT_INTERNAL_SUBTITLE_POSITION)
    args["subtitle_position"] = pos_value

    # generic -- args (excluding GUI-only keys)
    excluded = {
        "--keyboard_seek_step", "--default_output_dir", "--save_in_video_dir",
        "--send_notification", "--save_crop_box", "--language", "--use_dual_zone",
        "--subtitle_alignment", "--subtitle_alignment2", "--saved_crop_boxes",
        "--output",
    }
    for key, value in settings.items():
        if not key.startswith("--") or key in excluded:
            continue
        stripped = key.lstrip("-")
        if value is None or str(value).strip() == "":
            continue
        if key == "--directml_performance_preset":
            args[stripped] = C.DIRECTML_PERFORMANCE_TO_CLI.get(str(value), str(C.DIRECTML_PERFORMANCE_TO_CLI.get("Balanced (recommended)", "balanced")))
        elif key == "--directml_recognition_mode":
            args[stripped] = C.DIRECTML_RECOGNITION_TO_CLI.get(str(value), "stable")
        elif key == "--directml_frame_scan_mode":
            args[stripped] = C.DIRECTML_FRAME_SCAN_TO_CLI.get(str(value), "cpu_ssim")
        elif key == "--onnx_directml_tuning":
            args[stripped] = C.ONNX_DIRECTML_TUNING_TO_CLI.get(str(value), "balanced")
        elif isinstance(value, bool):
            args[stripped] = value
        else:
            args[stripped] = str(value).strip()

    # subtitle alignment
    # settings store the internal value (e.g. "top-left"); also accept the
    # localized display text for compatibility with older configs.
    if settings.get("enable_subtitle_alignment"):
        align_internal_values = [internal for _, internal in C.SUBTITLE_ALIGNMENT_LIST]
        align_map = {i18n.tr(key, internal): internal for key, internal in C.SUBTITLE_ALIGNMENT_LIST}

        def _align_value(raw: Any) -> str:
            s = str(raw)
            return s if s in align_internal_values else align_map.get(s, C.DEFAULT_SUBTITLE_ALIGNMENT)

        args["subtitle_alignment"] = _align_value(settings.get("--subtitle_alignment", ""))
        if use_dual_zone:
            args["subtitle_alignment2"] = _align_value(settings.get("--subtitle_alignment2", ""))

    # label detection (text outside the subtitle crop, e.g. names)
    if settings.get("enable_label_detection"):
        args["enable_label_detection"] = True

    # notifications handled by GUI, not CLI
    args["send_notification"] = settings.get("--send_notification", True)
    args["allow_system_sleep"] = True

    # crop zones
    use_fullframe = settings.get("--use_fullframe", False)
    if use_dual_zone and len(crop_boxes) == 2:
        args.update(crop_boxes[0]["coords"])
        box2 = crop_boxes[1]["coords"]
        args.update({f"{k}2": v for k, v in box2.items()})
    elif not use_fullframe and crop_boxes:
        args.update(crop_boxes[0]["coords"])

    out = output_path or str(generate_output_path(video_path, settings))
    args["output"] = out

    return args, []


def _fmt_time(seconds: float | int) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
