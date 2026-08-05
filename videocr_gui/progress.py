"""CLI output parsing and progress/ETA computation (ported from the legacy GUI).

The worker thread feeds raw CLI lines here; this module decides which lines are
progress updates, status messages, or plain log output, and emits structured
progress payloads.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from . import i18n


@dataclass
class ProgressState:
    """Accumulates per-step progress state, mirroring legacy handle_progress."""

    last_key: str | None = None
    start_time: float = 0.0
    last_update_time: float = 0.0
    start_percent: float = 0.0
    last_eta: str = ""
    last_taskbar_val: int = -1


@dataclass
class ProgressUpdate:
    text: str = ""
    percent: float | None = None
    eta: str = ""
    step: int = 0


_STATE = ProgressState()


def parse_srt_time_to_seconds(time_str: str) -> float:
    """Parses '00:00:01,500' or '00:01:00' into seconds."""
    try:
        parts = time_str.replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except Exception:
        return 0.0
    return 0.0


def format_seconds(seconds: float | int | None) -> str:
    """Converts seconds to '1h 05m' or '05m 30s' format."""
    if seconds is None or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m:02d}m {s:02d}s"


def handle_progress(
    label_format_key: str,
    step_num: int,
    curr_item: str,
    total: str,
    extra: str | None = None,
) -> ProgressUpdate:
    """Computes a ProgressUpdate for one step from parsed CLI match groups."""
    global _STATE
    state = _STATE
    current_time = time.time()
    is_time_based = label_format_key == "progress_step1"

    if is_time_based:
        curr_sec = parse_srt_time_to_seconds(curr_item)
        target_sec = parse_srt_time_to_seconds(total)
        current_percent = (curr_sec / target_sec) * 100.0 if target_sec > 0 else 0.0
        current_percent = min(max(current_percent, 0.0), 100.0)
        current_item_display = curr_item
        display_total = total
        frame_num = extra or ""
    else:
        try:
            current_item = int(curr_item)
        except ValueError:
            current_item = 0
        if total == "Unknown":
            display_total = i18n.tr("unknown", "unknown")
            current_percent = 0.0
        else:
            try:
                total_items = int(total)
                display_total = str(total_items)
            except ValueError:
                total_items = 0
                display_total = total
            current_percent = (current_item / total_items) * 100.0 if total_items > 0 else 0.0
        current_item_display = str(current_item)
        frame_num = ""

    if state.last_key != label_format_key:
        state.last_key = label_format_key
        state.start_time = current_time
        state.last_update_time = 0.0
        state.start_percent = current_percent

    time_delta = current_time - state.last_update_time
    if current_percent < 100 and current_percent < 99.9 and time_delta < 0.2:
        return ProgressUpdate()

    state.last_update_time = current_time

    step_word = i18n.tr("lbl_step", "Step")
    prefix = f"{step_word} {step_num}/3:"

    if label_format_key == "progress_step1":
        action_text = i18n.tr("progress_step1_action", "Processing video...")
        frame_lbl = i18n.tr("lbl_frame", "Frame")
        msg_template = f"{prefix} {action_text} {curr_item} / {total}, {frame_lbl}: {frame_num} ({{percent}}%)"
    elif label_format_key == "progress_step2":
        raw_msg = i18n.tr("progress_step2_action", "Performing Text-Detection on image {current} of {total} ({percent}%)")
        msg_template = prefix + " " + raw_msg
    else:
        raw_msg = i18n.tr("progress_step3_action", "Performing OCR on image {current} of {total} ({percent}%)")
        msg_template = prefix + " " + raw_msg

    eta_prefix = f"{i18n.tr('eta_step', 'ETA Step')} {step_num}/3"
    elapsed = current_time - state.start_time
    percent_done = current_percent - state.start_percent
    eta_str = state.last_eta
    if percent_done > 0 and elapsed > 0:
        rate = percent_done / elapsed
        remaining = 100.0 - current_percent
        if rate > 0:
            eta_str = f"{eta_prefix}: {format_seconds(remaining / rate)}"
            state.last_eta = eta_str

    if label_format_key in ("progress_step2", "progress_step3"):
        display_text = msg_template.format(
            current=current_item_display, total=display_total, percent=f"{current_percent:.1f}"
        )
    else:
        display_text = msg_template.format(percent=f"{current_percent:.1f}")

    return ProgressUpdate(text=display_text, percent=current_percent, eta=eta_str, step=step_num)


# --- Regexes shared by the worker -------------------------------------------------
STEP1_PROGRESS_PATTERN = re.compile(r"Step (\d+)/\d+: Processing video\.\.\. Current: ([\d:]+) / ([\d:]+|Unknown), Frame: (\d+)")
STEP_IMAGE_PROGRESS_PATTERN = re.compile(r"Step (\d+)/\d+: Performing (?:Text-Detection|OCR|ONNX DirectML OCR) on image (\d+) of (\d+)")
REPACKING_PATTERN = re.compile(r"Analyzing frame (\d+) of (\d+)")
STARTING_PADDLEOCR_PATTERN = re.compile(r"Starting PaddleOCR\.\.\.")
STARTING_LENS_PATTERN = re.compile(r"Starting Google Lens CLI\.\.\.")
STARTING_EASYOCR_PATTERN = re.compile(r"Starting EasyOCR(?: DirectML)?\.\.\.")
STARTING_ONNX_PATTERN = re.compile(r"Starting ONNX Runtime DirectML OCR")
INFO_PASS_PATTERN = re.compile(r"Running Text-Detection-Only pass on (\d+) filtered frame\(s\) stitched into (\d+) image grid\(s\)\.\.\.")
FILTERED_PATTERN = re.compile(r"Filtered out (\d+) redundant frame\(s\) via Text-Detection and tight-box SSIM analysis\.")
GENERATING_SUBTITLES_PATTERN = re.compile(r"Generating subtitles\.\.\.")
REACHED_END_TIME_PATTERN = re.compile(r"Reached end time\. Stopping\.")
UNSUPPORTED_HARDWARE_ERROR_PATTERN = re.compile(r"Unsupported Hardware Error: (.*)")
WARNING_HARDWARE_PATTERN = re.compile(r"Hardware Check Warning: (.*)")
PROCESS_ERROR_PATTERN = re.compile(r"Error: Process failed.")


def classify_line(line: str) -> tuple[str, dict[str, Any]]:
    """Classifies one CLI stdout line.

    Returns (kind, payload) where kind is one of:
      'progress', 'log', 'fatal', 'warning', 'process_error', 'starting',
      'info_pass', 'filtered', 'repacking', 'generating', 'reached_end'
    """
    match1 = STEP1_PROGRESS_PATTERN.search(line)
    if match1:
        return "progress", {
            "key": "progress_step1",
            "step": int(match1.group(1)),
            "curr": match1.group(2),
            "total": match1.group(3),
            "extra": match1.group(4),
        }

    match2 = STEP_IMAGE_PROGRESS_PATTERN.search(line)
    if match2:
        step = int(match2.group(1))
        key = "progress_step2" if step == 2 else "progress_step3"
        return "progress", {"key": key, "step": step, "curr": match2.group(2), "total": match2.group(3), "extra": None}

    if REPACKING_PATTERN.search(line):
        return "repacking", {"line": line}

    m = UNSUPPORTED_HARDWARE_ERROR_PATTERN.search(line)
    if m:
        return "fatal", {"message": m.group(1)}

    m = WARNING_HARDWARE_PATTERN.search(line)
    if m:
        return "warning", {"message": m.group(1)}

    if PROCESS_ERROR_PATTERN.search(line):
        return "process_error", {"line": line}

    if STARTING_PADDLEOCR_PATTERN.search(line):
        return "starting", {"key": "cli_starting_paddleocr", "line": line}
    if STARTING_LENS_PATTERN.search(line):
        return "starting", {"key": "cli_starting_lens", "line": line}
    if STARTING_EASYOCR_PATTERN.search(line) or STARTING_ONNX_PATTERN.search(line):
        return "starting", {"key": None, "line": line.strip()}

    m = INFO_PASS_PATTERN.search(line)
    if m:
        return "info_pass", {"frames": m.group(1), "grids": m.group(2)}

    m = FILTERED_PATTERN.search(line)
    if m:
        return "filtered", {"frames": m.group(1)}

    if GENERATING_SUBTITLES_PATTERN.search(line):
        return "generating", {"line": line}

    if REACHED_END_TIME_PATTERN.search(line):
        return "reached_end", {"line": line}

    return "log", {"line": line}
