import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from typing import IO, Any

import av
import fast_ssim  # type: ignore
import numpy as np
from cpuid import cpuid, xgetbv  # type: ignore
from PIL import Image

from .lang_dictionaries import PADDLEOCR_LANGS
from .models import PredictedText

ALIGNMENT_MAP = {
    "bottom-left": "an1",
    "bottom-center": "an2",
    "bottom-right": "an3",
    "middle-left": "an4",
    "middle-center": "an5",
    "middle-right": "an6",
    "top-left": "an7",
    "top-center": "an8",
    "top-right": "an9",
}
VALID_ALIGNMENT_NAMES = set(ALIGNMENT_MAP.keys())


def get_ms_from_time_str(time_str: str) -> float:
    """Convert time string to milliseconds."""
    t = [float(x) for x in time_str.split(":")]
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(f'Time data "{time_str}" does not match format "%H:%M:%S"')
    return td.total_seconds() * 1000


def get_srt_timestamp(frame_index: int, fps: float, offset_ms: float = 0.0) -> str:
    """Convert frame index into SRT timestamp."""
    td = datetime.timedelta(milliseconds=(frame_index / fps * 1000 + offset_ms))
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_srt_timestamp_from_ms(ms: float) -> str:
    """Convert milliseconds into SRT timestamp."""
    td = datetime.timedelta(milliseconds=ms)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def get_ass_timestamp_from_ms(ms: float) -> str:
    """Convert milliseconds into ASS timestamp (H:MM:SS.cc)."""
    td = datetime.timedelta(milliseconds=ms)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    centiseconds = td.microseconds // 10000
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def ass_anchor_point(
    align: int, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """ASS \\an1..\\an9 anchor point of the bounding box (x1, y1, x2, y2).

    ASS places \\pos at the alignment point of the text block, so with \\an7
    (top-left) the anchor is the box's top-left corner, with \\an5 its center,
    etc. Out-of-range values fall back to top-left.
    """
    if not 1 <= align <= 9:
        align = 7
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    xs = {1: x1, 2: cx, 3: x2, 4: x1, 5: cx, 6: x2, 7: x1, 8: cx, 9: x2}
    ys = {1: y2, 2: y2, 3: y2, 4: cy, 5: cy, 6: cy, 7: y1, 8: y1, 9: y1}
    return xs[align], ys[align]


# Sampled frames per label run before a force-keep flush. Bounds worst-case
# label-detection latency and the OCR floor for content whose 64-bit hash
# never moves past the similarity threshold.
LABEL_HASH_MAX_RUN = 20


def dhash64(img: np.ndarray[Any, Any]) -> int:
    """64-bit difference hash of an image array (gradient signature).

    Grayscale -> 9x8 bilinear resize -> 8 horizontal brightness comparisons
    per row -> 64 bits (1 when the left pixel is brighter). Gradient-based,
    so global brightness shifts (fades) barely move the hash.
    """
    if img.ndim == 3:
        pil: Image.Image = Image.fromarray(img.astype(np.uint8), mode="RGB")
    else:
        pil = Image.fromarray(img.astype(np.uint8))
    small = np.asarray(
        pil.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.uint8
    ).reshape(8, 9)
    bits = (small[:, :8] > small[:, 1:9]).astype(np.uint8).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes (py39: no int.bit_count)."""
    return bin(a ^ b).count("1")


class LabelFrameDedup:
    """Pre-OCR frame dedup for label bands via last-kept-reference dHash.

    A frame is skipped when its hash is within ``threshold`` bits of the last
    KEPT representative (comparing against the kept reference — not the
    previous frame — so slow fade-ins still accumulate a detectable delta).
    Every skipped frame is folded into the representative's weight so
    confirmation counting recovers the skipped frames. Runs force-flush after
    ``max_run`` sampled frames so static content periodically re-enters OCR.
    """

    def __init__(
        self, threshold: int = 4, max_run: int = LABEL_HASH_MAX_RUN
    ) -> None:
        self.threshold = threshold
        self.max_run = max(1, max_run)
        self.weights: dict[int, int] = {}
        self.skipped = 0
        self._prev_hash: dict[int, int] = {}
        self._rep: dict[int, int] = {}
        self._run_len: dict[int, int] = {}

    def consider(
        self, zone_idx: int, frame_idx: int, img: np.ndarray[Any, Any]
    ) -> bool:
        """True = keep the frame for OCR; False = redundant, skip it."""
        h = dhash64(img)
        prev = self._prev_hash.get(zone_idx)
        if prev is None or hamming_distance(h, prev) > self.threshold:
            self._flush(zone_idx)
            self._start(zone_idx, frame_idx, h)
            return True
        self._run_len[zone_idx] += 1
        if self._run_len[zone_idx] >= self.max_run:
            self._flush(zone_idx)
            self._start(zone_idx, frame_idx, h)
            return True
        self.skipped += 1
        return False

    def finish(self) -> None:
        """Flush all pending runs (call once the scan is complete)."""
        for zone_idx in list(self._rep):
            self._flush(zone_idx)
        self._prev_hash.clear()

    def _start(self, zone_idx: int, frame_idx: int, h: int) -> None:
        self._prev_hash[zone_idx] = h
        self._rep[zone_idx] = frame_idx
        self._run_len[zone_idx] = 1

    def _flush(self, zone_idx: int) -> None:
        rep = self._rep.pop(zone_idx, None)
        if rep is not None:
            self.weights[rep] = self._run_len.pop(zone_idx, 1)


def resolve_label_overlaps(
    items: list[dict[str, Any]], frame_height: int, margin: float = 4.0
) -> None:
    """Shift colliding label events apart vertically (in place).

    Each item needs start_ms/end_ms and a frame-coordinate bbox
    (fx1/fy1/fx2/fy2). Items are ordered by start time; a later item that
    overlaps an already-positioned one in time AND in both axes is pushed
    down until clear. The final position is clamped to the frame bottom —
    if clamping re-introduces an overlap, the overlap is accepted (visible
    beats pushed off-screen).
    """

    def _collides(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return bool(
            a["start_ms"] < b["end_ms"]
            and b["start_ms"] < a["end_ms"]
            and a["fx2"] > b["fx1"]
            and b["fx2"] > a["fx1"]
            and a["fy2"] > b["fy1"]
            and b["fy2"] > a["fy1"]
        )

    ordered = sorted(items, key=lambda it: (it["start_ms"], it["fy1"]))
    for i, cur in enumerate(ordered):
        height = cur["fy2"] - cur["fy1"]
        # Strictly monotonic downward shifts, so this terminates.
        for _ in range(len(ordered)):
            hit = next(
                (other for other in ordered[:i] if _collides(cur, other)), None
            )
            if hit is None:
                break
            cur["fy1"] = float(hit["fy2"]) + margin
            cur["fy2"] = cur["fy1"] + height
        if cur["fy2"] > frame_height:
            cur["fy2"] = float(frame_height)
            cur["fy1"] = max(0.0, cur["fy2"] - height)


def compute_label_zones(
    width: int,
    height: int,
    subtitle_zones: list[dict[str, Any]],
    min_height: int = 32,
    max_bands: int = 4,
) -> list[dict[str, int]]:
    """Compute the label-detection areas: horizontal bands of the frame NOT
    covered by the subtitle crop zone(s) (e.g. people/place names that appear
    above or below the hardcoded subtitle area).

    Subtitles are usually a horizontal band, so the free area splits into
    horizontal bands; every band at least ``min_height`` pixels tall is
    returned (top to-bottom), capped at the ``max_bands`` tallest. Returning
    multiple bands lets labels in a smaller free region (e.g. a top band)
    be detected alongside the dominant one.

    Args:
        width: Video width in pixels.
        height: Video height in pixels.
        subtitle_zones: Validated subtitle zones with 'x_start'/'x_end'/
                       'y_start'/'y_end' (original video coordinates).
        min_height: Minimum band height (px) to be worth scanning.
        max_bands: Maximum number of bands (caps split-branch/OCR cost).

    Returns:
        A list of {'x', 'y', 'w', 'h'} dicts in original video coordinates
        (full width), ordered top to bottom. Empty when nothing qualifies
        (invalid dims, full coverage, or all bands below min_height).
    """
    if not subtitle_zones or width <= 0 or height <= 0:
        return []

    # Merge the y-intervals covered by the subtitle zones.
    covered: list[list[int]] = []
    for z in subtitle_zones:
        y1 = int(max(0, z.get("y_start", 0)))
        y2 = int(min(height, z.get("y_end", height)))
        if y2 > y1:
            covered.append([y1, y2])
    covered.sort()

    merged: list[list[int]] = []
    for y1, y2 in covered:
        if merged and y1 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y2)
        else:
            merged.append([y1, y2])

    # Free horizontal bands: frame edges + gaps between merged intervals.
    bands: list[list[int]] = []
    cursor = 0
    for y1, y2 in merged:
        if y1 > cursor:
            bands.append([cursor, y1])
        cursor = max(cursor, y2)
    if cursor < height:
        bands.append([cursor, height])

    qualifying = [b for b in bands if b[1] - b[0] >= min_height]
    if len(qualifying) > max_bands:
        # Keep the tallest, then restore top-to-bottom order.
        qualifying = sorted(qualifying, key=lambda b: b[1] - b[0], reverse=True)
        qualifying = sorted(qualifying[:max_bands], key=lambda b: b[0])
    return [
        {"x": 0, "y": y1, "w": width, "h": y2 - y1} for y1, y2 in qualifying
    ]


def frame_to_array(frame: av.VideoFrame, fmt: str) -> np.ndarray[Any, Any]:
    """Converts a frame to an array, safely falls back if threads arg is unsupported."""
    if not hasattr(frame_to_array, "supports_threads"):
        frame_to_array.supports_threads = True  # type: ignore

    if frame_to_array.supports_threads:  # type: ignore
        try:
            return frame.to_ndarray(format=fmt, threads=1)
        except TypeError:
            frame_to_array.supports_threads = False  # type: ignore

    return frame.to_ndarray(format=fmt)


def is_on_same_line(word1: PredictedText, word2: PredictedText) -> bool:
    """Checks if two words are on the same line based on vertical overlap."""
    y_min1 = min(p[1] for p in word1.bounding_box)
    y_max1 = max(p[1] for p in word1.bounding_box)
    y_min2 = min(p[1] for p in word2.bounding_box)
    y_max2 = max(p[1] for p in word2.bounding_box)

    midpoint1 = (y_min1 + y_max1) / 2
    midpoint2 = (y_min2 + y_max2) / 2

    return (y_min1 < midpoint2 < y_max1) or (y_min2 < midpoint1 < y_max2)


def is_language_rtl(lang: str) -> bool:
    """Checks if a given language code is written Right-to-Left (RTL)."""
    google_lens_extra_rtl = {"iw", "yi", "dv", "syr"}

    return lang in PADDLEOCR_LANGS["arabic"] or lang in google_lens_extra_rtl


def extract_non_chinese_segments(text: str) -> list[tuple[str, str]]:
    """Extracts non chinese segments out of the detected text for post processing."""
    segments: list[tuple[str, str]] = []
    current_segment = ""

    def is_chinese(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"

    for char in text:
        if is_chinese(char):
            if current_segment:
                segments.append(("non_chinese", current_segment))
                current_segment = ""
            segments.append(("chinese", char))
        else:
            current_segment += char

    if current_segment:
        segments.append(("non_chinese", current_segment))

    return segments


def find_executable(program_name: str) -> str:
    """Finds an executable inside a directory starting with the program name."""
    program_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ext = ".exe" if sys.platform == "win32" else ".bin"
    executable_name = f"{program_name}{ext}"

    for entry in os.listdir(program_dir):
        if entry.lower().startswith(f"{program_name.lower()}-"):
            path = os.path.join(program_dir, entry, executable_name)
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        f"Could not find {executable_name} in any folder starting with '{program_name}'"
    )


def resolve_model_dirs(lang: str, use_server_model: bool) -> tuple[str, str, str]:
    """Resolves the model directory for the specified language and mode."""
    program_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_path = os.path.join(program_dir, "PaddleOCR.PP-OCRv6.support.files")
    det_path = os.path.join(base_path, "det")
    rec_path = os.path.join(base_path, "rec")
    cls_path = os.path.join(base_path, "cls", "PP-LCNet_x1_0_textline_ori")
    # The PP-OCRv6 support archive names the tiers "small" (lightweight) and
    # "medium" (higher accuracy) — unlike PP-OCRv5's "mobile"/"server".
    mode = "medium" if use_server_model else "small"

    # DET
    if lang == "ka":
        det_sub = "PP-OCRv3_mobile_det"
    else:
        det_sub = f"PP-OCRv6_{mode}_det"

    # REC
    # PP-OCRv6 supports ch, chinese_cht, japan, en, and all latin languages with a single unified model
    if lang in ("ch", "chinese_cht", "japan", "en") or lang in PADDLEOCR_LANGS["latin"]:
        rec_sub = f"PP-OCRv6_{mode}_rec"
    elif lang in PADDLEOCR_LANGS["arabic"]:
        rec_sub = "arabic_PP-OCRv5_mobile_rec"
    elif lang in PADDLEOCR_LANGS["eslav"]:
        rec_sub = "eslav_PP-OCRv5_mobile_rec"
    elif lang in PADDLEOCR_LANGS["cyrillic"]:
        rec_sub = "cyrillic_PP-OCRv5_mobile_rec"
    elif lang in PADDLEOCR_LANGS["devanagari"]:
        rec_sub = "devanagari_PP-OCRv5_mobile_rec"
    elif lang in ("korean", "th", "el", "te", "ta"):
        rec_sub = f"{lang}_PP-OCRv5_mobile_rec"
    elif lang == "ka":
        rec_sub = "ka_PP-OCRv3_mobile_rec"

    return (os.path.join(det_path, det_sub), os.path.join(rec_path, rec_sub), cls_path)


def perform_hardware_check(paddleocr_path: str, use_gpu: bool) -> None:
    """Checks if the current system supports the hardware requirements."""
    error_prefix = "Unsupported Hardware Error:"
    warning_prefix = "Hardware Check Warning:"

    def has_avx() -> bool:
        # CPUID leaf 1: Check AVX and OSXSAVE flags
        _, _, ecx, _ = cpuid(1)
        osxsave = bool(ecx & (1 << 27))
        avx = bool(ecx & (1 << 28))

        # OS support check: XGETBV for YMM registers
        ymm_supported = False
        if osxsave:
            try:
                xcr0 = xgetbv(0)
                ymm_supported = (xcr0 & 0b110) == 0b110
            except Exception:
                ymm_supported = False

        return avx and ymm_supported

    def check_cpu() -> None:
        try:
            if not has_avx():
                raise SystemExit(
                    f"{error_prefix} CPU or Operating System does not support the AVX instruction set, which is required."
                )
        except Exception as e:
            print(
                f"{warning_prefix} Could not determine CPU AVX support due to an error: {e}. Functionality is uncertain.",
                flush=True,
            )

    CUDA_COMPATIBILITY_MAP = {
        "CUDA-11.8": (6.1, 8.9) if sys.platform == "win32" else (6.0, 8.9),
        "CUDA-12.9": (7.5, 12.0),
    }

    CUDA_DRIVER_MAP = {
        "CUDA-11.8": "451.22" if sys.platform == "win32" else "450.36.06",
        "CUDA-12.9": "527.41" if sys.platform == "win32" else "525.60.13",
    }

    def parse_version(v_str: str) -> tuple[int, ...]:
        return tuple(map(int, v_str.split(".")))

    def check_gpu() -> None:
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=driver_version,compute_cap",
                "--format=csv,noheader",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding="utf-8"
            )

            first_gpu_info = result.stdout.strip().split("\n")[0]
            if not first_gpu_info:
                raise SystemExit(
                    f"{error_prefix} GPU mode enabled, but 'nvidia-smi' returned no GPU info."
                )

            driver_version_str, compute_cap_str = [
                item.strip() for item in first_gpu_info.split(",")
            ]
            compute_capability = float(compute_cap_str)

            detected_cuda_version = next(
                (v for v in CUDA_COMPATIBILITY_MAP if v in paddleocr_path), None
            )

            if detected_cuda_version:
                # Check Compute Capability
                min_cc, max_cc = CUDA_COMPATIBILITY_MAP[detected_cuda_version]
                if not (min_cc <= compute_capability <= max_cc):
                    raise SystemExit(
                        f"{error_prefix} GPU compute capability is {compute_capability}, but this build "
                        f"({detected_cuda_version}) requires a value between {min_cc} and {max_cc}."
                    )

                # Check NVIDIA Driver Version
                required_driver = CUDA_DRIVER_MAP[detected_cuda_version]
                if parse_version(driver_version_str) < parse_version(required_driver):
                    raise SystemExit(
                        f"{error_prefix} NVIDIA driver version is {driver_version_str}, but this build "
                        f"({detected_cuda_version}) requires version {required_driver} or newer."
                    )

        except Exception as e:
            print(
                f"{warning_prefix} Could not determine GPU support due to an error: {e}. Functionality is uncertain.",
                flush=True,
            )

    check_cpu()

    build_folder_name = os.path.basename(os.path.dirname(paddleocr_path))

    if use_gpu and "GPU" in build_folder_name.upper():
        check_gpu()


def read_pipe(pipe: IO[str], output_list: list[str]) -> None:
    """Reads lines from a pipe and appends them to a list."""
    try:
        for line in iter(pipe.readline, ""):
            output_list.append(line)
    finally:
        pipe.close()


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in result.stdout
        else:
            if os.path.exists(f"/proc/{pid}"):
                return True
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return False


def is_running_in_container() -> bool:
    """Check if the app is running inside a Docker container."""
    return os.path.exists("/.dockerenv")


def create_clean_temp_dir() -> str:
    """Cleans up orphaned temporary directories from previous crashed runs and creates a fresh one for the current process."""
    current_pid = os.getpid()
    temp_prefix = f"videocr_temp_{current_pid}_"
    base_temp = tempfile.gettempdir()

    for name in os.listdir(base_temp):
        if name.startswith("videocr_temp_"):
            temp_path = os.path.join(base_temp, name)
            try:
                match = re.match(r"videocr_temp_(\d+)_", name)
                if match:
                    dir_pid = int(match.group(1))

                    if dir_pid == current_pid:
                        continue

                    if os.path.isdir(temp_path):
                        if not is_process_running(dir_pid):
                            shutil.rmtree(temp_path, ignore_errors=True)
            except Exception as e:
                print(f"Could not remove leftover temp dir '{name}': {e}", flush=True)

    return tempfile.mkdtemp(prefix=temp_prefix)


def log_error(message: str, log_name: str = "error_log.txt") -> str:
    """Saves errors to a log file."""
    if sys.platform == "win32":
        log_dir = os.path.join(
            os.getenv("LOCALAPPDATA") or os.path.expanduser("~"), "VideOCR"
        )
    else:
        log_dir = os.path.join(os.path.expanduser("~"), ".config", "VideOCR")

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_name)
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")

    return log_path


def prepare_stitch_batch(
    batch: list[Any],
    counter: int,
    zone_idx: int,
    prefix: str,
    out_dir: str,
    target_map: dict[str, Any],
    max_width: int,
    grid_spacing: int,
    zero_pad_length: int,
) -> tuple[str, int, int, list[tuple[Any, int, int]]]:
    """Calculates grid dimensions and maps coordinates for a batch. Returns queue arguments."""
    h, w = batch[0]["img"].shape[:2]
    cols = max(1, (max_width + grid_spacing) // (w + grid_spacing))

    actual_cols = min(len(batch), cols)
    actual_rows = (len(batch) + cols - 1) // cols
    canvas_w = actual_cols * w + (actual_cols - 1) * grid_spacing
    canvas_h = actual_rows * h + (actual_rows - 1) * grid_spacing

    mapping: list[dict[str, Any]] = []
    draw_instructions: list[tuple[Any, int, int]] = []

    filename = f"{prefix}_{counter:0{zero_pad_length}d}_zone{zone_idx}.jpg"
    filepath = os.path.join(out_dir, filename)

    for i, item in enumerate(batch):
        row_idx = i // cols
        col_idx = i % cols
        x_offset = col_idx * (w + grid_spacing)
        y_offset = row_idx * (h + grid_spacing)

        draw_instructions.append((item["img"], x_offset, y_offset))

        mapping.append(
            {
                "grid_file": filepath,
                "frame_idx": item["frame_idx"],
                "zone_idx": zone_idx,
                "x": x_offset,
                "y": y_offset,
                "w": w,
                "h": h,
            }
        )

    target_map[filename] = mapping

    return filepath, canvas_w, canvas_h, draw_instructions


def get_batch_limit(
    w: int, h: int, max_width: int, max_height: int, padding: int
) -> int:
    """Calculates the maximum number of frames that can fit in a stitched grid."""
    cols = max(1, (max_width + padding) // (w + padding))
    rows = max(1, (max_height + padding) // (h + padding))
    return cols * rows


def unstitch_polygon(
    poly: list[list[float]], mapping: list[dict[str, Any]]
) -> list[tuple[list[list[float]], dict[str, Any]]]:
    """Finds all grid frames a polygon intersects and clips coordinates to their local space."""
    poly_min_x = min(pt[0] for pt in poly)
    poly_max_x = max(pt[0] for pt in poly)
    poly_min_y = min(pt[1] for pt in poly)
    poly_max_y = max(pt[1] for pt in poly)

    intersections = []

    for m in mapping:
        cell_min_x = m["x"]
        cell_max_x = m["x"] + m["w"]
        cell_min_y = m["y"]
        cell_max_y = m["y"] + m["h"]

        if (
            poly_min_x < cell_max_x
            and poly_max_x > cell_min_x
            and poly_min_y < cell_max_y
            and poly_max_y > cell_min_y
        ):

            inter_min_x = max(poly_min_x, cell_min_x)
            inter_max_x = min(poly_max_x, cell_max_x)
            inter_min_y = max(poly_min_y, cell_min_y)
            inter_max_y = min(poly_max_y, cell_max_y)

            if inter_max_x - inter_min_x < 5 or inter_max_y - inter_min_y < 5:
                continue

            local_min_x = inter_min_x - m["x"]
            local_max_x = inter_max_x - m["x"]
            local_min_y = inter_min_y - m["y"]
            local_max_y = inter_max_y - m["y"]

            adjusted_poly = [
                [local_min_x, local_min_y],
                [local_max_x, local_min_y],
                [local_max_x, local_max_y],
                [local_min_x, local_max_y],
            ]
            intersections.append((adjusted_poly, m))

    if not intersections:
        cx = sum(pt[0] for pt in poly) / 4.0
        cy = sum(pt[1] for pt in poly) / 4.0

        best_m = min(
            mapping,
            key=lambda m: (cx - (m["x"] + m["w"] / 2.0)) ** 2
            + (cy - (m["y"] + m["h"] / 2.0)) ** 2,
        )

        adjusted_poly = [[pt[0] - best_m["x"], pt[1] - best_m["y"]] for pt in poly]
        intersections.append((adjusted_poly, best_m))

    return intersections


def stream_cli_process(args: list[str], log_name: str) -> Iterator[str]:
    """Executes a CLI process, yields its stdout lines, and handles errors/logging."""
    cli_env = os.environ.copy()
    cli_env["PYTHONIOENCODING"] = "utf-8"
    cli_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=cli_env,
        bufsize=1,
    )

    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(
        target=read_pipe, args=(process.stderr, stderr_lines)
    )
    stderr_thread.start()

    stdout_lines: list[str] = []
    is_interrupted = False

    try:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            stdout_lines.append(line)
            yield line
    except KeyboardInterrupt:
        is_interrupted = True
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        if process.stdout:
            process.stdout.close()
        exit_code = process.wait()
        stderr_thread.join()

        if exit_code != 0 and not is_interrupted:
            full_stdout = "".join(stdout_lines)
            full_stderr = "".join(stderr_lines)
            command_str = " ".join(args)
            log_message = (
                f"Process failed with exit code {exit_code}.\n"
                f"Command: {command_str}\n\n"
                f"--- STDOUT ---\n{full_stdout}\n\n"
                f"--- STDERR ---\n{full_stderr}\n"
            )
            log_file_path = log_error(log_message, log_name=log_name)
            print(
                f"\nError: Process failed. See the log file for technical details:\n{log_file_path}",
                flush=True,
            )
            sys.exit(1)


def get_line_rects(polys: list[list[list[float]]]) -> list[list[float]]:
    """Converts a list of polygons into merged line bounding boxes [min_x, min_y, max_x, max_y]."""
    if not polys:
        return []

    rects: list[list[float]] = []
    for poly in polys:
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        rects.append([min(xs), min(ys), max(xs), max(ys)])

    rects.sort(key=lambda r: r[1])

    merged_lines: list[list[float]] = []
    for r in rects:
        if not merged_lines:
            merged_lines.append(r)
        else:
            last = merged_lines[-1]
            overlap_top = max(last[1], r[1])
            overlap_bottom = min(last[3], r[3])

            if overlap_top < overlap_bottom:
                merged_lines[-1] = [
                    min(last[0], r[0]),
                    min(last[1], r[1]),
                    max(last[2], r[2]),
                    max(last[3], r[3]),
                ]
            else:
                merged_lines.append(r)

    return merged_lines


def are_rect_lists_similar(
    rects1: list[list[float]], rects2: list[list[float]], tolerance: float
) -> bool:
    """Compares two lists of bounding boxes to see if they are spatially similar within a tolerance."""
    if len(rects1) != len(rects2):
        return False

    for r1, r2 in zip(rects1, rects2):
        w1, h1 = r1[2] - r1[0], r1[3] - r1[1]
        w2, h2 = r2[2] - r2[0], r2[3] - r2[1]
        cx1, cy1 = r1[0] + w1 / 2, r1[1] + h1 / 2
        cx2, cy2 = r2[0] + w2 / 2, r2[1] + h2 / 2

        max_w, max_h = max(w1, w2, 1), max(h1, h2, 1)
        if not (
            abs(w1 - w2) / max_w <= tolerance
            and abs(h1 - h2) / max_h <= tolerance
            and abs(cx1 - cx2) / max_w <= tolerance
            and abs(cy1 - cy2) / max_h <= tolerance
        ):
            return False

    return True


def load_grid(g_file: str) -> tuple[str, Any]:
    """Loads a grid image."""
    return g_file, np.array(Image.open(g_file))


def process_ssim_group(
    union_rects: list[list[float]],
    group_frames: list[tuple[int, list[list[float]], float, dict[str, Any]]],
    loaded_grids: dict[str, Any],
    ssim_threshold: float,
    frame_weights: dict[int, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Processes a group for SSIM, keeping the frame with the highest detection score per contiguous block.

    ``frame_weights`` optionally maps frame_idx -> number of original sampled
    frames it represents (set by pre-OCR label-frame dedup), so a survivor's
    weight sums represented frames instead of counting stitched frames.
    """
    local_surviving_items: list[dict[str, Any]] = []
    current_similar_batch: list[dict[str, Any]] = []
    prev_crops: list[Any] = []

    def _batch_weight(items: list[dict[str, Any]]) -> int:
        if frame_weights is None:
            return len(items)
        return sum(frame_weights.get(it["frame_idx"], 1) for it in items)

    for i, (_, _, det_score, m) in enumerate(group_frames):
        grid_img = loaded_grids[m["grid_file"]]
        img = grid_img[m["y"] : m["y"] + m["h"], m["x"] : m["x"] + m["w"]]
        h, w = img.shape[:2]

        current_crops: list[Any] = []
        for rect in union_rects:
            cx1, cy1 = max(0, int(rect[0])), max(0, int(rect[1]))
            cx2, cy2 = min(w, int(rect[2])), min(h, int(rect[3]))
            current_crops.append(img[cy1:cy2, cx1:cx2])

        item_dict = {
            "img": img.copy(),
            "frame_idx": m["frame_idx"],
            "det_score": det_score,
        }

        if i == 0:
            prev_crops = current_crops
            current_similar_batch.append(item_dict)
            continue

        all_lines_match = True
        for prev_c, curr_c in zip(prev_crops, current_crops):
            if prev_c.size == 0 or curr_c.size == 0:
                all_lines_match = False
                break
            score = fast_ssim.ssim(prev_c, curr_c, data_range=255)
            if score <= ssim_threshold:
                all_lines_match = False
                break

        if all_lines_match:
            current_similar_batch.append(item_dict)
        else:
            best_item = max(current_similar_batch, key=lambda x: x["det_score"])
            # Weight before the frame_idx rewrite below — best_item may not be
            # batch[0], and _batch_weight reads each item's own frame_idx.
            weight = _batch_weight(current_similar_batch)
            best_item["frame_idx"] = current_similar_batch[0]["frame_idx"]
            best_item["weight"] = weight
            local_surviving_items.append(best_item)

            current_similar_batch = [item_dict]
            prev_crops = current_crops

    if current_similar_batch:
        best_item = max(current_similar_batch, key=lambda x: x["det_score"])
        weight = _batch_weight(current_similar_batch)
        best_item["frame_idx"] = current_similar_batch[0]["frame_idx"]
        best_item["weight"] = weight
        local_surviving_items.append(best_item)

    local_deleted = len(group_frames) - len(local_surviving_items)

    return local_surviving_items, local_deleted
