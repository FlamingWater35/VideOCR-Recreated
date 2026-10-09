from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, NamedTuple, cast

import av
import fast_ssim  # type: ignore
import numpy as np
import wordninja_enhanced as wordninja  # type: ignore
from PIL import Image

from . import utils
from .models import PredictedFrames, PredictedSubtitle
from .pyav_adapter import Capture, get_video_properties


def _label_text_norm(text: str) -> str:
    """Normalize a label OCR text for similarity comparison (spaces removed)."""
    return (text or "").replace(" ", "")


def _is_single_ascii_label(text: str) -> bool:
    """True if the label is a single ASCII character/digit (OCR noise)."""
    t = (text or "").strip()
    if len(t) != 1:
        return False
    return t.isascii() and (t.isalnum())


class _LabelLine(NamedTuple):
    """One OCR'd label line: text, centroid, mean confidence, bbox."""

    text: str
    cx: float
    cy: float
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Video:
    path: str
    lang: str
    use_fullframe: bool
    paddleocr_path: str
    google_lens_path: str
    post_processing: bool
    det_model_dir: str
    rec_model_dir: str
    cls_model_dir: str
    duration_ms: int
    height: int
    width: int
    pred_frames_zone1: list[PredictedFrames]
    pred_frames_zone2: list[PredictedFrames]
    pred_subs: list[PredictedSubtitle]
    validated_zones: list[dict[str, Any]]
    label_zones: list[dict[str, Any]]
    label_frames: list[PredictedFrames]
    frame_timestamps: dict[int, float]
    start_time_offset_ms: float
    avg_frame_duration_ms: float
    fps: float

    def __init__(
        self,
        path: str,
        paddleocr_path: str,
        det_model_dir: str,
        rec_model_dir: str,
        cls_model_dir: str,
        google_lens_path: str,
    ) -> None:
        self.path = path
        self.paddleocr_path = paddleocr_path
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.google_lens_path = google_lens_path
        self.frame_timestamps = {}
        self.start_time_offset_ms = 0.0
        self.avg_frame_duration_ms = 0.0
        self.label_zones = []
        self.label_frames = []
        self._label_frame_weights: dict[int, int] = {}
        self._label_hash_weights: dict[int, int] = {}
        self._label_hash_threshold = 4
        self._label_dedup_enabled = True
        self._label_min_zone_height = 32
        # Pre-run_ocr placeholder only: run_ocr always assigns
        # `label_lang_code or self.lang` (never "") before any consumer reads
        # it (PredictedFrames construction, split-recognition passes).
        # _merge_label_frames/_render_ass never touch _label_lang, and
        # get_subtitles is only called after run_ocr, so "" is unreachable
        # as an OCR language.
        self._label_lang = ""
        self._label_split_lang = False
        self._label_model_dirs: dict[str, str] | None = None
        self._label_style: dict[str, Any] = {
            "font": "Arial",
            "fontsize": 22,
            "primary": "&H00FFFFFF",
            "outline": "&H00000000",
            "alignment": "an7",
        }
        self.label_time_start_ms = 0.0
        self.label_time_end_ms: float | None = None
        props = get_video_properties(self.path)
        self.height = props["height"]
        self.width = props["width"]
        self.duration_ms = props["duration_ms"]
        self.start_time_offset_ms = props["start_time_offset_ms"]
        self.fps = float(props.get("fps", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # Frame-duration helpers (fix 8.3 / 8.4)
    # ------------------------------------------------------------------
    def _frame_duration_ms(self) -> float:
        """Best-effort duration of one video frame in milliseconds.

        Derived from the measured average frame duration when available,
        otherwise from the nominal FPS, falling back to 33 ms (~30 fps).
        """
        if self.avg_frame_duration_ms and self.avg_frame_duration_ms > 0:
            return self.avg_frame_duration_ms
        if self.fps and self.fps > 0:
            return 1000.0 / self.fps
        return 33.0

    def _estimate_frame_timestamp_ms(self, frame_index: int) -> float | None:
        """Estimate a frame's container timestamp when the map is sparse.

        Extrapolates from the earliest known frame using the average frame
        duration, so an unknown frame is still checked against the label time
        window instead of leaking through unconditionally.
        """
        if not self.frame_timestamps:
            return None
        ref_index = min(self.frame_timestamps.keys())
        ref_ts = self.frame_timestamps[ref_index]
        return ref_ts + (frame_index - ref_index) * self._frame_duration_ms()

    def _run_easyocr_directml_ffmpeg_hw_scan(
        self,
        temp_dir: str,
        target_end_str: str,
        ssim_threshold_ratio: float,
        subtitle_position: str,
        brightness_threshold: int | None,
        frames_to_skip: int,
        max_stitch_width: int,
        max_stitch_height: int,
        directml_recognition_mode: str,
        use_gpu: bool,
        ocr_engine: str,
        conf_threshold_ratio: float,
        lang: str,
        normalize_to_simplified_chinese: bool,
        step1_start: float,
        perf_total_start: float,
        onnx_directml_tuning: str = "balanced",
        benchmark_compare_engine: bool = False,
        benchmark_compare_sample_grids: int = 3,
        target_start_ms: float = 0.0,
        target_end_ms: float | None = None,
    ) -> bool:
        """Experimental AMD path: FFmpeg D3D11VA decode + raw cropped frames.

        Supports the common single subtitle crop workflow plus the optional
        label-detection zone. When a label zone is present, FFmpeg emits a
        second output (via a ``split`` filter) with the cropped/scaled label
        frames written to a temporary raw file, so label detection keeps
        hardware decode instead of falling back to PyAV.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print(
                "Warning: FFmpeg was not found on PATH. Falling back to PyAV CPU frame scan.",
                flush=True,
            )
            return False
        if len(self.validated_zones) != 1:
            print(
                "Warning: FFmpeg D3D11VA prototype currently supports one OCR zone. Falling back to PyAV CPU frame scan.",
                flush=True,
            )
            return False
        if self.fps <= 0:
            print(
                "Warning: Could not determine input FPS for FFmpeg D3D11VA timestamp mapping. Falling back to PyAV CPU frame scan.",
                flush=True,
            )
            return False

        label_zones = self.label_zones
        use_label = bool(label_zones)

        z = self.validated_zones[0]
        target_w = int(z["w"])
        target_h = int(z["h"])
        frame_bytes = target_w * target_h * 3

        GRID_SPACING = 10
        FILENAME_ZERO_PADDING = 8

        label_raw_path = ""
        label_target_w = label_frame_bytes = 0
        label_batches: dict[int, list[Any]] = {}
        label_batch_limits: dict[int, int] = {}
        if use_label:
            # All bands share target_w (built with one scale ratio), so the
            # FFmpeg label output is the bands vstacked into one composite.
            label_target_w = int(label_zones[0]["target_w"])
            total_label_h = sum(int(b["target_h"]) for b in label_zones)
            label_frame_bytes = label_target_w * total_label_h * 3
            label_raw_path = os.path.join(temp_dir, "label_raw.rgb")
            for k, band in enumerate(label_zones):
                label_batches[2 + k] = []
                label_batch_limits[2 + k] = utils.get_batch_limit(
                    label_target_w,
                    int(band["target_h"]),
                    max_stitch_width,
                    max_stitch_height,
                    GRID_SPACING,
                )

        modulo = frames_to_skip + 1
        det_stitched_dir = os.path.join(temp_dir, "det_stitched")
        os.makedirs(det_stitched_dir, exist_ok=True)
        det_stitch_map: dict[str, list[dict[str, Any]]] = {}
        det_counter = 0
        det_batch: list[Any] = []
        batch_limit = utils.get_batch_limit(
            target_w, target_h, max_stitch_width, max_stitch_height, GRID_SPACING
        )
        prev_sample = None
        dml_frame_filter = None
        if ssim_threshold_ratio < 1:
            try:
                requested_index = os.environ.get(
                    "VIDEOCR_DIRECTML_DEVICE_INDEX", ""
                ).strip()
                device_index = int(requested_index) if requested_index else None
                from .directml_frame_filter import DirectMLFrameSimilarityFilter

                dml_frame_filter = DirectMLFrameSimilarityFilter(
                    ssim_threshold_ratio, device_index=device_index
                )
                print(
                    f"Step 1 frame scan mode: FFmpeg D3D11VA hardware decode + DirectML SSIM on adapter {dml_frame_filter.device_label}.",
                    flush=True,
                )
            except Exception as e:
                dml_frame_filter = None
                print(
                    f"Warning: DirectML SSIM failed in FFmpeg mode; using CPU SSIM after hardware decode: {e}",
                    flush=True,
                )
        else:
            print(
                "Step 1 frame scan mode: FFmpeg D3D11VA hardware decode without SSIM filtering.",
                flush=True,
            )

        def save_stitch_batch(
            batch: list[Any], counter: int, batch_zone_idx: int
        ) -> int:
            frame_path, canvas_w, canvas_h, draw_instructions = (
                utils.prepare_stitch_batch(
                    batch,
                    counter,
                    batch_zone_idx,
                    "det_stitched",
                    det_stitched_dir,
                    det_stitch_map,
                    max_stitch_width,
                    GRID_SPACING,
                    FILENAME_ZERO_PADDING,
                )
            )
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for img, x, y in draw_instructions:
                h, w = img.shape[:2]
                canvas[y : y + h, x : x + w] = img
            Image.fromarray(canvas).save(frame_path, quality=80)
            return counter + 1

        if use_label:
            select_part = f"select=not(mod(n\\,{modulo}))"
            if len(label_zones) == 1:
                # Single band: keep the original two-output graph as-is.
                fc = (
                    f"[0:v]{select_part},split=2[s1][s2];"
                    f"[s1]crop={z['crop_str']},scale={z['scale_str']},format=rgb24[sub];"
                    f"[s2]crop={label_zones[0]['crop_str']},scale={label_zones[0]['scale_str']},format=rgb24[label]"
                )
            else:
                # Multiple bands: split once more per band, scale each to the
                # shared target width, then vstack into the single [label]
                # output (still exactly two FFmpeg outputs total).
                k = len(label_zones)
                fc = (
                    f"[0:v]{select_part},split={k + 1}[s0]"
                    + "".join(f"[l{i}]" for i in range(k))
                    + ";"
                    + f"[s0]crop={z['crop_str']},scale={z['scale_str']},format=rgb24[sub];"
                )
                for i, band in enumerate(label_zones):
                    fc += (
                        f"[l{i}]crop={band['crop_str']},scale={band['scale_str']},"
                        f"format=rgb24[ls{i}];"
                    )
                fc += (
                    "".join(f"[ls{i}]" for i in range(k))
                    + f"vstack=inputs={k}[label]"
                )
            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-hwaccel",
                "d3d11va",
                "-ss",
                str(target_start_ms / 1000.0),
                "-i",
                self.path,
                "-filter_complex",
                fc,
                "-an",
                "-sn",
            ]
            if target_end_ms is not None and target_end_ms > target_start_ms:
                cmd += ["-t", str((target_end_ms - target_start_ms) / 1000.0)]
            cmd += [
                "-map",
                "[sub]",
                "-fps_mode",
                "vfr",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
                "-map",
                "[label]",
                "-fps_mode",
                "vfr",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                label_raw_path,
            ]
            vf_note = fc
        else:
            vf = f"select=not(mod(n\\,{modulo})),crop={z['crop_str']},scale={z['scale_str']},format=rgb24"
            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-hwaccel",
                "d3d11va",
                "-ss",
                str(target_start_ms / 1000.0),
                "-i",
                self.path,
                "-vf",
                vf,
                # -vsync 0 was removed in newer ffmpeg; use -fps_mode vfr (equivalent
                # for raw-video pipe output: every selected frame is preserved).
                "-fps_mode",
                "vfr",
                "-an",
                "-sn",
            ]
            if target_end_ms is not None and target_end_ms > target_start_ms:
                cmd += ["-t", str((target_end_ms - target_start_ms) / 1000.0)]
            cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            vf_note = vf

        print("Starting FFmpeg D3D11VA hardware decode prototype...", flush=True)
        print(f"[Perf] FFmpeg filter: {vf_note}", flush=True)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4
        )
        stderr_lines: list[str] = []

        def stderr_reader() -> None:
            assert proc.stderr is not None
            for raw_line in iter(proc.stderr.readline, b""):
                try:
                    line = raw_line.decode(errors="replace").strip()
                except Exception:
                    line = str(raw_line)
                if line:
                    stderr_lines.append(line)
                    if len(stderr_lines) > 40:
                        del stderr_lines[:20]

        err_thread = threading.Thread(target=stderr_reader, daemon=True)
        err_thread.start()

        ocr_end = 0
        kept_frames = 0
        skipped_similar = 0
        try:
            assert proc.stdout is not None
            while True:
                data = proc.stdout.read(frame_bytes)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise RuntimeError(
                        f"FFmpeg returned a partial raw frame ({len(data)} / {frame_bytes} bytes)."
                    )
                frame_index = ocr_end
                # With input seeking (-ss), decoded frame 0 corresponds to
                # target_start_ms (container time), not the video's own start.
                timestamp_ms = target_start_ms + (
                    frame_index * modulo * 1000.0 / self.fps
                )
                self.frame_timestamps[frame_index] = timestamp_ms
                curr_str = utils.get_srt_timestamp_from_ms(
                    timestamp_ms - self.start_time_offset_ms
                ).split(",")[0]
                # Stop if we have reached the requested end time.
                if target_end_ms is not None and timestamp_ms > target_end_ms:
                    break
                img = (
                    np.frombuffer(data, dtype=np.uint8)
                    .reshape((target_h, target_w, 3))
                    .copy()
                )
                if brightness_threshold is not None:
                    gray = (
                        (
                            img[..., 0].astype(np.uint16) * 77
                            + img[..., 1].astype(np.uint16) * 150
                            + img[..., 2].astype(np.uint16) * 29
                        )
                        >> 8
                    ).astype(np.uint8)
                    mask = gray > brightness_threshold
                    img *= mask[..., None]
                keep_frame = True
                if ssim_threshold_ratio < 1:
                    w = img.shape[1]
                    if subtitle_position == "center":
                        w_margin = int(w * 0.35)
                        sample = img[:, w_margin : w - w_margin]
                    elif subtitle_position == "left":
                        sample = img[:, : int(w * 0.3)]
                    elif subtitle_position == "right":
                        sample = img[:, int(w * 0.7) :]
                    elif subtitle_position == "any":
                        sample = img
                    else:
                        raise ValueError(
                            f"Invalid subtitle_position: {subtitle_position}"
                        )
                    if prev_sample is not None:
                        if dml_frame_filter is not None:
                            keep_frame = not dml_frame_filter.is_similar(
                                prev_sample, sample
                            )
                        else:
                            keep_frame = not (
                                fast_ssim.ssim(prev_sample, sample, data_range=255)
                                > ssim_threshold_ratio
                            )
                    prev_sample = sample
                if keep_frame:
                    det_batch.append({"img": img, "frame_idx": frame_index})
                    kept_frames += 1
                    if len(det_batch) >= batch_limit:
                        det_counter = save_stitch_batch(det_batch, det_counter, 0)
                        det_batch = []
                else:
                    skipped_similar += 1
                if frame_index % 15 == 0:
                    # Clamp displayed position so it never exceeds the target.
                    if target_end_ms is not None and timestamp_ms > target_end_ms:
                        show_ts = target_end_ms
                    else:
                        show_ts = timestamp_ms
                    curr_str = utils.get_srt_timestamp_from_ms(
                        show_ts - self.start_time_offset_ms
                    ).split(",")[0]
                    print(
                        f"\rStep 1/3: FFmpeg/D3D11VA scan... Current: {curr_str} / {target_end_str}, Sample: {frame_index + 1}",
                        end="",
                        flush=True,
                    )
                    # Also emit the standard progress line the GUI parses for the
                    # smooth progress bar / ETA.
                    print(
                        f"\nStep {1}/3: Processing video... Current: {curr_str} / {target_end_str}, Frame: {frame_index + 1}",
                        flush=True,
                    )
                ocr_end += 1
        finally:
            if proc.stdout:
                proc.stdout.close()
            return_code = proc.wait()
            err_thread.join(timeout=2)
            print(flush=True)
            if return_code != 0:
                recent = "\n".join(stderr_lines[-20:])
                raise RuntimeError(
                    f"FFmpeg D3D11VA scan failed with exit code {return_code}. Recent FFmpeg output:\n{recent}"
                )

        if det_batch:
            det_counter = save_stitch_batch(det_batch, det_counter, 0)
            det_batch = []

        # Label zone (fix 6): read the second FFmpeg output and stitch the
        # label frames. No SSIM here, but the optional pre-OCR dHash dedup
        # (label_ssim_dedup) skips redundant frames per band while keeping
        # confirmation weights. Frame indices match the subtitle sampled frames.
        # The output holds all bands vstacked; split each composite frame back
        # into per-band crops so every band stitches under its own zone index.
        if use_label:
            try:
                with open(label_raw_path, "rb") as lf:
                    label_data = lf.read()
            except OSError as e:
                print(
                    f"Warning: could not read FFmpeg label-zone output: {e}", flush=True
                )
                label_data = b""
            num_label_frames = (
                len(label_data) // label_frame_bytes if label_frame_bytes > 0 else 0
            )
            label_frame_count = min(num_label_frames, ocr_end)
            band_heights = [int(b["target_h"]) for b in label_zones]
            row_strides = [label_target_w * th * 3 for th in band_heights]
            row_offsets: list[int] = []
            acc = 0
            for stride in row_strides:
                row_offsets.append(acc)
                acc += stride
            label_dedup: utils.LabelFrameDedup | None = None
            if self._label_dedup_enabled:
                label_dedup = utils.LabelFrameDedup(
                    threshold=self._label_hash_threshold
                )
            for i in range(label_frame_count):
                frame_base = i * label_frame_bytes
                for k, band_height in enumerate(band_heights):
                    zidx = 2 + k
                    off = frame_base + row_offsets[k]
                    img = (
                        np.frombuffer(
                            label_data[off : off + row_strides[k]], dtype=np.uint8
                        )
                        .reshape((band_height, label_target_w, 3))
                        .copy()
                    )
                    if (
                        label_dedup is not None
                        and not label_dedup.consider(zidx, i, img)
                    ):
                        continue
                    label_batches[zidx].append({"img": img, "frame_idx": i})
                    if len(label_batches[zidx]) >= label_batch_limits[zidx]:
                        det_counter = save_stitch_batch(
                            label_batches[zidx], det_counter, zidx
                        )
                        label_batches[zidx] = []
            for k in range(len(label_zones)):
                zidx = 2 + k
                if label_batches[zidx]:
                    det_counter = save_stitch_batch(
                        label_batches[zidx], det_counter, zidx
                    )
                    label_batches[zidx] = []
            if label_dedup is not None:
                label_dedup.finish()
                self._publish_label_hash_weights(label_dedup)
            with contextlib.suppress(OSError):
                os.remove(label_raw_path)

        if ocr_end <= 0 or det_counter <= 0:
            print(
                "Warning: FFmpeg D3D11VA scan produced no OCR grids. Falling back to PyAV CPU frame scan.",
                flush=True,
            )
            return False

        if len(self.frame_timestamps) > 1:
            min_idx = min(self.frame_timestamps.keys())
            max_idx = max(self.frame_timestamps.keys())
            if max_idx > min_idx:
                total_duration = (
                    self.frame_timestamps[max_idx] - self.frame_timestamps[min_idx]
                )
                self.avg_frame_duration_ms = total_duration / (max_idx - min_idx)

        step1_end = time.perf_counter()
        if dml_frame_filter is not None:
            print(
                f"[Perf] DirectML frame scan comparisons: {dml_frame_filter.comparisons}, filtered: {dml_frame_filter.filtered}",
                flush=True,
            )
        print(
            f"[Perf] FFmpeg/D3D11VA sampled frames: {ocr_end}; kept for OCR: {kept_frames}; SSIM-skipped: {skipped_similar}",
            flush=True,
        )

        if ocr_engine == "onnx_directml":
            from .onnx_directml_ocr import run_onnx_directml_on_stitched_images

            run_directml_ocr = run_onnx_directml_on_stitched_images
            directml_label = "ONNX Runtime DirectML OCR"
        else:
            from .easyocr_directml import run_easyocr_on_stitched_images

            run_directml_ocr = run_easyocr_on_stitched_images
            directml_label = "EasyOCR DirectML hybrid OCR"

        total_stitched_frames = sum(
            len(mappings) for mappings in det_stitch_map.values()
        )
        total_grids = len(det_stitch_map)
        avg_frames_per_grid = (
            total_stitched_frames / total_grids if total_grids else 0.0
        )
        print(
            f"Running {directml_label} pass on {total_stitched_frames} filtered frame(s) "
            f"stitched into {total_grids} image grid(s) "
            f"({avg_frames_per_grid:.1f} frame(s)/grid)...",
            flush=True,
        )
        directml_ocr_start = time.perf_counter()
        ocr_outputs = self._run_label_aware_directml_ocr(
            run_directml_ocr,
            det_stitched_dir,
            det_stitch_map,
            self.lang,
            use_gpu,
            directml_recognition_mode,
            ocr_engine,
        )
        directml_ocr_end = time.perf_counter()

        if ocr_engine == "onnx_directml" and benchmark_compare_engine:
            try:
                sample_count = max(
                    1, min(int(benchmark_compare_sample_grids or 3), total_grids)
                )
                sample_names = sorted(det_stitch_map.keys())[:sample_count]
                sample_map = {name: det_stitch_map[name] for name in sample_names}
                sample_dir = os.path.join(temp_dir, "benchmark_compare_grids")
                os.makedirs(sample_dir, exist_ok=True)
                for name in sample_names:
                    shutil.copy2(
                        os.path.join(det_stitched_dir, name),
                        os.path.join(sample_dir, name),
                    )
                print(
                    f"[BenchCompare] Running EasyOCR Hybrid sample on {sample_count} stitched grid(s)...",
                    flush=True,
                )
                from .easyocr_directml import run_easyocr_on_stitched_images

                compare_start = time.perf_counter()
                _ = run_easyocr_on_stitched_images(
                    sample_dir, sample_map, self.lang, use_gpu, "stable"
                )
                compare_elapsed = time.perf_counter() - compare_start
                onnx_per_grid = (
                    (directml_ocr_end - directml_ocr_start) / total_grids
                    if total_grids
                    else 0.0
                )
                easy_per_grid = compare_elapsed / sample_count if sample_count else 0.0
                print(
                    f"[BenchCompare] ONNX full OCR: {directml_ocr_end - directml_ocr_start:.2f}s / {total_grids} grid(s) = {onnx_per_grid:.2f}s/grid",
                    flush=True,
                )
                print(
                    f"[BenchCompare] EasyOCR Hybrid sample: {compare_elapsed:.2f}s / {sample_count} grid(s) = {easy_per_grid:.2f}s/grid; projected full={easy_per_grid * total_grids:.2f}s",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[BenchCompare] EasyOCR-vs-ONNX sample benchmark failed: {e}",
                    flush=True,
                )

        active_frame_coords: set[tuple[int, int]] = set()
        for mappings in det_stitch_map.values():
            for m in mappings:
                active_frame_coords.add((int(m["frame_idx"]), int(m["zone_idx"])))
        frame_predictions_dict: dict[int, dict[int, PredictedFrames]] = {0: {}, 1: {}}
        for k in range(len(label_zones)):
            frame_predictions_dict[2 + k] = {}
        label_conf_ratio = getattr(
            self, "_label_conf_threshold_ratio", conf_threshold_ratio
        )
        for frame_index, zone_index in active_frame_coords:
            ocr_result = ocr_outputs.get((frame_index, zone_index), [])
            pred_data = [ocr_result] if ocr_result else [[]]
            zone_conf = (
                label_conf_ratio if zone_index >= 2 else conf_threshold_ratio
            )
            predicted_frame = PredictedFrames(
                ocr_engine,
                frame_index,
                pred_data,
                zone_conf,
                zone_index,
                self._label_lang if zone_index >= 2 else lang,
                normalize_to_simplified_chinese,
            )
            frame_predictions_dict[zone_index][frame_index] = predicted_frame
        frame_predictions_list: dict[int, list[PredictedFrames]] = {}
        for z_idx in frame_predictions_dict:
            frames = sorted(
                frame_predictions_dict[z_idx].values(), key=lambda f: f.start_index
            )
            if not frames:
                continue
            for i in range(len(frames) - 1):
                frames[i].end_index = frames[i + 1].start_index - 1
            frames[-1].end_index = ocr_end - 1
            frame_predictions_list[z_idx] = frames

        self.pred_frames_zone1 = frame_predictions_list.get(0, [])
        self.pred_frames_zone2 = frame_predictions_list.get(1, [])
        self.label_frames = (
            self._collect_label_frames(frame_predictions_list)
            if use_label
            else []
        )

        total_elapsed = time.perf_counter() - perf_total_start
        print(
            f"[Perf] Step 1 FFmpeg/D3D11VA decode/filter/stitch: {step1_end - step1_start:.2f}s",
            flush=True,
        )
        print(
            f"[Perf] Step 2 {directml_label}: {directml_ocr_end - directml_ocr_start:.2f}s",
            flush=True,
        )
        if ocr_engine == "onnx_directml":
            print(
                f"[Perf] ONNX tuning: {onnx_directml_tuning}; grid target: {max_stitch_width}x{max_stitch_height}",
                flush=True,
            )
        print(
            f"[Perf] Filtered OCR frames: {total_stitched_frames}; stitched grids: {total_grids}; average frames/grid: {avg_frames_per_grid:.1f}",
            flush=True,
        )
        print(
            f"[Perf] Total OCR preparation/runtime before subtitle merge: {total_elapsed:.2f}s",
            flush=True,
        )
        return True

    @staticmethod
    def _resolve_directml_grid_size(
        ocr_engine: str,
        directml_performance_preset: str,
        directml_grid_max_width: int,
        directml_grid_max_height: int,
        onnx_directml_tuning: str,
    ) -> tuple[int, int, str]:
        """Resolve stitched-grid dimensions for DirectML OCR engines.

        EasyOCR's detector benefits from very large 4096 grids on high-end AMD
        cards. RapidOCR/ONNX DirectML can reserve huge VRAM on 4096x4096 grids,
        so v14 gives ONNX its own safer tuning curve.
        """
        engine = str(ocr_engine or "").strip().lower()
        preset = str(directml_performance_preset or "balanced").strip().lower()
        tuning = str(onnx_directml_tuning or "balanced").strip().lower()
        if engine == "onnx_directml":
            onnx_sizes = {
                "low_vram": (1600, 1600),
                "balanced": (2048, 2048),
                "max": (3072, 3072),
            }
            if tuning in onnx_sizes:
                w, h = onnx_sizes[tuning]
                return w, h, f"ONNX tuning={tuning}"
            # manual ONNX tuning intentionally uses the visible grid boxes.
            try:
                return (
                    max(512, int(directml_grid_max_width or 2048)),
                    max(512, int(directml_grid_max_height or 2048)),
                    "ONNX tuning=manual",
                )
            except Exception:
                return 2048, 2048, "ONNX tuning=manual fallback"
        preset_sizes = {
            "compatibility": (1600, 1600),
            "balanced": (2400, 2400),
            "max": (4096, 4096),
        }
        if preset in preset_sizes:
            w, h = preset_sizes[preset]
            return w, h, f"DirectML preset={preset}"
        try:
            return (
                max(512, int(directml_grid_max_width or 2400)),
                max(512, int(directml_grid_max_height or 2400)),
                "DirectML preset=manual",
            )
        except Exception:
            return 2400, 2400, "DirectML preset=manual fallback"

    @staticmethod
    def _build_label_band(
        band: dict[str, int], crop_w: int, crop_h: int, ratio: float
    ) -> dict[str, Any]:
        """Derive crop/scale strings for one label band (shared scale ratio)."""
        crop_x = max(0, band["x"]) & ~1
        crop_y = max(0, band["y"]) & ~1
        target_w = max(2, int(crop_w * ratio) & ~1)
        target_h = max(2, int(crop_h * ratio) & ~1)
        return {
            "x": band["x"],  # original video coords
            "y": band["y"],
            "w": band["w"],
            "h": band["h"],
            "crop_x": crop_x,
            "crop_y": crop_y,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "crop_str": f"{crop_w}:{crop_h}:{crop_x}:{crop_y}",
            "scale_str": f"{target_w}:{target_h}:flags=area:threads=1",
            "target_w": target_w,
            "target_h": target_h,
        }

    def _label_band(self, zone_idx: int) -> dict[str, Any] | None:
        """Band dict for a label zone index (label zones occupy 2..2+K-1)."""
        i = zone_idx - 2
        return self.label_zones[i] if 0 <= i < len(self.label_zones) else None

    def _collect_label_frames(
        self, frame_predictions_list: dict[int, list[PredictedFrames]]
    ) -> list[PredictedFrames]:
        """Gather label-band frames (zone indices >= 2) in time order and
        apply the label-detection time window."""
        out: list[PredictedFrames] = []
        for zone_idx in sorted(k for k in frame_predictions_list if k >= 2):
            out.extend(frame_predictions_list[zone_idx])
        out.sort(key=lambda f: f.start_index)
        return self._filter_label_frames_by_time(out)

    def _run_label_aware_directml_ocr(
        self,
        runner: Any,
        stitched_dir: str,
        stitch_map: dict[str, list[dict[str, Any]]],
        subtitle_lang: str,
        use_gpu: bool,
        recognition_mode: str,
        ocr_engine: str,
    ) -> dict[tuple[int, int], list[Any]]:
        """Run DirectML recognition, splitting label bands into a second pass
        with the label language when it differs.

        Only easyocr_directml honors the split — the ONNX PP-OCRv6 model is
        unified and ignores language (the CLI prints a note for it).
        """
        if not (self._label_split_lang and ocr_engine == "easyocr_directml"):
            return cast(
                dict[tuple[int, int], list[Any]],
                runner(
                    stitched_dir, stitch_map, subtitle_lang, use_gpu, recognition_mode
                ),
            )
        sub_map: dict[str, list[dict[str, Any]]] = {}
        label_map: dict[str, list[dict[str, Any]]] = {}
        for grid, mappings in stitch_map.items():
            for m in mappings:
                target = label_map if int(m["zone_idx"]) >= 2 else sub_map
                target.setdefault(grid, []).append(m)
        outputs: dict[tuple[int, int], list[Any]] = {}
        if sub_map:
            outputs.update(
                runner(
                    stitched_dir, sub_map, subtitle_lang, use_gpu, recognition_mode
                )
            )
        if label_map:
            print(
                f"Running label OCR pass (lang={self._label_lang})...",
                flush=True,
            )
            outputs.update(
                runner(
                    stitched_dir,
                    label_map,
                    self._label_lang,
                    use_gpu,
                    recognition_mode,
                )
            )
        return outputs

    def _publish_label_hash_weights(self, dedup: utils.LabelFrameDedup) -> None:
        """Merge finished dHash run weights into the label weight maps.

        ``_label_hash_weights`` feeds Step-2 SSIM weight composition;
        ``_label_frame_weights`` is what the merge step reads (and the only
        consumer in paths without a Step-2 pass).
        """
        for rep, weight in dedup.weights.items():
            self._label_hash_weights[rep] = weight
            self._label_frame_weights[rep] = weight
        if dedup.skipped:
            kept = len(dedup.weights)
            print(
                f"[Perf] Label frame dedup (dHash): kept {kept} of "
                f"{kept + dedup.skipped} sampled label frame(s); "
                f"skipped {dedup.skipped} redundant.",
                flush=True,
            )

    def run_ocr(
        self,
        use_gpu: bool,
        ocr_engine: str,
        lang: str,
        use_angle_cls: bool,
        time_start: str,
        time_end: str,
        conf_threshold: int,
        use_fullframe: bool,
        brightness_threshold: int | None,
        ssim_threshold: int,
        subtitle_position: str,
        frames_to_skip: int,
        crop_zones: list[dict[str, int]],
        ocr_image_max_width: int,
        normalize_to_simplified_chinese: bool,
        directml_grid_max_width: int = 2400,
        directml_grid_max_height: int = 2400,
        directml_performance_preset: str = "balanced",
        directml_recognition_mode: str = "stable",
        directml_frame_scan_mode: str = "cpu_ssim",
        onnx_directml_tuning: str = "balanced",
        benchmark_compare_engine: bool = False,
        benchmark_compare_sample_grids: int = 3,
        enable_label_detection: bool = False,
        label_ocr_image_max_width: int = 1080,
        label_min_display_duration_sec: float = 1.0,
        label_min_confirmation_frames: int = 2,
        label_reappear_merge_gap_sec: float = 2.0,
        label_filter_single_char: bool = True,
        label_time_start: str = "",
        label_time_end: str = "",
        label_text_similarity: int = 55,
        label_pos_drift: int = 160,
        label_close_pos_distance: int = 40,
        label_close_pos_length_ratio: float = 0.5,
        label_conf_threshold: int = 60,
        label_ssim_dedup: bool = True,
        label_hash_threshold: int = 4,
        label_min_zone_height: int = 32,
        label_font: str = "Arial",
        label_fontsize: int = 22,
        label_primary_color: str = "&H00FFFFFF",
        label_outline_color: str = "&H00000000",
        label_alignment: str = "an7",
        label_lang: str = "",
        label_model_dirs: dict[str, str] | None = None,
    ) -> None:
        perf_total_start = time.perf_counter()
        step1_start = perf_total_start
        conf_threshold_ratio = conf_threshold / 100
        ssim_threshold_ratio = ssim_threshold / 100
        # Fix 7: label-specific confidence threshold (defaults slightly lower
        # than the subtitle threshold to keep smaller/harder name text).
        self._label_conf_threshold_ratio = (
            float(label_conf_threshold if label_conf_threshold is not None else 60)
            / 100
        )
        self.lang = lang
        self.use_fullframe = use_fullframe
        self.validated_zones = []
        self.pred_frames_zone1 = []
        self.pred_frames_zone2 = []
        self.label_zones = []
        self.label_frames = []
        self._label_frame_weights = {}
        self._label_hash_weights = {}
        self._label_hash_threshold = max(0, min(64, int(label_hash_threshold)))
        self._label_dedup_enabled = bool(label_ssim_dedup)
        self._label_min_zone_height = max(8, int(label_min_zone_height or 32))
        # Label ASS style (see _render_ass). Alignment normalizes both
        # "anN" codes and GUI alignment names to a valid an1..an9 tag.
        align = str(label_alignment or "an7")
        align = utils.ALIGNMENT_MAP.get(align, align)
        if not re.fullmatch(r"an[1-9]", align):
            align = "an7"
        self._label_style = {
            "font": str(label_font or "Arial").replace(",", ""),
            "fontsize": max(1, int(label_fontsize or 22)),
            "primary": str(label_primary_color or "&H00FFFFFF"),
            "outline": str(label_outline_color or "&H00000000"),
            "alignment": align,
        }
        # Label-specific OCR language (empty = subtitle language). When it
        # differs, recognition runs a second pass for label images only.
        label_lang_code = (label_lang or "").strip()
        self._label_lang = label_lang_code or self.lang
        self._label_split_lang = bool(
            label_lang_code and label_lang_code != self.lang
        )
        self._label_model_dirs = label_model_dirs
        self.label_time_start_ms = 0.0
        self.label_time_end_ms = None
        self._label_min_display_duration_ms = (
            float(label_min_display_duration_sec or 1.0) * 1000.0
        )
        self._label_min_confirmation_frames = max(
            1, int(label_min_confirmation_frames or 1)
        )
        self._label_reappear_merge_gap_ms = (
            float(label_reappear_merge_gap_sec or 0.0) * 1000.0
        )
        self._label_filter_single_char = bool(label_filter_single_char)
        # Fix 3: configurable label-merge thresholds (previously hardcoded).
        self._label_text_similarity_threshold = float(
            label_text_similarity if label_text_similarity is not None else 55
        )
        self._label_pos_drift_px = float(
            label_pos_drift if label_pos_drift is not None else 160
        )
        self._label_close_pos_distance_px = float(
            label_close_pos_distance if label_close_pos_distance is not None else 40
        )
        self._label_close_pos_length_ratio = float(
            label_close_pos_length_ratio
            if label_close_pos_length_ratio is not None
            else 0.5
        )

        if ocr_engine == "onnx_directml":
            os.environ["VIDEOCR_ONNX_DIRECTML_TUNING"] = (
                str(onnx_directml_tuning or "balanced").strip().lower() or "balanced"
            )
            print(
                f"ONNX DirectML tuning mode: {os.environ['VIDEOCR_ONNX_DIRECTML_TUNING']}",
                flush=True,
            )

        user_start_ms = 0.0
        if time_start:
            user_start_ms = utils.get_ms_from_time_str(time_start)
        target_start_ms = user_start_ms + self.start_time_offset_ms
        target_end_ms = None
        target_end_str = "Unknown"
        if time_end:
            user_end_ms = utils.get_ms_from_time_str(time_end)
            target_end_ms = user_end_ms + self.start_time_offset_ms
            target_end_str = utils.get_srt_timestamp_from_ms(user_end_ms).split(",")[0]
        elif self.duration_ms > 0:
            target_end_str = utils.get_srt_timestamp_from_ms(self.duration_ms).split(
                ","
            )[0]

        # Label-detection time window, independent of the subtitle extraction
        # window. Empty values fall back to the main window so labels follow
        # the overall run unless the user explicitly overrides them.
        label_start_ms = float(user_start_ms)
        label_end_ms = target_end_ms
        if label_time_start:
            label_start_ms = utils.get_ms_from_time_str(label_time_start)
        if label_time_end:
            label_end_ms = utils.get_ms_from_time_str(label_time_end)
        self.label_time_start_ms = label_start_ms + self.start_time_offset_ms
        self.label_time_end_ms = (
            label_end_ms + self.start_time_offset_ms
            if label_end_ms is not None
            else None
        )

        for zone in crop_zones:
            if zone["y"] >= self.height:
                raise ValueError(
                    f"Crop Y position ({zone['y']}) is outside video height ({self.height})."
                )
            if zone["x"] >= self.width:
                raise ValueError(
                    f"Crop X position ({zone['x']}) is outside video width ({self.width})."
                )
            if zone["y"] + zone["height"] > self.height:
                print(
                    f"Warning: Crop area extends out of bounds (crop_y + crop_height > video height ({self.height})). The crop area will be clipped.",
                    flush=True,
                )
            if zone["x"] + zone["width"] > self.width:
                print(
                    f"Warning: Crop area extends out of bounds (crop_x + crop_width > video width ({self.width})). The crop area will be clipped.",
                    flush=True,
                )
            self.validated_zones.append(
                {
                    "x_start": zone["x"],
                    "y_start": zone["y"],
                    "x_end": zone["x"] + zone["width"],
                    "y_end": zone["y"] + zone["height"],
                    "midpoint_y": zone["y"] + (zone["height"] / 2),
                }
            )

        # Handle full frame and fallback scenarios by injecting them as standard zones
        if self.use_fullframe:
            # Overwrite any user zones with a single full-frame zone
            self.validated_zones = [
                {
                    "x_start": 0,
                    "y_start": 0,
                    "x_end": self.width,
                    "y_end": self.height,
                    "midpoint_y": self.height / 2,
                }
            ]
        elif not self.validated_zones:
            # Default to bottom third if no zones were provided
            self.validated_zones.append(
                {
                    "x_start": 0,
                    "y_start": 2 * self.height // 3,
                    "x_end": self.width,
                    "y_end": self.height,
                    "midpoint_y": (2 * self.height // 3) + (self.height // 6),
                }
            )

        # Pre-calculate FFmpeg crop and scale strings for all zones
        for val_zone in self.validated_zones:
            crop_w = max(
                2,
                (min(self.width, val_zone["x_end"]) - max(0, val_zone["x_start"])) & ~1,
            )
            crop_h = max(
                2,
                (min(self.height, val_zone["y_end"]) - max(0, val_zone["y_start"]))
                & ~1,
            )
            crop_x = max(0, val_zone["x_start"]) & ~1
            crop_y = max(0, val_zone["y_start"]) & ~1
            # Resize image
            MIN_SIDE = 64
            scale_ratio = 1.0
            if ocr_image_max_width and crop_w > ocr_image_max_width:
                scale_ratio = ocr_image_max_width / crop_w
                min_required_ratio = MIN_SIDE / min(crop_w, crop_h)
                if scale_ratio < min_required_ratio:
                    scale_ratio = min_required_ratio
            target_w = max(2, int(crop_w * scale_ratio) & ~1)
            target_h = max(2, int(crop_h * scale_ratio) & ~1)
            val_zone["w"] = target_w
            val_zone["h"] = target_h
            val_zone["crop_str"] = f"{crop_w}:{crop_h}:{crop_x}:{crop_y}"
            val_zone["scale_str"] = f"{target_w}:{target_h}:flags=area:threads=1"

        # Label detection area: horizontal bands OUTSIDE the subtitle crop
        # zone(s). Every free band >= min_height is scanned so labels in
        # smaller regions (e.g. a top band) are not missed.
        if enable_label_detection:
            min_zone_height = self._label_min_zone_height
            bands = utils.compute_label_zones(
                self.width, self.height, self.validated_zones,
                min_height=min_zone_height,
            )
            if not bands:
                print(
                    "Warning: Label detection requested, but no free band "
                    f"(>= {min_zone_height}px) remains outside the subtitle "
                    "crop area (this includes --use_fullframe). Label detection skipped.",
                    flush=True,
                )
                self.label_zones = []
            else:
                # One shared scale ratio for all bands so every band's OCR
                # crop has the same target width (required to vstack bands
                # into a single FFmpeg label output).
                MIN_SIDE = 64
                crop_ws = [
                    max(
                        2,
                        (
                            min(self.width, b["x"] + b["w"])
                            - max(0, b["x"])
                        )
                        & ~1,
                    )
                    for b in bands
                ]
                crop_hs = [
                    max(
                        2,
                        (
                            min(self.height, b["y"] + b["h"])
                            - max(0, b["y"])
                        )
                        & ~1,
                    )
                    for b in bands
                ]
                # Per-band ratio (same formula as the subtitle zones), then one
                # shared value so every band's OCR crop has the same target
                # width (required to vstack bands into one FFmpeg output).
                band_ratios: list[float] = []
                for crop_w, crop_h in zip(crop_ws, crop_hs):
                    band_ratio = 1.0
                    if label_ocr_image_max_width and crop_w > label_ocr_image_max_width:
                        band_ratio = label_ocr_image_max_width / crop_w
                        min_required_ratio = MIN_SIDE / min(crop_w, crop_h)
                        if band_ratio < min_required_ratio:
                            band_ratio = min_required_ratio
                    band_ratios.append(band_ratio)
                ratio = max(band_ratios)
                self.label_zones = [
                    self._build_label_band(b, crop_w, crop_h, ratio)
                    for b, crop_w, crop_h in zip(bands, crop_ws, crop_hs)
                ]
                for band in self.label_zones:
                    print(
                        f"Label detection band (outside subtitle crop): x={band['x']} y={band['y']} "
                        f"w={band['w']} h={band['h']}",
                        flush=True,
                    )

        temp_dir = utils.create_clean_temp_dir()
        try:
            ffmpeg_mode = (
                str(directml_frame_scan_mode or "cpu_ssim").strip().lower()
                == "ffmpeg_d3d11va"
            )
            # Fix 6: label detection no longer forces the PyAV CPU fallback —
            # the FFmpeg D3D11VA path now emits the label zone as a second
            # output. The FFmpeg path itself still falls back when it cannot
            # handle the current zone configuration (e.g. two subtitle zones).

            if ocr_engine in ("easyocr_directml", "onnx_directml") and ffmpeg_mode:
                ffmpeg_grid_w, ffmpeg_grid_h, grid_note = (
                    self._resolve_directml_grid_size(
                        ocr_engine,
                        directml_performance_preset,
                        directml_grid_max_width,
                        directml_grid_max_height,
                        onnx_directml_tuning,
                    )
                )
                print(
                    f"DirectML stitched grid target: {ffmpeg_grid_w}x{ffmpeg_grid_h}px ({grid_note})",
                    flush=True,
                )
                used_ffmpeg_hw_path = self._run_easyocr_directml_ffmpeg_hw_scan(
                    temp_dir,
                    target_end_str,
                    ssim_threshold_ratio,
                    subtitle_position,
                    brightness_threshold,
                    frames_to_skip,
                    ffmpeg_grid_w,
                    ffmpeg_grid_h,
                    directml_recognition_mode,
                    use_gpu,
                    ocr_engine,
                    conf_threshold_ratio,
                    lang,
                    normalize_to_simplified_chinese,
                    step1_start,
                    perf_total_start,
                    onnx_directml_tuning,
                    benchmark_compare_engine,
                    benchmark_compare_sample_grids,
                    target_start_ms,
                    target_end_ms,
                )
                if used_ffmpeg_hw_path:
                    return

            raw_queue: queue.Queue[Any] = queue.Queue(maxsize=100)
            processed_queue: queue.Queue[Any] = queue.Queue(maxsize=100)
            write_queue: queue.Queue[Any] = queue.Queue(maxsize=200)
            start_index_queue: queue.Queue[Any] = queue.Queue()
            stop_event = threading.Event()
            drain_event = threading.Event()
            error_list: list[Exception] = []

            def producer_thread() -> None:
                try:
                    with Capture(self.path) as v:
                        is_seeking = user_start_ms > 0
                        if is_seeking:
                            v.seek(target_start_ms)
                        current_index = 0
                        modulo = frames_to_skip + 1
                        first_queued = False
                        while not stop_event.is_set():
                            success, raw_frame, timestamp_ms = v.read()
                            if not success:
                                break
                            curr_str = utils.get_srt_timestamp_from_ms(
                                timestamp_ms - self.start_time_offset_ms
                            ).split(",")[0]
                            # Check Start Time
                            if is_seeking:
                                if timestamp_ms < target_start_ms:
                                    continue
                                else:
                                    is_seeking = False
                            # Check End Time
                            if (
                                target_end_ms is not None
                                and timestamp_ms > target_end_ms
                            ):
                                break
                            if not first_queued:
                                start_index_queue.put(current_index)
                                first_queued = True
                            should_process_frame = current_index % modulo == 0
                            if should_process_frame:
                                raw_queue.put(
                                    (current_index, timestamp_ms, raw_frame, curr_str)
                                )
                            else:
                                raw_queue.put(
                                    (current_index, timestamp_ms, None, curr_str)
                                )
                            current_index += 1

                except Exception as e:
                    error_list.append(e)
                    stop_event.set()

                finally:
                    if not first_queued:
                        start_index_queue.put(None)
                    raw_queue.put(None)

            def worker_thread() -> None:
                graph = None
                sinks: list[Any] = []

                try:
                    while not stop_event.is_set():
                        try:
                            item = raw_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        if item is None:
                            raw_queue.put(None)
                            break

                        current_index, timestamp_ms, raw_frame, curr_str = item

                        if raw_frame is None:
                            processed_queue.put(
                                (current_index, timestamp_ms, None, curr_str)
                            )
                            continue
                        images_to_process: list[dict[str, Any]] = []

                        # Filter Graph
                        if graph is None:
                            graph = av.filter.Graph()
                            buffer_node = graph.add_buffer(template=raw_frame)
                            num_zones = len(self.validated_zones)
                            label_zones = self.label_zones
                            num_label_bands = len(label_zones)
                            num_outputs = num_zones + num_label_bands
                            label_sink_base = (
                                num_zones if num_label_bands > 0 else -1
                            )
                            if num_outputs == 1:
                                # Single Zone (User crop, Bottom Third, Full Frame)
                                # Pipeline: Buffer -> Crop -> Scale -> Sink
                                z = self.validated_zones[0]
                                crop_node = graph.add("crop", z["crop_str"])
                                scale_node = graph.add("scale", z["scale_str"])
                                sink_node = graph.add("buffersink")
                                buffer_node.link_to(crop_node)
                                crop_node.link_to(scale_node)
                                scale_node.link_to(sink_node)
                                sinks.append(sink_node)
                            else:
                                # Multiple outputs (Dual Zone and/or Label bands):
                                # Buffer -> Split(N) -> (Crop -> Scale -> Sink) x N.
                                # Every output goes through the split so each branch
                                # gets an explicit output pad (avoids PyAV linking
                                # every branch to buffersrc output 0).
                                split_node = graph.add("split", str(num_outputs))
                                buffer_node.link_to(split_node)
                                all_zones = list(self.validated_zones)
                                all_zones.extend(label_zones)
                                for i, z in enumerate(all_zones):
                                    crop_node = graph.add("crop", z["crop_str"])
                                    scale_node = graph.add("scale", z["scale_str"])
                                    sink_node = graph.add("buffersink")
                                    split_node.link_to(crop_node, output_idx=i)
                                    crop_node.link_to(scale_node)
                                    scale_node.link_to(sink_node)
                                    sinks.append(sink_node)

                            graph.configure()

                        graph.push(raw_frame)

                        for idx, sink in enumerate(sinks):
                            # Each branch is driven by a split, so every sink
                            # yields one frame per pushed frame. Guard the pull
                            # anyway so a missing label frame never blocks the
                            # whole worker (missing label = skip this frame).
                            is_label_sink = (
                                label_sink_base >= 0 and idx >= label_sink_base
                            )
                            try:
                                processed_raw_frame = cast(av.VideoFrame, sink.pull())
                            except Exception:
                                if is_label_sink:
                                    continue
                                raise
                            img = utils.frame_to_array(processed_raw_frame, fmt="rgb24")
                            # Label sinks are the label-zone branches (2..2+K-1).
                            if is_label_sink:
                                images_to_process.append(
                                    {
                                        "zone_idx": 2 + (idx - label_sink_base),
                                        "img": img,
                                        "ssim_sample": None,
                                    }
                                )
                                continue
                            zone_idx = idx
                            if brightness_threshold is not None:
                                gray = (
                                    (
                                        img[..., 0].astype(np.uint16) * 77
                                        + img[..., 1].astype(np.uint16) * 150
                                        + img[..., 2].astype(np.uint16) * 29
                                    )
                                    >> 8
                                ).astype(np.uint8)
                                mask = gray > brightness_threshold
                                img *= mask[..., None]
                            sample = None
                            if ssim_threshold_ratio < 1:
                                w = img.shape[1]
                                if subtitle_position == "center":
                                    w_margin = int(w * 0.35)
                                    sample = img[:, w_margin : w - w_margin]
                                elif subtitle_position == "left":
                                    sample = img[:, : int(w * 0.3)]
                                elif subtitle_position == "right":
                                    sample = img[:, int(w * 0.7) :]
                                elif subtitle_position == "any":
                                    sample = img
                                else:
                                    raise ValueError(
                                        f"Invalid subtitle_position: {subtitle_position}"
                                    )
                            images_to_process.append(
                                {
                                    "zone_idx": zone_idx,
                                    "img": img,
                                    "ssim_sample": sample,
                                }
                            )
                        processed_queue.put(
                            (current_index, timestamp_ms, images_to_process, curr_str)
                        )
                except Exception as e:
                    error_list.append(e)
                    stop_event.set()

            def writer_thread() -> None:
                try:
                    while not stop_event.is_set():
                        try:
                            item = write_queue.get(timeout=0.5)
                        except queue.Empty:
                            if drain_event.is_set():
                                break
                            continue
                        frame_path, canvas_w, canvas_h, draw_instructions = item
                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        for img, x, y in draw_instructions:
                            h, w = img.shape[:2]
                            canvas[y : y + h, x : x + w] = img
                        Image.fromarray(canvas).save(frame_path, quality=80)
                except Exception as e:
                    error_list.append(e)
                    stop_event.set()

            # Start Threads
            producer = threading.Thread(target=producer_thread)
            producer.start()
            num_workers = num_writers = (os.cpu_count() or 1) // 4 + 1
            workers: list[threading.Thread] = []
            for _ in range(num_workers):
                t = threading.Thread(target=worker_thread)
                t.start()
                workers.append(t)
            writers: list[threading.Thread] = []
            for _ in range(num_writers):
                t = threading.Thread(target=writer_thread)
                t.start()
                writers.append(t)

            DEFAULT_STITCH_WIDTH = 1500
            DEFAULT_STITCH_HEIGHT = 1500
            MAX_STITCH_WIDTH = DEFAULT_STITCH_WIDTH
            MAX_STITCH_HEIGHT = DEFAULT_STITCH_HEIGHT
            if ocr_engine in ("easyocr_directml", "onnx_directml"):
                MAX_STITCH_WIDTH, MAX_STITCH_HEIGHT, grid_note = (
                    self._resolve_directml_grid_size(
                        ocr_engine,
                        directml_performance_preset,
                        directml_grid_max_width,
                        directml_grid_max_height,
                        onnx_directml_tuning,
                    )
                )
            GRID_SPACING = 10
            FILENAME_ZERO_PADDING = 8
            if ocr_engine in ("easyocr_directml", "onnx_directml"):
                print(
                    f"DirectML performance preset: {directml_performance_preset}",
                    flush=True,
                )
                print(
                    f"DirectML recognition mode: {directml_recognition_mode}",
                    flush=True,
                )
                if ocr_engine == "onnx_directml":
                    print(
                        f"ONNX DirectML tuning mode: {onnx_directml_tuning}", flush=True
                    )
                print(
                    f"DirectML stitched grid target: {MAX_STITCH_WIDTH}x{MAX_STITCH_HEIGHT}px ({grid_note})",
                    flush=True,
                )

            batch_limits: dict[int, int] = {}
            for z_idx, z in enumerate(self.validated_zones):
                batch_limits[z_idx] = utils.get_batch_limit(
                    z["w"], z["h"], MAX_STITCH_WIDTH, MAX_STITCH_HEIGHT, GRID_SPACING
                )
            for k, band in enumerate(self.label_zones):
                batch_limits[2 + k] = utils.get_batch_limit(
                    band["target_w"],
                    band["target_h"],
                    MAX_STITCH_WIDTH,
                    MAX_STITCH_HEIGHT,
                    GRID_SPACING,
                )

            def flush_batch(
                batch: list[Any],
                counter: int,
                zone_idx: int,
                prefix: str,
                out_dir: str,
                target_map: dict[str, Any],
            ) -> int:
                queue_args = utils.prepare_stitch_batch(
                    batch,
                    counter,
                    zone_idx,
                    prefix,
                    out_dir,
                    target_map,
                    MAX_STITCH_WIDTH,
                    GRID_SPACING,
                    FILENAME_ZERO_PADDING,
                )
                write_queue.put(queue_args)
                return counter + 1

            # Consumer Logic
            det_stitched_dir = os.path.join(temp_dir, "det_stitched")
            os.makedirs(det_stitched_dir, exist_ok=True)
            det_stitch_map: dict[str, list[dict[str, Any]]] = {}
            det_counter = 0
            det_batches: dict[int, list[Any]] = {0: [], 1: []}
            for k in range(len(self.label_zones)):
                det_batches[2 + k] = []
            # Dict (not list): zone indices are sparse — subtitle zones are
            # 0/1 while label bands start at 2 regardless of zone count.
            prev_samples: dict[int, Any] = {}
            # Pre-OCR dHash dedup for label bands (gated by label_ssim_dedup).
            label_dedup: utils.LabelFrameDedup | None = None
            if self.label_zones and self._label_dedup_enabled:
                label_dedup = utils.LabelFrameDedup(
                    threshold=self._label_hash_threshold
                )
            dml_frame_filter = None
            dml_frame_filter_failed = False
            dml_frame_scan_mode_normalized = (
                str(directml_frame_scan_mode or "cpu_ssim").strip().lower()
            )
            if (
                ocr_engine in ("easyocr_directml", "onnx_directml")
                and dml_frame_scan_mode_normalized == "directml_ssim"
                and ssim_threshold_ratio < 1
            ):
                try:
                    requested_index = os.environ.get(
                        "VIDEOCR_DIRECTML_DEVICE_INDEX", ""
                    ).strip()
                    device_index = int(requested_index) if requested_index else None
                    from .directml_frame_filter import DirectMLFrameSimilarityFilter

                    dml_frame_filter = DirectMLFrameSimilarityFilter(
                        ssim_threshold_ratio, device_index=device_index
                    )
                    print(
                        f"Step 1 frame scan mode: AMD DirectML SSIM on adapter {dml_frame_filter.device_label} "
                        "(experimental; video decode/crop still uses PyAV/CPU).",
                        flush=True,
                    )
                except Exception as e:
                    dml_frame_filter = None
                    dml_frame_filter_failed = True
                    print(
                        f"Warning: DirectML frame scan could not start, falling back to CPU SSIM: {e}",
                        flush=True,
                    )
            elif ocr_engine in ("easyocr_directml", "onnx_directml"):
                print("Step 1 frame scan mode: CPU SSIM (compatible).", flush=True)

            expected_index = None
            success = False
            try:
                while expected_index is None and not stop_event.is_set():
                    if error_list:
                        break
                    try:
                        expected_index = start_index_queue.get(timeout=0.1)
                    except queue.Empty:
                        if not producer.is_alive():
                            break
                        continue
                if expected_index is not None:
                    buffer: dict[int, tuple[float, Any, str]] = {}
                    while not stop_event.is_set():
                        if error_list:
                            break
                        try:
                            item = processed_queue.get(timeout=0.1)
                        except queue.Empty:
                            if (
                                not producer.is_alive()
                                and all(not w.is_alive() for w in workers)
                                and processed_queue.empty()
                            ):
                                break
                            continue
                        current_index, timestamp_ms, images_to_process, curr_str = item
                        buffer[current_index] = (
                            timestamp_ms,
                            images_to_process,
                            curr_str,
                        )
                        # Process buffer sequentially
                        while expected_index in buffer:
                            timestamp_ms, images_to_process, curr_str = buffer.pop(
                                expected_index
                            )
                            self.frame_timestamps[expected_index] = timestamp_ms
                            if expected_index % 15 == 0:
                                print(
                                    f"\rStep 1/3: Processing video... Current: {curr_str} / {target_end_str}, Frame: {expected_index + 1}",
                                    end="",
                                    flush=True,
                                )
                            if images_to_process is not None:
                                for zone_data in images_to_process:
                                    zone_idx = zone_data["zone_idx"]
                                    img = zone_data["img"]
                                    sample = zone_data["ssim_sample"]
                                    # Label bands (indices >= 2) are never
                                    # SSIM-filtered: labels must be detected
                                    # on every appearance.
                                    is_label_zone = zone_idx >= 2
                                    if not is_label_zone and ssim_threshold_ratio < 1:
                                        if prev_samples.get(zone_idx) is not None:
                                            if dml_frame_filter is not None:
                                                try:
                                                    if dml_frame_filter.is_similar(
                                                        prev_samples[zone_idx], sample
                                                    ):
                                                        prev_samples[zone_idx] = sample
                                                        continue
                                                except Exception as e:
                                                    # Do not kill the whole OCR job if the experimental GPU frame scan hits
                                                    # a DirectML operator/runtime issue. Fall back to the known-good CPU path.
                                                    if not dml_frame_filter_failed:
                                                        print(
                                                            f"Warning: DirectML frame scan failed during processing; falling back to CPU SSIM: {e}",
                                                            flush=True,
                                                        )
                                                    dml_frame_filter = None
                                                    dml_frame_filter_failed = True
                                            score = fast_ssim.ssim(
                                                prev_samples[zone_idx],
                                                sample,
                                                data_range=255,
                                            )
                                            if score > ssim_threshold_ratio:
                                                prev_samples[zone_idx] = sample
                                                continue
                                        # First sample per zone: nothing to
                                        # compare against yet — always keep it.
                                    prev_samples[zone_idx] = sample
                                    if (
                                        is_label_zone
                                        and label_dedup is not None
                                        and not label_dedup.consider(
                                            zone_idx, expected_index, img
                                        )
                                    ):
                                        # Redundant label frame: folded into
                                        # the kept representative's weight.
                                        continue
                                    det_batches[zone_idx].append(
                                        {"img": img, "frame_idx": expected_index}
                                    )
                                    if (
                                        len(det_batches[zone_idx])
                                        >= batch_limits[zone_idx]
                                    ):
                                        det_counter = flush_batch(
                                            det_batches[zone_idx],
                                            det_counter,
                                            zone_idx,
                                            "det_stitched",
                                            det_stitched_dir,
                                            det_stitch_map,
                                        )
                                        det_batches[zone_idx] = []
                            expected_index += 1
                    for z_idx in sorted(det_batches.keys()):
                        if det_batches[z_idx]:
                            det_counter = flush_batch(
                                det_batches[z_idx],
                                det_counter,
                                z_idx,
                                "det_stitched",
                                det_stitched_dir,
                                det_stitch_map,
                            )
                    if label_dedup is not None:
                        label_dedup.finish()
                        self._publish_label_hash_weights(label_dedup)
                    if not error_list and expected_index > 0:
                        last_idx = expected_index - 1
                        final_ms = self.frame_timestamps.get(last_idx, 0)
                        final_str = utils.get_srt_timestamp_from_ms(
                            final_ms - self.start_time_offset_ms
                        ).split(",")[0]
                        if target_end_ms is not None:
                            print(
                                f"\rStep 1/3: Processing video... Current: {target_end_str} / {target_end_str}, Frame: {expected_index}",
                                end="",
                            )
                            print("\nReached end time. Stopping.", flush=True)
                        else:
                            print(
                                f"\rStep 1/3: Processing video... Current: {final_str} / {target_end_str}, Frame: {expected_index}",
                                flush=True,
                            )
                    success = True
            except KeyboardInterrupt:
                raise
            finally:
                is_aborting = not success or len(error_list) > 0
                if is_aborting:
                    stop_event.set()
                else:
                    drain_event.set()
                while not raw_queue.empty():
                    try:
                        raw_queue.get_nowait()
                    except queue.Empty:
                        break
                while not processed_queue.empty():
                    try:
                        processed_queue.get_nowait()
                    except queue.Empty:
                        break
                if is_aborting:
                    while not write_queue.empty():
                        try:
                            write_queue.get_nowait()
                        except queue.Empty:
                            break
                producer.join()
                for w in workers:
                    w.join()
                for w in writers:
                    w.join()
                if error_list:
                    raise error_list[0]

            ocr_end = expected_index if expected_index is not None else 0
            step1_end = time.perf_counter()
            if dml_frame_filter is not None:
                print(
                    f"[Perf] DirectML frame scan comparisons: {dml_frame_filter.comparisons}, "
                    f"filtered: {dml_frame_filter.filtered}",
                    flush=True,
                )
            if len(self.frame_timestamps) > 1:
                min_idx = min(self.frame_timestamps.keys())
                max_idx = max(self.frame_timestamps.keys())
                if max_idx > min_idx:
                    total_duration = (
                        self.frame_timestamps[max_idx] - self.frame_timestamps[min_idx]
                    )
                    self.avg_frame_duration_ms = total_duration / (max_idx - min_idx)

            if det_counter == 0:
                print(
                    f"[Perf] Step 1 frame scan/filter/stitch: {step1_end - step1_start:.2f}s",
                    flush=True,
                )
                print(
                    "[Perf] No frames survived OCR filtering. Nothing to recognize.",
                    flush=True,
                )
                self.label_zones = []
                self.label_frames = []
                return

            if ocr_engine in ("easyocr_directml", "onnx_directml"):
                # DirectML backends handle detection and recognition in a single
                # pass over the stitched frame grids, then map every recognized
                # line back into VideOCR's normal per-frame prediction model.
                if ocr_engine == "onnx_directml":
                    from .onnx_directml_ocr import run_onnx_directml_on_stitched_images

                    run_directml_ocr = run_onnx_directml_on_stitched_images
                    directml_label = "ONNX Runtime DirectML OCR"
                else:
                    from .easyocr_directml import run_easyocr_on_stitched_images

                    run_directml_ocr = run_easyocr_on_stitched_images
                    directml_label = "EasyOCR DirectML hybrid OCR"

                total_stitched_frames = sum(
                    len(mappings) for mappings in det_stitch_map.values()
                )
                total_grids = len(det_stitch_map)
                avg_frames_per_grid = (
                    total_stitched_frames / total_grids if total_grids else 0.0
                )
                print(
                    f"Running {directml_label} pass on {total_stitched_frames} filtered frame(s) "
                    f"stitched into {total_grids} image grid(s) "
                    f"({avg_frames_per_grid:.1f} frame(s)/grid)...",
                    flush=True,
                )
                directml_ocr_start = time.perf_counter()
                ocr_outputs = self._run_label_aware_directml_ocr(
                    run_directml_ocr,
                    det_stitched_dir,
                    det_stitch_map,
                    self.lang,
                    use_gpu,
                    directml_recognition_mode,
                    ocr_engine,
                )
                directml_ocr_end = time.perf_counter()

                if ocr_engine == "onnx_directml" and benchmark_compare_engine:
                    try:
                        sample_count = max(
                            1,
                            min(int(benchmark_compare_sample_grids or 3), total_grids),
                        )
                        sample_names = sorted(det_stitch_map.keys())[:sample_count]
                        sample_map = {
                            name: det_stitch_map[name] for name in sample_names
                        }
                        sample_dir = os.path.join(temp_dir, "benchmark_compare_grids")
                        os.makedirs(sample_dir, exist_ok=True)
                        for name in sample_names:
                            shutil.copy2(
                                os.path.join(det_stitched_dir, name),
                                os.path.join(sample_dir, name),
                            )
                        print(
                            f"[BenchCompare] Running EasyOCR Hybrid sample on {sample_count} stitched grid(s)...",
                            flush=True,
                        )
                        from .easyocr_directml import run_easyocr_on_stitched_images

                        compare_start = time.perf_counter()
                        _ = run_easyocr_on_stitched_images(
                            sample_dir, sample_map, self.lang, use_gpu, "stable"
                        )
                        compare_elapsed = time.perf_counter() - compare_start
                        onnx_per_grid = (
                            (directml_ocr_end - directml_ocr_start) / total_grids
                            if total_grids
                            else 0.0
                        )
                        easy_per_grid = (
                            compare_elapsed / sample_count if sample_count else 0.0
                        )
                        print(
                            f"[BenchCompare] ONNX full OCR: {directml_ocr_end - directml_ocr_start:.2f}s / {total_grids} grid(s) = {onnx_per_grid:.2f}s/grid",
                            flush=True,
                        )
                        print(
                            f"[BenchCompare] EasyOCR Hybrid sample: {compare_elapsed:.2f}s / {sample_count} grid(s) = {easy_per_grid:.2f}s/grid; projected full={easy_per_grid * total_grids:.2f}s",
                            flush=True,
                        )
                    except Exception as e:
                        print(
                            f"[BenchCompare] EasyOCR-vs-ONNX sample benchmark failed: {e}",
                            flush=True,
                        )

                active_frame_coords: set[tuple[int, int]] = set()
                for mappings in det_stitch_map.values():
                    for m in mappings:
                        active_frame_coords.add(
                            (int(m["frame_idx"]), int(m["zone_idx"]))
                        )
                frame_predictions_dict: dict[int, dict[int, PredictedFrames]] = {
                    0: {},
                    1: {},
                }
                for k in range(len(self.label_zones)):
                    frame_predictions_dict[2 + k] = {}
                for frame_index, zone_index in active_frame_coords:
                    ocr_result = ocr_outputs.get((frame_index, zone_index), [])
                    pred_data = [ocr_result] if ocr_result else [[]]
                    # Fix 7: label bands use their own (lower) confidence gate.
                    zone_conf = (
                        self._label_conf_threshold_ratio
                        if zone_index >= 2
                        else conf_threshold_ratio
                    )
                    predicted_frame = PredictedFrames(
                        ocr_engine,
                        frame_index,
                        pred_data,
                        zone_conf,
                        zone_index,
                        self._label_lang if zone_index >= 2 else lang,
                        normalize_to_simplified_chinese,
                    )
                    frame_predictions_dict[zone_index][frame_index] = predicted_frame
                frame_predictions_list: dict[int, list[PredictedFrames]] = {}
                for zone_idx in frame_predictions_dict:
                    frames = sorted(
                        frame_predictions_dict[zone_idx].values(),
                        key=lambda f: f.start_index,
                    )
                    if not frames:
                        continue
                    for i in range(len(frames) - 1):
                        current_pred = frames[i]
                        next_pred = frames[i + 1]
                        current_pred.end_index = next_pred.start_index - 1
                    frames[-1].end_index = ocr_end - 1
                    frame_predictions_list[zone_idx] = frames

                self.pred_frames_zone1 = frame_predictions_list.get(0, [])
                self.pred_frames_zone2 = frame_predictions_list.get(1, [])
                self.label_frames = (
                    self._collect_label_frames(frame_predictions_list)
                    if self.label_zones
                    else []
                )

                total_elapsed = time.perf_counter() - perf_total_start
                print(
                    f"[Perf] Step 1 frame scan/filter/stitch: {step1_end - step1_start:.2f}s",
                    flush=True,
                )
                print(
                    f"[Perf] Step 2 {directml_label}: {directml_ocr_end - directml_ocr_start:.2f}s",
                    flush=True,
                )
                if ocr_engine == "onnx_directml":
                    print(
                        f"[Perf] ONNX tuning: {onnx_directml_tuning}; grid target: {MAX_STITCH_WIDTH}x{MAX_STITCH_HEIGHT}",
                        flush=True,
                    )
                print(
                    f"[Perf] Filtered OCR frames: {total_stitched_frames}; stitched grids: {total_grids}; average frames/grid: {avg_frames_per_grid:.1f}",
                    flush=True,
                )
                print(
                    f"[Perf] Total OCR preparation/runtime before subtitle merge: {total_elapsed:.2f}s",
                    flush=True,
                )
                return

            # --------------------------------------------------------
            # Detection pass and SSIM filtering on detected text boxes
            # --------------------------------------------------------
            TIGHT_BOX_SSIM_THRESHOLD = 0.85
            total_stitched_frames = sum(
                len(mappings) for mappings in det_stitch_map.values()
            )
            print(
                f"Running Text-Detection-Only pass on {total_stitched_frames} filtered frame(s) stitched into {det_counter} image grid(s)...",
                flush=True,
            )
            det_res_dir = os.path.join(temp_dir, "det_results")
            os.makedirs(det_res_dir, exist_ok=True)
            args = [
                self.paddleocr_path,
                "text_detection",
                "--input",
                det_stitched_dir,
                "--model_dir",
                self.det_model_dir,
                "--model_name",
                os.path.basename(self.det_model_dir),
                "--save_path",
                det_res_dir,
            ]
            print("Starting PaddleOCR...", flush=True)
            for line in utils.stream_cli_process(args, "paddleocr_error.log"):
                if "ppocr INFO: Processed item" in line:
                    match = re.search(r"Processed item (\d+)", line)
                    if match:
                        current_item = match.group(1)
                        print(
                            f"\rStep 2/3: Performing Text-Detection on image {current_item} of {det_counter}",
                            end="",
                            flush=True,
                        )
            print()

            # Parse JSON Outputs and unstitch coordinates
            parsed_detections: dict[int, list[Any]] = {0: [], 1: []}
            for k in range(len(self.label_zones)):
                parsed_detections[2 + k] = []
            for json_file in os.listdir(det_res_dir):
                if not json_file.endswith(".json"):
                    continue
                with open(os.path.join(det_res_dir, json_file), encoding="utf-8") as f:
                    data = json.load(f)
                stitched_filename = os.path.basename(data["input_path"])
                if stitched_filename not in det_stitch_map:
                    continue
                mapping = det_stitch_map[stitched_filename]
                zone_idx = mapping[0]["zone_idx"]
                temp_polys_dict: dict[int, list[Any]] = {
                    m["frame_idx"]: [] for m in mapping
                }
                dt_polys = data["dt_polys"]
                dt_scores = data["dt_scores"]
                for poly, score in zip(dt_polys, dt_scores):
                    for adjusted_poly, m in utils.unstitch_polygon(poly, mapping):
                        temp_polys_dict[m["frame_idx"]].append(
                            {"poly": adjusted_poly, "score": score}
                        )
                for m in mapping:
                    polys_data = temp_polys_dict[m["frame_idx"]]
                    frame_score = (
                        sum(p["score"] for p in polys_data) / len(polys_data)
                        if polys_data
                        else 0.0
                    )
                    extracted_polygons = [p["poly"] for p in polys_data]
                    parsed_detections[zone_idx].append(
                        (m["frame_idx"], extracted_polygons, frame_score, m)
                    )
            for z_idx in parsed_detections:
                parsed_detections[z_idx].sort(key=lambda x: x[0])

            frames_processed = 0
            frames_deleted_count = 0
            next_print_target = 15
            rec_images_dir = os.path.join(temp_dir, "rec_images")
            os.makedirs(rec_images_dir, exist_ok=True)
            # Separate directory for label rec-images when labels OCR in a
            # different language (second recognition pass over just these).
            label_rec_images_dir = ""
            label_rec_written = 0
            if self._label_split_lang:
                label_rec_images_dir = os.path.join(temp_dir, "rec_images_label")
                os.makedirs(label_rec_images_dir, exist_ok=True)
            empty_frames_meta: set[tuple[int, int]] = set()
            surviving_frames_meta: set[tuple[int, int]] = set()
            rec_image_map: dict[str, dict[str, int]] = {}
            rec_counter = 0
            drain_event.clear()
            rec_writers: list[threading.Thread] = []
            for _ in range(max(1, (os.cpu_count() or 1) - 1)):
                t = threading.Thread(target=writer_thread)
                t.start()
                rec_writers.append(t)
            success = False

            # Process Zones
            try:
                for z_idx, zone_data in parsed_detections.items():
                    if not zone_data:
                        continue
                    groups: list[Any] = []
                    current_group: list[Any] = []
                    current_union_rects: list[list[float]] = []
                    for _, polys, frame_score, m in zone_data:
                        if not polys or len(polys) == 0:
                            frames_deleted_count += 1
                            frames_processed += 1
                            coord = (m["frame_idx"], z_idx)
                            if coord not in empty_frames_meta:
                                empty_frames_meta.add(coord)
                            if frames_processed >= next_print_target:
                                print(
                                    f"\rAnalyzing frame {frames_processed} of {total_stitched_frames}",
                                    end="",
                                    flush=True,
                                )
                                next_print_target = frames_processed + 15
                            continue
                        line_rects = utils.get_line_rects(polys)
                        if not current_group:
                            current_group = [
                                (m["frame_idx"], line_rects, frame_score, m)
                            ]
                            current_union_rects = line_rects
                        else:
                            if utils.are_rect_lists_similar(
                                current_union_rects, line_rects, tolerance=0.05
                            ):
                                current_group.append(
                                    (m["frame_idx"], line_rects, frame_score, m)
                                )
                                new_unions: list[list[float]] = []
                                for u_rect, l_rect in zip(
                                    current_union_rects, line_rects
                                ):
                                    new_unions.append(
                                        [
                                            min(u_rect[0], l_rect[0]),
                                            min(u_rect[1], l_rect[1]),
                                            max(u_rect[2], l_rect[2]),
                                            max(u_rect[3], l_rect[3]),
                                        ]
                                    )
                                current_union_rects = new_unions
                            else:
                                groups.append((current_union_rects, current_group))
                                current_group = [
                                    (m["frame_idx"], line_rects, frame_score, m)
                                ]
                                current_union_rects = line_rects
                    if current_group:
                        groups.append((current_union_rects, current_group))

                    # SSIM & Repacking
                    MAX_GRIDS_PER_CHUNK = 30
                    chunks: list[tuple[list[Any], set[str]]] = []
                    current_chunk_groups: list[Any] = []
                    current_chunk_grids: set[str] = set()
                    for union_rects, group_frames in groups:
                        group_grids = set()
                        for _, _, _, m in group_frames:
                            group_grids.add(m["grid_file"])
                        if (
                            len(current_chunk_grids | group_grids) > MAX_GRIDS_PER_CHUNK
                            and current_chunk_groups
                        ):
                            chunks.append((current_chunk_groups, current_chunk_grids))
                            current_chunk_groups = []
                            current_chunk_grids = set()
                        current_chunk_groups.append((union_rects, group_frames))
                        current_chunk_grids.update(group_grids)
                    if current_chunk_groups:
                        chunks.append((current_chunk_groups, current_chunk_grids))

                    for chunk_groups, chunk_grids in chunks:
                        loaded_grids: dict[str, Any] = {}
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            for g_file, img_array in executor.map(
                                utils.load_grid, chunk_grids
                            ):
                                loaded_grids[g_file] = img_array
                        group_args = [
                            (
                                union_rects,
                                group_frames,
                                loaded_grids,
                                TIGHT_BOX_SSIM_THRESHOLD,
                            )
                            for union_rects, group_frames in chunk_groups
                        ]
                        for ssim_args in group_args:
                            if z_idx >= 2 and not label_ssim_dedup:
                                # Label bands without dedup: keep every detected
                                # frame so label appearances are not lost.
                                _union_rects, group_frames, loaded_grids, _thr = (
                                    ssim_args
                                )
                                surviving_items = []
                                for _fidx, _lr, _score, m in group_frames:
                                    grid_img = loaded_grids[m["grid_file"]]
                                    gh, gw = grid_img.shape[:2]
                                    y1 = min(int(m["y"]), max(0, gh - 1))
                                    x1 = min(int(m["x"]), max(0, gw - 1))
                                    y2 = min(int(m["y"]) + int(m["h"]), gh)
                                    x2 = min(int(m["x"]) + int(m["w"]), gw)
                                    img = grid_img[y1:y2, x1:x2]
                                    surviving_items.append(
                                        {
                                            "img": img.copy(),
                                            "frame_idx": m["frame_idx"],
                                            "det_score": _score,
                                        }
                                    )
                                local_deleted = 0
                            else:
                                surviving_items, local_deleted = (
                                    utils.process_ssim_group(
                                        *ssim_args,
                                        frame_weights=(
                                            self._label_hash_weights
                                            if z_idx >= 2
                                            else None
                                        ),
                                    )
                                )
                                frames_deleted_count += local_deleted
                                if z_idx >= 2:
                                    # Label band: SSIM-dedup keeps one
                                    # representative per visually-identical run.
                                    # Record how many frames each survivor stands
                                    # for so confirmation counts survive dedup.
                                    for item in surviving_items:
                                        self._label_frame_weights[item["frame_idx"]] = item.get("weight", 1)
                            for item in surviving_items:
                                surviving_frames_meta.add((item["frame_idx"], z_idx))
                                filename = f"rec_image_{rec_counter:0{FILENAME_ZERO_PADDING}d}_zone{z_idx}.jpg"
                                route_to_label_dir = (
                                    self._label_split_lang and z_idx >= 2
                                )
                                zone_dir = (
                                    label_rec_images_dir
                                    if route_to_label_dir
                                    else rec_images_dir
                                )
                                filepath = os.path.join(zone_dir, filename)
                                h, w = item["img"].shape[:2]
                                write_queue.put((filepath, w, h, [(item["img"], 0, 0)]))
                                rec_image_map[filename] = {
                                    "frame_idx": item["frame_idx"],
                                    "zone_idx": z_idx,
                                }
                                if route_to_label_dir:
                                    label_rec_written += 1
                                rec_counter += 1
                            frames_processed += len(ssim_args[1])
                            if frames_processed >= next_print_target:
                                print(
                                    f"\rAnalyzing frame {frames_processed} of {total_stitched_frames}",
                                    end="",
                                    flush=True,
                                )
                                next_print_target = frames_processed + 15
                        loaded_grids.clear()
                print(
                    f"\rAnalyzing frame {frames_processed} of {total_stitched_frames}",
                    end="",
                    flush=True,
                )
                success = True
            except KeyboardInterrupt:
                raise
            finally:
                is_aborting = not success or len(error_list) > 0
                if is_aborting:
                    stop_event.set()
                else:
                    drain_event.set()
                if is_aborting:
                    while not write_queue.empty():
                        try:
                            write_queue.get_nowait()
                        except queue.Empty:
                            break
                for w in rec_writers:
                    w.join()
                if error_list:
                    raise error_list[0]

            print(
                f"\nFiltered out {frames_deleted_count} redundant frame(s) via Text-Detection and tight-box SSIM analysis.",
                flush=True,
            )
            if rec_counter == 0:
                self.label_zones = []
                self.label_frames = []
                return

            # --------------------------------------------------------
            # Recognition Pass
            # --------------------------------------------------------
            rec_ocr_outputs: dict[str, list[Any]] = {}
            ocr_image_index = 0
            if ocr_engine == "google_lens":

                def _run_lens(input_dir: str, code: str) -> None:
                    nonlocal ocr_image_index
                    lens_args = [
                        self.google_lens_path,
                        input_dir,
                        code,
                        "--get-coords",
                        "--oneline",
                        "-q",
                    ]
                    for line in utils.stream_cli_process(
                        lens_args, "google_lens_error.log"
                    ):
                        line = line.strip()
                        if (
                            not line
                            or not line.startswith("{")
                            or '"file"' not in line
                        ):
                            continue
                        data = json.loads(line)
                        stitched_filename = data["file"]
                        if stitched_filename not in rec_image_map:
                            continue
                        grid_w = data["dimensions"]["original_width"]
                        grid_h = data["dimensions"]["original_height"]
                        results: list[list[Any]] = []
                        for word_item in data["words"]:
                            text = word_item["text"]
                            separator = word_item["separator"]
                            geom = word_item["geometry"]
                            if not geom:
                                continue
                            cx = geom["center_x"] * grid_w
                            cy = geom["center_y"] * grid_h
                            w = geom["width"] * grid_w
                            h = geom["height"] * grid_h
                            x1, x2 = cx - w / 2.0, cx + w / 2.0
                            y1, y2 = cy - h / 2.0, cy + h / 2.0
                            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            combined_text = text + separator
                            results.append([box, (combined_text, 1.0)])
                        rec_ocr_outputs[stitched_filename] = results
                        ocr_image_index += 1
                        print(
                            f"\rStep 3/3: Performing OCR on image {ocr_image_index} of {len(rec_image_map)}",
                            end="",
                            flush=True,
                        )

                print("Starting Google Lens CLI...", flush=True)
                _run_lens(rec_images_dir, self.lang)
                if self._label_split_lang and label_rec_written > 0:
                    print(
                        "Starting Google Lens CLI label pass "
                        f"(lang={self._label_lang})...",
                        flush=True,
                    )
                    _run_lens(label_rec_images_dir, self._label_lang)
                print()
            elif ocr_engine == "paddleocr":

                def _run_paddle(input_dir: str, code: str, model_dirs: dict[str, str]) -> None:
                    nonlocal ocr_image_index
                    rec_args = [
                        self.paddleocr_path,
                        "ocr",
                        "--input",
                        input_dir,
                        "--device",
                        "gpu" if use_gpu else "cpu",
                        "--use_textline_orientation",
                        "true" if use_angle_cls else "false",
                        "--use_doc_orientation_classify",
                        "false",
                        "--use_doc_unwarping",
                        "false",
                        "--lang",
                        code,
                        "--text_detection_model_dir",
                        model_dirs["det"],
                        "--text_detection_model_name",
                        os.path.basename(model_dirs["det"]),
                        "--text_recognition_model_dir",
                        model_dirs["rec"],
                        "--text_recognition_model_name",
                        os.path.basename(model_dirs["rec"]),
                    ]
                    if use_angle_cls:
                        rec_args += [
                            "--textline_orientation_model_dir",
                            model_dirs["cls"],
                        ]
                        rec_args += [
                            "--textline_orientation_model_name",
                            os.path.basename(model_dirs["cls"]),
                        ]
                    current_image = None
                    for line in utils.stream_cli_process(
                        rec_args, "paddleocr_error.log"
                    ):
                        line = line.strip()
                        if "ppocr INFO: **********" in line:
                            match = re.search(r"\*+(.+?)\*+$", line)
                            if match:
                                current_image = os.path.basename(
                                    match.group(1)
                                ).strip()
                                rec_ocr_outputs[current_image] = []
                                ocr_image_index += 1
                                print(
                                    f"\rStep 3/3: Performing OCR on image {ocr_image_index} of {len(rec_image_map)}",
                                    end="",
                                    flush=True,
                                )
                        elif current_image and "[[" in line:
                            try:
                                match = re.search(
                                    r"ppocr INFO:\s*(\[.+\])", line
                                )
                                if match:
                                    parsed = ast.literal_eval(match.group(1))
                                    rec_ocr_outputs[current_image].append(parsed)
                            except Exception as e:
                                print(
                                    f"Error parsing OCR for {current_image}: {e}",
                                    flush=True,
                                )

                print("Starting PaddleOCR...", flush=True)
                main_dirs = {
                    "det": self.det_model_dir,
                    "rec": self.rec_model_dir,
                    "cls": self.cls_model_dir,
                }
                _run_paddle(rec_images_dir, self.lang, main_dirs)
                if self._label_split_lang and label_rec_written > 0:
                    print(
                        "Starting PaddleOCR label pass "
                        f"(lang={self._label_lang})...",
                        flush=True,
                    )
                    label_dirs = self._label_model_dirs or main_dirs
                    _run_paddle(
                        label_rec_images_dir, self._label_lang, label_dirs
                    )
                print()

            # Map 2D coordinates
            ocr_outputs: dict[tuple[int, int], list[Any]] = {}
            for filename, results in rec_ocr_outputs.items():
                if filename not in rec_image_map:
                    continue
                m = rec_image_map[filename]
                coord_key = (m["frame_idx"], m["zone_idx"])
                if coord_key not in ocr_outputs:
                    ocr_outputs[coord_key] = []
                for word_pred in results:
                    ocr_outputs[coord_key].append([word_pred[0], word_pred[1]])

            active_frame_coords = surviving_frames_meta | empty_frames_meta
            frame_predictions_dict: dict[int, dict[int, PredictedFrames]] = {
                0: {},
                1: {},
            }
            for k in range(len(self.label_zones)):
                frame_predictions_dict[2 + k] = {}
            for frame_index, zone_index in active_frame_coords:
                ocr_result = ocr_outputs.get((frame_index, zone_index), [])
                pred_data = [ocr_result] if ocr_result else [[]]
                # Fix 7: label bands use their own (lower) confidence gate.
                zone_conf = (
                    self._label_conf_threshold_ratio
                    if zone_index >= 2
                    else conf_threshold_ratio
                )
                predicted_frame = PredictedFrames(
                    ocr_engine,
                    frame_index,
                    pred_data,
                    zone_conf,
                    zone_index,
                    self._label_lang if zone_index >= 2 else lang,
                    normalize_to_simplified_chinese,
                )
                frame_predictions_dict[zone_index][frame_index] = predicted_frame
            frame_predictions_list: dict[int, list[PredictedFrames]] = {}
            for zone_idx in frame_predictions_dict:
                frames = sorted(
                    frame_predictions_dict[zone_idx].values(),
                    key=lambda f: f.start_index,
                )
                if not frames:
                    continue
                for i in range(len(frames) - 1):
                    current_pred = frames[i]
                    next_pred = frames[i + 1]
                    current_pred.end_index = next_pred.start_index - 1
                if frames:
                    frames[-1].end_index = ocr_end - 1
                frame_predictions_list[zone_idx] = frames

            self.pred_frames_zone1 = frame_predictions_list.get(0, [])
            self.pred_frames_zone2 = frame_predictions_list.get(1, [])
            self.label_frames = (
                self._collect_label_frames(frame_predictions_list)
                if self.label_zones
                else []
            )

            total_elapsed = time.perf_counter() - perf_total_start
            print(
                f"[Perf] Step 1 frame scan/filter/stitch: {step1_end - step1_start:.2f}s",
                flush=True,
            )
            print(
                f"[Perf] Total OCR preparation/runtime before subtitle merge: {total_elapsed:.2f}s",
                flush=True,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_subtitles(
        self,
        sim_threshold: int,
        max_merge_gap_sec: float,
        lang: str,
        post_processing: bool,
        min_subtitle_duration_sec: float,
        subtitle_alignments: list[str | None],
    ) -> str:
        self._generate_subtitles(
            sim_threshold,
            max_merge_gap_sec,
            lang,
            post_processing,
            min_subtitle_duration_sec,
            subtitle_alignments,
        )
        # Label detection always produces ASS output with proper ASS timestamps
        # and \pos-positioned label events.
        if self.label_zones:
            return self._render_ass(subtitle_alignments)
        srt_lines: list[str] = []
        for i, sub in enumerate(self.pred_subs, 1):
            start_time, end_time = self._get_srt_timestamps(sub)
            text = sub.text
            tag = subtitle_alignments[sub.zone_index]
            if tag:
                text = f"{{\\{tag}}}{sub.text}"
            srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        return "".join(srt_lines)

    def _render_ass(self, subtitle_alignments: list[str | None]) -> str:
        """Render the final .ass file: dialogue + label events.

        Label events are positioned at the actual on-screen location of the
        detected text (mapped back from the label zone crop to the original
        frame coordinates) using ASS \\pos.
        """

        def _esc(text: str) -> str:
            # Collapse newlines (an ASS event must be a single line) and escape
            # ASS override-block / backslash characters. Backslash is escaped
            # first so the escapes introduced for { } % are not re-escaped.
            text = re.sub(r"[\r\n]+", " ", text)
            return (
                text.replace("\\", "\\\\")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("%", "\\%")
            )

        def _ass_time(ms: float) -> str:
            return utils.get_ass_timestamp_from_ms(max(0.0, ms))

        lines: list[str] = []
        lines.append("[Script Info]")
        lines.append("ScriptType: v4.00+")
        lines.append("WrapStyle: 2")
        lines.append("ScaledBorderAndShadow: yes")
        lines.append("")
        lines.append("[V4+ Styles]")
        lines.append(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        )
        lines.append(
            "Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1"
        )
        style = getattr(self, "_label_style", {})
        label_font = str(style.get("font", "Arial")).replace(",", "")
        label_fontsize = int(style.get("fontsize", 22))
        label_primary = str(style.get("primary", "&H00FFFFFF"))
        label_outline = str(style.get("outline", "&H00000000"))
        label_align = int(str(style.get("alignment", "an7")).removeprefix("an") or 7)
        lines.append(
            f"Style: Label,{label_font},{label_fontsize},{label_primary},"
            f"&H000000FF,{label_outline},&H80000000,0,0,0,0,100,100,0,0,1,1.5,0.5,"
            f"{label_align},10,10,10,1"
        )
        lines.append("")
        lines.append("[Events]")
        lines.append(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        )

        # Dialogue events (existing subtitle pipeline, zone 0/1).
        for sub in self.pred_subs:
            start_ms, end_ms = self._get_subtitle_ms_times(sub)
            tag = subtitle_alignments[sub.zone_index]
            text = sub.text
            if tag:
                text = f"{{\\{tag}}}{text}"
            lines.append(
                f"Dialogue: 0,{_ass_time(start_ms)},{_ass_time(end_ms)},Default,,0,0,0,,{_esc(text)}"
            )

        # Label events: cluster per label text and position each at its actual
        # on-screen location (original video coordinates). Each event maps
        # back through its own band's crop origin/scale, then colliding
        # events are shifted apart vertically.
        render_items: list[dict[str, Any]] = []
        for label in self._merge_label_frames():
            band = self._label_band(label["zone_index"])
            if band is None:
                continue
            # Use the clipped crop origin (crop_x/crop_y) — matches the
            # coordinates the OCR actually saw.
            zone_x = int(band["crop_x"])
            zone_y = int(band["crop_y"])
            # Scale factors: label OCR crop size -> original frame coordinates.
            sx = band["crop_w"] / band["target_w"] if band["target_w"] else 1.0
            sy = band["crop_h"] / band["target_h"] if band["target_h"] else 1.0
            bx1, by1, bx2, by2 = label["bbox"]
            render_items.append(
                {
                    "start_ms": label["start_ms"],
                    "end_ms": label["end_ms"],
                    "text": label["text"],
                    "fx1": zone_x + bx1 * sx,
                    "fy1": zone_y + by1 * sy,
                    "fx2": zone_x + bx2 * sx,
                    "fy2": zone_y + by2 * sy,
                }
            )
        utils.resolve_label_overlaps(
            render_items,
            self.height,
            # Scale the gap with the label font size so large labels still
            # read as separate events; the floor keeps the 4 px default at
            # the standard 22 pt.
            margin=max(4.0, label_fontsize / 6.0),
        )
        for item in render_items:
            # \pos carries the bbox corner/edge/center matching the Label
            # style's \anN alignment, so the text renders exactly in place.
            anchor_x, anchor_y = utils.ass_anchor_point(
                label_align, item["fx1"], item["fy1"], item["fx2"], item["fy2"]
            )
            pos_x = int(round(anchor_x))
            pos_y = int(round(anchor_y))
            lines.append(
                f"Dialogue: 2,{_ass_time(item['start_ms'])},{_ass_time(item['end_ms'])},Label,,0,0,0,,"
                f"{{\\pos({pos_x},{pos_y})}}{_esc(item['text'])}"
            )
        return "\n".join(lines) + "\n"

    def _merge_label_frames(self) -> list[dict[str, Any]]:
        """Cluster raw label detections into stable label events.

        OCR text for the same on-screen label is unstable between frames
        (e.g. "乔" vs "乔羽", "主" vs "楼城城主", "楼城少城丰" vs "楼城少城主"),
        so events are merged by text similarity AND position proximity rather
        than exact equality. Events never span label bands — a frame only
        matches an event from its own band (zone_index).

        Returns a list of dicts with start_ms/end_ms/text/cx/cy/zone_index
        where cx,cy is the average label position in band crop coordinates,
        plus "bbox" — the averaged (x1,y1,x2,y2) text bounding box used for
        ASS \\pos anchoring.
        """
        from thefuzz import fuzz  # type: ignore

        frames = sorted(self.label_frames, key=lambda f: f.start_index)
        if not frames:
            return []

        # Fix 3: configurable thresholds (previously hardcoded constants).
        TEXT_SIM_THRESHOLD = float(
            getattr(self, "_label_text_similarity_threshold", 55)
        )
        POS_DRIFT = float(getattr(self, "_label_pos_drift_px", 160.0))
        CLOSE_POS_DIST = float(getattr(self, "_label_close_pos_distance_px", 40.0))
        CLOSE_POS_LEN_RATIO = float(getattr(self, "_label_close_pos_length_ratio", 0.5))

        def _line_text_and_pos(line: list[Any]) -> _LabelLine:
            text = "".join(w.text for w in line).strip()
            xs: list[float] = []
            ys: list[float] = []
            confs: list[float] = []
            for w in line:
                xs.extend(p[0] for p in w.bounding_box)
                ys.extend(p[1] for p in w.bounding_box)
                confs.append(float(w.confidence))
            cx = sum(xs) / len(xs) if xs else 0.0
            cy = sum(ys) / len(ys) if ys else 0.0
            return _LabelLine(
                text=text,
                cx=cx,
                cy=cy,
                conf=sum(confs) / len(confs) if confs else 1.0,
                x1=min(xs) if xs else 0.0,
                y1=min(ys) if ys else 0.0,
                x2=max(xs) if xs else 0.0,
                y2=max(ys) if ys else 0.0,
            )

        def _band_scale(zone_idx: int) -> tuple[float, float]:
            # Per-band crop->frame scale so POS_DRIFT is in *original video*
            # coordinates regardless of which band the OCR crop came from.
            band = self._label_band(zone_idx)
            if band is None:
                return 1.0, 1.0
            cw = int(band.get("crop_w") or 1)
            tw = int(band.get("target_w") or 1)
            ch = int(band.get("crop_h") or 1)
            th = int(band.get("target_h") or 1)
            return (cw / tw if tw > 0 else 1.0, ch / th if th > 0 else 1.0)

        def _drift(ev: dict[str, Any], cx: float, cy: float) -> float:
            sx, sy = _band_scale(ev["zone_index"])
            dx = (ev["cx"] - cx) * sx
            dy = (ev["cy"] - cy) * sy
            return (dx * dx + dy * dy) ** 0.5

        # A label may have multiple simultaneous text blocks (e.g. person name
        # "乔羽" plus title "楼城城主"), so several events can be "open" at the
        # same time. Each frame contributes every detected line to the best
        # matching open event; unmatched lines open new events.
        events: list[dict[str, Any]] = []

        def _close_event(ev: dict[str, Any]) -> None:
            ev["text"] = self._pick_best_label_text(ev)
            count = ev.get("count") or 1
            ev["bbox"] = (
                ev["x1_sum"] / count,
                ev["y1_sum"] / count,
                ev["x2_sum"] / count,
                ev["y2_sum"] / count,
            )

        def _open_event(info: _LabelLine, frame: Any, weight: int) -> dict[str, Any]:
            ev = {
                "start_ms": self._label_frame_start_ms(frame),
                "end_ms": self._label_frame_end_ms(frame),
                "text": info.text,
                "cx": info.cx,
                "cy": info.cy,
                "zone_index": frame.zone_index,
                # Votes weighted by run length x confidence so a noisy low-conf
                # reading counts less than a stable high-conf one.
                "votes": {_label_text_norm(info.text): weight * info.conf},
                "text_samples": [info.text],
                "cx_sum": info.cx * weight,
                "cy_sum": info.cy * weight,
                "x1_sum": info.x1 * weight,
                "y1_sum": info.y1 * weight,
                "x2_sum": info.x2 * weight,
                "y2_sum": info.y2 * weight,
                "count": weight,
                "last_frame_idx": frame.start_index,
                "_matched_this_frame": True,
            }
            events.append(ev)
            return ev

        gap_frames = self._label_reappear_gap_frames()
        for frame in frames:
            if not frame.lines:
                continue
            # Fix 2: reset the per-frame match guard up-front so it never
            # carries over between frames. Within a frame it still prevents a
            # single line from matching the same event twice.
            for ev in events:
                ev["_matched_this_frame"] = False
            # Number of original frames this (SSIM-deduped) frame represents,
            # so confirmation counts stay correct after dedup.
            weight = self._label_frame_weights.get(frame.start_index, 1)
            for line in frame.lines:
                info = _line_text_and_pos(line)
                text, cx, cy = info.text, info.cx, info.cy
                if not text:
                    continue
                best_ev: dict[str, Any] | None = None
                best_score: float = -1.0
                for ev in events:
                    # Bands are disjoint: a frame only ever matches an event
                    # from its own band.
                    if ev["zone_index"] != frame.zone_index:
                        continue
                    # Skip events whose last detection is older than the
                    # reappear-merge gap (they belong to an earlier, separate
                    # appearance), and skip events this frame already extended
                    # (one frame line per event).
                    if ev.get("last_frame_idx", -1) < frame.start_index - gap_frames:
                        continue
                    if ev.get("last_frame_idx") == frame.start_index and ev.get(
                        "_matched_this_frame"
                    ):
                        continue
                    d = _drift(ev, cx, cy)
                    if d > POS_DRIFT:
                        continue
                    ev_norm = _label_text_norm(ev["text"])
                    text_norm = _label_text_norm(text)
                    same_text = ev_norm == text_norm
                    # Compare against every known variant of the event (not just
                    # its current representative text): OCR may have seen both
                    # "主" and "楼城城主" for the same label, and a later frame
                    # reading "楼城城" must still match via "楼城城主".
                    sim = max(
                        max(
                            fuzz.ratio(ev_norm, text_norm),
                            fuzz.partial_ratio(ev_norm, text_norm),
                        ),
                        max(
                            (
                                max(
                                    fuzz.ratio(_label_text_norm(s), text_norm),
                                    fuzz.partial_ratio(_label_text_norm(s), text_norm),
                                )
                                for s in ev.get("text_samples", [])
                            ),
                            default=0.0,
                        ),
                    )
                    score = 100.0 if same_text else sim
                    # Very close positions almost certainly mean the same label
                    # even when OCR splits/merges characters ("主" vs
                    # "楼城城主" at the same spot). Use a length-ratio guard so
                    # a real 2-char label at a coincidentally close spot is not
                    # merged with an unrelated longer one.
                    if d <= CLOSE_POS_DIST:
                        short_len = min(len(text_norm), len(ev_norm))
                        long_len = max(len(text_norm), len(ev_norm))
                        if long_len > 0 and short_len / long_len >= CLOSE_POS_LEN_RATIO:
                            score = max(score, 85.0)
                    # Slight position bonus: closer detections are more likely
                    # the same label.
                    score -= d / POS_DRIFT * 20.0
                    if score > best_score:
                        best_score = score
                        best_ev = ev
                if best_ev is not None and best_score >= TEXT_SIM_THRESHOLD:
                    # Extend the matched event (weighted by the dedup run
                    # length and the line's OCR confidence).
                    best_ev["votes"][_label_text_norm(text)] = (
                        best_ev["votes"].get(_label_text_norm(text), 0)
                        + weight * info.conf
                    )
                    best_ev["text_samples"].append(text)
                    best_ev["end_ms"] = self._label_frame_end_ms(frame)
                    best_ev["cx_sum"] += cx * weight
                    best_ev["cy_sum"] += cy * weight
                    best_ev["x1_sum"] += info.x1 * weight
                    best_ev["y1_sum"] += info.y1 * weight
                    best_ev["x2_sum"] += info.x2 * weight
                    best_ev["y2_sum"] += info.y2 * weight
                    best_ev["count"] += weight
                    best_ev["cx"] = best_ev["cx_sum"] / best_ev["count"]
                    best_ev["cy"] = best_ev["cy_sum"] / best_ev["count"]
                    best_ev["last_frame_idx"] = frame.start_index
                    best_ev["_matched_this_frame"] = True
                else:
                    _open_event(info, frame, weight)

        for ev in events:
            _close_event(ev)

        # Minimum confirmation frames: drop events that only appeared once
        # (or fewer than the configured threshold) — usually OCR noise.
        min_conf = getattr(self, "_label_min_confirmation_frames", 2)
        events = [ev for ev in events if ev.get("count", 1) >= min_conf]

        # Optional single-ASCII-character noise filter ("M", "3", "L", ...).
        # CJK single characters are kept (a person's name may be one char).
        if getattr(self, "_label_filter_single_char", True):
            events = [ev for ev in events if not _is_single_ascii_label(ev["text"])]

        # Enforce a minimum display duration for readability. Guard against
        # degenerate timestamps by clamping end >= start + one frame duration.
        min_ms = getattr(self, "_label_min_display_duration_ms", 1000.0)
        for ev in events:
            if ev["end_ms"] - ev["start_ms"] < min_ms:
                ev["end_ms"] = max(ev["start_ms"] + min_ms, ev["end_ms"])

        # Clamp events to the label-detection time window so a label never
        # starts before its window opens or outlives its window close.
        win_start = float(getattr(self, "label_time_start_ms", 0.0))
        win_end = getattr(self, "label_time_end_ms", None)
        if win_start > 0.0 or win_end is not None:
            clamped: list[dict[str, Any]] = []
            for ev in events:
                if (
                    win_end is not None
                    and ev["start_ms"] > win_end - self.start_time_offset_ms
                ):
                    continue  # event starts after the window closed
                start_ms = max(ev["start_ms"], win_start - self.start_time_offset_ms)
                end_ms = ev["end_ms"]
                if win_end is not None:
                    end_ms = min(end_ms, win_end - self.start_time_offset_ms)
                if end_ms <= start_ms:
                    continue
                ev["start_ms"], ev["end_ms"] = start_ms, end_ms
                clamped.append(ev)
            events = clamped
        return events

    @staticmethod
    def _pick_best_label_text(event: dict[str, Any]) -> str:
        """Pick the most stable OCR text for a label event.

        Votes are weighted by frame-run length x OCR confidence, so frequent
        high-confidence readings beat occasional noisy ones; ties go to the
        longest variant (which usually carries the full label, e.g. prefer
        "楼城城主" over "主").
        """
        votes = event.get("votes", {})
        if not votes:
            samples = event.get("text_samples", [])
            return max(samples, key=len) if samples else ""
        best_count = max(votes.values())
        candidates = [t for t, c in votes.items() if c == best_count]
        # Fix 5: group original samples by their normalized form so a key
        # shared by several originals (e.g. "乔 羽" and "乔羽") keeps every
        # variant instead of silently dropping all but the last one.
        samples_by_norm: dict[str, list[str]] = {}
        for s in event.get("text_samples", []):
            samples_by_norm.setdefault(_label_text_norm(s), []).append(s)
        originals: list[str] = []
        for norm in candidates:
            variants = samples_by_norm.get(norm)
            if variants:
                # Longest variant usually carries the full label.
                originals.append(max(variants, key=len))
            else:
                originals.append(norm)
        return max(originals, key=len)

    def _label_reappear_gap_frames(self) -> int:
        """Frames a label event can stay "open" for reappearance merging.

        Derived from the configured reappear-merge gap (in seconds) and the
        average sampled-frame duration. Falls back to a 20-frame window.
        """
        gap_ms = getattr(self, "_label_reappear_merge_gap_ms", 0.0)
        if gap_ms <= 0:
            return 20
        frame_ms = self._frame_duration_ms()
        return max(1, int(round(gap_ms / frame_ms)))

    def _filter_label_frames_by_time(
        self, frames: list[PredictedFrames]
    ) -> list[PredictedFrames]:
        """Drop label detections outside the label-detection time window.

        The window is set via --label_time_start/--label_time_end and is
        independent of the subtitle extraction window. Timestamps come from
        frame_timestamps (container time, including start_time_offset), so the
        comparison uses the offset-adjusted label window.
        """
        start_ms = float(getattr(self, "label_time_start_ms", 0.0))
        end_ms = getattr(self, "label_time_end_ms", None)
        if start_ms <= 0.0 and end_ms is None:
            return frames
        kept: list[PredictedFrames] = []
        for frame in frames:
            ts = self.frame_timestamps.get(frame.start_index)
            if ts is None:
                # Fix 8.4: estimate the timestamp from the frame index instead
                # of unconditionally keeping the frame, so a sparse map cannot
                # leak out-of-window frames. Only fall back to keeping when no
                # estimate is possible at all.
                ts = self._estimate_frame_timestamp_ms(frame.start_index)
                if ts is None:
                    kept.append(frame)
                    continue
            if start_ms > 0.0 and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            kept.append(frame)
        return kept

    def _label_frame_start_ms(self, frame: Any) -> float:
        # frame_timestamps is normally fully populated by the producer, but
        # fall back to the frame's end timestamp (and then 0) defensively so a
        # sparse map never yields a spurious 0 start that the label time-window
        # clamp would misinterpret as "before the window".
        start_ms = self.frame_timestamps.get(
            frame.start_index,
            self.frame_timestamps.get(frame.end_index, 0),
        )
        return start_ms - self.start_time_offset_ms

    def _label_frame_end_ms(self, frame: Any) -> float:
        end_ms = self.frame_timestamps.get(frame.end_index + 1)
        if end_ms is None:
            # The frame after the last detection is not in the sampled map.
            # Fix 8.3: derive the one-frame floor from the actual FPS /
            # measured frame duration instead of a hardcoded 33 ms.
            start_ms = self.frame_timestamps.get(frame.start_index, 0)
            last_ms = self.frame_timestamps.get(frame.end_index, start_ms)
            floor_ms = self._frame_duration_ms()
            end_ms = max(last_ms + floor_ms, start_ms + floor_ms)
        return end_ms - self.start_time_offset_ms

    def _generate_subtitles(
        self,
        sim_threshold: int,
        max_merge_gap_sec: float,
        lang: str,
        post_processing: bool,
        min_subtitle_duration_sec: float,
        subtitle_alignments: list[str | None],
    ) -> None:
        print("Generating subtitles...", flush=True)
        subs_zone1 = self._process_single_zone(
            self.pred_frames_zone1,
            sim_threshold,
            max_merge_gap_sec,
            lang,
            post_processing,
            min_subtitle_duration_sec,
        )
        subs_zone2 = self._process_single_zone(
            self.pred_frames_zone2,
            sim_threshold,
            max_merge_gap_sec,
            lang,
            post_processing,
            min_subtitle_duration_sec,
        )
        if subs_zone1 and not subs_zone2:
            self.pred_subs = subs_zone1
        elif not subs_zone1 and subs_zone2:
            self.pred_subs = subs_zone2
        elif subs_zone1 and subs_zone2:
            if subtitle_alignments[0] != subtitle_alignments[1]:
                self.pred_subs = sorted(
                    subs_zone1 + subs_zone2, key=lambda s: s.index_start
                )
            else:
                self.pred_subs = self._merge_dual_zone_subtitles(subs_zone1, subs_zone2)
        else:
            self.pred_subs = []

    def _process_single_zone(
        self,
        pred_frames: list[PredictedFrames],
        sim_threshold: int,
        max_merge_gap_sec: float,
        lang: str,
        post_processing: bool,
        min_subtitle_duration_sec: float,
    ) -> list[PredictedSubtitle]:
        if not pred_frames:
            return []
        language_mapping = {
            "en": "en",
            "fr": "fr",
            "german": "de",
            "it": "it",
            "es": "es",
            "pt": "pt",
        }
        language_model = None
        if post_processing:
            if lang in language_mapping:
                language_model = wordninja.LanguageModel(
                    language=language_mapping[lang]
                )
        subs: list[PredictedSubtitle] = []
        for frame in sorted(pred_frames, key=lambda f: f.start_index):
            new_sub = PredictedSubtitle(
                [frame], frame.zone_index, sim_threshold, lang, language_model
            )
            if not new_sub.text:
                continue
            if subs:
                last_sub = subs[-1]
                if self._is_gap_mergeable(
                    last_sub, new_sub, max_merge_gap_sec
                ) and last_sub.is_similar_to(new_sub):
                    last_sub.frames.extend(new_sub.frames)
                    last_sub.frames.sort(key=lambda f: f.start_index)
                else:
                    subs.append(new_sub)
            else:
                subs.append(new_sub)
        for sub in subs:
            sub.finalize_text(post_processing)
        # Filter out subs that are too short
        filtered_subs = [
            sub
            for sub in subs
            if self._get_subtitle_duration_sec(sub) >= min_subtitle_duration_sec
        ]
        if not filtered_subs:
            return []
        # Re-merge the cleaned-up list of subtitles if applicable
        cleaned_subs = [filtered_subs[0]]
        for next_sub in filtered_subs[1:]:
            last_sub = cleaned_subs[-1]
            if self._is_gap_mergeable(
                last_sub, next_sub, max_merge_gap_sec
            ) and last_sub.is_similar_to(next_sub):
                last_sub.frames.extend(next_sub.frames)
                last_sub.frames.sort(key=lambda f: f.start_index)
                last_sub.finalize_text(post_processing)
            else:
                cleaned_subs.append(next_sub)
        return cleaned_subs

    def _merge_dual_zone_subtitles(
        self, subs1: list[PredictedSubtitle], subs2: list[PredictedSubtitle]
    ) -> list[PredictedSubtitle]:
        all_subs = sorted(subs1 + subs2, key=lambda s: s.index_start)
        if not all_subs:
            return []
        merged_subs = [all_subs[0]]
        for current_sub in all_subs[1:]:
            last_sub = merged_subs[-1]
            if current_sub.index_start <= last_sub.index_end:
                last_zone_info = self.validated_zones[last_sub.zone_index]
                current_zone_info = self.validated_zones[current_sub.zone_index]
                if current_zone_info["midpoint_y"] < last_zone_info["midpoint_y"]:
                    last_sub.text = f"{current_sub.text}\n{last_sub.text}"
                else:
                    last_sub.text = f"{last_sub.text}\n{current_sub.text}"
                last_sub.frames.extend(current_sub.frames)
                last_sub.frames.sort(key=lambda f: f.start_index)
            else:
                merged_subs.append(current_sub)
        return merged_subs

    def _get_srt_timestamps(self, sub: PredictedSubtitle) -> tuple[str, str]:
        start_ms, end_ms = self._get_subtitle_ms_times(sub)
        start_time = utils.get_srt_timestamp_from_ms(start_ms)
        end_time = utils.get_srt_timestamp_from_ms(end_ms)
        return start_time, end_time

    def _get_subtitle_ms_times(self, sub: PredictedSubtitle) -> tuple[float, float]:
        start_time_ms = self.frame_timestamps.get(sub.index_start, 0)
        end_time_ms = self.frame_timestamps.get(sub.index_end + 1)
        # For the end time, we try to get the timestamp of the next frame, if it doesn't exist, we fall back to estimating duration of last frame
        if end_time_ms is None:
            last_frame_ms = self.frame_timestamps.get(sub.index_end, start_time_ms)
            end_time_ms = last_frame_ms + self.avg_frame_duration_ms
        # Apply the correction to align with the container's start time
        corrected_start_time_ms = start_time_ms - self.start_time_offset_ms
        corrected_end_time_ms = end_time_ms - self.start_time_offset_ms
        return corrected_start_time_ms, corrected_end_time_ms

    def _get_subtitle_duration_sec(self, sub: PredictedSubtitle) -> float:
        start_ms, end_ms = self._get_subtitle_ms_times(sub)
        return (end_ms - start_ms) / 1000

    def _is_gap_mergeable(
        self,
        last_sub: PredictedSubtitle,
        next_sub: PredictedSubtitle,
        max_merge_gap_sec: float,
    ) -> bool:
        _, last_end_ms = self._get_subtitle_ms_times(last_sub)
        next_start_ms, _ = self._get_subtitle_ms_times(next_sub)
        gap_ms = next_start_ms - last_end_ms
        return gap_ms <= (max_merge_gap_sec * 1000)
