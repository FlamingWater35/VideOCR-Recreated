from __future__ import annotations

import os
import sys
import time

from . import utils
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path: str = 'subtitle.srt', ocr_engine: str = 'google_lens', lang: str = 'en',
        time_start: str = '0:00', time_end: str = '', conf_threshold: int = 75, sim_threshold: int = 80, max_merge_gap_sec: float = 0.1,
        use_fullframe: bool = False, use_gpu: bool = False, use_angle_cls: bool = False, use_server_model: bool = False,
        brightness_threshold: int | None = None, ssim_threshold: int = 92, subtitle_position: str = "center", frames_to_skip: int = 1,
        crop_zones: list[dict[str, int]] | None = None, ocr_image_max_width: int = 720, post_processing: bool = False, min_subtitle_duration_sec: float = 0.2,
        normalize_to_simplified_chinese: bool = True, subtitle_alignments: list[str | None] | None = None,
        directml_grid_max_width: int = 2400, directml_grid_max_height: int = 2400,
        directml_performance_preset: str = "balanced", directml_recognition_mode: str = "stable",
        directml_frame_scan_mode: str = "cpu_ssim", onnx_directml_tuning: str = "balanced",
        benchmark_compare_engine: bool = False, benchmark_compare_sample_grids: int = 3,
        enable_label_detection: bool = False, label_ocr_image_max_width: int = 720,
        label_time_start: str = '', label_time_end: str = '',
        label_min_display_duration_sec: float = 1.0,
        label_min_confirmation_frames: int = 2,
        label_reappear_merge_gap_sec: float = 2.0,
        label_filter_single_char: bool = True) -> None:

    total_start = time.perf_counter()

    if crop_zones is None:
        crop_zones = []

    if subtitle_alignments is None:
        subtitle_alignments = [None, None]
    elif len(subtitle_alignments) == 1:
        subtitle_alignments.append(None)

    # Label detection (e.g. people/place names outside the subtitle crop area)
    # is an ASS-only feature: the final result is written as an .ass subtitle
    # file with proper ASS timestamps and \pos-positioned Label events.
    label_detection = bool(enable_label_detection)
    if label_detection:
        if file_path.lower().endswith(".srt"):
            file_path = file_path[:-4] + ".ass"
        print(f"Label detection enabled: output will be written as .ass subtitle file: {file_path}", flush=True)
        if os.path.isfile(file_path):
            print(f"Warning: overwriting existing .ass file: {file_path}", flush=True)

    # PaddleOCR and Chrome Lens are standalone helper executables.
    # EasyOCR DirectML is a pure-Python backend, so it must not require the
    # PaddleOCR CUDA/CPU bundle or the NVIDIA hardware check.
    paddleocr_path = ""
    google_lens_path = ""
    det_model_dir = ""
    rec_model_dir = ""
    cls_model_dir = ""

    if ocr_engine in ("paddleocr", "google_lens"):
        try:
            paddleocr_path = utils.find_executable("paddleocr")
        except FileNotFoundError as e:
            helper = "PaddleOCR" if ocr_engine == "paddleocr" else "PaddleOCR (text detection) and Chrome Lens (recognition)"
            install_hint = (
                "install the VideOCR Recreated PaddleOCR support files (see the build/install instructions)"
                if ocr_engine == "paddleocr"
                else "install the VideOCR Recreated PaddleOCR and Chrome Lens support files (see the build/install instructions)"
            )
            print(
                f"Error: {helper} helper executable not found.\n"
                f"{e}\n\n"
                f"To use the '{ocr_engine}' OCR engine, {install_hint}. "
                f"Alternatively select the 'ONNX Runtime DirectML (AMD GPU Experimental)' engine, which runs fully from "
                f"Python without external helper executables.",
                flush=True,
            )
            sys.exit(1)
        try:
            utils.perform_hardware_check(paddleocr_path, use_gpu)
        except SystemExit as e:
            print(e, flush=True)
            sys.exit(1)

        if ocr_engine == 'paddleocr':
            det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs(lang, use_server_model)
        else:
            # For the Text-Detection-Only Pass just the default detection model is needed
            det_model_dir, rec_model_dir, cls_model_dir = utils.resolve_model_dirs('en', use_server_model)

    if ocr_engine == "google_lens":
        try:
            google_lens_path = utils.find_executable("chrome-lens")
        except FileNotFoundError as e:
            print(
                f"Error: Chrome Lens helper executable not found.\n"
                f"{e}\n\n"
                f"To use the 'google_lens' OCR engine, install the VideOCR Recreated Chrome Lens support files. "
                f"Alternatively select the 'ONNX Runtime DirectML (AMD GPU Experimental)' engine, which runs fully from "
                f"Python without external helper executables.",
                flush=True,
            )
            sys.exit(1)

    v = Video(video_path, paddleocr_path, det_model_dir, rec_model_dir, cls_model_dir, google_lens_path)
    try:
        ocr_start = time.perf_counter()
        v.run_ocr(
            use_gpu, ocr_engine, lang, use_angle_cls, time_start, time_end, conf_threshold,
            use_fullframe, brightness_threshold, ssim_threshold, subtitle_position,
            frames_to_skip, crop_zones, ocr_image_max_width, normalize_to_simplified_chinese,
            directml_grid_max_width, directml_grid_max_height,
            directml_performance_preset, directml_recognition_mode, directml_frame_scan_mode,
            onnx_directml_tuning, benchmark_compare_engine, benchmark_compare_sample_grids,
            enable_label_detection, label_ocr_image_max_width, label_min_display_duration_sec,
            label_min_confirmation_frames, label_reappear_merge_gap_sec, label_filter_single_char,
            label_time_start, label_time_end
        )
        ocr_end = time.perf_counter()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    merge_start = time.perf_counter()
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec, subtitle_alignments)

    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)

    # Run the bundled ASS QA auto-fixer (ass_qafix) on the generated .ass file
    # as post-processing. Label detection always produces .ass output.
    if label_detection:
        qafix_errors: list[str] = []
        try:
            import subprocess as _sp
            import sys as _sys

            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "tools", "ass_qafix", "ass_qafix.py",
            )
            if not os.path.isfile(script):
                script = ""
                for root, _dirs, files in os.walk(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))):
                    if "ass_qafix.py" in files:
                        script = os.path.join(root, "ass_qafix.py")
                        break
            if script:
                env = os.environ.copy()
                # Let the bundled ass-qafix import jieba3/rapidfuzz/rich from the
                # same interpreter/environment that runs the VideOCR CLI, and
                # force UTF-8 so any CJK text in its output survives the pipe.
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUNBUFFERED"] = "1"
                result = _sp.run(
                    [_sys.executable, script, "--inplace", file_path],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                )
                if result.returncode == 0:
                    print("ASS post-processing (ass-qafix) completed successfully.", flush=True)
                else:
                    print(
                        f"Warning: ass-qafix exited with code {result.returncode}. "
                        f"Output: {(result.stdout or '').strip()[:500]}",
                        flush=True,
                    )
            else:
                print("Warning: ass-qafix script not found; skipping ASS post-processing.", flush=True)
        except Exception as e:
            qafix_errors.append(str(e))
            print(f"Warning: ASS post-processing (ass-qafix) failed: {e}", flush=True)
        if qafix_errors:
            print(f"ASS post-processing errors: {'; '.join(qafix_errors)}", flush=True)

    total_end = time.perf_counter()

    merge_write_sec = total_end - merge_start
    ocr_runtime_sec = ocr_end - ocr_start
    total_runtime_sec = total_end - total_start
    print(f"[Perf] Step 3 subtitle merge/write: {merge_write_sec:.2f}s", flush=True)
    print(f"[Perf] End-to-end runtime: {total_runtime_sec:.2f}s", flush=True)
    if getattr(v, 'duration_ms', 0) and total_runtime_sec > 0:
        video_sec = float(v.duration_ms) / 1000.0
        speed_x = video_sec / total_runtime_sec
        print(
            f"[Bench] Video duration: {video_sec:.2f}s; OCR runtime: {ocr_runtime_sec:.2f}s; "
            f"total runtime: {total_runtime_sec:.2f}s; speed: {speed_x:.2f}x real-time; "
            f"engine: {ocr_engine}; frame scan: {directml_frame_scan_mode}; recognition: {directml_recognition_mode}; onnx tuning: {onnx_directml_tuning}",
            flush=True,
        )
