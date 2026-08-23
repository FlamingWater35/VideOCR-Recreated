from __future__ import annotations

import os
import time
import traceback
from typing import Any

from . import utils


def _format_exception(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


# ---------------------------------------------------------------------------
# Language forwarding
# ---------------------------------------------------------------------------
# The unified `rapidocr` package runs PP-OCRv6, a single model family that
# covers 50 languages (Chinese, English, Japanese and 46 Latin-script
# languages). The requested subtitle language is forwarded to the engine so
# the ONNX DirectML backend behaves like the PaddleOCR / EasyOCR backends.
#
# Project language codes are normalized to RapidOCR-compatible codes below.
# Unknown two-letter codes are passed through; anything else falls back to "en".
RAPIDOCR_LANG_MAP: dict[str, str] = {
    # PaddleOCR-style codes used across VideOCR
    "en": "en",
    "ch": "ch",
    "chinese_cht": "chinese_cht",
    "japan": "japan",
    "korean": "korean",
    "th": "th",
    "el": "el",
    "te": "te",
    "ta": "ta",
    "ka": "ka",
    # Latin-script and other common codes
    "fr": "fr",
    "french": "fr",
    "de": "de",
    "german": "de",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ar": "ar",
    "id": "id",
    "vi": "vi",
    "nl": "nl",
    "pl": "pl",
    "tr": "tr",
    "sv": "sv",
    "da": "da",
    "no": "no",
    "fi": "fi",
    "cs": "cs",
    "sk": "sk",
    "hu": "hu",
    "ro": "ro",
    "hr": "hr",
    "sl": "sl",
    "et": "et",
    "lv": "lv",
    "lt": "lt",
    "bg": "bg",
    "uk": "uk",
    "be": "be",
    "hi": "hi",
    "mr": "mr",
    "ne": "ne",
    "fa": "fa",
    "ur": "ur",
    "ms": "ms",
    "tl": "tl",
    "af": "af",
    "az": "az",
    "bs": "bs",
    "ca": "ca",
    "cy": "cy",
    "eu": "eu",
    "ga": "ga",
    "gl": "gl",
    "is": "is",
    "ku": "ku",
    "la": "la",
    "lb": "lb",
    "mi": "mi",
    "mt": "mt",
    "oc": "oc",
    "qu": "qu",
    "rm": "rm",
    "rs_latin": "sr",
    "sq": "sq",
    "uz": "uz",
}


def normalize_rapidocr_lang(lang: str) -> str:
    """Return a RapidOCR-compatible language code for a VideOCR language code."""
    key = (lang or "en").strip()
    if key in RAPIDOCR_LANG_MAP:
        return RAPIDOCR_LANG_MAP[key]
    if len(key) == 2:
        return key
    return "en"


def _normalize_box(box: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for point in box:
        if len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    if len(points) != 4:
        raise ValueError(f"Unexpected ONNX OCR box format: {box!r}")
    return points


def _make_session_options(tuning: str) -> tuple[Any | None, str]:
    """Build conservative ONNX Runtime session options when possible.

    DirectML can reserve a lot of VRAM with very large OCR grids. These options
    do not force true batching, but they reduce extra memory patterns/arenas and
    make the selected tuning visible in the logs. RapidOCR versions vary, so the
    loader tries these options first and then gracefully falls back.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return None, "session options unavailable"

    tuning = (tuning or "balanced").strip().lower()
    so = ort.SessionOptions()
    try:
        if tuning in ("low_vram", "balanced"):
            so.enable_mem_pattern = False
            so.enable_cpu_mem_arena = False
            if tuning == "low_vram":
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
            elif tuning == "balanced":
                so.intra_op_num_threads = 2
        elif tuning == "max":
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    except Exception:
        pass
    return so, f"session_options={tuning}"


# PP-OCRv6 medium model identifiers (PaddleOCR naming convention). The unified
# `rapidocr` package bundles the PP-OCRv6 *small* tier by default; we request
# the medium tier for higher accuracy and fall back to the bundled defaults
# when the medium models cannot be resolved.
PP_OCRV6_MEDIUM_MODELS: dict[str, str] = {
    "Det.model_path": "PP-OCRv6_medium_det",
    "Rec.model_path": "PP-OCRv6_medium_rec",
}


def _load_rapidocr_engine(lang: str = "en") -> tuple[Any | None, str]:
    """Create a RapidOCR engine on ONNX Runtime DirectML with PP-OCRv6 models.

    The unified ``rapidocr`` package is used. This loader:
      * verifies ONNX Runtime exposes ``DmlExecutionProvider``,
      * attempts to select the PP-OCRv6 *medium* det/rec models for the best
        accuracy, falling back to the bundled defaults (PP-OCRv6 small tier)
        when the medium models cannot be resolved, and
      * forwards the requested language so recognition matches the other
        engines.
    """
    tuning = (
        os.environ.get("VIDEOCR_ONNX_DIRECTML_TUNING", "balanced").strip().lower()
        or "balanced"
    )

    # 1. Verify ONNX Runtime is present and exposes the DirectML provider.
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return None, f"onnxruntime-directml is not importable: {e}"

    providers = []
    try:
        providers = list(ort.get_available_providers())
    except Exception:
        providers = []
    if "DmlExecutionProvider" not in providers:
        return (
            None,
            f"ONNX Runtime is installed, but DmlExecutionProvider is not available. Providers: {providers}",
        )

    requested_index = os.environ.get("VIDEOCR_DIRECTML_DEVICE_INDEX", "").strip()
    provider_options: list[dict[str, Any]] = []
    if requested_index:
        os.environ["ORT_DML_DEVICE_ID"] = requested_index
        provider_options = (
            [{"device_id": int(requested_index)}, {}]
            if requested_index.isdigit()
            else []
        )

    if tuning in ("low_vram", "balanced"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    # 2. Import the unified rapidocr package.
    try:
        from rapidocr import RapidOCR  # type: ignore
    except Exception as e:
        return None, f"rapidocr is not importable: {e}"

    session_options, session_note = _make_session_options(tuning)
    provider_list = ["DmlExecutionProvider", "CPUExecutionProvider"]
    rapidocr_lang = normalize_rapidocr_lang(lang)

    # 3. ONNX Runtime session/provider kwargs. The exact keyword names accepted
    #    by the unified rapidocr constructor vary between releases, so several
    #    shapes are attempted.
    ort_variants: list[dict[str, Any]] = []
    if session_options is not None and provider_options:
        ort_variants.append(
            {
                "providers": provider_list,
                "provider_options": provider_options,
                "sess_options": session_options,
            }
        )
        ort_variants.append(
            {
                "providers": provider_list,
                "provider_options": provider_options,
                "session_options": session_options,
            }
        )
    if session_options is not None:
        ort_variants.append(
            {"providers": provider_list, "sess_options": session_options}
        )
        ort_variants.append(
            {"providers": provider_list, "session_options": session_options}
        )
    if provider_options:
        ort_variants.append(
            {"providers": provider_list, "provider_options": provider_options}
        )
    ort_variants.append({"providers": provider_list})
    ort_variants.append({})

    # 4. Model/language configuration variants, most preferred first. The
    #    `params` dict uses dot-notation config keys understood by the unified
    #    rapidocr package. If a variant is rejected (unknown keys, missing
    #    models, etc.) the loader falls through to the next one, ultimately
    #    landing on the bundled PP-OCRv6 small defaults.
    model_variants: list[tuple[dict[str, Any], str]] = [
        (
            {"params": {**PP_OCRV6_MEDIUM_MODELS, "lang": rapidocr_lang}},
            f"PP-OCRv6 medium + lang={rapidocr_lang}",
        ),
        ({"params": dict(PP_OCRV6_MEDIUM_MODELS)}, "PP-OCRv6 medium"),
        (
            {"params": {"lang": rapidocr_lang}},
            f"bundled defaults + lang={rapidocr_lang}",
        ),
        ({}, "bundled defaults (PP-OCRv6 small)"),
    ]

    last_error: BaseException | None = None
    last_desc = "no attempt made"
    for model_kwargs, model_desc in model_variants:
        for ort_kwargs in ort_variants:
            kwargs = {**model_kwargs, **ort_kwargs}
            try:
                engine = RapidOCR(**kwargs)
                return engine, (
                    f"ONNX Runtime providers: {providers}; provider order: {provider_list}; "
                    f"tuning: {tuning}; {session_note}; models: {model_desc}"
                )
            except TypeError as e:
                last_error = e
                last_desc = model_desc
                continue
            except Exception as e:
                last_error = e
                last_desc = model_desc
                continue

    reason = (
        _format_exception(last_error)
        if last_error
        else "unknown RapidOCR initialization error"
    )
    return None, f"RapidOCR (unified) could not initialize ({last_desc}): {reason}"


def _run_rapidocr_on_grid(
    engine: Any, image_path: str
) -> list[tuple[list[list[float]], str, float]]:
    raw = engine(image_path)

    # Unwrap the legacy (result, elapse) tuple shape.
    if isinstance(raw, tuple) and raw:
        result = raw[0]
    else:
        result = raw

    if result is None:
        return []

    # Newer unified `rapidocr` releases may return an OcrResult-like object
    # exposing parallel boxes/txts/scores collections instead of a flat
    # [[box, text, score], ...] list. Normalize both shapes.
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and txts is not None and scores is not None:
        items = list(zip(boxes, txts, scores))
    else:
        try:
            items = list(result)
        except TypeError:
            return []

    normalized: list[tuple[list[list[float]], str, float]] = []
    for item in items:
        if not item or len(item) < 3:
            continue
        try:
            box = _normalize_box(item[0])
            text = str(item[1]).strip()
            conf = float(item[2])
        except Exception:
            continue
        if text:
            normalized.append((box, text, conf))
    return normalized


def run_onnx_directml_on_stitched_images(
    input_dir: str,
    stitch_map: dict[str, list[dict[str, Any]]],
    lang: str,
    use_gpu: bool,
    directml_recognition_mode: str = "stable",
) -> dict[tuple[int, int], list[Any]]:
    """Experimental ONNXRuntime DirectML OCR pass.

    When RapidOCR + ONNXRuntime DirectML are available, this reads the already
    stitched VideOCR Recreated grids with ONNXRuntime. The selected subtitle
    language is forwarded to RapidOCR so it uses the matching PP-OCRv6
    recognition configuration, mirroring the PaddleOCR / EasyOCR backends. If
    the optional ONNX stack is not available or cannot initialize on DirectML,
    it falls back to the proven EasyOCR DirectML Hybrid backend so the run
    still finishes.
    """
    filenames = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    )
    outputs: dict[tuple[int, int], list[Any]] = {}
    total = len(filenames)
    if total == 0:
        return outputs

    if not use_gpu:
        print(
            "ONNX Runtime DirectML OCR was selected, but GPU usage is disabled. Falling back to EasyOCR CPU/Hybrid path.",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images

        return run_easyocr_on_stitched_images(
            input_dir, stitch_map, lang, use_gpu, directml_recognition_mode
        )

    print("Starting ONNX Runtime DirectML OCR (experimental)...", flush=True)
    engine, reason = _load_rapidocr_engine(lang)
    print(f"ONNX DirectML status: {reason}", flush=True)

    if engine is None:
        print(
            "ONNX DirectML OCR is not ready on this install. Falling back to EasyOCR DirectML Hybrid for this run.",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images

        return run_easyocr_on_stitched_images(
            input_dir, stitch_map, lang, use_gpu, directml_recognition_mode
        )

    try:
        image_times: list[float] = []
        recognized_lines = 0
        active_images = 0
        for index, filename in enumerate(filenames, 1):
            image_path = os.path.join(input_dir, filename)
            mapping = stitch_map.get(filename)
            if not mapping:
                continue

            img_start = time.perf_counter()
            raw_lines = _run_rapidocr_on_grid(engine, image_path)
            img_elapsed = time.perf_counter() - img_start
            image_times.append(img_elapsed)
            active_images += 1
            recognized_lines += len(raw_lines)

            for box, text, confidence in raw_lines:
                for adjusted_poly, meta in utils.unstitch_polygon(box, mapping):
                    key = (int(meta["frame_idx"]), int(meta["zone_idx"]))
                    outputs.setdefault(key, []).append(
                        [adjusted_poly, (text, confidence)]
                    )

            print(
                f"\rStep 2/3: Performing ONNX DirectML OCR on image {index} of {total}",
                end="",
                flush=True,
            )
        print()

        if image_times:
            avg_t = sum(image_times) / len(image_times)
            print(
                f"[Perf] ONNX image timing: images={active_images}; avg={avg_t:.2f}s; "
                f"fastest={min(image_times):.2f}s; slowest={max(image_times):.2f}s; recognized lines={recognized_lines}",
                flush=True,
            )
        return outputs
    except Exception as e:
        print(
            "\nONNX DirectML OCR failed during processing. Falling back to EasyOCR DirectML Hybrid.\n"
            f"ONNX error:\n{_format_exception(e)}",
            flush=True,
        )
        from .easyocr_directml import run_easyocr_on_stitched_images

        return run_easyocr_on_stitched_images(
            input_dir, stitch_map, lang, use_gpu, directml_recognition_mode
        )
