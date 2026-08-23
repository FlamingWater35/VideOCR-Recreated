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
# PP-OCRv6 is a *unified* model family covering ~50 languages in a single
# network, so there are no per-language rec models to select (unlike PP-OCRv5).
# The requested language is normalized and logged for verification, but it is
# not passed as a model-selection parameter — RapidOCR rejects a `lang` key in
# its params for the unified PP-OCRv6 models.
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
    # Common Latin-script / other codes
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
    # Many language IDs are already two-letter ISO codes.
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


# PP-OCRv6 medium model identifiers (PaddleOCR naming convention).
# Detection: PP-OCRv6_medium_det (~59 MB), Recognition: PP-OCRv6_medium_rec (~73 MB).
# The unified `rapidocr` package bundles the PP-OCRv6 *small* tier by default;
# requesting the medium tier via `params` gives higher accuracy.
PP_OCRV6_MEDIUM_DET = "PP-OCRv6_medium_det"
PP_OCRV6_MEDIUM_REC = "PP-OCRv6_medium_rec"


def _load_rapidocr_engine(lang: str = "en") -> tuple[Any | None, str]:
    """Create a RapidOCR engine on ONNX Runtime DirectML with PP-OCRv6 medium models.

    The unified ``rapidocr`` package is used. Its constructor accepts EITHER a
    ``params`` dict (model/config overrides) OR top-level provider/session
    kwargs — not both at once. Since PP-OCRv6 medium model selection requires
    ``params``, the engine relies on ONNX Runtime's default provider ordering
    (which includes ``DmlExecutionProvider`` because ``onnxruntime-directml``
    is installed) rather than explicit provider kwargs.
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

    # 2. Device selection + CPU thread tuning via environment variables.
    #    These apply regardless of how the ONNX Runtime session is created.
    requested_index = os.environ.get("VIDEOCR_DIRECTML_DEVICE_INDEX", "").strip()
    if requested_index:
        # ORT DirectML honors this environment variable on many builds.
        os.environ["ORT_DML_DEVICE_ID"] = requested_index
    # Keep OpenMP from spinning hard on the CPU while DirectML is the target.
    if tuning in ("low_vram", "balanced"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    # 3. Import the unified rapidocr package.
    try:
        from rapidocr import RapidOCR  # type: ignore
    except Exception as e:
        return None, f"rapidocr is not importable: {e}"

    # 4. Language forwarding: normalize + log. PP-OCRv6 is a unified model, so
    #    the language is verified here but not passed to the engine.
    rapidocr_lang = normalize_rapidocr_lang(lang)
    print(
        f"[ONNX] Language forwarding: input='{lang}' -> rapidocr='{rapidocr_lang}' "
        f"(mapped={lang in RAPIDOCR_LANG_MAP})",
        flush=True,
    )

    provider_list = ["DmlExecutionProvider", "CPUExecutionProvider"]

    # 5. Create the engine with PP-OCRv6 medium models. This is the known-good
    #    configuration: `params` with `model_name` keys and no provider kwargs.
    try:
        engine = RapidOCR(
            params={
                "Det.model_name": PP_OCRV6_MEDIUM_DET,
                "Rec.model_name": PP_OCRV6_MEDIUM_REC,
            }
        )
        return engine, (
            f"ONNX Runtime providers: {providers}; provider order: {provider_list}; "
            f"tuning: {tuning}; models: PP-OCRv6 medium (model_name); lang: {rapidocr_lang}"
        )
    except Exception as medium_err:
        # Fall back to the bundled defaults (PP-OCRv6 small) if the medium
        # model config is rejected for any reason.
        print(
            f"[ONNX] PP-OCRv6 medium config rejected ({type(medium_err).__name__}); "
            f"falling back to bundled defaults.",
            flush=True,
        )
        try:
            engine = RapidOCR()
            return engine, (
                f"ONNX Runtime providers: {providers}; provider order: {provider_list}; "
                f"tuning: {tuning}; models: bundled defaults (PP-OCRv6 small); lang: {rapidocr_lang}"
            )
        except Exception as default_err:
            reason = _format_exception(default_err)
            return None, f"RapidOCR (unified) could not initialize: {reason}"


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
    stitched VideOCR Recreated grids with ONNXRuntime using PP-OCRv6 medium
    models. The selected subtitle language is normalized and logged. If the
    optional ONNX stack is not available or cannot initialize on DirectML, it
    falls back to the proven EasyOCR DirectML Hybrid backend so the run still
    finishes.
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
