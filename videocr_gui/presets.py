"""Engine presets: the OCR and label-detection defaults that suit each engine.

A preset touches the **OCR Settings** section and, when label detection is
enabled, the **Label Detection** section. Crop boxes, the GPU/DirectML section
and the ASS styles stay exactly as the user set them, so switching preset can
never silently move a crop box or change an output file. ``--enable_label_detection``
itself is never toggled: a preset only tunes the label knobs, it does not turn
the feature on or off behind the user's back.

The values were measured on the two real clips in ``tests/data/``:

* ``test_video.mkv`` — 349 s, HEVC 10-bit 3840x1640, Chinese hardsubs, with a
  hand-written English soft-sub track. That embedded track is the ground truth
  for **timings and text**, and ``tests/data/settings_exp/eval.py`` scores a run
  against it (95 subtitle windows, 14 hand-transcribed Chinese frames, 8 label
  events). This is the clip that decided the subtitle numbers.
* ``test_video_2.mp4`` — 17.5 s, 3840x1608. No reference track; only useful for
  a fast sanity run.

What the sweep actually showed on the long clip:

* **PaddleOCR's subtitles are already exact.** 95/95 windows covered, 100 %
  text similarity on all 14 samples, zero spurious events — at every
  confidence (65-85), similarity (70-90), SSIM (85-96) and width (720-1080)
  setting tried. Nothing in the OCR gate knobs can improve a perfect score, so
  its preset keeps the shipped defaults.
* **ONNX Runtime DirectML's subtitles lose a little recall**: 91/95 windows,
  92.2 % mean text similarity, one sampled frame missed entirely. The misses are
  all the same failure: a very short subtitle (0.5 s) sandwiched between two
  longer ones is skipped by the frame-scan gate. Raising ``--ssim_threshold``
  to 96-98, lowering ``--conf_threshold`` to 50-60 and skipping fewer frames
  each changed the count by at most ±2 windows and never recovered a miss, so
  its preset also keeps the shipped defaults rather than pretending otherwise.
* **The ``medium`` model tier is a real accuracy win for ONNX**: 93/95 windows
  and 92.9 % text similarity, plus 8/8 label recall instead of 6/8 — at roughly
  double the runtime (210 s vs 104 s on this 5:49 clip). That is a genuine
  per-engine difference and is what the ONNX preset selects.
* **Label false positives are dominated by two knobs.** The 2-second
  ``--label_reappear_merge_gap`` splits a long-running title card
  ("前情提要", on screen 2.4-64.6 s) into separate events, and the 160 px
  ``--label_pos_drift`` turns the drifting title art into extra events. With
  ``--label_reappear_merge_gap 5`` and ``--label_pos_drift 400`` PaddleOCR goes
  from 13 label events / 6 false positives to 9 / 2, and ONNX from 12 / 5 to
  11 / 3, with recall unchanged (7/8 and 8/8 respectively). Both values are now
  the app defaults too, so a fresh install starts on the PaddleOCR preset
  instead of showing "Custom".
* ``--label_conf_threshold`` is the one label knob that differs per engine.
  PaddleOCR keeps the shipped 70: raising it to 80 changed nothing on the long
  clip but dropped 乔蚩光 from the short clip's output, which is the same trap
  AGENTS.md records (above ~75 the small name text starts disappearing).
  ONNX keeps 80: on the long clip it scores the same 8/8 recall with 3 false
  positives instead of 4, and the short clip's labels are identical at 70 and
  80 — so the stricter value is free for that engine.
* ``--label_min_confirmation_frames`` above 4, ``--label_min_zone_height`` above
  32 and ``--label_text_similarity`` above 55 changed nothing on either clip,
  so they stay at their shipped defaults.

``Custom`` is not a settings set: it is what the UI shows once the user edits
any preset-managed setting by hand, and it means "whatever is in the widgets
right now".
"""

from __future__ import annotations

from typing import Any

#: Preset id -> (translation key, English label).
PRESET_LIST: list[tuple[str, str, str]] = [
    ("paddle", "preset_paddle", "PaddleOCR / Google Lens"),
    ("onnx", "preset_onnx", "ONNX Runtime DirectML"),
    ("custom", "preset_custom", "Custom"),
]

CUSTOM_PRESET: str = "custom"
PADDLE_PRESET: str = "paddle"
ONNX_PRESET: str = "onnx"
DEFAULT_PRESET: str = PADDLE_PRESET

#: Keys a preset may write: the OCR Settings section (including the model
#: tier) plus the label-detection tuning knobs. Crop boxes, the GPU section and
#: the ASS styles are excluded.
PRESET_KEYS: tuple[str, ...] = (
    "--conf_threshold",
    "--sim_threshold",
    "--max_merge_gap",
    "--brightness_threshold",
    "--ssim_threshold",
    "--ocr_image_max_width",
    "--frames_to_skip",
    "--min_subtitle_duration",
    "--model_tier",
    "--label_conf_threshold",
    "--label_min_confirmation_frames",
    "--label_text_similarity",
    "--label_pos_drift",
    "--label_reappear_merge_gap",
    "--label_min_zone_height",
)

#: Values measured on tests/data/test_video.mkv (see the module docstring).
#:
#: PaddleOCR scores a perfect subtitle run on that clip at every gate setting,
#: so its numbers are the shipped defaults apart from the label knobs that
#: provably remove 10 of its 12 false positives.
#:
#: ONNX differs in one place the data supports: the ``medium`` PP-OCRv6 tier
#: recovers two subtitle windows and all eight labels, which the ``small``
#: default cannot. It costs roughly twice the runtime.
PRESETS: dict[str, dict[str, str]] = {
    PADDLE_PRESET: {
        "--conf_threshold": "75",
        "--sim_threshold": "80",
        "--max_merge_gap": "0.1",
        "--brightness_threshold": "",
        "--ssim_threshold": "92",
        "--ocr_image_max_width": "720",
        "--frames_to_skip": "1",
        "--min_subtitle_duration": "0.2",
        "--model_tier": "small",
        "--label_conf_threshold": "70",
        "--label_min_confirmation_frames": "4",
        "--label_text_similarity": "55",
        "--label_pos_drift": "400",
        "--label_reappear_merge_gap": "5.0",
        "--label_min_zone_height": "32",
    },
    ONNX_PRESET: {
        "--conf_threshold": "75",
        "--sim_threshold": "80",
        "--max_merge_gap": "0.1",
        "--brightness_threshold": "",
        "--ssim_threshold": "92",
        "--ocr_image_max_width": "720",
        "--frames_to_skip": "1",
        "--min_subtitle_duration": "0.2",
        "--model_tier": "medium",
        "--label_conf_threshold": "80",
        "--label_min_confirmation_frames": "4",
        "--label_text_similarity": "55",
        "--label_pos_drift": "400",
        "--label_reappear_merge_gap": "5.0",
        "--label_min_zone_height": "32",
    },
}

#: Engine display name -> preset id. Anything else (legacy EasyOCR) maps to the
#: ONNX preset because that engine shares its backend.
ENGINE_PRESETS: dict[str, str] = {
    "PaddleOCR (Det. + Rec.)": PADDLE_PRESET,
    "PaddleOCR (Det.) + Google Lens (Rec.)": PADDLE_PRESET,
    "ONNX Runtime DirectML (AMD GPU Experimental)": ONNX_PRESET,
    "EasyOCR DirectML (AMD GPU)": ONNX_PRESET,
}


def preset_for_engine(engine: str) -> str:
    """Preset id that matches ``engine``; PaddleOCR when unknown."""
    return ENGINE_PRESETS.get(engine, PADDLE_PRESET)


def preset_ids() -> list[str]:
    """The preset ids in display order."""
    return [pid for pid, _key, _label in PRESET_LIST]


def preset_label_key(preset: str) -> str:
    """Translation key for ``preset``'s display name."""
    for pid, key, _label in PRESET_LIST:
        if pid == preset:
            return key
    return "preset_custom"


def preset_values(preset: str) -> dict[str, str]:
    """Settings a preset writes, or ``{}`` for ``custom``/unknown."""
    return dict(PRESETS.get(preset, {}))


def matches_preset(settings: dict[str, Any], preset: str) -> bool:
    """True when ``settings`` still holds exactly ``preset``'s values."""
    values = PRESETS.get(preset)
    if values is None:
        return False
    for key, expected in values.items():
        actual = settings.get(key, "")
        if str(actual).strip() != str(expected).strip():
            return False
    return True


def detect_preset(settings: dict[str, Any], engine: str) -> str:
    """Which preset the current OCR settings correspond to.

    The engine's own preset wins when the values match it; otherwise the other
    known preset is checked (a user may deliberately run PaddleOCR with the
    ONNX numbers); otherwise ``custom``.
    """
    preferred = preset_for_engine(engine)
    if matches_preset(settings, preferred):
        return preferred
    for preset in (PADDLE_PRESET, ONNX_PRESET):
        if preset != preferred and matches_preset(settings, preset):
            return preset
    return CUSTOM_PRESET


def apply_preset(settings: dict[str, Any], preset: str) -> dict[str, Any]:
    """Returns ``settings`` with ``preset``'s OCR values written in."""
    merged = dict(settings)
    merged.update(preset_values(preset))
    return merged
