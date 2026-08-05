"""Crop box model and coordinate math (ported from the legacy GUI)."""

from __future__ import annotations

import math
from typing import Any


def compute_crop_coords(
    rect_x1: float, rect_y1: float, rect_x2: float, rect_y2: float,
    original_w: int, original_h: int, resized_w: int, resized_h: int,
) -> dict[str, int]:
    """Converts image-space rectangle (resized) into absolute video pixel coords."""
    crop_x = math.floor(rect_x1 * original_w / resized_w)
    crop_y = math.floor(rect_y1 * original_h / resized_h)
    crop_w = math.ceil((rect_x2 - rect_x1) * original_w / resized_w)
    crop_h = math.ceil((rect_y2 - rect_y1) * original_h / resized_h)
    return {"crop_x": crop_x, "crop_y": crop_y, "crop_width": crop_w, "crop_height": crop_h}


def restore_box(
    cx: float, cy: float, cw: float, ch: float,
    orig_w: int, orig_h: int, res_w: int, res_h: int,
) -> dict[str, Any]:
    """Builds a crop box dict from absolute video coords for display at resized size."""
    sx = res_w / orig_w if orig_w > 0 else 0
    sy = res_h / orig_h if orig_h > 0 else 0
    rx1 = cx * sx
    ry1 = cy * sy
    rx2 = (cx + cw) * sx
    ry2 = (cy + ch) * sy
    return {
        "coords": {
            "crop_x": int(cx), "crop_y": int(cy),
            "crop_width": int(cw), "crop_height": int(ch),
        },
        "img_points": ((rx1, ry1), (rx2, ry2)),
    }


def absolute_from_relative(rel: dict[str, Any], orig_w: int, orig_h: int) -> dict[str, int]:
    """Converts relative (0..1) crop coords to absolute video pixels."""
    return {
        "crop_x": math.floor(rel.get("crop_x", 0) * orig_w),
        "crop_y": math.floor(rel.get("crop_y", 0) * orig_h),
        "crop_width": math.ceil(rel.get("crop_width", 0) * orig_w),
        "crop_height": math.ceil(rel.get("crop_height", 0) * orig_h),
    }


def relative_from_absolute(abs_coords: dict[str, int], orig_w: int, orig_h: int) -> dict[str, float]:
    """Converts absolute video pixel coords to relative (0..1) coords for storage."""
    return {
        "crop_x": abs_coords["crop_x"] / orig_w,
        "crop_y": abs_coords["crop_y"] / orig_h,
        "crop_width": abs_coords["crop_width"] / orig_w,
        "crop_height": abs_coords["crop_height"] / orig_h,
    }
