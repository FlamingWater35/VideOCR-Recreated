"""Tests for videocr_gui.crop — crop box coordinate math (all pure functions)."""

from __future__ import annotations

import pytest

from videocr_gui import crop


class TestComputeCropCoords:
    def test_exact_scale(self):
        # 640x360 image displayed at 320x180: rect maps 1:2.
        result = crop.compute_crop_coords(10, 20, 110, 70, 640, 360, 320, 180)
        assert result == {
            "crop_x": 20,
            "crop_y": 40,
            "crop_width": 200,
            "crop_height": 100,
        }

    def test_floor_and_ceil_rounding(self):
        # Non-integer scaling: x/y floored, w/h ceiled (never under-covers).
        result = crop.compute_crop_coords(1, 1, 4, 4, 1920, 1080, 640, 360)
        assert result["crop_x"] == 3   # floor(1 * 1920/640) = 3
        assert result["crop_y"] == 3   # floor(1 * 1080/360) = 3
        assert result["crop_width"] == 9   # ceil(3 * 3) = 9
        assert result["crop_height"] == 9

    def test_full_frame(self):
        result = crop.compute_crop_coords(0, 0, 320, 180, 640, 360, 320, 180)
        assert result == {"crop_x": 0, "crop_y": 0, "crop_width": 640, "crop_height": 360}

    def test_returns_ints(self):
        result = crop.compute_crop_coords(0.5, 0.5, 2.5, 2.5, 100, 100, 10, 10)
        assert all(isinstance(v, int) for v in result.values())


class TestRestoreBox:
    def test_basic(self):
        result = crop.restore_box(100, 50, 200, 60, 640, 360, 320, 180)
        assert result["coords"] == {
            "crop_x": 100, "crop_y": 50, "crop_width": 200, "crop_height": 60,
        }
        (x1, y1), (x2, y2) = result["img_points"]
        assert (x1, y1) == (50.0, 25.0)
        assert (x2, y2) == (150.0, 55.0)

    def test_zero_original_size_does_not_divide_by_zero(self):
        result = crop.restore_box(0, 0, 10, 10, 0, 0, 100, 100)
        assert result["img_points"] == ((0, 0), (0, 0))

    def test_coords_coerced_to_int(self):
        result = crop.restore_box(10.9, 5.1, 20.7, 6.2, 100, 100, 100, 100)
        assert result["coords"] == {
            "crop_x": 10, "crop_y": 5, "crop_width": 20, "crop_height": 6,
        }


class TestRelativeAbsoluteRoundTrip:
    def test_relative_to_absolute(self):
        rel = {"crop_x": 0.25, "crop_y": 0.5, "crop_width": 0.5, "crop_height": 0.25}
        abs_coords = crop.absolute_from_relative(rel, 1920, 1080)
        assert abs_coords == {
            "crop_x": 480, "crop_y": 540, "crop_width": 960, "crop_height": 270,
        }

    def test_absolute_to_relative(self):
        abs_coords = {"crop_x": 480, "crop_y": 540, "crop_width": 960, "crop_height": 270}
        rel = crop.relative_from_absolute(abs_coords, 1920, 1080)
        assert rel == {"crop_x": 0.25, "crop_y": 0.5, "crop_width": 0.5, "crop_height": 0.25}

    @pytest.mark.parametrize("w,h", [(1920, 1080), (1280, 720), (3840, 2160)])
    def test_round_trip_within_one_pixel(self, w, h):
        abs_coords = {"crop_x": 17, "crop_y": 29, "crop_width": 333, "crop_height": 111}
        rel = crop.relative_from_absolute(abs_coords, w, h)
        back = crop.absolute_from_relative(rel, w, h)
        for key in abs_coords:
            assert abs(back[key] - abs_coords[key]) <= 1, key

    def test_missing_relative_keys_default_to_zero(self):
        assert crop.absolute_from_relative({}, 100, 100) == {
            "crop_x": 0, "crop_y": 0, "crop_width": 0, "crop_height": 0,
        }

    def test_fractional_floor_ceil(self):
        rel = {"crop_x": 0.333, "crop_y": 0.666, "crop_width": 0.111, "crop_height": 0.111}
        result = crop.absolute_from_relative(rel, 999, 999)
        assert result["crop_x"] == 332    # floor(332.667)
        assert result["crop_y"] == 665    # floor(665.334)
        assert result["crop_width"] == 111  # ceil(110.889)

    def test_relative_of_zero_size(self):
        rel = crop.relative_from_absolute(
            {"crop_x": 0, "crop_y": 0, "crop_width": 0, "crop_height": 0}, 100, 100
        )
        assert rel == {"crop_x": 0.0, "crop_y": 0.0, "crop_width": 0.0, "crop_height": 0.0}
