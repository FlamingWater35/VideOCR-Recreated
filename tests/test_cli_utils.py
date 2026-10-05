"""Tests for videocr.utils — timestamps, text/geometry helpers, model dirs."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from videocr import utils
from videocr.lang_dictionaries import GOOGLE_LENS_LANGS, PADDLEOCR_LANGS


class TestGetMsFromTimeStr:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0:00", 0.0),
            ("1:30", 90_000.0),
            ("10:59", 659_000.0),
            ("1:02:03", 3_723_000.0),
            ("0:00:01.5", 1_500.0),
            ("1:61", 121_000.0),  # minutes/seconds are not range-checked
        ],
    )
    def test_valid(self, value, expected):
        assert utils.get_ms_from_time_str(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["", "90", "1:2:3:4"])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            utils.get_ms_from_time_str(value)


class TestTimestamps:
    def test_srt_from_frame(self):
        # frame 25 at 25fps = 1.000s
        assert utils.get_srt_timestamp(25, 25.0) == "00:00:01,000"

    def test_srt_from_frame_with_offset(self):
        assert utils.get_srt_timestamp(0, 25.0, offset_ms=1500.0) == "00:00:01,500"

    @pytest.mark.parametrize(
        ("ms", "expected"),
        [
            (0, "00:00:00,000"),
            (1, "00:00:00,001"),
            (1_500, "00:00:01,500"),
            (3_723_456, "01:02:03,456"),
            (3_600_000, "01:00:00,000"),
        ],
    )
    def test_srt_from_ms(self, ms, expected):
        assert utils.get_srt_timestamp_from_ms(ms) == expected

    @pytest.mark.parametrize(
        ("ms", "expected"),
        [
            (0, "0:00:00.00"),
            (1_500, "0:00:01.50"),
            (3_723_456, "1:02:03.45"),
            (3_600_000, "1:00:00.00"),
            (10 * 3_600_000, "10:00:00.00"),
        ],
    )
    def test_ass_from_ms(self, ms, expected):
        assert utils.get_ass_timestamp_from_ms(ms) == expected


class TestComputeLabelZones:
    def test_no_zones_returns_empty(self):
        assert utils.compute_label_zones(100, 100, []) == []

    def test_zero_dimensions_returns_empty(self):
        zone = {"y_start": 10, "y_end": 20}
        assert utils.compute_label_zones(0, 100, [zone]) == []
        assert utils.compute_label_zones(100, 0, [zone]) == []

    def test_free_band_below_subtitle(self):
        # Subtitle band occupies y 80..100 of a 100px-high frame → label zone 0..80.
        zones = [{"x_start": 0, "x_end": 100, "y_start": 80, "y_end": 100}]
        result = utils.compute_label_zones(100, 100, zones)
        assert result == [{"x": 0, "y": 0, "w": 100, "h": 80}]

    def test_single_gap_band(self):
        zones = [
            {"y_start": 0, "y_end": 20},    # top band occupied
            {"y_start": 60, "y_end": 100},  # bottom band occupied
        ]
        # Free gap between them (20..60) is the only band.
        result = utils.compute_label_zones(100, 100, zones)
        assert result == [{"x": 0, "y": 20, "w": 100, "h": 40}]

    def test_all_free_bands_returned_top_to_bottom(self):
        # Subtitles in the middle → free bands above AND below are both kept
        # (the old single-zone API only returned the largest).
        zones = [{"y_start": 30, "y_end": 70}]
        result = utils.compute_label_zones(100, 100, zones, min_height=25)
        assert result == [
            {"x": 0, "y": 0, "w": 100, "h": 30},
            {"x": 0, "y": 70, "w": 100, "h": 30},
        ]

    def test_small_bands_filtered_by_min_height(self):
        # Free bands: 40px top (kept) and 10px bottom (below min_height → dropped).
        zones = [{"y_start": 40, "y_end": 90}]
        result = utils.compute_label_zones(100, 100, zones, min_height=20)
        assert result == [{"x": 0, "y": 0, "w": 100, "h": 40}]

    def test_max_bands_keeps_tallest(self):
        # Three free bands: 30, 30, 20 px; with max_bands=2 the 20px one goes.
        zones = [
            {"y_start": 30, "y_end": 40},
            {"y_start": 70, "y_end": 80},
        ]
        result = utils.compute_label_zones(100, 100, zones, min_height=15, max_bands=2)
        assert result == [
            {"x": 0, "y": 0, "w": 100, "h": 30},
            {"x": 0, "y": 40, "w": 100, "h": 30},
        ]

    def test_overlapping_zones_merge(self):
        zones = [
            {"y_start": 30, "y_end": 60},
            {"y_start": 50, "y_end": 80},
        ]
        # Merged 30..80 → free bands [0..30] and [80..100].
        result = utils.compute_label_zones(100, 100, zones, min_height=15)
        assert result == [
            {"x": 0, "y": 0, "w": 100, "h": 30},
            {"x": 0, "y": 80, "w": 100, "h": 20},
        ]

    def test_zones_cover_full_height(self):
        zones = [{"y_start": 0, "y_end": 100}]
        assert utils.compute_label_zones(100, 100, zones) == []


class TestAssAnchorPoint:
    X1, Y1, X2, Y2 = 10.0, 20.0, 110.0, 60.0  # center = (60, 40)

    @pytest.mark.parametrize(
        ("align", "expected"),
        [
            (1, (10.0, 60.0)),   # bottom-left
            (2, (60.0, 60.0)),   # bottom-center
            (3, (110.0, 60.0)),  # bottom-right
            (4, (10.0, 40.0)),   # middle-left
            (5, (60.0, 40.0)),   # center
            (6, (110.0, 40.0)),  # middle-right
            (7, (10.0, 20.0)),   # top-left (default label alignment)
            (8, (60.0, 20.0)),   # top-center
            (9, (110.0, 20.0)),  # top-right
        ],
    )
    def test_alignment_points(self, align, expected):
        assert utils.ass_anchor_point(
            align, self.X1, self.Y1, self.X2, self.Y2
        ) == expected

    def test_out_of_range_falls_back_to_top_left(self):
        assert utils.ass_anchor_point(0, self.X1, self.Y1, self.X2, self.Y2) == (
            10.0,
            20.0,
        )
        assert utils.ass_anchor_point(10, self.X1, self.Y1, self.X2, self.Y2) == (
            10.0,
            20.0,
        )


class TestDhash:
    @staticmethod
    def _band(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, (48, 192, 3), dtype=np.uint8)

    def test_deterministic(self):
        img = self._band(42)
        assert utils.dhash64(img) == utils.dhash64(img)

    def test_identical_images_distance_zero(self):
        img = np.full((32, 96, 3), 40, dtype=np.uint8)
        assert utils.hamming_distance(utils.dhash64(img), utils.dhash64(img)) == 0

    def test_brightness_shift_stays_close(self):
        # dHash is gradient-based: a global fade must stay within threshold.
        # Use smooth content (a horizontal ramp) — the gradient signs survive
        # the shift, so the hash barely moves.
        ramp = np.arange(192, dtype=np.uint8)[None, :].repeat(48, axis=0)
        base = np.stack([ramp] * 3, axis=-1)
        brighter = (base.astype(np.int16) + 50).astype(np.uint8)  # max 241
        d = utils.hamming_distance(utils.dhash64(base), utils.dhash64(brighter))
        assert d <= 4

    def test_text_appearance_exceeds_threshold(self):
        # Safety: a label appearing in an otherwise static band must move the
        # hash past the default threshold, or real labels would be skipped.
        blank = np.full((32, 384, 3), 30, dtype=np.uint8)
        with_text = blank.copy()
        with_text[8:24, 40:200] = 255  # bright "label" rectangle
        d = utils.hamming_distance(utils.dhash64(blank), utils.dhash64(with_text))
        assert d > 4

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(0, 0, 0), (0b1010, 0b0110, 2), (0, 2**64 - 1, 64), (0b1, 0b0, 1)],
    )
    def test_hamming_distance(self, a, b, expected):
        assert utils.hamming_distance(a, b) == expected


class TestLabelFrameDedup:
    @staticmethod
    def _band(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 255, (48, 192, 3), dtype=np.uint8)

    def test_static_band_skips_and_weights_run(self):
        dedup = utils.LabelFrameDedup(threshold=4, max_run=20)
        img = self._band(1)
        assert dedup.consider(2, 0, img) is True  # first frame always kept
        for i in range(1, 15):
            assert dedup.consider(2, i, img) is False
        dedup.finish()
        assert dedup.weights == {0: 15}  # rep 0 stands for all 15 frames
        assert dedup.skipped == 14

    def test_force_keep_after_max_run(self):
        dedup = utils.LabelFrameDedup(threshold=4, max_run=5)
        img = self._band(2)
        kept = [dedup.consider(2, i, img) for i in range(12)]
        assert kept == [
            True, False, False, False, True,
            False, False, False, True, False, False, False,
        ]
        dedup.finish()
        assert dedup.weights == {0: 5, 4: 5, 8: 4}
        assert dedup.skipped == 9  # 12 frames - 3 kept

    def test_changed_frame_breaks_run(self):
        dedup = utils.LabelFrameDedup(threshold=4, max_run=20)
        a, b = self._band(3), self._band(99)
        assert dedup.consider(2, 0, a) is True
        assert dedup.consider(2, 1, a) is False
        assert dedup.consider(2, 2, b) is True  # dissimilar → keep
        dedup.finish()
        assert dedup.weights == {0: 2, 2: 1}

    def test_bands_are_independent(self):
        dedup = utils.LabelFrameDedup(threshold=4)
        img = self._band(4)
        assert dedup.consider(2, 0, img) is True
        assert dedup.consider(3, 1, img) is True  # other band starts fresh
        dedup.finish()
        assert dedup.weights == {0: 1, 1: 1}


class TestProcessSsimGroupWeights:
    @staticmethod
    def _group():
        rng = np.random.default_rng(11)
        grid = rng.integers(0, 255, (50, 100, 3), dtype=np.uint8)
        loaded = {"g.png": grid}
        m0 = {
            "grid_file": "g.png",
            "x": 0,
            "y": 0,
            "w": 100,
            "h": 50,
            "frame_idx": 10,
        }
        m1 = dict(m0)
        m1["frame_idx"] = 11
        group = [
            (10, [[0.0, 0.0, 100.0, 50.0]], 0.9, m0),
            (11, [[0.0, 0.0, 100.0, 50.0]], 0.8, m1),
        ]
        rects = [[0.0, 0.0, 100.0, 50.0]]
        return rects, group, loaded

    def test_weight_defaults_to_batch_length(self):
        rects, group, loaded = self._group()
        items, deleted = utils.process_ssim_group(rects, group, loaded, 0.85)
        assert deleted == 1
        assert items[0]["frame_idx"] == 10
        assert items[0]["weight"] == 2

    def test_frame_weights_compose_hash_runs(self):
        # Frame 10 represents 6 sampled frames, frame 11 represents 3 →
        # the surviving SSIM representative must stand for all 9.
        rects, group, loaded = self._group()
        items, deleted = utils.process_ssim_group(
            rects, group, loaded, 0.85, frame_weights={10: 6, 11: 3}
        )
        assert deleted == 1
        assert items[0]["frame_idx"] == 10
        assert items[0]["weight"] == 9


class TestResolveLabelOverlaps:
    @staticmethod
    def _item(start, end, x1, y1, x2, y2):
        return {
            "start_ms": start,
            "end_ms": end,
            "fx1": float(x1),
            "fy1": float(y1),
            "fx2": float(x2),
            "fy2": float(y2),
        }

    def test_disjoint_items_untouched(self):
        a = self._item(0, 1000, 0, 0, 100, 40)
        b = self._item(0, 1000, 200, 0, 300, 40)   # horizontal gap
        c = self._item(2000, 3000, 0, 0, 100, 40)  # temporal gap
        d = self._item(0, 1000, 0, 100, 100, 140)  # vertical gap
        utils.resolve_label_overlaps([a, b, c, d], 600)
        assert [x["fy1"] for x in (a, b, c, d)] == [0.0, 0.0, 0.0, 100.0]

    def test_overlap_shifts_down_and_keeps_height(self):
        a = self._item(0, 1000, 0, 0, 100, 40)
        b = self._item(500, 1500, 10, 10, 110, 50)  # overlaps a
        utils.resolve_label_overlaps([a, b], 600)
        assert a["fy1"] == 0.0
        assert b["fy1"] == a["fy2"] + 4.0  # pushed below a + margin
        assert b["fy2"] - b["fy1"] == 40.0  # height preserved

    def test_horizontal_disjoint_simultaneous_untouched(self):
        # Two simultaneous labels side by side must NOT be stacked.
        a = self._item(0, 1000, 0, 0, 100, 40)
        b = self._item(0, 1000, 150, 0, 250, 40)
        utils.resolve_label_overlaps([a, b], 600)
        assert b["fy1"] == 0.0

    def test_clamped_to_frame_bottom(self):
        # b extends past the frame bottom with no collision → clamp in place.
        b = self._item(0, 1000, 0, 575, 100, 620)  # height 45
        utils.resolve_label_overlaps([b], 600)
        assert b["fy2"] == 600.0
        assert b["fy1"] == 555.0

    def test_clamp_accepts_remaining_overlap(self):
        # Shifting would push past the bottom; the overlap is accepted.
        a = self._item(0, 1000, 0, 560, 100, 600)
        b = self._item(0, 1000, 10, 565, 110, 605)
        utils.resolve_label_overlaps([a, b], 600)
        assert b["fy2"] == 600.0
        assert b["fy1"] == 560.0  # height 40 kept; overlaps a by design


class TestRtlAndLanguages:
    @pytest.mark.parametrize("lang", ["ar", "fa", "ur", "iw", "yi", "dv", "syr"])
    def test_rtl(self, lang):
        assert utils.is_language_rtl(lang) is True

    @pytest.mark.parametrize("lang", ["en", "de", "japan", "korean", "ch", "ru"])
    def test_ltr(self, lang):
        assert utils.is_language_rtl(lang) is False

    def test_arabic_group_members_are_rtl(self):
        for lang in PADDLEOCR_LANGS["arabic"]:
            assert utils.is_language_rtl(lang)


class TestExtractNonChineseSegments:
    def test_pure_ascii(self):
        assert utils.extract_non_chinese_segments("hello") == [("non_chinese", "hello")]

    def test_pure_chinese(self):
        assert utils.extract_non_chinese_segments("中文字") == [
            ("chinese", "中"), ("chinese", "文"), ("chinese", "字"),
        ]

    def test_mixed(self):
        result = utils.extract_non_chinese_segments("abc中文def")
        assert result == [
            ("non_chinese", "abc"),
            ("chinese", "中"), ("chinese", "文"),
            ("non_chinese", "def"),
        ]

    def test_empty(self):
        assert utils.extract_non_chinese_segments("") == []


class TestIsOnSameLine:
    def _box(self, x1, y1, x2, y2):
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_same_line(self):
        a = utils.PredictedText(self._box(0, 10, 20, 30), 0.9, "a")
        b = utils.PredictedText(self._box(25, 12, 45, 32), 0.9, "b")
        assert utils.is_on_same_line(a, b) is True

    def test_different_lines(self):
        a = utils.PredictedText(self._box(0, 0, 20, 20), 0.9, "a")
        b = utils.PredictedText(self._box(0, 40, 20, 60), 0.9, "b")
        assert utils.is_on_same_line(a, b) is False


class TestAlignmentMap:
    def test_nine_alignments(self):
        assert len(utils.ALIGNMENT_MAP) == 9
        assert set(utils.VALID_ALIGNMENT_NAMES) == set(utils.ALIGNMENT_MAP)

    @pytest.mark.parametrize(
        ("name", "code"),
        [
            ("bottom-left", "an1"), ("bottom-center", "an2"), ("bottom-right", "an3"),
            ("middle-left", "an4"), ("middle-center", "an5"), ("middle-right", "an6"),
            ("top-left", "an7"), ("top-center", "an8"), ("top-right", "an9"),
        ],
    )
    def test_codes(self, name, code):
        assert utils.ALIGNMENT_MAP[name] == code


class TestResolveModelDirs:
    def test_latin_english(self):
        det, rec, cls = utils.resolve_model_dirs("en", use_server_model=False)
        assert det.endswith(os.path.join("det", "PP-OCRv6_small_det"))
        assert rec.endswith(os.path.join("rec", "PP-OCRv6_small_rec"))
        assert cls.endswith(os.path.join("cls", "PP-LCNet_x1_0_textline_ori"))

    def test_server_model_uses_medium(self):
        det, rec, _ = utils.resolve_model_dirs("en", use_server_model=True)
        assert det.endswith("PP-OCRv6_medium_det")
        assert rec.endswith("PP-OCRv6_medium_rec")

    @pytest.mark.parametrize("lang", ["ar", "fa", "ur"])
    def test_arabic_branch(self, lang):
        _, rec, _ = utils.resolve_model_dirs(lang, use_server_model=False)
        assert rec.endswith("arabic_PP-OCRv5_mobile_rec")

    @pytest.mark.parametrize("lang", ["ru", "uk", "be"])
    def test_eslav_before_cyrillic(self, lang):
        _, rec, _ = utils.resolve_model_dirs(lang, use_server_model=False)
        assert rec.endswith("eslav_PP-OCRv5_mobile_rec")

    @pytest.mark.parametrize(
        ("lang", "rec_suffix"),
        [
            ("ch", "PP-OCRv6_small_rec"),
            ("japan", "PP-OCRv6_small_rec"),
            ("korean", "korean_PP-OCRv5_mobile_rec"),
            ("th", "th_PP-OCRv5_mobile_rec"),
        ],
    )
    def test_specific_branches(self, lang, rec_suffix):
        det, rec, _ = utils.resolve_model_dirs(lang, use_server_model=False)
        assert det.endswith("PP-OCRv6_small_det")
        assert rec.endswith(rec_suffix)

    def test_ka_uses_v3_models(self):
        det, rec, _ = utils.resolve_model_dirs("ka", use_server_model=False)
        assert det.endswith("PP-OCRv3_mobile_det")
        assert rec.endswith("ka_PP-OCRv3_mobile_rec")

    def test_ku_resolves_latin_not_arabic(self):
        _, rec, _ = utils.resolve_model_dirs("ku", use_server_model=False)
        assert rec.endswith("PP-OCRv6_small_rec")  # latin unified rec model


class TestGeometryHelpers:
    def test_get_batch_limit(self):
        # 2 cols (500 // 200) × 2 rows (300 // 150) → 4.
        assert utils.get_batch_limit(200, 150, max_width=500, max_height=300, padding=0) == 4

    def test_get_batch_limit_single_cell(self):
        assert utils.get_batch_limit(600, 400, max_width=500, max_height=300, padding=0) == 1

    def test_get_line_rects_merges_vertical_overlap(self):
        polys = [
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[0, 4], [12, 4], [12, 9], [0, 9]],   # overlaps previous vertically
            [[0, 20], [8, 20], [8, 25], [0, 25]],  # separate line
        ]
        rects = utils.get_line_rects(polys)
        assert len(rects) == 2
        assert rects[0][1] <= rects[0][3]  # y1 <= y2
        assert rects[0][1] < rects[1][1]   # sorted by y

    def test_are_rect_lists_similar_identical(self):
        rects = [[0, 0, 100, 20]]
        assert utils.are_rect_lists_similar(rects, [list(r) for r in rects], tolerance=0.1)

    def test_are_rect_lists_similar_length_mismatch(self):
        assert not utils.are_rect_lists_similar([[0, 0, 10, 10]], [], tolerance=0.1)

    def test_are_rect_lists_similar_far_apart(self):
        a = [[0, 0, 10, 10]]
        b = [[500, 500, 510, 510]]
        assert not utils.are_rect_lists_similar(a, b, tolerance=0.1)


class TestFindExecutable:
    def test_missing_raises(self, monkeypatch, tmp_path):
        # argv[0] dir contains no helper → FileNotFoundError.
        fake0 = tmp_path / "entry.py"
        fake0.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [str(fake0)])
        with pytest.raises(FileNotFoundError):
            utils.find_executable("paddleocr")

    def test_finds_helper_directory(self, monkeypatch, tmp_path):
        helper_dir = tmp_path / "paddleocr-2.7.0"
        helper_dir.mkdir()
        exe_name = "paddleocr.exe" if sys.platform == "win32" else "paddleocr.bin"
        exe = helper_dir / exe_name
        exe.write_bytes(b"")
        entry = tmp_path / "entry.py"
        entry.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [str(entry)])
        found = utils.find_executable("paddleocr")
        assert found == str(exe)


class TestReadPipe:
    def test_reads_all_lines(self):
        import io

        pipe = io.StringIO("a\nb\nc\n")
        out: list[str] = []
        utils.read_pipe(pipe, out)
        assert out == ["a\n", "b\n", "c\n"]


class TestStreamCliProcess:
    def test_yields_stdout_lines(self):
        gen = utils.stream_cli_process(
            [sys.executable, "-c", "print('one'); print('two')"], "test_stream.log"
        )
        assert list(gen) == ["one\n", "two\n"]

    def test_nonzero_exit_raises_system_exit(self, monkeypatch):
        logged: list[str] = []
        monkeypatch.setattr(
            utils, "log_error", lambda msg, log_name="error_log.txt": logged.append(msg) or "log"
        )
        gen = utils.stream_cli_process(
            [sys.executable, "-c", "import sys; print('bad'); sys.exit(2)"],
            "test_stream_fail.log",
        )
        with pytest.raises(SystemExit):
            list(gen)
        assert logged and "exit code 2" in logged[0]


def test_prepare_stitch_batch_grid_geometry():
    def _img(w, h):
        return np.zeros((h, w, 3), dtype=np.uint8)

    batch = [
        {"img": _img(100, 50), "frame_idx": 10},
        {"img": _img(100, 50), "frame_idx": 11},
        {"img": _img(100, 50), "frame_idx": 12},
    ]
    target_map: dict[str, object] = {}
    # 250px budget fits 2 × 100px images + 10px spacing → 2 columns, 2 rows.
    filepath, canvas_w, canvas_h, instructions = utils.prepare_stitch_batch(
        batch, 0, 0, "batch", "/tmp", target_map,
        max_width=250, grid_spacing=10, zero_pad_length=3,
    )
    assert filepath.replace("\\", "/").endswith("/tmp/batch_000_zone0.jpg")
    assert (canvas_w, canvas_h) == (210, 110)  # 2×100+10, 2×50+10
    assert len(instructions) == 3
    assert len(target_map) == 1
    mapping = target_map["batch_000_zone0.jpg"]
    assert [m["frame_idx"] for m in mapping] == [10, 11, 12]
    assert [(m["x"], m["y"]) for m in mapping] == [(0, 0), (110, 0), (0, 60)]
    assert all(m["grid_file"] == filepath for m in mapping)


def test_unstitch_polygon_splits_across_cells():
    # Two side-by-side 100px cells; polygon straddles the boundary.
    mapping = [
        {"grid_file": "g", "frame_idx": 0, "zone_idx": 0, "x": 0, "y": 0, "w": 100, "h": 50},
        {"grid_file": "g", "frame_idx": 1, "zone_idx": 0, "x": 100, "y": 0, "w": 100, "h": 50},
    ]
    poly = [[90, 10], [110, 10], [110, 40], [90, 40]]
    results = utils.unstitch_polygon(poly, mapping)
    assert len(results) == 2
    cells = sorted(m["frame_idx"] for _, m in results)
    assert cells == [0, 1]
    # Clipped coordinates are relative to each cell.
    for adj, m in results:
        xs = [p[0] for p in adj]
        assert all(0 <= x <= m["w"] for x in xs)


def test_unstitch_polygon_nearest_cell_fallback():
    mapping = [
        {"grid_file": "g", "frame_idx": 0, "zone_idx": 0, "x": 0, "y": 0, "w": 100, "h": 50},
    ]
    # Fully outside the cell → nearest-cell fallback (single result).
    poly = [[200, 200], [210, 200], [210, 210], [200, 210]]
    results = utils.unstitch_polygon(poly, mapping)
    assert len(results) == 1
    assert results[0][1]["frame_idx"] == 0


def test_google_lens_langs_contains_english_and_german():
    assert "en" in GOOGLE_LENS_LANGS
    assert "de" in GOOGLE_LENS_LANGS
