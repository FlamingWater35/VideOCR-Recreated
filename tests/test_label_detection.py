"""Unit tests for the label-detection pipeline (merge, voting, rendering)."""

from __future__ import annotations

from videocr.models import PredictedFrames
from videocr.video import Video


def _bare_video() -> Video:
    """A Video with just the state _merge_label_frames/_render_ass need."""
    v = Video.__new__(Video)
    v.pred_subs = []
    v.label_zones = []
    v.label_frames = []
    v._label_frame_weights = {}
    v.frame_timestamps = {}
    v.start_time_offset_ms = 0.0
    v.avg_frame_duration_ms = 40.0
    v.fps = 25.0
    v.height = 600
    v.width = 640
    v.label_time_start_ms = 0.0
    v.label_time_end_ms = None
    return v


def _label_frame(
    index: int,
    text: str,
    box: list[list[float]],
    conf: float,
    zone_index: int = 2,
) -> PredictedFrames:
    pred_data = [[[box, (text, conf)]]]
    return PredictedFrames(
        "onnx_directml", index, pred_data, 0.6, zone_index, "en", False
    )


class TestPickBestLabelText:
    def test_confidence_weighted_votes_win(self):
        # Two votes each, but "ab" was read with higher confidence.
        event = {
            "votes": {"ab": 2 * 0.95, "abc": 2 * 0.6},
            "text_samples": ["ab", "abc"],
        }
        assert Video._pick_best_label_text(event) == "ab"

    def test_tie_prefers_longest_original_variant(self):
        event = {
            "votes": {"ab": 1.0, "abc": 1.0},
            "text_samples": ["ab", "abc"],
        }
        assert Video._pick_best_label_text(event) == "abc"

    def test_no_votes_falls_back_to_longest_sample(self):
        event = {"votes": {}, "text_samples": ["a", "abc"]}
        assert Video._pick_best_label_text(event) == "abc"

    def test_empty_event_returns_empty(self):
        assert Video._pick_best_label_text({}) == ""


class TestRenderAssPos:
    def test_pos_uses_bbox_top_left_not_center(self):
        r"""\pos must anchor the bbox top-left for the default \an7 style."""
        v = _bare_video()
        band = {
            "x": 0,
            "y": 0,
            "w": 640,
            "h": 100,
            "crop_x": 0,
            "crop_y": 0,
            "crop_w": 640,   # 1:1 scale: crop coords == frame coords
            "crop_h": 100,
            "target_w": 640,
            "target_h": 100,
        }
        v.label_zones = [band]
        box = [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]
        v.label_frames = [
            _label_frame(0, "Hunter", box, 0.95),
            _label_frame(1, "Hunter", box, 0.95),
        ]
        v.frame_timestamps = {0: 0.0, 1: 40.0, 2: 80.0}
        out = v._render_ass([None, None])
        assert "Dialogue: 2," in out
        # bbox top-left (10,20); the old center-based pos would be (60,40).
        assert "{\\pos(10,20)}" in out
        assert "{\\pos(60,40)}" not in out

    def test_events_never_merge_across_bands(self):
        """Simultaneous labels in two different bands stay separate events."""
        v = _bare_video()
        top_band = {
            "x": 0, "y": 0, "w": 640, "h": 100,
            "crop_x": 0, "crop_y": 0,
            "crop_w": 640, "crop_h": 100,
            "target_w": 640, "target_h": 100,
        }
        bottom_band = {
            "x": 0, "y": 500, "w": 640, "h": 100,
            "crop_x": 0, "crop_y": 500,
            "crop_w": 640, "crop_h": 100,
            "target_w": 640, "target_h": 100,
        }
        v.label_zones = [top_band, bottom_band]
        box = [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]
        # Identical text/box at the same frame in BOTH bands.
        f_top0 = _label_frame(0, "Hunter", box, 0.95, 2)
        f_top1 = _label_frame(1, "Hunter", box, 0.95, 2)
        f_bot0 = _label_frame(0, "Hunter", box, 0.95, 3)
        f_bot1 = _label_frame(1, "Hunter", box, 0.95, 3)
        v.label_frames = [f_top0, f_top1, f_bot0, f_bot1]
        v.frame_timestamps = {0: 0.0, 1: 40.0, 2: 80.0}
        out = v._render_ass([None, None])
        label_lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue: 2,")]
        # Two events, one per band — mapped to different frame Y positions.
        assert len(label_lines) == 2
        pos_values = [
            ln.split("{\\pos(")[1].split(")}")[0] for ln in label_lines
        ]
        assert "10,20" in pos_values          # top band (crop_y=0)
        assert "10,520" in pos_values         # bottom band (crop_y=500)

    def test_custom_style_and_alignment_rendered(self):
        v = _bare_video()
        band = {
            "x": 0, "y": 0, "w": 640, "h": 100,
            "crop_x": 0, "crop_y": 0,
            "crop_w": 640, "crop_h": 100,
            "target_w": 640, "target_h": 100,
        }
        v.label_zones = [band]
        v._label_style = {
            "font": "Verdana",
            "fontsize": 30,
            "primary": "&H0000FF00",
            "outline": "&H000000FF",
            "alignment": "an9",
        }
        box = [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]
        v.label_frames = [
            _label_frame(0, "Hunter", box, 0.95),
            _label_frame(1, "Hunter", box, 0.95),
        ]
        v.frame_timestamps = {0: 0.0, 1: 40.0, 2: 80.0}
        out = v._render_ass([None, None])
        assert (
            "Style: Label,Verdana,30,&H0000FF00,&H000000FF,&H000000FF,"
            "&H80000000,0,0,0,0,100,100,0,0,1,1.5,0.5,9,10,10,10,1"
        ) in out
        # \an9 anchors the bbox top-right → pos = (110, 20).
        assert "{\\pos(110,20)}" in out

    def test_overlap_margin_scales_with_fontsize(self):
        """Larger label fonts get a larger vertical gap between colliding labels."""
        v = _bare_video()
        band = {
            "x": 0, "y": 0, "w": 640, "h": 200,
            "crop_x": 0, "crop_y": 0,
            "crop_w": 640, "crop_h": 200,
            "target_w": 640, "target_h": 200,
        }
        v.label_zones = [band]
        v._label_style = {
            "font": "Arial",
            "fontsize": 60,
            "primary": "&H00FFFFFF",
            "outline": "&H00000000",
            "alignment": "an7",
        }
        box_a = [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]
        # Shifted >40 px in x so the close-position merge shortcut does not
        # swallow "Beta" into the "Alpha" event.
        box_b = [[60.0, 20.0], [160.0, 20.0], [160.0, 60.0], [60.0, 60.0]]
        v.label_frames = [
            _label_frame(0, "Alpha", box_a, 0.95),
            _label_frame(1, "Alpha", box_a, 0.95),
            _label_frame(2, "Beta", box_b, 0.95),
            _label_frame(3, "Beta", box_b, 0.95),
        ]
        v.frame_timestamps = {0: 0.0, 1: 40.0, 2: 80.0, 3: 120.0, 4: 160.0}
        out = v._render_ass([None, None])
        # margin = max(4.0, 60/6) = 10 → Beta pushed to y = 60 + 10 = 70;
        # at the default 22 pt the gap would only be 4 px (y = 64).
        assert "{\\pos(10,20)}" in out   # Alpha untouched
        assert "{\\pos(60,70)}" in out   # Beta shifted by the scaled margin
        assert "{\\pos(60,20)}" not in out
