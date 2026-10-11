"""Regression tests for tools/ass_qafix layer handling.

ass_qafix's merge functions used to hardcode ``Layer: 0`` when rebuilding
merged lines, which silently stripped Layer 2 from merged label events.
"""

from __future__ import annotations

import pytest

from videocr.api import _load_qafix

FIELDS = [
    "Layer",
    "Start",
    "End",
    "Style",
    "Name",
    "MarginL",
    "MarginR",
    "MarginV",
    "Effect",
    "Text",
]


@pytest.fixture(scope="module")
def qafix(repo_root):
    mod = _load_qafix(str(repo_root / "tools" / "ass_qafix" / "ass_qafix.py"))
    assert mod is not None, "ass_qafix.py failed to load"
    return mod


def _dlg(layer: int, start: str, end: str, text: str) -> str:
    return f"Dialogue: {layer},{start},{end},Label,,0,0,0,,{text}"


class TestMergePreservesLayer:
    def test_same_layer_merge_keeps_layer_2(self, qafix):
        lines = [
            _dlg(2, "0:00:01.00", "0:00:02.00", "abc"),
            _dlg(2, "0:00:02.10", "0:00:03.00", "abc"),  # 10cs gap <= 500ms
        ]
        merged, count = qafix.merge_consecutive_dialogues(lines, FIELDS)
        assert count == 1
        assert len(merged) == 1
        assert merged[0].startswith("Dialogue: 2,")

    def test_layer_0_merge_still_keeps_layer_0(self, qafix):
        lines = [
            _dlg(0, "0:00:01.00", "0:00:02.00", "abc"),
            _dlg(0, "0:00:02.10", "0:00:03.00", "abc"),
        ]
        merged, count = qafix.merge_consecutive_dialogues(lines, FIELDS)
        assert count == 1
        assert merged[0].startswith("Dialogue: 0,")

    def test_cross_layer_not_merged(self, qafix):
        lines = [
            _dlg(0, "0:00:01.00", "0:00:02.00", "abc"),
            _dlg(2, "0:00:02.10", "0:00:03.00", "abc"),
        ]
        merged, count = qafix.merge_consecutive_dialogues(lines, FIELDS)
        assert count == 0
        assert merged == lines

    def test_alternating_merge_keeps_layer_2(self, qafix):
        lines = [
            _dlg(2, "0:00:01.00", "0:00:02.00", "ABCD"),
            _dlg(2, "0:00:02.00", "0:00:03.00", "ABCE"),
            _dlg(2, "0:00:03.00", "0:00:04.00", "ABCD"),
        ]
        merged, count = qafix.merge_alternating_ocr_variants(lines, FIELDS)
        assert count >= 1
        assert len(merged) == 1
        assert merged[0].startswith("Dialogue: 2,")

    def test_alternating_cross_layer_not_merged(self, qafix):
        lines = [
            _dlg(2, "0:00:01.00", "0:00:02.00", "ABCD"),
            _dlg(0, "0:00:02.00", "0:00:03.00", "ABCE"),
            _dlg(2, "0:00:03.00", "0:00:04.00", "ABCD"),
        ]
        merged, count = qafix.merge_alternating_ocr_variants(lines, FIELDS)
        assert count == 0
        assert merged == lines


class TestOverlapFixRespectsLayer:
    def test_same_layer_overlap_is_fixed(self, qafix):
        lines = [
            _dlg(2, "0:00:01.00", "0:00:02.02", "abc"),  # ends 2cs into next
            _dlg(2, "0:00:02.00", "0:00:03.00", "abc"),
        ]
        fixed, count = qafix.fix_overlapping_timestamps(lines, FIELDS)
        assert count == 1
        assert ",0:00:02.00," in fixed[0]  # first end pulled back
        assert ",0:00:02.02," in fixed[1]  # next start pushed forward

    def test_cross_layer_overlap_is_left_alone(self, qafix):
        lines = [
            _dlg(0, "0:00:01.00", "0:00:02.02", "abc"),
            _dlg(2, "0:00:02.00", "0:00:03.00", "abc"),
        ]
        fixed, count = qafix.fix_overlapping_timestamps(lines, FIELDS)
        assert count == 0
        assert fixed[0] == lines[0]
        assert fixed[1] == lines[1]


class TestFakeTextSeesThroughOverrideBlocks:
    """Label events wrap their text in a {\\pos(...)} override block (and
    subtitle lines may carry {\\anN} alignment tags). The fake-text checks are
    fullmatch-based, so classification must run on the visible text —
    otherwise digit/short-letter OCR noise behind a tag is never removed."""

    @staticmethod
    def _process(qafix, text: str):
        stats = qafix.QAStats()
        return qafix.process_dialogue_line(
            _dlg(2, "0:00:01.00", "0:00:02.00", text),
            FIELDS,
            {"Label"},
            stats,
            False,
        )

    def test_pure_digits_behind_pos_removed(self, qafix):
        assert self._process(qafix, "{\\pos(10,20)}00000") is None

    def test_alignment_tagged_digits_removed(self, qafix):
        assert self._process(qafix, "{\\an8}28") is None

    def test_single_letter_behind_pos_removed(self, qafix):
        assert self._process(qafix, "{\\pos(10,20)}M") is None

    def test_real_text_behind_pos_kept(self, qafix):
        out = self._process(qafix, "{\\pos(10,20)}Hello world")
        assert out is not None
        assert "{\\pos(10,20)}" in out
        assert "Hello world" in out

    def test_cjk_text_behind_pos_kept(self, qafix):
        out = self._process(qafix, "{\\pos(801,356)}乔羽")
        assert out is not None
        assert "乔羽" in out
