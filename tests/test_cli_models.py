"""Tests for videocr.models — PredictedText, PredictedFrames, PredictedSubtitle."""

from __future__ import annotations

import pytest

from videocr.models import PredictedFrames, PredictedSubtitle, PredictedText


def _box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _word(x1, y1, x2, y2, text, conf):
    return [_box(x1, y1, x2, y2), (text, conf)]


def _frame(index=0, words=None, engine="paddleocr", conf_threshold=50,
           zone=0, lang="en", normalize=False, pred_data=None):
    data = pred_data if pred_data is not None else [words if words is not None else []]
    return PredictedFrames(engine, index, data, conf_threshold, zone, lang, normalize)


class TestPredictedText:
    def test_dataclass_fields(self):
        word = PredictedText(_box(0, 0, 10, 10), 0.9, "hi")
        assert word.confidence == 0.9
        assert word.text == "hi"
        assert word.bounding_box[0] == [0, 0]


class TestPredictedFrames:
    def test_basic_text_with_space_join(self):
        words = [
            _word(0, 10, 20, 30, "Hello", 90),
            _word(25, 10, 45, 30, "World", 80),
        ]
        frame = _frame(words=words)
        assert frame.text == "Hello World"
        assert frame.confidence == pytest.approx(85.0)
        assert len(frame.lines) == 1

    def test_google_lens_joins_without_space(self):
        words = [
            _word(0, 10, 20, 30, "Hel", 90),
            _word(25, 10, 45, 30, "lo", 80),
        ]
        frame = _frame(words=words, engine="google_lens")
        assert frame.text == "Hello"

    def test_words_below_threshold_dropped(self):
        words = [
            _word(0, 10, 20, 30, "keep", 90),
            _word(25, 10, 45, 30, "drop", 30),
        ]
        frame = _frame(words=words, conf_threshold=50)
        assert frame.text == "keep"
        assert frame.confidence == 90.0

    def test_threshold_is_inclusive(self):
        frame = _frame(words=[_word(0, 10, 20, 30, "exact", 50)], conf_threshold=50)
        assert frame.text == "exact"

    def test_empty_prediction_high_confidence(self):
        frame = _frame(words=[])
        assert frame.text == ""
        assert frame.confidence == 100  # no detections at all

    def test_all_filtered_out_zero_confidence(self):
        frame = _frame(words=[_word(0, 10, 20, 30, "low", 10)], conf_threshold=50)
        assert frame.text == ""
        assert frame.confidence == 0

    def test_malformed_word_skipped(self):
        words = [
            ["only-a-box"],  # len < 2 → skipped
            _word(0, 10, 20, 30, "ok", 90),
        ]
        frame = _frame(words=words)
        assert frame.text == "ok"

    def test_multiple_lines_grouped_and_sorted_by_y(self):
        words = [
            _word(0, 50, 30, 70, "bottom", 90),
            _word(0, 10, 30, 30, "top", 90),
        ]
        frame = _frame(words=words)
        assert len(frame.lines) == 2
        assert frame.text == "top\nbottom"

    def test_words_within_line_sorted_by_x(self):
        words = [
            _word(50, 10, 80, 30, "world", 90),
            _word(0, 10, 20, 30, "hello", 90),
        ]
        frame = _frame(words=words)
        assert frame.text == "hello world"

    def test_rtl_language_reverses_word_order(self):
        words = [
            _word(50, 10, 80, 30, "سلام", 90),
            _word(0, 10, 20, 30, "جهان", 90),
        ]
        frame = _frame(words=words, lang="ar")
        # RTL → rightmost word first.
        assert frame.text == "سلام جهان"

    def test_index_and_zone_recorded(self):
        frame = _frame(index=42, words=[_word(0, 10, 20, 30, "x", 90)], zone=1)
        assert frame.start_index == 42
        assert frame.end_index == 42
        assert frame.zone_index == 1

    def test_chinese_normalization_applies(self):
        # Traditional "電腦" → simplified "电脑".
        frame = _frame(
            words=[_word(0, 10, 60, 30, "電腦", 90)],
            lang="ch", normalize=True,
        )
        assert frame.text == "电脑"

    def test_chinese_normalization_skipped_when_disabled(self):
        frame = _frame(
            words=[_word(0, 10, 60, 30, "電腦", 90)],
            lang="ch", normalize=False,
        )
        assert frame.text == "電腦"

    def test_normalization_only_for_chinese_lang(self):
        frame = _frame(
            words=[_word(0, 10, 60, 30, "電腦", 90)],
            lang="en", normalize=True,
        )
        assert frame.text == "電腦"  # lang not in (ch, zh-CN) → untouched


class TestPredictedSubtitle:
    def _sub(self, frames, sim_threshold=80, lang="en", lm=None):
        return PredictedSubtitle(frames, 0, sim_threshold, lang, lm)

    def test_filters_zero_confidence_frames(self):
        good = _frame(index=5, words=[_word(0, 10, 20, 30, "good", 90)])
        # conf below threshold → all words filtered → confidence 0 → dropped.
        bad = _frame(index=6, words=[_word(0, 10, 20, 30, "x", 10)], conf_threshold=50)
        sub = self._sub([good, bad])
        assert sub.frames == [good]

    def test_frames_sorted_by_index(self):
        f2 = _frame(index=9, words=[_word(0, 10, 20, 30, "later", 80)])
        f1 = _frame(index=3, words=[_word(0, 10, 20, 30, "earlier", 70)])
        sub = self._sub([f2, f1])
        assert [f.start_index for f in sub.frames] == [3, 9]

    def test_text_from_highest_confidence_frame(self):
        low = _frame(index=1, words=[_word(0, 10, 20, 30, "low conf", 60)])
        high = _frame(index=2, words=[_word(0, 10, 20, 30, "high conf", 95)])
        sub = self._sub([low, high])
        assert sub.text == "high conf"

    def test_empty_frames(self):
        sub = self._sub([])
        assert sub.text == ""
        assert sub.index_start == 0
        assert sub.index_end == 0

    def test_index_range(self):
        f1 = _frame(index=3, words=[_word(0, 10, 20, 30, "a", 90)])
        f2 = _frame(index=7, words=[_word(0, 10, 20, 30, "b", 90)])
        sub = self._sub([f1, f2])
        assert sub.index_start == 3
        assert sub.index_end == 7

    def test_is_similar_identical(self):
        a = self._sub([_frame(words=[_word(0, 0, 10, 10, "Hello World", 90)])])
        b = self._sub([_frame(words=[_word(0, 0, 10, 10, "Hello World", 90)])])
        assert a.is_similar_to(b) is True

    def test_is_similar_ignores_spaces(self):
        a = self._sub([_frame(words=[_word(0, 0, 10, 10, "HelloWorld", 90)])])
        b = self._sub([_frame(words=[_word(0, 0, 10, 10, "Hello World", 90)])])
        assert a.is_similar_to(b) is True

    def test_not_similar_when_different(self):
        a = self._sub([_frame(words=[_word(0, 0, 10, 10, "completely different text", 90)])], sim_threshold=80)
        b = self._sub([_frame(words=[_word(0, 0, 10, 10, "nothing alike at all", 90)])])
        assert a.is_similar_to(b) is False

    def test_similarity_threshold_respected(self):
        a = self._sub([_frame(words=[_word(0, 0, 10, 10, "hello", 90)])], sim_threshold=100)
        b = self._sub([_frame(words=[_word(0, 0, 10, 10, "hallo", 90)])])
        assert a.is_similar_to(b) is False  # fuzz ratio < 100


class TestFinalizeText:
    def _frame_with(self, text, conf, index=0):
        return _frame(index=index, words=[_word(0, 10, 60, 30, text, conf)])

    def test_majority_vote(self):
        frames = [
            self._frame_with("Hello", 70, 0),
            self._frame_with("Hxllo", 99, 1),
            self._frame_with("Hello", 80, 2),
        ]
        sub = PredictedSubtitle(frames, 0, 80, "en", None)
        sub.finalize_text(post_processing=False)
        assert sub.text == "Hello"  # 2 votes vs 1

    def test_tie_broken_by_mean_confidence(self):
        frames = [
            self._frame_with("variantA", 60, 0),
            self._frame_with("variantB", 95, 1),
        ]
        sub = PredictedSubtitle(frames, 0, 80, "en", None)
        sub.finalize_text(post_processing=False)
        assert sub.text == "variantB"

    def test_single_frame(self):
        sub = PredictedSubtitle([self._frame_with("only", 88)], 0, 80, "en", None)
        sub.finalize_text(post_processing=False)
        assert sub.text == "only"

    def test_post_processing_uses_language_model(self):
        class FakeLM:
            def __init__(self):
                self.seen = []

            def rejoin(self, text):
                self.seen.append(text)
                return "REJOINED"

        lm = FakeLM()
        sub = PredictedSubtitle(
            [self._frame_with("helloworld", 90)], 0, 80, "en", lm
        )
        sub.finalize_text(post_processing=True)
        assert lm.seen == ["helloworld"]
        assert sub.text == "REJOINED"

    def test_post_processing_without_language_model_is_noop_for_en(self):
        sub = PredictedSubtitle(
            [self._frame_with("helloworld", 90)], 0, 80, "en", None
        )
        sub.finalize_text(post_processing=True)
        assert sub.text == "helloworld"  # lm is None → untouched

    def test_post_processing_chinese_segments(self):
        sub = PredictedSubtitle(
            [self._frame_with("abc def 中文", 90)], 0, 80, "ch", None
        )
        sub.finalize_text(post_processing=True)
        assert isinstance(sub.text, str)
        assert "中" in sub.text and "文" in sub.text  # chinese chars preserved

    def test_post_processing_skipped_when_disabled(self):
        class FakeLM:
            def rejoin(self, text):
                raise AssertionError("should not be called")

        sub = PredictedSubtitle(
            [self._frame_with("helloworld", 90)], 0, 80, "en", FakeLM()
        )
        sub.finalize_text(post_processing=False)
        assert sub.text == "helloworld"
