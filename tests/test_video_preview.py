"""Tests for the preview's async, latest-wins seek pipeline.

Frame decoding runs on a background worker so fast seeking never blocks the
UI thread; these tests pin the coalescing contract without needing a real
video (``VideoHandler.get_frame`` is faked).
"""

from __future__ import annotations

import threading
import time

from videocr_gui.video_preview import VideoPreview


class TestAsyncSeek:
    def test_rapid_seeks_decode_only_first_and_latest(self, qtbot):
        preview = VideoPreview()
        decoded: list[float] = []

        def fake_get_frame(ms, display_size, brightness_threshold=None):
            time.sleep(0.05)
            decoded.append(ms)
            return (None, 0, 0, 0, 0)

        preview.handler.get_frame = fake_get_frame  # type: ignore[method-assign]
        try:
            for i in range(20):
                preview.seek_to(i * 100.0)

            qtbot.waitUntil(lambda: len(decoded) >= 2, timeout=3000)
            qtbot.waitUntil(lambda: not preview._decode_busy, timeout=1000)
            # First seek decodes immediately (snappy); the other 19 requests
            # coalesce while it runs and only the last target is decoded next.
            assert decoded == [0.0, 1900.0]
            assert preview._pending_seek_ms is None
        finally:
            preview.shutdown()

    def test_geometry_applied_before_first_frame(self):
        preview = VideoPreview()
        release = threading.Event()
        try:
            preview._orig_w, preview._orig_h = 1920, 1080

            def gated_get_frame(ms, display_size, brightness_threshold=None):
                release.wait(2.0)
                return (None, 0, 0, 0, 0)

            preview.handler.get_frame = gated_get_frame  # type: ignore[method-assign]
            preview.show_frame(500.0)
            # Crop-box restore in the main window maps coordinates right after
            # the first frame is *requested*, so geometry must be set eagerly.
            assert preview._resized_w > 0
            assert preview._resized_h > 0
        finally:
            release.set()
            preview.shutdown()
