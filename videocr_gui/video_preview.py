"""Video preview widget: PyAV decoding + crop-box canvas (ported from legacy GUI).

The legacy ``VideoHandler`` (PyAV seek/decode + filter-graph scale) is ported
almost verbatim; the PySimpleGUI ``Graph`` is replaced by a QGraphicsView scene.
"""

from __future__ import annotations

import io
from typing import Any, cast

import av
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from . import i18n
from .config import log_error
from .style import PALETTE

# --- VideoHandler port (PyAV) -------------------------------------------------


class VideoHandler:
    def __init__(self) -> None:
        self.container: av.container.InputContainer | None = None
        self.stream: av.video.stream.VideoStream | None = None
        self.path: str | None = None
        self.width: int = 0
        self.height: int = 0
        self.duration_ms: int = 0

        self.newly_opened: bool = False
        self.last_pts: int | None = None

        self.graph: av.filter.Graph | None = None
        self.buffer_node: Any = None
        self.sink_node: Any = None
        self.last_display_size: tuple[int, int] = (0, 0)
        self.current_new_w: int = 0
        self.current_new_h: int = 0

        self._supports_threads = True

    def _frame_to_array(self, frame: av.VideoFrame, fmt: str) -> np.ndarray[Any, Any]:
        if self._supports_threads:
            try:
                return frame.to_ndarray(format=fmt, threads=1)
            except TypeError:
                self._supports_threads = False
        return frame.to_ndarray(format=fmt)

    def _get_cached_properties(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "duration_ms": self.duration_ms,
        }

    def _setup_filter_graph(
        self, template_frame: av.VideoFrame, display_size: tuple[int, int]
    ) -> None:
        scale = min(display_size[0] / self.width, display_size[1] / self.height)
        self.current_new_w, self.current_new_h = (
            int(self.width * scale) & ~1,
            int(self.height * scale) & ~1,
        )

        self.graph = av.filter.Graph()
        self.buffer_node = self.graph.add_buffer(template=cast(Any, template_frame))
        scale_node = self.graph.add(
            "scale", f"{self.current_new_w}:{self.current_new_h}:flags=bicubic"
        )
        self.sink_node = self.graph.add("buffersink")
        self.buffer_node.link_to(scale_node)
        scale_node.link_to(self.sink_node)
        self.graph.configure()
        self.last_display_size = display_size

    def open(self, path: str) -> dict[str, int]:
        if self.path == path and self.container:
            return self._get_cached_properties()

        self.close()
        try:
            self.container = av.open(path)
            self.stream = self.container.streams.video[0]
            self.stream.thread_type = "FRAME"
            self.path = path
            self.width = int(self.stream.width)
            self.height = int(self.stream.height)
            self.newly_opened = True

            if self.container.duration is not None:
                self.duration_ms = int(self.container.duration / 1000.0)
            elif self.stream.duration is not None and self.stream.time_base is not None:
                self.duration_ms = int(
                    self.stream.duration * float(self.stream.time_base) * 1000.0
                )

            return self._get_cached_properties()
        except Exception as e:
            log_error(f"VideoHandler Open Error: {e}")
            self.close()
            return {"width": 0, "height": 0, "duration_ms": 0}

    def get_frame(
        self,
        timestamp_ms: float,
        display_size: tuple[int, int],
        brightness_threshold: int | None = None,
    ) -> tuple[io.BytesIO | None, int, int, int, int]:
        """Seeks or decodes forward to provide a frame at the requested timestamp."""
        if not self.container or not self.stream:
            return None, 0, 0, 0, 0

        try:
            if self.stream.time_base is None:
                raise ValueError("Stream time_base is None")

            tb = float(self.stream.time_base)
            container_start_ms = (
                (self.container.start_time / 1000.0)
                if self.container.start_time is not None
                else 0.0
            )
            target_ms = timestamp_ms + container_start_ms
            target_pts = int(target_ms / 1000.0 / tb)
            seek_threshold = int(1.5 / tb)

            should_seek = True
            if self.newly_opened and timestamp_ms == 0:
                should_seek = False
            elif self.last_pts is not None:
                if self.last_pts <= target_pts <= (self.last_pts + seek_threshold):
                    should_seek = False

            self.newly_opened = False

            if should_seek:
                try:
                    self.container.seek(target_pts, stream=self.stream)
                    self.last_pts = None
                except Exception as e:
                    if target_pts <= 0 and getattr(e, "errno", None) == 1:
                        saved_path = self.path
                        self.close()
                        if saved_path:
                            self.open(saved_path)
                        self.newly_opened = False
                    else:
                        raise

            frame: av.VideoFrame | None = None
            for f in self.container.decode(self.stream):
                if f.pts is not None and f.pts >= target_pts:
                    frame = f
                    self.last_pts = f.pts
                    break

            if not frame:
                return None, 0, 0, 0, 0

            if self.graph is None or self.last_display_size != display_size:
                self._setup_filter_graph(frame, display_size)

            off_x = (display_size[0] - self.current_new_w) // 2
            off_y = (display_size[1] - self.current_new_h) // 2

            self.buffer_node.push(frame)
            processed_frame: av.VideoFrame = self.sink_node.pull()

            img_np = self._frame_to_array(processed_frame, fmt="rgb24")

            if brightness_threshold is not None:
                gray = (
                    (
                        img_np[..., 0].astype(np.uint16) * 77
                        + img_np[..., 1].astype(np.uint16) * 150
                        + img_np[..., 2].astype(np.uint16) * 29
                    )
                    >> 8
                ).astype(np.uint8)
                mask = gray > brightness_threshold
                img_np *= mask[..., None]

            pil_img = Image.fromarray(img_np)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format="PNG")

            return (
                io.BytesIO(img_byte_arr.getvalue()),
                self.current_new_w,
                self.current_new_h,
                off_x,
                off_y,
            )
        except Exception as e:
            log_error(f"VideoHandler Seek Error: {e}")
            return None, 0, 0, 0, 0

    def close(self) -> None:
        if self.container:
            self.container.close()
        self.container = self.stream = self.path = self.graph = self.buffer_node = (
            self.sink_node
        ) = None
        self.width = self.height = 0
        self.duration_ms = 0
        self.last_pts = None
        self.last_display_size = (0, 0)
        self.current_new_w = self.current_new_h = 0


# --- Crop box overlay -----------------------------------------------------------


class ResizableRect(QGraphicsRectItem):
    """A crop rectangle with handles; calls ``on_change`` on geometry changes."""

    HANDLE = 10
    TOLERANCE = 8

    def __init__(self, rect: QRectF, movable: bool = True) -> None:
        top_left = rect.topLeft()
        super().__init__(QRectF(0.0, 0.0, rect.width(), rect.height()))
        self.setPos(top_left)
        self.on_change = None
        self._bounds: QRectF | None = None
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, movable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPen(QPen(QColor(PALETTE["accent"]), 2))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self._resize_mode: str | None = None
        self._moving = False
        self._move_start_scene = QPointF()
        self._move_start_pos = QPointF()
        self._handles: list[QGraphicsRectItem] = []
        self._create_handles()

    def set_bounds(self, bounds: QRectF | None) -> None:
        """Constrain this crop box to the given scene-coordinate rect."""
        self._bounds = bounds

    def _create_handles(self) -> None:
        self._handles = []
        for name in ("tl", "tr", "bl", "br", "l", "r", "t", "b"):
            h = QGraphicsRectItem(0, 0, self.HANDLE, self.HANDLE, self)
            h.setPen(QPen(QColor(PALETTE["text_emphasis"]), 1))
            h.setBrush(QBrush(QColor(PALETTE["accent"])))
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            # Make handles transparent to mouse events so the parent
            # ResizableRect receives presses/moves for handle-based resizing.
            h.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            h.setData(0, name)
            self._handles.append(h)
        self._place_handles()

    def _place_handles(self) -> None:
        r = self.rect()
        h = self.HANDLE / 2
        positions = {
            "tl": (r.left() - h, r.top() - h),
            "tr": (r.right() - h, r.top() - h),
            "bl": (r.left() - h, r.bottom() - h),
            "br": (r.right() - h, r.bottom() - h),
            "l": (r.left() - h, r.center().y() - h),
            "r": (r.right() - h, r.center().y() - h),
            "t": (r.center().x() - h, r.top() - h),
            "b": (r.center().x() - h, r.bottom() - h),
        }
        for handle in self._handles:
            name = handle.data(0)
            pos = positions[name]
            handle.setPos(pos[0], pos[1])

    def _resize_from(self, mode: str, scene_pos: QPointF) -> None:
        cur = self.rect()
        scene_rect = QRectF(
            self.pos().x() + cur.left(),
            self.pos().y() + cur.top(),
            cur.width(),
            cur.height(),
        )
        r = QRectF(scene_rect)
        sx, sy = scene_pos.x(), scene_pos.y()

        # Clamp the drag position to the video bounds
        if self._bounds is not None:
            sx = max(self._bounds.left(), min(sx, self._bounds.right()))
            sy = max(self._bounds.top(), min(sy, self._bounds.bottom()))

        if "l" in mode:
            r.setLeft(min(sx, r.right() - 5))
        if "r" in mode:
            r.setRight(max(sx, r.left() + 5))
        if "t" in mode:
            r.setTop(min(sy, r.bottom() - 5))
        if "b" in mode:
            r.setBottom(max(sy, r.top() + 5))

        r = r.normalized()

        # Clamp the resulting rect so it never leaves the video area
        if self._bounds is not None:
            r.setLeft(max(r.left(), self._bounds.left()))
            r.setRight(min(r.right(), self._bounds.right()))
            r.setTop(max(r.top(), self._bounds.top()))
            r.setBottom(min(r.bottom(), self._bounds.bottom()))

        self.setRect(QRectF(0.0, 0.0, r.width(), r.height()))
        self.setPos(r.topLeft())
        self._place_handles()
        if self.on_change:
            self.on_change()

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene()
        ):
            bounds = (
                self._bounds if self._bounds is not None else self.scene().sceneRect()
            )
            new_pos = value
            r = self.rect()
            if new_pos.x() < bounds.left():
                new_pos.setX(bounds.left())
            if new_pos.y() < bounds.top():
                new_pos.setY(bounds.top())
            if new_pos.x() + r.width() > bounds.right():
                new_pos.setX(bounds.right() - r.width())
            if new_pos.y() + r.height() > bounds.bottom():
                new_pos.setY(bounds.bottom() - r.height())
            value = new_pos

        result = super().itemChange(change, value)

        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._place_handles()
            if self.on_change:
                self.on_change()
        return result

    def mousePressEvent(self, event) -> None:  # noqa: N802
        # event.pos() is in this item's local coords; map to scene for hit-testing
        scene_pos = self.mapToScene(event.pos())
        for handle in self._handles:
            if handle.sceneBoundingRect().contains(scene_pos):
                self._resize_mode = handle.data(0)
                event.accept()
                return
        # interior press: begin manual move
        self._resize_mode = None
        self._move_start_scene = scene_pos
        self._move_start_pos = self.pos()
        self._moving = True
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        scene_pos = self.mapToScene(event.pos())
        if self._resize_mode:
            self._resize_from(self._resize_mode, scene_pos)
            event.accept()
            return
        if getattr(self, "_moving", False):
            delta = scene_pos - self._move_start_scene
            self.setPos(self._move_start_pos + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._resize_mode = None
        self._moving = False
        event.accept()


class PreviewView(QGraphicsView):
    """QGraphicsView that forwards mouse events to the controller for crop drawing.

    The crop-box drawing logic lives in the parent ``VideoPreview``, but the
    video is displayed inside this view, which would otherwise consume all
    mouse events. This subclass intercepts left-button presses on empty space
    and routes draw interactions to the controller, while letting presses on
    existing crop-box items fall through to the items (move/resize).
    """

    def __init__(self, controller: VideoPreview, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self._controller = controller
        self._pressed_item: ResizableRect | None = None

    def _forward_to_item(self, item, event, scene_pos) -> QGraphicsSceneMouseEvent:
        """Builds a QGraphicsSceneMouseEvent in the item's local coords."""
        local = item.mapFromScene(scene_pos)
        scene_ev = QGraphicsSceneMouseEvent(event.type())
        scene_ev.setScenePos(scene_pos)
        scene_ev.setPos(local)
        scene_ev.setScreenPos(event.globalPosition().toPoint())
        scene_ev.setButton(event.button())
        scene_ev.setButtons(event.buttons())
        scene_ev.setModifiers(event.modifiers())
        scene_ev.setAccepted(False)
        return scene_ev

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            scene_pos = self.mapToScene(pos)
            target = self._crop_item_at(scene_pos)
            if target is not None:
                # Forward the press directly to the crop item so handle
                # resizing and interior dragging work even where the handle
                # ring overlaps the video pixmap.
                self._pressed_item = target
                scene_ev = self._forward_to_item(target, event, scene_pos)
                target.mousePressEvent(scene_ev)
                event.accept()
                return
            if self._controller.start_draw(scene_pos):
                event.accept()
                return
        super().mousePressEvent(event)

    def _crop_item_at(self, scene_pos) -> ResizableRect | None:
        """Returns the crop box item at scene_pos, including its handle ring."""
        pad = ResizableRect.HANDLE
        for rr in self._controller.crop_rect_items():
            if (
                rr.sceneBoundingRect()
                .adjusted(-pad, -pad, pad, pad)
                .contains(scene_pos)
            ):
                return rr
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._controller.is_drawing():
            scene_pos = self.mapToScene(event.position().toPoint())
            self._controller.update_draw(scene_pos)
            event.accept()
            return
        # Forward to the item we pressed on (handle resize or interior move).
        if (
            self._pressed_item is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            scene_ev = self._forward_to_item(self._pressed_item, event, scene_pos)
            self._pressed_item.mouseMoveEvent(scene_ev)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if (
            self._controller.is_drawing()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            self._controller.end_draw(scene_pos)
            event.accept()
            return
        if (
            self._pressed_item is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            scene_ev = self._forward_to_item(self._pressed_item, event, scene_pos)
            self._pressed_item.mouseReleaseEvent(scene_ev)
            self._pressed_item = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VideoPreview(QWidget):
    """Video frame display with crop-box drawing.

    Emits:
        crop_changed(list[dict])  - crop boxes in image-space coords
        frame_loaded(str, int)    - video path, duration_ms
    """

    crop_changed = Signal(list)
    frame_loaded = Signal(str, int)
    video_error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.handler = VideoHandler()
        self._display_size = (720, 405)
        self._current_pixmap: QPixmap | None = None
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_bytes: bytes | None = None
        self._offset_x = 0
        self._offset_y = 0
        self._resized_w = 0
        self._resized_h = 0
        self._orig_w = 0
        self._orig_h = 0
        self._duration_ms = 0
        self._current_ms = 0.0
        self._brightness: int | None = None
        self._crop_boxes: list[dict[str, Any]] = []
        self._drawing: tuple[QPointF, QPointF] | None = None
        self._dual_zone = False
        self._max_boxes = 1

        self._scene = QGraphicsScene(self)
        self._view = PreviewView(self, self._scene)
        self._view.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self._view.setBackgroundBrush(QBrush(QColor(PALETTE["bg_deep"])))
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.viewport().setCursor(Qt.CursorShape.CrossCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self.setMinimumHeight(300)

    # --- public API ----------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    @property
    def crop_boxes(self) -> list[dict[str, Any]]:
        return self._crop_boxes

    @property
    def original_size(self) -> tuple[int, int]:
        return self._orig_w, self._orig_h

    def set_brightness_threshold(self, value: int | None) -> None:
        self._brightness = value
        if self._current_ms is not None:
            self.show_frame(self._current_ms)

    def set_dual_zone(self, enabled: bool) -> None:
        self._dual_zone = enabled
        self._max_boxes = 2 if enabled else 1
        if len(self._crop_boxes) > self._max_boxes:
            self._crop_boxes = self._crop_boxes[: self._max_boxes]
            self._redraw_boxes()
            self.crop_changed.emit(list(self._crop_boxes))

    def load_video(self, path: str) -> bool:
        props = self.handler.open(path)
        if props["width"] == 0 or props["height"] == 0 or props["duration_ms"] <= 0:
            self.video_error.emit(path)
            return False
        self._orig_w = props["width"]
        self._orig_h = props["height"]
        self._duration_ms = props["duration_ms"]
        self._current_ms = 0.0
        self._crop_boxes = []
        self._redraw_boxes()
        self.frame_loaded.emit(path, self._duration_ms)
        self.show_frame(0)
        return True

    def show_frame(self, timestamp_ms: float) -> None:
        self._current_ms = float(timestamp_ms)
        size = self._view.viewport().size()
        if size.width() > 20 and size.height() > 20:
            self._display_size = (size.width(), size.height())
        img_bytes, w, h, off_x, off_y = self.handler.get_frame(
            self._current_ms, self._display_size, brightness_threshold=self._brightness
        )
        if img_bytes is None:
            return
        self._image_bytes = img_bytes.getvalue()
        self._resized_w, self._resized_h = w, h
        self._offset_x, self._offset_y = off_x, off_y

        image = QImage.fromData(self._image_bytes, "PNG")
        self._current_pixmap = QPixmap.fromImage(image)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(self._current_pixmap)
        self._pixmap_item.setPos(self._offset_x, self._offset_y)
        self._scene.setSceneRect(0, 0, self._display_size[0], self._display_size[1])
        self._view.setScene(self._scene)
        self._redraw_boxes()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._current_pixmap is not None:
            self.show_frame(self._current_ms)

    def clear(self) -> None:
        self.handler.close()
        self._scene.clear()
        self._pixmap_item = None
        self._current_pixmap = None
        self._crop_boxes = []
        self._orig_w = self._orig_h = self._duration_ms = 0
        self._current_ms = 0.0

    # --- crop boxes ----------------------------------------------------------

    def _img_point(self, scene_pos: QPointF) -> QPointF | None:
        x = scene_pos.x() - self._offset_x
        y = scene_pos.y() - self._offset_y
        if x < 0 or y < 0 or x >= self._resized_w or y >= self._resized_h:
            return None
        return QPointF(x, y)

    def _redraw_boxes(self) -> None:
        for item in list(self._scene.items()):
            if isinstance(item, ResizableRect) or item.data(0) == "draw-preview":
                self._scene.removeItem(item)

        video_bounds = self._video_bounds()

        for box in self._crop_boxes:
            (x1, y1), (x2, y2) = box["img_points"]
            rect = QRectF(
                min(x1, x2) + self._offset_x,
                min(y1, y2) + self._offset_y,
                abs(x2 - x1),
                abs(y2 - y1),
            )
            rr = ResizableRect(rect)
            rr.set_bounds(video_bounds)
            rr.on_change = lambda r=rr: self._sync_box_from_item(r)
            box["_item"] = rr
            self._scene.addItem(rr)

        if self._drawing is not None:
            p1, p2 = self._drawing
            rect = QRectF(p1, p2).normalized()
            tmp = QGraphicsRectItem(rect)
            tmp.setData(0, "draw-preview")
            tmp.setPen(QPen(QColor(PALETTE["accent"]), 2))
            tmp.setBrush(Qt.BrushStyle.NoBrush)
            tmp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            tmp.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self._scene.addItem(tmp)

    def _sync_box_from_item(self, item: ResizableRect) -> None:
        # Use the item's geometric rect mapped to scene coordinates instead of
        # sceneBoundingRect(), which includes the pen width (2px → 1px extra
        # per side). Using sceneBoundingRect() caused the crop box to grow
        # slightly on every seek/redraw cycle, drifting the bottom edge down.
        scene_rect = item.mapRectToScene(item.rect())
        x1 = scene_rect.left() - self._offset_x
        y1 = scene_rect.top() - self._offset_y
        x2 = scene_rect.right() - self._offset_x
        y2 = scene_rect.bottom() - self._offset_y

        box = next((b for b in self._crop_boxes if b.get("_item") is item), None)
        if box is None:
            return
        box["img_points"] = ((x1, y1), (x2, y2))

        # keep absolute video coords in sync after move/resize
        from .crop import compute_crop_coords

        box["coords"] = compute_crop_coords(
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
            self._orig_w,
            self._orig_h,
            self._resized_w,
            self._resized_h,
        )
        self.crop_changed.emit(list(self._crop_boxes))

    def is_drawing(self) -> bool:
        return self._drawing is not None

    def crop_rect_items(self) -> list[ResizableRect]:
        """Returns the live ResizableRect items for the current crop boxes."""
        return [b.get("_item") for b in self._crop_boxes if b.get("_item") is not None]

    def start_draw(self, scene_pos: QPointF) -> bool:
        """Begins drawing a crop box from a scene-space point.

        Returns True if drawing started (a frame is loaded and the point is
        inside the video area), False otherwise.
        """
        if self._current_pixmap is None:
            return False
        p = self._img_point(scene_pos)
        if p is None:
            return False
        if len(self._crop_boxes) >= self._max_boxes:
            self._crop_boxes.clear()
            self._redraw_boxes()
            self.crop_changed.emit(list(self._crop_boxes))
        self._drawing = (scene_pos, scene_pos)
        self._redraw_boxes()
        return True

    def update_draw(self, scene_pos: QPointF) -> None:
        if self._drawing is not None:
            clamped = self._clamp_scene_point(scene_pos)
            self._drawing = (self._drawing[0], clamped)
            self._redraw_boxes()

    def end_draw(self, scene_pos: QPointF) -> None:
        if self._drawing is None:
            return
        p1, p2 = self._drawing
        self._drawing = None

        # Clamp both points to the video area so the box is never silently
        # discarded just because the mouse was released slightly outside.
        img1 = self._img_point_clamped(p1)
        img2 = self._img_point_clamped(scene_pos)
        rect = QRectF(img1, img2).normalized()
        if rect.width() < 7 or rect.height() < 7:
            self._redraw_boxes()
            return

        from .crop import compute_crop_coords

        coords = compute_crop_coords(
            rect.left(),
            rect.top(),
            rect.right(),
            rect.bottom(),
            self._orig_w,
            self._orig_h,
            self._resized_w,
            self._resized_h,
        )
        self._crop_boxes.append(
            {
                "coords": coords,
                "img_points": (
                    (rect.left(), rect.top()),
                    (rect.right(), rect.bottom()),
                ),
            }
        )
        self._redraw_boxes()
        self.crop_changed.emit(list(self._crop_boxes))

    def clear_crop(self) -> None:
        self._crop_boxes = []
        self._drawing = None
        self._redraw_boxes()
        self.crop_changed.emit(list(self._crop_boxes))

    def restore_crop_boxes(self, boxes: list[dict[str, Any]]) -> None:
        """boxes: list of {'coords': {...absolute...}} -> set img_points."""
        limit = self._max_boxes
        new_boxes: list[dict[str, Any]] = []
        for box_data in boxes[:limit]:
            coords = box_data.get("coords", {})
            if not coords:
                continue
            new_boxes.append(self._make_box(coords))
        self._crop_boxes = new_boxes
        self._redraw_boxes()
        self.crop_changed.emit(list(self._crop_boxes))

    def _make_box(self, coords: dict[str, int]) -> dict[str, Any]:
        scale_w = self._resized_w / self._orig_w if self._orig_w > 0 else 0
        scale_h = self._resized_h / self._orig_h if self._orig_h > 0 else 0
        rx1 = coords["crop_x"] * scale_w
        ry1 = coords["crop_y"] * scale_h
        rx2 = (coords["crop_x"] + coords["crop_width"]) * scale_w
        ry2 = (coords["crop_y"] + coords["crop_height"]) * scale_h
        return {"coords": dict(coords), "img_points": ((rx1, ry1), (rx2, ry2))}

    def crop_coords_text(self) -> str:
        """Human-readable crop coordinates label (Zone 1/2)."""
        if not self._crop_boxes:
            return i18n.tr("crop_not_set", "Not Set")
        parts = []
        zone_text = i18n.tr("crop_zone_text", "Zone")
        for i, box in enumerate(self._crop_boxes):
            c = box["coords"]
            parts.append(
                f"{zone_text} {i + 1}: ({c['crop_x']}, {c['crop_y']}, {c['crop_width']}, {c['crop_height']})"
            )
        return "  |  ".join(parts)

    def current_position_ms(self) -> float:
        return self._current_ms

    def seek_to(self, ms: float) -> None:
        self.show_frame(ms)

    def _video_bounds(self) -> QRectF:
        """The video pixmap area in scene coordinates."""
        return QRectF(self._offset_x, self._offset_y, self._resized_w, self._resized_h)

    def _clamp_scene_point(self, scene_pos: QPointF) -> QPointF:
        """Clamp a scene point into the video area."""
        bounds = self._video_bounds()
        x = max(bounds.left(), min(scene_pos.x(), bounds.right()))
        y = max(bounds.top(), min(scene_pos.y(), bounds.bottom()))
        return QPointF(x, y)

    def _img_point_clamped(self, scene_pos: QPointF) -> QPointF:
        """Scene point → image coordinates, clamped to the video area.

        Unlike _img_point this never returns None, so a box is never
        silently discarded just because the mouse ended slightly outside.
        """
        x = scene_pos.x() - self._offset_x
        y = scene_pos.y() - self._offset_y
        x = max(0.0, min(x, float(self._resized_w)))
        y = max(0.0, min(y, float(self._resized_h)))
        return QPointF(x, y)
