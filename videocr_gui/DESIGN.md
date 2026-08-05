# VideOCR Recreated — PySide6 GUI Design Notes

This document records the design decisions and feature inventory for the
"VideOCR Recreated" GUI, which replaces the old PySimpleGUI `VideOCR.py`.

## Decisions (user-confirmed)

1. The old GUI (`VideOCR.py`, PySimpleGUI, ~4600 lines) is **deleted**.
   The new PySide6 GUI is the application.
2. Benchmark-related removal is **GUI-only**:
   - The "Last Run Benchmark" panel, the benchmark compare checkboxes, and the
     **Apply Tested AMD Preset** button are NOT ported.
   - CLI arguments (`--benchmark_compare_engine`, `--benchmark_compare_sample_grids`),
     `[Perf]`/`[Bench]` CLI output, and `tools/benchmark_*.py` are left intact.
3. Update-checking is **removed entirely** from the GUI:
   - No startup update thread, no "Check Now" button, no `check_for_updates()`,
     no `update_*` popups.
   - The About tab keeps the GitHub releases/issues *links* (manual navigation only).
4. The new GUI lives in a small package: `videocr_gui/`, with root launcher
   `VideOCR_qt.py`.

## DirectML device-index persistence (new behavior)

- The DirectML GPU selector index is **saved to the config file** and restored
  on startup (the old GUI kept the index in a hidden input but the dropdown
  selection was not reliably persisted).
- Default index is **0** (first GPU / "Auto (recommended)" resolves to no index).

## vsync / ffmpeg compatibility

- `-vsync 0` was removed in newer ffmpeg. The equivalent modern option used here
  is `-fps_mode vfr` (raw-video pipe output preserves each selected frame).
  Applied in `CLI/videocr/video.py` (FFmpeg D3D11VA path) and
  `tools/benchmark_amd_decode.py` (same helper fix).

## Feature inventory ported from VideOCR.py

- Process Video tab: Open File/Folder, video preview (PyAV decode + seek),
  crop-box draw/move/resize (single + dual zone), brightness-threshold overlay,
  Run/Pause/Cancel, progress + ETA + taskbar progress, timestamped log pane,
  "When ready:" post-completion action with countdown, wake-lock, notifications.
- Queue tab: table, multi-select + shift-click, reorder, reset, edit, remove,
  clear, start/stop/pause, duplicate-output dialog, "Add All to Queue" scan.
- Advanced Settings tab: all OCR thresholds/toggles, subtitle position/alignment,
  DirectML controls (GPU dropdown + Refresh, performance preset, recognition
  mode, frame scan mode, ONNX tuning, grid max width/height), UI language,
  GUI scaling, save options, output dir, seek step, notification, sleep,
  normalize-chinese.
- Config: `videocr_gui_config.ini` (portable mode via `portable_mode.txt`,
  else `%APPDATA%/VideOCR` or XDG), autosave on change, relative crop-box
  persistence, defaults matching the old GUI except DirectML index default 0
  and the removed update/benchmark keys.

## Module map

```
videocr_gui/
  __init__.py      package metadata (APP_NAME = "VideOCR Recreated")
  main.py          QApplication bootstrap + entry point
  app.py           MainWindow: tabs, state, event wiring
  config.py        config file path + load/save + defaults
  i18n.py          languages/*.json loading + translation helper
  resources.py     icon paths, scaling helpers
  crop.py          crop box model + coordinate math + saved-relative parsing
  progress.py      CLI progress/ETA parsing (ported handle_progress logic)
  dialogs.py       centered modal dialogs + post-action countdown
  video_preview.py PyAV VideoHandler port + QGraphicsView crop canvas
  workers.py       videocr-cli subprocess worker (QThread) + kill/pause helpers
  settings_tab.py  Advanced Settings tab widget
  queue_tab.py     Queue tab widget
  about_tab.py     About tab widget
  DESIGN.md        this file
```
