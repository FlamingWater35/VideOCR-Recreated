# VideOCR Recreated — PySide6 GUI Design Notes

This document records the design decisions and feature inventory for the
"VideOCR Recreated" GUI, which replaces the old PySimpleGUI `VideOCR.py`.

## Design language: "warm obsidian & gold"

The GUI uses a warm, near-black **obsidian** foundation lit by a single
imperial-**gold** signal (Shēng Luó, `#e1a636`), with a small vocabulary of
meaningful semantic states. This replaces the original neon-dark theme's
cold navy + cyan/violet accent pair.

Inspiration: the "Imperial Command Deck" aesthetic — a console that glows
from within, gold on warm obsidian, with calm and exact controls.

### Palette (source of truth: `videocr_gui/style.py` → `PALETTE`)

| Token | Hex | Use |
|---|---|---|
| `bg_deep` | `#0c0a07` | window / dialog background (obsidian) |
| `bg_surface` | `#16110a` | tabs, inputs, group boxes (surface) |
| `bg_raised` | `#201810` | hovered surfaces (raised) |
| `bg_hover` | `#2a2114` | interactive hover wash |
| `border` | `#2c2316` | default borders (gold-tinted hairline) |
| `border_strong` | `#3a2f1c` | emphasized borders, scrollbar handles |
| `accent` | `#e1a636` | **gold** — primary actions, selection, focus |
| `accent_hi` | `#ffd98a` | gold highlight — hover, bloom core |
| `accent_deep` | `#8f661f` | gold deep — pressed, gradient foot |
| `amber` | `#f0b53e` | paused / needs a look |
| `text_emphasis` | `#fff4dc` | emphasis text, handle outlines |
| `danger` | `#ff8a8a` | coral — destructive actions, errors |
| `ok` | `#9bd6a0` | jade — success states |
| `text` | `#ecdfc6` | primary text |
| `text_dim` | `#9c8b69` | secondary text |
| `text_faint` | `#6f6347` | disabled text |

### Semantic state colors

- **Translating / live work** → gold `#e1a636`
- **Paused / needs a look** → amber `#f0b53e`
- **Done / clean** → jade `#9bd6a0`
- **Failed / cancelled** → coral `#ff8a8a`
- **Pending / waiting** → muted `#9c8b69`

These appear in the queue table status colors (`queue_tab.py`) and follow the
QSS theme.

### Named rules

- **The Signal Rule.** Gold appears on roughly ≤10% of any screen — actions,
  focus, the current selection, live work. Its rarity against the obsidian is
  why it reads as a signal.
- **Warm-Obsidian Ladder.** Elevation is expressed by stepping *up* the
  warm-obsidian ladder, never by darkening a shadow. The faint gold in every
  neutral keeps the dark reading as *lit* rather than grey.
- **State-Reads-Twice.** Every semantic color is paired with a text label
  (queue status text); hue alone never carries meaning.

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
5. The GUI is **dark-only** — there is no light theme. The warm-obsidian + gold
   identity is deliberate.
6. This project is a **fork** of the original VideOCR (by `timminator`); the
   About tab states this and links both this repository
   (`FlamingWater35/VideOCR-Recreated`) and the original.

## DirectML device-index persistence (new behavior)

- The DirectML GPU selector index is **saved to the config file** and restored
  on startup (the old GUI kept the index in a hidden input but the dropdown
  selection was not reliably persisted).
- Default index is **0** (first GPU / "Auto (recommended)" resolves to no index).

## Dependent-control disabling (new behavior)

- The DirectML controls (GPU dropdown, performance preset, recognition mode,
  frame scan mode, ONNX tuning, grid width/height, Refresh button) are
  **disabled in the GUI** while the **Enable GPU Usage** checkbox is unchecked.
- The label-detection controls (label OCR width, min duration, confirmation
  frames, reappear gap, single-char filter) are **disabled** while the
  **Enable Label Detection** checkbox is unchecked.
- Implemented in `settings_tab.py` (`DML_DEPENDENT_WIDGETS`,
  `LABEL_DEPENDENT_WIDGETS`, `_update_dependent_states()`), re-evaluated on
  toggle and on `populate()`.

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
- Label detection section: `enable_label_detection`, label OCR width, min
  display duration, min confirmation frames, reappear merge gap, single-char
  filter (`.ass` output with positioned Label events).
- Config: `videocr_gui_config.ini` (portable mode via `portable_mode.txt`,
  else `%APPDATA%/VideOCR` or XDG), autosave on change, relative crop-box
  persistence, defaults matching the old GUI except DirectML index default 0
  and the removed update/benchmark keys.

## Theme / styling

- `videocr_gui/style.py` owns the QSS stylesheet and the shared `PALETTE`
  dict. `apply(app)` sets the Fusion style + stylesheet.
- Hardcoded colors in drawing code (queue status colors, crop-box pen/brush,
  preview background) import `PALETTE` instead of duplicating hex values.
- Font stack: `Segoe UI` / `Noto Sans` (UI) and `Cascadia Mono` / `Consolas`
  (log pane).

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
  style.py         QSS theme ("warm obsidian & gold") + shared PALETTE
  DESIGN.md        this file
```
