# AGENTS.md

## Project overview

PySide6 GUI + CLI tool that extracts hardcoded/burned-in subtitles from videos using OCR. Fork of a fork (VideOCR -> VideOCR-AMD-DirectML -> this). Python 3.12 recommended; 3.13 not supported (torch-directml).

## Architecture

**Two packages, not one:**
- `videocr_gui/` — PySide6 GUI. Entry: `VideOCR_qt.py` → `videocr_gui.main.main()`. Runs the CLI as a subprocess via `videocr_gui/workers.py`.
- `CLI/videocr/` — Core OCR logic. Entry: `CLI/videocr_cli.py` → `CLI/videocr/api.py:save_subtitles_to_file()`. This is where all OCR engines, frame processing, and subtitle generation live.

The GUI never imports OCR libraries directly. It spawns `videocr-cli` (compiled exe or `python CLI/videocr_cli.py`) as a child process and parses stdout for progress.

**OCR engines** (in `CLI/videocr/`):
- `paddleocr` — local CPU/CUDA via PaddleOCR standalone helper executables
- `google_lens` — hybrid: local detection + Google Lens recognition
- `easyocr_directml` — AMD GPU detection via DirectML, CPU recognition (hybrid mode)
- `onnx_directml` — ONNX Runtime DirectML with PP-OCRv6, falls back to easyocr_directml

**Other key directories:**
- `tests/` — pytest suite (unit + integration). `tests/data/` holds a local test video (gitignored, see quirks).
- `tools/` — diagnostics (`test_directml.py`, `diagnose_easyocr_directml.py`), benchmarks, release notes generator, `ass_qafix/` (ASS subtitle post-processor)
- `languages/` — UI translation JSON files (14 languages)
- `Installer/` — Windows Inno Setup + Linux shell scripts
- `videocr_gui/DESIGN.md` — GUI design decisions, theme palette, module map

## Dev commands

### Lint and typecheck (pre-commit hooks)

```bash
ruff check . --fix          # lint (runs via pre-commit with --fix)
mypy .                      # typecheck (strict mode, see pyproject.toml)
```

Pre-commit config (`.pre-commit-config.yaml`) runs ruff then mypy. Install hooks with `pre-commit install`.

Notes:
- `mypy` has a **pre-existing error baseline** in the existing code; new changes should not add to it. The config must keep `explicit_package_bases = true` (otherwise `CLI/videocr/*.py` is discovered twice — as `videocr.*` and `CLI.videocr.*` — and mypy aborts before checking anything).
- `tests/` is excluded from mypy (the pytest suite is not strictly typed).

### Tests

```bash
python -m pytest tests/ -q                  # full suite (unit + integration)
python -m pytest -m "not integration" -q    # skip the real-OCR end-to-end run
python -m pytest tests/test_args.py -q      # single file
```

- Uses `pytest` + `pytest-qt` (both in the `dev` dependency group). On this machine the project env is `.conda`: `.conda\python.exe -m pytest tests/ -q`.
- Qt tests run offscreen: `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen` before any Qt import.
- `tests/conftest.py` puts the repo root and `CLI/` on `sys.path`, so both `videocr_gui` and `videocr` are importable. Autouse fixtures redirect config persistence, `log_error` writes, i18n state, and progress rate-limit state into per-test sandboxes — tests never touch the real `%APPDATA%` config.
- **Integration tests** (`-m integration`) run the real CLI on a real video through the ONNX Runtime DirectML engine (rapidocr's bundled models, no network) and parse its stdout through `progress.classify_line` — the exact GUI contract. They skip automatically when the test video or the DirectML runtime is missing.
- **The test video is not committed.** `tests/data/` is gitignored; place a clip at `tests/data/test_video_2.mp4`. The decode tests assert its properties (~17.5 s, 3840x1608, 25 fps — see `tests/test_integration.py`), so update those assertions if you use a different clip; when the file is absent they skip.

### Run from source

```bash
python VideOCR_qt.py                    # launch GUI
python CLI/videocr_cli.py -h            # CLI help
```

### DirectML development (Windows only)

```bash
python tools/test_directml.py                        # verify DirectML works
python tools/diagnose_easyocr_directml.py            # full DirectML diagnostics
python tools/list_directml_adapters.py               # list DirectML adapter indices
```

Set `VIDEOCR_DIRECTML_DEVICE_INDEX=1` to force discrete GPU on hybrid iGPU+dGPU systems.

### Build (Nuitka compilation)

```bash
python build.py --target cpu                 # CPU build
python build.py --target gpu-directml        # AMD DirectML build
python build.py --target gpu-cuda11.8        # CUDA 11.8 build
python build.py --target gpu-cuda12.9        # CUDA 12.9 build
python build.py --clean true --target cpu    # clean artifacts then build
python build.py -h                           # all options
```

Build requires: Nuitka, 7-Zip in PATH, PySide6, (Windows) C++ Build Tools + Inno Setup for installers.

### Install for development

```bash
# Standard
pip install . --group all

# DirectML (Windows, Python 3.12)
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

## Important quirks

- **Tests live in `tests/`** (pytest + pytest-qt, offscreen Qt; see the Tests section). No coverage gates are enforced; verification beyond the suite is still manual or via `tools/test_directml.py` / `tools/diagnose_easyocr_directml.py`.
- **`tests/data/` is gitignored** — local test media only; tests skip gracefully when the video is absent.
- **DirectML Windows-only.** `torch-directml`, `easyocr`, `onnxruntime-directml` are Windows-only deps. They won't install on Linux.
- **`onnxruntime` vs `onnxruntime-directml` conflict.** Both provide the `onnxruntime` module and cannot coexist. CI and setup must ensure `onnxruntime-directml` wins. See the `pyproject.toml` `[directml]` extras and `build-release.yml` for the uninstall/reinstall dance.
- **`sympy`/`mpmath` pin.** Must be `sympy==1.13.3` and `mpmath==1.3.0` for torch-directml compatibility.
- **Version lives in `_version.py`** (`__version__ = "1.6.3"`). Setuptools reads it via `attr = "_version.__version__"`.
- **Ruff config:** `target-version = "py39"`, select `E,F,I,UP,B,SIM,W`, ignore `E501,SIM108,SIM102,SIM114`. Preview mode enabled.
- **Mypy config:** strict mode with `explicit_package_bases = true`; excludes Nuitka build dirs, `Releases`, `build-artifacts`, `.conda`, and `tests`. Has a pre-existing error baseline — don't add to it.
- **Config persistence:** `videocr_gui_config.ini` (INI format via configparser). Portable mode when `portable_mode.txt` exists; otherwise `%APPDATA%/VideOCR` (Windows) or XDG (Linux).
- **GUI launches CLI as subprocess** (`workers.py`). The GUI's `workers.py` looks for: compiled `videocr-cli.exe`/`.bin` → PATH → `python CLI/videocr_cli.py` fallback.
- **Legacy OCR engine names** in old configs are remapped via `constants.LEGACY_OCR_ENGINE_MAP` (e.g., "EasyOCR DirectML (AMD GPU)" → "ONNX Runtime DirectML (AMD GPU Experimental)").
- **Build variant marker:** `build.py` writes `build_variant.txt` (`cpu`, `gpu-cuda11.8`, `gpu-cuda12.9`, `gpu-directml`) into each package root. The GUI reads it via `config.get_build_variant()` to grey out engines/backends the package cannot run; a missing marker means `"unknown"` and everything stays selectable (that is the from-source case). See `videocr_gui/DESIGN.md` § Build-variant gating.
- **GPU toggles are split:** the GUI keeps `--use_gpu` (CUDA for the PaddleOCR/Google Lens engines) and `--use_directml_gpu` (DirectML for the ONNX engine) apart, and `args.build_args()` collapses them onto the CLI's single `--use_gpu` flag based on the selected engine and the build variant.
- **Language codes are non-standard** — PaddleOCR uses custom codes (`ch`, `japan`, `korean`, `german`, `chinese_cht`). EasyOCR uses `ch_sim`, `ch_tra`. See `CLI/videocr/lang_dictionaries.py` and `easyocr_directml.py:EASYOCR_LANG_MAP`.
- **Label detection outputs .ass** (not .srt). When `enable_label_detection` is true, the output path is rewritten from `.srt` to `.ass`.
- **DirectML recognition mode:** keep on `stable` (CPU recognition). The `auto`/`experimental` modes hit EasyOCR LSTM operator crashes on DirectML (`aten::_thnn_fused_lstm_cell`).
- **FFmpeg D3D11VA path** uses `-fps_mode vfr` (not the removed `-vsync 0`).
- **GUI theme:** "warm obsidian & gold" — dark-only, no light theme. Palette and QSS in `videocr_gui/style.py`. Design docs in `videocr_gui/DESIGN.md`.
- **Nuitka GUI compilation** excludes heavy ML libs (`--nofollow-import-to=torch,torchvision,...`) to avoid MSVC heap exhaustion. The GUI never imports OCR directly.
- **CI builds** (`.github/workflows/build-release.yml`): tag-triggered, builds all targets (CPU + CUDA variants + DirectML on Windows), creates draft GitHub release with assets. No code signing in CI.
