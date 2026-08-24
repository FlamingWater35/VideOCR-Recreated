<p align="center">
  <img src="Pictures/VideOCR.png" alt="VideOCR Icon" width="128">
  <h1 align="center">VideOCR Recreated</h1>
  <p align="center">
    Extract hardcoded subtitles from videos with a modern PySide6 GUI, 200+ languages, and experimental AMD DirectML GPU support.
    <br>
    <em>Forked from <a href="https://github.com/timminator/VideOCR">VideOCR</a> (by <code>timminator</code>), via <a href="https://github.com/BaseCrunch/VideOCR-AMD-DirectML">VideOCR-AMD-DirectML</a> (by <code>BaseCrunch</code>).</em>
  </p>

  <p align="center">
    <a href="https://github.com/FlamingWater35/VideOCR-Recreated/releases">
      <img src="https://img.shields.io/badge/Download-Latest%20Release-2ea44f?style=for-the-badge&logo=github" alt="Download Latest Release">
    </a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9%20|%203.12-blue" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Windows%20|%20Linux-lightgrey" alt="Platform">
    <img src="https://img.shields.io/badge/GUI-PySide6-41CD52" alt="PySide6 GUI">
    <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
  </p>
</p>

---

## ✨ About & Key Features

VideOCR Recreated extracts hardcoded/burned-in subtitles from videos and exports them as `.srt` (or `.ass` with label detection) files. It builds upon previous forks to deliver a polished, feature-rich experience:

- **Modern PySide6 GUI**: A complete rewrite featuring a warm-obsidian & gold dark theme, native HiDPI support, resizable layouts, and persistent settings.
- **Label Detection**: Detects text *outside* the subtitle crop area (e.g., character/place names) and exports positioned `.ass` subtitle files with auto-fixing.
- **AMD DirectML Support**: Experimental hybrid GPU acceleration for Windows (DirectML text detection + CPU recognition fallback for maximum stability).
- **ONNX Runtime DirectML**: Experimental unified RapidOCR backend with PP-OCRv6 models for higher accuracy.
- **Refined FFmpeg Path**: Uses `-fps_mode vfr` for seamless compatibility with newer FFmpeg versions.
- **Streamlined Experience**: Removed GUI update polling and benchmarking buttons (CLI benchmark tools remain available for advanced users).

---

## ⚙️ Quick Setup

### 📦 Pre-built Releases (Recommended)

Get started instantly by downloading the latest installer or portable folder from the [Releases page](https://github.com/FlamingWater35/VideOCR-Recreated/releases).

- **Windows**: Run the setup installer or unzip the portable folder to your desired location.
- **Linux**: Download the tarball, unzip, and optionally run `./install_videocr.sh` to create a desktop shortcut.

### 💻 Development Setup (From Source)

*Requires **Python 3.12** (Python 3.13 is not recommended due to `torch-directml` wheel compatibility).*

**Windows (AMD DirectML):**

```bat
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
python VideOCR_qt.py
```

**Windows / Linux (NVIDIA CUDA):**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
# Build CUDA helpers: python build.py --target gpu-cuda12.9
python VideOCR_qt.py
```

> **Note:** NVIDIA users should use the normal CUDA builds. The DirectML path is strictly intended for AMD GPUs on Windows.

### 🔨 Building Standalone Executables

To compile portable releases (requires C++ Build Tools on Windows and 7-Zip):

```bash
git clone https://github.com/FlamingWater35/VideOCR-Recreated.git
cd VideOCR-Recreated
pip install . --group all
# Build targets: cpu, gpu-cuda11.8, gpu-cuda12.9, gpu-directml, or all
python build.py --target gpu-directml --clean true
```

---

## 🚀 Usage

### GUI Usage

1. Import a video and seek using the timeline or arrow keys.
2. Draw a crop box over the subtitle area (click and drag). *A tight crop is faster and more accurate than full-frame OCR.*
3. Click **Run** to start extraction.

> **Recommended 1080p Anime Settings:** `Frames to Skip: 2`, `OCR Max Width: 720`, `SSIM Threshold: 92`, `Confidence: 65–75`.

### CLI Usage

Run `videocr-cli.exe -h` (or `python CLI\videocr_cli.py -h`) for a full list of parameters.

**Windows Executable Example:**

```bat
.\videocr-cli.exe --video_path "video.mp4" --output "output.srt" --lang en --use_gpu true
```

**AMD DirectML Source Example:**

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
python CLI\videocr_cli.py --video_path "video.mp4" --output "output.srt" --ocr_engine easyocr_directml --lang en --use_gpu true --use_fullframe false
```

---

## ⚡ Performance & Tuning

| Mode | Best For | Notes |
| --- | --- | --- |
| `paddleocr` (CPU) | Compatibility | Fully local, but can be slow. |
| `paddleocr` (CUDA) | NVIDIA GPUs | Fastest official local GPU path. |
| `google_lens` | Accuracy | Requires internet; hybrid cloud recognition. |
| `easyocr_directml` | AMD Windows GPUs | Experimental hybrid DirectML mode. |

### AMD DirectML Tuning Tips

- **Expected GPU Usage**: 10–40% is normal. The pipeline bursts: CPU filters/stitches frames → GPU detects text → CPU recognizes/merges.
- **Grid Size**: Larger grids reduce per-image overhead. Try `DirectML Grid Max Width/Height: 2400` (Balanced) or `4096` (Max) for high-end GPUs.
- **Recognition Mode**: Keep on `Stable` (DirectML Detection + CPU Recognition) to avoid EasyOCR LSTM crashes. Use `Auto` or `Experimental` only for testing.
- **Discrete GPU Selection**: If workloads land on your iGPU, set the environment variable `VIDEOCR_DIRECTML_DEVICE_INDEX=1`, or select `GPU 1` in the GUI Advanced Settings (this preference is saved automatically).

---

## 🛠️ Troubleshooting

- **`No module named av` / `PySide6`**: Ensure you ran `pip install -e ".[directml]"` inside your activated virtual environment.
- **`No module named sympy.core`**: Run `pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"`.
- **DirectML using Integrated GPU**: Force Windows *Graphics Settings* to "High performance" for `.venv\Scripts\python.exe`, or set the device index as noted above.
- **ONNX DirectML `DmlExecutionProvider` not found**: Uninstall `onnxruntime` and install `onnxruntime-directml` instead. They cannot coexist in the same environment.
- **EasyOCR DirectML LSTM Crash**: This is a known operator compatibility issue. Ensure Recognition Mode is set to `Stable` to utilize the CPU fallback safely.

---

## 🙏 Credits

This project is a **fork of a fork**, building upon the excellent work of:

1. **[VideOCR](https://github.com/timminator/VideOCR)** by `timminator` (The base subtitle-extraction project).
2. **[VideOCR-AMD-DirectML](https://github.com/BaseCrunch/VideOCR-AMD-DirectML)** by `BaseCrunch` (Added the AMD DirectML backend, GPU selection, performance presets, and D3D11VA frame-scan).

This recreation adds the PySide6 GUI, Label Detection, ONNX DirectML support, and further refinements for a modern user experience.

```

### Key Improvements Made:
1. **Badges & Download Button**: Added a prominent, styled "Download Latest Release" button pointing to the releases page, alongside standard tech-stack badges.
2. **Consolidated Lineage**: Merged the repeated "fork of a fork" explanations from the header, "About", and "Credits" into a single, clean acknowledgment.
3. **Summarized CLI Parameters**: Replaced the massive, verbose list of every single CLI flag with a concise summary and a recommendation to use `-h`, keeping only the most critical examples.
4. **Streamlined Setup**: Grouped setup instructions logically (Pre-built vs. Source) and removed redundant step-by-step explanations that are obvious to developers.
5. **Condensed Troubleshooting**: Reduced the troubleshooting section to the most common, high-impact fixes, removing overly verbose error outputs.
6. **Improved Readability**: Used clear headings, emojis for visual scanning, and compact tables for performance tuning.
