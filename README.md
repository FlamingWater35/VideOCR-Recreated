<p align="center">
<img src="Pictures/VideOCR.png" alt="VideOCR Icon" width="128">
  <h1 align="center">VideOCR Recreated</h1>
  <p align="center">
    Extract hardcoded subtitles from videos with a modern PySide6 GUI, 200+ languages, and experimental AMD DirectML GPU support.
    <br />
  </p>
</p>

<br>

## ℹ About

VideOCR Recreated extracts hardcoded / burned-in subtitles from videos and exports them as `.srt` subtitle files.

This project is a recreation of the original VideOCR GUI in **PySide6 (Qt)** — a modern, easy-to-use interface that replaces the old PySimpleGUI application. It keeps the core subtitle-extraction features while adding an experimental **AMD GPU acceleration path for Windows** using:

- **DirectML**
- **torch-directml**
- **EasyOCR**
- **Hybrid OCR mode**

Supported OCR engines:

- Local OCR with **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
- Hybrid cloud recognition with **Google Lens**
- **EasyOCR DirectML (AMD GPU)** — experimental Windows AMD path
- **ONNX Runtime DirectML (AMD GPU Experimental)** — experimental

### What changed in the recreation

- GUI rewritten from scratch in **PySide6** (modern dark theme, resizable layout, native HiDPI support).
- **Update checking was removed** — the GUI no longer polls GitHub for new versions.
- **Benchmarking and the "Apply Tested AMD Preset" button were removed** from the GUI (the CLI benchmark flags and `tools/benchmark_*.py` helpers remain available for advanced users).
- The **DirectML GPU selection is now saved to the config file** and restored across sessions; it **defaults to GPU 0** (the first DirectML adapter).
- The FFmpeg D3D11VA path uses `-fps_mode vfr` instead of the removed `-vsync 0` option so it works with newer FFmpeg versions.

## AMD DirectML Backend Status

Tested on:

- **Windows**
- **AMD Radeon RX 7900 XTX**
- **Python 3.12**
- **torch-directml**
- **EasyOCR 1.7.2**

Current AMD backend:

| Stage | Device |
|---|---|
| Video decoding / frame filtering | CPU |
| Image preprocessing / stitching | CPU |
| EasyOCR text detection | AMD GPU through DirectML |
| EasyOCR text recognition | CPU fallback |
| Subtitle merging / SRT generation | CPU |

The recognition stage currently stays on CPU because EasyOCR's LSTM/CRNN recognizer can hit DirectML operator compatibility issues, such as:

```text
aten::_thnn_fused_lstm_cell
```

The hybrid mode is intentional. It avoids the crash while still using the AMD GPU for the text-detection part of the OCR pipeline.

## Important Notes

- The AMD DirectML path is **experimental**.
- DirectML mode is currently intended for **Windows AMD GPUs**.
- NVIDIA users should still use the normal CUDA builds.
- CPU mode still works.
- Google Lens mode still works where available.
- EasyOCR may download model files the first time it runs.
- Python **3.12** is recommended.
- Python **3.13** is not recommended because `torch-directml` may not provide compatible wheels for it.

## Setup

### Windows CPU / CUDA / Normal Use

You can either install VideOCR with the setup installer or download a folder containing the executable and required files, then unzip it to your desired location.

### Windows AMD DirectML Development Setup

Use this setup if you want to run the AMD DirectML fork directly from source.

Open CMD in the repository folder and run:

```bat
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

If your PC has both integrated AMD graphics and a discrete AMD GPU, force the DirectML adapter.

For RX 7900 XTX systems where the discrete GPU is adapter index `1`:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

Then test DirectML:

```bat
python tools\test_directml.py
python tools\diagnose_easyocr_directml.py
```

Expected result:

```text
DirectML tensor test: 3.0
DirectML device: privateuseone:1
Reader OK
All DirectML diagnostics passed.
```

Then start the GUI:

```bat
python VideOCR_qt.py
```

You can also use the helper:

```bat
run_gui_directml_dev.bat
```

This helper sets:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

before launching the GUI.

> **DirectML GPU persistence:** in the GUI, the DirectML GPU dropdown (Advanced Settings) is saved to `videocr_gui_config.ini` and restored on the next launch. The default is **GPU 0**. If your discrete Radeon card is adapter `1`, select `GPU 1` once — the GUI will remember it.

### Windows Graphics Preference

If Windows still sends DirectML workloads to the integrated GPU, force Python to use the high-performance GPU:

```text
Windows Settings
→ System
→ Display
→ Graphics
→ Add desktop app
→ Select:
  .venv\Scripts\python.exe
→ Options
→ High performance
→ AMD Radeon RX 7900 XTX
```

### Linux

Download the tarball archive from the releases page and unzip it to your desired location.

Optionally, you can add VideOCR to your app menus. Open a terminal where you unpacked the archive and run:

```bash
./install_videocr.sh
```

This creates a shortcut for VideOCR.

You can remove it with:

```bash
./uninstall_videocr.sh
```

### Docker

The VideOCR CLI can also be run within a Docker container.

#### Requirements

- **[Docker](https://docs.docker.com/get-docker/)** installed on your system.
- **For CUDA GPU acceleration:** An NVIDIA GPU with the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed on your host machine.

> AMD DirectML mode is Windows/DirectML-based and is not currently supported through the Docker flow.

#### Option A: Download from GitHub Container Registry

Pre-built images are automatically generated and hosted on GitHub.

CPU version:

```bash
docker pull ghcr.io/timminator/videocr-cli-cpu:latest
```

GPU version, CUDA 11.8 / NVIDIA 10 Series graphics cards:

```bash
docker pull ghcr.io/timminator/videocr-cli-gpu-cuda11.8:latest
```

GPU version, CUDA 12.9 / NVIDIA 16 - 50 Series graphics cards:

```bash
docker pull ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest
```

#### Option B: Build Locally

Clone the repository and use the provided Dockerfile. You can specify the hardware target with `BUILD_TARGET`.

Supported Docker build targets:

```text
cpu
gpu-cuda11.8
gpu-cuda12.9
```

Example CUDA 12.9 GPU build:

```bash
docker build --build-arg BUILD_TARGET=gpu-cuda12.9 -t videocr-cli-gpu:latest .
```

Example CPU build:

```bash
docker build --build-arg BUILD_TARGET=cpu -t videocr-cli-cpu:latest .
```

## GUI Usage

Import a video and seek through the video using the timeline. You can also use the left and right arrow keys.

Draw a crop box over the subtitle area using click and drag. After selecting the subtitle area, start subtitle extraction with the **Run** button.

For AMD DirectML mode, recommended GUI settings are:

```text
OCR Engine: EasyOCR DirectML / AMD GPU
GPU Usage: checked
Full Frame: unchecked
Crop Area: subtitle area only
Language: English, or your subtitle language
```

Recommended 1080p anime episode settings:

```text
Frames to Skip: 2
OCR Image Max Width: 720
SSIM Threshold: 92
Confidence Threshold: 65–75
Max Merge Gap: 0.1–0.3
```

If subtitles are missed, increase accuracy:

```text
Frames to Skip: 1
OCR Image Max Width: 960
```

If processing is too slow, increase speed:

```text
Frames to Skip: 3
OCR Image Max Width: 720
```

## CLI Usage

There is also a CLI version available. Open a terminal in the VideOCR folder and run:

### Windows

```bat
.\videocr-cli.exe -h
```

When running from source:

```bat
python CLI\videocr_cli.py -h
```

### Linux

```bash
./videocr-cli.bin -h
```

### Example Usage: Windows Executable

```bat
.\videocr-cli.exe --video_path "Path\to\your\video\example.mp4" --output "Path\to\your\desired\subtitle\location\example.srt" --lang en --time_start "18:40" --use_gpu true
```

### Example Usage: AMD DirectML Source Mode

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1

python CLI\videocr_cli.py ^
  --video_path "C:\Path\To\video.mp4" ^
  --output "C:\Path\To\output.en.srt" ^
  --ocr_engine easyocr_directml ^
  --lang en ^
  --use_gpu true ^
  --use_fullframe false ^
  --crop_x 36 ^
  --crop_y 793 ^
  --crop_width 1862 ^
  --crop_height 273 ^
  --frames_to_skip 2 ^
  --ssim_threshold 92 ^
  --ocr_image_max_width 720
```

### Example Usage: Docker

When running the Docker container, use Docker volumes with `-v` to mount your local video folder into the container's `/data` directory.

GPU example:

```bash
docker run --rm -it --gpus all \
-v /path/to/your/local/videos:/data \
ghcr.io/timminator/videocr-cli-gpu-cuda12.9:latest \
--video_path /data/my_video.mp4 \
--output /data/my_subtitle.srt \
--use_gpu true
```

CPU example:

```bash
docker run --rm -it \
-v /path/to/your/local/videos:/data \
ghcr.io/timminator/videocr-cli-cpu:latest \
--video_path /data/my_video.mp4 \
--output /data/my_subtitle.srt
```

Any CLI parameters listed below can be appended to the Docker command.

## Performance

Local OCR processing can be slow on CPU. Using a GPU is recommended when available.

The fork provides several practical performance paths:

| Mode | Best For | Notes |
|---|---|---|
| `paddleocr` CPU | Compatibility | Fully local but can be slow |
| `paddleocr` CUDA | NVIDIA GPUs | Fastest official local GPU path |
| `google_lens` | Accuracy / cloud recognition | Requires internet |
| `easyocr_directml` | AMD Windows GPUs | Experimental hybrid DirectML mode |

### AMD DirectML Frame Scan Modes

This fork includes experimental Step 1 frame-scan modes intended for AMD GPUs on Windows:

```text
CPU SSIM (compatible)
AMD DirectML SSIM (experimental)
AMD FFmpeg D3D11VA Decode + DirectML SSIM (prototype)
```

The FFmpeg D3D11VA prototype asks FFmpeg to use D3D11VA hardware decode, then feeds the cropped subtitle area into the EasyOCR DirectML pipeline. It is best tested with a single subtitle crop area and FFmpeg available on PATH. For reliable use, keep DirectML Recognition Mode on `Stable Hybrid`; use `AMD Max Auto` only for experiments.

A quick diagnostic is available:

```bash
python tools/benchmark_amd_decode.py "C:\path\to\video.mp4" --crop 1920:287:0:793 --scale 720:-2 --frames-to-skip 1 --seconds 60
```

### AMD DirectML Performance Notes

GPU usage may appear lower than gaming or rendering workloads. This is normal.

The AMD DirectML backend processes OCR in bursts:

1. CPU reads and filters frames.
2. CPU creates stitched OCR image grids.
3. RX 7900 XTX performs text detection through DirectML.
4. CPU performs text recognition for compatibility.
5. CPU merges subtitle lines and writes the `.srt`.

Because only part of the OCR pipeline runs on the GPU, GPU usage around `10–40%` can be normal. This does not mean the GPU is unused.

For RX-class GPUs, the GUI exposes DirectML grid tuning and a DirectML GPU dropdown. Larger stitched grids reduce per-image overhead and can keep the GPU busier during the detection pass. Good starting values are:

```text
DirectML GPU: GPU 1: AMD Radeon RX 7900 XTX
DirectML Grid Max Width: 2400
DirectML Grid Max Height: 2400
Frames to Skip: 2 for speed, 1 for accuracy
OCR Image Max Width: 720 for speed, 960 for accuracy
```

The DirectML performance preset controls the grid target automatically:

| Preset | Grid target | Intended use |
|---|---:|---|
| `compatibility` | 1600x1600 | Older/lower-VRAM GPUs, stability first |
| `balanced` | 2400x2400 | Recommended default |
| `max` | 4096x4096 | Larger batches to feed high-end AMD GPUs harder |
| `manual` | Uses the grid width/height fields | Manual tuning |

DirectML Recognition Mode behavior:

| Mode | Detection | Recognition | Notes |
|---|---|---|---|
| `stable` | DirectML GPU | CPU | Safest; avoids EasyOCR LSTM DirectML crash |
| `auto` | DirectML GPU | Try DirectML GPU, then CPU fallback | Best "max AMD" test mode |
| `experimental` | DirectML GPU | Try DirectML GPU | Still falls back for known LSTM compatibility failures |

## Tips

When cropping, leave a bit of buffer space above and below the subtitle text to improve detection, but do not make the crop box too large.

A tight crop box around the subtitle area is usually much faster and more accurate than full-frame OCR.

### Quick Configuration Cheatsheet

| Option | More Speed | More Accuracy | Notes |
|---|---|---|---|
| Input video quality | Use lower quality | Use higher quality | Cropping reduces the performance cost of high resolution |
| `frames_to_skip` | Higher number | Lower number | For perfectly accurate timestamps, set this to `0` |
| `ssim_threshold` | Lower threshold | Higher threshold | Lower values reduce the number of images sent to OCR |
| `ocr_image_max_width` | Lower value | Higher value | Lower values are faster; higher values help small text |
| `directml_grid_max_width` / `directml_grid_max_height` | Larger values can reduce overhead | Smaller values use less memory | AMD DirectML mode only |
| Crop area | Tighter crop | Slight buffer around text | Avoid full-frame OCR unless needed |

## Command Line Parameters

### `video_path`

Path to the video where subtitles should be extracted from.

### `output`

Path where the `.srt` subtitle file should be stored.

### `ocr_engine`

Select the OCR engine to use for text detection and recognition.

Valid values include:

```text
paddleocr
google_lens
easyocr_directml
onnx_directml
```

`paddleocr` uses local processing for both text detection and recognition.

`google_lens` uses hybrid processing where local detection is combined with Google Lens recognition. This mode requires an active internet connection.

`easyocr_directml` uses the AMD DirectML fork backend. In the current stable hybrid mode, EasyOCR text detection runs on DirectML / AMD GPU and recognition falls back to CPU.

`onnx_directml` is an experimental ONNX Runtime DirectML backend. If the ONNX stack is unavailable it falls back to the EasyOCR DirectML Hybrid path for that run.

### `lang`

Language of the subtitles.

Supported languages depend on the selected OCR engine.

- For `paddleocr`: see the PaddleOCR documentation.
- For `google_lens`: see the Google Lens / Vision language documentation.
- For `easyocr_directml`: use EasyOCR-supported language codes, such as `en`.

### `subtitle_position`

Specifies the alignment of subtitles in the video and allows for better text recognition.

### `conf_threshold`

Confidence threshold for word predictions. Words with lower confidence than this value are discarded.

Default value:

```text
75
```

Make it lower if you get too few words in each line.

Make it higher if there are too many extra words in each line.

### `sim_threshold`

Similarity threshold for subtitle lines. Subtitle lines with larger Levenshtein ratios than this threshold are merged together.

Default value:

```text
80
```

Make it lower if there are too many duplicated subtitle lines.

Make it higher if too few subtitle lines are being generated.

### `ssim_threshold`

If the SSIM between consecutive frames exceeds this threshold, the frame is considered similar and discarded during initial frame filtering in Step 1.

A lower value can greatly reduce the number of images OCR needs to process.

On tight subtitle crop boxes, good results may be possible around:

```text
85–92
```

### `post_processing`

Adds a post-processing step for detected text. This can analyze detected text for missing spaces and insert them automatically.

Currently available for:

```text
English
Spanish
Portuguese
German
Italian
French
```

### `max_merge_gap`

Maximum allowed time gap in seconds between two subtitles to be considered for merging if they are similar.

Default value:

```text
0.09
```

Increase this if the output SRT contains repeated subtitle lines that should have been merged.

### `time_start` and `time_end`

Extract subtitles from only part of the video.

Subtitle timestamps are still calculated according to the full video timeline.

### `use_fullframe`

By default, the specified crop area is used for OCR. If no crop is specified, the bottom third of the frame is used.

Set this to `True` to OCR the entire frame.

### `crop_x`, `crop_y`, `crop_width`, `crop_height`

Specify the bounding area in pixels used for OCR.

![Crop example](Pictures/crop_example.png)

### `crop_x2`, `crop_y2`, `crop_width2`, `crop_height2`

Specify a second bounding area in pixels for OCR when needed.

### `subtitle_alignment` and `subtitle_alignment2`

Subtitle alignment values for ASS / Advanced SubStation Alpha positioning.

Valid values:

```text
bottom-left
bottom-center
bottom-right
middle-left
middle-center
middle-right
top-left
top-center
top-right
```

### `ocr_image_max_width`

Downscales the cropped image frame so its width does not exceed this value before passing it to OCR.

Lower values improve speed.

Higher values may improve accuracy.

### `use_gpu`

Set to `True` to perform OCR with GPU acceleration where supported.

For AMD DirectML mode, the GUI saves your DirectML GPU selection in the config file (default GPU 0). From the CLI you can also set the adapter directly:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

if your discrete AMD GPU is adapter index `1`.

### `use_angle_cls`

Set to `True` if classification should be enabled.

For PaddleOCR only.

### `brightness_threshold`

If set, pixels whose brightness is less than the threshold are blackened out.

Valid brightness values range from:

```text
0
```

to:

```text
255
```

This can help improve accuracy when OCR is performed on videos with white subtitles.

### `frames_to_skip`

Number of frames to skip before sampling a frame for OCR.

For 1080p anime episodes, a good starting point is:

```text
2
```

Increase for speed.

Decrease for more accurate timing.

### `min_subtitle_duration`

Subtitles shorter than this threshold are omitted from the final subtitle file.

### `normalize_to_simplified_chinese`

Traditional Chinese characters are converted to Simplified Chinese before processing.

Only active for Chinese & English.

### `use_server_model`

Enables server models for OCR.

This may improve detection at the cost of more processing power.

Primarily for PaddleOCR GPU usage.

### `directml_device_index`

Selects the DirectML adapter index used by `easyocr_directml` / `onnx_directml`.

```bat
--directml_device_index 0
```

usually selects the first DirectML adapter. `1` may select the discrete GPU on systems with integrated graphics plus an RX 7900 XTX.

### `directml_performance_preset`

```bat
--directml_performance_preset compatibility
--directml_performance_preset balanced
--directml_performance_preset max
--directml_performance_preset manual
```

### `directml_recognition_mode`

```bat
--directml_recognition_mode stable
--directml_recognition_mode auto
--directml_recognition_mode experimental
```

### `directml_frame_scan_mode`

```bat
--directml_frame_scan_mode cpu_ssim
--directml_frame_scan_mode directml_ssim
--directml_frame_scan_mode ffmpeg_d3d11va
```

### `onnx_directml_tuning`

```bat
--onnx_directml_tuning low_vram
--onnx_directml_tuning balanced
--onnx_directml_tuning max
--onnx_directml_tuning manual
```

### `directml_grid_max_width` / `directml_grid_max_height`

Maximum stitched OCR grid size for the manual DirectML mode.

### Benchmark compare (CLI-only)

The CLI retains optional benchmarking helpers that were removed from the GUI:

```bat
--benchmark_compare_engine false
--benchmark_compare_sample_grids 3
```

These run a small EasyOCR-vs-ONNX sample after the main run and print `[BenchCompare]` lines. Leave them off for normal processing. Standalone tools:

```bash
python tools/benchmark_amd_presets.py "C:\path\to\video.mp4" --crop 0,793,1920,287 --seconds 180 --device-index 1 --keep-logs
python tools/benchmark_ocr_compare.py "C:\path\to\video.mp4" --crop 0:793:1920:287 --frames-to-skip 1 --onnx-directml-tuning balanced
```

## AMD DirectML Environment Variables

### `VIDEOCR_DIRECTML_DEVICE_INDEX`

Selects the DirectML adapter index.

Examples:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=0
```

usually selects the first DirectML adapter.

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

may select the discrete GPU on systems with integrated graphics plus an RX 7900 XTX.

### `VIDEOCR_DIRECTML_GRID_MAX_WIDTH` / `VIDEOCR_DIRECTML_GRID_MAX_HEIGHT`

Optional helper values used by the development launcher. The GUI and CLI settings are preferred.

Recommended RX 7900 XTX starting point:

```bat
set VIDEOCR_DIRECTML_GRID_MAX_WIDTH=2400
set VIDEOCR_DIRECTML_GRID_MAX_HEIGHT=2400
```

### `VIDEOCR_EASYOCR_RECOGNITION_DEVICE`

Optional advanced setting.

Supported values:

```text
cpu
directml
```

Default:

```text
cpu
```

Keep this on `cpu` unless you are testing experimental full DirectML recognition. DirectML recognition may fail on some EasyOCR models due to unsupported LSTM-related operators.

## Build and Compile Instructions

### Requirements

- Python 3.9 or higher
- Python 3.12 recommended for AMD DirectML mode
- **PySide6** (installed via `pip install PySide6` or the project dependencies)

Windows:

- C++ Build Tools, for example Visual Studio with **Desktop development with C++**
- 7-Zip available in PATH

Linux:

- 7-Zip
- Working dbus installation is recommended

### Clone Repository

```bash
git clone https://github.com/BaseCrunch/VideOCR.git
cd VideOCR
```

If you rename this fork, use your new repository URL instead.

### Install Dependencies

Standard install:

```bash
python -m pip install --upgrade pip
pip install . --group all
```

AMD DirectML source install:

```bat
py -3.12 -m venv .venv --upgrade-deps
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[directml]"
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

### Build

CPU build:

```bash
python build.py --target cpu
```

CUDA 11.8 build:

```bash
python build.py --target gpu-cuda11.8
```

CUDA 12.9 build:

```bash
python build.py --target gpu-cuda12.9
```

AMD DirectML build:

```bash
python build.py --target gpu-directml
```

More information:

```bash
python build.py -h
```

## Troubleshooting

### `No suitable Python runtime found`

Install Python 3.12:

```bat
winget install -e --id Python.Python.3.12
```

Then open a new CMD and verify:

```bat
py -0p
```

### `No module named pip`

Repair pip inside the active venv:

```bat
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
```

### `No module named av`

Install project dependencies inside the active venv:

```bat
python -m pip install -e ".[directml]"
```

### `No module named PySide6`

Install PySide6:

```bat
python -m pip install PySide6
```

### `No module named sympy.core`

Repair SymPy:

```bat
python -m pip install --force-reinstall --no-cache-dir "sympy==1.13.3" "mpmath==1.3.0"
```

### DirectML uses integrated GPU instead of RX 7900 XTX

Set the adapter index:

```bat
set VIDEOCR_DIRECTML_DEVICE_INDEX=1
```

Then run:

```bat
python tools\test_directml.py
python tools\diagnose_easyocr_directml.py
python VideOCR_qt.py
```

In the GUI, open **Advanced Settings → DirectML GPU** and select your discrete card (e.g. `GPU 1: AMD Radeon RX 7900 XTX`). The selection is saved and restored on the next launch.

Also set Windows graphics preference for:

```text
.venv\Scripts\python.exe
```

to **High performance**.

### EasyOCR DirectML fails on LSTM / CRNN recognition

Keep hybrid mode enabled.

Use:

```text
Detection: DirectML
Recognition: CPU
```

Do not force recognition to DirectML unless testing.

### ONNX DirectML: `DmlExecutionProvider` not available / not found

When running the **ONNX Runtime DirectML (AMD GPU Experimental)** engine you may see output like:

```text
ONNX DirectML status: ONNX Runtime is installed, but DmlExecutionProvider is not available. Providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
ONNX DirectML OCR is not ready on this install. Falling back to EasyOCR DirectML Hybrid for this run.
```

This means the ONNX Runtime package is installed but its **DirectML execution provider** is missing, so the ONNX engine automatically falls back to EasyOCR DirectML Hybrid (the run still completes). To actually use the ONNX DirectML engine, install the DirectML-enabled ONNX Runtime build:

```bat
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-directml
```

Verify the DirectML provider is now registered:

```bat
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output includes `DmlExecutionProvider`, for example:

```text
['DmlExecutionProvider', 'CPUExecutionProvider']
```

If it is still missing after installing `onnxruntime-directml`, try forcing a reinstall of the exact package:

```bat
python -m pip install --force-reinstall --no-cache-dir onnxruntime-directml
```

Notes:

- The `onnxruntime-directml` wheel is Windows-only and requires a DirectX 12 capable GPU.
- `onnxruntime` (plain CPU) and `onnxruntime-directml` cannot be installed at the same time — installing one removes the other.
- If the DirectML provider still does not appear, your Python environment may not match the wheel's platform (use a 64-bit Python 3.12 venv, as recommended for this project).

## Credits

This project is based on the original **VideOCR** project by `timminator`, with a recreated PySide6 GUI and an AMD DirectML backend.

Original project:

```text
https://github.com/timminator/VideOCR
```

Fork changes include:

- PySide6 GUI recreation ("VideOCR Recreated")
- `easyocr_directml` backend
- DirectML adapter selection (persisted in the config, default GPU 0)
- DirectML diagnostics and development helper scripts
- Hybrid EasyOCR mode for stable AMD GPU usage
- Newer-FFmpeg compatible frame-scan mode (`-fps_mode vfr`)
