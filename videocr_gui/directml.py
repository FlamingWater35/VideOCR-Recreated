"""DirectML adapter detection and display<->index conversion (ported from legacy GUI)."""

from __future__ import annotations

import re
import subprocess
import sys

from .constants import DIRECTML_AUTO_OPTION

DIRECTML_GPU_OPTION_PATTERN = re.compile(r"^GPU\s+(\d+)\s*:\s*(.*)$")


def _powershell_video_controller_names() -> list[str]:
    """Return Windows video controller names for the DirectML GPU dropdown."""
    if sys.platform != "win32":
        return []
    # Prevent the child console process (powershell/wmic) from flashing a
    # console window when launched from the GUI-subsystem executable.
    creationflags = subprocess.CREATE_NO_WINDOW
    commands = [
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ],
        ["wmic", "path", "win32_VideoController", "get", "name"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=creationflags,
            )
        except Exception:
            continue
        lines = []
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line or line.lower() == "name":
                continue
            clean = " ".join(line.split())
            if clean and clean not in lines:
                lines.append(clean)
        if lines:
            return lines
    return []


def _torch_directml_adapter_count() -> int | None:
    """Return the adapter count exposed by torch-directml, if available."""
    if sys.platform != "win32":
        return None
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None
    for name in ("device_count", "get_device_count"):
        fn = getattr(torch_directml, name, None)
        if callable(fn):
            try:
                count = int(fn())
                if count >= 0:
                    return count
            except Exception:
                pass
    return None


def _torch_directml_adapter_name(index: int) -> str | None:
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None
    for name in ("device_name", "get_device_name"):
        fn = getattr(torch_directml, name, None)
        if callable(fn):
            try:
                value = str(fn(index)).strip()
                if value:
                    return value
            except Exception:
                pass
    return None


def get_adapter_options() -> list[str]:
    """Build display options for the DirectML GPU selector."""
    options = [DIRECTML_AUTO_OPTION]
    gpu_names = _powershell_video_controller_names()
    dml_count = _torch_directml_adapter_count()

    if dml_count is not None and dml_count > 0:
        count = dml_count
    elif gpu_names:
        count = len(gpu_names)
    else:
        count = 2 if sys.platform == "win32" else 0

    for idx in range(count):
        name = _torch_directml_adapter_name(idx)
        if not name and idx < len(gpu_names):
            name = gpu_names[idx]
        if not name:
            name = "Unknown DirectML adapter"
        options.append(f"GPU {idx}: {name}")
    return options


def option_to_index(option: object) -> str:
    """Convert a DirectML combo display value into a CLI adapter index."""
    text = str(option or "").strip()
    if not text or text == DIRECTML_AUTO_OPTION:
        return ""
    match = DIRECTML_GPU_OPTION_PATTERN.match(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return ""


def index_to_option(options: list[str], selected_index: object) -> str:
    """Pick a DirectML combo display value from a stored numeric index."""
    selected = str(selected_index or "").strip()
    if selected:
        prefix = f"GPU {selected}:"
        for option in options:
            if option.startswith(prefix):
                return option
        return f"GPU {selected}: Unknown DirectML adapter"
    return DIRECTML_AUTO_OPTION
