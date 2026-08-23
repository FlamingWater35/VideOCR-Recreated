from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import requests  # type: ignore

from _version import __version__

# --- Configuration ---
APP_VERSION = __version__

# Directories left behind by previous builds (Nuitka .build/.dist caches and
# the packaged Releases output). --clean removes all of them.
BUILD_ARTIFACT_DIRS: list[str] = [
    "Releases",
    "VideOCR_qt.dist",
    "CLI/videocr_cli.dist",
    "VideOCR_qt.build",
    "CLI/videocr_cli.build",
]

SUPPORT_FILES_URLS: dict[str, str] = {
    "Windows": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR.PP-OCRv6.support.files.VideOCR.7z",
    "Linux": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR.PP-OCRv6.support.files.VideOCR.tar.xz",
}

PADDLE_URLS: dict[str, dict[str, str | list[str]]] = {
    "Windows": {
        "cpu": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-CPU-v{version}.7z",
        "gpu-cuda11.8": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-GPU-v{version}-CUDA-11.8.7z",
        "gpu-cuda12.9": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-GPU-v{version}-CUDA-12.9.7z",
        # DirectML uses EasyOCR for AMD GPU acceleration, but keeps the CPU
        # PaddleOCR helper so the original PaddleOCR/Google Lens modes still work.
        "gpu-directml": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-CPU-v{version}.7z",
    },
    "Linux": {
        "cpu": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-CPU-v{version}-Linux.7z",
        "gpu-cuda11.8": "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-GPU-v{version}-CUDA-11.8-Linux.7z",
        "gpu-cuda12.9": [
            "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-GPU-v{version}-CUDA-12.9-Linux.7z.001",
            "https://github.com/timminator/PaddleOCR-Standalone/releases/download/v{version}/PaddleOCR-GPU-v{version}-CUDA-12.9-Linux.7z.002",
        ],
    },
}

CHROME_LENS_URLS: dict[str, str] = {
    "Windows": "https://github.com/timminator/Chrome-Lens-OCR/releases/download/v{version}/Chrome-Lens-OCR-v{version}.7z",
    "Linux": "https://github.com/timminator/Chrome-Lens-OCR/releases/download/v{version}/Chrome-Lens-OCR-v{version}-Linux.7z",
}


# --- Helper Functions ---
def _github_headers() -> dict[str, str]:
    """Return headers for GitHub API requests, using GITHUB_TOKEN if available to avoid rate limits."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_latest_paddle_version() -> str:
    """Fetches the latest version tag for PaddleOCR."""
    url = "https://api.github.com/repos/timminator/PaddleOCR-Standalone/releases/latest"
    r = requests.get(url, timeout=15, headers=_github_headers())
    r.raise_for_status()
    tag = str(r.json()["tag_name"])
    return tag.lstrip("v")


def get_latest_chrome_lens_version() -> str:
    """Fetches the latest version tag for Chrome-Lens-OCR."""
    url = "https://api.github.com/repos/timminator/Chrome-Lens-OCR/releases/latest"
    r = requests.get(url, timeout=15, headers=_github_headers())
    r.raise_for_status()
    tag = str(r.json()["tag_name"])
    return tag.lstrip("v")


def print_header(message: str) -> None:
    """Prints a formatted header."""
    print("\n" + "=" * 60)
    print(f" {message}")
    print("=" * 60)


def clean_build_artifacts() -> None:
    """Removes residuals left behind by previous builds.

    Deletes the packaged Releases folder and the Nuitka .build/.dist caches
    (both the GUI's and the CLI's) so a fresh build starts from a clean slate.
    """
    print_header("Cleaning Previous Build Artifacts")
    for rel_path in BUILD_ARTIFACT_DIRS:
        path = Path(rel_path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {rel_path}")
            else:
                path.unlink()
                print(f"Removed file: {rel_path}")
        else:
            print(f"Nothing to clean: {rel_path} does not exist.")


def check_pyside6() -> None:
    """Checks if PySide6 is installed and importable."""
    print_header("Checking for PySide6 support...")
    try:
        import PySide6  # noqa: F401

        print("PySide6 support found.")
    except ImportError:
        print("ERROR: PySide6 is not installed or not available.")
        print("Please install it: python -m pip install PySide6")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: PySide6 found, but failed to initialize: {e}")
        sys.exit(1)


def check_dbus() -> None:
    """On Linux, checks if dbus is likely installed for better plyer support."""
    if sys.platform == "linux":
        print_header("Checking for D-Bus on Linux...")
        if shutil.which("dbus-daemon"):
            print("D-Bus daemon found. Plyer notifications should work well.")
        else:
            print(
                "WARNING: D-Bus daemon not found. Plyer notifications might not work correctly."
            )
            print(
                "On Debian/Ubuntu, you can install it with: sudo apt-get install dbus"
            )


def check_7zip() -> None:
    """Checks if 7-Zip is installed and available."""
    print_header("Checking for 7-Zip...")
    if not (shutil.which("7z") or shutil.which("7z.exe")):
        print(
            "ERROR: 7-Zip executable ('7z' or '7z.exe') not found in your system's PATH."
        )
        print("Please install 7-Zip and ensure it's added to your PATH.")
        print(" - Windows: https://www.7-zip.org/")
        print(" - Linux (Debian/Ubuntu): sudo apt-get install p7zip-full")
        sys.exit(1)
    print("7-Zip found.")


def clean_pyside6_qml_artifacts() -> None:
    """Removes CMake build artifacts shipped inside the PySide6 wheel.

    Recent PySide6 wheels ship 'objects-RelWithDebInfo' directories below
    'PySide6/qml' that contain ELF relocatable object files ('*.cpp.o').
    When Nuitka's PySide6 plugin bundles the QML directory, it tries to
    fix the rpath of every ELF file with patchelf, which fails with:

        FATAL: Error, call to 'patchelf' failed: ... -> b'patchelf: wrong ELF type'

    because patchelf can only handle shared libraries and executables.
    These object files are compile-time artifacts not needed at runtime,
    so it is safe to delete them before compilation.
    """
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return

    pyside6_init = getattr(PySide6, "__file__", None)
    if not pyside6_init:
        return
    pyside6_dir = Path(pyside6_init).resolve().parent
    removed = 0
    for directory in pyside6_dir.rglob("objects-*"):
        if directory.is_dir():
            print(f"Removing PySide6 build artifact: {directory}")
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    if removed:
        print(f"Removed {removed} PySide6 'objects-*' artifact folder(s).")
    else:
        print("No PySide6 'objects-*' artifact folders found.")


def run_command(command: list[str], cwd: str | Path | None = None) -> None:
    """Runs a command in the shell, streams its output, and exits if it fails."""
    try:
        print(
            f"\nRunning command: {' '.join(command)}" + (f" in '{cwd}'" if cwd else "")
        )
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Command '{command[0]}' not found. Is it in your PATH?")
        sys.exit(1)


def download_file(urls: str | list[str], dest_folder: str | Path) -> Path:
    """Downloads a file or a sequence of files from URLs into a destination folder."""
    if not isinstance(urls, list):
        urls = [urls]
    if not urls:
        raise ValueError("No URLs provided to download.")

    first_file_path = Path(dest_folder) / urls[0].split("/")[-1]
    for url in urls:
        local_filename = url.split("/")[-1]
        file_path = Path(dest_folder) / local_filename
        print(f"Downloading {local_filename}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded to {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to download {url}. Reason: {e}")
            sys.exit(1)
    return first_file_path


def extract_archive(file_path: Path, dest_folder: str | Path) -> None:
    """Extracts a .7z, multipart .7z, or .tar.xz archive."""
    print(f"Extracting {file_path.name}...")
    if file_path.suffix == ".7z" or str(file_path).endswith(".7z.001"):
        seven_zip_exe = shutil.which("7z") or shutil.which("7z.exe")
        if seven_zip_exe:
            command = [seven_zip_exe, "x", str(file_path), f"-o{dest_folder}", "-y"]
            run_command(command)
        else:
            print("ERROR: 7-Zip executable not found in PATH.")
            sys.exit(1)
    elif file_path.suffix == ".xz":
        try:
            with tarfile.open(file_path, "r:xz") as archive:
                try:
                    # Python >= 3.12 (and patched 3.8.17+/3.9.17+/3.10.12+/3.11.4+)
                    archive.extractall(path=dest_folder, filter="data")
                except TypeError:
                    archive.extractall(path=dest_folder)
        except Exception as e:
            print(f"ERROR: Failed to extract {file_path}. Reason: {e}")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported archive format: {file_path.suffix}")
    print(f"Extracted to {dest_folder}")


def sign_file(
    signtool_path: str | None, cert_name: str | None, file_to_sign: Path
) -> None:
    """Signs a file using signtool.exe on Windows."""
    if not signtool_path or sys.platform != "win32":
        return

    print(f"Signing {file_to_sign.name}...")
    if not Path(signtool_path).is_file():
        print(f"ERROR: Sign tool not found at '{signtool_path}'")
        sys.exit(1)

    command = [
        signtool_path,
        "sign",
        "/tr",
        "http://timestamp.digicert.com",
        "/td",
        "sha256",
        "/fd",
        "sha256",
    ]
    if cert_name:
        command.extend(["/n", cert_name])
    else:
        command.append("/a")
    command.append(str(file_to_sign))

    run_command(command)
    print(f"Successfully signed {file_to_sign.name}")


def create_final_archive(folder_path: Path, build_target: str) -> None:
    """Creates a compressed archive of the final build folder."""
    print_header(f"Creating final archive for {folder_path.name}")
    try:
        seven_zip_exe = shutil.which("7z") or shutil.which("7z.exe")
        if not seven_zip_exe:
            print("WARNING: 7-Zip not found, cannot create .7z archive. Skipping.")
            return

        archive_path = folder_path.parent / f"{folder_path.name}.7z"
        print(f"Creating {archive_path.name}...")

        is_linux_cuda12_split = (
            sys.platform == "linux" and build_target == "gpu-cuda12.9"
        )

        command = [
            seven_zip_exe,
            "a",
            "-t7z",
            "-mx=7",
            "-m0=lzma2",
            "-md=64m",
            "-mfb=64",
            "-ms=on",
        ]
        if is_linux_cuda12_split:
            print("Applying 1999MB volume splitting for Linux CUDA 12.9 build...")
            command.extend(["-v1999m"])
        command.extend([str(archive_path.name), str(folder_path.name)])

        run_command(command, cwd=str(folder_path.parent))
        print(f"Archive created successfully: {archive_path}")
    except Exception as e:
        print(f"ERROR: Failed to create archive. Reason: {e}")
        sys.exit(1)


def create_windows_installer(final_app_path: Path, args: argparse.Namespace) -> None:
    """Creates a Windows installer using Inno Setup by passing parameters to the compiler."""
    if sys.platform != "win32":
        return

    # 1. Check explicitly provided argument
    # 2. Check the default Inno Setup installation path
    # 3. Fallback to system PATH
    default_iscc = r"C:\Program Files\Inno Setup 7\ISCC.exe"
    iscc_exe = (
        args.iscc
        or (default_iscc if Path(default_iscc).is_file() else None)
        or shutil.which("iscc")
        or shutil.which("ISCC.exe")
    )

    if not iscc_exe or not Path(iscc_exe).is_file():
        print("\nWARNING: Inno Setup Compiler (iscc.exe) not found.")
        print("         Skipping installer creation.")
        print(
            "         To create an installer, install Inno Setup and add it to your PATH,"
        )
        print("         or provide the path to iscc.exe using the --iscc argument.")
        return

    display_target_name = final_app_path.name.replace("VideOCR-", "").replace(
        f"-v{APP_VERSION}", ""
    )
    print_header(f"Creating Windows Installer for {display_target_name}")

    script_path = Path("Installer/Windows/installer_template.iss")
    if not script_path.is_file():
        print(
            f"WARNING: Installer script not found at '{script_path}'. Skipping installer creation."
        )
        return

    releases_dir = final_app_path.parent
    output_filename = f"{final_app_path.name}-setup-x64"

    command = [
        iscc_exe,
        "/Qp",
        f"/DMyAppVersion={APP_VERSION}",
        f"/DSourceDir={str(final_app_path.resolve())}",
        f"/DOutputBaseFilename={output_filename}",
        f"/DOutputDir={str(releases_dir.resolve())}",
    ]

    if args.signtool:
        signtool_params = "sign /tr http://timestamp.digicert.com /td sha256 /fd sha256"
        if args.sign_cert_name:
            signtool_params += f" /n $q{args.sign_cert_name}$q $f"
        else:
            signtool_params += " /a $f"
        iscc_sign_param = f"/Ssigntool=$q{args.signtool}$q {signtool_params}"
        command.append(iscc_sign_param)
        command.append("/DUseSignTool")

    command.append(str(script_path))
    run_command(command)
    print("\nInstaller created successfully.")


# --- Main Build Logic ---
def package_target(
    build_target: str,
    args: argparse.Namespace,
    releases_dir: Path,
    base_gui_dist: Path | None,
    base_cli_dist: Path,
    paddle_version: str,
    chrome_lens_version: str,
) -> None:
    """Packages a single distribution for the specified target using pre-compiled files."""
    is_cli_only = args.cli_only.lower() == "true"
    is_gui_only = args.gui_only.lower() == "true"

    if build_target == "gpu-directml":
        display_target_name = "GPU-DirectML"
    elif "gpu" in build_target:
        display_target_name = build_target.replace("gpu-", "GPU-").replace(
            "cuda", "CUDA-"
        )
    else:
        display_target_name = build_target.upper()

    print_header(f"Packaging for Target: {display_target_name}")

    os_name = "Windows" if sys.platform == "win32" else "Linux"
    os_suffix = "-Linux" if os_name == "Linux" else ""

    if build_target == "gpu-directml" and os_name != "Windows":
        print(
            "ERROR: gpu-directml is Windows-only because torch-directml/DirectML is Windows-only."
        )
        sys.exit(1)

    # Create a temporary directory for this target's packaging process
    work_dir = releases_dir / f"work_{build_target}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    print(f"Creating temporary work directories in '{work_dir}'...")

    temp_cli_dist = work_dir / "cli_dist"
    shutil.copytree(base_cli_dist, temp_cli_dist)

    temp_gui_dist = None
    if not is_cli_only and base_gui_dist:
        temp_gui_dist = work_dir / "gui_dist"
        shutil.copytree(base_gui_dist, temp_gui_dist)

    # Download and Extract Dependencies into the temporary CLI folder
    print_header(f"Downloading Dependencies for {display_target_name} target")

    # Format URLs dynamically using the fetched versions
    raw_support_url = SUPPORT_FILES_URLS[os_name]
    support_archive_path = download_file(
        raw_support_url.format(version=paddle_version), temp_cli_dist
    )

    raw_paddle_urls = PADDLE_URLS[os_name][build_target]
    paddle_url: str | list[str]
    if isinstance(raw_paddle_urls, list):
        paddle_url = [u.format(version=paddle_version) for u in raw_paddle_urls]
    else:
        paddle_url = raw_paddle_urls.format(version=paddle_version)
    paddle_archive_path = download_file(paddle_url, temp_cli_dist)

    raw_chrome_lens_url = CHROME_LENS_URLS[os_name]
    chrome_lens_archive_path = download_file(
        raw_chrome_lens_url.format(version=chrome_lens_version), temp_cli_dist
    )

    # Extract all archives
    extract_archive(support_archive_path, temp_cli_dist)
    extract_archive(paddle_archive_path, temp_cli_dist)
    extract_archive(chrome_lens_archive_path, temp_cli_dist)

    print("Cleaning up downloaded archives...")
    os.remove(support_archive_path)
    os.remove(chrome_lens_archive_path)
    if isinstance(paddle_url, list):
        for url in paddle_url:
            filename = url.split("/")[-1]
            filepath = Path(temp_cli_dist) / filename
            if filepath.exists():
                os.remove(filepath)
    else:
        os.remove(paddle_archive_path)

    for file_path in Path(temp_cli_dist).rglob("*flash*"):
        if file_path.is_file():
            file_path.unlink(missing_ok=True)

    # Assemble Final Directory Structure
    print_header(f"Assembling Final Directory Structure for {display_target_name}")

    # Define final names
    release_tag = f"-{args.release_type}" if args.release_type else ""
    cuda_suffix = ""
    if build_target == "gpu-directml":
        base_target_name = "GPU"
        cuda_suffix = "-DirectML"
    elif "gpu" in build_target:
        base_target_name = "GPU"
        cuda_version = build_target.split("-")[-1]
        cuda_suffix = f"-{cuda_version.replace('cuda', 'CUDA-')}"
    else:
        base_target_name = build_target.upper()

    cli_final_name = f"videocr-cli-{base_target_name}-v{APP_VERSION}{cuda_suffix}{release_tag}{os_suffix}"
    final_cli_path = releases_dir / cli_final_name

    if not is_gui_only:
        print(f"Creating standalone CLI at '{final_cli_path}'")
        shutil.copytree(temp_cli_dist, final_cli_path)
    else:
        print(f"Skipping standalone CLI folder creation (--gui-only).")

    if not is_cli_only and temp_gui_dist:
        final_app_folder_name = f"VideOCR-{base_target_name}-v{APP_VERSION}{cuda_suffix}{release_tag}{os_suffix}"

        # Merge CLI into GUI
        print(f"Merging CLI files into GUI root for {final_app_folder_name}...")
        for item in temp_cli_dist.rglob("*"):
            relative_path = item.relative_to(temp_cli_dist)
            target_path = temp_gui_dist / relative_path
            if item.is_dir():
                target_path.mkdir(exist_ok=True)
            else:
                shutil.copy2(item, target_path)

        # Copy Linux installer scripts if applicable
        if os_name == "Linux":
            print("Copying Linux installer scripts...")
            installer_src = Path("Installer/Linux")
            for script_name in ["install_videocr.sh", "uninstall_videocr.sh"]:
                src_path = installer_src / script_name
                dest_path = temp_gui_dist / script_name
                if src_path.exists():
                    shutil.copy(src_path, dest_path)
                    os.chmod(dest_path, dest_path.stat().st_mode | stat.S_IEXEC)
                    print(f"Copied and set +x on {script_name}")
                else:
                    print(f"WARNING: Installer script not found at {src_path}")

        # Move final GUI folder to Releases
        final_app_path = releases_dir / final_app_folder_name
        print(f"Moving final application to '{final_app_path}'")
        shutil.move(str(temp_gui_dist), final_app_path)
        shutil.rmtree(work_dir)

        if (
            sys.platform == "win32"
            and args.windows_installer
            and args.windows_installer.lower() == "true"
        ):
            create_windows_installer(final_app_path, args)

        print_header("Preparing Portable Standalone Build")
        print("Injecting portable_mode.txt for standalone GUI archive...")
        portable_flag_gui = final_app_path / "portable_mode.txt"
        portable_flag_gui.touch()
    else:
        shutil.rmtree(work_dir)

    if args.archive and args.archive.lower() == "true":
        if not is_cli_only:
            create_final_archive(final_app_path, build_target)
        if not is_gui_only:
            create_final_archive(final_cli_path, build_target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VideOCR Build Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=["cpu", "gpu", "gpu-cuda11.8", "gpu-cuda12.9", "gpu-directml", "all"],
        default="cpu",
        help="The build target: 'cpu', 'gpu' (builds all GPU versions), specific gpu targets including 'gpu-directml', or 'all'. Defaults to 'cpu'.",
    )
    parser.add_argument(
        "--cli-only",
        default="false",
        help="(Optional) Set to 'true' to skip building the GUI and only build the CLI.",
    )
    parser.add_argument(
        "--gui-only",
        default="false",
        help="(Optional) Set to 'true' to skip building the standalone CLI release and only build the GUI.",
    )
    parser.add_argument(
        "--signtool",
        default=None,
        help="(Optional, Windows only) Path to signtool.exe for code signing.",
    )
    parser.add_argument(
        "--sign-cert-name",
        default=None,
        help="(Optional, Windows only) The subject name of the certificate to use for signing.",
    )
    parser.add_argument(
        "--iscc",
        default=None,
        help="(Optional, Windows only) Path to the Inno Setup compiler (iscc.exe).",
    )
    parser.add_argument(
        "--archive",
        default="false",
        help="(Optional) Set to 'true' to create a compressed archive of the final build folder.",
    )
    parser.add_argument(
        "--windows-installer",
        default="false",
        help="(Optional, Windows only) Set to 'true' to create an Inno Setup installer.",
    )
    parser.add_argument(
        "--release-type",
        default=None,
        help="(Optional) Specify a release type (e.g., 'Beta', 'RC1') to append to the output artifact names.",
    )
    parser.add_argument(
        "--clean",
        default="false",
        help="(Optional) Set to 'true' to remove residuals from previous builds (Releases output and Nuitka .build/.dist caches) before building. Can be combined with other options; with --clean true alone, only the cleanup runs.",
    )
    args = parser.parse_args()

    is_cli_only = args.cli_only.lower() == "true"

    if args.clean.lower() == "true":
        clean_build_artifacts()
        # --clean true with no other build options: only the cleanup runs.
        build_requested = any(
            (
                args.target != "cpu",
                is_cli_only,
                args.gui_only.lower() == "true",
                args.signtool is not None,
                args.sign_cert_name is not None,
                args.iscc is not None,
                args.archive.lower() == "true",
                args.windows_installer.lower() == "true",
                args.release_type is not None,
            )
        )
        if not build_requested:
            print_header("Clean Complete")
            print("No build requested; cleaned residuals and exiting.")
            return

    # Prerequisite Checks
    if not is_cli_only:
        check_pyside6()
    check_dbus()
    check_7zip()

    if args.target in ("gpu", "gpu-directml", "all") and sys.platform == "win32":
        print_header("Checking DirectML Python dependencies...")
        missing_directml_deps: list[str] = []
        try:
            import easyocr  # type: ignore  # noqa: F401
        except Exception:
            missing_directml_deps.append("easyocr")
        try:
            import torch_directml  # type: ignore  # noqa: F401
        except Exception:
            missing_directml_deps.append("torch-directml")
        if missing_directml_deps:
            print(
                "ERROR: Missing DirectML dependency/dependencies: "
                + ", ".join(missing_directml_deps)
            )
            print("Install them before building the DirectML target:")
            print('  python -m pip install ".[directml]"')
            sys.exit(1)
        print("DirectML dependencies found.")

    # Remove PySide6 QML build artifacts that break patchelf on Linux
    clean_pyside6_qml_artifacts()

    paddle_version = get_latest_paddle_version()
    chrome_lens_version = get_latest_chrome_lens_version()

    releases_dir = Path("Releases")
    if releases_dir.exists():
        print_header("Cleaning previous build artifacts")
        print(f"Removing existing directory: {releases_dir}")
        shutil.rmtree(releases_dir)
    releases_dir.mkdir(exist_ok=True)

    print_header("Compiling Binaries")

    # Compile GUI conditionally
    gui_dist_folder = None
    if not is_cli_only:
        gui_script = "VideOCR_qt.py"
        gui_dist_folder = Path("VideOCR_qt.dist")
        if gui_dist_folder.exists():
            shutil.rmtree(gui_dist_folder)

        run_command(
            [
                sys.executable,
                "-m",
                "nuitka",
                "--assume-yes-for-downloads",
                "--include-module=av.utils",
                # The GUI only spawns the CLI. It does not run OCR itself.
                # Exclude heavy AI/ML libraries to prevent MSVC C1002 "out of heap space"
                # errors and to slash GUI compile times.
                "--nofollow-import-to=torch,torchvision,sympy,mpmath,easyocr,onnxruntime,rapidocr,scipy,scikit-image",
                "--jobs=4",
                gui_script,
            ]
        )
        if not gui_dist_folder.is_dir():
            print(
                f"ERROR: Nuitka failed to create the GUI dist folder: {gui_dist_folder}"
            )
            sys.exit(1)

        gui_exe = gui_dist_folder / "VideOCR.exe"
        if gui_exe.exists():
            sign_file(args.signtool, args.sign_cert_name, gui_exe)
    else:
        print("Skipping GUI compilation due to --cli-only flag.")

    # Compile CLI
    cli_folder = Path("CLI")
    cli_script = "videocr_cli.py"
    cli_dist_folder = cli_folder / "videocr_cli.dist"
    if cli_dist_folder.exists():
        shutil.rmtree(cli_dist_folder)

    cli_command = [
        sys.executable,
        "-m",
        "nuitka",
        "--assume-yes-for-downloads",
        "--include-module=av.utils",
        # Bundle the ass_qafix script as raw data so runpy can execute it
        "--include-data-files=../tools/ass_qafix/ass_qafix.py=tools/ass_qafix/ass_qafix.py",
        # Ignore PyTorch's JIT C++ builder and setuptools to save compile time
        "--nofollow-import-to=setuptools,pkg_resources,distutils,torch.utils.cpp_extension",
        # Allow bytecode-included packages (sympy, mpmath) to be imported
        "--no-deployment-flag=excluded-module-usage",
        "--include-package-data=rapidocr",
        "--jobs=4",
        cli_script,
    ]

    # Conditionally bundle C-extensions and dynamic imports.
    # We only add them if they are actually installed in the current build environment
    # (e.g., DirectML/ONNX/EasyOCR are Windows-only and won't be present in Linux CI runners).
    optional_c_modules = [
        "jieba3",
        "rapidfuzz",
        "rich",
        "onnxruntime",
        "torch_directml",
        "easyocr",
        "torch",
        "rapidocr",
    ]
    for mod in optional_c_modules:
        try:
            __import__(mod)
            cli_command.append(f"--include-module={mod}")
        except ImportError:
            pass

    # Bundle heavy pure-Python libraries (sympy, mpmath) as bytecode instead of
    # compiling them to C. This saves massive amounts of CI compile time while
    # ensuring they are present at runtime to prevent missing module errors.
    try:
        __import__("sympy")
        cli_command.extend(["--nofollow-import-to=sympy", "--include-package=sympy"])
    except ImportError:
        pass
    try:
        __import__("mpmath")
        cli_command.extend(["--nofollow-import-to=mpmath", "--include-package=mpmath"])
    except ImportError:
        pass

    run_command(cli_command, cwd=str(cli_folder))
    if not cli_dist_folder.is_dir():
        print(f"ERROR: Nuitka failed to create the CLI dist folder: {cli_dist_folder}")
        sys.exit(1)

    cli_exe = cli_dist_folder / "videocr-cli.exe"
    if cli_exe.exists():
        sign_file(args.signtool, args.sign_cert_name, cli_exe)

    # --- Package for each target ---
    if args.target == "cpu":
        targets_to_build = ["cpu"]
    elif args.target == "gpu":
        targets_to_build = ["gpu-cuda11.8", "gpu-cuda12.9"]
        if sys.platform == "win32":
            targets_to_build.append("gpu-directml")
    elif args.target == "all":
        targets_to_build = ["cpu", "gpu-cuda11.8", "gpu-cuda12.9"]
        if sys.platform == "win32":
            targets_to_build.append("gpu-directml")
    else:
        targets_to_build = [args.target]

    for i, build_target in enumerate(targets_to_build):
        package_target(
            build_target,
            args,
            releases_dir,
            gui_dist_folder,
            cli_dist_folder,
            paddle_version,
            chrome_lens_version,
        )
        if i < len(targets_to_build) - 1:
            if build_target == "gpu-directml":
                completed_target_name = "GPU-DirectML"
            elif "gpu" in build_target:
                completed_target_name = build_target.replace("gpu-", "GPU-").replace(
                    "cuda", "CUDA-"
                )
            else:
                completed_target_name = build_target.upper()
            print("\n" + "#" * 60)
            print(
                f"Completed packaging for {completed_target_name}. Starting next target..."
            )
            print("#" * 60)

    # --- Final Cleanup ---
    print_header("Final Cleanup")
    print("Removing temporary compilation directories...")
    if gui_dist_folder:
        shutil.rmtree(gui_dist_folder)
    shutil.rmtree(cli_dist_folder)

    print_header("All Builds Complete!")
    print(f"All outputs are located in the '{releases_dir}' folder.")


if __name__ == "__main__":
    main()
