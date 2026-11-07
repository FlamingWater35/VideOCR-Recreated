"""Locating, versioning and replacing the FFmpeg executable.

FFmpeg is optional: the app always works through PyAV. Only the
``ffmpeg_d3d11va`` frame-scan mode shells out to an ``ffmpeg`` binary, so this
module exists to answer one question well — *which* ffmpeg should run.

Resolution order (first hit wins):

1. the managed copy shipped with the app (``<app root>/ffmpeg/ffmpeg[.exe]``),
   which the GUI's Advanced Settings can update in place;
2. whatever ``shutil.which("ffmpeg")`` finds on ``PATH``.

The managed copy comes first on purpose: a packaged build then behaves the same
on every machine instead of inheriting whatever FFmpeg the user happens to have
installed, while a from-source checkout with no managed copy still picks up the
system binary.

Replacing the managed copy needs no application restart. The GUI never runs
ffmpeg itself — it spawns the CLI per job, and the CLI resolves the path at the
start of each run — so a file swapped on disk is picked up by the next job. The
installer writes to a sibling temp name and ``os.replace``s it into position,
which also keeps a partially downloaded archive from ever becoming the live
binary.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable

#: Directory name (under the application root) holding the managed copy.
MANAGED_DIR_NAME = "ffmpeg"

#: Official Windows builds. ``release-version`` is a one-line plain-text file,
#: and the "packages" URL is stable per version, so a build can pin an exact
#: release while the update check compares against the moving one.
FFMPEG_VERSION_URL = "https://www.gyan.dev/ffmpeg/builds/release-version"
FFMPEG_PACKAGE_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-{version}-essentials_build.zip"
)
FFMPEG_LATEST_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

_VERSION_TIMEOUT_S = 15.0
_DOWNLOAD_TIMEOUT_S = 300.0
_USER_AGENT = "VideOCR-Recreated-ffmpeg"

#: Files kept from the upstream archive. ``ffplay`` is deliberately skipped.
_KEEP_BINARIES = ("ffmpeg", "ffprobe")

_VERSION_RE = re.compile(r"ffmpeg version (\S+)", re.IGNORECASE)


def executable_name(base: str = "ffmpeg") -> str:
    """Platform-appropriate binary name."""
    return f"{base}.exe" if sys.platform == "win32" else base


def candidate_roots() -> list[str]:
    """Directories that may contain the managed ``ffmpeg/`` folder.

    A packaged build puts the CLI and the GUI in the same folder, so one root
    covers both. Running from source they differ — ``VideOCR_qt.py`` sits in the
    repo root while ``CLI/videocr_cli.py`` is one level down — so the parent of
    a ``CLI`` directory is included too.
    """
    roots: list[str] = []
    try:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0] or ""))
    except Exception:
        script_dir = ""
    for candidate in (script_dir, os.getcwd()):
        if candidate and candidate not in roots:
            roots.append(candidate)
        if os.path.basename(candidate).lower() == "cli":
            parent = os.path.dirname(candidate)
            if parent and parent not in roots:
                roots.append(parent)
    return roots


def default_root(root: str | None = None) -> str:
    """The directory a managed install goes into."""
    if root is not None:
        return root
    roots = candidate_roots()
    return roots[0] if roots else os.getcwd()


def managed_path(root: str | None = None) -> str:
    """Path the managed ffmpeg binary has (or would have)."""
    return os.path.join(default_root(root), MANAGED_DIR_NAME, executable_name("ffmpeg"))


def find_managed(root: str | None = None) -> str | None:
    """The bundled copy, if it exists."""
    for base in [root] if root is not None else candidate_roots():
        if not base:
            continue
        path = os.path.join(base, MANAGED_DIR_NAME, executable_name("ffmpeg"))
        if os.path.isfile(path):
            return path
    return None


def find_ffmpeg(root: str | None = None) -> str | None:
    """Managed copy first, then ``PATH``. ``None`` when neither exists."""
    managed = find_managed(root)
    if managed:
        return managed
    return shutil.which("ffmpeg")


def resolve_ffmpeg(root: str | None = None) -> tuple[str | None, str]:
    """``(path, source)`` where source is ``"bundled"``, ``"system"`` or ``""``."""
    managed = find_managed(root)
    if managed:
        return managed, "bundled"
    system = shutil.which("ffmpeg")
    if system:
        return system, "system"
    return None, ""


def ffmpeg_version(path: str | None, timeout: float = 10.0) -> str | None:
    """Parses the version out of ``ffmpeg -version``; ``None`` if unreadable."""
    if not path:
        return None
    try:
        completed = subprocess.run(
            [path, "-version"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    text = ""
    for raw in (completed.stdout, completed.stderr):
        if raw:
            text += raw.decode("utf-8", errors="replace")
    match = _VERSION_RE.search(text)
    return match.group(1) if match else None


def latest_release_version(timeout: float = _VERSION_TIMEOUT_S) -> str | None:
    """Newest published FFmpeg release, or ``None`` when offline."""
    request = urllib.request.Request(
        FFMPEG_VERSION_URL, headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
    except Exception:
        # Network boundary: offline, proxy or DNS failure all mean "unknown".
        return None
    return text.splitlines()[0].strip() if text else None


def is_newer(latest: str | None, current: str | None) -> bool:
    """True when ``latest`` is a strictly newer release than ``current``.

    Returns ``False`` when either side is not a comparable release number. Git
    snapshot builds (``2026-07-20-git-c23123630e-...``) report a *date* where a
    release reports a version, and comparing those two number spaces would
    invent an update that does not exist.
    """
    latest_key = _version_key(latest or "")
    current_key = _version_key(current or "")
    if not latest_key or not current_key:
        return False
    return latest_key > current_key


def _version_key(version: str) -> tuple[int, ...]:
    """Numeric components of a release version, or ``()`` when not comparable.

    ``9.0.2-essentials_build-www.gyan.dev`` and ``9.0.2`` must compare equal, so
    only the leading dotted-numeric run is significant. A run whose first
    component is a four-digit year is a git snapshot date, not a release.
    """
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", version)
    if not match:
        return ()
    parts = [int(part) for part in match.group(1).split(".")]
    if parts and parts[0] >= 1900:
        return ()
    return tuple(parts)


def short_version(version: str | None) -> str:
    """Trims a verbose ``-version`` token for display.

    ``9.0.2-essentials_build-www.gyan.dev`` -> ``9.0.2``;
    ``2026-07-20-git-c23123630e-...`` -> ``2026-07-20 (git)``.
    """
    if not version:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})-git", version)
    if match:
        return f"{match.group(1)} (git)"
    key = _version_key(version)
    if key:
        return ".".join(str(part) for part in key)
    return version


def download_zip(
    destination: str,
    version: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Downloads the FFmpeg essentials archive; returns its path.

    ``progress`` receives ``(bytes_done, bytes_total)``; the total is 0 when the
    server omits ``Content-Length``.
    """
    url = (
        FFMPEG_PACKAGE_URL.format(version=version) if version else FFMPEG_LATEST_ZIP_URL
    )
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    return destination


def install_from_zip(
    archive: str,
    root: str | None = None,
    keep_probe: bool = True,
) -> str:
    """Replaces the managed ffmpeg with the binaries inside ``archive``.

    Writes each binary to a temporary sibling and ``os.replace``s it into
    position, so a failed or interrupted install leaves the previous working
    copy untouched. Returns the installed ffmpeg path.

    Raises ``ValueError`` when the archive has no ffmpeg binary, and lets
    ``OSError`` propagate when the target is locked (for example while a job is
    running) so the caller can report it plainly.
    """
    target_dir = os.path.join(default_root(root), MANAGED_DIR_NAME)
    os.makedirs(target_dir, exist_ok=True)

    wanted = list(_KEEP_BINARIES) if keep_probe else ["ffmpeg"]
    staged: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            for name in wanted:
                member = _find_binary_member(names, name)
                if member is None:
                    if name == "ffmpeg":
                        raise ValueError(
                            "The downloaded archive does not contain an ffmpeg binary."
                        )
                    continue
                final_path = os.path.join(target_dir, executable_name(name))
                temp_path = f"{final_path}.new"
                with bundle.open(member) as source, open(temp_path, "wb") as sink:
                    shutil.copyfileobj(source, sink, 1024 * 512)
                staged.append((temp_path, final_path))

        for temp_path, final_path in staged:
            os.replace(temp_path, final_path)
    finally:
        # Any staged file that never reached its final name is dead weight.
        for temp_path, _final in staged:
            if os.path.exists(temp_path):
                with contextlib.suppress(OSError):
                    os.remove(temp_path)

    return os.path.join(target_dir, executable_name("ffmpeg"))


def _find_binary_member(names: list[str], base: str) -> str | None:
    """Finds ``.../bin/<base>[.exe]`` inside an archive's member list."""
    wanted = executable_name(base).lower()
    for name in names:
        lowered = name.lower()
        if lowered.endswith("/bin/" + wanted) or lowered == "bin/" + wanted:
            return name
    # Fall back to any member with the right filename, in case the archive
    # layout differs from the upstream build.
    for name in names:
        lowered = name.lower()
        if lowered == wanted or lowered.endswith("/" + wanted):
            return name
    return None


def install_managed(
    root: str | None = None,
    version: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Downloads and installs the managed ffmpeg; returns its path."""
    handle, temp_path = tempfile.mkstemp(suffix=".zip", prefix="ffmpeg-")
    os.close(handle)
    try:
        download_zip(temp_path, version=version, progress=progress)
        return install_from_zip(temp_path, root=root)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temp_path)


def managed_status(root: str | None = None) -> dict[str, str]:
    """Small snapshot for the GUI: path, source, version and managed path."""
    path, source = resolve_ffmpeg(root)
    return {
        "path": path or "",
        "source": source,
        "version": ffmpeg_version(path) or "",
        "managed_path": managed_path(root),
    }


def describe(path: str | None, source: str) -> str:
    """One-line human summary used in logs and the GUI."""
    if not path:
        return "no FFmpeg found (frame scan will use PyAV)"
    version = short_version(ffmpeg_version(path)) or "unknown version"
    where = "bundled" if source == "bundled" else "system"
    return f"{where} FFmpeg {version} at {path}"
