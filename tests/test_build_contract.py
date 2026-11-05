"""Guards for the packaged (Nuitka) executable naming contract.

Regression tests for the Linux build shipping a GUI binary named
``VideOCR.exe`` instead of ``VideOCR.bin``. Nuitka reads the
``nuitka-project`` directives at the top of the entry scripts, so these tests
assert the directives stay platform-guarded and that build.py / the Linux
installer agree with the names they produce.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_ENTRY = REPO_ROOT / "VideOCR_qt.py"
CLI_ENTRY = REPO_ROOT / "CLI" / "videocr_cli.py"
BUILD_SCRIPT = REPO_ROOT / "build.py"
LINUX_INSTALLER = REPO_ROOT / "Installer" / "Linux" / "install_videocr.sh"


def _directive_header(path: Path) -> str:
    """Return the leading ``# nuitka-project...`` block of an entry script.

    Indented continuation lines (``#     nuitka-project: ...``) belong to the
    preceding ``nuitka-project-if`` guard, so they are kept verbatim.
    """
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip("# \t")
        if stripped.startswith("nuitka-project"):
            lines.append(line)
        elif lines:
            break
    return "\n".join(lines)


def _parse_directives(header: str) -> list[tuple[str | None, str]]:
    """Pair each ``nuitka-project`` directive with its guard OS (None = always).

    Nuitka treats an *indented* ``nuitka-project:`` line as a member of the
    preceding ``nuitka-project-if`` block; an unindented one is unconditional.
    Getting this distinction wrong is precisely the bug being guarded against,
    so the tests parse indentation rather than trusting line order.
    """
    pairs: list[tuple[str | None, str]] = []
    current_os: str | None = None
    for line in header.splitlines():
        guard = re.search(r'nuitka-project-if:\s*\{OS\}\s*==\s*"(\w+)"', line)
        if guard:
            current_os = guard.group(1)
            continue
        # Nuitka's convention: "# nuitka-project:" (one space) is top-level;
        # "#     nuitka-project:" (extra indent) continues the preceding guard.
        match = re.match(r"^#([ \t]*)(.*)$", line)
        if not match:
            continue
        gap, body = match.group(1), match.group(2).strip()
        if not body.startswith("nuitka-project:"):
            continue
        indented = len(gap) >= 2 or gap.startswith("\t")
        pairs.append((current_os if indented else None, body))
    return pairs


def _output_filenames_by_os(header: str) -> dict[str, str]:
    """Map OS -> output filename declared by guarded directives."""
    filenames: dict[str, str] = {}
    for os_name, directive in _parse_directives(header):
        name = re.search(r"--output-filename=(\S+)", directive)
        if name and os_name is not None:
            filenames[os_name] = name.group(1)
    return filenames


def test_gui_entry_declares_platform_specific_names():
    names = _output_filenames_by_os(_directive_header(GUI_ENTRY))
    assert names.get("Windows") == "VideOCR.exe"
    assert names.get("Linux") == "VideOCR.bin"


def test_cli_entry_declares_platform_specific_names():
    names = _output_filenames_by_os(_directive_header(CLI_ENTRY))
    assert names.get("Windows") == "videocr-cli"
    assert names.get("Linux") == "videocr-cli.bin"


def test_gui_entry_has_no_unguarded_output_filename():
    # An unconditional --output-filename applies to every OS and is exactly the
    # bug that shipped a Linux ELF named VideOCR.exe.
    unguarded = [
        directive
        for os_name, directive in _parse_directives(_directive_header(GUI_ENTRY))
        if os_name is None and "--output-filename" in directive
    ]
    assert not unguarded, f"unguarded --output-filename: {unguarded}"


def test_linux_installer_exec_targets_linux_binary():
    text = LINUX_INSTALLER.read_text(encoding="utf-8")
    assert "VideOCR.bin" in text
    assert "VideOCR.exe" not in text


def test_build_script_sign_targets_are_platform_aware():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    # Both sign targets must pick the platform suffix, never a hardcoded .exe.
    assert re.search(
        r'gui_exe\s*=\s*gui_dist_folder\s*/\s*\(\s*"VideOCR\.exe"\s*if\s*sys\.platform'
        r'\s*==\s*"win32"\s*else\s*"VideOCR\.bin"\s*\)',
        text,
    ), "gui_exe must select VideOCR.exe on Windows and VideOCR.bin elsewhere"
    assert re.search(
        r'"videocr-cli\.exe"\s*if\s*sys\.platform\s*==\s*"win32"\s*else\s*'
        r'"videocr-cli\.bin"',
        text,
    ), "cli_exe must select videocr-cli.exe on Windows and videocr-cli.bin elsewhere"


def test_build_script_forces_pyav_shared_utility_module():
    # av/logging imports av._cyutil from C code, which Nuitka's import
    # analysis cannot see; the build must include it explicitly.
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "def av_cyutil_include_args()" in text
    assert "--include-module=av._cyutil" in text
    guards = text.count("*av_cyutil_include_args()")
    assert guards >= 2, "expected the helper to be spliced into GUI and CLI builds"
