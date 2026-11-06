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
ISS_SCRIPT = REPO_ROOT / "Installer" / "Windows" / "installer_template.iss"
ISS_LANGUAGES = REPO_ROOT / "Installer" / "Windows" / "Languages"


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


def _load_build_module():
    """Import build.py from the repo root without executing a build."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_build_contract", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _png_icon_size() -> int:
    """Read the real pixel size of the repo icon (PNG IHDR bytes 16-24)."""
    import struct

    header = (REPO_ROOT / "Installer" / "VideOCR.png").read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])
    return max(width, height)


def _make_png(width: int, height: int) -> bytes:
    """Build a minimal PNG header with a real IHDR (enough for size probing)."""
    import struct

    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + ihdr
        + b"\x00\x00\x00\x00"
    )


def test_appimage_appdir_drops_portable_flag_and_keeps_payload_siblings(tmp_path):
    """An AppDir mounts read-only, so portable_mode.txt must not survive.

    The GUI resolves its data next to sys.executable, so the binary and its
    siblings have to stay in the same directory inside the AppDir.
    """
    build = _load_build_module()

    package = tmp_path / "VideOCR-CPU-v1.6.11-Linux"
    (package / "av").mkdir(parents=True)
    (package / "VideOCR.bin").write_text("", encoding="utf-8")
    (package / "videocr-cli.bin").write_text("", encoding="utf-8")
    (package / "build_variant.txt").write_text("cpu", encoding="utf-8")
    (package / "portable_mode.txt").write_text("", encoding="utf-8")
    # A deliberately non-256 icon: the hicolor directory must follow the real
    # pixel size instead of a hardcoded guess.
    (package / "VideOCR.png").write_bytes(_make_png(512, 512))

    appdir = tmp_path / "AppDir"
    build._build_appdir(package, appdir, "CPU")

    payload = appdir / "usr" / "bin"
    assert not (payload / "portable_mode.txt").exists(), (
        "portable_mode.txt must be removed: the AppDir is read-only at run time"
    )
    assert (payload / "VideOCR.bin").is_file()
    assert (payload / "videocr-cli.bin").is_file()
    assert (payload / "build_variant.txt").read_text(encoding="utf-8") == "cpu"
    assert (payload / "av").is_dir()

    # appimagetool requires these at the AppDir root, and AppRun must be
    # executable or the AppImage will not start.
    assert (appdir / "AppRun").is_file()
    assert (appdir / "VideOCR.desktop").is_file()
    assert (appdir / "videocr.png").is_file()
    icon = (
        appdir
        / "usr"
        / "share"
        / "icons"
        / "hicolor"
        / "512x512"
        / "apps"
        / "videocr.png"
    )
    assert icon.is_file(), "hicolor directory must match the icon's pixel size"

    import os
    import stat as stat_module

    assert os.stat(appdir / "AppRun").st_mode & stat_module.S_IXUSR
    assert os.stat(appdir / "VideOCR.desktop").st_mode & stat_module.S_IXUSR

    desktop = (appdir / "VideOCR.desktop").read_text(encoding="utf-8")
    # The Exec target must match the Linux binary the build actually produces.
    assert "Exec=VideOCR.bin" in desktop
    assert "Exec=VideOCR.exe" not in desktop
    assert "Icon=videocr" in desktop

    app_run = (appdir / "AppRun").read_text(encoding="utf-8")
    assert "VideOCR.bin" in app_run


def test_appimage_naming_matches_release_assets():
    """The AppImage name must follow the archive naming scheme.

    tools/generate_release_notes.py derives the download link from the portable
    archive name plus "-x86_64.AppImage", so the two must agree.
    """
    build = _load_build_module()
    arch = build._appimage_arch()
    assert arch in ("x86_64", "aarch64")

    # Mirrors the f-string in create_linux_appimage().
    produced = f"VideOCR-CPU-v{build.APP_VERSION}-Linux-{arch}.AppImage"
    assert produced.startswith("VideOCR-CPU-v")
    assert produced.endswith(f"-Linux-{arch}.AppImage")

    notes = (REPO_ROOT / "tools" / "generate_release_notes.py").read_text(
        encoding="utf-8"
    )
    assert ".AppImage" in notes, "release notes must list AppImage assets"
    assert "AppImage |" in notes, "release notes table needs an AppImage column"


def test_release_notes_parse_appimage_asset_names():
    """parse_asset() must fold an AppImage back onto its portable base name."""
    import importlib.util
    import sys

    notes_path = REPO_ROOT / "tools" / "generate_release_notes.py"
    spec = importlib.util.spec_from_file_location("_release_notes", notes_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    os_name, build_name, base = module.parse_asset(
        "VideOCR-CPU-v1.6.11-Linux-x86_64.AppImage"
    )
    assert (os_name, build_name) == ("Linux", "CPU")
    assert base == "VideOCR-CPU-v1.6.11-Linux"

    os_name, build_name, base = module.parse_asset(
        "VideOCR-GPU-v1.6.11-CUDA-11.8-Linux-x86_64.AppImage"
    )
    assert (os_name, build_name) == ("Linux", "CUDA 11.8")
    assert base == "VideOCR-GPU-v1.6.11-CUDA-11.8-Linux"


def test_appimage_only_built_on_linux_and_opted_in():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    # Linux-only guard plus the opt-in flag; building on Windows must be a no-op.
    assert 'sys.platform != "linux"' in text
    assert "args.appimage" in text
    assert '"--appimage"' in text or "'--appimage'" in text


def test_installer_language_files_are_vendored():
    """Every non-``compiler:`` MessagesFile must exist in the repository.

    Inno Setup aborts the whole compile with a bare "exit code 2" when a
    MessagesFile cannot be opened. ``chineseSimplified.isl`` is NOT part of
    Inno Setup's own Languages/ folder at 6.7.x (it lives under
    Files/Languages/Unofficial/ in issrc), so relying on the compiler's bundled
    copy failed every Windows build. The files are vendored instead.
    """
    build = _load_build_module()
    missing = build._missing_installer_languages(ISS_SCRIPT)
    assert not missing, f"missing installer language files: {missing}"


def test_installer_languages_are_relative_not_compiler_provided():
    """Non-English languages must not fall back to the compiler's folder.

    Only real ``Name:`` entries are inspected — the .iss explains the bug in a
    comment that mentions the old ``compiler:`` path, and that prose must not
    make this test fail.
    """
    text = ISS_SCRIPT.read_text(encoding="utf-8")
    entries = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("Name:") and "MessagesFile:" in line
    ]
    assert entries, "no [Languages] entries parsed"

    for line in entries:
        assert "compiler:Languages\\" not in line, (
            f"non-English language must load from the vendored folder: {line}"
        )

    for lang in (
        "German",
        "ChineseSimplified",
        "Spanish",
        "French",
        "Portuguese",
        "Italian",
        "Arabic",
        "Russian",
        "Indonesian",
        "Thai",
        "Korean",
        "Japanese",
        "Vietnamese",
    ):
        assert (ISS_LANGUAGES / f"{lang}.isl").is_file(), (
            f"vendored {lang}.isl is missing"
        )
        assert f"Languages\\{lang}.isl" in text, (
            f"{lang} must be referenced from the vendored Languages/ folder"
        )


def _installer_language_entries() -> list[tuple[str, str]]:
    """Return the (Name, MessagesFile) pairs from the .iss [Languages] section."""
    text = ISS_SCRIPT.read_text(encoding="utf-8")
    return re.findall(
        r'^\s*Name:\s*"([^"]+)";\s*MessagesFile:\s*"([^"]+)"', text, re.MULTILINE
    )


def _installer_language_case_map() -> dict[str, str]:
    """Return {installer language name: app language code} from [Code]."""
    text = ISS_SCRIPT.read_text(encoding="utf-8")
    pairs = re.findall(
        r"^\s*'([^']+)':\s*SetIniString\('Settings', '--language', '([^']+)'",
        text,
        re.MULTILINE,
    )
    return dict(pairs)


def test_installer_covers_every_app_language():
    """The installer must offer a language for every UI language the app ships.

    The app's UI languages are the ``languages/*.json`` files. A language that
    exists in the app but not in the installer leaves those users with an
    English-only setup wizard — which is how Indonesian and Vietnamese were
    commented out.
    """
    app_codes = {p.stem for p in (REPO_ROOT / "languages").glob("*.json")}
    assert app_codes, "no app language files found"

    entry_names = {name for name, _ in _installer_language_entries()}
    case_map = _installer_language_case_map()

    # A language only counts as offered when it is BOTH an active [Languages]
    # entry and has a --language mapping. Commenting out either one silently
    # drops it from the wizard, which is how Indonesian/Vietnamese were lost.
    orphan_mappings = sorted(set(case_map) - entry_names)
    assert not orphan_mappings, (
        f"these languages have a --language mapping but no active [Languages] "
        f"entry, so the wizard never offers them: {orphan_mappings}"
    )

    # English is the `else` fallback of the case statement, not an explicit arm.
    covered = {case_map[name] for name in entry_names if name in case_map} | {"en"}

    missing = sorted(app_codes - covered)
    assert not missing, (
        f"installer does not offer these app languages: {missing} "
        "(add them to [Languages] and to the [Code] case block)"
    )

    extra = sorted(covered - {"en"} - app_codes)
    assert not extra, (
        f"installer maps language codes the app does not ship: {extra}"
    )


def test_every_installer_language_sets_the_app_language():
    """Each [Languages] entry needs a --language mapping.

    Without one the wizard shows the right language but the app still starts in
    English, because the mapping is what writes the setting.
    """
    entries = _installer_language_entries()
    assert entries, "no [Languages] entries parsed"

    case_map = _installer_language_case_map()
    unmapped = sorted(name for name, _ in entries if name != "english" and name not in case_map)
    assert not unmapped, (
        f"installer languages with no --language mapping: {unmapped}"
    )

    # Every mapping must point at a language the app actually ships.
    app_codes = {p.stem for p in (REPO_ROOT / "languages").glob("*.json")}
    for name, code in case_map.items():
        assert code in app_codes, (
            f"installer language '{name}' maps to '{code}', "
            f"which is not one of {sorted(app_codes)}"
        )


def test_installer_preflight_flags_a_missing_language_file(tmp_path):
    """build.py must report the missing file instead of ISCC's bare exit 2."""
    build = _load_build_module()
    langs = tmp_path / "Languages"
    langs.mkdir()
    (langs / "German.isl").write_text("x", encoding="utf-8")
    script = tmp_path / "t.iss"
    script.write_text(
        "[Languages]\n"
        'Name: "english"; MessagesFile: "compiler:Default.isl"\n'
        'Name: "german"; MessagesFile: "Languages\\German.isl"\n'
        'Name: "thai"; MessagesFile: "Languages\\Thai.isl"\n'
        '; Name: "indonesian"; MessagesFile: "Languages\\Indonesian.isl"\n',
        encoding="utf-8",
    )

    missing = build._missing_installer_languages(script)
    names = sorted(p.name for p in missing)
    # Thai is referenced but absent; Indonesian is commented out and ignored;
    # Default.isl comes from the compiler and is not checked.
    assert names == ["Thai.isl"], names
