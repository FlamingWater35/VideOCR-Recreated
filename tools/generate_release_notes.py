#!/usr/bin/env python3
"""Generate the release notes for a VideOCR Recreated GitHub release.

Renders a download table (one row per OS / build target) with user-friendly
links pointing at the release assets, then appends an auto-generated changelog
derived from ``git log`` between two tags.

Two modes:

1. Asset-driven (recommended for CI): pass ``--assets-file`` with the real
   asset names (one per line, e.g. from
   ``gh release view TAG --json assets --jq '.assets[].name'``). The table is
   built from what is actually uploaded, so split volumes and any sizing
   surprises stay in sync automatically.

2. Static (useful for a preview before builds finish): pass ``--tag`` (and
   optionally ``--prev-tag``); the expected asset names are derived from the
   build script's naming rules.

Usage:
    python tools/generate_release_notes.py \
        --tag v1.6.1 \
        --repo FlamingWater35/VideOCR-Recreated \
        --assets-file assets.txt \
        [--prev-tag v1.6.0] \
        [--out RELEASE_NOTES.md]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DOWNLOAD_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"

# ---------------------------------------------------------------------------
# Static naming rules (kept in sync with build.py) — only used in static mode.
# ---------------------------------------------------------------------------
# target -> (display name, base name segment)
TARGETS = [
    ("cpu", "CPU", "CPU-v{VERSION}"),
    ("gpu-cuda11.8", "GPU CUDA 11.8", "GPU-v{VERSION}-CUDA-11.8"),
    ("gpu-cuda12.9", "GPU CUDA 12.9", "GPU-v{VERSION}-CUDA-12.9"),
    ("gpu-directml", "GPU DirectML", "GPU-v{VERSION}-DirectML"),
]

OSES = [
    ("Windows", ""),
    ("Linux", "-Linux"),
]

# Order used for both modes: Windows first, then Linux; CPU, CUDA 11.8,
# CUDA 12.9, DirectML.
BUILD_ORDER = {"CPU": 0, "CUDA 11.8": 1, "CUDA 12.9": 2, "DirectML": 3}


def asset_base(target: str, version: str, os_name: str) -> str:
    """Return the shared 'VideOCR-...' prefix (without extension) for an asset."""
    short = {t: s for t, _display, s in TARGETS}
    segment = short[target].format(VERSION=version)
    suffix = "-Linux" if os_name == "Linux" else ""
    return f"VideOCR-{segment}{suffix}"


def link(asset: str, repo: str, tag: str) -> str:
    url = DOWNLOAD_URL.format(repo=repo, tag=tag, asset=asset)
    return f"[{asset}]({url})"


def render_static_table(version: str, repo: str, tag: str) -> str:
    """Render the table from expected asset names (static/preview mode)."""
    lines = [
        "| Operating system | Build | Portable | Installer |",
        "| --- | --- | --- | --- |",
    ]
    for os_name, _suffix in OSES:
        for target, display, _seg in TARGETS:
            if target == "gpu-directml" and os_name == "Linux":
                continue
            base = asset_base(target, version, os_name)
            if os_name == "Linux" and target == "gpu-cuda12.9":
                portable = (
                    f"{link(base + '.7z.001', repo, tag)} + "
                    f"{link(base + '.7z.002', repo, tag)}"
                )
            else:
                portable = link(base + ".7z", repo, tag)
            installer = (
                link(base + "-setup-x64.exe", repo, tag) if os_name == "Windows" else "—"
            )
            lines.append(f"| **{os_name}** | {display} | {portable} | {installer} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Asset-driven parsing
# ---------------------------------------------------------------------------
def parse_asset(name: str) -> tuple[str, str, str]:
    """Map an asset name to (os, build, base) using its filename only."""
    if name.endswith(".exe"):
        base = name[: -len(".exe")]
    elif ".7z." in name:  # split volume, e.g. VideOCR-...-Linux.7z.001
        base = name.rsplit(".7z.", 1)[0]
    else:
        base = name.removesuffix(".7z")

    os_name = "Linux" if "-Linux" in base else "Windows"
    if "DirectML" in base:
        build = "DirectML"
    elif "CUDA-11.8" in base:
        build = "CUDA 11.8"
    elif "CUDA-12.9" in base:
        build = "CUDA 12.9"
    else:
        build = "CPU"
    return os_name, build, base


def volume_number(name: str) -> int | None:
    """Return the split-volume number for a name, or None if not a split part."""
    m = re.search(r"\.7z\.(\d+)$", name)
    return int(m.group(1)) if m else None


def render_asset_table(assets: list[str], repo: str, tag: str) -> str:
    """Render the table from the actual uploaded asset names."""
    rows: dict[tuple[str, str], list[str]] = {}
    for asset in assets:
        os_name, build, _base = parse_asset(asset)
        rows.setdefault((os_name, build), []).append(asset)

    def sort_key(item):
        (os_name, build), names = item
        os_order = 0 if os_name == "Windows" else 1
        return (os_order, BUILD_ORDER.get(build, 99), build)

    lines = [
        "| Operating system | Build | Portable | Installer |",
        "| --- | --- | --- | --- |",
    ]
    for (os_name, build), names in sorted(rows.items(), key=sort_key):
        parts = sorted(names, key=lambda n: (volume_number(n) is not None, volume_number(n) or 0, n))
        installers = [n for n in parts if n.endswith(".exe")]
        archives = [n for n in parts if not n.endswith(".exe")]

        portable = " + ".join(link(n, repo, tag) for n in archives)
        installer = " · ".join(link(n, repo, tag) for n in installers) if installers else "—"
        lines.append(f"| **{os_name}** | {build} | {portable} | {installer} |")
    return "\n".join(lines)


def changelog(prev_tag: str | None, tag: str, repo: str) -> str:
    """Build a concise changelog from git log between prev_tag and tag."""
    if prev_tag:
        spec = f"{prev_tag}..{tag}"
        header = f"### What's Changed since `{prev_tag}`"
    else:
        spec = tag
        header = f"### What's Changed (commits in `{tag}`)"
    try:
        log = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", spec],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        log = ""

    if not log:
        return (
            header
            + "\n\n_No commits listed. If this is the preview from the "
            "create-release step, it is regenerated from the real assets after "
            "the builds finish._"
        )
    linked = []
    for line in log.splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        linked.append(f"- [`{sha}`](https://github.com/{repo}/commit/{sha}) {subject}")
    return header + "\n\n" + "\n".join(linked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, e.g. v1.6.1")
    parser.add_argument(
        "--repo", default="FlamingWater35/VideOCR-Recreated", help="owner/repo"
    )
    parser.add_argument(
        "--assets-file",
        default=None,
        help="Path to a file with the real release asset names (one per line). "
        "When given, the table is built from these instead of the static naming rules.",
    )
    parser.add_argument(
        "--prev-tag", default=None, help="Previous tag for the changelog range"
    )
    parser.add_argument("--out", default="RELEASE_NOTES.md", help="Output path")
    args = parser.parse_args()

    version = args.tag[1:] if args.tag.startswith("v") else args.tag

    if args.assets_file:
        assets = [
            line.strip()
            for line in Path(args.assets_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        table = render_asset_table(assets, args.repo, args.tag)
    else:
        table = render_static_table(version, args.repo, args.tag)

    notes = [
        f"# VideOCR Recreated {args.tag}",
        "",
        "## Downloads",
        "",
        f"Assets are attached to this release (tag `{args.tag}`). Choose the build for "
        "your operating system and OCR backend. All builds include the GUI and the CLI.",
        "",
        table,
        "",
        changelog(args.prev_tag, args.tag, args.repo),
    ]

    Path(args.out).write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote release notes to {args.out}")


if __name__ == "__main__":
    main()
