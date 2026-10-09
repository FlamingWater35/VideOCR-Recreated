"""Startup update check against the project's GitHub releases.

Stdlib only (urllib) and always called from a background thread started by
the main window, so the UI never blocks on the network. Any failure —
offline, rate-limited, malformed payload — resolves to ``None``: the caller
simply shows no Update tab.
"""

from __future__ import annotations

import json
import re
import urllib.request

#: Delay between application start and the (single) update check.
BOOT_CHECK_DELAY_MS = 3000

#: Direct link to the newest published release (shown in the Update tab).
LATEST_RELEASE_URL = (
    "https://github.com/FlamingWater35/VideOCR-Recreated/releases/latest"
)
_RELEASES_API_URL = (
    "https://api.github.com/repos/FlamingWater35/VideOCR-Recreated/releases/latest"
)
_REQUEST_TIMEOUT_S = 8.0


def _version_tuple(version: str) -> tuple[int, ...]:
    """Extracts the numeric release components ('v1.10.2' -> (1, 10, 2))."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def is_newer(latest: str, current: str) -> bool:
    """True when ``latest`` is a strictly newer release than ``current``."""
    latest_key = _version_tuple(latest)
    if not latest_key:
        return False
    return latest_key > _version_tuple(current)


def fetch_latest_version(timeout: float = _REQUEST_TIMEOUT_S) -> str | None:
    """Returns the latest release tag without a leading 'v', or ``None``."""
    request = urllib.request.Request(
        _RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "VideOCR-Recreated-update-check",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception:
        # Network boundary: offline, rate-limited or unreadable response all
        # mean the same thing to the caller — no update information.
        return None
    if not isinstance(payload, dict):
        return None
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return None
    return tag[1:] if tag[:1] in ("v", "V") else tag
