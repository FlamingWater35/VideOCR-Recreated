"""Allow running as ``python -m videocr_gui``."""

from __future__ import annotations

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
