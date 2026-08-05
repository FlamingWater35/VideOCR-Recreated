"""Application entry point for VideOCR Recreated."""

from __future__ import annotations

import os
import sys


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Qt 6 is Per-Monitor V2 DPI aware by default on Windows. Do NOT call
    # SetProcessDpiAwareness/SetProcessDPIAware here: that would change the
    # process awareness context after Qt already set it, producing
    # "qt.qpa.window: SetProcessDpiAwarenessContext() failed: Access is denied."
    if hasattr(Qt, "ApplicationAttribute"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    from videocr_gui import config
    from videocr_gui.app import MainWindow

    # In source/dev mode, make the CLI package importable when the GUI is launched
    # from a different working directory.
    sys.path.insert(0, os.path.join(config.APP_DIR, "CLI"))

    app = QApplication(sys.argv)
    app.setApplicationName("VideOCR Recreated")
    app.setOrganizationName("VideOCR")

    from videocr_gui import resources

    icon = resources.icon_path()
    if icon:
        from PySide6.QtGui import QIcon

        app.setWindowIcon(QIcon(icon))

    from videocr_gui import style

    style.apply(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
