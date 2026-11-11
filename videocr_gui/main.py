"""Application entry point for VideOCR Recreated."""

from __future__ import annotations

import os
import sys


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Do this before anything can spawn a child: a windowed build has no console,
    # so every unmarked child would open its own window and flash on screen
    # (ffmpeg -version and the `ver` call inside `import torch_directml` did).
    from videocr_gui import no_console

    no_console.suppress_child_consoles()

    # Qt 6 is Per-Monitor V2 DPI aware by default on Windows. Do NOT call
    # SetProcessDpiAwareness/SetProcessDPIAware here: that would change the
    # process awareness context after Qt already set it, producing
    # "qt.qpa.window: SetProcessDpiAwarenessContext() failed: Access is denied."
    if hasattr(Qt, "ApplicationAttribute"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    from videocr_gui import config
    from videocr_gui import constants as C
    from videocr_gui.app import MainWindow

    # QT_SCALE_FACTOR must be in the environment before QApplication exists, so
    # the persisted GUI scaling is resolved here rather than in the settings tab.
    # "system_default" leaves the variable unset so Qt follows the OS DPI.
    _scale = C.GUI_SCALING_FACTORS.get(
        str(config.load_settings().get("gui_scaling", C.DEFAULT_GUI_SCALING))
    )
    if _scale is not None:
        os.environ["QT_SCALE_FACTOR"] = _scale

    # In source/dev mode, make the CLI package importable when the GUI is launched
    # from a different working directory.
    sys.path.insert(0, os.path.join(config.APP_DIR, "CLI"))

    app = QApplication(sys.argv)
    app.setApplicationName("VideOCR Recreated")
    app.setOrganizationName("VideOCR Recreated")

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
