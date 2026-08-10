# Compilation instructions
# nuitka-project: --standalone
# nuitka-project: --enable-plugin=pyside6
# nuitka-project-if: {OS} == "Windows":
#     nuitka-project: --include-qt-plugins=platforms,styles,imageformats,iconengines,platforminputcontexts
# nuitka-project-if: {OS} == "Linux":
#     nuitka-project: --include-qt-plugins=platforms,platformthemes,xcbglintegrations,imageformats,iconengines,platforminputcontexts
# nuitka-project: --windows-console-mode=disable
# nuitka-project: --include-windows-runtime-dlls=yes
# nuitka-project: --output-filename=VideOCR.exe
# nuitka-project: --include-data-files=Installer/*.ico=VideOCR.ico
# nuitka-project: --include-data-files=Installer/*.png=VideOCR.png
# nuitka-project: --include-data-dir=languages=languages

# Windows-specific metadata for the executable
# nuitka-project-if: {OS} == "Windows":
#     nuitka-project-set: APP_VERSION = __import__("_version").__version__
#     nuitka-project: --file-description="VideOCR Recreated"
#     nuitka-project: --file-version={APP_VERSION}
#     nuitka-project: --product-name="VideOCR Recreated GUI"
#     nuitka-project: --product-version={APP_VERSION}
#     nuitka-project: --copyright="timminator"
#     nuitka-project: --windows-icon-from-ico=Installer/VideOCR.ico

from videocr_gui.main import main

if __name__ == "__main__":
    raise SystemExit(main())
