"""
Iris — a small PySide6 desktop viewer for NetCDF / gridded scientific data.

Run with:
    python main.py

Companion project to Xenia (github.com/mixstam1821/Xenia), but native desktop
instead of browser-based: same "load gridded data, pick a variable, render it"
idea, built with real Qt widgets instead of FastAPI + MapLibre.
"""

import sys

from PySide6.QtWidgets import QApplication

from iris.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Iris")
    app.setOrganizationName("mixstam1821")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
