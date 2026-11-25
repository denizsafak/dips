"""Application entrypoint for DIP Studio across Windows, macOS, and Linux."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Ensure direct execution (`python src/dips/main.py`) resolves `dips.*` from any CWD
_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QFileOpenEvent, QGuiApplication, QIcon
from PyQt6.QtWidgets import QApplication

from dips import __app_name__, __version__
from dips.assets import asset_path
from dips.ui.main_window import MainWindow


class DIPSApplication(QApplication):
    """QApplication subclass supporting macOS Finder FileOpen events."""

    def __init__(self, argv: list[str]) -> None:
        super().__init__(argv)
        self.main_window: MainWindow | None = None
        self.pending_open_path: Path | None = None

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.FileOpen and isinstance(event, QFileOpenEvent):
            path = Path(event.file())
            if path.is_file():
                if self.main_window is not None:
                    self.main_window.open_image_path(path)
                else:
                    self.pending_open_path = path
            return True
        return super().event(event)


def _has_registered_desktop_file() -> bool:
    candidates = [
        Path.home() / ".local" / "share" / "applications" / "dips.desktop",
        Path("/usr/local/share/applications/dips.desktop"),
        Path("/usr/share/applications/dips.desktop"),
    ]
    for data_dir in os.environ.get("XDG_DATA_DIRS", "").split(":"):
        if data_dir.strip():
            candidates.append(Path(data_dir.strip()) / "applications" / "dips.desktop")
    return any(p.is_file() for p in candidates)


def _configure_platform_identity() -> None:
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("denizsafak.dips.studio")
        except Exception:
            pass
    elif system == "Linux" and _has_registered_desktop_file():
        QGuiApplication.setDesktopFileName("dips")


def main(argv: list[str] | None = None) -> int:
    """Launch DIP Studio application."""
    multiprocessing.freeze_support()
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    args = argv if argv is not None else sys.argv
    app = DIPSApplication(args)
    app.setQuitOnLastWindowClosed(True)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("DenizSafak")
    _configure_platform_identity()

    icon_file = asset_path("dips.png")
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))

    initial_path = Path(args[1]) if len(args) > 1 and Path(args[1]).is_file() else app.pending_open_path
    window = MainWindow(initial_image=initial_path)
    app.main_window = window
    window.show()
    code = app.exec()
    from dips.core.ops_ai import release_ai_sessions
    release_ai_sessions()
    for child in multiprocessing.active_children():
        try:
            child.terminate()
            child.join(timeout=0.2)
            if child.is_alive():
                child.kill()
        except Exception:
            pass
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
