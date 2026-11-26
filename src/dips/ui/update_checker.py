"""GitHub Releases update checker and cross-platform automatic self-updater for DIPS."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import __repo_slug__, __version__

GITHUB_API_LATEST = f"https://api.github.com/repos/{__repo_slug__}/releases/latest"
GITHUB_RELEASES_URL = f"https://github.com/{__repo_slug__}/releases/latest"


def parse_version(tag: str) -> tuple[int, ...]:
    """Parse a semantic version or tag like 'v1.2.3' into an integer tuple."""
    cleaned = tag.strip().lstrip("vV")
    nums = [int(x) for x in re.findall(r"\d+", cleaned)]
    return tuple(nums) if nums else (0,)


def is_newer_version(latest_tag: str, current_version: str = __version__) -> bool:
    """Return True when `latest_tag` is strictly newer than `current_version`."""
    latest = parse_version(latest_tag)
    current = parse_version(current_version)
    max_len = max(len(latest), len(current))
    return (latest + (0,) * (max_len - len(latest))) > (current + (0,) * (max_len - len(current)))


def select_platform_asset(assets: list[dict]) -> dict | None:
    """Choose the best downloadable GitHub Release asset for the current OS & architecture."""
    system = platform.system()
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    if system == "Windows":
        targets = ("windows-amd64.zip",)
    elif system == "Darwin":
        targets = ("macos-arm64.zip",) if is_arm else ("macos-amd64.zip",)
    elif system == "Linux":
        if os.environ.get("APPIMAGE"):
            targets = ("linux-amd64.appimage", "linux-amd64.tar.gz")
        else:
            targets = ("linux-amd64.tar.gz", "linux-amd64.appimage")
    else:
        return None

    for suffix in targets:
        for asset in assets:
            name = str(asset.get("name", "")).lower()
            if name.endswith(suffix) and asset.get("browser_download_url"):
                return asset
    return None


class UpdateCheckerThread(QThread):
    """Checks GitHub Releases API in the background without blocking UI startup."""

    updateAvailable = pyqtSignal(dict)  # release metadata dict
    upToDate = pyqtSignal(str)
    checkFailed = pyqtSignal(str)

    def __init__(self, current_version: str = __version__, parent=None) -> None:
        super().__init__(parent)
        self._current = current_version

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                GITHUB_API_LATEST,
                headers={"Accept": "application/vnd.github+json", "User-Agent": f"DIPS/{self._current}"},
            )
            with urllib.request.urlopen(req, timeout=4.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("draft") or data.get("prerelease"):
                self.upToDate.emit(self._current)
                return
            tag = str(data.get("tag_name", "")).strip()
            if tag and is_newer_version(tag, self._current):
                self.updateAvailable.emit(
                    {
                        "version": tag.lstrip("vV"),
                        "html_url": str(data.get("html_url", GITHUB_RELEASES_URL)),
                        "assets": data.get("assets", []),
                        "body": str(data.get("body", "")),
                    }
                )
            else:
                self.upToDate.emit(self._current)
        except Exception as exc:
            self.checkFailed.emit(str(exc))


class UpdateDownloadThread(QThread):
    """Downloads the platform update package with progress reporting."""

    progress = pyqtSignal(int)
    finishedDownload = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, url: str, dest: Path, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._dest = dest
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self) -> None:
        try:
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(self._url, headers={"User-Agent": f"DIPS/{__version__}"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(self._dest, "wb") as fh:
                total = int(resp.headers.get("Content-Length", "0") or "0")
                done = 0
                while True:
                    if self._cancelled or self.isInterruptionRequested():
                        break
                    chunk = resp.read(1024 * 128)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        self.progress.emit(min(100, int(done * 100 / total)))
            if self._cancelled or self.isInterruptionRequested():
                try:
                    self._dest.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            self.progress.emit(100)
            self.finishedDownload.emit(str(self._dest))
        except Exception as exc:
            if not (self._cancelled or self.isInterruptionRequested()):
                self.failed.emit(str(exc))


def apply_downloaded_update(package_path: Path) -> tuple[bool, str]:
    """Install the downloaded update package and relaunch the app when running as a packaged binary."""
    appimage_path = os.environ.get("APPIMAGE")
    if appimage_path and package_path.name.lower().endswith(".appimage"):
        target = Path(appimage_path)
        tmp_new = target.with_suffix(".AppImage.new")
        shutil.copy2(package_path, tmp_new)
        tmp_new.chmod(tmp_new.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        tmp_new.replace(target)
        subprocess.Popen([str(target)], start_new_session=True)
        return True, "Relaunching updated AppImage..."

    if not getattr(sys, "frozen", False):
        return False, (
            f"Downloaded update archive to:\n{package_path}\n\n"
            "You are running from source. Pull the latest repository changes or extract the release package."
        )

    pid = os.getpid()
    staging = Path(tempfile.mkdtemp(prefix="dips_update_"))
    if package_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(package_path, "r") as zf:
            zf.extractall(staging)
    elif package_path.name.lower().endswith(".tar.gz"):
        import tarfile
        with tarfile.open(package_path, "r:gz") as tf:
            tf.extractall(staging)
    else:
        return False, f"Downloaded installer to:\n{package_path}"

    system = platform.system()
    exe_path = Path(sys.executable).resolve()
    if system == "Windows":
        app_dir = exe_path.parent
        src_dir = staging / "DIPS" if (staging / "DIPS").is_dir() else staging
        try:
            for src_file in src_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel = src_file.relative_to(src_dir)
                dst_file = app_dir / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_file, dst_file)
                except PermissionError:
                    old_file = dst_file.with_name(f"{dst_file.name}.old")
                    if old_file.exists():
                        try:
                            old_file.unlink()
                        except OSError:
                            old_file = dst_file.with_name(f"{dst_file.name}.{pid}.old")
                    dst_file.rename(old_file)
                    shutil.copy2(src_file, dst_file)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return False, (
            "Update files have been installed.\n\n"
            "Click OK to close DIPS, then reopen it to launch the updated version."
        )

    if system == "Darwin":
        # exe_path is .../DIPS.app/Contents/MacOS/DIPS
        app_bundle = exe_path.parents[2] if exe_path.parents[2].name.endswith(".app") else exe_path.parent
        new_app = next(staging.glob("*.app"), None)
        if new_app is None:
            return False, f"Update extracted to {staging}"
        sh_script = staging / "apply_update.sh"
        sh_script.write_text(
            "#!/bin/sh\n"
            f"while kill -0 {pid} 2>/dev/null; do sleep 0.3; done\n"
            f"rm -rf \"{app_bundle}\"\n"
            f"mv \"{new_app}\" \"{app_bundle}\"\n"
            f"open \"{app_bundle}\"\n",
            encoding="utf-8",
        )
        sh_script.chmod(0o755)
        subprocess.Popen(["/bin/sh", str(sh_script)], start_new_session=True)
        return True, "Applying macOS update and restarting..."

    # Frozen Linux tarball
    app_dir = exe_path.parent
    src_dir = staging / "DIPS" if (staging / "DIPS").is_dir() else staging
    sh_script = staging / "apply_update.sh"
    sh_script.write_text(
        "#!/bin/sh\n"
        f"while kill -0 {pid} 2>/dev/null; do sleep 0.3; done\n"
        f"cp -rf \"{src_dir}\"/* \"{app_dir}/\"\n"
        f"chmod +x \"{exe_path}\"\n"
        f"\"{exe_path}\" &\n",
        encoding="utf-8",
    )
    sh_script.chmod(0o755)
    subprocess.Popen(["/bin/sh", str(sh_script)], start_new_session=True)
    return True, "Applying Linux update and restarting..."


class SelfUpdateDialog(QDialog):
    """Progress dialog downloading and installing a GitHub Release package."""

    def __init__(self, version: str, asset_name: str, download_url: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Updating DIPS")
        self.setMinimumWidth(420)
        self._restarting = False

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        self._lbl = QLabel(f"Downloading DIPS v{version} ({asset_name})...")
        lay.addWidget(self._lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        lay.addWidget(self._bar)

        row = QHBoxLayout()
        row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        row.addWidget(self._cancel_btn)
        lay.addLayout(row)

        dest = Path(tempfile.gettempdir()) / asset_name
        self._dl = UpdateDownloadThread(download_url, dest)
        self._dl.finished.connect(self._dl.deleteLater)
        self._dl.progress.connect(self._bar.setValue)
        self._dl.finishedDownload.connect(self._on_downloaded)
        self._dl.failed.connect(self._on_failed)
        self._dl.start()

    def _cancel_download(self) -> None:
        self._dl.cancel()
        for sig in (self._dl.progress, self._dl.finishedDownload, self._dl.failed):
            try:
                sig.disconnect()
            except Exception:
                pass
        if self._dl.isRunning():
            self._dl.wait(250)

    def reject(self) -> None:
        self._cancel_download()
        super().reject()

    def closeEvent(self, event) -> None:
        self._cancel_download()
        super().closeEvent(event)

    def _on_downloaded(self, path_str: str) -> None:
        self._lbl.setText("Installing update...")
        QApplication.processEvents()
        try:
            will_restart, msg = apply_downloaded_update(Path(path_str))
            if not will_restart:
                QMessageBox.information(self, "Update Ready", msg)
            self._restarting = will_restart or (
                platform.system() == "Windows" and bool(getattr(sys, "frozen", False))
            )
            self.accept()
        except Exception as exc:
            self._on_failed(str(exc))

    def _on_failed(self, err: str) -> None:
        QMessageBox.warning(self, "Update Failed", f"Automatic update failed:\n{err}")
        self.reject()

    @property
    def should_quit(self) -> bool:
        return self._restarting
