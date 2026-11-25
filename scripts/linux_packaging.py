"""Linux packaging helpers for DIPS: .desktop entry, .deb builder, .tar.gz, and .AppImage."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.request

PACKAGE_NAME = "dips"
PACKAGE_REVISION = "1"
MAINTAINER = "Deniz Şafak <denizsafak98@gmail.com>"
HOMEPAGE = "https://github.com/denizsafak/dips"
SECTION = "graphics"
PRIORITY = "optional"
INSTALL_ROOT = "opt/dips"
BINARY_NAME = "DIPS"
COMMAND_NAME = "dips"
DESKTOP_FILE_NAME = "dips.desktop"

BASE_DEPENDS = (
    "libc6",
    "libdrm2",
    "libegl1",
    "libgl1",
    "libwayland-client0",
    "libwayland-egl1",
    "libxcb1",
    "zlib1g",
)
RECOMMENDS = ("fonts-dejavu-core", "xdg-utils")

_TAR_COMPRESS_LEVEL = 6
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def normalize_version(version: str) -> str:
    cleaned = version.strip().lstrip("vV")
    if not cleaned:
        raise ValueError("Empty version")
    return cleaned


def build_desktop_entry(exec_target: str, *, include_appimage_keys: bool = False, app_version: str = "1.0.1") -> str:
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=DIPS",
        "GenericName=Digital Image Processing Studio",
        "Comment=Non-destructive image processing, FFT, morphology, and feature analysis studio",
        f"Exec={exec_target} %F",
        "Icon=dips",
        "Terminal=false",
        "Categories=Graphics;Science;ImageProcessing;",
        "MimeType=image/png;image/jpeg;image/bmp;image/tiff;image/webp;",
        "StartupWMClass=DIPS",
    ]
    if include_appimage_keys:
        lines.append(f"X-AppImage-Version={app_version}")
    return "\n".join(lines) + "\n"


def _minimum_libc_version() -> str | None:
    libc_name, libc_ver = platform.libc_ver()
    parts = libc_ver.split(".")
    if libc_name == "glibc" and len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
        return ".".join(parts[:2])
    return None


def _dependency_field() -> str:
    deps = list(BASE_DEPENDS)
    libc_ver = _minimum_libc_version()
    if libc_ver:
        deps[0] = f"libc6 (>= {libc_ver})"
    return ", ".join(deps)


def _iter_stage(root: str):
    root = os.path.abspath(root)

    def walk(directory: str):
        with os.scandir(directory) as it:
            for entry in sorted(it, key=lambda e: e.name):
                rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
                if entry.is_symlink():
                    yield rel, entry.path, "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    yield rel, entry.path, "dir"
                    yield from walk(entry.path)
                else:
                    yield rel, entry.path, "file"

    yield from walk(root)


def _installed_size_kb(stage: str) -> int:
    total = 0
    for _rel, abspath, kind in _iter_stage(stage):
        if kind == "file":
            total += os.path.getsize(abspath)
    return max(1, (total + 1023) // 1024)


def _md5sums(stage: str) -> str:
    lines: list[str] = []
    for rel, abspath, kind in _iter_stage(stage):
        if kind == "file":
            digest = hashlib.md5(Path(abspath).read_bytes()).hexdigest()
            lines.append(f"{digest}  {rel}")
    return ("\n".join(lines) + "\n") if lines else ""


def _add_tar_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: int, mode: int) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = mtime
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    import io
    tar.addfile(info, io.BytesIO(data))


def _add_staged_member(tar: tarfile.TarFile, relpath: str, abspath: str, kind: str, mtime: int) -> None:
    info = tarfile.TarInfo(name=f"./{relpath}")
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    if kind == "dir":
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        tar.addfile(info)
    elif kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = os.readlink(abspath)
        info.mode = 0o777
        tar.addfile(info)
    else:
        st = os.lstat(abspath)
        info.size = st.st_size
        info.mode = 0o755 if (st.st_mode & 0o111) else 0o644
        with open(abspath, "rb") as fh:
            tar.addfile(info, fh)


def _write_ar_member(handle, name: str, size: int, mtime: int, *, source: str | None = None, data: bytes | None = None) -> None:
    header = f"{name:<16}{mtime:<12}{0:<6}{0:<6}{0o100644:<8o}{size:<10}`\n"
    handle.write(header.encode("ascii"))
    if data is not None:
        handle.write(data)
    elif source is not None:
        with open(source, "rb") as fh:
            shutil.copyfileobj(fh, handle, 1024 * 1024)
    if size % 2:
        handle.write(b"\n")


def build_deb(repo_root: str | Path, version: str, architecture: str = "amd64", output_dir: str | Path | None = None) -> Path:
    """Build DIPS-linux-<arch>.deb from dist/DIPS bundle."""
    if not re.fullmatch(r"[a-z0-9]+", architecture):
        raise ValueError(f"Invalid Debian architecture: {architecture!r}")

    root = Path(repo_root).resolve()
    bundle_dir = root / "dist" / BINARY_NAME
    exe = bundle_dir / BINARY_NAME
    icon_png = root / "src" / "dips" / "assets" / "dips.png"
    if not exe.is_file():
        raise FileNotFoundError(f"PyInstaller bundle executable missing: {exe}")
    if not icon_png.is_file():
        raise FileNotFoundError(f"Icon missing: {icon_png}")

    norm_ver = normalize_version(version)
    deb_ver = f"{norm_ver}-{PACKAGE_REVISION}"
    out_dir = Path(output_dir or root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mtime = int(time.time())

    stage_dir = root / "dist" / "deb" / f"{PACKAGE_NAME}_{norm_ver}_{architecture}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    try:
        install_dir = stage_dir / INSTALL_ROOT
        shutil.copytree(bundle_dir, install_dir, symlinks=True)
        os.chmod(install_dir / BINARY_NAME, 0o755)

        bin_dir = stage_dir / "usr" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(f"../../{INSTALL_ROOT}/{BINARY_NAME}", bin_dir / COMMAND_NAME)

        apps_dir = stage_dir / "usr" / "share" / "applications"
        apps_dir.mkdir(parents=True, exist_ok=True)
        (apps_dir / DESKTOP_FILE_NAME).write_text(build_desktop_entry(f"/usr/bin/{COMMAND_NAME}"), encoding="utf-8")

        icon_dir = stage_dir / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
        icon_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon_png, icon_dir / "dips.png")

        doc_dir = stage_dir / "usr" / "share" / "doc" / PACKAGE_NAME
        doc_dir.mkdir(parents=True, exist_ok=True)
        license_path = root / "LICENSE"
        if license_path.is_file():
            shutil.copy2(license_path, doc_dir / "copyright")

        tm = time.gmtime(mtime)
        rfc_date = f"{_DAYS[tm.tm_wday]}, {tm.tm_mday:02d} {_MONTHS[tm.tm_mon - 1]} {tm.tm_year} {tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d} +0000"
        changelog = (
            f"{PACKAGE_NAME} ({deb_ver}) unstable; urgency=low\n\n"
            f"  * Release {norm_ver}.\n\n"
            f" -- {MAINTAINER}  {rfc_date}\n"
        ).encode("utf-8")
        with open(doc_dir / "changelog.Debian.gz", "wb") as raw_cl:
            with gzip.GzipFile(fileobj=raw_cl, mode="wb", compresslevel=6, mtime=mtime) as gz:
                gz.write(changelog)

        control_txt = (
            f"Package: {PACKAGE_NAME}\n"
            f"Version: {deb_ver}\n"
            f"Section: {SECTION}\n"
            f"Priority: {PRIORITY}\n"
            f"Architecture: {architecture}\n"
            f"Maintainer: {MAINTAINER}\n"
            f"Homepage: {HOMEPAGE}\n"
            f"Depends: {_dependency_field()}\n"
            f"Recommends: {', '.join(RECOMMENDS)}\n"
            f"Installed-Size: {_installed_size_kb(str(stage_dir))}\n"
            "Description: cross-platform digital image processing studio\n"
            " DIPS is a non-destructive desktop studio for spatial filtering,\n"
            " 2D FFT frequency analysis, morphology, restoration, color grading,\n"
            " and feature detection.\n"
        ).encode("utf-8")

        postinst = (
            "#!/bin/sh\nset -e\n"
            "if command -v update-desktop-database >/dev/null 2>&1; then\n"
            "  update-desktop-database -q /usr/share/applications || true\n"
            "fi\n"
            "if command -v gtk-update-icon-cache >/dev/null 2>&1; then\n"
            "  gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true\n"
            "fi\n"
        ).encode("utf-8")

        control_tar = root / "dist" / "deb" / "control.tar.gz"
        data_tar = root / "dist" / "deb" / "data.tar.gz"

        with open(control_tar, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=_TAR_COMPRESS_LEVEL, mtime=mtime) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                    _add_tar_bytes(tar, "./control", control_txt, mtime, 0o644)
                    _add_tar_bytes(tar, "./md5sums", _md5sums(str(stage_dir)).encode("utf-8"), mtime, 0o644)
                    _add_tar_bytes(tar, "./postinst", postinst, mtime, 0o755)
                    _add_tar_bytes(tar, "./postrm", postinst, mtime, 0o755)

        with open(data_tar, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=_TAR_COMPRESS_LEVEL, mtime=mtime) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                    for relpath, abspath, kind in _iter_stage(str(stage_dir)):
                        _add_staged_member(tar, relpath, abspath, kind, mtime)

        deb_path = out_dir / f"DIPS-linux-{architecture}.deb"
        with open(deb_path, "wb") as fh:
            fh.write(b"!<arch>\n")
            _write_ar_member(fh, "debian-binary", 4, mtime, data=b"2.0\n")
            _write_ar_member(fh, "control.tar.gz", control_tar.stat().st_size, mtime, source=str(control_tar))
            _write_ar_member(fh, "data.tar.gz", data_tar.stat().st_size, mtime, source=str(data_tar))
        return deb_path
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def build_tarball(repo_root: str | Path, architecture: str = "amd64", output_dir: str | Path | None = None) -> Path:
    """Package dist/DIPS bundle into DIPS-linux-<arch>.tar.gz."""
    root = Path(repo_root).resolve()
    bundle_dir = root / "dist" / BINARY_NAME
    out_dir = Path(output_dir or root).resolve()
    out_path = out_dir / f"DIPS-linux-{architecture}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=BINARY_NAME)
    return out_path


def build_appimage(
    repo_root: str | Path,
    version: str,
    architecture: str = "amd64",
    output_dir: str | Path | None = None,
) -> Path:
    """Create DIPS-linux-<arch>.AppImage using appimagetool."""
    root = Path(repo_root).resolve()
    bundle_dir = root / "dist" / BINARY_NAME
    icon_png = root / "src" / "dips" / "assets" / "dips.png"
    out_dir = Path(output_dir or root).resolve()

    tool = shutil.which("appimagetool") or shutil.which("appimagetool-x86_64.AppImage")
    if not tool:
        dl_dir = root / "build" / "tools"
        dl_dir.mkdir(parents=True, exist_ok=True)
        tool_path = dl_dir / "appimagetool-x86_64.AppImage"
        if not tool_path.is_file():
            url = "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
            urllib.request.urlretrieve(url, tool_path)
            tool_path.chmod(tool_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        tool = str(tool_path)

    appdir = root / "dist" / "DIPS.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    appdir.mkdir(parents=True)

    usr_bin = appdir / "usr" / "bin"
    shutil.copytree(bundle_dir, usr_bin, symlinks=True)
    os.chmod(usr_bin / BINARY_NAME, 0o755)

    (appdir / DESKTOP_FILE_NAME).write_text(
        build_desktop_entry(BINARY_NAME, include_appimage_keys=True, app_version=normalize_version(version)),
        encoding="utf-8",
    )
    shutil.copy2(icon_png, appdir / "dips.png")
    shutil.copy2(icon_png, appdir / ".DirIcon")

    apprun = appdir / "AppRun"
    apprun.write_text(
        "#!/bin/sh\n"
        'HERE="$(dirname "$(readlink -f "$0")")"\n'
        'export PATH="${HERE}/usr/bin:${PATH}"\n'
        f'exec "${{HERE}}/usr/bin/{BINARY_NAME}" "$@"\n',
        encoding="utf-8",
    )
    apprun.chmod(0o755)

    out_appimage = out_dir / f"DIPS-linux-{architecture}.AppImage"
    env = os.environ.copy()
    env["ARCH"] = "x86_64" if architecture == "amd64" else architecture
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"
    subprocess.run([tool, str(appdir), str(out_appimage)], check=True, env=env)
    return out_appimage
