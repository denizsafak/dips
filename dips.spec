# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller specification for DIPS (Windows, macOS, Linux)."""

from pathlib import Path
import os
import platform
import tomllib
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH)
SYSTEM = platform.system()
IS_MACOS = SYSTEM == "Darwin"

VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
EXE_NAME = os.environ.get("DIPS_EXE_NAME", "DIPS")
ONEFILE = os.environ.get("DIPS_ONEFILE", "0") == "1"

datas = [
    *copy_metadata("dips"),
    (str(ROOT / "src" / "dips" / "assets" / "dips.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "dips.png"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "dips.ico"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "dips.icns"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "chevron_dark.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "chevron_light.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "chevron_up_dark.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "chevron_up_light.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "assets" / "check_white.svg"), "dips/assets"),
    (str(ROOT / "src" / "dips" / "models" / "*.onnx"), "dips/models"),
    (str(ROOT / "src" / "dips" / "samples" / "*"), "dips/samples"),
]

binaries = collect_dynamic_libs("onnxruntime")
hidden = collect_submodules("dips") + ["cv2", "numpy", "PIL", "onnxruntime"]


def _icon_path() -> str:
    if IS_MACOS:
        return str(ROOT / "src" / "dips" / "assets" / "dips.icns")
    return str(ROOT / "src" / "dips" / "assets" / "dips.ico")


a = Analysis(
    [str(ROOT / "scripts" / "pyinstaller_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas"],
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE and not IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=EXE_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=_icon_path(),
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=EXE_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=_icon_path(),
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=EXE_NAME,
    )
    if IS_MACOS:
        app = BUNDLE(
            coll,
            name=f"{EXE_NAME}.app",
            icon=str(ROOT / "src" / "dips" / "assets" / "dips.icns"),
            bundle_identifier="com.denizsafak.dips",
            info_plist={
                "CFBundleName": "DIPS",
                "CFBundleDisplayName": "DIPS",
                "CFBundleExecutable": EXE_NAME,
                "CFBundleIdentifier": "com.denizsafak.dips",
                "CFBundleShortVersionString": VERSION,
                "CFBundleVersion": VERSION,
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "12.0",
                "CFBundleDocumentTypes": [
                    {
                        "CFBundleTypeName": "Image Document",
                        "CFBundleTypeRole": "Editor",
                        "LSItemContentTypes": [
                            "public.png",
                            "public.jpeg",
                            "public.tiff",
                            "com.microsoft.bmp",
                            "org.webmproject.webp",
                        ],
                    }
                ],
            },
        )
