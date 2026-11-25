"""Cross-platform build & packaging script for DIP Studio (Windows, macOS, Linux)."""

from __future__ import annotations

import argparse
import os
import importlib.util
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile


def read_version(root: Path) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"]).strip()


def arch_tag() -> str:
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m


def run_pyinstaller(root: Path, exe_name: str = "DIPS", *, onefile: bool = False) -> None:
    env = os.environ.copy()
    env["DIPS_EXE_NAME"] = exe_name
    env["DIPS_ONEFILE"] = "1" if onefile else "0"
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(root / "dips.spec")]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=root, env=env, check=True)


def zip_directory(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() or p.is_symlink():
                zf.write(p, p.relative_to(src_dir.parent))


def package_windows(root: Path, exe_name: str = "DIPS") -> Path:
    bundle = root / "dist" / exe_name
    out_zip = root / f"{exe_name}-windows-{arch_tag()}.zip"
    if bundle.is_dir():
        zip_directory(bundle, out_zip)
    else:
        exe = root / "dist" / f"{exe_name}.exe"
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(exe, exe.name)
    print("Created:", out_zip)
    return out_zip


def package_macos(root: Path, exe_name: str = "DIPS") -> Path:
    app_bundle = root / "dist" / f"{exe_name}.app"
    if not app_bundle.is_dir():
        raise FileNotFoundError(f"macOS bundle not found: {app_bundle}")
    # Ad-hoc codesign so Gatekeeper/local launches work cleanly
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app_bundle)], check=False)
    out_zip = root / f"{exe_name}-macos-{arch_tag()}.zip"
    subprocess.run(
        ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app_bundle), str(out_zip)],
        check=True,
    )
    print("Created:", out_zip)
    return out_zip


def package_linux(root: Path, version: str, *, skip_appimage: bool = False) -> list[Path]:
    sys.path.insert(0, str(root / "scripts"))
    import linux_packaging

    arch = arch_tag()
    outputs: list[Path] = []
    deb = linux_packaging.build_deb(root, version, architecture=arch, output_dir=root)
    print("Created:", deb)
    outputs.append(deb)

    tar_path = linux_packaging.build_tarball(root, architecture=arch, output_dir=root)
    print("Created:", tar_path)
    outputs.append(tar_path)

    if not skip_appimage:
        appimage = linux_packaging.build_appimage(root, version, architecture=arch, output_dir=root)
        print("Created:", appimage)
        outputs.append(appimage)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and package DIP Studio for the current platform.")
    parser.add_argument("--output-name", default="DIPS", help="Base executable/bundle name (default: DIPS)")
    parser.add_argument("--onefile", action="store_true", help="Build a single-file binary (Windows/Linux)")
    parser.add_argument("--skip-appimage", action="store_true", help="Skip AppImage generation on Linux")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    version = read_version(root)
    system = platform.system()

    spec = importlib.util.spec_from_file_location("dips_model_manager", root / "src" / "dips" / "core" / "model_manager.py")
    assert spec is not None and spec.loader is not None
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    mm.download_all_models(root / "src" / "dips" / "models")

    run_pyinstaller(root, args.output_name, onefile=args.onefile)

    if system == "Windows":
        package_windows(root, args.output_name)
    elif system == "Darwin":
        package_macos(root, args.output_name)
    elif system == "Linux" and not args.onefile:
        package_linux(root, version, skip_appimage=args.skip_appimage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
