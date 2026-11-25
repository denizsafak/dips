"""Icon and static asset helpers for DIPS."""

from pathlib import Path
import sys


def asset_path(*parts: str) -> Path:
    """Resolve an asset file path in source or PyInstaller frozen bundle."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidate = Path(base) / "dips" / "assets" / Path(*parts)
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.joinpath(*parts)
