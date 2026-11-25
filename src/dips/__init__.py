"""DIPS - Digital Image Processing Studio."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import tomllib


def _read_version() -> str:
    try:
        return _pkg_version("dips")
    except PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            ver = data.get("project", {}).get("version")
            if isinstance(ver, str) and ver.strip():
                return ver.strip()
        except Exception:
            pass
    return "1.0.1"


__app_name__ = "DIPS"
__version__ = _read_version()
__author__ = "Deniz Şafak"
__email__ = "denizsafak98@gmail.com"
__repo_slug__ = "denizsafak/dips"
