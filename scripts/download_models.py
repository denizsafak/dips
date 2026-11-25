"""Build-time helper to fetch all AI ONNX models into src/dips/models/."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dips.core.model_manager import download_all_models


if __name__ == "__main__":
    download_all_models(ROOT / "src" / "dips" / "models")
