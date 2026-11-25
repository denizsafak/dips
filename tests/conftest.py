"""Pytest session configuration and clean C++ extension teardown handling."""

from __future__ import annotations

import os
import sys
import pytest

_exit_status: int | None = None


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    if _exit_status is None:
        return
    ops_ai = sys.modules.get("dips.core.ops_ai")
    if ops_ai is not None and hasattr(ops_ai, "release_ai_sessions"):
        ops_ai.release_ai_sessions()
    if "onnxruntime" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status)
