"""Shared fixtures for ophyd-mmcore tests."""

from __future__ import annotations

from typing import Iterator

import pytest
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore._worker import MMCoreWorker


@pytest.fixture
def core() -> Iterator[CMMCorePlus]:
    """Yield a CMMCorePlus instance loaded with the MM demo configuration."""
    instance = CMMCorePlus()
    instance.loadSystemConfiguration()
    yield instance


@pytest.fixture
def worker(core: CMMCorePlus) -> Iterator[MMCoreWorker]:
    """Yield an MMCoreWorker wrapping the demo core."""
    w = MMCoreWorker(core)
    yield w
    w.stop()
