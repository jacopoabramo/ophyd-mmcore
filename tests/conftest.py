"""Shared fixtures for ophyd-mmcore tests."""

from __future__ import annotations

from typing import Iterator

import pytest
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore import get_worker
from ophyd_mmcore._base import _worker_cache
from ophyd_mmcore._worker import MMCoreWorker


@pytest.fixture
def core() -> Iterator[CMMCorePlus]:
    """Yield a CMMCorePlus instance loaded with the MM demo configuration."""
    instance = CMMCorePlus()
    instance.loadSystemConfiguration()
    yield instance


@pytest.fixture
def worker(core: CMMCorePlus) -> Iterator[MMCoreWorker]:
    """Yield the shared MMCoreWorker for the demo core."""
    w = get_worker(core)
    yield w


@pytest.fixture(autouse=True)
def _cleanup_workers() -> Iterator[None]:
    """Stop and evict all cached workers after each test."""
    yield
    for w in list(_worker_cache.values()):
        w.stop()
    _worker_cache.clear()
