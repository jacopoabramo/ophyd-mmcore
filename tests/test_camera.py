"""Tests for MMCamera — triggerable and flyable interfaces.

Tests exercise:
  - MMTriggerLogic: frame count storage, exposure setting
  - MMArmLogic: arm/disarm sequence acquisition
  - MMZarrDataLogic: zarr store creation and cleanup
  - MMCamera (full): step-scan trigger and fly-scan kickoff/complete
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ophyd_async.core import TriggerInfo
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore._camera import MMArmLogic, MMCamera, MMTriggerLogic, MMZarrDataLogic
from ophyd_mmcore._worker import MMCoreWorker


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "cam.zarr"


@pytest.fixture
def cam(worker: MMCoreWorker, store_path: Path) -> MMCamera:
    return MMCamera("Camera", worker, store_path, name="cam")


# ---------------------------------------------------------------------------
# MMTriggerLogic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_logic_stores_frame_count(worker: MMCoreWorker) -> None:
    logic = MMTriggerLogic("Camera", worker)
    await logic.prepare_internal(num=5, livetime=0.0, deadtime=0.0)
    assert logic._n_frames == 5


@pytest.mark.asyncio
async def test_trigger_logic_sets_exposure(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    logic = MMTriggerLogic("Camera", worker)
    await logic.prepare_internal(num=1, livetime=0.05, deadtime=0.0)
    assert float(core.getProperty("Camera", "Exposure")) == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_trigger_logic_skips_exposure_when_zero(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    original = float(core.getProperty("Camera", "Exposure"))
    logic = MMTriggerLogic("Camera", worker)
    await logic.prepare_internal(num=1, livetime=0.0, deadtime=0.0)
    assert float(core.getProperty("Camera", "Exposure")) == pytest.approx(original)


# ---------------------------------------------------------------------------
# MMArmLogic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arm_starts_sequence(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    trigger = MMTriggerLogic("Camera", worker)
    trigger._n_frames = 10
    arm = MMArmLogic("Camera", worker, trigger)
    await arm.arm()
    assert core.isSequenceRunning("Camera")
    await arm.disarm()


@pytest.mark.asyncio
async def test_disarm_stops_sequence(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    trigger = MMTriggerLogic("Camera", worker)
    trigger._n_frames = 10
    arm = MMArmLogic("Camera", worker, trigger)
    await arm.arm()
    await arm.disarm()
    assert not core.isSequenceRunning("Camera")


# ---------------------------------------------------------------------------
# MMZarrDataLogic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zarr_data_logic_creates_store(
    worker: MMCoreWorker, store_path: Path
) -> None:
    logic = MMZarrDataLogic(store_path, "Camera", worker)
    await logic.prepare_unbounded("cam")
    assert store_path.exists()
    await logic.stop()


@pytest.mark.asyncio
async def test_zarr_data_logic_stop_is_idempotent(
    worker: MMCoreWorker, store_path: Path
) -> None:
    logic = MMZarrDataLogic(store_path, "Camera", worker)
    await logic.prepare_unbounded("cam")
    await logic.stop()
    await logic.stop()  # must not raise


# ---------------------------------------------------------------------------
# MMCamera — step scan (trigger)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_camera_trigger_acquires_one_frame(cam: MMCamera) -> None:
    """trigger() arms, acquires one frame, and reports index 1."""
    await cam.connect()
    await cam.stage()
    await cam.trigger()
    assert await cam.get_index() == 1
    await cam.unstage()


@pytest.mark.asyncio
async def test_camera_trigger_emits_stream_docs(cam: MMCamera) -> None:
    """collect_asset_docs() yields stream_resource then stream_datum."""
    await cam.connect()
    await cam.stage()
    await cam.trigger()

    doc_names = [name async for name, _ in cam.collect_asset_docs()]
    assert "stream_resource" in doc_names
    assert "stream_datum" in doc_names
    await cam.unstage()


# ---------------------------------------------------------------------------
# MMCamera — fly scan (kickoff / complete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_camera_flyable_acquires_n_frames(cam: MMCamera) -> None:
    """kickoff/complete acquires the requested number of frames."""
    n = 3
    await cam.connect()
    await cam.stage()
    await cam.prepare(TriggerInfo(number_of_events=n))
    await cam.kickoff()
    await cam.complete()
    assert await cam.get_index() == n
    await cam.unstage()


@pytest.mark.asyncio
async def test_camera_flyable_stream_datum_covers_all_frames(cam: MMCamera) -> None:
    """stream_datum indices stop == number of frames acquired."""
    n = 4
    await cam.connect()
    await cam.stage()
    await cam.prepare(TriggerInfo(number_of_events=n))
    await cam.kickoff()
    await cam.complete()

    indices_stop = 0
    async for name, doc in cam.collect_asset_docs():
        if name == "stream_datum":
            indices_stop = doc["indices"]["stop"]
    assert indices_stop == n
    await cam.unstage()
