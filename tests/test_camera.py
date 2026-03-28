"""Tests for MMCamera — triggerable and flyable interfaces.

Tests exercise:
  - MMTriggerLogic: frame count storage, exposure setting
  - MMArmLogic: arm/disarm sequence acquisition
  - MMZarrDataLogic: zarr store creation and cleanup
  - MMCamera (full): step-scan trigger and fly-scan kickoff/complete
  - ZarrStore: shared store with mixed MM and simulated data logics
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import acquire_zarr as zarr
import numpy as np
import pytest
from ophyd_async.core import DetectorDataLogic, TriggerInfo
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore._camera import (
    MMArmLogic,
    MMCamera,
    MMTriggerLogic,
    MMZarrDataLogic,
    MMZarrStreamProvider,
    ZarrStore,
)
from ophyd_mmcore._worker import MMCoreWorker


# ---------------------------------------------------------------------------
# Simulated data logic (pure-Python, no MM dependency)
# ---------------------------------------------------------------------------


class _SimDataLogic(DetectorDataLogic):
    """Minimal data logic that writes synthetic frames into a shared ZarrStore.

    Generates zero-filled uint16 frames at ~50 ms intervals, gated on the
    store being open.  Designed to share a ZarrStore with MMZarrDataLogic.
    """

    def __init__(
        self, store: ZarrStore, array_key: str, shape: tuple[int, int]
    ) -> None:
        self._store = store
        self._array_key = array_key
        self._shape = shape
        self._provider: MMZarrStreamProvider | None = None
        self._task: asyncio.Task[None] | None = None

    async def prepare_unbounded(self, datakey_name: str) -> MMZarrStreamProvider:
        await self.stop()
        h, w = self._shape
        dtype = np.dtype("uint16")
        self._store.register_array(
            self._array_key,
            zarr.ArraySettings(
                output_key=self._array_key,
                data_type=zarr.DataType.UINT16,
                dimensions=[
                    zarr.Dimension(
                        name="t",
                        kind=zarr.DimensionType.TIME,
                        array_size_px=0,
                        chunk_size_px=1,
                        shard_size_chunks=1,
                    ),
                    zarr.Dimension(
                        name="y",
                        kind=zarr.DimensionType.SPACE,
                        array_size_px=h,
                        chunk_size_px=h,
                        shard_size_chunks=1,
                    ),
                    zarr.Dimension(
                        name="x",
                        kind=zarr.DimensionType.SPACE,
                        array_size_px=w,
                        chunk_size_px=w,
                        shard_size_chunks=1,
                    ),
                ],
            ),
        )
        self._provider = MMZarrStreamProvider(
            store=self._store,
            array_key=self._array_key,
            datakey_name=datakey_name,
            dtype=dtype,
            width=w,
            height=h,
        )
        self._task = asyncio.create_task(self._generate_loop())
        return self._provider

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _generate_loop(self) -> None:
        assert self._provider is not None
        frame = np.zeros(self._shape, dtype=np.uint16)
        frames_written = 0
        while True:
            if self._store.is_open:
                self._store.append(frame, key=self._array_key)
                frames_written += 1
                await self._provider._frames_written.set(frames_written)
            await asyncio.sleep(0.05)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "cam.zarr"


@pytest.fixture
def cam(core: CMMCorePlus, store_path: Path) -> MMCamera:
    return MMCamera("Camera", core, store_path, name="cam")


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


@pytest.fixture
def arm_logic(worker: MMCoreWorker, store_path: Path) -> MMArmLogic:
    """MMArmLogic wired to a store with one dummy array registered."""
    store = ZarrStore(store_path)
    # Register a minimal array so the store can open successfully.
    store.register_array(
        "frames",
        zarr.ArraySettings(
            output_key="frames",
            data_type=zarr.DataType.UINT16,
            dimensions=[
                zarr.Dimension(name="t", kind=zarr.DimensionType.TIME, array_size_px=0, chunk_size_px=1, shard_size_chunks=1),
                zarr.Dimension(name="y", kind=zarr.DimensionType.SPACE, array_size_px=512, chunk_size_px=512, shard_size_chunks=1),
                zarr.Dimension(name="x", kind=zarr.DimensionType.SPACE, array_size_px=512, chunk_size_px=512, shard_size_chunks=1),
            ],
        ),
    )
    trigger = MMTriggerLogic("Camera", worker)
    trigger._n_frames = 10
    return MMArmLogic("Camera", worker, trigger, store)


@pytest.mark.asyncio
async def test_arm_starts_sequence(
    arm_logic: MMArmLogic, core: CMMCorePlus
) -> None:
    await arm_logic.arm()
    assert core.isSequenceRunning("Camera")
    await arm_logic.disarm()


@pytest.mark.asyncio
async def test_disarm_stops_sequence(
    arm_logic: MMArmLogic, core: CMMCorePlus
) -> None:
    await arm_logic.arm()
    await arm_logic.disarm()
    assert not core.isSequenceRunning("Camera")


# ---------------------------------------------------------------------------
# MMZarrDataLogic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zarr_data_logic_creates_store(
    worker: MMCoreWorker, store_path: Path
) -> None:
    store = ZarrStore(store_path)
    logic = MMZarrDataLogic(store, "frames", "Camera", worker)
    await logic.prepare_unbounded("cam")
    store.open()
    assert store_path.exists()
    await logic.stop()
    store.close()


@pytest.mark.asyncio
async def test_zarr_data_logic_stop_is_idempotent(
    worker: MMCoreWorker, store_path: Path
) -> None:
    store = ZarrStore(store_path)
    logic = MMZarrDataLogic(store, "frames", "Camera", worker)
    await logic.prepare_unbounded("cam")
    await logic.stop()
    await logic.stop()  # must not raise


@pytest.mark.asyncio
async def test_shared_store_registers_multiple_arrays(
    worker: MMCoreWorker, store_path: Path
) -> None:
    """Two MMZarrDataLogic instances sharing one store write to separate arrays."""
    store = ZarrStore(store_path)
    logic_a = MMZarrDataLogic(store, "ch0", "Camera", worker)
    logic_b = MMZarrDataLogic(store, "ch1", "Camera", worker)

    # Both logics register their arrays before the stream opens.
    await logic_a.prepare_unbounded("cam-ch0")
    await logic_b.prepare_unbounded("cam-ch1")

    assert "ch0" in store._array_settings
    assert "ch1" in store._array_settings

    # Stream opens with both arrays.
    store.open()
    assert store.is_open
    assert store_path.exists()

    await logic_a.stop()
    await logic_b.stop()
    store.close()
    assert not store.is_open
    # After close, array settings are cleared ready for the next acquisition.
    assert not store._array_settings


@pytest.mark.asyncio
async def test_shared_store_open_is_idempotent(
    worker: MMCoreWorker, store_path: Path
) -> None:
    """Calling store.open() multiple times only creates the stream once."""
    store = ZarrStore(store_path)
    logic = MMZarrDataLogic(store, "frames", "Camera", worker)
    await logic.prepare_unbounded("cam")

    store.open()
    stream_id = id(store._stream)
    store.open()  # must not recreate the stream
    assert id(store._stream) == stream_id

    await logic.stop()
    store.close()


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


# ---------------------------------------------------------------------------
# ZarrStore — mixed MM and simulated data logics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_store_mm_and_sim(
    worker: MMCoreWorker, store_path: Path, core: CMMCorePlus
) -> None:
    """ZarrStore accepts concurrent writes from an MM camera and a simulated source."""
    store = ZarrStore(store_path)
    n = 3

    mm_logic = MMZarrDataLogic(store, "mm_frames", "Camera", worker)
    sim_logic = _SimDataLogic(store, "sim_frames", shape=(512, 512))

    mm_provider = await mm_logic.prepare_unbounded("cam")
    sim_provider = await sim_logic.prepare_unbounded("sim")

    assert "mm_frames" in store._array_settings
    assert "sim_frames" in store._array_settings

    store.open()
    core.startSequenceAcquisition("Camera", n, 0, False)
    while core.isSequenceRunning("Camera"):
        await asyncio.sleep(0.02)

    # Give both drain loops time to flush pending frames
    await asyncio.sleep(0.2)

    mm_frames = await mm_provider._frames_written.get_value()
    sim_frames = await sim_provider._frames_written.get_value()

    await mm_logic.stop()
    await sim_logic.stop()
    store.close()

    assert store_path.exists()
    assert mm_frames == n
    assert sim_frames > 0
