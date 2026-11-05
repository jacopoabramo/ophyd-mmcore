import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import FakeCamera
from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    wait_for_value,
)
from ophyd_async.testing import MonitorQueue
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore

from ophyd_mmcore import CameraState, MMCamera


async def connected_camera(
    core: CMMCorePlus, label: str, tmp_path: Path, **options
) -> MMCamera:
    """Return the camera `label` on `core`, connected."""
    camera = MMCamera(
        label,
        core,
        path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        name=label.lower(),
        **options,
    )
    await camera.connect()
    return camera


async def stop(camera: MMCamera) -> None:
    """Close the camera's gate, stop its sequence and close its stores."""
    await camera.capture.set(False)
    await camera.acquire.set(False)
    for output in camera.outputs.values():
        await output.file_path.set("")


@pytest.fixture
async def cam(uni_core: UniMMCore, tmp_path: Path) -> AsyncIterator[MMCamera]:
    """Return the camera `Cam` with its `Gain` property, stopped after the test."""
    camera = await connected_camera(uni_core, "Cam", tmp_path, properties=["Gain"])
    yield camera
    await stop(camera)


@pytest.fixture
async def cam2(uni_core: UniMMCore, tmp_path: Path) -> AsyncIterator[MMCamera]:
    """Return the camera `Cam2`, stopped after the test."""
    camera = await connected_camera(uni_core, "Cam2", tmp_path)
    yield camera
    await stop(camera)


@pytest.fixture
def pair_core(uni_core: UniMMCore) -> UniMMCore:
    """Return the `uni_core` fixture with one more camera, `Pair`."""
    uni_core.loadPyDevice("Pair", FakeCamera())
    uni_core.initializeDevice("Pair")
    return uni_core


async def start_capture(cam: MMCamera, path: Path, num_capture: int) -> None:
    out = cam.outputs["cam"]
    await out.file_path.set(str(path))
    await out.data_key.set("cam")
    await cam.acquire.set(True)
    await cam.num_capture.set(num_capture)
    await cam.capture.set(True)


async def wait_until(condition) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


def raising_append(self, frame: np.ndarray) -> None:
    raise OSError("disk full")


async def test_bounded_capture_writes_n_frames(cam, tmp_path):
    """Write exactly num_capture frames, then close the gate."""
    out = cam.outputs["cam"]
    await start_capture(cam, tmp_path / "a.zarr", 5)
    await wait_for_value(cam.capture, False, timeout=5)
    assert await out.captured.get_value() == 5
    await out.file_path.set("")
    assert zarr.open(str(tmp_path / "a.zarr"))["cam"].shape == (5, 16, 24)  # type: ignore[index]


async def test_store_stays_open_across_gates(cam, tmp_path):
    """Append every capture to the same store until file_path is cleared."""
    out = cam.outputs["cam"]
    await start_capture(cam, tmp_path / "a.zarr", 2)
    await wait_for_value(cam.capture, False, timeout=5)
    await cam.capture.set(True)
    await wait_for_value(cam.capture, False, timeout=5)
    assert await out.captured.get_value() == 4
    await out.file_path.set("")
    assert zarr.open(str(tmp_path / "a.zarr"))["cam"].shape == (4, 16, 24)  # type: ignore[index]


async def test_layout_reports_frame_shape_and_dtype(cam):
    """Report the width, height and dtype of the frames the camera sends."""
    out = cam.outputs["cam"]
    assert await out.width.get_value() == 24
    assert await out.height.get_value() == 16
    assert await out.dtype_numpy.get_value() == "<u2"


async def test_write_while_acquiring_pauses(cam, uni_core):
    """Change a property the adapter refuses mid-sequence while acquiring."""
    gain = cam.properties["gain"]
    await cam.acquire.set(True)
    await gain.set(4)
    assert await gain.get_value() == 4
    assert uni_core.isSequenceRunning("Cam")


async def test_write_while_capturing_is_refused(cam, tmp_path):
    """Refuse an exposure change during a capture."""
    await start_capture(cam, tmp_path / "a.zarr", 0)
    with pytest.raises(RuntimeError, match="Cam"):
        await cam.exposure.set(20.0)


async def test_exposure_and_state_updates_reach_subscribers(cam):
    """Send exposure writes and state changes to subscribers."""
    with MonitorQueue(cam.exposure) as exposure:
        await exposure.assert_updates(10.0)
        await cam.exposure.set(20.0)
        await exposure.assert_updates(20.0)
    with MonitorQueue(cam.state) as state:
        await state.assert_updates(CameraState.IDLE)
        await cam.acquire.set(True)
        await state.assert_updates(CameraState.ACQUIRING)


async def test_acquire_sets_and_restores_core_state(cam, uni_core):
    """Make the camera current with AutoShutter off while acquiring, and restore both after."""
    uni_core.setCameraDevice("Cam2")
    await cam.acquire.set(True)
    assert uni_core.getCameraDevice() == "Cam" and uni_core.getAutoShutter() is False
    await cam.acquire.set(False)
    assert uni_core.getCameraDevice() == "Cam2" and uni_core.getAutoShutter() is True


async def test_cameras_on_one_core_acquire_one_at_a_time(uni_core, tmp_path):
    """Refuse a second camera's sequence while another camera built on the same core acquires."""
    provider = StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    first = MMCamera("Cam", uni_core, path_provider=provider, name="first")
    second = MMCamera("Cam2", uni_core, path_provider=provider, name="second")
    await first.connect()
    await second.connect()
    await first.acquire.set(True)
    try:
        with pytest.raises(RuntimeError, match="'Cam'"):
            await second.acquire.set(True)
    finally:
        await first.acquire.set(False)


async def test_existing_store_is_refused(cam, tmp_path):
    """Refuse to open a store over an existing path."""
    (tmp_path / "a.zarr").mkdir()
    with pytest.raises(FileExistsError, match="a.zarr"):
        await cam.outputs["cam"].file_path.set(str(tmp_path / "a.zarr"))


async def test_stop_while_capturing_closes_the_store(cam, cam2, uni_core, tmp_path):
    """Keep every captured frame when acquiring stops mid-capture, and free the core."""
    out = cam.outputs["cam"]
    await start_capture(cam, tmp_path / "a.zarr", 0)
    await wait_for_value(out.captured, lambda n: n > 0, timeout=5)
    await cam.acquire.set(False)
    await out.file_path.set("")
    captured = await out.captured.get_value()
    assert zarr.open(str(tmp_path / "a.zarr"))["cam"].shape[0] == captured  # type: ignore[index]
    assert uni_core.getAutoShutter() is True
    await cam2.acquire.set(True)


async def test_store_change_while_capturing_is_refused(cam, tmp_path):
    """Refuse to replace the store during a capture."""
    await start_capture(cam, tmp_path / "a.zarr", 0)
    with pytest.raises(RuntimeError, match="Cam"):
        await cam.outputs["cam"].file_path.set(str(tmp_path / "b.zarr"))


async def test_lower_num_capture_closes_the_gate(cam, tmp_path):
    """Close the gate when num_capture drops to frames already written."""
    await start_capture(cam, tmp_path / "a.zarr", 0)
    await wait_for_value(cam.outputs["cam"].captured, lambda n: n >= 2, timeout=5)
    await cam.num_capture.set(1)
    await wait_for_value(cam.capture, False, timeout=1)


async def test_cancelled_acquire_can_be_stopped(cam, cam2, uni_core, monkeypatch):
    """Stop a camera whose acquire put was cancelled, and let another camera acquire."""
    entered = threading.Event()
    set_auto_shutter = uni_core.setAutoShutter

    def slow_set_auto_shutter(state: bool) -> None:
        entered.set()
        time.sleep(0.1)
        set_auto_shutter(state)

    monkeypatch.setattr(uni_core, "setAutoShutter", slow_set_auto_shutter)
    uni_core.setCameraDevice("Cam2")
    status = cam.acquire.set(True)
    await asyncio.to_thread(entered.wait, 5)
    status.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await status
    assert await cam.acquire.get_value() is True
    await cam.acquire.set(False)
    await cam2.acquire.set(True)
    assert uni_core.isSequenceRunning("Cam2")


async def test_colour_camera_is_refused(rgb_core, tmp_path):
    """Refuse to open a store for a camera with several components per pixel."""
    rgb = await connected_camera(rgb_core, "Rgb", tmp_path)
    with pytest.raises(ValueError, match="PixelType|RGB"):
        await rgb.outputs["rgb"].file_path.set(str(tmp_path / "a.zarr"))


async def test_outputs_with_different_layouts_are_refused(pair_core, tmp_path):
    """Refuse to open a store when the outputs of one camera differ in frame size."""
    pair_core.setROI("Cam2", 0, 0, 8, 8)
    pair = await connected_camera(pair_core, "Pair", tmp_path, physical=("Cam", "Cam2"))
    with pytest.raises(ValueError, match="Cam2"):
        await pair.outputs["cam"].file_path.set(str(tmp_path / "a.zarr"))


async def test_camera_with_other_physical_cameras_is_refused(pair_core, tmp_path):
    """Refuse to connect a second camera on one label with different physical cameras."""
    await connected_camera(pair_core, "Pair", tmp_path, physical=("Cam", "Cam2"))
    second = MMCamera(
        "Pair",
        pair_core,
        path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        physical=("Cam",),
        name="second",
    )
    with pytest.raises(ValueError, match="Pair"):
        await second.connect()


async def test_write_error_faults_the_camera(cam, uni_core, monkeypatch, tmp_path):
    """Fault the camera and stop its sequence when a frame cannot be written."""
    monkeypatch.setattr("ophyd_mmcore._camera_io.FrameStore.append", raising_append)
    await start_capture(cam, tmp_path / "a.zarr", 3)
    await wait_for_value(cam.capture, False, timeout=5)
    assert await cam.state.get_value() == CameraState.FAULTED
    assert "disk full" in await cam.last_error.get_value()
    await wait_until(lambda: not uni_core.isSequenceRunning("Cam"))


async def test_buffer_overflow_faults_the_camera(cam, uni_core, monkeypatch):
    """Fault the camera and stop its sequence when the circular buffer overflows."""
    overflowed = threading.Event()
    monkeypatch.setattr(uni_core, "isBufferOverflowed", overflowed.is_set)
    await cam.acquire.set(True)
    overflowed.set()
    await wait_for_value(cam.state, CameraState.FAULTED, timeout=5)
    assert "overflowed" in await cam.last_error.get_value()
    await wait_until(lambda: not uni_core.isSequenceRunning("Cam"))


async def test_cancelled_change_logs_its_error(cam, uni_core, monkeypatch, caplog):
    """Log the error of a camera change whose caller was cancelled."""
    entered = threading.Event()

    def failing_start(*args) -> None:
        entered.set()
        time.sleep(0.1)
        raise RuntimeError("camera unplugged")

    monkeypatch.setattr(uni_core, "startSequenceAcquisition", failing_start)
    status = cam.acquire.set(True)
    await asyncio.to_thread(entered.wait, 5)
    status.task.cancel()
    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await status
    assert "camera unplugged" in caplog.text
