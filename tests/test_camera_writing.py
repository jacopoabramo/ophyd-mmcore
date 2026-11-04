from collections.abc import AsyncIterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from ophyd_async.core import SignalR, SignalRW, wait_for_value
from ophyd_async.testing import MonitorQueue

from ophyd_mmcore import CameraState, MMController
from ophyd_mmcore._backend import MMSignalBackend


@dataclass
class OutputSignals:
    width: SignalR[int]
    height: SignalR[int]
    dtype_numpy: SignalR[str]
    file_path: SignalRW[str]
    data_key: SignalRW[str]
    captured: SignalR[int]


@dataclass
class CameraSignals:
    exposure: SignalRW[float]
    acquire: SignalRW[bool]
    capture: SignalRW[bool]
    num_capture: SignalRW[int]
    state: SignalR[CameraState]
    last_error: SignalR[str]
    out: OutputSignals


async def connect_all(signals: object) -> None:
    for field in fields(signals):  # type: ignore[arg-type]
        if field.name != "out":
            await getattr(signals, field.name).connect()


async def camera_signals(ctrl: MMController, label: str) -> CameraSignals:
    io = ctrl.camera_io(label)

    def sig(cls, datatype, source) -> Any:
        return cls(MMSignalBackend(datatype, source), name=source.uri)

    out = OutputSignals(
        width=sig(SignalR, int, io.output_source(label, "width")),
        height=sig(SignalR, int, io.output_source(label, "height")),
        dtype_numpy=sig(SignalR, str, io.output_source(label, "dtype_numpy")),
        file_path=sig(SignalRW, str, io.output_source(label, "file_path")),
        data_key=sig(SignalRW, str, io.output_source(label, "data_key")),
        captured=sig(SignalR, int, io.output_source(label, "captured")),
    )
    await connect_all(out)
    signals = CameraSignals(
        exposure=sig(SignalRW, float, io.source("exposure")),
        acquire=sig(SignalRW, bool, io.source("acquire")),
        capture=sig(SignalRW, bool, io.source("capture")),
        num_capture=sig(SignalRW, int, io.source("num_capture")),
        state=sig(SignalR, CameraState, io.source("state")),
        last_error=sig(SignalR, str, io.source("last_error")),
        out=out,
    )
    await connect_all(signals)
    return signals


@pytest.fixture
async def cam_signals(ctrl: MMController) -> AsyncIterator[CameraSignals]:
    yield await camera_signals(ctrl, "Cam")


async def start_capture(s: CameraSignals, path: Path, num_capture: int) -> None:
    await s.out.file_path.set(str(path))
    await s.out.data_key.set("cam")
    await s.acquire.set(True)
    await s.num_capture.set(num_capture)
    await s.capture.set(True)


def raising_append(self, frame: np.ndarray) -> None:
    raise OSError("disk full")


async def test_bounded_capture_writes_n_frames(cam_signals, tmp_path):
    """Write exactly num_capture frames, then close the gate."""
    s, out = cam_signals, cam_signals.out
    await start_capture(s, tmp_path / "a.zarr", 5)
    await wait_for_value(s.capture, False, timeout=5)
    assert await out.captured.get_value() == 5
    await out.file_path.set("")
    assert zarr.open(str(tmp_path / "a.zarr"))["cam"].shape == (5, 16, 24)  # type: ignore[index]


async def test_store_stays_open_across_gates(cam_signals, tmp_path):
    """Append every capture to the same store until file_path is cleared."""
    s, out = cam_signals, cam_signals.out
    await start_capture(s, tmp_path / "a.zarr", 2)
    await wait_for_value(s.capture, False, timeout=5)
    await s.capture.set(True)
    await wait_for_value(s.capture, False, timeout=5)
    assert await out.captured.get_value() == 4
    await out.file_path.set("")
    assert zarr.open(str(tmp_path / "a.zarr"))["cam"].shape == (4, 16, 24)  # type: ignore[index]


async def test_layout_reports_frame_shape_and_dtype(cam_signals):
    """Report the width, height and dtype of the frames the camera sends."""
    out = cam_signals.out
    assert await out.width.get_value() == 24
    assert await out.height.get_value() == 16
    assert await out.dtype_numpy.get_value() == "<u2"


async def test_write_while_acquiring_pauses(cam_signals, ctrl):
    """Change a property the adapter refuses mid-sequence while acquiring."""
    gain = SignalRW(
        MMSignalBackend(int, ctrl.property_source("Cam", "Gain")), name="gain"
    )
    await gain.connect()
    await cam_signals.acquire.set(True)
    await gain.set(4)
    assert await gain.get_value() == 4
    assert ctrl.core.isSequenceRunning("Cam")


async def test_write_while_capturing_is_refused(cam_signals, tmp_path):
    """Refuse an exposure change during a capture."""
    await start_capture(cam_signals, tmp_path / "a.zarr", 0)
    with pytest.raises(RuntimeError, match="Cam"):
        await cam_signals.exposure.set(20.0)


async def test_exposure_and_state_updates_reach_subscribers(cam_signals):
    """Send exposure writes and state changes to subscribers."""
    with MonitorQueue(cam_signals.exposure) as exposure:
        await exposure.assert_updates(10.0)
        await cam_signals.exposure.set(20.0)
        await exposure.assert_updates(20.0)
    with MonitorQueue(cam_signals.state) as state:
        await state.assert_updates(CameraState.IDLE)
        await cam_signals.acquire.set(True)
        await state.assert_updates(CameraState.ACQUIRING)


async def test_acquire_sets_and_restores_core_state(cam_signals, uni_core):
    """Make the camera current with AutoShutter off while acquiring, and restore both after."""
    uni_core.setCameraDevice("Cam2")
    await cam_signals.acquire.set(True)
    assert uni_core.getCameraDevice() == "Cam" and uni_core.getAutoShutter() is False
    await cam_signals.acquire.set(False)
    assert uni_core.getCameraDevice() == "Cam2" and uni_core.getAutoShutter() is True


async def test_second_sequence_on_core_is_refused(cam_signals, ctrl):
    """Refuse a second acquiring camera on the same core, and allow it once the first stops."""
    cam2 = SignalRW(
        MMSignalBackend(bool, ctrl.camera_io("Cam2").source("acquire")), name="acquire2"
    )
    await cam2.connect()
    await cam_signals.acquire.set(True)
    with pytest.raises(RuntimeError, match="'Cam'"):
        await cam2.set(True)
    await cam_signals.acquire.set(False)
    await cam2.set(True)


async def test_existing_store_is_refused(cam_signals, tmp_path):
    """Refuse to open a store over an existing path."""
    (tmp_path / "a.zarr").mkdir()
    with pytest.raises(FileExistsError, match="a.zarr"):
        await cam_signals.out.file_path.set(str(tmp_path / "a.zarr"))


async def test_close_while_capturing_cleans_up(cam_signals, ctrl, uni_core, tmp_path):
    """Stop, close the store and restore AutoShutter when the controller closes."""
    await start_capture(cam_signals, tmp_path / "a.zarr", 0)
    await wait_for_value(cam_signals.out.captured, lambda n: n > 0, timeout=5)
    ctrl.close()
    ctrl.close()
    assert not uni_core.isSequenceRunning("Cam") and uni_core.getAutoShutter() is True
    zarr.open(str(tmp_path / "a.zarr"))


async def test_colour_camera_is_refused(ctrl_with_rgb, tmp_path):
    """Refuse to open a store for a camera with several components per pixel."""
    source = ctrl_with_rgb.camera_io("Rgb").output_source("Rgb", "file_path")
    file_path = SignalRW(MMSignalBackend(str, source), name="file_path")
    await file_path.connect()
    with pytest.raises(ValueError, match="PixelType|RGB"):
        await file_path.set(str(tmp_path / "a.zarr"))


async def test_outputs_with_different_layouts_are_refused(ctrl, uni_core, tmp_path):
    """Refuse to open a store when the outputs of one camera differ in frame size."""
    uni_core.setROI("Cam2", 0, 0, 8, 8)
    source = ctrl.camera_io("Pair", ("Cam", "Cam2")).output_source("Cam", "file_path")
    file_path = SignalRW(MMSignalBackend(str, source), name="file_path")
    await file_path.connect()
    with pytest.raises(ValueError, match="Cam2"):
        await file_path.set(str(tmp_path / "a.zarr"))


def test_camera_io_refuses_other_physical_cameras(ctrl):
    """Refuse a second camera_io call for a label with different physical cameras."""
    ctrl.camera_io("Pair", ("Cam", "Cam2"))
    with pytest.raises(ValueError, match="Pair"):
        ctrl.camera_io("Pair", ("Cam",))


async def test_write_error_faults_the_camera(cam_signals, monkeypatch, tmp_path):
    """Fault the camera with the error message when a frame cannot be written."""
    monkeypatch.setattr("ophyd_mmcore._camera_io.FrameStore.append", raising_append)
    await start_capture(cam_signals, tmp_path / "a.zarr", 3)
    await wait_for_value(cam_signals.capture, False, timeout=5)
    assert await cam_signals.state.get_value() == CameraState.FAULTED
    assert "disk full" in await cam_signals.last_error.get_value()
