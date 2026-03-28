"""Tests for MMZStageMixin and MMXYStageMixin.

Move completion is simulated by controlling a ``deviceBusy`` side-effect
that returns True for the first N polls then False, while firing position
events on a background task to verify WatcherUpdate delivery.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Annotated as A
from typing import Any
from unittest.mock import patch

from pymmcore_plus.core._constants import PropertyType

import pytest
from bluesky.protocols import Location
from bluesky.run_engine import RunEngine, TransitionError
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.core import WatcherUpdate
from ophyd_async.plan_stubs import ensure_connected
from pymmcore_plus import CMMCorePlus

import bluesky.plan_stubs as bps

from ophyd_mmcore import PropName, XPositionMethod, YPositionMethod
from ophyd_mmcore._backend import MMPropertyBackend
from ophyd_mmcore._connector import CoreMethod
from ophyd_mmcore._devices import MMXYStage, MMZStage
from ophyd_mmcore._worker import MMCoreWorker


# ---------------------------------------------------------------------------
# Concrete device classes under test
# ---------------------------------------------------------------------------


class ZStage(MMZStage):
    """Z stage with an extra config property (e.g. speed)."""

    velocity: A[float, PropName("Speed"), Format.CONFIG_SIGNAL]


XYStage = MMXYStage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def core_z() -> Generator[CMMCorePlus, None, None]:
    """CMMCorePlus with a simulated Z stage at position 0.0."""
    instance = CMMCorePlus()
    _position = {"z": 0.0}

    def _get_pos(label: str) -> float:
        return _position["z"]

    def _set_pos(label: str, value: float) -> None:
        _position["z"] = value

    def _get_property_type(device: str, prop: str) -> Any:
        return PropertyType.Float

    def _get_property(device: str, prop: str) -> str:
        return {"Speed": "1.0"}.get(prop, "")

    with (
        patch.object(instance, "getPosition", side_effect=_get_pos),
        patch.object(instance, "setPosition", side_effect=_set_pos),
        patch.object(instance, "deviceBusy", return_value=False),
        patch.object(instance, "stop"),
        patch.object(instance, "getPropertyType", side_effect=_get_property_type),
        patch.object(instance, "getProperty", side_effect=_get_property),
        patch.object(instance, "setProperty"),
        patch.object(instance, "isPropertyReadOnly", return_value=False),
        patch.object(instance, "hasPropertyLimits", return_value=False),
        patch.object(instance, "getAllowedPropertyValues", return_value=()),
    ):
        yield instance


@pytest.fixture
def core_xy() -> Generator[CMMCorePlus, None, None]:
    """CMMCorePlus with a simulated XY stage at position (0.0, 0.0)."""
    instance = CMMCorePlus()
    _position = {"x": 0.0, "y": 0.0}

    def _get_x(label: str) -> float:
        return _position["x"]

    def _get_y(label: str) -> float:
        return _position["y"]

    def _set_xy(label: str, x: float, y: float) -> None:
        _position["x"] = x
        _position["y"] = y

    with (
        patch.object(instance, "getXPosition", side_effect=_get_x),
        patch.object(instance, "getYPosition", side_effect=_get_y),
        patch.object(instance, "setXYPosition", side_effect=_set_xy),
        patch.object(instance, "deviceBusy", return_value=False),
        patch.object(instance, "stop"),
        # no extra properties on bare XY stage
        patch.object(instance, "getPropertyType"),
        patch.object(instance, "getProperty", return_value=""),
        patch.object(instance, "setProperty"),
        patch.object(instance, "isPropertyReadOnly", return_value=False),
        patch.object(instance, "hasPropertyLimits", return_value=False),
        patch.object(instance, "getAllowedPropertyValues", return_value=()),
    ):
        yield instance


@pytest.fixture
def worker_z(core_z: CMMCorePlus) -> Generator[MMCoreWorker, None, None]:
    w = MMCoreWorker(core_z)
    yield w
    w.stop()


@pytest.fixture
def worker_xy(core_xy: CMMCorePlus) -> Generator[MMCoreWorker, None, None]:
    w = MMCoreWorker(core_xy)
    yield w
    w.stop()


@pytest.fixture
def RE() -> Generator[RunEngine, None, None]:
    import asyncio

    loop = asyncio.new_event_loop()
    re = RunEngine({}, call_returns_result=True, loop=loop)
    yield re
    if re.state not in ("idle", "panicked"):
        try:
            re.halt()
        except TransitionError:
            pass
    loop.call_soon_threadsafe(loop.stop)
    re._th.join()
    loop.close()


# ---------------------------------------------------------------------------
# Z stage tests
# ---------------------------------------------------------------------------


async def test_z_locate_returns_current_position(core_z: CMMCorePlus) -> None:
    """locate() returns the current Z position as setpoint and readback."""
    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    loc: Location[float] = await stage.locate()
    assert loc["setpoint"] == pytest.approx(0.0)
    assert loc["readback"] == pytest.approx(0.0)


async def test_z_set_calls_set_position(core_z: CMMCorePlus) -> None:
    """set() calls core.setPosition with the right label and value."""
    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    await stage.set(100.0)

    core_z.setPosition.assert_called_with("ZDrive", 100.0)  # type: ignore[attr-defined]


async def test_z_set_updates_locate_after_move(core_z: CMMCorePlus) -> None:
    """After set(), locate() reflects the new position."""
    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    await stage.set(250.0)
    loc = await stage.locate()
    assert loc["readback"] == pytest.approx(250.0)


async def test_z_set_emits_watcher_updates(core_z: CMMCorePlus) -> None:
    """set() emits at least one WatcherUpdate before completing."""
    # Make deviceBusy return True once, then False, to force one polling cycle
    core_z.deviceBusy.side_effect = [True, False]  # type: ignore[attr-defined]

    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    updates: list[WatcherUpdate[float]] = []

    def _collect(**kwargs: object) -> None:
        updates.append(WatcherUpdate(**kwargs))  # type: ignore[arg-type]

    status = stage.set(100.0)
    status.watch(_collect)
    await status

    assert len(updates) >= 1
    last = updates[-1]
    assert last.target == pytest.approx(100.0)
    assert last.initial == pytest.approx(0.0)


async def test_z_stop_calls_core_stop(core_z: CMMCorePlus) -> None:
    """stop() calls core.stop with the device label."""
    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    await stage.stop(success=True)

    core_z.stop.assert_called_with("ZDrive")  # type: ignore[attr-defined]


async def test_z_stop_marks_move_as_failed(core_z: CMMCorePlus) -> None:
    """stop(success=False) causes the in-progress set() to raise."""
    core_z.deviceBusy.side_effect = lambda label: True  # type: ignore[attr-defined]

    stage = ZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)

    move_status = stage.set(100.0)

    # Let the move start, then stop it
    await asyncio.sleep(0.02)
    await stage.stop(success=False)

    with pytest.raises(Exception):
        await move_status


# ---------------------------------------------------------------------------
# XY stage tests
# ---------------------------------------------------------------------------


async def test_xy_locate_returns_current_position(core_xy: CMMCorePlus) -> None:
    """locate() returns the current (x, y) position."""
    stage = XYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)

    loc: Location[tuple[float, float]] = await stage.locate()
    assert loc["setpoint"] == (pytest.approx(0.0), pytest.approx(0.0))
    assert loc["readback"] == (pytest.approx(0.0), pytest.approx(0.0))


async def test_xy_set_calls_set_xy_position(core_xy: CMMCorePlus) -> None:
    """set((x, y)) calls core.setXYPosition with the right arguments."""
    stage = XYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)

    await stage.set((100.0, 200.0))

    core_xy.setXYPosition.assert_called_with("XYDrive", 100.0, 200.0)  # type: ignore[attr-defined]


async def test_xy_set_updates_locate_after_move(core_xy: CMMCorePlus) -> None:
    """After set(), locate() reflects the new XY position."""
    stage = XYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)

    await stage.set((50.0, 75.0))
    loc = await stage.locate()
    assert loc["readback"] == (pytest.approx(50.0), pytest.approx(75.0))


async def test_xy_set_emits_watcher_updates(core_xy: CMMCorePlus) -> None:
    """set() emits WatcherUpdates, and the final target matches the requested position."""
    core_xy.deviceBusy.side_effect = [True, False]  # type: ignore[attr-defined]

    stage = XYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)

    updates: list[WatcherUpdate[tuple[float, float]]] = []

    def _collect(**kwargs: object) -> None:
        updates.append(WatcherUpdate(**kwargs))  # type: ignore[arg-type]

    status = stage.set((100.0, 200.0))
    status.watch(_collect)
    await status

    assert len(updates) >= 1
    assert updates[-1].target == (pytest.approx(100.0), pytest.approx(200.0))


async def test_xy_stop_calls_core_stop(core_xy: CMMCorePlus) -> None:
    """stop() calls core.stop with the device label."""
    stage = XYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)

    await stage.stop(success=True)

    core_xy.stop.assert_called_with("XYDrive")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Plan integration tests
# ---------------------------------------------------------------------------


def test_z_mv_in_plan(RE: RunEngine, core_z: CMMCorePlus) -> None:
    """bps.mv works with a Z stage inside a plan."""
    stage = ZStage("ZDrive", core_z, name="z")

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(stage)
        yield from bps.mv(stage, 150.0)

    RE(plan())
    core_z.setPosition.assert_called_with("ZDrive", 150.0)  # type: ignore[attr-defined]


def test_xy_mv_in_plan(RE: RunEngine, core_xy: CMMCorePlus) -> None:
    """bps.mv works with an XY stage inside a plan."""
    stage = XYStage("XYDrive", core_xy, name="xy")

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(stage)
        yield from bps.mv(stage, (300.0, 400.0))

    RE(plan())
    core_xy.setXYPosition.assert_called_with("XYDrive", 300.0, 400.0)  # type: ignore[attr-defined]


def test_z_locate_in_plan(RE: RunEngine, core_z: CMMCorePlus) -> None:
    """bps.locate returns the current Z position inside a plan."""
    stage = ZStage("ZDrive", core_z, name="z")
    result: list[Location[float]] = []

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(stage)
        loc = yield from bps.locate(stage)
        result.append(loc)

    RE(plan())
    assert result[0]["readback"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# XPositionMethod / YPositionMethod signal tests
# ---------------------------------------------------------------------------


def _make_xy_backend(axis_name: str, method: CoreMethod, worker: MMCoreWorker) -> MMPropertyBackend[float]:
    backend: MMPropertyBackend[float] = MMPropertyBackend("XY", axis_name, worker, datatype=float)
    backend.configure_core_method(method.get, method.set, method.event, method.event_value)
    return backend


@pytest.mark.asyncio
async def test_x_position_method_get_returns_x(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """XPositionMethod.get reads the X axis, not Y."""
    core_xy.setXYPosition("XY", 10.0, 20.0)
    backend = _make_xy_backend("XPosition", XPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)
    assert await backend.get_value() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_y_position_method_get_returns_y(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """YPositionMethod.get reads the Y axis, not X."""
    core_xy.setXYPosition("XY", 10.0, 20.0)
    backend = _make_xy_backend("YPosition", YPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)
    assert await backend.get_value() == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_x_position_method_put_preserves_y(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """XPositionMethod.set moves X while keeping the current Y unchanged."""
    core_xy.setXYPosition("XY", 0.0, 20.0)
    backend = _make_xy_backend("XPosition", XPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)
    await backend.put(50.0)
    core_xy.setXYPosition.assert_called_with("XY", 50.0, 20.0)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_y_position_method_put_preserves_x(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """YPositionMethod.set moves Y while keeping the current X unchanged."""
    core_xy.setXYPosition("XY", 30.0, 0.0)
    backend = _make_xy_backend("YPosition", YPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)
    await backend.put(75.0)
    core_xy.setXYPosition.assert_called_with("XY", 30.0, 75.0)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_x_position_method_callback_extracts_x(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """XPositionMethod event callback delivers X, not Y, from XYStagePositionChanged."""
    backend = _make_xy_backend("XPosition", XPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)

    received: list[float] = []
    backend.set_callback(lambda r: received.append(r["value"]))
    await asyncio.sleep(0.05)  # consume initial delivery
    received.clear()

    core_xy.events.XYStagePositionChanged.emit("XY", 111.0, 222.0)
    await asyncio.sleep(0.05)
    backend.set_callback(None)

    assert received == [pytest.approx(111.0)]


@pytest.mark.asyncio
async def test_y_position_method_callback_extracts_y(
    worker_xy: MMCoreWorker, core_xy: CMMCorePlus
) -> None:
    """YPositionMethod event callback delivers Y, not X, from XYStagePositionChanged."""
    backend = _make_xy_backend("YPosition", YPositionMethod, worker_xy)
    await backend.connect(timeout=5.0)

    received: list[float] = []
    backend.set_callback(lambda r: received.append(r["value"]))
    await asyncio.sleep(0.05)
    received.clear()

    core_xy.events.XYStagePositionChanged.emit("XY", 111.0, 222.0)
    await asyncio.sleep(0.05)
    backend.set_callback(None)

    assert received == [pytest.approx(222.0)]


# ---------------------------------------------------------------------------
# MMZStage.position and MMXYStage.x/y device-level signal tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_z_stage_position_signal_reads_position(core_z: CMMCorePlus) -> None:
    """MMZStage.position reads the current Z position."""
    core_z.setPosition("ZDrive", 42.0)
    stage = MMZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)
    assert await stage.position.get_value() == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_z_stage_position_signal_set_moves_z(core_z: CMMCorePlus) -> None:
    """MMZStage.position.set() calls setPosition on the core."""
    stage = MMZStage("ZDrive", core_z, name="z")
    await stage.connect(mock=False)
    await stage.position.set(99.0)
    core_z.setPosition.assert_called_with("ZDrive", 99.0)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_xy_stage_x_y_signals_read_correct_axes(core_xy: CMMCorePlus) -> None:
    """MMXYStage.x and .y read from the correct axes."""
    core_xy.setXYPosition("XYDrive", 10.0, 20.0)
    stage = MMXYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)
    assert await stage.x.get_value() == pytest.approx(10.0)
    assert await stage.y.get_value() == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_xy_stage_x_signal_set_preserves_y(core_xy: CMMCorePlus) -> None:
    """MMXYStage.x.set() moves X while keeping the current Y."""
    core_xy.setXYPosition("XYDrive", 0.0, 30.0)
    stage = MMXYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)
    await stage.x.set(50.0)
    core_xy.setXYPosition.assert_called_with("XYDrive", 50.0, 30.0)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_xy_stage_y_signal_set_preserves_x(core_xy: CMMCorePlus) -> None:
    """MMXYStage.y.set() moves Y while keeping the current X."""
    core_xy.setXYPosition("XYDrive", 40.0, 0.0)
    stage = MMXYStage("XYDrive", core_xy, name="xy")
    await stage.connect(mock=False)
    await stage.y.set(80.0)
    core_xy.setXYPosition.assert_called_with("XYDrive", 40.0, 80.0)  # type: ignore[attr-defined]