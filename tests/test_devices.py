"""Tests for all non-stage MMDevice mixins.

Each mixin gets a fixture with a patched CMMCorePlus and tests covering:
  - locate() returns current state
  - set() calls the correct core method
  - stop() halts an in-progress operation
  - trigger() (where applicable)
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from ophyd_async.core import WatcherUpdate
from pymmcore_plus import CMMCorePlus


from ophyd_mmcore._devices import (
    MMAutoFocus,
    MMGalvo,
    MMPump,
    MMShutter,
    MMStateDevice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_patches(instance: CMMCorePlus):
    """Return an ExitStack with all property-level MM patches applied."""
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(instance, "getPropertyType"))
    stack.enter_context(patch.object(instance, "getProperty", return_value=""))
    stack.enter_context(patch.object(instance, "setProperty"))
    stack.enter_context(patch.object(instance, "isPropertyReadOnly", return_value=False))
    stack.enter_context(patch.object(instance, "hasPropertyLimits", return_value=False))
    stack.enter_context(patch.object(instance, "getAllowedPropertyValues", return_value=()))
    stack.enter_context(patch.object(instance, "deviceBusy", return_value=False))
    return stack


def _watcher_collector() -> tuple[list[WatcherUpdate[Any]], Any]:
    """Return (updates list, watch callable) for status.watch()."""
    updates: list[WatcherUpdate[Any]] = []

    def _collect(**kwargs: Any) -> None:
        updates.append(WatcherUpdate(**kwargs))  # type: ignore[arg-type]

    return updates, _collect


# ---------------------------------------------------------------------------
# Shutter
# ---------------------------------------------------------------------------


MyShutter = MMShutter


@pytest.fixture
def core_shutter() -> Generator[CMMCorePlus, None, None]:
    instance = CMMCorePlus()
    _state = {"open": False}

    with _base_patches(instance) as stack:
        for cm in [patch.object(instance, "getShutterOpen", side_effect=lambda lbl: _state["open"]),
        patch.object(instance, "setShutterOpen", side_effect=lambda lbl, v: _state.update({"open": v})),
        patch.object(instance, "stop")]:
            stack.enter_context(cm)
        yield instance


async def test_shutter_locate(core_shutter: CMMCorePlus) -> None:
    s = MyShutter("Shutter", core_shutter, name="shutter")
    await s.connect(mock=False)
    loc = await s.locate()
    assert loc["readback"] is False


async def test_shutter_set_opens(core_shutter: CMMCorePlus) -> None:
    s = MyShutter("Shutter", core_shutter, name="shutter")
    await s.connect(mock=False)
    await s.set(True)
    core_shutter.setShutterOpen.assert_called_with("Shutter", True)  # type: ignore[attr-defined]


async def test_shutter_locate_reflects_set(core_shutter: CMMCorePlus) -> None:
    s = MyShutter("Shutter", core_shutter, name="shutter")
    await s.connect(mock=False)
    await s.set(True)
    loc = await s.locate()
    assert loc["readback"] is True


async def test_shutter_emits_watcher_updates(core_shutter: CMMCorePlus) -> None:
    core_shutter.deviceBusy.side_effect = [True, False]  # type: ignore[attr-defined]
    s = MyShutter("Shutter", core_shutter, name="shutter")
    await s.connect(mock=False)
    updates, collect = _watcher_collector()
    status = s.set(True)
    status.watch(collect)
    await status
    assert len(updates) >= 1
    assert updates[-1].target is True


async def test_shutter_stop(core_shutter: CMMCorePlus) -> None:
    core_shutter.deviceBusy.side_effect = lambda lbl: True  # type: ignore[attr-defined]
    s = MyShutter("Shutter", core_shutter, name="shutter")
    await s.connect(mock=False)
    move = s.set(True)
    await asyncio.sleep(0.02)
    await s.stop(success=False)
    with pytest.raises(Exception):
        await move


# ---------------------------------------------------------------------------
# State device
# ---------------------------------------------------------------------------


MyFilterWheel = MMStateDevice


@pytest.fixture
def core_state() -> Generator[CMMCorePlus, None, None]:
    instance = CMMCorePlus()
    _state = {"label": "DAPI"}

    with _base_patches(instance) as stack:
        for cm in [patch.object(instance, "getStateLabel", side_effect=lambda lbl: _state["label"]),
        patch.object(instance, "setStateLabel", side_effect=lambda lbl, v: _state.update({"label": v})),
        patch.object(instance, "stop")]:
            stack.enter_context(cm)
        yield instance


async def test_state_locate(core_state: CMMCorePlus) -> None:
    fw = MyFilterWheel("FilterWheel", core_state, name="fw")
    await fw.connect(mock=False)
    loc = await fw.locate()
    assert loc["readback"] == "DAPI"


async def test_state_set(core_state: CMMCorePlus) -> None:
    fw = MyFilterWheel("FilterWheel", core_state, name="fw")
    await fw.connect(mock=False)
    await fw.set("FITC")
    core_state.setStateLabel.assert_called_with("FilterWheel", "FITC")  # type: ignore[attr-defined]


async def test_state_locate_reflects_set(core_state: CMMCorePlus) -> None:
    fw = MyFilterWheel("FilterWheel", core_state, name="fw")
    await fw.connect(mock=False)
    await fw.set("FITC")
    loc = await fw.locate()
    assert loc["readback"] == "FITC"


async def test_state_emits_watcher_updates(core_state: CMMCorePlus) -> None:
    core_state.deviceBusy.side_effect = [True, False]  # type: ignore[attr-defined]
    fw = MyFilterWheel("FilterWheel", core_state, name="fw")
    await fw.connect(mock=False)
    updates, collect = _watcher_collector()
    status = fw.set("FITC")
    status.watch(collect)
    await status
    assert len(updates) >= 1
    assert updates[-1].target == "FITC"


async def test_state_stop(core_state: CMMCorePlus) -> None:
    core_state.deviceBusy.side_effect = lambda lbl: True  # type: ignore[attr-defined]
    fw = MyFilterWheel("FilterWheel", core_state, name="fw")
    await fw.connect(mock=False)
    move = fw.set("FITC")
    await asyncio.sleep(0.02)
    await fw.stop(success=False)
    with pytest.raises(Exception):
        await move


# ---------------------------------------------------------------------------
# AutoFocus
# ---------------------------------------------------------------------------


MyAutoFocus = MMAutoFocus


@pytest.fixture
def core_af() -> Generator[CMMCorePlus, None, None]:
    instance = CMMCorePlus()
    _state = {"offset": 0.0, "locked": False}

    with _base_patches(instance) as stack:
        for cm in [patch.object(instance, "getAutoFocusOffset", side_effect=lambda: _state["offset"]),
        patch.object(instance, "setAutoFocusOffset", side_effect=lambda v: _state.update({"offset": v})),
        patch.object(instance, "fullFocus"),
        patch.object(instance, "isContinuousFocusLocked", side_effect=lambda: _state["locked"])]:
            stack.enter_context(cm)
        yield instance


async def test_af_locate(core_af: CMMCorePlus) -> None:
    af = MyAutoFocus("AutoFocus", core_af, name="af")
    await af.connect(mock=False)
    loc = await af.locate()
    assert loc["readback"] == pytest.approx(0.0)


async def test_af_set_offset(core_af: CMMCorePlus) -> None:
    af = MyAutoFocus("AutoFocus", core_af, name="af")
    await af.connect(mock=False)
    await af.set(10.0)
    core_af.setAutoFocusOffset.assert_called_with(10.0)  # type: ignore[attr-defined]


async def test_af_locate_reflects_set(core_af: CMMCorePlus) -> None:
    af = MyAutoFocus("AutoFocus", core_af, name="af")
    await af.connect(mock=False)
    await af.set(10.0)
    loc = await af.locate()
    assert loc["readback"] == pytest.approx(10.0)


async def test_af_trigger_calls_full_focus(core_af: CMMCorePlus) -> None:
    # Make isContinuousFocusLocked return True on second call so trigger exits
    core_af.isContinuousFocusLocked.side_effect = [False, True]  # type: ignore[attr-defined]
    af = MyAutoFocus("AutoFocus", core_af, name="af")
    await af.connect(mock=False)
    await af.trigger()
    core_af.fullFocus.assert_called_once()  # type: ignore[attr-defined]


async def test_af_stop(core_af: CMMCorePlus) -> None:
    core_af.deviceBusy.side_effect = lambda lbl: True  # type: ignore[attr-defined]
    af = MyAutoFocus("AutoFocus", core_af, name="af")
    await af.connect(mock=False)
    move = af.set(50.0)
    await asyncio.sleep(0.02)
    await af.stop(success=False)
    with pytest.raises(Exception):
        await move


# ---------------------------------------------------------------------------
# Galvo
# ---------------------------------------------------------------------------


class MyGalvo(MMGalvo):
    _dwell_us = 500.0


@pytest.fixture
def core_galvo() -> Generator[CMMCorePlus, None, None]:
    instance = CMMCorePlus()
    _pos = {"x": 0.0, "y": 0.0}

    with _base_patches(instance) as stack:
        for cm in [patch.object(instance, "getGalvoPosition", side_effect=lambda lbl: (_pos["x"], _pos["y"])),
        patch.object(instance, "setGalvoPosition", side_effect=lambda lbl, x, y: _pos.update({"x": x, "y": y})),
        patch.object(instance, "pointGalvoAndFire")]:
            stack.enter_context(cm)
        yield instance


async def test_galvo_locate(core_galvo: CMMCorePlus) -> None:
    g = MyGalvo("Galvo", core_galvo, name="galvo")
    await g.connect(mock=False)
    loc = await g.locate()
    assert loc["readback"] == (pytest.approx(0.0), pytest.approx(0.0))


async def test_galvo_set(core_galvo: CMMCorePlus) -> None:
    g = MyGalvo("Galvo", core_galvo, name="galvo")
    await g.connect(mock=False)
    await g.set((100.0, 200.0))
    core_galvo.setGalvoPosition.assert_called_with("Galvo", 100.0, 200.0)  # type: ignore[attr-defined]


async def test_galvo_locate_reflects_set(core_galvo: CMMCorePlus) -> None:
    g = MyGalvo("Galvo", core_galvo, name="galvo")
    await g.connect(mock=False)
    await g.set((100.0, 200.0))
    loc = await g.locate()
    assert loc["readback"] == (pytest.approx(100.0), pytest.approx(200.0))


async def test_galvo_trigger_fires(core_galvo: CMMCorePlus) -> None:
    g = MyGalvo("Galvo", core_galvo, name="galvo")
    await g.connect(mock=False)
    await g.set((50.0, 75.0))
    await g.trigger()
    core_galvo.pointGalvoAndFire.assert_called_with("Galvo", 50.0, 75.0, 500.0)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Pump
# ---------------------------------------------------------------------------


MyPump = MMPump


@pytest.fixture
def core_pump() -> Generator[CMMCorePlus, None, None]:
    instance = CMMCorePlus()
    _state = {"volume": 0.0}

    with _base_patches(instance) as stack:
        for cm in [patch.object(instance, "getPumpVolume", side_effect=lambda lbl: _state["volume"]),
        patch.object(instance, "pumpDispenseVolumeUl", side_effect=lambda lbl, v: _state.update({"volume": v})),
        patch.object(instance, "volumetricPumpStop")]:
            stack.enter_context(cm)
        yield instance


async def test_pump_set_dispenses(core_pump: CMMCorePlus) -> None:
    p = MyPump("Pump", core_pump, name="pump")
    await p.connect(mock=False)
    await p.set(50.0)
    core_pump.pumpDispenseVolumeUl.assert_called_with("Pump", 50.0)  # type: ignore[attr-defined]


async def test_pump_emits_watcher_updates(core_pump: CMMCorePlus) -> None:
    core_pump.deviceBusy.side_effect = [True, False]  # type: ignore[attr-defined]
    p = MyPump("Pump", core_pump, name="pump")
    await p.connect(mock=False)
    updates, collect = _watcher_collector()
    status = p.set(50.0)
    status.watch(collect)
    await status
    assert len(updates) >= 1
    assert updates[-1].target == pytest.approx(50.0)


async def test_pump_stop(core_pump: CMMCorePlus) -> None:
    core_pump.deviceBusy.side_effect = lambda lbl: True  # type: ignore[attr-defined]
    p = MyPump("Pump", core_pump, name="pump")
    await p.connect(mock=False)
    dispense = p.set(100.0)
    await asyncio.sleep(0.02)
    await p.stop(success=False)
    with pytest.raises(Exception):
        await dispense
    core_pump.volumetricPumpStop.assert_called_with("Pump")  # type: ignore[attr-defined]
