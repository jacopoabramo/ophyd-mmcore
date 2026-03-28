"""Concrete ophyd-async devices for Micro-Manager device types.

Each class inherits from :class:`~ophyd_mmcore.MMDevice`, which supplies
``_mm_label``, ``_worker``, and the ``StandardReadable`` property-signal
infrastructure.  The motion / control logic is implemented directly on the
class, so no mixin or multiple inheritance is required.

Extra properties are added by subclassing and annotating with ``PropName``::

    from typing import Annotated as A
    from ophyd_async.core import SignalRW, StandardReadableFormat as Format
    from ophyd_mmcore._devices import MMShutter
    from ophyd_mmcore._connector import PropName


    class MyShutter(MMShutter):
        delay_ms: A[SignalRW[float], PropName("ClosingDelay"), Format.CONFIG_SIGNAL]

====================  =====================================  ======================================================
Device type           MM API                                 Bluesky protocols
====================  =====================================  ======================================================
Shutter               setShutterOpen / getShutterOpen        Movable[bool], Locatable[bool]
State device          setState / getState / getStateLabel    Movable[str], Locatable[str]
AutoFocus             fullFocus / getAutoFocusOffset         Movable[float] (offset), Triggerable (snap focus)
Galvo                 setGalvoPosition / pointGalvoAndFire   Movable[tuple], Triggerable (fire)
Pump (volumetric)     pumpDispenseVolumeUl / pumpStop        Movable[float] (volume), Stoppable
====================  =====================================  ======================================================

Z stage and XY stage are defined below.

Laser / generic devices whose entire API is property-based need no subclass —
``PropName`` annotations on a plain ``MMDevice`` subclass are sufficient.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Annotated as A

from bluesky.protocols import Location
from ophyd_async.core import (
    AsyncStatus,
    SignalRW,
    WatchableAsyncStatus,
    WatcherUpdate,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)

from ._base import MMDevice
from ._connector import XPositionMethod, YPositionMethod, ZPositionMethod

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


# ---------------------------------------------------------------------------
# Shutter
# ---------------------------------------------------------------------------


class MMShutter(MMDevice):
    """Micro-Manager shutter device.

    ``set(True)`` opens the shutter; ``set(False)`` closes it.
    ``locate()`` returns the current open/closed state as both setpoint
    and readback.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"Shutter"``).
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._shutter_wakeup: asyncio.Event | None = None
        self._shutter_stopped = False
        self._shutter_set_success = True
        super().__init__(mm_label, core, name=name)

    async def locate(self) -> Location[bool]:
        """Return the current open state as both setpoint and readback."""
        state: bool = await self._worker.run(
            lambda: self._worker.core.getShutterOpen(self._mm_label)
        )
        return Location(setpoint=state, readback=state)

    @WatchableAsyncStatus.wrap
    async def set(self, value: bool):  # type: ignore[override]
        """Open (``True``) or close (``False``) the shutter."""
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()
        self._shutter_wakeup = wakeup
        self._shutter_stopped = False
        self._shutter_set_success = True

        initial: bool = await self._worker.run(
            lambda: self._worker.core.getShutterOpen(self._mm_label)
        )
        current = initial

        def _on_changed(label: str, state: bool) -> None:
            nonlocal current
            if label == self._mm_label:
                current = state
                loop.call_soon_threadsafe(wakeup.set)

        self._worker.core.events.shutterOpenChanged.connect(_on_changed)
        await self._worker.run(
            lambda: self._worker.core.setShutterOpen(self._mm_label, value)
        )

        try:
            while True:
                wakeup.clear()
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current,
                    initial=initial,
                    target=value,
                    name=self.name,
                )
                if not busy or self._shutter_stopped:
                    break
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._worker.core.events.shutterOpenChanged.disconnect(_on_changed)
            self._shutter_wakeup = None

        if not self._shutter_set_success:
            raise RuntimeError(f"{self.name}: shutter move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Interrupt a shutter transition."""
        self._shutter_set_success = success
        self._shutter_stopped = True
        if self._shutter_wakeup is not None:
            self._shutter_wakeup.set()


# ---------------------------------------------------------------------------
# State device  (filter wheels, objective turrets, etc.)
# ---------------------------------------------------------------------------


class MMStateDevice(MMDevice):
    """Micro-Manager state device (filter wheel, objective turret, etc.).

    ``set("DAPI")`` selects the named position; ``locate()`` returns the
    current label as both setpoint and readback.

    To move by integer index instead, use ``PropName("State")`` directly.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"FilterWheel"``).
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._state_wakeup: asyncio.Event | None = None
        self._state_stopped = False
        self._state_set_success = True
        super().__init__(mm_label, core, name=name)

    async def locate(self) -> Location[str]:
        """Return the current state label."""
        label: str = await self._worker.run(
            lambda: self._worker.core.getStateLabel(self._mm_label)
        )
        return Location(setpoint=label, readback=label)

    @WatchableAsyncStatus.wrap
    async def set(self, value: str):  # type: ignore[override]
        """Move to the named state position (e.g. ``"DAPI"``)."""
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()
        self._state_wakeup = wakeup
        self._state_stopped = False
        self._state_set_success = True

        initial: str = await self._worker.run(
            lambda: self._worker.core.getStateLabel(self._mm_label)
        )
        current = initial

        def _on_changed(label: str, prop: str, val: str) -> None:
            nonlocal current
            if label == self._mm_label and prop == "Label":
                current = val
                loop.call_soon_threadsafe(wakeup.set)

        self._worker.core.events.propertyChanged.connect(_on_changed)
        await self._worker.run(
            lambda: self._worker.core.setStateLabel(self._mm_label, value)
        )

        try:
            while True:
                wakeup.clear()
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current,
                    initial=initial,
                    target=value,
                    name=self.name,
                )
                if not busy or self._state_stopped:
                    break
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._worker.core.events.propertyChanged.disconnect(_on_changed)
            self._state_wakeup = None

        if not self._state_set_success:
            raise RuntimeError(f"{self.name}: state move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Interrupt a state transition."""
        self._state_set_success = success
        self._state_stopped = True
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._state_wakeup is not None:
            self._state_wakeup.set()


# ---------------------------------------------------------------------------
# AutoFocus
# ---------------------------------------------------------------------------


class MMAutoFocus(MMDevice):
    """Micro-Manager autofocus device.

    ``set(offset)`` adjusts the focus offset (µm); ``locate()`` reads it
    back.  ``trigger()`` runs a full focus snap and waits until the device
    reports locked.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"AutoFocus"``).
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._af_wakeup: asyncio.Event | None = None
        self._af_stopped = False
        self._af_set_success = True
        super().__init__(mm_label, core, name=name)

    async def locate(self) -> Location[float]:
        """Return the current autofocus offset in µm."""
        offset: float = await self._worker.run(
            lambda: self._worker.core.getAutoFocusOffset()
        )
        return Location(setpoint=offset, readback=offset)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Move the autofocus offset to *value* µm."""
        self._af_stopped = False
        self._af_set_success = True

        initial: float = await self._worker.run(
            lambda: self._worker.core.getAutoFocusOffset()
        )
        await self._worker.run(lambda: self._worker.core.setAutoFocusOffset(value))

        current = initial
        while not self._af_stopped:
            busy: bool = await self._worker.run(
                lambda: self._worker.core.deviceBusy(self._mm_label)
            )
            yield WatcherUpdate(
                current=current,
                initial=initial,
                target=value,
                name=self.name,
                unit="µm",
            )
            if not busy:
                break
            current = await self._worker.run(
                lambda: self._worker.core.getAutoFocusOffset()
            )
            await asyncio.sleep(0)

        if not self._af_set_success:
            raise RuntimeError(f"{self.name}: autofocus offset move was stopped")

    @AsyncStatus.wrap
    async def trigger(self) -> None:  # type: ignore[override]
        """Run a full focus snap and wait until the device reports locked."""
        await self._worker.run(lambda: self._worker.core.fullFocus())
        while True:
            locked: bool = await self._worker.run(
                lambda: self._worker.core.isContinuousFocusLocked()
            )
            if locked:
                break
            busy: bool = await self._worker.run(
                lambda: self._worker.core.deviceBusy(self._mm_label)
            )
            if not busy:
                break
            await asyncio.sleep(0)

    async def stop(self, success: bool = True) -> None:
        """Interrupt an in-progress offset move."""
        self._af_set_success = success
        self._af_stopped = True


# ---------------------------------------------------------------------------
# Galvo
# ---------------------------------------------------------------------------

GalvoPosition = tuple[float, float]


class MMGalvo(MMDevice):
    """Micro-Manager galvo device.

    ``set((x, y))`` positions the mirrors; ``locate()`` reads the current
    position.  ``trigger()`` fires at the current position using
    :attr:`_dwell_us` microseconds dwell time.

    Override ``_dwell_us`` on a subclass or per-instance to change the dwell
    time if the ``DwellTime`` property is not available on the adapter::

        class MyGalvo(MMGalvo):
            _dwell_us = 500.0

    Parameters
    ----------
    mm_label:
        Micro-Manager device label.
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    _dwell_us: float = 1000.0  # override per subclass or instance

    async def locate(self) -> Location[GalvoPosition]:
        """Return the current galvo (x, y) position."""
        pos = await self._worker.run(
            lambda: self._worker.core.getGalvoPosition(self._mm_label)
        )
        return Location(setpoint=tuple(pos), readback=tuple(pos))

    @AsyncStatus.wrap
    async def set(self, value: GalvoPosition) -> None:  # type: ignore[override]
        """Position the galvo mirrors at *(x, y)*."""
        x, y = value
        await self._worker.run(
            lambda: self._worker.core.setGalvoPosition(self._mm_label, x, y)
        )

    @AsyncStatus.wrap
    async def trigger(self) -> None:  # type: ignore[override]
        """Fire the galvo at its current position."""
        dwell = self._dwell_us
        pos = await self._worker.run(
            lambda: self._worker.core.getGalvoPosition(self._mm_label)
        )
        x, y = pos
        await self._worker.run(
            lambda: self._worker.core.pointGalvoAndFire(self._mm_label, x, y, dwell)
        )


# ---------------------------------------------------------------------------
# Volumetric pump
# ---------------------------------------------------------------------------


class MMPump(MMDevice):
    """Micro-Manager volumetric pump device.

    ``set(volume_ul)`` dispenses *volume_ul* microlitres at the current
    flowrate and waits for the dispense to complete.  ``stop()`` halts the
    pump immediately.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label.
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._pump_stopped = False
        self._pump_set_success = True
        super().__init__(mm_label, core, name=name)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Dispense *value* µL, yielding progress updates until done."""
        self._pump_stopped = False
        self._pump_set_success = True

        await self._worker.run(
            lambda: self._worker.core.pumpDispenseVolumeUl(self._mm_label, value)
        )

        dispensed = 0.0
        while not self._pump_stopped:
            busy: bool = await self._worker.run(
                lambda: self._worker.core.deviceBusy(self._mm_label)
            )
            dispensed = value - await self._worker.run(
                lambda: self._worker.core.getPumpVolume(self._mm_label)
            )
            yield WatcherUpdate(
                current=dispensed,
                initial=0.0,
                target=value,
                name=self.name,
                unit="uL",
            )
            if not busy:
                break
            await asyncio.sleep(0)

        if not self._pump_set_success:
            raise RuntimeError(f"{self.name}: pump dispense was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the pump immediately."""
        self._pump_set_success = success
        self._pump_stopped = True
        await self._worker.run(
            lambda: self._worker.core.volumetricPumpStop(self._mm_label)
        )


# ---------------------------------------------------------------------------
# Z stage
# ---------------------------------------------------------------------------


class MMZStage(MMDevice):
    """Micro-Manager Z (focus) stage.

    ``position`` is a readable/writable signal for the current Z position in µm.
    For moves with progress tracking use :meth:`set` instead.

    Extra properties are added by subclassing::

        from typing import Annotated as A
        from ophyd_async.core import SignalRW, StandardReadableFormat as Format
        from ophyd_mmcore import MMZStage
        from ophyd_mmcore._connector import PropName


        class ZStage(MMZStage):
            velocity: A[SignalRW[float], PropName("Speed"), Format.CONFIG_SIGNAL]

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"ZStage"``).
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    position: A[SignalRW[float], ZPositionMethod, Format.HINTED_SIGNAL]

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._z_wakeup: asyncio.Event | None = None
        self._z_stopped = False
        self._z_set_success = True
        super().__init__(mm_label, core, name=name)

    async def locate(self) -> Location[float]:
        """Return the current Z position in µm."""
        pos: float = await self._worker.run(
            lambda: self._worker.core.getPosition(self._mm_label)
        )
        return Location(setpoint=pos, readback=pos)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Move the Z stage to *value* µm, yielding WatcherUpdates until done."""
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()
        self._z_wakeup = wakeup
        self._z_stopped = False
        self._z_set_success = True

        initial: float = await self._worker.run(
            lambda: self._worker.core.getPosition(self._mm_label)
        )
        current_pos = initial

        def _on_pos_changed(label: str, pos: float) -> None:
            nonlocal current_pos
            if label == self._mm_label:
                current_pos = pos
                loop.call_soon_threadsafe(wakeup.set)

        self._worker.core.events.stagePositionChanged.connect(_on_pos_changed)
        await self._worker.run(
            lambda: self._worker.core.setPosition(self._mm_label, value)
        )

        try:
            while True:
                wakeup.clear()
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current_pos,
                    initial=initial,
                    target=value,
                    name=self.name,
                    unit="µm",
                )
                if not busy or self._z_stopped:
                    break
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._worker.core.events.stagePositionChanged.disconnect(_on_pos_changed)
            self._z_wakeup = None

        if not self._z_set_success:
            raise RuntimeError(f"{self.name}: move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the Z stage immediately."""
        self._z_set_success = success
        self._z_stopped = True
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._z_wakeup is not None:
            self._z_wakeup.set()


# ---------------------------------------------------------------------------
# XY stage
# ---------------------------------------------------------------------------

XYPosition = tuple[float, float]


class MMXYStage(MMDevice):
    """Micro-Manager XY stage.

    ``x`` and ``y`` are readable/writable signals for the individual axes in µm.
    For coordinated moves with progress tracking use :meth:`set` instead.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"XYStage"``).
    core:
        The shared ``CMMCorePlus`` instance.
    name:
        ophyd-async device name.
    """

    x: A[SignalRW[float], XPositionMethod, Format.HINTED_SIGNAL]
    y: A[SignalRW[float], YPositionMethod, Format.HINTED_SIGNAL]

    def __init__(self, mm_label: str, core: CMMCorePlus, name: str = "") -> None:
        self._xy_wakeup: asyncio.Event | None = None
        self._xy_stopped = False
        self._xy_set_success = True
        super().__init__(mm_label, core, name=name)

    async def locate(self) -> Location[XYPosition]:
        """Return the current (x, y) position in µm."""

        def _get_xy() -> XYPosition:
            core = self._worker.core
            return core.getXPosition(self._mm_label), core.getYPosition(self._mm_label)

        x, y = await self._worker.run(_get_xy)
        return Location(setpoint=(x, y), readback=(x, y))

    @WatchableAsyncStatus.wrap
    async def set(self, value: XYPosition):  # type: ignore[override]
        """Move the XY stage to *(x, y)* µm, yielding WatcherUpdates until done."""
        loop = asyncio.get_running_loop()
        wakeup = asyncio.Event()
        self._xy_wakeup = wakeup
        self._xy_stopped = False
        self._xy_set_success = True
        target_x, target_y = value

        def _get_xy() -> XYPosition:
            core = self._worker.core
            return core.getXPosition(self._mm_label), core.getYPosition(self._mm_label)

        initial: XYPosition = await self._worker.run(_get_xy)
        current_pos: XYPosition = initial

        def _on_xy_changed(label: str, x: float, y: float) -> None:
            nonlocal current_pos
            if label == self._mm_label:
                current_pos = (x, y)
                loop.call_soon_threadsafe(wakeup.set)

        self._worker.core.events.XYStagePositionChanged.connect(_on_xy_changed)
        await self._worker.run(
            lambda: self._worker.core.setXYPosition(self._mm_label, target_x, target_y)
        )

        try:
            while True:
                wakeup.clear()
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current_pos,
                    initial=initial,
                    target=value,
                    name=self.name,
                    unit="µm",
                )
                if not busy or self._xy_stopped:
                    break
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
        finally:
            self._worker.core.events.XYStagePositionChanged.disconnect(_on_xy_changed)
            self._xy_wakeup = None

        if not self._xy_set_success:
            raise RuntimeError(f"{self.name}: move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the XY stage immediately."""
        self._xy_set_success = success
        self._xy_stopped = True
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._xy_wakeup is not None:
            self._xy_wakeup.set()
