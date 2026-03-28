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

Z stage and XY stage are in ``_stage.py``.

Laser / generic devices whose entire API is property-based need no subclass —
``PropName`` annotations on a plain ``MMDevice`` subclass are sufficient.
"""

from __future__ import annotations

import asyncio

from bluesky.protocols import Location
from ophyd_async.core import AsyncStatus, WatchableAsyncStatus, WatcherUpdate

from ._device import MMDevice

_POLL_INTERVAL = 0.05  # 50 ms


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
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._shutter_stop_event: asyncio.Event | None = None
        self._shutter_set_success = True
        super().__init__(mm_label, worker, name=name)

    async def locate(self) -> Location[bool]:
        """Return the current open state as both setpoint and readback."""
        state: bool = await self._worker.run(
            lambda: self._worker.core.getShutterOpen(self._mm_label)
        )
        return Location(setpoint=state, readback=state)

    @WatchableAsyncStatus.wrap
    async def set(self, value: bool):  # type: ignore[override]
        """Open (``True``) or close (``False``) the shutter."""
        self._shutter_set_success = True
        self._shutter_stop_event = asyncio.Event()

        initial: bool = await self._worker.run(
            lambda: self._worker.core.getShutterOpen(self._mm_label)
        )
        await self._worker.run(
            lambda: self._worker.core.setShutterOpen(self._mm_label, value)
        )

        current = initial

        def _on_changed(label: str, state: bool) -> None:
            nonlocal current
            if label == self._mm_label:
                current = state

        self._worker.core.events.shutterOpenChanged.connect(_on_changed)
        try:
            while not self._shutter_stop_event.is_set():
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current,
                    initial=initial,
                    target=value,
                    name=self.name,
                )
                if not busy:
                    break
                await asyncio.sleep(_POLL_INTERVAL)
        finally:
            self._worker.core.events.shutterOpenChanged.disconnect(_on_changed)
            self._shutter_stop_event = None

        if not self._shutter_set_success:
            raise RuntimeError(f"{self.name}: shutter move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Interrupt a shutter transition."""
        self._shutter_set_success = success
        if self._shutter_stop_event is not None:
            self._shutter_stop_event.set()


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
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._state_stop_event: asyncio.Event | None = None
        self._state_set_success = True
        super().__init__(mm_label, worker, name=name)

    async def locate(self) -> Location[str]:
        """Return the current state label."""
        label: str = await self._worker.run(
            lambda: self._worker.core.getStateLabel(self._mm_label)
        )
        return Location(setpoint=label, readback=label)

    @WatchableAsyncStatus.wrap
    async def set(self, value: str):  # type: ignore[override]
        """Move to the named state position (e.g. ``"DAPI"``)."""
        self._state_set_success = True
        self._state_stop_event = asyncio.Event()

        initial: str = await self._worker.run(
            lambda: self._worker.core.getStateLabel(self._mm_label)
        )
        await self._worker.run(
            lambda: self._worker.core.setStateLabel(self._mm_label, value)
        )

        current = initial

        def _on_changed(label: str, prop: str, val: str) -> None:
            nonlocal current
            if label == self._mm_label and prop == "Label":
                current = val

        self._worker.core.events.propertyChanged.connect(_on_changed)
        try:
            while not self._state_stop_event.is_set():
                busy: bool = await self._worker.run(
                    lambda: self._worker.core.deviceBusy(self._mm_label)
                )
                yield WatcherUpdate(
                    current=current,
                    initial=initial,
                    target=value,
                    name=self.name,
                )
                if not busy:
                    break
                await asyncio.sleep(_POLL_INTERVAL)
        finally:
            self._worker.core.events.propertyChanged.disconnect(_on_changed)
            self._state_stop_event = None

        if not self._state_set_success:
            raise RuntimeError(f"{self.name}: state move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Interrupt a state transition."""
        self._state_set_success = success
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._state_stop_event is not None:
            self._state_stop_event.set()


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
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._af_stop_event: asyncio.Event | None = None
        self._af_set_success = True
        super().__init__(mm_label, worker, name=name)

    async def locate(self) -> Location[float]:
        """Return the current autofocus offset in µm."""
        offset: float = await self._worker.run(
            lambda: self._worker.core.getAutoFocusOffset()
        )
        return Location(setpoint=offset, readback=offset)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Move the autofocus offset to *value* µm."""
        self._af_set_success = True
        self._af_stop_event = asyncio.Event()

        initial: float = await self._worker.run(
            lambda: self._worker.core.getAutoFocusOffset()
        )
        await self._worker.run(lambda: self._worker.core.setAutoFocusOffset(value))

        current = initial
        while not self._af_stop_event.is_set():
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
            await asyncio.sleep(_POLL_INTERVAL)

        self._af_stop_event = None
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
            await asyncio.sleep(_POLL_INTERVAL)

    async def stop(self, success: bool = True) -> None:
        """Interrupt an in-progress offset move."""
        self._af_set_success = success
        if self._af_stop_event is not None:
            self._af_stop_event.set()


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
    worker:
        The shared ``MMCoreWorker``.
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
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._pump_stop_event: asyncio.Event | None = None
        self._pump_set_success = True
        super().__init__(mm_label, worker, name=name)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Dispense *value* µL, yielding progress updates until done."""
        self._pump_set_success = True
        self._pump_stop_event = asyncio.Event()

        await self._worker.run(
            lambda: self._worker.core.pumpDispenseVolumeUl(self._mm_label, value)
        )

        dispensed = 0.0
        while not self._pump_stop_event.is_set():
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
                unit="µL",
            )
            if not busy:
                break
            await asyncio.sleep(_POLL_INTERVAL)

        self._pump_stop_event = None
        if not self._pump_set_success:
            raise RuntimeError(f"{self.name}: pump dispense was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the pump immediately."""
        self._pump_set_success = success
        await self._worker.run(
            lambda: self._worker.core.volumetricPumpStop(self._mm_label)
        )
        if self._pump_stop_event is not None:
            self._pump_stop_event.set()
