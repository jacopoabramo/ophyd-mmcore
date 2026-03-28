"""Concrete ophyd-async devices for Micro-Manager stage types.

Two classes are provided:

- :class:`MMZStage` — Z (focus) stage; implements ``Movable``, ``Stoppable``,
  and ``Locatable[float]``.
- :class:`MMXYStage` — XY stage; implements ``Movable``, ``Stoppable``, and
  ``Locatable[tuple[float, float]]``.

Both use ``deviceBusy()`` polling at 10 Hz to detect move completion and
subscribe to the relevant MM position-changed events to emit
:class:`~ophyd_async.core.WatcherUpdate` progress during the move.

Extra properties are added by subclassing::

    from typing import Annotated as A
    from ophyd_async.core import SignalRW, StandardReadableFormat as Format
    from ophyd_mmcore._stage import MMZStage
    from ophyd_mmcore._connector import PropName


    class ZStage(MMZStage):
        velocity: A[SignalRW[float], PropName("Speed"), Format.CONFIG_SIGNAL]
"""

from __future__ import annotations

import asyncio

from bluesky.protocols import Location
from ophyd_async.core import WatchableAsyncStatus, WatcherUpdate

from ._device import MMDevice

_POLL_INTERVAL = 0.05  # 50 ms


# ---------------------------------------------------------------------------
# Z stage
# ---------------------------------------------------------------------------


class MMZStage(MMDevice):
    """Micro-Manager Z (focus) stage.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"ZStage"``).
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._z_stop_event: asyncio.Event | None = None
        self._z_set_success = True
        super().__init__(mm_label, worker, name=name)

    async def locate(self) -> Location[float]:
        """Return the current Z position in µm."""
        pos: float = await self._worker.run(
            lambda: self._worker.core.getPosition(self._mm_label)
        )
        return Location(setpoint=pos, readback=pos)

    @WatchableAsyncStatus.wrap
    async def set(self, value: float):  # type: ignore[override]
        """Move the Z stage to *value* µm, yielding WatcherUpdates until done."""
        self._z_set_success = True
        self._z_stop_event = asyncio.Event()

        initial: float = await self._worker.run(
            lambda: self._worker.core.getPosition(self._mm_label)
        )
        await self._worker.run(
            lambda: self._worker.core.setPosition(self._mm_label, value)
        )

        current_pos = initial

        def _on_pos_changed(label: str, pos: float) -> None:
            nonlocal current_pos
            if label == self._mm_label:
                current_pos = pos

        self._worker.core.events.stagePositionChanged.connect(_on_pos_changed)
        try:
            while not self._z_stop_event.is_set():
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
                if not busy:
                    break
                await asyncio.sleep(_POLL_INTERVAL)
        finally:
            self._worker.core.events.stagePositionChanged.disconnect(_on_pos_changed)
            self._z_stop_event = None

        if not self._z_set_success:
            raise RuntimeError(f"{self.name}: move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the Z stage immediately."""
        self._z_set_success = success
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._z_stop_event is not None:
            self._z_stop_event.set()


# ---------------------------------------------------------------------------
# XY stage
# ---------------------------------------------------------------------------

XYPosition = tuple[float, float]


class MMXYStage(MMDevice):
    """Micro-Manager XY stage.

    ``set((x, y))`` moves to the given position in µm; ``locate()`` returns
    the current position as both setpoint and readback.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"XYStage"``).
    worker:
        The shared ``MMCoreWorker``.
    name:
        ophyd-async device name.
    """

    def __init__(self, mm_label: str, worker, name: str = "") -> None:
        self._xy_stop_event: asyncio.Event | None = None
        self._xy_set_success = True
        super().__init__(mm_label, worker, name=name)

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
        self._xy_set_success = True
        self._xy_stop_event = asyncio.Event()
        target_x, target_y = value

        def _get_xy() -> XYPosition:
            core = self._worker.core
            return core.getXPosition(self._mm_label), core.getYPosition(self._mm_label)

        initial: XYPosition = await self._worker.run(_get_xy)
        current_pos: XYPosition = initial

        await self._worker.run(
            lambda: self._worker.core.setXYPosition(self._mm_label, target_x, target_y)
        )

        def _on_xy_changed(label: str, x: float, y: float) -> None:
            nonlocal current_pos
            if label == self._mm_label:
                current_pos = (x, y)

        self._worker.core.events.XYStagePositionChanged.connect(_on_xy_changed)
        try:
            while not self._xy_stop_event.is_set():
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
                if not busy:
                    break
                await asyncio.sleep(_POLL_INTERVAL)
        finally:
            self._worker.core.events.XYStagePositionChanged.disconnect(_on_xy_changed)
            self._xy_stop_event = None

        if not self._xy_set_success:
            raise RuntimeError(f"{self.name}: move was stopped")

    async def stop(self, success: bool = True) -> None:
        """Stop the XY stage immediately."""
        self._xy_set_success = success
        await self._worker.run(lambda: self._worker.core.stop(self._mm_label))
        if self._xy_stop_event is not None:
            self._xy_stop_event.set()
