from collections.abc import Sequence
from functools import cached_property
from typing import Annotated as A
from typing import Literal

from ophyd_async.core import SignalRW, TriggerableCommand
from ophyd_async.core import StandardReadableFormat as Format

from ._connector import Halt, MMConnector, Position, XYAxis
from ._controller import MMController
from ._devices import MMDevice, MMMovable, MMMoveLogic


def stage_connector(
    label: str,
    ctrl: MMController | None,
    axis: Literal["x", "y"] | None,
    properties: Sequence[str],
    move_timeout: float | None,
    connector: MMConnector | None,
) -> tuple[str, MMController | None, Sequence[str], MMConnector | None]:
    """Return the arguments for `MMDevice`, with the stage options put into a connector."""
    if connector is not None:
        if axis is not None or properties or move_timeout != 10.0:
            raise ValueError(
                "A device given a connector takes no axis, properties or move_timeout"
            )
        return label, ctrl, properties, connector
    if ctrl is None:
        return label, ctrl, properties, connector
    return (
        "",
        None,
        (),
        MMConnector(
            ctrl, label, axis=axis, properties=properties, move_timeout=move_timeout
        ),
    )


class MMAxis(MMMovable[float], MMDevice):
    """One axis of a stage, moved in micrometres.

    A put on `position` returns when Micro-Manager reports the stage idle, and
    `halt` stops the stage. `axis` is `"x"` or `"y"` for an axis of an XY
    stage, and `None` for a one-axis stage. `move_timeout` is the seconds a
    move may take, and `None` disables the limit. `axis`, `properties` and
    `move_timeout` cannot be combined with a `connector`.
    """

    position: A[SignalRW[float], Position(), Format.HINTED_SIGNAL]
    halt: A[TriggerableCommand, Halt()]

    def __init__(
        self,
        label: str = "",
        ctrl: MMController | None = None,
        *,
        axis: Literal["x", "y"] | None = None,
        properties: Sequence[str] = (),
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        label, ctrl, properties, connector = stage_connector(
            label, ctrl, axis, properties, move_timeout, connector
        )
        super().__init__(
            label, ctrl, properties=properties, name=name, connector=connector
        )

    @cached_property
    def movable_logic(self) -> MMMoveLogic[float]:
        """The move logic of the axis, built from `position` and `halt`."""
        return MMMoveLogic(
            setpoint=self.position,
            readback=self.position,
            halt=self.halt,
            move_timeout=self._mm_connector.move_timeout,
        )


class MMZStage(MMAxis):
    """A one-axis stage, such as a focus drive, moved in micrometres."""

    def __init__(
        self,
        label: str = "",
        ctrl: MMController | None = None,
        *,
        properties: Sequence[str] = (),
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        super().__init__(
            label,
            ctrl,
            properties=properties,
            move_timeout=move_timeout,
            name=name,
            connector=connector,
        )


class MMXYStage(MMDevice):
    """A two-axis stage whose axes `x` and `y` move in micrometres.

    Moving both axes at once ends at both targets. Halting either axis halts
    the stage.
    """

    x: A[MMAxis, XYAxis("x"), Format.CHILD]
    y: A[MMAxis, XYAxis("y"), Format.CHILD]

    def __init__(
        self,
        label: str = "",
        ctrl: MMController | None = None,
        *,
        properties: Sequence[str] = (),
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        label, ctrl, properties, connector = stage_connector(
            label, ctrl, None, properties, move_timeout, connector
        )
        super().__init__(
            label, ctrl, properties=properties, name=name, connector=connector
        )
