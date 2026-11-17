from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Literal
from typing import Annotated as A

from bluesky.protocols import Location
from ophyd_async.core import (
    DeviceMap,
    MovableLogic,
    SignalDatatypeT,
    SignalRW,
    StandardMovable,
    StandardReadable,
    TimeoutCalculator,
    TriggerableCommand,
)
from ophyd_async.core import StandardReadableFormat as Format

from ._connector import (
    Halt,
    MMConnector,
    Preset,
    Property,
    ShutterOpen,
    StateLabel,
    clean_name,
)

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus


def property_map(connector: MMConnector) -> DeviceMap[SignalRW]:
    """Return a signal for each of the connector's properties, keyed by its cleaned name.

    Raises
    ------
    ValueError
        If two properties have the same cleaned name.
    """
    signals: dict[str, SignalRW] = {}
    raw_names: dict[str, str] = {}
    for prop in connector.properties:
        key = clean_name(prop)
        if key in signals:
            raise ValueError(
                f"Properties {raw_names[key]!r} and {prop!r} both become {key!r}"
            )
        raw_names[key] = prop
        signals[key] = SignalRW(connector.property_backend(prop))
    return DeviceMap(signals)


class MMDevice(StandardReadable):
    """A Micro-Manager device, addressed by its label.

    Pass either `label` with `core`, or a `connector`, as a declared child
    receives. Annotated signals are declared with [`Property`][ophyd_mmcore.Property].
    Each name in `properties` becomes a signal in the `properties` map, which
    `read_configuration` reports. The core is first used at `connect`, which
    raises `NotConnectedError` when the core has not loaded the label.
    """

    def __init__(
        self,
        label: str = "",
        core: "CMMCorePlus | None" = None,
        *,
        properties: Sequence[str] = (),
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        if connector is not None:
            if label or core is not None or properties:
                raise ValueError(
                    "MMDevice takes either a connector or a label, core and "
                    "properties, not both"
                )
        elif core is None:
            raise ValueError("MMDevice needs either a core or a connector")
        else:
            connector = MMConnector(core, label, properties=properties)
        self._label = connector.label
        self._mm_connector = connector
        super().__init__(name=name, connector=connector)
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.properties = property_map(connector)

    @property
    def label(self) -> str:
        """The Micro-Manager label of the device."""
        return self._label


@dataclass
class MMMoveLogic(MovableLogic[SignalDatatypeT]):
    """How a device moves: a put on `setpoint` returns once the device is idle.

    `halt` stops the move, and `move_timeout` is the time a move may take, or
    `None` for no limit.
    """

    halt: TriggerableCommand | None
    move_timeout: float | None

    async def move(
        self, new_position: SignalDatatypeT, timeout: TimeoutCalculator
    ) -> None:
        """Put `new_position` on the setpoint signal."""
        await self.setpoint.set(new_position, timeout=timeout())

    async def stop(self) -> None:
        """Run `halt`, if there is one."""
        if self.halt is not None:
            await self.halt.trigger()

    async def calculate_timeout(
        self, old_position: SignalDatatypeT, new_position: SignalDatatypeT
    ) -> float | None:
        """Return `move_timeout`."""
        return self.move_timeout


class MMMovable(StandardMovable[SignalDatatypeT]):
    """A device that moves through [`MMMoveLogic`][ophyd_mmcore._devices.MMMoveLogic]."""

    async def locate(self) -> Location[SignalDatatypeT]:
        """Return the stored target as the setpoint and the hardware value as the readback."""
        return await self.movable_logic.setpoint.locate()


def movable_connector(
    label: str,
    core: "CMMCorePlus | None",
    axis: Literal["x", "y"] | None,
    properties: Sequence[str],
    move_timeout: float | None,
    connector: MMConnector | None,
) -> tuple[str, "CMMCorePlus | None", Sequence[str], MMConnector | None]:
    """Return the arguments for `MMDevice`, with the options of a moving device put into a connector."""
    if connector is not None:
        if axis is not None or properties or move_timeout != 10.0:
            raise ValueError(
                "A device given a connector takes no axis, properties or move_timeout"
            )
        return label, core, properties, connector
    if core is None:
        return label, core, properties, connector
    return (
        "",
        None,
        (),
        MMConnector(
            core, label, axis=axis, properties=properties, move_timeout=move_timeout
        ),
    )


class MMShutter(MMMovable[bool], MMDevice):
    """A shutter, moved by setting it open or closed.

    A put on `position` returns when Micro-Manager reports the shutter idle,
    and `halt` stops it. `move_timeout` is the seconds a move may take, and
    `None` disables the limit. `properties` and `move_timeout` cannot be
    combined with a `connector`.
    """

    position: A[SignalRW[bool], ShutterOpen(), Format.HINTED_SIGNAL]
    halt: A[TriggerableCommand, Halt()]

    def __init__(
        self,
        label: str = "",
        core: "CMMCorePlus | None" = None,
        *,
        properties: Sequence[str] = (),
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        label, core, properties, connector = movable_connector(
            label, core, None, properties, move_timeout, connector
        )
        super().__init__(
            label, core, properties=properties, name=name, connector=connector
        )

    @cached_property
    def movable_logic(self) -> MMMoveLogic[bool]:
        """The move logic of the shutter, built from `position` and `halt`."""
        return MMMoveLogic(
            setpoint=self.position,
            readback=self.position,
            halt=self.halt,
            move_timeout=self._mm_connector.move_timeout,
        )


class MMStateDevice(MMMovable[str], MMDevice):
    """A device with named states, such as a filter wheel, moved by state label.

    The labels of the device are the `choices` of the `position` signal. A put
    on `position` returns when Micro-Manager reports the device idle, and
    `halt` stops it. `move_timeout` is the seconds a move may take, and `None`
    disables the limit. `properties` and `move_timeout` cannot be combined
    with a `connector`.
    """

    position: A[SignalRW[str], StateLabel(), Format.HINTED_SIGNAL]
    halt: A[TriggerableCommand, Halt()]

    def __init__(
        self,
        label: str = "",
        core: "CMMCorePlus | None" = None,
        *,
        properties: Sequence[str] = (),
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        label, core, properties, connector = movable_connector(
            label, core, None, properties, move_timeout, connector
        )
        super().__init__(
            label, core, properties=properties, name=name, connector=connector
        )

    @cached_property
    def movable_logic(self) -> MMMoveLogic[str]:
        """The move logic of the device, built from `position` and `halt`."""
        return MMMoveLogic(
            setpoint=self.position,
            readback=self.position,
            halt=self.halt,
            move_timeout=self._mm_connector.move_timeout,
        )


class MMConfigGroup(MMMovable[str], StandardReadable):
    """A Micro-Manager configuration group, moved by preset name.

    A put on `position` applies the preset and returns when every device it
    touches is idle. `position` reads `""` when no preset matches the
    hardware. A group has no `halt`, so `stop` returns at once. The presets
    are the `choices` of the `position` signal. `move_timeout` is the seconds
    a move may take, and `None` disables the limit. `move_timeout` cannot be
    combined with a `connector`.
    """

    refused_markers: ClassVar[tuple[type, ...]] = (Property, Halt)
    position: A[SignalRW[str], Preset(), Format.HINTED_SIGNAL]

    def __init__(
        self,
        group: str = "",
        core: "CMMCorePlus | None" = None,
        *,
        move_timeout: float | None = 10.0,
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        group, core, _, connector = movable_connector(
            group, core, None, (), move_timeout, connector
        )
        if connector is None:
            raise ValueError("MMConfigGroup needs either a core or a connector")
        if group or core is not None:
            raise ValueError(
                "MMConfigGroup takes either a connector or a group and core, not both"
            )
        connector.group = True
        self._mm_connector = connector
        super().__init__(name=name, connector=connector)

    @cached_property
    def movable_logic(self) -> MMMoveLogic[str]:
        """The move logic of the group, built from `position`."""
        return MMMoveLogic(
            setpoint=self.position,
            readback=self.position,
            halt=None,
            move_timeout=self._mm_connector.move_timeout,
        )
