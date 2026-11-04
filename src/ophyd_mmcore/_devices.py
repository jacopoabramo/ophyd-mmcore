from collections.abc import Sequence
from dataclasses import dataclass

from bluesky.protocols import Location
from ophyd_async.core import (
    DeviceMap,
    MovableLogic,
    PathProvider,
    SignalDatatypeT,
    SignalRW,
    StandardMovable,
    StandardReadable,
    TimeoutCalculator,
    TriggerableCommand,
)
from ophyd_async.core import StandardReadableFormat as Format

from ._connector import MMConnector, clean_name
from ._controller import MMController


class MMDevice(StandardReadable):
    """A Micro-Manager device, addressed by its label.

    Pass either `label` with `ctrl`, or a `connector`, as a declared child
    receives. Annotated signals are declared with [`Property`][ophyd_mmcore.Property].
    Each name in `properties` becomes a signal in the `properties` map, which
    `read_configuration` reports.
    """

    def __init__(
        self,
        label: str = "",
        ctrl: MMController | None = None,
        *,
        properties: Sequence[str] = (),
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        if connector is not None:
            if label or ctrl is not None or properties:
                raise ValueError(
                    "MMDevice takes either a connector or a label, ctrl and "
                    "properties, not both"
                )
        elif ctrl is None:
            raise ValueError("MMDevice needs either a ctrl or a connector")
        else:
            connector = MMConnector(ctrl, label, properties=properties)
        self._label = connector.label
        super().__init__(name=name, connector=connector)
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
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.properties = DeviceMap(signals)

    @property
    def label(self) -> str:
        """The Micro-Manager label of the device."""
        return self._label


class MMSystem(StandardReadable):
    """A microscope declared as a class, whose children are marked with [`Label`][ophyd_mmcore.Label]."""

    def __init__(
        self,
        ctrl: MMController,
        *,
        path_provider: PathProvider | None = None,
        move_timeout: float | None = 10.0,
        name: str = "",
    ) -> None:
        super().__init__(
            name=name,
            connector=MMConnector(
                ctrl, path_provider=path_provider, move_timeout=move_timeout
            ),
        )


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
    """A device that moves through [`MMMoveLogic`][ophyd_mmcore.MMMoveLogic]."""

    async def locate(self) -> Location[SignalDatatypeT]:
        """Return the stored target as the setpoint and the hardware value as the readback."""
        return await self.movable_logic.setpoint.locate()
