import re
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from ophyd_async.core import (
    Command,
    Device,
    DeviceConnector,
    DeviceFiller,
    PathProvider,
    Signal,
)

from ._backend import MMCommandBackend, MMSignalBackend
from ._controller import MMController


@dataclass(frozen=True)
class Property:
    """Mark a signal that reads and writes the property `name` of the device's label."""

    name: str


@dataclass(frozen=True)
class Label:
    """Mark a child device addressed by a Micro-Manager label.

    `physical` lists the labels of the physical cameras of a multi-camera
    device, and `properties` the properties to report in the child's
    configuration.
    """

    label: str
    physical: tuple[str, ...] = ()
    properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class Position:
    """Mark the position signal of a moving device, in micrometres for a stage."""


@dataclass(frozen=True)
class Halt:
    """Mark the command that stops the device's label."""


@dataclass(frozen=True)
class XYAxis:
    """Mark a child device that is one axis of the parent's XY stage."""

    axis: Literal["x", "y"]


SIGNAL_MARKERS = (Property, Position)
COMMAND_MARKERS = (Halt,)
DEVICE_MARKERS = (Label, XYAxis)
MARKERS = SIGNAL_MARKERS + COMMAND_MARKERS + DEVICE_MARKERS


def clean_name(name: str) -> str:
    """Turn a Micro-Manager name into a lower-case identifier.

    Camel case is split and every run of characters other than letters and
    digits becomes `_`, so `"CCDTemperature RO"` becomes `"ccd_temperature_ro"`.
    """
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    return re.sub(r"[^A-Za-z0-9]+", "_", split).strip("_").lower()


def take_marker(extras: list[Any]) -> Any:
    """Remove and return the marker among a child's annotations."""
    marker = next(extra for extra in extras if isinstance(extra, MARKERS))
    extras.remove(marker)
    return marker


class MMConnector(DeviceConnector):
    """Builds the declared children of a device from their markers.

    Holds what a device needs to address the core: the controller, the label
    and the options of the device.
    """

    def __init__(
        self,
        ctrl: MMController,
        label: str = "",
        *,
        axis: Literal["x", "y"] | None = None,
        output: str | None = None,
        physical: Sequence[str] = (),
        properties: Sequence[str] = (),
        path_provider: PathProvider | None = None,
        move_timeout: float | None = 10.0,
    ) -> None:
        self.ctrl = ctrl
        self.path_provider = path_provider
        self.move_timeout = move_timeout
        self.set_address(
            label=label,
            axis=axis,
            output=output,
            physical=physical,
            properties=properties,
        )

    def set_address(
        self,
        *,
        label: str | None = None,
        axis: Literal["x", "y"] | None = None,
        output: str | None = None,
        physical: Sequence[str] | None = None,
        properties: Sequence[str] = (),
    ) -> None:
        """Replace the fields given, and clear `axis`, `output` and `properties` if not given."""
        if label is not None:
            self.label = label
        if physical is not None:
            self.physical = tuple(physical)
        self.axis: Literal["x", "y"] | None = axis
        self.output = output
        self.properties = tuple(properties)

    def child(self) -> "MMConnector":
        """Return a connector for a declared child, with the same controller and options."""
        return MMConnector(
            self.ctrl,
            self.label,
            physical=self.physical,
            path_provider=self.path_provider,
            move_timeout=self.move_timeout,
        )

    def property_backend(self, name: str) -> MMSignalBackend:
        """Return a bound backend, with no declared datatype, for a property of the label."""
        return MMSignalBackend(None, self.ctrl.property_source(self.label, name))

    def check_markers(self, device: Device) -> None:
        """Raise `TypeError` for a declared child whose marker is missing or does not fit."""
        cls = type(device)
        for attr, hint in typing.get_type_hints(cls, include_extras=True).items():
            if attr == "parent" or attr.startswith("_"):
                continue
            extras: list[Any] = []
            if get_origin(hint) is Annotated:
                hint, *extras = get_args(hint)
            if get_origin(hint) in (Union, types.UnionType):
                args = [arg for arg in get_args(hint) if arg is not types.NoneType]
                if len(args) == 1:
                    hint = args[0]
            origin = get_origin(hint) or hint
            if not (isinstance(origin, type) and issubclass(origin, Device)):
                continue
            if issubclass(origin, Signal):
                allowed = SIGNAL_MARKERS
            elif issubclass(origin, Command):
                allowed = COMMAND_MARKERS
            else:
                allowed = DEVICE_MARKERS
            where = f"{cls.__name__}.{attr}"
            markers = [extra for extra in extras if isinstance(extra, MARKERS)]
            if len(markers) != 1:
                raise TypeError(f"{where}: expected one marker, found {len(markers)}")
            marker = markers[0]
            if not isinstance(marker, allowed):
                raise TypeError(
                    f"{where}: {type(marker).__name__} does not fit a {origin.__name__}"
                )
            if isinstance(marker, Property | Position | Halt) and not self.label:
                raise TypeError(
                    f"{where}: {type(marker).__name__} needs a device with a label"
                )

    def create_children_from_annotations(self, device: Device) -> None:
        """Create every annotated signal, command and device of `device`."""
        if hasattr(self, "filler"):
            return
        self.check_markers(device)
        self.filler = DeviceFiller(
            device,
            signal_backend_factory=MMSignalBackend,
            device_connector_factory=self.child,
            command_backend_factory=lambda sig: MMCommandBackend(),
        )
        for backend, extras in self.filler.create_signals_from_annotations():
            marker = take_marker(extras)
            match marker:
                case Property(name):
                    backend.bind(self.ctrl.property_source(self.label, name))
                case Position():
                    backend.bind(
                        self.ctrl.position_source(self.label, self.axis or "z")  # pyright: ignore[reportArgumentType]
                    )
                case _:
                    raise TypeError(f"{marker!r} does not fit a Signal")
        for backend, extras in self.filler.create_commands_from_annotations():
            marker = take_marker(extras)
            match marker:
                case Halt():
                    backend.bind(
                        self.ctrl.halt(self.label), f"mmcore://{self.label}/halt"
                    )
                case _:
                    raise TypeError(f"{marker!r} does not fit a Command")
        for connector, extras in self.filler.create_devices_from_annotations():
            marker = take_marker(extras)
            match marker:
                case Label(label, physical, properties):
                    connector.set_address(
                        label=label, physical=physical, properties=properties
                    )
                case XYAxis(axis):
                    connector.set_address(axis=axis)
                case _:
                    raise TypeError(f"{marker!r} does not fit a Device")
