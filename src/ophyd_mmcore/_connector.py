"""Declarative device connector for Micro-Manager backed devices.

Usage
-----
Instead of manually constructing signals in ``__init__``, subclass
:class:`MMDevice` and declare signals as annotated class attributes::

    from typing import Annotated as A
    from ophyd_async.core import SignalR, SignalRW
    from ophyd_async.core import StandardReadableFormat as Format
    from ophyd_mmcore import MMDevice
    from ophyd_mmcore._connector import PropName


    class MMCamera(MMDevice):
        exposure: A[SignalRW[float], PropName("Exposure"), Format.HINTED_SIGNAL]
        binning: A[SignalRW[str], PropName("Binning"), Format.CONFIG_SIGNAL]
        read_mode: A[SignalR[str], PropName("ReadMode"), Format.CONFIG_SIGNAL]

The connector reads the ``PropName`` annotation to wire each signal to
the corresponding Micro-Manager device property, using the shared
:class:`~ophyd_mmcore._worker.MMCoreWorker` passed to ``MMDevice.__init__``.

Pre-arranged :class:`CoreMethod` instances are provided for properties that
have dedicated API methods on ``CMMCorePlus``:

- :data:`ExposureMethod` — ``getExposure`` / ``setExposure``
- :data:`ZPositionMethod` — ``getPosition`` / ``setPosition``
- :data:`XPositionMethod` — ``getXPosition`` (read-only)
- :data:`YPositionMethod` — ``getYPosition`` (read-only)
- :data:`ShutterOpenMethod` — ``getShutterOpen`` / ``setShutterOpen``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from ophyd_async.core import Device, DeviceConnector, DeviceFiller

from ._backend import MMPropertyBackend

if TYPE_CHECKING:
    from collections.abc import Callable

    from psygnal import SignalInstance
    from pymmcore_plus import CMMCorePlus

    from ._worker import MMCoreWorker


@dataclass
class PropName:
    """Annotation that maps a signal attribute to a MM device property name.

    Parameters
    ----------
    name:
        The Micro-Manager property name (e.g. ``"Exposure"``).

    Examples
    --------
    ::

        class MyDevice(MMDevice):
            exposure: Annotated[
                SignalRW[float], PropName("Exposure"), Format.HINTED_SIGNAL
            ]
    """

    name: str


@dataclass
class CoreMethod:
    """Annotation that maps a signal to dedicated ``CMMCorePlus`` API methods.

    Use this instead of :class:`PropName` when MM exposes a typed getter/setter
    (e.g. ``getExposure``/``setExposure``) that should be preferred over the
    generic ``getProperty``/``setProperty`` path.

    Parameters
    ----------
    get:
        Callable ``(core, label) -> value`` that reads the current value.
    set:
        Callable ``(core, label, value) -> None`` that writes a new value.
        Omit for read-only signals.
    event:
        Callable ``(core) -> SignalInstance`` that returns the psygnal signal
        to subscribe to for live updates.  The signal must emit
        ``(label: str, value, ...)`` so the backend can filter by device label
        and extract the new value.  Omit if live updates are not needed.

    Examples
    --------
    Use a pre-arranged constant::

        from ophyd_mmcore._connector import ExposureMethod


        class MyCamera(MMDevice):
            exposure: Annotated[SignalRW[float], ExposureMethod, Format.HINTED_SIGNAL]

    Or define a custom one::

        from ophyd_mmcore._connector import CoreMethod


        class MyDevice(MMDevice):
            pos: Annotated[
                SignalRW[float],
                CoreMethod(
                    get=lambda core, label: core.getPosition(label),
                    set=lambda core, label, v: core.setPosition(label, v),
                    event=lambda core: core.events.stagePositionChanged,
                ),
            ]
    """

    get: Callable[[CMMCorePlus, str], Any]
    set: Callable[[CMMCorePlus, str, Any], None] | None = None
    event: Callable[[CMMCorePlus], SignalInstance] | None = field(default=None)
    event_value: Callable[..., Any] | None = field(default=None)
    """Extract the value from event args after ``label``.

    Defaults to taking the first argument.  Override for events that emit
    multiple values — e.g. ``XYStagePositionChanged(label, x, y)``.
    """


# ---------------------------------------------------------------------------
# Pre-arranged CoreMethod constants
# ---------------------------------------------------------------------------

ExposureMethod: CoreMethod = CoreMethod(
    get=lambda core, label: core.getExposure(label),
    set=lambda core, label, v: core.setExposure(label, v),
    event=lambda core: core.events.exposureChanged,
)
"""CoreMethod for camera exposure (``getExposure`` / ``setExposure``)."""

ZPositionMethod: CoreMethod = CoreMethod(
    get=lambda core, label: core.getPosition(label),
    set=lambda core, label, v: core.setPosition(label, v),
    event=lambda core: core.events.stagePositionChanged,
)
"""CoreMethod for Z-stage position (``getPosition`` / ``setPosition``)."""

XPositionMethod: CoreMethod = CoreMethod(
    get=lambda core, label: core.getXPosition(label),
    set=lambda core, label, v: core.setXYPosition(label, v, core.getYPosition(label)),
    event=lambda core: core.events.XYStagePositionChanged,
    event_value=lambda x, _: x,
)
"""CoreMethod for the X axis of an XY stage.

The setter moves only X by reading the current Y and calling ``setXYPosition``.
"""

YPositionMethod: CoreMethod = CoreMethod(
    get=lambda core, label: core.getYPosition(label),
    set=lambda core, label, v: core.setXYPosition(label, core.getXPosition(label), v),
    event=lambda core: core.events.XYStagePositionChanged,
    event_value=lambda _, y: y,
)
"""CoreMethod for the Y axis of an XY stage.

The setter moves only Y by reading the current X and calling ``setXYPosition``.
"""

ShutterOpenMethod: CoreMethod = CoreMethod(
    get=lambda core, label: core.getShutterOpen(label),
    set=lambda core, label, v: core.setShutterOpen(label, v),
    event=lambda core: core.events.shutterOpenChanged,
)
"""CoreMethod for shutter open/closed state (``getShutterOpen`` / ``setShutterOpen``)."""


# ---------------------------------------------------------------------------
# MMDeviceConnector
# ---------------------------------------------------------------------------


class MMDeviceConnector(DeviceConnector):
    """Connector that wires ``PropName``-annotated signals to MM properties.

    Created automatically by :class:`~ophyd_mmcore._device.MMDevice` when
    the class has ``PropName``-annotated attributes.  You do not normally
    need to instantiate this directly.
    """

    def __init__(self, mm_label: str, worker: MMCoreWorker) -> None:
        self._mm_label = mm_label
        self._worker = worker

    def create_children_from_annotations(self, device: Device) -> None:
        if hasattr(self, "_filler"):
            return  # already created

        self._filler = DeviceFiller(
            device,
            signal_backend_factory=lambda datatype: MMPropertyBackend(
                self._mm_label,
                "",
                self._worker,
                cast("Any", datatype),  # DeviceFiller uses full SignalDatatype union
            ),
            device_connector_factory=DeviceConnector,
        )

        for backend, ann in self._filler.create_signals_from_annotations():
            _fill_backend_from_annotation(backend, ann)

        list(self._filler.create_devices_from_annotations())


def _fill_backend_from_annotation(
    backend: MMPropertyBackend[Any], annotations: list[Any]
) -> None:
    """Consume ``PropName`` or ``CoreMethod`` annotations and configure the backend."""
    unhandled = []
    for annotation in annotations:
        if isinstance(annotation, PropName):
            backend.configure_prop(annotation.name)
        elif isinstance(annotation, CoreMethod):
            backend.configure_core_method(
                annotation.get, annotation.set, annotation.event, annotation.event_value
            )
        else:
            unhandled.append(annotation)
    annotations[:] = unhandled
