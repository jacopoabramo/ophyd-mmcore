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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ophyd_async.core import Device, DeviceConnector, DeviceFiller

from ._backend import MMPropertyBackend

if TYPE_CHECKING:
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
            _fill_backend_with_prop_name(self._mm_label, backend, ann)

        list(self._filler.create_devices_from_annotations())


def _fill_backend_with_prop_name(
    mm_label: str, backend: MMPropertyBackend[Any], annotations: list[Any]
) -> None:
    """Consume a ``PropName`` annotation and set ``backend._prop``."""
    unhandled = []
    for annotation in annotations:
        if isinstance(annotation, PropName):
            backend._prop = annotation.name
        else:
            unhandled.append(annotation)
    annotations[:] = unhandled
