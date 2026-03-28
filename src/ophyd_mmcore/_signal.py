"""Factory functions for creating ophyd-async Signals backed by MM properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ophyd_async.core import DEFAULT_TIMEOUT, SignalR, SignalRW, SignalW

from ._backend import MMPropertyBackend, PrimitiveT

if TYPE_CHECKING:
    from ._worker import MMCoreWorker


def mmcore_signal_rw(
    datatype: type[PrimitiveT],
    device_label: str,
    property_name: str,
    worker: MMCoreWorker,
    name: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> SignalRW[PrimitiveT]:
    """Create a read-write Signal backed by a Micro-Manager device property.

    Parameters
    ----------
    datatype:
        The Python type for the signal value (``str``, ``float``, or ``int``).
    device_label:
        The MM device label (e.g. ``"Camera"``).
    property_name:
        The MM property name (e.g. ``"Exposure"``).
    worker:
        The shared ``MMCoreWorker`` for this application.
    name:
        Optional signal name.
    timeout:
        Timeout in seconds for read operations.
    """
    backend: MMPropertyBackend[PrimitiveT] = MMPropertyBackend(
        device_label, property_name, worker, datatype
    )
    return SignalRW(backend, name=name, timeout=timeout)


def mmcore_signal_r(
    datatype: type[PrimitiveT],
    device_label: str,
    property_name: str,
    worker: MMCoreWorker,
    name: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> SignalR[PrimitiveT]:
    """Create a read-only Signal backed by a Micro-Manager device property.

    Parameters
    ----------
    datatype:
        The Python type for the signal value (``str``, ``float``, or ``int``).
    device_label:
        The MM device label (e.g. ``"Camera"``).
    property_name:
        The MM property name (e.g. ``"Exposure"``).
    worker:
        The shared ``MMCoreWorker`` for this application.
    name:
        Optional signal name.
    timeout:
        Timeout in seconds for read operations.
    """
    backend: MMPropertyBackend[PrimitiveT] = MMPropertyBackend(
        device_label, property_name, worker, datatype
    )
    return SignalR(backend, name=name, timeout=timeout)


def mmcore_signal_w(
    datatype: type[PrimitiveT],
    device_label: str,
    property_name: str,
    worker: MMCoreWorker,
    name: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> SignalW[PrimitiveT]:
    """Create a write-only Signal backed by a Micro-Manager device property.

    Parameters
    ----------
    datatype:
        The Python type for the signal value (``str``, ``float``, or ``int``).
    device_label:
        The MM device label (e.g. ``"Camera"``).
    property_name:
        The MM property name (e.g. ``"Exposure"``).
    worker:
        The shared ``MMCoreWorker`` for this application.
    name:
        Optional signal name.
    timeout:
        Timeout in seconds for write operations.
    """
    backend: MMPropertyBackend[PrimitiveT] = MMPropertyBackend(
        device_label, property_name, worker, datatype
    )
    return SignalW(backend, name=name, timeout=timeout)


def mmcore_signal_auto(
    device_label: str,
    property_name: str,
    worker: MMCoreWorker,
    name: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> SignalRW[PrimitiveT] | SignalR[PrimitiveT]:
    """Create a Signal whose type and read/write access are inferred from MM.

    The datatype is inferred from ``PropertyType`` (falls back to ``str``).
    The signal is ``SignalR`` if the property is read-only, ``SignalRW`` otherwise.

    Parameters
    ----------
    device_label:
        The MM device label.
    property_name:
        The MM property name.
    worker:
        The shared ``MMCoreWorker`` for this application.
    name:
        Optional signal name.
    timeout:
        Timeout in seconds for read operations.
    """
    core = worker.core
    backend: MMPropertyBackend[PrimitiveT] = MMPropertyBackend(
        device_label,
        property_name,
        worker,  # datatype inferred inside __init__
    )
    if core.isPropertyReadOnly(device_label, property_name):
        return SignalR(backend, name=name, timeout=timeout)
    return SignalRW(backend, name=name, timeout=timeout)
