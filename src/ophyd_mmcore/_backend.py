from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

from ophyd_async.core import SignalBackend, make_datakey
from ophyd_async.core._signal_backend import Primitive, PrimitiveT, make_metadata

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from bluesky.protocols import Reading
    from event_model import DataKey
    from ophyd_async.core import Callback
    from pymmcore_plus import CMMCorePlus

    from ._worker import MMCoreWorker


class MMPropertyBackend(SignalBackend[PrimitiveT]):
    """ophyd-async ``SignalBackend`` for a single Micro-Manager device property.

    Micro-Manager properties are always one of ``str``, ``float``, or ``int``
    (i.e. :data:`~ophyd_async.core.Primitive`), so this backend is typed over
    ``PrimitiveT`` rather than the full ``SignalDatatypeT``.

    Parameters
    ----------
    device_label:
        The MM device label (e.g. ``"Camera"``).
    property_name:
        The MM property name (e.g. ``"Exposure"``).
    worker:
        The shared ``MMCoreWorker``.  Inject one instance per application so
        that all backends serialise through the same thread.
    datatype:
        The Python type for signal values.  Inferred from ``PropertyType`` if
        not given (falls back to ``str`` for undefined properties).
    """

    def __init__(
        self,
        label: str,
        property_name: str,
        worker: MMCoreWorker,
        datatype: type[PrimitiveT] | None = None,
    ) -> None:
        self._dev = label
        self._prop = property_name
        self._worker = worker

        if datatype is None:
            pt = worker.core.getPropertyType(label, property_name)
            # to_python() returns str | float | int | None; fall back to str
            inferred: type[Primitive] = pt.to_python() or str
            datatype = cast("type[PrimitiveT]", inferred)

        self._reading_callback: Callback[Reading[PrimitiveT]] | None = None
        # The psygnal slot we connected (kept so we can disconnect it later)
        self._psygnal_slot: Callable[..., None] | None = None
        # We need the event loop when wiring callbacks; captured on connect()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._getter: Callable[[CMMCorePlus, str], Any] | None = None
        self._setter: Callable[[CMMCorePlus, str, Any], None] | None = None
        self._event_prop: str = self._prop
        self._event_factory: Callable[[CMMCorePlus], Any] | None = None
        self._event_value: Callable[..., Any] | None = None

        super().__init__(datatype)

    def configure_prop(self, name: str) -> None:
        """Configure the backend to use the generic property API for *name*."""
        self._prop = name
        self._event_prop = name

    def configure_core_method(
        self,
        getter: Callable[[CMMCorePlus, str], Any],
        setter: Callable[[CMMCorePlus, str, Any], None] | None,
        event_factory: Callable[[CMMCorePlus], Any] | None,
        event_value: Callable[..., Any] | None = None,
    ) -> None:
        """Configure the backend to use dedicated core API methods."""
        self._getter = getter
        self._setter = setter
        self._event_factory = event_factory
        self._event_value = event_value

    def source(self, name: str, read: bool) -> str:
        """Return the source URI for this signal."""
        return f"mmcore://{self._dev}/{self._prop}"

    async def connect(self, timeout: float) -> None:
        """Connect to the device property, verifying it is reachable."""
        self._loop = asyncio.get_running_loop()
        if (getter := self._getter) is not None:
            await self._worker.run(lambda: getter(self._worker.core, self._dev))
        else:
            await self._worker.run(
                lambda: self._worker.core.getProperty(self._dev, self._prop)
            )

    async def put(self, value: PrimitiveT | None) -> None:
        """Write *value* to the MM property."""
        if (setter := self._setter) is not None:
            await self._worker.run(lambda: setter(self._worker.core, self._dev, value))
        else:
            await self._worker.run(
                lambda: self._worker.core.setProperty(
                    self._dev, self._prop, cast("Primitive", value)
                )
            )

    async def get_value(self) -> PrimitiveT:
        """Return the current property value cast to the declared datatype."""
        assert self.datatype is not None
        if (getter := self._getter) is not None:
            raw = await self._worker.run(lambda: getter(self._worker.core, self._dev))
            return cast("PrimitiveT", self.datatype(raw))
        raw_str: str = await self._worker.run(
            lambda: self._worker.core.getProperty(self._dev, self._prop)
        )
        return cast("PrimitiveT", self.datatype(raw_str))

    async def get_setpoint(self) -> PrimitiveT:
        """Return the setpoint (same as readback for MM properties)."""
        return await self.get_value()

    async def get_reading(self) -> Reading[PrimitiveT]:
        """Return a Bluesky Reading with the current value and timestamp."""
        value = await self.get_value()
        return {
            "value": value,
            "timestamp": time.time(),
            "alarm_severity": 0,
        }

    async def get_datakey(self, source: str) -> DataKey:
        """Return event-model DataKey metadata for this signal."""
        assert self.datatype is not None
        dev, prop = self._dev, self._prop

        def _read_all() -> tuple[str, bool, float, float, tuple[str, ...]]:
            core = self._worker.core
            raw = core.getProperty(dev, prop)
            has_limits = core.hasPropertyLimits(dev, prop)
            lo = core.getPropertyLowerLimit(dev, prop) if has_limits else 0.0
            hi = core.getPropertyUpperLimit(dev, prop) if has_limits else 0.0
            allowed = core.getAllowedPropertyValues(dev, prop)
            return raw, has_limits, lo, hi, allowed

        raw, has_limits, lo, hi, allowed = await self._worker.run(_read_all)

        value = cast("PrimitiveT", self.datatype(raw))
        metadata = make_metadata(self.datatype)

        if has_limits:
            from event_model import Limits, LimitsRange

            metadata["limits"] = Limits(
                control=LimitsRange(low=lo, high=hi),
                display=LimitsRange(low=lo, high=hi),
            )

        if allowed:
            metadata["choices"] = list(allowed)

        return make_datakey(self.datatype, value, source, metadata)

    def set_callback(self, callback: Callback[Reading[PrimitiveT]] | None) -> None:
        """Wire or unwire the per-property psygnal change signal.

        The psygnal slot fires on the worker thread; the Reading is delivered
        back to the asyncio loop via ``call_soon_threadsafe``.
        """
        core = self._worker.core

        # Always disconnect the existing slot first
        if self._psygnal_slot is not None:
            if self._event_factory is not None:
                self._event_factory(core).disconnect(self._psygnal_slot)
            elif self._event_prop:
                core.events.devicePropertyChanged(
                    self._dev, self._event_prop
                ).disconnect(self._psygnal_slot)
            self._psygnal_slot = None
        self._reading_callback = None

        if callback is None:
            return

        loop = self._loop
        if loop is None:
            raise RuntimeError(
                "set_callback() called before connect(); "
                "the event loop has not been captured yet."
            )

        datatype = self.datatype
        assert datatype is not None
        label = self._dev

        if self._event_factory is not None:
            # CoreMethod path: signal emits (label, *args) — filter by label,
            # then extract the value via event_value (if given) or first arg.
            event_value_fn = self._event_value

            def _on_method_event(event_label: str, *args: Any) -> None:
                if event_label != label:
                    return
                raw = event_value_fn(*args) if event_value_fn is not None else args[0]
                reading: Reading[PrimitiveT] = {
                    "value": cast("PrimitiveT", datatype(raw)),
                    "timestamp": time.time(),
                    "alarm_severity": 0,
                }
                loop.call_soon_threadsafe(cast("Any", callback), reading)

            self._event_factory(core).connect(_on_method_event)
            self._psygnal_slot = _on_method_event
        elif self._event_prop:
            # PropName path: devicePropertyChanged emits (device, prop, value)
            def _on_property_changed(new_value: str) -> None:
                reading = {
                    "value": cast("PrimitiveT", datatype(new_value)),
                    "timestamp": time.time(),
                    "alarm_severity": 0,
                }
                loop.call_soon_threadsafe(cast("Any", callback), reading)

            core.events.devicePropertyChanged(self._dev, self._event_prop).connect(
                _on_property_changed
            )
            self._psygnal_slot = _on_property_changed

        self._reading_callback = callback

        getter = self._getter

        def _deliver_initial() -> None:
            if getter is not None:
                raw = getter(core, self._dev)
                value = cast("PrimitiveT", datatype(raw))
            else:
                value = cast(
                    "PrimitiveT", datatype(core.getProperty(self._dev, self._prop))
                )
            loop.call_soon_threadsafe(
                cast("Any", callback),
                {"value": value, "timestamp": time.time(), "alarm_severity": 0},
            )

        # Immediately deliver the current value so the subscriber is
        # synchronised on registration (matches SoftSignalBackend behaviour)
        self.submit(_deliver_initial)

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        """Direct access to the worker queue (for subclasses or tests)."""
        return self._worker.submit(fn)
