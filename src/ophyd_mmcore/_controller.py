import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from event_model import Limits
from ophyd_async.core import SignalMetadata
from pymmcore_plus import CMMCorePlus, PropertyType

from ._backend import MMSource

P = ParamSpec("P")
R = TypeVar("R")

EventKey = tuple[str, str]


@dataclass
class Subscription:
    """A callback to run on the loop that was running when it subscribed."""

    loop: asyncio.AbstractEventLoop
    callback: Callable[[Any], None]
    active: bool = True

    def deliver(self, value: Any) -> None:
        """Queue the callback on the loop; do nothing once inactive or closed."""
        if not self.active:
            return
        try:
            self.loop.call_soon_threadsafe(self._run, value)
        except RuntimeError:
            self.active = False

    def _run(self, value: Any) -> None:
        if self.active:
            self.callback(value)


class MMController:
    """The only object that calls the Micro-Manager core on behalf of devices."""

    def __init__(self, core: CMMCorePlus) -> None:
        self._core = core
        self._lock = threading.Lock()
        self._subscriptions: dict[EventKey, list[Subscription]] = {}
        self._property_types: dict[EventKey, type] = {}
        self._closed = False
        events = core.events
        self._connections: list[tuple[Any, Callable[..., None]]] = [
            (events.propertyChanged, self._on_property),
            (events.stagePositionChanged, self._on_stage_position),
            (events.XYStagePositionChanged, self._on_xy_position),
            (events.exposureChanged, self._on_exposure),
            (events.shutterOpenChanged, self._on_shutter),
            (events.configSet, self._on_config_set),
        ]
        for signal, slot in self._connections:
            signal.connect(slot)

    @property
    def core(self) -> CMMCorePlus:
        """The Micro-Manager core the controller calls."""
        return self._core

    async def call(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a core call in a worker thread and return its result."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def subscribe(
        self, key: EventKey, callback: Callable[[Any], None]
    ) -> Callable[[], None]:
        """Run `callback` on the running loop for each event with `key`.

        Returns the function that cancels the subscription. An event that
        arrives after cancelling, after `close` or after the loop closed is
        dropped.
        """
        subscription = Subscription(asyncio.get_running_loop(), callback)
        with self._lock:
            if not self._closed:
                self._subscriptions.setdefault(key, []).append(subscription)

        def unsubscribe() -> None:
            subscription.active = False
            with self._lock:
                subscriptions = self._subscriptions.get(key, [])
                if subscription in subscriptions:
                    subscriptions.remove(subscription)

        return unsubscribe

    def property_source(self, label: str, prop: str) -> MMSource[Any]:
        """Return the source of a device property, `mmcore://<label>/<prop>`."""
        key = (label, prop)

        def property_type() -> type:
            if key not in self._property_types:
                match self._core.getPropertyType(label, prop):
                    case PropertyType.Float:
                        self._property_types[key] = float
                    case PropertyType.Integer:
                        self._property_types[key] = int
                    case _:
                        self._property_types[key] = str
            return self._property_types[key]

        def get() -> Any:
            return property_type()(self._core.getProperty(label, prop))

        def set_(value: Any) -> None:
            self._core.setProperty(label, prop, value)

        def metadata() -> SignalMetadata:
            result = SignalMetadata()
            if self._core.hasPropertyLimits(label, prop):
                result["limits"] = Limits(
                    control={
                        "low": self._core.getPropertyLowerLimit(label, prop),
                        "high": self._core.getPropertyUpperLimit(label, prop),
                    }
                )
            if choices := self._core.getAllowedPropertyValues(label, prop):
                result["choices"] = list(choices)
            return result

        async def read() -> Any:
            return await self.call(get)

        async def write(value: Any) -> None:
            await self.call(set_, value)

        async def infer() -> type:
            return await self.call(property_type)

        async def describe() -> SignalMetadata:
            return await self.call(metadata)

        def subscribe(callback: Callable[[Any], None]) -> Callable[[], None]:
            return self.subscribe(key, lambda raw: callback(property_type()(raw)))

        return MMSource(
            uri=f"mmcore://{label}/{prop}",
            read=read,
            write=write,
            subscribe=subscribe,
            infer=infer,
            metadata=describe,
        )

    def close(self) -> None:
        """Disconnect every core event and drop every subscription."""
        with self._lock:
            self._closed = True
            subscriptions = [s for subs in self._subscriptions.values() for s in subs]
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.active = False
        for signal, slot in self._connections:
            signal.disconnect(slot)
        self._connections.clear()

    def _dispatch(self, key: EventKey, value: Any) -> None:
        with self._lock:
            subscriptions = list(self._subscriptions.get(key, ()))
        for subscription in subscriptions:
            subscription.deliver(value)

    def _on_property(self, label: str, prop: str, value: Any) -> None:
        self._dispatch((label, prop), value)

    def _on_stage_position(self, label: str, position: float) -> None:
        self._dispatch((label, "core:position"), position)

    def _on_xy_position(self, label: str, x: float, y: float) -> None:
        self._dispatch((label, "core:x"), x)
        self._dispatch((label, "core:y"), y)

    def _on_exposure(self, label: str, exposure: float) -> None:
        self._dispatch((label, "core:exposure"), exposure)

    def _on_shutter(self, label: str, is_open: bool) -> None:
        self._dispatch((label, "core:open"), is_open)

    def _on_config_set(self, group: str, preset: str) -> None:
        self._dispatch((group, "core:preset"), preset)
