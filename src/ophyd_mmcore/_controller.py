import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar

import numpy as np
from event_model import Limits
from ophyd_async.core import SignalMetadata
from pymmcore_plus import CMMCorePlus, PropertyType

from ._backend import MMSource
from ._camera_io import SEQUENCE_LENGTH, CameraIO
from ._frames import FrameRouter

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

EventKey = tuple[str, str]


@dataclass
class Subscription:
    """A callback to run on the loop that was running when it subscribed.

    The callback receives the value and the time the event arrived.
    """

    loop: asyncio.AbstractEventLoop
    callback: Callable[[Any, float], None]
    active: bool = True

    def deliver(self, value: Any) -> None:
        """Queue the callback on the loop; do nothing once inactive or closed."""
        if not self.active:
            return
        try:
            self.loop.call_soon_threadsafe(self._run, value, time.time())
        except RuntimeError:
            self.active = False

    def _run(self, value: Any, timestamp: float) -> None:
        if self.active:
            self.callback(value, timestamp)


class MMController:
    """The only object that calls the Micro-Manager core on behalf of devices."""

    def __init__(self, core: CMMCorePlus) -> None:
        self._core = core
        self._lock = threading.Lock()
        self._subscriptions: dict[EventKey, list[Subscription]] = {}
        self._closed = False
        self._targets: dict[tuple[str, str], Any] = {}
        self._xy_locks: dict[str, asyncio.Lock] = {}
        self._cameras: dict[str, CameraIO] = {}
        self._acquiring: str | None = None
        self._restore: tuple[str, bool] | None = None
        self._router = FrameRouter(core, self._deliver)
        self._dropped: set[str] = set()
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

    @classmethod
    def from_config(
        cls, path: str | Path, core: CMMCorePlus | None = None
    ) -> "MMController":
        """Load a Micro-Manager configuration file into a core and wrap it.

        A new core is created when `core` is not given. The load blocks until
        the devices are initialised.
        """
        core = CMMCorePlus() if core is None else core
        core.loadSystemConfiguration(path)
        return cls(core)

    @property
    def core(self) -> CMMCorePlus:
        """The Micro-Manager core the controller calls."""
        return self._core

    async def call(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Run a core call in a worker thread and return its result."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def subscribe(
        self, key: EventKey, callback: Callable[[Any, float], None]
    ) -> Callable[[], None]:
        """Run `callback(value, timestamp)` on the running loop for each event with `key`.

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

        def get() -> Any:
            return self._core.getProperty(label, prop)

        def property_type() -> type:
            match self._core.getPropertyType(label, prop):
                case PropertyType.Float:
                    return float
                case PropertyType.Integer:
                    return int
                case _:
                    return str

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
            if (camera := self._cameras.get(label)) is not None:
                return await camera.read_setting(prop, get)
            return await self.call(get)

        async def write(value: Any) -> None:
            if (camera := self._cameras.get(label)) is not None:
                await camera.write_setting(prop, lambda: set_(value), get)
            else:
                await self.call(set_, value)

        async def infer() -> type:
            return await self.call(property_type)

        async def describe() -> SignalMetadata:
            return await self.call(metadata)

        return MMSource(
            uri=f"mmcore://{label}/{prop}",
            read=read,
            write=write,
            subscribe=lambda callback: self.subscribe(key, callback),
            infer=infer,
            metadata=describe,
        )

    def position_source(
        self, label: str, axis: Literal["z", "x", "y"]
    ) -> MMSource[float]:
        """Return the source of a stage position, in micrometres.

        `axis` is `"z"` for a one-axis stage, and `"x"` or `"y"` for an axis of
        an XY stage. Writing starts the move and returns when the stage is
        idle, and `setpoint` reports the last target written, which starts at
        the position at connect. A write the core rejects leaves the target
        as it was. The two axes of an XY stage move with one core call, so
        writes to them share a lock and each carries the other axis's target.
        """
        getters = {
            "z": self._core.getPosition,
            "x": self._core.getXPosition,
            "y": self._core.getYPosition,
        }
        other = "y" if axis == "x" else "x"

        async def read() -> float:
            position = await self.call(getters[axis], label)
            self._targets.setdefault((label, axis), position)
            return position

        async def setpoint() -> float:
            if (label, axis) not in self._targets:
                await read()
            return self._targets[label, axis]

        async def store_and_move(value: float, move: Callable[[], None]) -> None:
            await self._store_and_move((label, axis), value, move)

        async def write(value: float) -> None:
            if axis == "z":
                await store_and_move(
                    value, lambda: self._core.setPosition(label, value)
                )
            else:
                async with self._xy_locks.setdefault(label, asyncio.Lock()):
                    if (label, other) not in self._targets:
                        self._targets[label, other] = await self.call(
                            getters[other], label
                        )
                    target = self._targets[label, other]
                    x, y = (value, target) if axis == "x" else (target, value)
                    await store_and_move(
                        value, lambda: self._core.setXYPosition(label, x, y)
                    )
            await self.call(self._core.waitForDevice, label)
            # Most adapters and the core send no position event after a move.
            if axis == "z":
                position = await self.call(self._core.getPosition, label)
                self._on_stage_position(label, position)
            else:
                x, y = await self.call(self._core.getXYPosition, label)
                self._on_xy_position(label, x, y)

        async def metadata() -> SignalMetadata:
            return SignalMetadata(units="um")

        name = "position" if axis == "z" else axis
        return MMSource(
            uri=f"mmcore://{label}/{name}",
            read=read,
            write=write,
            setpoint=setpoint,
            subscribe=lambda callback: self.subscribe(
                (label, f"core:{name}"), callback
            ),
            metadata=metadata,
        )

    async def _store_and_move(
        self, key: tuple[str, str], value: Any, move: Callable[[], None]
    ) -> None:
        previous = self._targets.get(key)
        self._targets[key] = value
        try:
            await self.call(move)
        except Exception:
            if previous is None:
                del self._targets[key]
            else:
                self._targets[key] = previous
            raise

    def _moving_source(
        self,
        uri: str,
        key: tuple[str, str],
        get: Callable[[], T],
        move: Callable[[T], None],
        wait: Callable[[T], None],
        events: dict[EventKey, Callable[[Any], T] | None],
        metadata: Callable[[], SignalMetadata] | None = None,
    ) -> MMSource[T]:
        async def read() -> T:
            value = await self.call(get)
            self._targets.setdefault(key, value)
            return value

        async def setpoint() -> T:
            if key not in self._targets:
                await read()
            return self._targets[key]

        async def write(value: T) -> None:
            await self._store_and_move(key, value, lambda: move(value))
            await self.call(wait, value)

        async def describe() -> SignalMetadata:
            return SignalMetadata() if metadata is None else await self.call(metadata)

        def subscribe(callback: Callable[[T, float], None]) -> Callable[[], None]:
            cancels = [
                self.subscribe(
                    key,
                    callback
                    if convert is None
                    else lambda value, time, convert=convert: callback(
                        convert(value), time
                    ),
                )
                for key, convert in events.items()
            ]

            def cancel() -> None:
                for cancel_one in cancels:
                    cancel_one()

            return cancel

        return MMSource(
            uri=uri,
            read=read,
            write=write,
            setpoint=setpoint,
            subscribe=subscribe,
            metadata=describe,
        )

    def shutter_source(self, label: str) -> MMSource[bool]:
        """Return the source of a shutter, true when open.

        Writing opens or closes the shutter and returns when the device is
        idle. `setpoint` reports the last target written, which starts at the
        state at connect, and a write the core rejects leaves it as it was.
        """
        return self._moving_source(
            f"mmcore://{label}/open",
            (label, "open"),
            lambda: self._core.getShutterOpen(label),
            lambda value: self._core.setShutterOpen(label, value),
            lambda value: self._core.waitForDevice(label),
            {
                (label, "core:open"): None,
                (label, "State"): lambda value: bool(int(value)),
            },
        )

    def state_source(self, label: str) -> MMSource[str]:
        """Return the source of a state device's label.

        Writing moves to the state with that label and returns when the device
        is idle. The labels of the device are the `choices` of its metadata.
        `setpoint` behaves as it does for [`shutter_source`][ophyd_mmcore.MMController.shutter_source].
        """
        return self._moving_source(
            f"mmcore://{label}/Label",
            (label, "state"),
            lambda: self._core.getStateLabel(label),
            lambda value: self._core.setStateLabel(label, value),
            lambda value: self._core.waitForDevice(label),
            {(label, "Label"): None},
            lambda: SignalMetadata(choices=list(self._core.getStateLabels(label))),
        )

    def preset_source(self, group: str) -> MMSource[str]:
        """Return the source of a configuration group's current preset.

        Writing applies the preset and returns when every device it touches is
        idle. Reading gives `""` when no preset matches the hardware. The
        presets of the group are the `choices` of its metadata. `setpoint`
        behaves as it does for [`shutter_source`][ophyd_mmcore.MMController.shutter_source].
        """
        return self._moving_source(
            f"mmcore://{group}/preset",
            (group, "preset"),
            lambda: self._core.getCurrentConfig(group),
            lambda value: self._core.setConfig(group, value),
            lambda value: self._core.waitForConfig(group, value),
            {(group, "core:preset"): None},
            lambda: SignalMetadata(choices=list(self._core.getAvailableConfigs(group))),
        )

    def halt(self, label: str) -> Callable[[], Awaitable[None]]:
        """Return the call that stops the device `label`.

        The targets written to the device are dropped, so the next setpoint
        read and the next move of another XY axis start from the hardware.
        """

        async def run() -> None:
            await self.call(self._core.stop, label)
            for key in [key for key in self._targets if key[0] == label]:
                del self._targets[key]

        return run

    def camera_io(self, label: str, physical: Sequence[str] = ()) -> CameraIO:
        """Return the sequence flags and stores of the camera `label`.

        The camera has one output per label in `physical`, or one output for
        `label` itself when `physical` is empty. Reads and writes of the
        camera's properties go through it from the first call on.

        Raises
        ------
        ValueError
            If an earlier call for `label` named other physical cameras.
        """
        outputs = tuple(physical) or (label,)
        if (camera := self._cameras.get(label)) is None:
            camera = self._cameras[label] = CameraIO(self, label, outputs)
        if camera.physical != outputs:
            raise ValueError(
                f"Camera {label!r} has the outputs {camera.physical}, not {outputs}"
            )
        return camera

    def _sequence_running(self) -> bool:
        return self._acquiring is not None

    async def _start_sequence(self, label: str, prepare: Callable[[], None]) -> None:
        """Make `label` the current camera with AutoShutter off and start its sequence.

        `prepare` runs once the camera is current, before the sequence starts.
        """
        if self._acquiring is not None:
            raise RuntimeError(
                f"Camera {self._acquiring!r} is already acquiring, and only one "
                f"camera per core can run a sequence; stop it before {label!r}"
            )
        self._acquiring = label
        try:
            await self.call(self._begin_sequence, label, prepare)
        except Exception:
            self._acquiring = None
            raise
        self._router.start()

    def _begin_sequence(self, label: str, prepare: Callable[[], None]) -> None:
        # Recorded before the core changes, so that `close` restores it even
        # when the caller stopped waiting for this call.
        restore = self._restore = (
            self._core.getCameraDevice(),
            self._core.getAutoShutter(),
        )
        self._core.setCameraDevice(label)
        self._core.setAutoShutter(False)
        try:
            prepare()
            self._core.startSequenceAcquisition(label, SEQUENCE_LENGTH, 0, False)
        except Exception:
            self._restore = None
            self._core.setCameraDevice(restore[0])
            self._core.setAutoShutter(restore[1])
            raise

    async def _stop_sequence(self, label: str) -> None:
        """Stop the sequence of `label` and restore the camera and AutoShutter."""
        if self._acquiring == label:
            await self.call(self._end_sequence)

    def _end_sequence(self) -> None:
        restore, self._restore = self._restore, None
        try:
            if self._acquiring is not None:
                self._core.stopSequenceAcquisition(self._acquiring)
        finally:
            try:
                self._router.stop()
                if restore is not None:
                    self._core.setCameraDevice(restore[0])
                    self._core.setAutoShutter(restore[1])
            finally:
                self._acquiring = None

    def _deliver(self, physical: str, frame: np.ndarray) -> None:
        camera = self._cameras.get(self._acquiring or "")
        delivered = camera is not None and camera.deliver(physical, frame)
        if not delivered and physical not in self._dropped:
            self._dropped.add(physical)
            logger.warning("Dropping the frames of %r, which has no output", physical)

    def close(self) -> None:
        """Stop the acquiring camera, close every store and drop every subscription.

        The camera and AutoShutter current before the sequence started are
        restored. Closing again does nothing.
        """
        for camera in self._cameras.values():
            camera.close()
        self._end_sequence()
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
