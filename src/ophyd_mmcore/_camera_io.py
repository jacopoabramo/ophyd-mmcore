import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

import acquire_zarr as az
import numpy as np
from ophyd_async.core import SignalMetadata

from ._backend import CameraState, MMSource

if TYPE_CHECKING:
    from ._controller import MMController

SEQUENCE_LENGTH: Final = 2**31 - 1
"""The number of frames a sequence asks for, the largest Micro-Manager accepts."""

CameraValueName = Literal[
    "exposure", "acquire", "capture", "num_capture", "state", "last_error"
]
"""The camera values a [`CameraIO`][ophyd_mmcore._camera_io.CameraIO] serves."""

OutputValueName = Literal[
    "width", "height", "dtype_numpy", "file_path", "data_key", "captured"
]
"""The values of one output a [`CameraIO`][ophyd_mmcore._camera_io.CameraIO] serves."""

T = TypeVar("T")

logger = logging.getLogger(__name__)


class FrameStore:
    """Zarr store with one array of frames along an unbounded time axis.

    The array is named `data_key`, has the dimensions `t, y, x` and holds one
    frame per chunk.

    Raises
    ------
    FileExistsError
        If `path` exists.
    """

    def __init__(
        self, path: Path, data_key: str, shape: tuple[int, int], dtype: np.dtype
    ) -> None:
        if path.exists():
            raise FileExistsError(f"Store {path} already exists")
        height, width = shape
        self._key = data_key
        self._shape = shape
        self._dtype = dtype
        self._stream = az.ZarrStream(
            az.StreamSettings(
                store_path=str(path),
                overwrite=False,
                arrays=[
                    az.ArraySettings(
                        output_key=data_key,
                        data_type=dtype,
                        dimensions=[
                            dimension("t", az.DimensionType.TIME, 0, 1),
                            dimension("y", az.DimensionType.SPACE, height, height),
                            dimension("x", az.DimensionType.SPACE, width, width),
                        ],
                    )
                ],
            )
        )

    def append(self, frame: np.ndarray) -> None:
        """Append one frame.

        Raises
        ------
        ValueError
            If the frame's shape or dtype differs from the store's.
        """
        if frame.shape != self._shape or frame.dtype != self._dtype:
            raise ValueError(
                f"A frame of shape {frame.shape} and dtype {frame.dtype} does not "
                f"fit a store of shape {self._shape} and dtype {self._dtype}"
            )
        self._stream.append(frame, key=self._key)

    def close(self) -> None:
        """Write what is buffered and close the store."""
        self._stream.close()


def dimension(name: str, kind: az.DimensionType, size: int, chunk: int) -> az.Dimension:
    """Return a dimension of `size` pixels in chunks of `chunk`, 0 for unbounded."""
    return az.Dimension(
        name=name,
        kind=kind,
        array_size_px=size,
        chunk_size_px=chunk,
        shard_size_chunks=1,
    )


@dataclass(frozen=True)
class Layout:
    """Layout of the frames a physical camera sends."""

    width: int
    height: int
    bytes_per_pixel: int
    components: int
    pixel_type: str = field(compare=False)

    @property
    def dtype(self) -> np.dtype:
        """Unsigned integer dtype of one pixel component."""
        return np.dtype(f"<u{self.bytes_per_pixel}")

    @property
    def dtype_numpy(self) -> str:
        """The dtype as a numpy type string such as `"<u2"`."""
        return self.dtype.str


@dataclass
class Output:
    """The store of one physical camera and what was written to it."""

    file_path: str = ""
    data_key: str = ""
    captured: int = 0
    written: int = 0
    store: FrameStore | None = None


class CameraIO:
    """Sequence flags, cached settings and stores of one camera.

    `acquire` runs the camera's sequence and `capture` opens the gate that
    lets frames into the stores, one per physical camera. A setting written
    while the camera acquires is written between a stop and a restart of the
    sequence, and refused while it captures. While it acquires, a setting
    reads the value last written or read with no sequence running. A
    physical camera of the acquiring camera counts as acquiring and
    capturing with it. An output's store cannot be replaced while the camera captures, and clearing
    its `file_path` then closes the gate before the store.

    A frame that cannot be written, or frames lost because the core's
    circular buffer overflowed, stop the sequence and fault the camera with
    the error as `last_error`, until `acquire` is set to true again.
    """

    def __init__(
        self, ctrl: "MMController", label: str, physical: tuple[str, ...]
    ) -> None:
        self._ctrl = ctrl
        self.label = label
        self.physical = physical
        self._outputs = {name: Output() for name in physical}
        self._settings: dict[str, Any] = {}
        self._layouts: dict[str, Layout] = {}
        self._values: dict[str, Any] = {
            "acquire": False,
            "capture": False,
            "num_capture": 0,
            "last_error": "",
            "state": CameraState.IDLE,
        }
        self._faulted = False
        # The frame thread and the event loop both change flags and outputs.
        self._lock = threading.Lock()
        self._write_lock = asyncio.Lock()

    def _sequence_camera(self) -> "CameraIO":
        """Return the acquiring camera when it sends this camera's frames, else this camera."""
        acquiring = self._ctrl._cameras.get(self._ctrl._acquiring or "")
        if acquiring is not None and self.label in acquiring.physical:
            return acquiring
        return self

    @property
    def _running(self) -> bool:
        camera = self._sequence_camera()
        return camera._values["acquire"] and not camera._faulted

    def source(self, name: CameraValueName) -> MMSource[Any]:
        """Return the source of the camera's exposure or one of its flags."""
        if name == "exposure":
            return self._exposure_source()
        key = (self.label, f"camera:{name}")

        async def read() -> Any:
            return self._values[name]

        writers = {
            "acquire": partial(self._serialised, self._change_acquire),
            "capture": partial(self._serialised, self._change_capture),
            "num_capture": self._write_num_capture,
        }
        return MMSource(
            uri=f"mmcore://{self.label}/{name}",
            read=read,
            write=writers.get(name),
            subscribe=lambda callback: self._ctrl.subscribe(key, callback),
        )

    def output_source(self, physical: str, name: OutputValueName) -> MMSource[Any]:
        """Return the source of a frame layout value or of the store of one output.

        Raises
        ------
        ValueError
            If the camera has no output for `physical`.
        """
        if physical not in self._outputs:
            raise ValueError(f"Camera {self.label!r} has no output {physical!r}")
        output = self._outputs[physical]
        key = (self.label, f"output:{physical}:{name}")

        async def read() -> Any:
            if name in ("width", "height", "dtype_numpy"):
                return getattr((await self._read_layouts())[physical], name)
            return getattr(output, name)

        async def write_file_path(value: str) -> None:
            await self._serialised(self._change_file_path, physical, value)

        async def write_data_key(value: str) -> None:
            with self._lock:
                output.data_key = value
            self._ctrl._dispatch(key, value)

        writers = {"file_path": write_file_path, "data_key": write_data_key}
        return MMSource(
            uri=f"mmcore://{self.label}/{physical}/{name}",
            read=read,
            write=writers.get(name),
            subscribe=lambda callback: self._ctrl.subscribe(key, callback),
        )

    async def read_setting(self, key: str, get: Callable[[], Any]) -> Any:
        """Read a setting, from the cache while the camera acquires."""
        running = self._running
        if running and key in self._settings:
            return self._settings[key]
        value = await self._ctrl.call(get)
        if not running:
            self._settings[key] = value
        return value

    async def write_setting(
        self, key: str, set_: Callable[[], None], get: Callable[[], Any]
    ) -> Any:
        """Write a setting, pausing the sequence if one runs, and return its new value.

        Raises
        ------
        RuntimeError
            If the camera is capturing.
        """
        return await self._sequence_camera()._serialised(
            self._change_setting, key, set_, get
        )

    def deliver(self, physical: str, frame: np.ndarray) -> bool:
        """Append a frame to its output's store while the gate is open.

        Runs on the frame thread. Returns false when the camera has no output
        for `physical`.
        """
        output = self._outputs.get(physical)
        if output is None:
            return False
        error = ""
        with self._lock:
            limit = self._values["num_capture"]
            if (
                not self._values["capture"]
                or output.store is None
                or (limit and output.written >= limit)
            ):
                return True
            try:
                output.store.append(frame)
            except Exception as exception:
                error = f"Cannot write a frame of camera {physical!r}: {exception}"
            else:
                output.written += 1
                output.captured += 1
                self._ctrl._dispatch(
                    (self.label, f"output:{physical}:captured"), output.captured
                )
                if limit and all(o.written >= limit for o in self._outputs.values()):
                    self._update(capture=False)
        if error:
            self.fault(error)
        return True

    def fault(self, error: str) -> None:
        """Close the gate, stop the sequence and fault the camera with `error`."""
        with self._lock:
            self._faulted = True
            self._update(capture=False, last_error=error)
        self._ctrl.core.stopSequenceAcquisition(self.label)

    def close(self) -> None:
        """Close the gate and every store, and clear `acquire`."""
        with self._lock:
            self._update(capture=False, acquire=False)
            stores = [o.store for o in self._outputs.values() if o.store is not None]
            for output in self._outputs.values():
                output.store = None
        for store in stores:
            try:
                store.close()
            except Exception:
                logger.exception("Closing a store of camera %r failed", self.label)

    def _update(self, **changes: Any) -> None:
        """Change flags and send each changed one to subscribers; hold the lock."""
        values = {**self._values, **changes}
        if self._faulted:
            values["state"] = CameraState.FAULTED
        elif values["capture"]:
            values["state"] = CameraState.CAPTURING
        elif values["acquire"]:
            values["state"] = CameraState.ACQUIRING
        else:
            values["state"] = CameraState.IDLE
        for name, value in values.items():
            if value != self._values[name]:
                self._values[name] = value
                self._ctrl._dispatch((self.label, f"camera:{name}"), value)

    def _exposure_source(self) -> MMSource[float]:
        core = self._ctrl.core
        key = (self.label, "core:exposure")

        def get() -> float:
            return core.getExposure(self.label)

        async def read() -> float:
            return await self.read_setting("exposure", get)

        async def write(value: float) -> None:
            exposure = await self.write_setting(
                "exposure", lambda: core.setExposure(self.label, value), get
            )
            # UniMMCore emits no exposureChanged for a Python camera.
            self._ctrl._dispatch(key, exposure)

        async def metadata() -> SignalMetadata:
            return SignalMetadata(units="ms")

        return MMSource(
            uri=f"mmcore://{self.label}/exposure",
            read=read,
            write=write,
            subscribe=lambda callback: self._ctrl.subscribe(key, callback),
            metadata=metadata,
        )

    async def _serialised(self, change: Callable[..., Awaitable[T]], *args: Any) -> T:
        """Run a change that touches the sequence or the stores, one at a time.

        A cancelled caller waits for the change to finish before the
        cancellation is raised, because the core calls it makes run on in
        their threads and the flags must end up matching them. An error of
        the change is then logged.
        """
        async with self._write_lock:
            task = asyncio.ensure_future(change(*args))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.wait({task})
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                if not task.cancelled() and (error := task.exception()) is not None:
                    logger.error(
                        "A change of camera %r failed after its caller was cancelled",
                        self.label,
                        exc_info=error,
                    )
                raise asyncio.CancelledError
            return task.result()

    async def _change_setting(
        self, key: str, set_: Callable[[], None], get: Callable[[], Any]
    ) -> Any:
        camera = self._sequence_camera()
        if camera._values["capture"]:
            raise RuntimeError(
                f"Cannot change {key} of camera {self.label!r} while camera "
                f"{camera.label!r} captures"
            )
        value = await self._ctrl.call(camera._write_paused, set_, get, self._running)
        self._settings[key] = value
        return value

    async def _change_acquire(self, value: bool) -> None:
        if self._values["acquire"]:
            if value and not self._faulted:
                return
            with self._lock:
                self._update(capture=False)
            try:
                await self._ctrl._stop_sequence(self.label)
            finally:
                with self._lock:
                    self._update(acquire=False)
        if value:
            await self._ctrl._start_sequence(self.label, self._refresh_layouts)
            with self._lock:
                self._faulted = False
                self._update(acquire=True, last_error="")

    async def _change_capture(self, value: bool) -> None:
        if not value:
            with self._lock:
                self._update(capture=False)
            return
        if self._faulted:
            raise RuntimeError(
                f"Camera {self.label!r} faulted: {self._values['last_error']}"
            )
        if not self._values["acquire"]:
            raise RuntimeError(f"Camera {self.label!r} is not acquiring")
        if missing := [p for p, o in self._outputs.items() if not o.file_path]:
            raise RuntimeError(
                f"Camera {self.label!r} has no file_path for {', '.join(missing)}"
            )
        await self._ctrl.call(self._open_gate)

    async def _write_num_capture(self, value: int) -> None:
        with self._lock:
            self._update(num_capture=value)
            written = [o.written for o in self._outputs.values()]
            if self._values["capture"] and value and min(written) >= value:
                self._update(capture=False)

    async def _change_file_path(self, physical: str, path: str) -> None:
        if self._values["capture"]:
            if path:
                raise RuntimeError(
                    f"Cannot change the store of camera {self.label!r} while it "
                    "captures"
                )
            # Unstaging clears file_path and the gate at once, so closing the
            # store also ends the capture rather than racing with it.
            with self._lock:
                self._update(capture=False)
        if path:
            self._check_layouts(physical, await self._read_layouts())
            if Path(path).exists():
                raise FileExistsError(f"Store {path} already exists")
        output = self._outputs[physical]
        await self._ctrl.call(self._close_store, output)
        with self._lock:
            output.file_path = path
            if path:
                output.captured = 0
        for name in ("file_path", "captured"):
            self._ctrl._dispatch(
                (self.label, f"output:{physical}:{name}"), getattr(output, name)
            )

    def _check_layouts(self, physical: str, layouts: dict[str, Layout]) -> None:
        layout = layouts[physical]
        # shortcut: only grayscale frames of 1, 2 or 4 bytes per pixel are
        # written, upgrade when a colour camera has to write frames.
        if layout.components > 1 or layout.bytes_per_pixel not in (1, 2, 4):
            raise ValueError(
                f"Cannot write frames of camera {physical!r} with pixel type "
                f"{layout.pixel_type}: only grayscale frames of 1, 2 or 4 bytes "
                "per pixel can be written"
            )
        if len(set(layouts.values())) > 1:
            described = ", ".join(
                f"{name!r} ({each.width} x {each.height}, {each.dtype_numpy})"
                for name, each in layouts.items()
            )
            raise ValueError(
                f"The cameras of {self.label!r} send frames of different layouts: "
                f"{described}"
            )

    async def _read_layouts(self) -> dict[str, Layout]:
        """Return the layout of every output, from the cache while a sequence runs."""
        if not self._ctrl._sequence_running():
            await self._ctrl.call(self._refresh_layouts)
        if missing := [p for p in self.physical if p not in self._layouts]:
            raise RuntimeError(
                f"The frame layout of {', '.join(missing)} is unknown while a "
                "sequence runs"
            )
        return {name: self._layouts[name] for name in self.physical}

    def _refresh_layouts(self) -> None:
        """Read every output's layout, making each physical camera current in turn."""
        with self._ctrl._camera_lock:
            core = self._ctrl.core
            current = core.getCameraDevice()
            try:
                for name in self.physical:
                    if core.getCameraDevice() != name:
                        core.setCameraDevice(name)
                    components = core.getNumberOfComponents()
                    layout = Layout(
                        width=core.getImageWidth(),
                        height=core.getImageHeight(),
                        bytes_per_pixel=core.getBytesPerPixel(),
                        components=components,
                        pixel_type=core.getProperty(name, "PixelType")
                        if core.hasProperty(name, "PixelType")
                        else ("RGB" if components > 1 else "grayscale"),
                    )
                    previous = self._layouts.get(name)
                    self._layouts[name] = layout
                    for value in ("width", "height", "dtype_numpy"):
                        if previous is None or getattr(previous, value) != getattr(
                            layout, value
                        ):
                            self._ctrl._dispatch(
                                (self.label, f"output:{name}:{value}"),
                                getattr(layout, value),
                            )
            finally:
                if core.getCameraDevice() != current:
                    core.setCameraDevice(current)

    def _write_paused(
        self, set_: Callable[[], None], get: Callable[[], Any], pause: bool
    ) -> Any:
        if not pause:
            set_()
            return get()
        core = self._ctrl.core
        core.stopSequenceAcquisition(self.label)
        try:
            set_()
            value = get()
            self._refresh_layouts()
        finally:
            core.startSequenceAcquisition(self.label, SEQUENCE_LENGTH, 0, False)
        return value

    def _open_gate(self) -> None:
        with self._lock:
            for name, output in self._outputs.items():
                if output.store is None:
                    layout = self._layouts[name]
                    output.store = FrameStore(
                        Path(output.file_path),
                        output.data_key,
                        (layout.height, layout.width),
                        layout.dtype,
                    )
                output.written = 0
            # shortcut: a frame already being exposed when the gate opens can be
            # written, upgrade to a check against the frame's timestamp if motion
            # blur appears.
            self._update(capture=True)

    def _close_store(self, output: Output) -> None:
        with self._lock:
            store, output.store = output.store, None
        if store is not None:
            store.close()
