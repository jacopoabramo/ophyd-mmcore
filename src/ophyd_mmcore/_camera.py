import asyncio
import contextlib
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated as A

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTriggerLogic,
    Device,
    DeviceMap,
    PathProvider,
    SignalR,
    SignalRW,
    StandardDetector,
    StreamableDataProvider,
    StreamResourceDataProvider,
    StreamResourceInfo,
    TriggerInfo,
    WatchableAsyncStatus,
    WatcherUpdate,
    wait_for_value,
)

from ._backend import CameraState
from ._connector import CameraValue, MMConnector, OutputValue, clean_name
from ._controller import MMController
from ._devices import property_map


class MMCameraOutput(Device):
    """The frames of one physical camera and the store they are written to.

    `captured` counts the frames written since `file_path` was set.
    """

    width: A[SignalR[int], OutputValue("width")]
    height: A[SignalR[int], OutputValue("height")]
    dtype_numpy: A[SignalR[str], OutputValue("dtype_numpy")]
    file_path: A[SignalRW[str], OutputValue("file_path")]
    data_key: A[SignalRW[str], OutputValue("data_key")]
    captured: A[SignalR[int], OutputValue("captured")]

    def __init__(self, *, connector: MMConnector, name: str = "") -> None:
        super().__init__(name=name, connector=connector)


@dataclass
class MMTriggerLogic(DetectorTriggerLogic):
    """Set the exposure and the number of frames to write per output."""

    exposure: SignalRW[float]
    num_capture: SignalRW[int]
    outputs: Sequence[MMCameraOutput]

    def config_sigs(self) -> set[SignalR]:
        """Return the exposure and every output's frame layout."""
        return {
            self.exposure,
            *(
                signal
                for output in self.outputs
                for signal in (output.width, output.height, output.dtype_numpy)
            ),
        }

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Set the exposure to `livetime` unless it is 0, and write `num` frames, 0 for no limit."""
        if livetime:
            await self.exposure.set(livetime * 1000)
        await self.num_capture.set(num)

    async def default_trigger_info(self) -> TriggerInfo:
        """Return the trigger info of one event, which a step scan triggers."""
        return TriggerInfo()


@dataclass
class MMAcquireLogic(DetectorAcquireLogic):
    """Run the camera's sequence from stage to unstage and open the gate to the stores."""

    acquire: SignalRW[bool]
    capture: SignalRW[bool]
    num_capture: SignalR[int]
    state: SignalR[CameraState]
    last_error: SignalR[str]

    async def ensure_ready(self) -> None:
        """Start the camera's sequence."""
        await self.acquire.set(True)

    async def start_acquiring(self) -> None:
        """Open the gate to the stores."""
        await self.capture.set(True)

    async def wait_for_idle(self) -> None:
        """Wait for a bounded capture to finish, or end an unbounded one.

        Raises
        ------
        RuntimeError
            With the camera's last error, if the camera faulted.
        """
        if await self.num_capture.get_value():
            await wait_for_value(self.capture, False, timeout=DEFAULT_TIMEOUT)
        else:
            await self.capture.set(False)
        if await self.state.get_value() == CameraState.FAULTED:
            raise RuntimeError(await self.last_error.get_value())

    async def ensure_stopped(self) -> None:
        """Close the gate and stop the sequence."""
        await self.capture.set(False)
        await self.acquire.set(False)


@dataclass
class MMDataLogic(DetectorDataLogic):
    """Name the store of one output with the path provider and count what it writes."""

    output: MMCameraOutput
    path_provider: PathProvider
    datakey_suffix: str = ""

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        """Point the output at a new zarr store whose array is named `datakey_name`."""
        info = self.path_provider(datakey_name)
        Path(info.directory_path).mkdir(parents=True, exist_ok=True)
        filename = f"{info.filename}.zarr"
        await self.output.data_key.set(datakey_name)
        await self.output.file_path.set(str(Path(info.directory_path) / filename))
        await wait_for_value(self.output.captured, 0, timeout=DEFAULT_TIMEOUT)
        width, height, dtype_numpy = await asyncio.gather(
            self.output.width.get_value(),
            self.output.height.get_value(),
            self.output.dtype_numpy.get_value(),
        )
        return StreamResourceDataProvider(
            uri=f"{info.directory_uri}{filename}",
            resources=[
                StreamResourceInfo(
                    data_key=datakey_name,
                    shape=(height, width),
                    chunk_shape=(1, height, width),
                    dtype_numpy=dtype_numpy,
                    parameters={"dataset": datakey_name},
                )
            ],
            mimetype="application/x-zarr",
            collections_written_signal=self.output.captured,
        )

    async def stop(self) -> None:
        """Close the output's store."""
        await self.output.file_path.set("")


class MMCamera(StandardDetector):
    """A Micro-Manager camera that writes its frames to zarr stores.

    Pass either `label` with `ctrl`, or a `connector`, as a declared child
    receives. The camera has one output per label in `physical`, as a
    `Multi Camera` has, or one output named after its own label when
    `physical` is empty. The path provider names each output's store at
    `prepare`. Each name in `properties` becomes a signal in the
    `properties` map, which `read_configuration` reports.

    Raises
    ------
    ValueError
        If both or neither of `ctrl` and `connector` are given, if options are
        given with a `connector`, or if there is no path provider.
    """

    exposure: A[SignalRW[float], CameraValue("exposure")]
    acquire: A[SignalRW[bool], CameraValue("acquire")]
    capture: A[SignalRW[bool], CameraValue("capture")]
    num_capture: A[SignalRW[int], CameraValue("num_capture")]
    state: A[SignalR[CameraState], CameraValue("state")]
    last_error: A[SignalR[str], CameraValue("last_error")]

    def __init__(
        self,
        label: str = "",
        ctrl: MMController | None = None,
        *,
        path_provider: PathProvider | None = None,
        physical: Sequence[str] = (),
        properties: Sequence[str] = (),
        name: str = "",
        connector: MMConnector | None = None,
    ) -> None:
        if connector is not None:
            if (
                label
                or ctrl is not None
                or path_provider is not None
                or physical
                or properties
            ):
                raise ValueError(
                    "MMCamera takes either a connector or a label, ctrl and "
                    "options, not both"
                )
        elif ctrl is None:
            raise ValueError("MMCamera needs either a ctrl or a connector")
        else:
            connector = MMConnector(
                ctrl,
                label,
                physical=physical,
                properties=properties,
                path_provider=path_provider,
            )
        path_provider = connector.path_provider
        if path_provider is None:
            raise ValueError(f"Camera {connector.label!r} needs a path provider")
        super().__init__(name=name, connector=connector)
        outputs: dict[str, MMCameraOutput] = {}
        for physical_label in connector.physical or (connector.label,):
            output_connector = connector.child()
            output_connector.set_address(output=physical_label)
            outputs[clean_name(physical_label)] = MMCameraOutput(
                connector=output_connector
            )
        self.outputs = DeviceMap(outputs)
        self.properties = property_map(connector)
        self.add_config_signals(*self.properties.values())
        self.add_detector_logics(
            MMTriggerLogic(self.exposure, self.num_capture, list(outputs.values())),
            MMAcquireLogic(
                self.acquire,
                self.capture,
                self.num_capture,
                self.state,
                self.last_error,
            ),
            *(
                MMDataLogic(
                    output, path_provider, f"-{key}" if connector.physical else ""
                )
                for key, output in outputs.items()
            ),
        )

    @contextlib.asynccontextmanager
    async def raise_on_fault(self) -> AsyncGenerator[None, None]:
        """Stop the block as soon as the camera faults.

        Raises
        ------
        RuntimeError
            With the camera's last error, if the camera faults.
        """
        scope = asyncio.timeout(None)

        async def expire_on_fault() -> None:
            await wait_for_value(self.state, CameraState.FAULTED, timeout=None)
            scope.reschedule(asyncio.get_running_loop().time())

        try:
            async with scope:
                watcher = asyncio.ensure_future(expire_on_fault())
                try:
                    yield
                finally:
                    watcher.cancel()
        except TimeoutError:
            if not scope.expired():
                raise
            raise RuntimeError(await self.last_error.get_value()) from None

    # The base methods are unwrapped so that their own watcher updates pass
    # through unchanged while the fault scope runs in the same task.
    @WatchableAsyncStatus.wrap
    async def trigger(self) -> AsyncIterator[WatcherUpdate[int]]:
        """Take one collection, failing with the camera's last error if it faults."""
        async with self.raise_on_fault():
            async for update in inspect.unwrap(StandardDetector.trigger)(self):
                yield update

    @WatchableAsyncStatus.wrap
    async def complete(self) -> AsyncIterator[WatcherUpdate[int]]:
        """Wait for the kicked-off collections, failing with the camera's last error if it faults."""
        async with self.raise_on_fault():
            async for update in inspect.unwrap(StandardDetector.complete)(self):
                yield update
