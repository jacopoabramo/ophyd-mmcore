from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import acquire_zarr as zarr
import numpy as np
from event_model import ComposeStreamResource, DataKey, StreamRange
from ophyd_async.core import (
    DetectorArmLogic,
    DetectorDataLogic,
    DetectorTriggerLogic,
    StandardDetector,
    StreamableDataProvider,
    soft_signal_rw,
)

from ._base import get_worker
from ._connector import MMDeviceConnector

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from bluesky.protocols import StreamAsset
    from pymmcore_plus import CMMCorePlus

    from ._worker import MMCoreWorker


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------

_NP_TO_ZARR: dict[np.dtype, zarr.DataType] = {
    np.dtype("uint8"): zarr.DataType.UINT8,
    np.dtype("uint16"): zarr.DataType.UINT16,
    np.dtype("uint32"): zarr.DataType.UINT32,
    np.dtype("uint64"): zarr.DataType.UINT64,
    np.dtype("int8"): zarr.DataType.INT8,
    np.dtype("int16"): zarr.DataType.INT16,
    np.dtype("int32"): zarr.DataType.INT32,
    np.dtype("int64"): zarr.DataType.INT64,
    np.dtype("float32"): zarr.DataType.FLOAT32,
    np.dtype("float64"): zarr.DataType.FLOAT64,
}


def _zarr_dtype(dtype: np.dtype) -> zarr.DataType:
    try:
        return _NP_TO_ZARR[dtype]
    except KeyError:
        raise ValueError(
            f"No acquire-zarr DataType for numpy dtype {dtype!r}"
        ) from None


# ---------------------------------------------------------------------------
# ZarrStore
# ---------------------------------------------------------------------------


class ZarrStore:
    """Shared acquire-zarr store that multiple data logics write into.

    Each :class:`MMZarrDataLogic` registers its array dimensions here during
    ``prepare_unbounded``.  The underlying ``ZarrStream`` is opened lazily on
    the first call to :meth:`open` — subsequent calls from other logics are
    no-ops.  This mirrors the pattern used in *redsun*'s ``ZarrWriter``.

    Closing the store (via :meth:`close`) resets it so it can be reused for
    the next acquisition.

    Parameters
    ----------
    store_path:
        Filesystem path for the ``.zarr`` store directory.
    overwrite:
        Whether to overwrite an existing store at the same path.
    """

    def __init__(self, store_path: Path, *, overwrite: bool = True) -> None:
        self._store_path = store_path
        self._overwrite = overwrite
        self._array_settings: dict[str, zarr.ArraySettings] = {}
        self._stream: zarr.ZarrStream | None = None

    @property
    def store_uri(self) -> str:
        return f"file://{self._store_path.resolve()}"

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def register_array(self, key: str, settings: zarr.ArraySettings) -> None:
        """Register an array to be created when the stream opens.

        Must be called before :meth:`open`.

        Parameters
        ----------
        key:
            Output key / array name within the zarr store.
        settings:
            acquire-zarr ``ArraySettings`` describing dtype and dimensions.
        """
        if self.is_open:
            raise RuntimeError(
                "Cannot register arrays after the store has been opened."
            )
        self._array_settings[key] = settings

    def open(self) -> None:
        """Open the ZarrStream with all registered arrays.

        Idempotent — subsequent calls are no-ops.
        """
        if self.is_open:
            return
        self._stream = zarr.ZarrStream(
            zarr.StreamSettings(
                store_path=str(self._store_path),
                arrays=list(self._array_settings.values()),
                overwrite=self._overwrite,
            )
        )

    def append(self, frame: np.ndarray, key: str) -> None:
        """Append *frame* to the array identified by *key*.

        Parameters
        ----------
        frame:
            NumPy array representing a single frame.
        key:
            Output key of the target array (must match a registered key).
        """
        if self._stream is None:
            raise RuntimeError("Store is not open; call open() first.")
        self._stream.append(frame, key=key)

    def close(self) -> None:
        """Close the ZarrStream and reset registered arrays."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._array_settings.clear()


# ---------------------------------------------------------------------------
# MMTriggerLogic
# ---------------------------------------------------------------------------


class MMTriggerLogic(DetectorTriggerLogic):
    """DetectorTriggerLogic for a pymmcore-plus camera device.

    Implements ``prepare_internal()`` for internally triggered sequence
    acquisition.  Stores the requested frame count so that
    :class:`MMArmLogic` can start the sequence.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label.
    worker:
        The shared ``MMCoreWorker``.
    """

    def __init__(self, mm_label: str, worker: MMCoreWorker) -> None:
        self._mm_label = mm_label
        self._worker = worker
        self._n_frames: int = 0

    def get_deadtime(self, config_values: object) -> float:
        """Return a conservative 1 ms inter-frame deadtime estimate."""
        return 0.001

    async def prepare_internal(
        self, num: int, livetime: float, deadtime: float
    ) -> None:
        """Store the frame count and optionally set the exposure time."""
        self._n_frames = num
        if livetime != 0.0:
            exposure_ms = livetime * 1_000.0
            await self._worker.run(
                lambda: self._worker.core.setExposure(self._mm_label, exposure_ms)
            )


# ---------------------------------------------------------------------------
# MMArmLogic
# ---------------------------------------------------------------------------


class MMArmLogic(DetectorArmLogic):
    """DetectorArmLogic for a pymmcore-plus camera device.

    Starts and stops Micro-Manager sequence acquisition.

    Parameters
    ----------
    mm_label:
        Micro-Manager device label.
    worker:
        The shared ``MMCoreWorker``.
    trigger_logic:
        The associated :class:`MMTriggerLogic`; the arm logic reads
        ``_n_frames`` from it.
    store:
        The shared :class:`ZarrStore`; opened here on the first arm so
        all data logics have registered their arrays before the stream starts.
    """

    def __init__(
        self,
        mm_label: str,
        worker: MMCoreWorker,
        trigger_logic: MMTriggerLogic,
        store: ZarrStore,
    ) -> None:
        self._mm_label = mm_label
        self._worker = worker
        self._trigger_logic = trigger_logic
        self._store = store

    async def arm(self) -> None:
        """Open the zarr store (if not already open) then start MM sequence acquisition."""
        self._store.open()
        n = self._trigger_logic._n_frames
        await self._worker.run(
            lambda: self._worker.core.startSequenceAcquisition(
                self._mm_label, n, 0, False
            )
        )

    async def wait_for_idle(self) -> None:
        """Poll until the sequence acquisition finishes."""
        while await self._worker.run(
            lambda: self._worker.core.isSequenceRunning(self._mm_label)
        ):
            await asyncio.sleep(0.02)

    async def disarm(self) -> None:
        """Stop sequence acquisition and close the zarr store."""
        if await self._worker.run(
            lambda: self._worker.core.isSequenceRunning(self._mm_label)
        ):
            await self._worker.run(
                lambda: self._worker.core.stopSequenceAcquisition(self._mm_label)
            )
        self._store.close()


# ---------------------------------------------------------------------------
# MMZarrStreamProvider
# ---------------------------------------------------------------------------


class MMZarrStreamProvider(StreamableDataProvider):
    """StreamableDataProvider for an array within a shared :class:`ZarrStore`.

    Holds a soft signal that the drain loop updates with the running frame
    count.  The detector framework reads this signal to track progress and
    know when to emit stream documents.

    Parameters
    ----------
    store:
        The shared zarr store.
    array_key:
        Key of the array within the zarr store.
    datakey_name:
        Bluesky data key (e.g. the detector name).
    dtype:
        NumPy dtype of the frames.
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    """

    def __init__(
        self,
        store: ZarrStore,
        array_key: str,
        datakey_name: str,
        dtype: np.dtype,
        width: int,
        height: int,
    ) -> None:
        self._frames_written = soft_signal_rw(int)
        self.collections_written_signal = self._frames_written
        self._datakey_name = datakey_name
        self._dtype = dtype
        self._width = width
        self._height = height
        self._last_emitted = 0

        self._bundle = ComposeStreamResource()(
            mimetype="application/x-zarr",
            uri=store.store_uri,
            data_key=datakey_name,
            parameters={"path": array_key},
            uid=None,
            validate=True,
        )

    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        """Return the Bluesky DataKey for this zarr array."""
        return {
            self._datakey_name: DataKey(
                source=self._bundle.stream_resource_doc["uri"],
                shape=[collections_per_event, self._height, self._width],
                dtype="array",
                dtype_numpy=self._dtype.str,
                external="STREAM:",
            )
        }

    async def make_stream_docs(
        self, collections_written: int, collections_per_event: int
    ) -> AsyncGenerator[StreamAsset, None]:
        """Yield pending stream_resource / stream_datum documents."""
        indices_written = collections_written // collections_per_event
        if indices_written and not self._last_emitted:
            yield "stream_resource", self._bundle.stream_resource_doc
        if indices_written > self._last_emitted:
            indices: StreamRange = {
                "start": self._last_emitted,
                "stop": indices_written,
            }
            self._last_emitted = indices_written
            yield "stream_datum", self._bundle.compose_stream_datum(indices)


# ---------------------------------------------------------------------------
# MMZarrDataLogic
# ---------------------------------------------------------------------------


class MMZarrDataLogic(DetectorDataLogic):
    """DetectorDataLogic that streams MM frames into a shared :class:`ZarrStore`.

    Multiple ``MMZarrDataLogic`` instances can share one ``ZarrStore``,
    each writing to a different array key within the same zarr store on disk.
    Frames are drained from the MM circular buffer by a background asyncio
    task and handed to the store via :meth:`ZarrStore.append`.

    The drain task runs from ``prepare_unbounded`` until cancelled by
    :meth:`stop`.  The store itself is opened by :class:`MMArmLogic` once all
    data logics have registered their arrays.

    Parameters
    ----------
    store:
        The shared :class:`ZarrStore` to write into.
    array_key:
        Output key for this logic's array within the store.
    mm_label:
        Micro-Manager device label for the camera.
    worker:
        The shared ``MMCoreWorker``.
    chunk_t:
        Number of frames per time-axis chunk.  Defaults to 1 (one frame per
        chunk), which minimises write latency.  Larger values improve read
        throughput at the cost of buffering.
    """

    def __init__(
        self,
        store: ZarrStore,
        array_key: str,
        mm_label: str,
        worker: MMCoreWorker,
        chunk_t: int = 1,
    ) -> None:
        self._store = store
        self._array_key = array_key
        self._mm_label = mm_label
        self._worker = worker
        self._chunk_t = chunk_t

        self._provider: MMZarrStreamProvider | None = None
        self._drain_task: asyncio.Task[None] | None = None

    async def prepare_unbounded(self, datakey_name: str) -> MMZarrStreamProvider:
        """Register this logic's array with the store and start the drain loop."""
        await self.stop()

        width, height, dtype = await self._worker.run(self._get_frame_shape)

        self._store.register_array(
            self._array_key,
            zarr.ArraySettings(
                output_key=self._array_key,
                data_type=_zarr_dtype(dtype),
                dimensions=[
                    zarr.Dimension(
                        name="t",
                        kind=zarr.DimensionType.TIME,
                        array_size_px=0,  # unlimited / streaming
                        chunk_size_px=self._chunk_t,
                        shard_size_chunks=1,
                    ),
                    zarr.Dimension(
                        name="y",
                        kind=zarr.DimensionType.SPACE,
                        array_size_px=height,
                        chunk_size_px=height,
                        shard_size_chunks=1,
                    ),
                    zarr.Dimension(
                        name="x",
                        kind=zarr.DimensionType.SPACE,
                        array_size_px=width,
                        chunk_size_px=width,
                        shard_size_chunks=1,
                    ),
                ],
            ),
        )

        self._provider = MMZarrStreamProvider(
            store=self._store,
            array_key=self._array_key,
            datakey_name=datakey_name,
            dtype=dtype,
            width=width,
            height=height,
        )
        self._drain_task = asyncio.create_task(self._drain_loop())
        return self._provider

    async def stop(self) -> None:
        """Cancel the drain task."""
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        self._drain_task = None

    def _get_frame_shape(self) -> tuple[int, int, np.dtype]:
        """Read frame dimensions from MM; runs on the worker thread."""
        w = self._worker.core.getImageWidth()
        h = self._worker.core.getImageHeight()
        bpp = self._worker.core.getBytesPerPixel()
        dtype = np.dtype("uint8") if bpp == 1 else np.dtype("uint16")
        return w, h, dtype

    def _drain_batch(self) -> int:
        """Pop all available frames from the MM buffer into the store.

        Returns the number of frames popped.  Runs on the worker thread.
        """
        n = self._worker.core.getRemainingImageCount()
        for _ in range(n):
            self._store.append(self._worker.core.popNextImage(), key=self._array_key)
        return n

    async def _drain_loop(self) -> None:
        """Drain the MM circular buffer into the store continuously.

        Polls every 10 ms until cancelled by :meth:`stop`.  The loop starts
        before the sequence acquisition is armed, so it must not exit on its
        own — it runs until cancelled.
        """
        assert self._provider is not None, (
            "_drain_loop called before prepare_unbounded()"
        )
        frames_written = 0

        while True:
            n = await self._worker.run(self._drain_batch)
            if n > 0:
                frames_written += n
                await self._provider._frames_written.set(frames_written)
            await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# MMCamera
# ---------------------------------------------------------------------------


class MMCamera(StandardDetector):
    """Micro-Manager camera as an ophyd-async ``StandardDetector``.

    Combines :class:`MMTriggerLogic`, :class:`MMArmLogic`, and
    :class:`MMZarrDataLogic` into a single device that participates in both
    step scans (via ``trigger()``) and fly scans (via ``prepare()`` /
    ``kickoff()`` / ``complete()``).

    An :class:`ZarrStore` is created internally and shared between the arm
    logic (which opens the stream) and the data logic (which registers its
    array and drains frames into it).  To share a store across multiple
    cameras, create an ``ZarrStore`` explicitly and pass it via
    ``store``.

    Property signals (exposure, binning, etc.) are declared on subclasses
    using ``PropName`` annotations, exactly as with :class:`~ophyd_mmcore.MMDevice`.
    After calling ``super().__init__()``, add signals to
    ``read_configuration()`` via :meth:`add_config_signals`.

    Example
    -------
    ::

        from typing import Annotated as A
        from ophyd_async.core import SignalRW
        from ophyd_mmcore import MMCamera, MMCoreWorker
        from ophyd_mmcore._connector import PropName


        class DemoCamera(MMCamera):
            exposure: A[SignalRW[float], PropName("Exposure")]
            binning: A[SignalRW[str], PropName("Binning")]

            def __init__(self, mm_label, worker, store_path, name=""):
                super().__init__(mm_label, worker, store_path, name=name)
                self.add_config_signals(self.exposure, self.binning)

    Parameters
    ----------
    mm_label:
        Micro-Manager device label (e.g. ``"Camera"``).
    worker:
        The shared ``MMCoreWorker`` instance.
    store_path:
        Filesystem path for the output ``.zarr`` store.  Ignored when
        ``store`` is provided.
    store:
        Optional pre-created :class:`ZarrStore`.  When given,
        ``store_path`` is not used.  Useful for sharing one zarr store
        across multiple cameras.
    array_key:
        Key for this camera's array within the store.  Defaults to
        ``"frames"``.
    chunk_t:
        Frames per time-axis chunk (passed to :class:`MMZarrDataLogic`).
    name:
        ophyd-async device name.
    """

    def __init__(
        self,
        mm_label: str,
        core: CMMCorePlus,
        store_path: Path | None = None,
        *,
        store: ZarrStore | None = None,
        array_key: str = "frames",
        chunk_t: int = 1,
        name: str = "",
    ) -> None:
        worker = get_worker(core)
        if store is None:
            if store_path is None:
                raise ValueError("Either store_path or store must be provided.")
            store = ZarrStore(store_path)
        trigger_logic = MMTriggerLogic(mm_label, worker)
        arm_logic = MMArmLogic(mm_label, worker, trigger_logic, store)
        data_logic = MMZarrDataLogic(store, array_key, mm_label, worker, chunk_t)
        super().__init__(
            name=name,
            connector=MMDeviceConnector(mm_label, worker),
        )
        self.add_detector_logics(trigger_logic, arm_logic, data_logic)
