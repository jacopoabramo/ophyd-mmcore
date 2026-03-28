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

from ._connector import MMDeviceConnector

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from bluesky.protocols import StreamAsset

    from ._worker import MMCoreWorker


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
    """

    def __init__(
        self,
        mm_label: str,
        worker: MMCoreWorker,
        trigger_logic: MMTriggerLogic,
    ) -> None:
        self._mm_label = mm_label
        self._worker = worker
        self._trigger_logic = trigger_logic

    async def arm(self) -> None:
        """Start MM sequence acquisition."""
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
        """Stop sequence acquisition if still running."""
        if await self._worker.run(
            lambda: self._worker.core.isSequenceRunning(self._mm_label)
        ):
            await self._worker.run(
                lambda: self._worker.core.stopSequenceAcquisition(self._mm_label)
            )


class MMZarrStreamProvider(StreamableDataProvider):
    """StreamableDataProvider for a zarr store written by acquire-zarr.

    Holds a soft signal that the drain loop updates with the running frame
    count.  The detector framework reads this signal to track progress and
    know when to emit stream documents.

    Parameters
    ----------
    store_uri:
        ``file://`` URI to the zarr store root.
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
        store_uri: str,
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
            uri=store_uri,
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
    """DetectorDataLogic that streams MM frames into an OME-Zarr store.

    Frames are drained from the MM circular buffer by a background asyncio
    task and handed to an *acquire-zarr* ``ZarrStream``.  The drain task
    starts when :meth:`prepare_unbounded` is called and terminates after
    the sequence acquisition stops and all remaining frames are flushed.

    Parameters
    ----------
    store_path:
        Filesystem path for the output ``.zarr`` store directory.
    mm_label:
        Micro-Manager device label for the camera.
    worker:
        The shared ``MMCoreWorker``.
    """

    _ARRAY_KEY = "frames"

    def __init__(
        self,
        store_path: Path,
        mm_label: str,
        worker: MMCoreWorker,
    ) -> None:
        self._store_path = store_path
        self._mm_label = mm_label
        self._worker = worker

        self._stream: zarr.ZarrStream | None = None
        self._provider: MMZarrStreamProvider | None = None
        self._drain_task: asyncio.Task[None] | None = None

    async def prepare_unbounded(self, datakey_name: str) -> MMZarrStreamProvider:
        """Open the zarr store and return the stream provider."""
        await self.stop()

        width, height, dtype = await self._worker.run(self._get_frame_shape)

        self._stream = zarr.ZarrStream(
            zarr.StreamSettings(
                store_path=str(self._store_path),
                arrays=[
                    zarr.ArraySettings(
                        output_key=self._ARRAY_KEY,
                        data_type=_zarr_dtype(dtype),
                        dimensions=[
                            zarr.Dimension(
                                name="t",
                                kind=zarr.DimensionType.TIME,
                                array_size_px=0,  # unlimited / streaming
                                chunk_size_px=1,
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
                    )
                ],
                overwrite=True,
            )
        )

        self._provider = MMZarrStreamProvider(
            store_uri=f"file://{self._store_path.resolve()}",
            array_key=self._ARRAY_KEY,
            datakey_name=datakey_name,
            dtype=dtype,
            width=width,
            height=height,
        )
        self._drain_task = asyncio.create_task(self._drain_loop())
        return self._provider

    async def stop(self) -> None:
        """Cancel the drain task and close the zarr stream."""
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        self._drain_task = None

        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _get_frame_shape(self) -> tuple[int, int, np.dtype]:
        """Read frame dimensions from MM; runs on the worker thread."""
        core = self._worker.core
        w = core.getImageWidth()
        h = core.getImageHeight()
        bpp = core.getBytesPerPixel()
        dtype = np.dtype("uint8") if bpp == 1 else np.dtype("uint16")
        return w, h, dtype

    def _drain_batch(self) -> int:
        """Pop all available frames from the MM buffer into the zarr stream.

        Returns the number of frames popped.  Runs on the worker thread.
        """
        assert self._stream is not None, (
            "_drain_batch called before prepare_unbounded()"
        )
        n = self._worker.core.getRemainingImageCount()
        for _ in range(n):
            self._stream.append(self._worker.core.popNextImage())
        return n

    async def _drain_loop(self) -> None:
        """Drain the MM circular buffer into the zarr stream continuously.

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


class MMCamera(StandardDetector):
    """Micro-Manager camera as an ophyd-async ``StandardDetector``.

    Combines :class:`MMTriggerLogic`, :class:`MMArmLogic`, and
    :class:`MMZarrDataLogic` into a single device that participates in both
    step scans (via ``trigger()``) and fly scans (via ``prepare()`` /
    ``kickoff()`` / ``complete()``).

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
        Filesystem path for the output ``.zarr`` store written during
        acquisition.
    name:
        ophyd-async device name.
    """

    def __init__(
        self,
        mm_label: str,
        worker: MMCoreWorker,
        store_path: Path,
        name: str = "",
    ) -> None:
        trigger_logic = MMTriggerLogic(mm_label, worker)
        arm_logic = MMArmLogic(mm_label, worker, trigger_logic)
        data_logic = MMZarrDataLogic(store_path, mm_label, worker)
        super().__init__(
            name=name,
            connector=MMDeviceConnector(mm_label, worker),
        )
        self.add_detector_logics(trigger_logic, arm_logic, data_logic)
