import logging
import threading
import time
from collections.abc import Callable

import numpy as np
from pymmcore_plus import CMMCorePlus

logger = logging.getLogger(__name__)

IDLE_INTERVAL = 0.1
"""Seconds between two calls of a router's `idle` while the buffer stays empty."""

MAX_FAILURES = 3
"""The number of failures in a row after which a router calls `failed`."""


class FrameRouter:
    """Thread that empties the core's circular buffer.

    Each frame goes to `deliver` with the label of the physical camera that
    sent it, read from the frame's `Camera` metadata tag. When the buffer is
    empty the thread sleeps for 1 ms, and calls `idle` at most once every
    100 ms. The first time the buffer has overflowed since the thread
    started, `overflow` is called. An error while popping or delivering a
    frame is logged and the thread goes on after 100 ms. The third error in
    a row is passed to `failed`, and later ones are no longer logged until a
    frame gets through.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        deliver: Callable[[str, np.ndarray], None],
        overflow: Callable[[], None],
        idle: Callable[[], None],
        failed: Callable[[Exception], None],
    ) -> None:
        self._core = core
        self._deliver = deliver
        self._overflow = overflow
        self._idle = idle
        self._failed = failed
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the thread, unless it is running."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ophyd-mmcore-frames", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for it to end, unless it is stopped."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        overflowed = False
        failures = 0
        next_idle = time.monotonic() + IDLE_INTERVAL
        while not self._stop.is_set():
            try:
                if not overflowed and self._core.isBufferOverflowed():
                    overflowed = True
                    self._overflow()
                if self._core.getRemainingImageCount() == 0:
                    if time.monotonic() >= next_idle:
                        next_idle = time.monotonic() + IDLE_INTERVAL
                        self._idle()
                    failures = 0
                    time.sleep(0.001)
                    continue
                frame, metadata = self._core.popNextImageAndMD()
                self._deliver(metadata["Camera"], frame)
                failures = 0
            except Exception as error:
                failures += 1
                if failures <= MAX_FAILURES:
                    logger.exception("Frame routing failed")
                if failures == MAX_FAILURES:
                    self._report(error)
                time.sleep(0.1)

    def _report(self, error: Exception) -> None:
        try:
            self._failed(error)
        except Exception:
            logger.exception("Reporting the frame routing failure failed")
