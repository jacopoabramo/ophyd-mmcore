import logging
import threading
import time
from collections.abc import Callable

import numpy as np
from pymmcore_plus import CMMCorePlus

logger = logging.getLogger(__name__)


class FrameRouter:
    """Thread that empties the core's circular buffer.

    Each frame goes to `deliver` with the label of the physical camera that
    sent it, read from the frame's `Camera` metadata tag. When the buffer is
    empty the thread sleeps for 1 ms. The first time the buffer has
    overflowed since the thread started, `overflow` is called. An error while
    popping or delivering a frame is logged and the thread goes on.
    """

    def __init__(
        self,
        core: CMMCorePlus,
        deliver: Callable[[str, np.ndarray], None],
        overflow: Callable[[], None],
    ) -> None:
        self._core = core
        self._deliver = deliver
        self._overflow = overflow
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
        while not self._stop.is_set():
            try:
                if not overflowed and self._core.isBufferOverflowed():
                    overflowed = True
                    self._overflow()
                if self._core.getRemainingImageCount() == 0:
                    time.sleep(0.001)
                    continue
                frame, metadata = self._core.popNextImageAndMD()
                self._deliver(metadata["Camera"], frame)
            except Exception:
                logger.exception("Frame routing failed")
                time.sleep(0.1)
