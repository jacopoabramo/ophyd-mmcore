"""Base class for all Micro-Manager backed ophyd-async devices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ophyd_async.core import StandardReadable

from ._connector import MMDeviceConnector

if TYPE_CHECKING:
    from ._worker import MMCoreWorker


class MMDevice(StandardReadable):
    """Base class for ophyd-async devices backed by a Micro-Manager device.

    Signals can be declared in two ways:

    **Declarative (recommended)** — annotate class attributes with
    :class:`~ophyd_mmcore._connector.PropName` and
    :class:`~ophyd_async.core.StandardReadableFormat`::

        from typing import Annotated as A
        from ophyd_async.core import SignalR, SignalRW
        from ophyd_async.core import StandardReadableFormat as Format
        from ophyd_mmcore import MMDevice
        from ophyd_mmcore._connector import PropName


        class MMCamera(MMDevice):
            exposure: A[SignalRW[float], PropName("Exposure"), Format.HINTED_SIGNAL]
            binning: A[SignalRW[str], PropName("Binning"), Format.CONFIG_SIGNAL]
            read_mode: A[SignalR[str], PropName("ReadMode"), Format.CONFIG_SIGNAL]


        cam = MMCamera("Camera", worker, name="cam")

    **Imperative** — construct signals manually in ``__init__`` using
    :func:`~ophyd_mmcore._signal.mmcore_signal_rw` and
    :meth:`~ophyd_async.core.StandardReadable.add_children_as_readables`::

        class MMCamera(MMDevice):
            def __init__(self, mm_label, worker, name=""):
                with self.add_children_as_readables(Format.HINTED_SIGNAL):
                    self.exposure = mmcore_signal_rw(
                        float, mm_label, "Exposure", worker
                    )
                super().__init__(mm_label, worker, name=name)

    Parameters
    ----------
    mm_label:
        The Micro-Manager device label (e.g. ``"Camera"``).
    worker:
        The shared :class:`~ophyd_mmcore._worker.MMCoreWorker` instance.
    name:
        The ophyd-async device name.
    """

    def __init__(
        self,
        mm_label: str,
        worker: MMCoreWorker,
        name: str = "",
    ) -> None:
        self._mm_label = mm_label
        self._worker = worker
        super().__init__(name=name, connector=MMDeviceConnector(mm_label, worker))

    @property
    def mm_label(self) -> str:
        """The Micro-Manager device label."""
        return self._mm_label

    @property
    def worker(self) -> MMCoreWorker:
        """The shared MMCoreWorker for this device."""
        return self._worker
