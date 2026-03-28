"""Base class for all Micro-Manager backed ophyd-async devices."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from ophyd_async.core import StandardReadable

from ._connector import MMDeviceConnector
from ._worker import MMCoreWorker

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

_worker_cache: weakref.WeakKeyDictionary[CMMCorePlus, MMCoreWorker] = (
    weakref.WeakKeyDictionary()
)


def get_worker(core: CMMCorePlus) -> MMCoreWorker:
    """Return the shared :class:`MMCoreWorker` for *core*, creating it if needed.

    All devices backed by the same ``CMMCorePlus`` instance share a single
    worker thread, ensuring that MM API calls are serialised correctly.

    This is called automatically by :class:`MMDevice.__init__`.  Call it
    explicitly only when you need the worker before constructing the device —
    e.g. in an imperative-style ``__init__`` that creates signals manually
    before calling ``super().__init__``.

    Parameters
    ----------
    core:
        The ``CMMCorePlus`` instance to wrap.
    """
    try:
        return _worker_cache[core]
    except KeyError:
        worker = MMCoreWorker(core)
        _worker_cache[core] = worker
        return worker


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


        cam = MMCamera("Camera", core, name="cam")

    **Imperative** — construct signals manually in ``__init__`` using
    :func:`~ophyd_mmcore._signal.mmcore_signal_rw` and
    :meth:`~ophyd_async.core.StandardReadable.add_children_as_readables`::

        class MMCamera(MMDevice):
            def __init__(self, mm_label, core, name=""):
                worker = get_worker(core)
                with self.add_children_as_readables(Format.HINTED_SIGNAL):
                    self.exposure = mmcore_signal_rw(
                        float, mm_label, "Exposure", worker
                    )
                super().__init__(mm_label, core, name=name)

    Parameters
    ----------
    mm_label:
        The Micro-Manager device label (e.g. ``"Camera"``).
    core:
        The shared :class:`~pymmcore_plus.CMMCorePlus` instance.
    name:
        The ophyd-async device name.
    """

    def __init__(
        self,
        mm_label: str,
        core: CMMCorePlus,
        name: str = "",
    ) -> None:
        self._mm_label = mm_label
        self._worker = get_worker(core)
        super().__init__(name=name, connector=MMDeviceConnector(mm_label, self._worker))

    @property
    def mm_label(self) -> str:
        """The Micro-Manager device label."""
        return self._mm_label
