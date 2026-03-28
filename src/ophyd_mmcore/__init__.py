"""ophyd-async devices backed by Micro-Manager via pymmcore-plus."""

from __future__ import annotations

from importlib.metadata import version

from ._camera import MMArmLogic, MMCamera, MMTriggerLogic, MMZarrDataLogic
from ._connector import PropName
from ._device import MMDevice
from ._devices import MMAutoFocus, MMGalvo, MMPump, MMShutter, MMStateDevice
from ._signal import (
    mmcore_signal_auto,
    mmcore_signal_r,
    mmcore_signal_rw,
    mmcore_signal_w,
)
from ._stage import MMXYStage, MMZStage
from ._worker import MMCoreWorker

try:
    __version__ = version("ophyd-mmcore")
except Exception:
    __version__ = "unknown"

__all__ = [
    "MMCoreWorker",
    "PropName",
    "MMDevice",
    "MMCamera",
    "MMTriggerLogic",
    "MMArmLogic",
    "MMZarrDataLogic",
    "MMShutter",
    "MMStateDevice",
    "MMAutoFocus",
    "MMGalvo",
    "MMPump",
    "MMZStage",
    "MMXYStage",
    "mmcore_signal_rw",
    "mmcore_signal_r",
    "mmcore_signal_w",
    "mmcore_signal_auto",
]
