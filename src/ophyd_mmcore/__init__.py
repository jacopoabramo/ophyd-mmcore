"""ophyd-async devices backed by Micro-Manager via pymmcore-plus."""

from __future__ import annotations

from importlib.metadata import version

from ._base import MMDevice, get_worker
from ._camera import (
    MMArmLogic,
    MMCamera,
    MMTriggerLogic,
    MMZarrDataLogic,
    MMZarrStreamProvider,
    ZarrStore,
)
from ._connector import (
    CoreMethod,
    ExposureMethod,
    PropName,
    ShutterOpenMethod,
    XPositionMethod,
    YPositionMethod,
    ZPositionMethod,
)
from ._devices import (
    MMAutoFocus,
    MMGalvo,
    MMPump,
    MMShutter,
    MMStateDevice,
    MMXYStage,
    MMZStage,
)
from ._signal import (
    mmcore_signal_auto,
    mmcore_signal_r,
    mmcore_signal_rw,
    mmcore_signal_w,
)
from ._worker import MMCoreWorker

try:
    __version__ = version("ophyd-mmcore")
except Exception:
    __version__ = "unknown"

__all__ = [
    "MMCoreWorker",
    "get_worker",
    "CoreMethod",
    "ExposureMethod",
    "ZPositionMethod",
    "XPositionMethod",
    "YPositionMethod",
    "ShutterOpenMethod",
    "PropName",
    "MMDevice",
    "MMCamera",
    "MMTriggerLogic",
    "MMArmLogic",
    "MMZarrDataLogic",
    "ZarrStore",
    "MMZarrStreamProvider",
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
