"""ophyd-async devices for Micro-Manager hardware driven through pymmcore-plus."""

from importlib.metadata import version

from ._connector import Label, Property
from ._controller import MMController
from ._devices import (
    MMConfigGroup,
    MMDevice,
    MMMoveLogic,
    MMShutter,
    MMStateDevice,
    MMSystem,
)
from ._stage import MMAxis, MMXYStage, MMZStage

__version__ = version("ophyd-mmcore")

__all__ = [
    "Label",
    "MMAxis",
    "MMConfigGroup",
    "MMController",
    "MMDevice",
    "MMMoveLogic",
    "MMShutter",
    "MMStateDevice",
    "MMSystem",
    "MMXYStage",
    "MMZStage",
    "Property",
    "__version__",
]
