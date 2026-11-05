"""ophyd-async devices for Micro-Manager hardware driven through pymmcore-plus."""

from importlib.metadata import version

from ._backend import CameraState
from ._camera import MMCamera, MMCameraOutput
from ._config import make_config_groups, make_devices
from ._connector import Label, Property
from ._devices import (
    MMConfigGroup,
    MMDevice,
    MMShutter,
    MMStateDevice,
    MMSystem,
)
from ._stage import MMAxis, MMXYStage, MMZStage

__version__ = version("ophyd-mmcore")

__all__ = [
    "CameraState",
    "Label",
    "MMAxis",
    "MMCamera",
    "MMCameraOutput",
    "MMConfigGroup",
    "MMDevice",
    "MMShutter",
    "MMStateDevice",
    "MMSystem",
    "MMXYStage",
    "MMZStage",
    "Property",
    "__version__",
    "make_config_groups",
    "make_devices",
]
