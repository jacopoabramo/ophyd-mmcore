"""ophyd-async devices for Micro-Manager hardware driven through pymmcore-plus."""

from importlib.metadata import version

from ._controller import MMController

__version__ = version("ophyd-mmcore")

__all__ = ["MMController", "__version__"]
