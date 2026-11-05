import asyncio
from typing import TYPE_CHECKING

from ophyd_async.core import Device, PathProvider, StandardReadable

from ._config import load_config, open_source
from ._connector import MMConnector

if TYPE_CHECKING:
    import os

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.model import Microscope


class SystemConnector(MMConnector):
    """The connector of an [`MMSystem`][ophyd_mmcore.MMSystem].

    Holds the parsed configuration file, if the system was built from one, and
    loads it into the core before it connects the children.
    """

    def __init__(
        self,
        core: "CMMCorePlus",
        model: "Microscope | None",
        *,
        path_provider: PathProvider | None,
        move_timeout: float | None,
    ) -> None:
        super().__init__(core, path_provider=path_provider, move_timeout=move_timeout)
        self.model = model

    async def connect_real(
        self, device: Device, timeout: float, force_reconnect: bool
    ) -> None:
        """Load the configuration file into the core, then connect as an `MMConnector` does."""
        if self.model is not None and self.core is not None:
            await asyncio.to_thread(load_config, self.core, self.model)
        await super().connect_real(device, timeout, force_reconnect)


class MMSystem(StandardReadable):
    """A microscope declared as a class, whose children are marked with [`Label`][ophyd_mmcore.Label].

    `source` is a core, or the path of a Micro-Manager configuration file. A
    relative path that does not exist is looked up in the core's device
    adapter search paths. A core is created for a path unless `core` is
    given. `connect` loads the file into the core, device by device, and
    reuses a device the core already has loaded with the same library and
    adapter. A connect in mock mode loads nothing.

    Raises
    ------
    TypeError
        `source` is neither a path nor a core, or `core` is given with a core
        as `source`.
    """

    def __init__(
        self,
        source: "str | os.PathLike[str] | CMMCorePlus",
        *,
        core: "CMMCorePlus | None" = None,
        path_provider: PathProvider | None = None,
        move_timeout: float | None = 10.0,
        name: str = "",
    ) -> None:
        core, model = open_source(source, core)
        super().__init__(
            name=name,
            connector=SystemConnector(
                core, model, path_provider=path_provider, move_timeout=move_timeout
            ),
        )
