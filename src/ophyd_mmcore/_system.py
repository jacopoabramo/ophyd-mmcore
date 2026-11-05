import asyncio
from typing import TYPE_CHECKING

from ophyd_async.core import (
    AsyncStageable,
    AsyncStatus,
    Device,
    PathProvider,
    StandardReadable,
)
from ophyd_async.core._device import DEVICE_RESERVED_ATTRS

from ._config import build_devices, build_groups, load_config, open_source
from ._connector import MMConnector

if TYPE_CHECKING:
    import os

    from pymmcore_plus import CMMCorePlus
    from pymmcore_plus.model import Microscope


def connector_of(child: Device) -> MMConnector:
    """Return the connector of a device that `MMSystem` builds or declares from a label."""
    connector = getattr(child, "_connector", None)
    assert isinstance(connector, MMConnector)
    return connector


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
        """Load the configuration file into the core, add the children not declared, then connect.

        Every device and configuration group of the core that no declared child
        addresses becomes a child, under its cleaned label (with `_group` for a
        group).

        Raises
        ------
        ValueError
            A key of a new child is also the key of another new child, the
            attribute of a declared child for another label, or an attribute
            of the root.
        """
        if self.core is not None and isinstance(device, StandardReadable):
            if self.model is not None:
                await asyncio.to_thread(load_config, self.core, self.model)
            await self.discover(self.core, device)
        await super().connect_real(device, timeout, force_reconnect)

    async def discover(self, core: "CMMCorePlus", device: StandardReadable) -> None:
        """Attach a child to `device` for each device and group of `core` it does not declare."""
        declared: dict[str, MMConnector] = {}
        for attr, child in device.children():
            connector = getattr(child, "_connector", None)
            if isinstance(connector, MMConnector) and connector.label:
                declared[attr] = connector
        devices, groups = await asyncio.to_thread(
            lambda: (
                build_devices(
                    core,
                    path_provider=self.path_provider,
                    move_timeout=self.move_timeout,
                    skip={c.label for c in declared.values() if not c.group},
                ),
                build_groups(
                    core,
                    move_timeout=self.move_timeout,
                    skip={c.label for c in declared.values() if c.group},
                ),
            )
        )
        for key in devices.keys() & groups.keys():
            raise ValueError(
                f"Device {connector_of(devices[key]).label!r} and configuration "
                f"group {connector_of(groups[key]).label!r} both become {key!r}"
            )
        found = devices | groups
        for key, child in found.items():
            label = connector_of(child).label
            if key in declared:
                raise ValueError(
                    f"{label!r} would be added as {key!r}, which is declared "
                    f"for {declared[key].label!r}"
                )
            if key in DEVICE_RESERVED_ATTRS or hasattr(device, key):
                raise ValueError(
                    f"{label!r} would be added as {key!r}, which the root "
                    "already has as an attribute"
                )
        for key, child in found.items():
            setattr(device, key, child)
        if device.name:
            device.set_name(device.name)


class MMSystem(StandardReadable):
    """A microscope whose children are the devices and configuration groups of one core.

    A subclass declares the children it wants typed with [`Label`][ophyd_mmcore.Label].

    `source` is a core, or the path of a Micro-Manager configuration file. A
    relative path that does not exist is looked up in the core's device
    adapter search paths. A core is created for a path unless `core` is
    given. `connect` loads the file into the core, device by device, and
    reuses a device the core already has loaded with the same library and
    adapter. A connect then adds a child for every device and configuration
    group of the core that no declared child addresses, keyed by the cleaned
    label (the cleaned group name followed by `_group` for a group). A connect
    in mock mode loads and adds nothing.

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

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage the root, then every child that can be staged, together.

        Bluesky stages the root of a device, so a plan that stages a child
        reaches it through this method.
        """
        await super().stage()
        await asyncio.gather(
            *(
                child.stage()
                for _, child in self.children()
                if isinstance(child, AsyncStageable)
            )
        )

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Unstage every child that can be staged, together, then the root."""
        await asyncio.gather(
            *(
                child.unstage()
                for _, child in self.children()
                if isinstance(child, AsyncStageable)
            )
        )
        await super().unstage()
