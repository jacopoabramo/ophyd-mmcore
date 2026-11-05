import contextlib
import logging
import os
from pathlib import Path

from ophyd_async.core import Device, PathProvider
from pymmcore_plus import CMMCorePlus, DeviceType
from pymmcore_plus.model import Device as ModelDevice
from pymmcore_plus.model import Microscope

from ._camera import MMCamera
from ._connector import clean_name
from ._devices import MMConfigGroup, MMDevice, MMShutter, MMStateDevice
from ._stage import MMXYStage, MMZStage

logger = logging.getLogger(__name__)

MULTI_CAMERA_PROPERTIES = tuple(f"Physical Camera {n}" for n in range(1, 5))

CORE_ROLES = {
    "Camera": DeviceType.Camera,
    "Shutter": DeviceType.Shutter,
    "Focus": DeviceType.Stage,
    "XYStage": DeviceType.XYStage,
    "AutoFocus": DeviceType.AutoFocus,
    "SLM": DeviceType.SLM,
    "Galvo": DeviceType.Galvo,
}

INIT_ORDER = {DeviceType.Serial: 0, DeviceType.Hub: 1}


def open_source(
    source: object, core: CMMCorePlus | None
) -> tuple[CMMCorePlus, Microscope | None]:
    """Return the core to drive and the model of the file to load into it, if any.

    A core as `source` is used as it is, with no model. A path creates a core
    unless `core` is given, and parses the configuration file it resolves to.

    Raises
    ------
    TypeError
        `source` is neither a path nor a core, or `core` is given with a core
        as `source`.
    """
    match source:
        case CMMCorePlus():
            if core is not None:
                raise TypeError("core= cannot be combined with a core as the source")
            return source, None
        case str() | os.PathLike():
            if core is None:
                core = CMMCorePlus()
            return core, Microscope.create_from_config(resolve_config(source, core))
        case _:
            raise TypeError(
                f"Expected a configuration file path or a core, got {source!r}"
            )


def resolve_config(path: str | os.PathLike[str], core: CMMCorePlus) -> Path:
    """Return the configuration file `path` names.

    A path that exists is used as given. A relative path that does not is
    looked up in each of the core's device adapter search paths.

    Raises
    ------
    FileNotFoundError
        The file is in none of those places.
    """
    given = Path(path).expanduser()
    if given.exists():
        return given
    if not given.is_absolute():
        for directory in core.getDeviceAdapterSearchPaths():
            found = Path(directory) / given
            if found.exists():
                return found
    raise FileNotFoundError(f"Configuration file {str(path)!r} not found")


# TODO: replace steps 2 to 4 and 6 of this function with
# `Microscope.initialize(core, keep_loaded=True)` and
# `Microscope.apply_to_core(core, exclude={"devices"})` once pymmcore-plus can
# keep a matching loaded device, apply a parsed model and skip empty core values.
def load_config(core: CMMCorePlus, model: Microscope) -> None:
    """Load the devices of `model` into `core` the way `loadSystemConfiguration` does.

    A device the core has loaded with the same library and adapter is reused
    as it is. Core roles that are still empty go to the first new device of
    their type, and the configuration groups, the pixel size presets and the
    core properties of the model are applied.

    Raises
    ------
    ValueError
        A label is loaded with another library or adapter than the file
        names. Nothing is loaded.
    """
    loaded = set(core.getLoadedDevices())
    new: list[ModelDevice] = []
    for device in [*model.assigned_com_ports, *model.devices]:
        if device.name not in loaded:
            new.append(device)
            continue
        found = (core.getDeviceLibrary(device.name), core.getDeviceName(device.name))
        if found != (device.library, device.adapter_name):
            raise ValueError(
                f"{device.name!r} is loaded as {found}, but the file names it "
                f"{(device.library, device.adapter_name)}"
            )
        logger.info("%r is already loaded, reusing it", device.name)

    uninitialized = [device.name for device in new]
    try:
        for device in new:
            device.load(core)
            for prop in device.properties:
                if prop.is_pre_init:
                    prop.apply_to_core(core, then_update=False)
            if device.parent_label:
                core.setParentLabel(device.name, device.parent_label)
        for device in sorted(
            new, key=lambda d: INIT_ORDER.get(core.getDeviceType(d.name), 2)
        ):
            core.initializeDevice(device.name)
            uninitialized.remove(device.name)
            device.apply_to_core(core, then_update=False)
    except BaseException:
        for name in reversed(uninitialized):
            with contextlib.suppress(RuntimeError):
                core.unloadDevice(name)
        raise

    for role, device_type in CORE_ROLES.items():
        if core.getProperty("Core", role):
            continue
        for device in new:
            if core.getDeviceType(device.name) == device_type:
                core.setProperty("Core", role, device.name)
                break

    for group in model.config_groups.values():
        group.apply_to_core(core, then_update=False)
    model.pixel_size_group.apply_to_core(core, then_update=False)
    for prop in model.core_device.properties:
        if prop.value:
            core.setProperty("Core", prop.name, prop.value)

    if core.isConfigDefined("System", "Startup"):
        core.setConfig("System", "Startup")
    core.waitForSystem()


def physical_cameras(core: CMMCorePlus, label: str) -> list[str]:
    """Return the labels set in the physical camera properties of a Multi Camera."""
    values = (core.getProperty(label, prop) for prop in MULTI_CAMERA_PROPERTIES)
    return [value for value in values if value != "Undefined"]


def make_devices(
    core: CMMCorePlus,
    *,
    path_provider: PathProvider,
    move_timeout: float | None = 10.0,
) -> dict[str, Device]:
    """Build an ophyd-async device for each device the core has loaded.

    The class follows the Micro-Manager device type: `MMCamera`, `MMZStage`,
    `MMXYStage`, `MMShutter`, `MMStateDevice` or `MMDevice`. Other types, such
    as the core, hubs and serial ports, get none. A camera of the `Utilities`
    adapter named `Multi Camera` gets the labels in its physical camera
    properties as `physical`. Each key is the cleaned label and each device is
    named by its key. Nothing is connected.

    All cameras share `path_provider`, which ophyd-async calls with each data
    key. Use a filename provider that depends on that key: two outputs with the
    same filename name the same store, and the second to open it is refused,
    at the latest when its capture starts.

    Raises
    ------
    ValueError
        Two labels clean to the same key.
    """
    devices: dict[str, Device] = {}
    labels: dict[str, str] = {}
    for label in core.getLoadedDevices():
        key = clean_name(label)
        match core.getDeviceType(label):
            case DeviceType.Camera:
                physical = (
                    physical_cameras(core, label)
                    if core.getDeviceLibrary(label) == "Utilities"
                    and core.getDeviceName(label) == "Multi Camera"
                    else []
                )
                device = MMCamera(
                    label,
                    core,
                    path_provider=path_provider,
                    physical=physical,
                    name=key,
                )
            case DeviceType.Stage:
                device = MMZStage(label, core, move_timeout=move_timeout, name=key)
            case DeviceType.XYStage:
                device = MMXYStage(label, core, move_timeout=move_timeout, name=key)
            case DeviceType.Shutter:
                device = MMShutter(label, core, move_timeout=move_timeout, name=key)
            case DeviceType.State:
                device = MMStateDevice(label, core, move_timeout=move_timeout, name=key)
            case DeviceType.Generic:
                device = MMDevice(label, core, name=key)
            case _:
                continue
        if key in devices:
            raise ValueError(
                f"Device labels {labels[key]!r} and {label!r} both clean to {key!r}"
            )
        devices[key] = device
        labels[key] = label
    return devices


def make_config_groups(
    core: CMMCorePlus, *, move_timeout: float | None = 10.0
) -> dict[str, MMConfigGroup]:
    """Build an `MMConfigGroup` for each configuration group of the core.

    Each key is the cleaned group name and each group is named `<key>_group`,
    so its data keys differ from those of a device with the same name. Nothing
    is connected.

    Raises
    ------
    ValueError
        Two group names clean to the same key.
    """
    groups: dict[str, MMConfigGroup] = {}
    names: dict[str, str] = {}
    for group in core.getAvailableConfigGroups():
        key = clean_name(group)
        if key in groups:
            raise ValueError(
                f"Group names {names[key]!r} and {group!r} both clean to {key!r}"
            )
        groups[key] = MMConfigGroup(
            group, core, move_timeout=move_timeout, name=f"{key}_group"
        )
        names[key] = group
    return groups
