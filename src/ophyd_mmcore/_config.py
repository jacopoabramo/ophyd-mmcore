from ophyd_async.core import Device, PathProvider
from pymmcore_plus import DeviceType

from ._camera import MMCamera
from ._connector import clean_name
from ._controller import MMController
from ._devices import MMConfigGroup, MMDevice, MMShutter, MMStateDevice
from ._stage import MMXYStage, MMZStage

MULTI_CAMERA_PROPERTIES = tuple(f"Physical Camera {n}" for n in range(1, 5))


def physical_cameras(ctrl: MMController, label: str) -> list[str]:
    """Return the labels set in the physical camera properties of a Multi Camera."""
    core = ctrl.core
    values = (core.getProperty(label, prop) for prop in MULTI_CAMERA_PROPERTIES)
    return [value for value in values if value != "Undefined"]


def make_devices(
    ctrl: MMController,
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
    same filename share a store, and the second fails when it is prepared.

    Raises
    ------
    ValueError
        Two labels clean to the same key.
    """
    core = ctrl.core
    devices: dict[str, Device] = {}
    labels: dict[str, str] = {}
    for label in core.getLoadedDevices():
        key = clean_name(label)
        match core.getDeviceType(label):
            case DeviceType.Camera:
                physical = (
                    physical_cameras(ctrl, label)
                    if core.getDeviceLibrary(label) == "Utilities"
                    and core.getDeviceName(label) == "Multi Camera"
                    else []
                )
                device = MMCamera(
                    label,
                    ctrl,
                    path_provider=path_provider,
                    physical=physical,
                    name=key,
                )
            case DeviceType.Stage:
                device = MMZStage(label, ctrl, move_timeout=move_timeout, name=key)
            case DeviceType.XYStage:
                device = MMXYStage(label, ctrl, move_timeout=move_timeout, name=key)
            case DeviceType.Shutter:
                device = MMShutter(label, ctrl, move_timeout=move_timeout, name=key)
            case DeviceType.State:
                device = MMStateDevice(label, ctrl, move_timeout=move_timeout, name=key)
            case DeviceType.Generic:
                device = MMDevice(label, ctrl, name=key)
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
    ctrl: MMController, *, move_timeout: float | None = 10.0
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
    for group in ctrl.core.getAvailableConfigGroups():
        key = clean_name(group)
        if key in groups:
            raise ValueError(
                f"Group names {names[key]!r} and {group!r} both clean to {key!r}"
            )
        groups[key] = MMConfigGroup(
            group, ctrl, move_timeout=move_timeout, name=f"{key}_group"
        )
        names[key] = group
    return groups
