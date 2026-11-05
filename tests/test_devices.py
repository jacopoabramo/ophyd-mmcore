import asyncio
from typing import Annotated as A

import pytest
from ophyd_async.core import (
    Command,
    SignalR,
    SignalRW,
    TriggerableCommand,
    init_devices,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.testing import MonitorQueue, assert_reading, partial_reading

from ophyd_mmcore import (
    Label,
    MMConfigGroup,
    MMDevice,
    MMShutter,
    MMStateDevice,
    MMSystem,
    MMXYStage,
    MMZStage,
    Property,
)
from ophyd_mmcore._connector import Halt, MMConnector, clean_name


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("CCDTemperature RO", "ccd_temperature_ro"),
        ("PixelType", "pixel_type"),
        ("Camera-1", "camera_1"),
        ("XY", "xy"),
        ("Exposure", "exposure"),
    ],
)
def test_clean_name(raw, clean):
    """Turn Micro-Manager names into lower-case identifiers."""
    assert clean_name(raw) == clean


class Wheel(MMDevice):
    label_: A[SignalRW[str], Property("Label"), Format.HINTED_SIGNAL]


async def test_annotated_property_is_read(ctrl):
    """Read an annotated property as a hinted signal."""
    async with init_devices():
        wheel = Wheel("Wheel", ctrl)
    await assert_reading(wheel, {"wheel-label_": partial_reading("A")})


async def test_allow_list_goes_to_configuration(ctrl):
    """Report allow-listed properties in read_configuration only."""
    async with init_devices():
        cam = MMDevice("Cam", ctrl, properties=["Gain"])
    assert "cam-properties-gain" in await cam.read_configuration()
    assert await cam.read() == {}


def test_colliding_property_names_raise(bare_ctrl):
    """Refuse two properties that clean to the same key."""
    with pytest.raises(ValueError, match="Pixel Type"):
        MMDevice("Cam", bare_ctrl, properties=["PixelType", "Pixel Type"])


class Bench(MMSystem):
    wheel: A[Wheel, Label("Wheel", properties=("State",)), Format.CHILD]


async def test_system_builds_labelled_children(ctrl):
    """Build a declared child from its Label and name it after the system."""
    async with init_devices():
        bench = Bench(ctrl)
    await assert_reading(bench, {"bench-wheel-label_": partial_reading("A")})
    assert "bench-wheel-properties-state" in await bench.wheel.read_configuration()


class Unmarked(MMDevice):
    power: SignalRW[float]


class Doubled(MMDevice):
    power: A[SignalRW[float], Property("Power"), Property("Level")]


class LabelOnSignal(MMDevice):
    power: A[SignalR[float], Label("Laser")]


class PropertyOnDevice(MMSystem):
    wheel: A[Wheel, Property("Label")]


class UnmarkedCommand(MMDevice):
    halt: Command[[], None]


class OptionalUnmarked(MMSystem):
    wheel: Wheel | None


class PropertyOnSystem(MMSystem):
    power: A[SignalR[float], Property("Power")]


@pytest.mark.parametrize(
    ("cls", "make", "match"),
    [
        (Unmarked, lambda c, ctrl: c("Laser", ctrl), "Unmarked.power"),
        (Doubled, lambda c, ctrl: c("Laser", ctrl), "Doubled.power"),
        (LabelOnSignal, lambda c, ctrl: c("Laser", ctrl), "LabelOnSignal.power"),
        (PropertyOnDevice, lambda c, ctrl: c(ctrl), "PropertyOnDevice.wheel"),
        (UnmarkedCommand, lambda c, ctrl: c("Laser", ctrl), "UnmarkedCommand.halt"),
        (OptionalUnmarked, lambda c, ctrl: c(ctrl), "OptionalUnmarked.wheel"),
        (PropertyOnSystem, lambda c, ctrl: c(ctrl), "PropertyOnSystem.power"),
    ],
)
def test_bad_marker_raises(bare_ctrl, cls, make, match):
    """Refuse an annotated child with no marker, several, or one that does not fit."""
    with pytest.raises(TypeError, match=match):
        make(cls, bare_ctrl)


def test_label_and_connector_are_exclusive(bare_ctrl):
    """Refuse a device given both a label and a connector."""
    with pytest.raises(ValueError):
        MMDevice("Wheel", bare_ctrl, connector=MMConnector(bare_ctrl, "Wheel"))


def test_device_needs_a_label_or_a_connector():
    """Refuse a device given neither a controller nor a connector."""
    with pytest.raises(ValueError):
        MMDevice("Wheel")


async def test_shutter_opens(ctrl):
    """Open the shutter and read it back as open."""
    async with init_devices():
        shutter = MMShutter("Shutter", ctrl)
    await shutter.set(True)
    assert await shutter.position.get_value() is True


async def test_state_device_moves_by_label(ctrl):
    """Move a state device by label and describe its labels as choices."""
    async with init_devices():
        wheel = MMStateDevice("Wheel", ctrl)
    await wheel.set("C")
    assert await wheel.position.get_value() == "C"
    assert (await wheel.describe())["wheel"].get("choices") == ["A", "B", "C", "D"]


async def test_config_group_applies_preset(ctrl, uni_core):
    """Apply a preset and read it back, or an empty name when none matches."""
    async with init_devices():
        channel = MMConfigGroup("Channel", ctrl)
    await channel.set("B")
    assert await channel.position.get_value() == "B"
    uni_core.setProperty("Wheel", "Label", "D")
    assert await channel.position.get_value() == ""
    assert await channel.locate() == {"setpoint": "B", "readback": ""}


async def test_rejected_state_keeps_previous_target(ctrl):
    """Keep the last target when the core refuses a state label."""
    async with init_devices():
        wheel = MMStateDevice("Wheel", ctrl)
    await wheel.set("B")
    with pytest.raises(RuntimeError, match="Label not defined"):
        await wheel.set("Z")
    assert await wheel.locate() == {"setpoint": "B", "readback": "B"}


@pytest.mark.parametrize(
    ("cls", "options"),
    [
        (MMShutter, {"properties": ["Name"]}),
        (MMStateDevice, {"move_timeout": 3.0}),
        (MMConfigGroup, {"move_timeout": 3.0}),
    ],
)
def test_connector_excludes_options(bare_ctrl, cls, options):
    """Refuse a device given a connector together with an option the connector holds."""
    with pytest.raises(ValueError):
        cls(connector=MMConnector(bare_ctrl, "Wheel"), **options)


class Scope(MMSystem):
    z: A[MMZStage, Label("Z"), Format.CHILD]
    xy: A[MMXYStage, Label("XY")]
    shutter: A[MMShutter, Label("Shutter")]
    channel: A[MMConfigGroup, Label("Channel")]


async def test_system_moves_declared_devices(ctrl):
    """Move every kind of declared child of a system with the system's move timeout."""
    async with init_devices():
        scope = Scope(ctrl, move_timeout=5.0)
    await asyncio.gather(
        scope.z.set(4.0),
        scope.xy.x.set(1.0),
        scope.shutter.set(True),
        scope.channel.set("B"),
    )
    assert (await scope.describe()).keys() == {"scope-z"}
    assert await scope.xy.x.position.get_value() == 1.0
    assert scope.xy.x.movable_logic.move_timeout == 5.0


@pytest.mark.parametrize(
    ("build", "signal", "change", "initial", "expected"),
    [
        (
            lambda ctrl: MMZStage("Z", ctrl),
            lambda d: d.position,
            lambda core: core.events.stagePositionChanged.emit("Z", 4.0),
            0.0,
            4.0,
        ),
        (
            lambda ctrl: MMXYStage("XY", ctrl),
            lambda d: d.x.position,
            lambda core: core.events.XYStagePositionChanged.emit("XY", 1.0, 2.0),
            0.0,
            1.0,
        ),
        (
            lambda ctrl: MMXYStage("XY", ctrl),
            lambda d: d.y.position,
            lambda core: core.events.XYStagePositionChanged.emit("XY", 1.0, 2.0),
            0.0,
            2.0,
        ),
        (
            lambda ctrl: MMShutter("Shutter", ctrl),
            lambda d: d.position,
            lambda core: core.setShutterOpen("Shutter", True),
            False,
            True,
        ),
        (
            lambda ctrl: MMStateDevice("Wheel", ctrl),
            lambda d: d.position,
            lambda core: core.setState("Wheel", 2),
            "A",
            "C",
        ),
        (
            lambda ctrl: MMStateDevice("Wheel", ctrl),
            lambda d: d.position,
            lambda core: core.setStateLabel("Wheel", "D"),
            "A",
            "D",
        ),
        (
            lambda ctrl: MMConfigGroup("Channel", ctrl),
            lambda d: d.position,
            lambda core: core.setConfig("Channel", "B"),
            "A",
            "B",
        ),
    ],
    ids=["z", "x", "y", "shutter", "state", "state-label", "config-group"],
)
async def test_change_on_the_core_reaches_the_position_signal(
    ctrl, uni_core, build, signal, change, initial, expected
):
    """Deliver a change the core reports to a subscriber of the position signal."""
    async with init_devices():
        device = build(ctrl)
    with MonitorQueue(signal(device)) as queue:
        await queue.assert_updates(initial)
        change(uni_core)
        await queue.assert_updates(expected)


class GroupProperty(MMConfigGroup):
    power: A[SignalRW[float], Property("Power")]


class GroupHalt(MMConfigGroup):
    halt: A[TriggerableCommand, Halt()]


@pytest.mark.parametrize(
    ("cls", "match"),
    [(GroupProperty, "GroupProperty.power"), (GroupHalt, "GroupHalt.halt")],
)
def test_config_group_refuses_property_and_halt(bare_ctrl, cls, match):
    """Refuse a Property or a Halt declared on a configuration group."""
    with pytest.raises(TypeError, match=match):
        cls("Channel", bare_ctrl)
