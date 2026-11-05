import asyncio
from typing import Annotated as A

import pytest
from ophyd_async.core import (
    Command,
    NotConnectedError,
    SignalR,
    SignalRW,
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
def test_property_key_is_the_cleaned_name(bare_core, raw, clean):
    """Key a property signal by its Micro-Manager name turned into a lower-case identifier."""
    assert list(MMDevice("Cam", bare_core, properties=[raw]).properties) == [clean]


class Wheel(MMDevice):
    label_: A[SignalRW[str], Property("Label"), Format.HINTED_SIGNAL]


async def test_annotated_property_is_read(uni_core):
    """Read an annotated property as a hinted signal."""
    async with init_devices():
        wheel = Wheel("Wheel", uni_core)
    await assert_reading(wheel, {"wheel-label_": partial_reading("A")})


async def test_allow_list_goes_to_configuration(uni_core):
    """Report allow-listed properties in read_configuration only."""
    async with init_devices():
        cam = MMDevice("Cam", uni_core, properties=["Gain"])
    assert "cam-properties-gain" in await cam.read_configuration()
    assert await cam.read() == {}


def test_colliding_property_names_raise(bare_core):
    """Refuse two properties that clean to the same key."""
    with pytest.raises(ValueError, match="Pixel Type"):
        MMDevice("Cam", bare_core, properties=["PixelType", "Pixel Type"])


class Bench(MMSystem):
    wheel: A[Wheel, Label("Wheel", properties=("State",)), Format.CHILD]


async def test_system_builds_labelled_children(uni_core):
    """Build a declared child from its Label and name it after the system."""
    async with init_devices():
        bench = Bench(uni_core)
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
        (Unmarked, lambda c, core: c("Laser", core), "Unmarked.power"),
        (Doubled, lambda c, core: c("Laser", core), "Doubled.power"),
        (LabelOnSignal, lambda c, core: c("Laser", core), "LabelOnSignal.power"),
        (PropertyOnDevice, lambda c, core: c(core), "PropertyOnDevice.wheel"),
        (UnmarkedCommand, lambda c, core: c("Laser", core), "UnmarkedCommand.halt"),
        (OptionalUnmarked, lambda c, core: c(core), "OptionalUnmarked.wheel"),
        (PropertyOnSystem, lambda c, core: c(core), "PropertyOnSystem.power"),
    ],
)
def test_bad_marker_raises(bare_core, cls, make, match):
    """Refuse an annotated child with no marker, several, or one that does not fit."""
    with pytest.raises(TypeError, match=match):
        make(cls, bare_core)


def test_device_needs_a_label_or_a_connector():
    """Refuse a device given neither a core nor a connector."""
    with pytest.raises(ValueError):
        MMDevice("Wheel")


def test_construction_does_not_touch_the_core(bare_core):
    """Build a device on a core that lacks its label without raising."""
    MMZStage("Z", bare_core, name="z")


async def test_device_on_missing_label_fails_to_connect(uni_core):
    """Refuse to connect a device whose label the core has not loaded."""
    stage = MMZStage("Nope", uni_core, name="stage")
    with pytest.raises(NotConnectedError, match="Nope"):
        await stage.connect()


async def test_group_on_missing_name_fails_to_connect(uni_core):
    """Refuse to connect a group the core does not define."""
    group = MMConfigGroup("Nope", uni_core, name="group")
    with pytest.raises(NotConnectedError, match="Nope"):
        await group.connect()


async def test_shutter_opens(uni_core):
    """Open the shutter and read it back as open."""
    async with init_devices():
        shutter = MMShutter("Shutter", uni_core)
    await shutter.set(True)
    assert await shutter.position.get_value() is True


async def test_state_device_moves_by_label(uni_core):
    """Move a state device by label and describe its labels as choices."""
    async with init_devices():
        wheel = MMStateDevice("Wheel", uni_core)
    await wheel.set("C")
    assert await wheel.position.get_value() == "C"
    assert (await wheel.describe())["wheel"].get("choices") == ["A", "B", "C", "D"]


async def test_config_group_applies_preset(uni_core):
    """Apply a preset and read it back, or an empty name when none matches."""
    async with init_devices():
        channel = MMConfigGroup("Channel", uni_core)
    await channel.set("B")
    assert await channel.position.get_value() == "B"
    uni_core.setProperty("Wheel", "Label", "D")
    assert await channel.position.get_value() == ""
    assert await channel.locate() == {"setpoint": "B", "readback": ""}


async def test_rejected_state_keeps_previous_target(uni_core):
    """Keep the last target when the core refuses a state label."""
    async with init_devices():
        wheel = MMStateDevice("Wheel", uni_core)
    await wheel.set("B")
    with pytest.raises(RuntimeError, match="Label not defined"):
        await wheel.set("Z")
    assert await wheel.locate() == {"setpoint": "B", "readback": "B"}


class Scope(MMSystem):
    z: A[MMZStage, Label("Z"), Format.CHILD]
    xy: A[MMXYStage, Label("XY")]
    shutter: A[MMShutter, Label("Shutter")]
    channel: A[MMConfigGroup, Label("Channel")]


async def test_system_moves_declared_devices(uni_core):
    """Move every kind of declared child of a system with the system's move timeout."""
    async with init_devices():
        scope = Scope(uni_core, move_timeout=5.0)
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
            lambda core: MMZStage("Z", core),
            lambda d: d.position,
            lambda core: core.events.stagePositionChanged.emit("Z", 4.0),
            0.0,
            4.0,
        ),
        (
            lambda core: MMXYStage("XY", core),
            lambda d: d.x.position,
            lambda core: core.events.XYStagePositionChanged.emit("XY", 1.0, 2.0),
            0.0,
            1.0,
        ),
        (
            lambda core: MMXYStage("XY", core),
            lambda d: d.y.position,
            lambda core: core.events.XYStagePositionChanged.emit("XY", 1.0, 2.0),
            0.0,
            2.0,
        ),
        (
            lambda core: MMShutter("Shutter", core),
            lambda d: d.position,
            lambda core: core.setShutterOpen("Shutter", True),
            False,
            True,
        ),
        (
            lambda core: MMStateDevice("Wheel", core),
            lambda d: d.position,
            lambda core: core.setState("Wheel", 2),
            "A",
            "C",
        ),
        (
            lambda core: MMStateDevice("Wheel", core),
            lambda d: d.position,
            lambda core: core.setStateLabel("Wheel", "D"),
            "A",
            "D",
        ),
        (
            lambda core: MMConfigGroup("Channel", core),
            lambda d: d.position,
            lambda core: core.setConfig("Channel", "B"),
            "A",
            "B",
        ),
    ],
    ids=["z", "x", "y", "shutter", "state", "state-label", "config-group"],
)
async def test_change_on_the_core_reaches_the_position_signal(
    uni_core, build, signal, change, initial, expected
):
    """Deliver a change the core reports to a subscriber of the position signal."""
    async with init_devices():
        device = build(uni_core)
    with MonitorQueue(signal(device)) as queue:
        await queue.assert_updates(initial)
        change(uni_core)
        await queue.assert_updates(expected)


class GroupProperty(MMConfigGroup):
    power: A[SignalRW[float], Property("Power")]


def test_config_group_refuses_property(bare_core):
    """Refuse a Property declared on a configuration group."""
    with pytest.raises(TypeError, match="GroupProperty.power"):
        GroupProperty("Channel", bare_core)
