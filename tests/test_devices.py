from typing import Annotated as A

import pytest
from ophyd_async.core import Command, SignalR, SignalRW, init_devices
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.testing import assert_reading, partial_reading

from ophyd_mmcore import Label, MMDevice, MMSystem, Property
from ophyd_mmcore._connector import MMConnector, clean_name


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
