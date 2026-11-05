"""Tests for building devices and groups from a loaded core."""

import pytest
from conftest import FakeStage
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from pymmcore_plus.experimental.unicore import UniMMCore

from ophyd_mmcore import (
    MMCamera,
    MMController,
    MMShutter,
    MMStateDevice,
    MMXYStage,
    MMZStage,
    make_config_groups,
    make_devices,
)


def test_make_devices_picks_classes_by_type(ctrl, tmp_path):
    """Build one device per loaded device, chosen by its type."""
    devices = make_devices(
        ctrl, path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    )
    assert {k: type(v) for k, v in devices.items()} == {
        "cam": MMCamera,
        "cam2": MMCamera,
        "z": MMZStage,
        "xy": MMXYStage,
        "shutter": MMShutter,
        "wheel": MMStateDevice,
    }
    assert devices["xy"].name == "xy"


def test_make_config_groups_names_with_suffix(ctrl):
    """Key groups by clean name and name them with a group suffix."""
    groups = make_config_groups(ctrl)
    assert list(groups) == ["channel"]
    assert groups["channel"].name == "channel_group"


def test_colliding_labels_raise(tmp_path):
    """Refuse two device labels that clean to the same key."""
    core = UniMMCore()
    for label in ("Cam-1", "Cam 1"):
        core.loadPyDevice(label, FakeStage())
        core.initializeDevice(label)
    with pytest.raises(ValueError, match="Cam-1"):
        make_devices(
            MMController(core),
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        )
    core.unloadAllDevices()
