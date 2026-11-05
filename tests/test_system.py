import logging
from collections import defaultdict
from typing import Annotated as A
from typing import cast

import bluesky.plans as bp
import pytest
from bluesky import RunEngine
from conftest import FakeStage, provider, release
from ophyd_async.plan_stubs import ensure_connected
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore import (
    Label,
    MMCamera,
    MMConfigGroup,
    MMShutter,
    MMStateDevice,
    MMSystem,
    MMXYStage,
    MMZStage,
)


@pytest.mark.demo
async def test_load_matches_load_system_configuration(demo_core, tmp_path):
    """Leave a core in the state loadSystemConfiguration leaves it in."""
    reference = CMMCorePlus()
    reference.loadSystemConfiguration("MMConfig_demo.cfg")
    await MMSystem(
        "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
    ).connect()
    assert sorted(demo_core.getLoadedDevices()) == sorted(reference.getLoadedDevices())
    for role in ("Camera", "Shutter", "Focus", "XYStage", "AutoFocus", "ChannelGroup"):
        assert demo_core.getProperty("Core", role) == reference.getProperty(
            "Core", role
        )
    assert demo_core.getAvailableConfigGroups() == reference.getAvailableConfigGroups()
    for group in reference.getAvailableConfigGroups():
        assert demo_core.getCurrentConfig(group) == reference.getCurrentConfig(group)
    assert demo_core.getPixelSizeUm() == reference.getPixelSizeUm()
    assert demo_core.getParentLabel("Camera") == reference.getParentLabel("Camera")
    reference.unloadAllDevices()


@pytest.mark.demo
async def test_load_applies_device_delays(demo_core, tmp_path):
    """Set the delay a configuration file gives a device, as loadSystemConfiguration does."""
    cfg = tmp_path / "scope.cfg"
    cfg.write_text(
        "Property,Core,Initialize,0\nDevice,Wheel,DemoCamera,DWheel\n"
        "Property,Core,Initialize,1\nDelay,Wheel,50.0\n"
    )
    reference = CMMCorePlus()
    try:
        reference.loadSystemConfiguration(cfg)
        await MMSystem(cfg, core=demo_core, path_provider=provider(tmp_path)).connect()
        assert reference.getDeviceDelayMs("Wheel") == 50.0
        assert demo_core.getDeviceDelayMs("Wheel") == 50.0
    finally:
        release(reference)


@pytest.mark.demo
async def test_relative_path_is_found_in_adapter_paths(
    demo_core, tmp_path, monkeypatch
):
    """Find a relative configuration file in the adapter search paths."""
    monkeypatch.chdir(tmp_path)
    await MMSystem(
        "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
    ).connect()
    assert "Camera" in demo_core.getLoadedDevices()


@pytest.mark.demo
async def test_other_adapter_under_a_label_is_refused(demo_core, tmp_path):
    """Refuse a file that names a loaded label with another adapter, loading nothing."""
    demo_core.loadDevice("Camera", "DemoCamera", "DXYStage")
    with pytest.raises(ValueError, match="Camera"):
        await MMSystem(
            "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
        ).connect()
    assert sorted(demo_core.getLoadedDevices()) == ["Camera", "Core"]


@pytest.mark.demo
async def test_failed_initialize_unloads_uninitialized_devices(
    demo_core, tmp_path, monkeypatch
):
    """Unload every device loaded but not initialized when one fails to initialize."""
    original = demo_core.initializeDevice
    initialized = []

    def failing(label):
        if label == "Z":
            raise RuntimeError("boom")
        original(label)
        initialized.append(label)

    monkeypatch.setattr(demo_core, "initializeDevice", failing)
    scope = MMSystem(
        "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
    )
    with pytest.raises(RuntimeError, match="boom"):
        await scope.connect()
    assert initialized
    assert set(demo_core.getLoadedDevices()) == {"Core", *initialized}
    monkeypatch.undo()
    await scope.connect(force_reconnect=True)
    assert "Z" in demo_core.getLoadedDevices()


@pytest.mark.demo
async def test_reconnect_reuses_and_logs(demo_core, tmp_path, caplog):
    """Reuse the devices of an earlier connect and log each at info."""
    scope = MMSystem(
        "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
    )
    await scope.connect()
    with caplog.at_level(logging.INFO):
        await scope.connect(force_reconnect=True)
    assert "'Camera' is already loaded, reusing it" in caplog.text


async def test_mock_connect_loads_nothing(tmp_path):
    """Connect a root built from a file in mock mode without loading the file."""
    core = CMMCorePlus()
    cfg = tmp_path / "scope.cfg"
    cfg.write_text("Device,Z,DemoCamera,DStage\n")
    await MMSystem(cfg, core=core, path_provider=provider(tmp_path)).connect(mock=True)
    assert list(core.getLoadedDevices()) == ["Core"]


@pytest.mark.parametrize("source", [42, None])
def test_unknown_source_is_refused(source):
    """Refuse a source that is neither a path nor a core."""
    with pytest.raises(TypeError):
        MMSystem(source)


def test_core_keyword_with_a_core_is_refused(bare_core):
    """Refuse core= when the source is already a core."""
    with pytest.raises(TypeError):
        MMSystem(bare_core, core=bare_core)


async def test_root_on_core_discovers_devices_and_groups(uni_core, tmp_path):
    """Add one child per loaded device and per group, chosen by type."""
    scope = MMSystem(uni_core, path_provider=provider(tmp_path), name="scope")
    await scope.connect()
    assert {key: type(child) for key, child in scope.children()} == {
        "cam": MMCamera,
        "cam2": MMCamera,
        "z": MMZStage,
        "xy": MMXYStage,
        "shutter": MMShutter,
        "wheel": MMStateDevice,
        "channel_group": MMConfigGroup,
    }
    assert dict(scope.children())["xy"].name == "scope-xy"


async def test_declared_child_is_not_discovered_twice(uni_core, tmp_path):
    """Skip a label a declared child binds under another attribute name."""

    class Scope(MMSystem):
        camera: A[MMCamera, Label("Cam")]

    scope = Scope(uni_core, path_provider=provider(tmp_path))
    await scope.connect()
    assert not hasattr(scope, "cam")


async def test_declared_attribute_with_other_label_clashes(uni_core, tmp_path):
    """Refuse a discovered key that a declared child holds for another label."""

    class Scope(MMSystem):
        cam: A[MMZStage, Label("Z")]

    with pytest.raises(ValueError, match="Cam"):
        await Scope(uni_core, path_provider=provider(tmp_path)).connect()


async def test_device_and_group_keys_clash(uni_core, tmp_path):
    """Refuse a device whose key equals a group's key."""
    uni_core.loadPyDevice("Channel_group", FakeStage())
    uni_core.initializeDevice("Channel_group")
    with pytest.raises(ValueError, match="Channel_group"):
        await MMSystem(uni_core, path_provider=provider(tmp_path)).connect()


@pytest.mark.parametrize("label", ["Name", "Parent", "Connect", "Trigger", "Kickoff"])
async def test_label_naming_a_root_attribute_is_refused(uni_core, tmp_path, label):
    """Refuse a device whose key is an attribute the root already has."""
    uni_core.loadPyDevice(label, FakeStage())
    uni_core.initializeDevice(label)
    with pytest.raises(ValueError, match=label):
        await MMSystem(uni_core, path_provider=provider(tmp_path)).connect()


async def test_camera_without_path_provider_is_refused(uni_core):
    """Refuse to discover a camera when the root has no path provider."""
    with pytest.raises(ValueError, match="Cam"):
        await MMSystem(uni_core).connect()


async def test_colliding_labels_raise(bare_core, tmp_path):
    """Refuse two device labels that clean to the same key."""
    for label in ("Cam-1", "Cam 1"):
        bare_core.loadPyDevice(label, FakeStage())
        bare_core.initializeDevice(label)
    with pytest.raises(ValueError, match="Cam-1"):
        await MMSystem(bare_core, path_provider=provider(tmp_path)).connect()


async def test_label_cleaning_to_nothing_is_refused(bare_core, tmp_path):
    """Refuse a device whose label has no letters or digits to make a key from."""
    bare_core.loadPyDevice("-", FakeStage())
    bare_core.initializeDevice("-")
    with pytest.raises(ValueError, match="empty key"):
        await MMSystem(bare_core, path_provider=provider(tmp_path)).connect()


async def test_reconnect_keeps_discovered_children(uni_core, tmp_path):
    """Connect again without adding or refusing the children found the first time."""
    scope = MMSystem(uni_core, path_provider=provider(tmp_path), name="scope")
    await scope.connect()
    xy = dict(scope.children())["xy"]
    await scope.connect(force_reconnect=True)
    assert dict(scope.children())["xy"] is xy


def test_count_of_one_camera_with_two_cameras_in_the_root(uni_core, tmp_path):
    """Stage a root with two cameras and write the events of one of them."""
    scope = MMSystem(uni_core, path_provider=provider(tmp_path))
    RE = RunEngine()
    docs = defaultdict(list)
    RE.subscribe(lambda name, doc: docs[name].append(doc))

    def plan():
        yield from ensure_connected(scope)
        yield from bp.count([cast("MMCamera", dict(scope.children())["cam2"])], 2)

    RE(plan())
    assert len(docs["event"]) == 2


def test_runs_of_each_camera_stage_and_unstage_through_the_root(uni_core, tmp_path):
    """Stage and unstage the cameras of the root so each run ends with the core idle."""
    scope = MMSystem(uni_core, path_provider=provider(tmp_path), name="scope")
    RE = RunEngine()
    docs = defaultdict(list)
    RE.subscribe(lambda name, doc: docs[name].append(doc))
    RE(ensure_connected(scope))
    for key in ("cam", "cam2"):
        docs.clear()
        RE(bp.count([cast("MMCamera", dict(scope.children())[key])]))
        assert len(docs["event"]) == 1
        assert not uni_core.isSequenceRunning()
        assert uni_core.getAutoShutter()
