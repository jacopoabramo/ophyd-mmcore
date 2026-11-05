import logging

import pytest
from conftest import provider
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore import MMSystem


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

    def failing(label):
        if label == "Z":
            raise RuntimeError("boom")
        original(label)

    monkeypatch.setattr(demo_core, "initializeDevice", failing)
    scope = MMSystem(
        "MMConfig_demo.cfg", core=demo_core, path_provider=provider(tmp_path)
    )
    with pytest.raises(RuntimeError, match="boom"):
        await scope.connect()
    assert "Z" not in demo_core.getLoadedDevices()
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
