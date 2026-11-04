import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Annotated as A

import bluesky.plans as bp
import pytest
from bluesky import RunEngine
from ophyd_async.core import (
    StaticPathProvider,
    TriggerInfo,
    UUIDFilenameProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)

from ophyd_mmcore import CameraState, Label, MMCamera, MMController, MMSystem


async def test_prepare_names_the_store(bare_ctrl, tmp_path):
    """Set file_path and data_key from the path provider at prepare."""
    async with init_devices(mock=True):
        cam = MMCamera(
            "Cam",
            bare_ctrl,
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        )
    await cam.stage()
    await cam.prepare(TriggerInfo(number_of_events=3))
    path = await cam.outputs["cam"].file_path.get_value()
    assert path.startswith(str(tmp_path)) and path.endswith(".zarr")
    assert await cam.outputs["cam"].data_key.get_value() == "cam"


async def test_multi_camera_has_one_data_key_per_output(bare_ctrl, tmp_path):
    """Describe one stream per physical camera of a Multi Camera."""
    async with init_devices(mock=True):
        multi = MMCamera(
            "Multi",
            bare_ctrl,
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
            physical=("Cam1", "Cam2"),
        )
    await multi.stage()
    await multi.prepare(TriggerInfo(number_of_events=2))
    assert set(await multi.describe_collect()) == {"multi-cam1", "multi-cam2"}
    paths = {await o.file_path.get_value() for o in multi.outputs.values()}
    assert len(paths) == 2


def test_count_emits_stream_documents(ctrl, tmp_path):
    """Run count with three points and emit one resource and three datums for a zarr store."""
    RE = RunEngine()
    docs = defaultdict(list)
    RE.subscribe(lambda n, d: docs[n].append(d))
    with init_devices():
        cam = MMCamera(
            "Cam",
            ctrl,
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        )
    RE(bp.count([cam], num=3))
    assert len(docs["stream_resource"]) == 1
    assert docs["stream_resource"][0]["mimetype"] == "application/x-zarr"
    assert (
        sum(d["indices"]["stop"] - d["indices"]["start"] for d in docs["stream_datum"])
        == 3
    )


async def faulting_camera(ctrl: MMController, tmp_path: Path) -> MMCamera:
    """Return a mock camera that faults with "boom" when its capture starts."""
    async with init_devices(mock=True):
        cam = MMCamera(
            "Cam",
            ctrl,
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        )

    def fault(value: bool) -> bool:
        if value:
            set_mock_value(cam.state, CameraState.FAULTED)
            set_mock_value(cam.last_error, "boom")
        return False

    callback_on_mock_put(cam.capture, fault)
    await cam.stage()
    return cam


async def test_trigger_raises_fault(bare_ctrl, tmp_path):
    """Fail a trigger with the camera's last error as soon as the camera faults."""
    cam = await faulting_camera(bare_ctrl, tmp_path)
    async with asyncio.timeout(2):
        with pytest.raises(RuntimeError, match="boom"):
            await cam.trigger()


async def test_complete_raises_fault(bare_ctrl, tmp_path):
    """Fail a fly scan's complete with the camera's last error as soon as the camera faults."""
    cam = await faulting_camera(bare_ctrl, tmp_path)
    await cam.prepare(TriggerInfo(number_of_events=2))
    await cam.kickoff()
    async with asyncio.timeout(2):
        with pytest.raises(RuntimeError, match="boom"):
            await cam.complete()


class Rig(MMSystem):
    cam: A[MMCamera, Label("Cam")]


async def test_system_camera_uses_system_path_provider(bare_ctrl, tmp_path):
    """Name a declared camera's store with the system's path provider."""
    async with init_devices(mock=True):
        rig = Rig(
            bare_ctrl,
            path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path),
        )
    await rig.cam.stage()
    await rig.cam.prepare(TriggerInfo(number_of_events=1))
    assert (await rig.cam.outputs["cam"].file_path.get_value()).startswith(
        str(tmp_path)
    )
    assert await rig.cam.outputs["cam"].data_key.get_value() == "rig-cam"


def test_system_camera_needs_path_provider(bare_ctrl):
    """Refuse a declared camera when its system has no path provider."""
    with pytest.raises(ValueError, match="Cam"):
        Rig(bare_ctrl)
