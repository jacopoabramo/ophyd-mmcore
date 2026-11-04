"""End-to-end plans on the Micro-Manager demo adapters."""

from collections import defaultdict
from typing import cast
from urllib.parse import urlparse

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import pytest
import zarr
from bluesky import RunEngine
from ophyd_async.core import StaticPathProvider, TriggerInfo, UUIDFilenameProvider
from ophyd_async.plan_stubs import ensure_connected

from ophyd_mmcore import MMCamera, make_config_groups, make_devices

pytestmark = pytest.mark.demo


def fly(det, info):
    """Return a plan that connects `det`, flies `info` and collects its frames."""

    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def inner():
        yield from bps.prepare(det, info, wait=True)
        yield from bps.kickoff(det, wait=True)
        yield from bps.complete(det, wait=True)
        yield from bps.declare_stream(det, name="primary")
        yield from bps.collect(det, name="primary")

    yield from ensure_connected(det)
    yield from inner()


def run(plan):
    """Run `plan` and return the documents it emitted, by name."""
    RE = RunEngine()
    docs = defaultdict(list)
    RE.subscribe(lambda name, doc: docs[name].append(doc))
    RE(plan)
    return docs


def open_array(resource):
    """Return the array a stream resource points at."""
    store = zarr.open(urlparse(resource["uri"]).path.lstrip("/"))
    return cast("zarr.Array", store[resource["parameters"]["dataset"]])


def build(ctrl, tmp_path):
    """Return the devices of `ctrl` writing into `tmp_path`."""
    return make_devices(
        ctrl, path_provider=StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    )


def test_demo_builds_expected_devices(demo_ctrl, tmp_path):
    """Build the demo's camera, stages, shutter and groups from its cfg."""
    devices = build(demo_ctrl, tmp_path)
    assert {"camera", "z", "xy", "white_light_shutter"} <= devices.keys()
    assert {"camera", "channel", "objective"} <= make_config_groups(demo_ctrl).keys()


def test_step_scan_writes_one_store(demo_ctrl, tmp_path):
    """Write one frame per point of a Z scan into one zarr store."""
    devices = build(demo_ctrl, tmp_path)
    cam, z = cast("MMCamera", devices["camera"]), devices["z"]

    def plan():
        yield from ensure_connected(cam, z)
        yield from bp.scan([cam], z, 0, 10, 5)

    docs = run(plan())
    assert len(docs["stream_resource"]) == 1
    assert open_array(docs["stream_resource"][0]).shape[0] == 5


def test_fly_scan_with_two_points_per_event(demo_ctrl, tmp_path):
    """Write a fly scan's frames and describe them with the right shape."""
    cam = build(demo_ctrl, tmp_path)["camera"]
    info = TriggerInfo(number_of_events=10, collections_per_event=2)
    docs = run(fly(cam, info))
    array = open_array(docs["stream_resource"][0])
    height, width = array.shape[1:]
    assert array.shape[0] == 20
    descriptor = docs["descriptor"][0]["data_keys"]["camera"]
    assert descriptor["shape"] == [2, height, width]
    assert (
        sum(d["indices"]["stop"] - d["indices"]["start"] for d in docs["stream_datum"])
        == 10
    )


def test_channel_group_moves(demo_ctrl):
    """Apply a Channel preset from a plan."""
    group = make_config_groups(demo_ctrl)["channel"]

    def plan():
        yield from ensure_connected(group)
        yield from bps.mv(group, "DAPI")

    run(plan())
    assert demo_ctrl.core.getCurrentConfig("Channel") == "DAPI"


def test_make_devices_reads_multi_camera(multi_ctrl, tmp_path):
    """Give a Multi Camera one output per physical camera."""
    devices = build(multi_ctrl, tmp_path)
    assert set(cast("MMCamera", devices["multi"]).outputs) == {"cam1", "cam2"}


def test_multi_camera_fly_scan_writes_two_stores(multi_ctrl, tmp_path):
    """Write one store per physical camera from one Multi Camera sequence."""
    multi = build(multi_ctrl, tmp_path)["multi"]
    docs = run(fly(multi, TriggerInfo(number_of_events=6)))
    resources = docs["stream_resource"]
    assert len({r["uri"] for r in resources}) == 2
    assert {r["parameters"]["dataset"] for r in resources} == {
        "multi-cam1",
        "multi-cam2",
    }
    assert [open_array(r).shape[0] for r in resources] == [6, 6]
