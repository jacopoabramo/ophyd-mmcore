"""Examples of using MMDevice subclasses inside Bluesky plans.

Demonstrates both the imperative and declarative approaches to defining devices,
and shows how to use them in Bluesky plans.

No real Micro-Manager installation is required — CMMCorePlus calls are patched
in the ``core`` and ``worker`` fixtures defined in conftest.py.

Key design note
---------------
The Bluesky RunEngine runs on its own asyncio event loop (separate from the
pytest-asyncio loop).  Devices must be connected via ``ensure_connected``
*inside a plan* so that the connection — and crucially the asyncio.Event that
signals readiness — lives on the RE's loop.  Connecting on the pytest-asyncio
loop and then using the device in a plan will cause a TimeoutError when the
RE tries to read the signal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from collections.abc import Generator
from typing import Annotated as A
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import pytest
from bluesky.run_engine import RunEngine, TransitionError
from ophyd_async.core import SignalRW, TriggerInfo
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.plan_stubs import ensure_connected

from ophyd_async.testing import MovableBeamstop

from ophyd_mmcore import MMCamera, MMDevice, PropName
from ophyd_mmcore._signal import mmcore_signal_rw
from ophyd_mmcore._worker import MMCoreWorker


# ---------------------------------------------------------------------------
# Two equivalent camera devices: imperative and declarative
# ---------------------------------------------------------------------------


class MMCameraImperative(MMDevice):
    """Camera defined the imperative way — signals constructed in __init__."""

    def __init__(
        self, mm_label: str, worker: MMCoreWorker, name: str = ""
    ) -> None:
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.exposure = mmcore_signal_rw(float, mm_label, "Exposure", worker)
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.binning = mmcore_signal_rw(str, mm_label, "Binning", worker)
        super().__init__(mm_label, worker, name=name)


class MMCameraDeclarative(MMDevice):
    """Camera defined the declarative way — signals as annotated class attributes.

    Each attribute is typed as ``Annotated[SignalRW[T], PropName("..."), Format.*]``.
    The ``PropName`` annotation maps the attribute to a MM property name.
    The ``Format`` annotation controls how the signal participates in Bluesky verbs.
    No ``__init__`` override needed.
    """

    exposure: A[SignalRW[float], PropName("Exposure"), Format.HINTED_SIGNAL]
    binning:  A[SignalRW[str],   PropName("Binning"),  Format.CONFIG_SIGNAL]


@pytest.fixture
def RE() -> Generator[RunEngine, None, None]:
    """A Bluesky RunEngine on a dedicated event loop (one per test)."""
    loop = asyncio.new_event_loop()
    re = RunEngine({}, call_returns_result=True, loop=loop)
    yield re
    if re.state not in ("idle", "panicked"):
        try:
            re.halt()
        except TransitionError:
            pass
    loop.call_soon_threadsafe(loop.stop)
    re._th.join()
    loop.close()


@pytest.fixture(params=["imperative", "declarative"])
def cam(request: pytest.FixtureRequest, worker: MMCoreWorker) -> MMDevice:
    """Both camera styles, exercised by every plan test via parametrize."""
    if request.param == "imperative":
        return MMCameraImperative("Camera", worker, name="cam")
    return MMCameraDeclarative("Camera", worker, name="cam")



def test_read_single_signal_with_rd(RE: RunEngine, cam: MMDevice) -> None:
    """bps.rd reads a single signal value inside a plan."""
    result: list[float] = []

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(cam)
        value = yield from bps.rd(cam.exposure)  # type: ignore[attr-defined]
        result.append(value)

    RE(plan())
    assert result[0] == pytest.approx(10.0)


def test_set_signal_with_mv(
    RE: RunEngine, cam: MMDevice, worker: MMCoreWorker
) -> None:
    """bps.mv sets a signal value inside a plan."""
    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(cam)
        yield from bps.mv(cam.exposure, 50.0)  # type: ignore[attr-defined]

    RE(plan())
    assert float(worker.core.getProperty("Camera", "Exposure")) == pytest.approx(50.0)


def test_count_plan_reads_camera(RE: RunEngine, cam: MMDevice) -> None:
    """bp.count produces start/event/stop documents with the exposure value."""
    docs: list[tuple[str, dict[Any, Any]]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(cam)
        yield from bp.count([cam])

    RE(plan())

    doc_names = [name for name, _ in docs]
    assert "start" in doc_names
    assert "event" in doc_names
    assert "stop" in doc_names

    events = [doc for name, doc in docs if name == "event"]
    assert len(events) == 1
    assert "cam-exposure" in events[0]["data"]
    assert events[0]["data"]["cam-exposure"] == pytest.approx(10.0)


def test_count_plan_multiple_shots(RE: RunEngine, cam: MMDevice) -> None:
    """bp.count with num=3 produces three event documents."""
    events: list[dict[Any, Any]] = []
    RE.subscribe(lambda name, doc: events.append(doc) if name == "event" else None)

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(cam)
        yield from bp.count([cam], num=3)

    RE(plan())

    assert len(events) == 3
    for event in events:
        assert event["data"]["cam-exposure"] == pytest.approx(10.0)


def test_read_configuration(RE: RunEngine, cam: MMDevice) -> None:
    """Configuration signals (binning) are readable via bps.rd."""
    config: list[dict[str, Any]] = []

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(cam)
        binning = yield from bps.rd(cam.binning)  # type: ignore[attr-defined]
        reading = yield from bps.read(cam)
        config.append({"binning": binning, "reading": reading})

    RE(plan())

    assert config[0]["binning"] == "1"
    assert config[0]["reading"]["cam-exposure"]["value"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Fly-scan tests (MMCamera via kickoff / complete)
# ---------------------------------------------------------------------------


@pytest.fixture
def fly_cam(worker: MMCoreWorker, tmp_path: Path) -> MMCamera:
    return MMCamera("Camera", worker, tmp_path / "fly.zarr", name="flycam")


def _fly_plan(
    flyer: MMCamera, trigger_info: TriggerInfo
) -> Generator[Any, Any, None]:
    """Minimal fly-scan plan compatible with ophyd-async's new stream model.

    ``bp.fly`` uses the old doubly-nested ``describe_collect`` API.  New-style
    detectors (ophyd-async 0.17+) return singly-nested DataKeys from
    ``describe_collect``, which must be pre-declared with ``bps.declare_stream``
    before ``bps.collect`` is called.
    """
    yield from ensure_connected(flyer)
    yield from bps.stage(flyer)
    yield from bps.prepare(flyer, trigger_info, wait=True)
    yield from bps.open_run()
    yield from bps.declare_stream(flyer, name="primary", collect=True)
    yield from bps.kickoff(flyer, wait=True)
    yield from bps.complete(flyer, wait=True)
    yield from bps.collect(flyer)
    yield from bps.close_run()
    yield from bps.unstage(flyer)


def test_fly_plan_produces_stream_documents(
    RE: RunEngine, fly_cam: MMCamera
) -> None:
    """Fly scan emits stream_resource and stream_datum documents."""
    docs: list[tuple[str, dict[Any, Any]]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))

    RE(_fly_plan(fly_cam, TriggerInfo(number_of_events=3)))

    doc_names = [name for name, _ in docs]
    assert "stream_resource" in doc_names
    assert "stream_datum" in doc_names


def test_mixed_device_with_non_mm_child(
    RE: RunEngine, worker: MMCoreWorker
) -> None:
    """An MMDevice subclass can carry a non-MM child device as a sibling.

    The MM connector wires PropName-annotated signals; the non-MM child
    (MovableBeamstop) is wired by DeviceFiller using its own connector and
    is completely independent of the MM backend.
    """

    class CameraWithBeamstop(MMDevice):
        exposure: A[SignalRW[float], PropName("Exposure"), Format.HINTED_SIGNAL]

        def __init__(self, mm_label: str, worker: MMCoreWorker, name: str = "") -> None:
            self.beamstop = MovableBeamstop()
            super().__init__(mm_label, worker, name=name)

    device = CameraWithBeamstop("Camera", worker, name="cam")

    results: list[Any] = []

    def plan() -> Generator[Any, Any, None]:
        yield from ensure_connected(device)
        # Read the MM-backed signal
        exposure = yield from bps.rd(device.exposure)
        # Move and read the soft-signal child
        yield from bps.mv(device.beamstop.x, 0.0)
        yield from bps.mv(device.beamstop.y, 0.0)
        bstop_x = yield from bps.rd(device.beamstop.x)
        results.append({"exposure": exposure, "beamstop_x": bstop_x})

    RE(plan())
    assert results[0]["exposure"] == pytest.approx(10.0)
    assert results[0]["beamstop_x"] == pytest.approx(0.0)


def test_fly_plan_stream_datum_covers_all_frames(
    RE: RunEngine, fly_cam: MMCamera
) -> None:
    """stream_datum indices stop == number of frames requested."""
    n = 4
    stream_datums: list[dict[Any, Any]] = []
    RE.subscribe(
        lambda name, doc: stream_datums.append(doc) if name == "stream_datum" else None
    )

    RE(_fly_plan(fly_cam, TriggerInfo(number_of_events=n)))

    assert stream_datums, "no stream_datum documents emitted"
    assert stream_datums[-1]["indices"]["stop"] == n