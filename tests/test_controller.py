import asyncio
import logging
import subprocess
import sys
import threading
import time
from typing import Annotated as A

import pytest
from ophyd_async.core import SignalR, SignalRW
from ophyd_async.testing import MonitorQueue
from pymmcore_plus.experimental.unicore import UniMMCore

from ophyd_mmcore import MMDevice, MMStateDevice, Property


async def property_signal(core: UniMMCore, label: str, name: str) -> SignalRW:
    """Return the connected signal of the property `name` of the device `label`."""
    device = MMDevice(label, core, properties=[name], name=label.lower())
    await device.connect()
    return device.properties[name.lower()]


class WheelState(MMDevice):
    as_float: A[SignalR[float], Property("State")]
    as_str: A[SignalR[str], Property("State")]


async def test_property_signal_round_trip(uni_core):
    """Write a property through a signal and read it back as an int."""
    sig = await property_signal(uni_core, "Cam", "Gain")
    await sig.set(3)
    assert await sig.get_value() == 3
    assert sig.datatype is int


async def test_property_event_updates_subscriber(uni_core):
    """Deliver the current value on subscribe, then a change made on the core."""
    sig = await property_signal(uni_core, "Wheel", "Label")
    with MonitorQueue(sig) as q:
        await q.assert_updates("A")
        uni_core.setProperty("Wheel", "Label", "C")
        await q.assert_updates("C")


async def test_property_event_from_core_thread_reaches_the_loop(uni_core):
    """Deliver a change made on a worker thread to the loop that subscribed."""
    sig = await property_signal(uni_core, "Wheel", "Label")
    with MonitorQueue(sig) as q:
        await q.assert_updates("A")
        await asyncio.to_thread(uni_core.setProperty, "Wheel", "Label", "B")
        await q.assert_updates("B")


async def test_property_datakey_has_choices(uni_core):
    """Describe a property with allowed values as choices."""
    sig = await property_signal(uni_core, "Wheel", "Label")
    assert (await sig.describe())[sig.name].get("choices") == ["A", "B", "C", "D"]


async def test_event_after_unsubscribe_is_ignored(uni_core):
    """Drop an event that arrives after the subscriber left."""
    sig = await property_signal(uni_core, "Wheel", "Label")
    seen = []
    sig.subscribe_reading(seen.append)
    await asyncio.sleep(0.05)
    sig.clear_sub(seen.append)
    uni_core.setProperty("Wheel", "Label", "D")
    await asyncio.sleep(0.05)
    assert [r[sig.name]["value"] for r in seen] == ["A"]


def test_event_after_loop_closed_is_ignored(uni_core):
    """Emit a core event after the subscribing loop closed without raising."""
    wheel = MMStateDevice("Wheel", uni_core, name="wheel")
    loop = asyncio.new_event_loop()

    async def subscribe() -> None:
        await wheel.connect()
        wheel.position.subscribe_reading(lambda reading: None)
        await asyncio.sleep(0.05)

    loop.run_until_complete(subscribe())
    loop.close()
    uni_core.setProperty("Wheel", "Label", "D")


@pytest.mark.parametrize(
    ("attr", "datatype", "expected"), [("as_float", float, 3.0), ("as_str", str, "3")]
)
async def test_declared_datatype_converts_value_and_event(
    uni_core, attr, datatype, expected
):
    """Return the declared datatype for a read and for an event, not the property's own."""
    wheel = WheelState("Wheel", uni_core, name="wheel")
    await wheel.connect()
    sig = getattr(wheel, attr)
    updates: asyncio.Queue = asyncio.Queue()
    sig.subscribe_reading(updates.put_nowait)
    await updates.get()
    uni_core.setProperty("Wheel", "State", 3)
    value = (await updates.get())[sig.name]["value"]
    assert (value, type(value)) == (expected, datatype)
    value = await sig.get_value()
    assert (value, type(value)) == (expected, datatype)


async def test_failed_initial_read_is_logged(uni_core, monkeypatch, caplog):
    """Log the source of a signal whose first read after subscribing fails."""
    sig = await property_signal(uni_core, "Wheel", "Label")

    def unplugged(label: str, prop: str) -> str:
        raise OSError("device gone")

    monkeypatch.setattr(uni_core, "getProperty", unplugged)
    with caplog.at_level(logging.ERROR):
        sig.subscribe_reading(lambda reading: None)
        await asyncio.sleep(0.05)
    assert "mmcore://Wheel/Label" in caplog.text


async def test_event_timestamp_is_arrival_time(uni_core):
    """Stamp an event when it reaches the core thread, not when the loop handles it."""
    sig = await property_signal(uni_core, "Wheel", "Label")
    updates: asyncio.Queue = asyncio.Queue()
    sig.subscribe_reading(updates.put_nowait)
    await updates.get()
    thread = threading.Thread(target=uni_core.setProperty, args=("Wheel", "Label", "B"))
    thread.start()
    thread.join()
    time.sleep(0.3)
    reading = (await updates.get())[sig.name]
    assert time.time() - reading["timestamp"] > 0.25


EXIT_SCRIPT = """
import atexit
from pymmcore_plus import CMMCorePlus
core = CMMCorePlus()
core.loadSystemConfiguration("MMConfig_demo.cfg")
# Registered before ophyd_mmcore is imported, so it runs after the package's hook.
atexit.register(lambda: print(core.isSequenceRunning(), core.getAutoShutter()))
import asyncio, tempfile
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from ophyd_mmcore import MMCamera

async def main():
    provider = StaticPathProvider(UUIDFilenameProvider(), tempfile.mkdtemp())
    camera = MMCamera("Camera", core, path_provider=provider, name="camera")
    await camera.connect()
    await camera.stage()
    await camera.acquire.set(True)

asyncio.run(main())
"""


@pytest.mark.demo
def test_exit_stops_a_staged_camera(demo_core, tmp_path):
    """Stop the sequence and restore AutoShutter at exit when a camera was never unstaged."""
    script = tmp_path / "staged.py"
    script.write_text(EXIT_SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == ["False", "True"]
