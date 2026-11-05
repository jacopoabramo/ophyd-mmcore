import asyncio
import logging
import threading
import time

import pytest
from ophyd_async.core import SignalR, SignalRW
from ophyd_async.testing import MonitorQueue
from pymmcore_plus.experimental.unicore import UniMMCore

from ophyd_mmcore import MMController
from ophyd_mmcore._backend import MMSignalBackend, MMSource


def prop(ctrl: MMController, label: str, name: str, datatype=None):
    return MMSignalBackend(datatype, ctrl.property_source(label, name))


async def test_property_signal_round_trip(ctrl):
    """Write a property through a signal and read it back as an int."""
    sig = SignalRW(prop(ctrl, "Cam", "Gain"), name="gain")
    await sig.connect()
    await sig.set(3)
    assert await sig.get_value() == 3
    assert sig.datatype is int


async def test_property_event_updates_subscriber(ctrl, uni_core):
    """Deliver the current value on subscribe, then a change made on the core."""
    sig = SignalR(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    with MonitorQueue(sig) as q:
        await q.assert_updates("A")
        uni_core.setProperty("Wheel", "Label", "C")
        await q.assert_updates("C")


async def test_property_event_from_core_thread_reaches_the_loop(ctrl, uni_core):
    """Deliver a change made on a worker thread to the loop that subscribed."""
    sig = SignalR(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    with MonitorQueue(sig) as q:
        await q.assert_updates("A")
        await asyncio.to_thread(uni_core.setProperty, "Wheel", "Label", "B")
        await q.assert_updates("B")


async def test_property_datakey_has_choices(ctrl):
    """Describe a property with allowed values as choices."""
    sig = SignalRW(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    assert (await sig.describe())["label"].get("choices") == ["A", "B", "C", "D"]


async def test_unbound_backend_refuses_to_connect():
    """Refuse to connect a signal whose backend was never bound to a source."""
    with pytest.raises(RuntimeError, match="gain"):
        await SignalRW(MMSignalBackend(int), name="gain").connect()


async def test_write_without_setter_is_refused(ctrl):
    """Refuse a put on a signal whose source cannot write."""
    source = ctrl.property_source("Wheel", "Label")
    backend = MMSignalBackend(str, MMSource(uri=source.uri, read=source.read))
    sig = SignalRW(backend, name="label")
    await sig.connect()
    with pytest.raises(RuntimeError, match="mmcore://Wheel/Label"):
        await sig.set("B")


async def test_event_after_unsubscribe_is_ignored(ctrl, uni_core):
    """Drop an event that arrives after the subscriber left."""
    sig = SignalR(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    seen = []
    sig.subscribe_reading(seen.append)
    await asyncio.sleep(0.05)
    sig.clear_sub(seen.append)
    uni_core.setProperty("Wheel", "Label", "D")
    await asyncio.sleep(0.05)
    assert [r["label"]["value"] for r in seen] == ["A"]


async def test_event_after_close_is_ignored(ctrl, uni_core):
    """Emit a core event after close without raising on the core thread."""
    sig = SignalR(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    sig.subscribe_reading(lambda r: None)
    ctrl.close()
    uni_core.setProperty("Wheel", "Label", "D")
    ctrl.close()


def test_event_after_loop_closed_is_ignored(uni_core: UniMMCore):
    """Emit a core event after the subscribing loop closed without raising."""
    controller = MMController(uni_core)
    loop = asyncio.new_event_loop()

    async def subscribe() -> None:
        controller.subscribe(("Wheel", "Label"), lambda value, timestamp: None)

    loop.run_until_complete(subscribe())
    loop.close()
    uni_core.setProperty("Wheel", "Label", "D")
    controller.close()


@pytest.mark.parametrize(("datatype", "expected"), [(float, 3.0), (str, "3")])
async def test_declared_datatype_converts_value_and_event(
    ctrl, uni_core, datatype, expected
):
    """Return the declared datatype for a read and for an event, not the property's own."""
    sig = SignalR(prop(ctrl, "Wheel", "State", datatype), name="gain")
    await sig.connect()
    updates: asyncio.Queue = asyncio.Queue()
    sig.subscribe_reading(updates.put_nowait)
    await updates.get()
    uni_core.setProperty("Wheel", "State", 3)
    value = (await updates.get())["gain"]["value"]
    assert (value, type(value)) == (expected, datatype)
    value = await sig.get_value()
    assert (value, type(value)) == (expected, datatype)


async def test_source_without_subscribe_delivers_value_once():
    """Deliver the current value once to a subscriber of a source that cannot monitor."""

    async def read() -> int:
        return 7

    sig = SignalR(MMSignalBackend(int, MMSource("mmcore://x", read)), name="x")
    await sig.connect()
    updates: asyncio.Queue = asyncio.Queue()
    sig.subscribe_reading(updates.put_nowait)
    assert (await asyncio.wait_for(updates.get(), 1))["x"]["value"] == 7
    await asyncio.sleep(0.05)
    assert updates.empty()


async def test_failed_initial_read_is_logged(caplog):
    """Log the source of a signal whose first read after subscribing fails."""
    reads = 0

    async def read() -> int:
        nonlocal reads
        reads += 1
        if reads > 1:
            raise OSError("device gone")
        return 1

    sig = SignalR(MMSignalBackend(int, MMSource("mmcore://x", read)), name="x")
    await sig.connect()
    with caplog.at_level(logging.ERROR):
        sig.subscribe_reading(lambda reading: None)
        await asyncio.sleep(0.05)
    assert "mmcore://x" in caplog.text


async def test_event_timestamp_is_arrival_time(ctrl, uni_core):
    """Stamp an event when it reaches the core thread, not when the loop handles it."""
    sig = SignalR(prop(ctrl, "Wheel", "Label"), name="label")
    await sig.connect()
    updates: asyncio.Queue = asyncio.Queue()
    sig.subscribe_reading(updates.put_nowait)
    await updates.get()
    thread = threading.Thread(target=uni_core.setProperty, args=("Wheel", "Label", "B"))
    thread.start()
    thread.join()
    time.sleep(0.3)
    reading = (await updates.get())["label"]
    assert time.time() - reading["timestamp"] > 0.25
