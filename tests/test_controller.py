import asyncio

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
        controller.subscribe(("Wheel", "Label"), lambda v: None)

    loop.run_until_complete(subscribe())
    loop.close()
    uni_core.setProperty("Wheel", "Label", "D")
    controller.close()
