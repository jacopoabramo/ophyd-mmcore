from __future__ import annotations

import asyncio

import pytest
from pymmcore_plus import CMMCorePlus

from ophyd_mmcore._backend import MMPropertyBackend
from ophyd_mmcore._worker import MMCoreWorker


@pytest.mark.asyncio
async def test_backend_infers_float_datatype(worker: MMCoreWorker) -> None:
    """Datatype is inferred from PropertyType when not supplied."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    assert backend.datatype is float


@pytest.mark.asyncio
async def test_backend_get_value_returns_correct_type(worker: MMCoreWorker) -> None:
    """get_value() casts the raw MM string to the declared datatype."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)
    value = await backend.get_value()
    assert value == pytest.approx(10.0)
    assert isinstance(value, float)


@pytest.mark.asyncio
async def test_backend_get_reading_structure(worker: MMCoreWorker) -> None:
    """get_reading() returns a dict with value, timestamp, alarm_severity."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)
    reading = await backend.get_reading()
    assert reading["value"] == pytest.approx(10.0)
    assert reading["alarm_severity"] == 0
    assert reading["timestamp"] == pytest.approx(pytest.approx(__import__("time").time(), abs=2))


@pytest.mark.asyncio
async def test_backend_get_datakey_includes_limits(worker: MMCoreWorker) -> None:
    """get_datakey() populates limits from MM property limits."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)
    datakey = await backend.get_datakey(backend.source("exposure", read=True))
    assert "limits" in datakey
    assert datakey["limits"]["control"]["low"] == pytest.approx(0.0)
    assert datakey["limits"]["control"]["high"] == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_backend_get_datakey_includes_choices(worker: MMCoreWorker) -> None:
    """get_datakey() populates choices from MM allowed property values."""
    backend: MMPropertyBackend[str] = MMPropertyBackend(
        "Camera", "Binning", worker, datatype=str
    )
    await backend.connect(timeout=5.0)
    datakey = await backend.get_datakey(backend.source("binning", read=True))
    assert datakey["choices"] == ["1", "2", "4"]


@pytest.mark.asyncio
async def test_backend_put_calls_set_property(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    """put() calls CMMCorePlus.setProperty with the correct arguments."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)
    await backend.put(50.0)
    core.setProperty.assert_called_once_with("Camera", "Exposure", 50.0)  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_backend_set_callback_delivers_initial_value(
    worker: MMCoreWorker,
) -> None:
    """set_callback() immediately delivers the current value to the subscriber."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)

    received: list[float] = []
    backend.set_callback(lambda r: received.append(r["value"]))
    await asyncio.sleep(0.05)  # let the initial delivery land via call_soon_threadsafe

    assert len(received) == 1
    assert received[0] == pytest.approx(10.0)
    backend.set_callback(None)


@pytest.mark.asyncio
async def test_backend_set_callback_receives_property_changes(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    """set_callback() receives subsequent property change events from MM."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)

    received: list[float] = []
    backend.set_callback(lambda r: received.append(r["value"]))
    await asyncio.sleep(0.05)  # initial value

    # Simulate MM emitting a property change (e.g. hardware changed the exposure)
    core.events.propertyChanged.emit("Camera", "Exposure", "75.0")
    await asyncio.sleep(0.05)

    backend.set_callback(None)

    assert len(received) == 2  # initial + change
    assert received[-1] == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_backend_disconnect_callback(
    worker: MMCoreWorker, core: CMMCorePlus
) -> None:
    """set_callback(None) stops further deliveries."""
    backend: MMPropertyBackend[float] = MMPropertyBackend(
        "Camera", "Exposure", worker
    )
    await backend.connect(timeout=5.0)

    received: list[float] = []
    backend.set_callback(lambda r: received.append(r["value"]))
    await asyncio.sleep(0.05)
    backend.set_callback(None)

    count_after_disconnect = len(received)
    core.events.propertyChanged.emit("Camera", "Exposure", "99.0")
    await asyncio.sleep(0.05)

    # No new readings after disconnect
    assert len(received) == count_after_disconnect