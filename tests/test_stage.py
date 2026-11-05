import asyncio
import threading

import pytest
from ophyd_async.core import init_devices

from ophyd_mmcore import MMXYStage, MMZStage


async def test_z_move_waits_for_idle(uni_core):
    """Complete a Z move only once the stage reports idle."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
    await z.set(12.5)
    assert (await z.locate()) == {"setpoint": 12.5, "readback": 12.5}
    assert (await z.read()).keys() == {"z"}


async def test_staged_read_reports_the_position_after_a_move(uni_core):
    """Read the new position of a staged Z stage and XY axis after each move."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
        xy = MMXYStage("XY", uni_core)
    await z.stage()
    await xy.stage()
    await z.set(5.0)
    await xy.x.set(3.0)
    assert (await z.read())["z"]["value"] == 5.0
    assert (await xy.read())["xy-x"]["value"] == 3.0
    await z.unstage()
    await xy.unstage()


async def test_concurrent_xy_axis_moves_both_land(uni_core):
    """Move both XY axes at once and land at both targets."""
    async with init_devices():
        xy = MMXYStage("XY", uni_core)
    await asyncio.gather(xy.x.set(1.0), xy.y.set(2.0))
    assert tuple(uni_core.getXYPosition("XY")) == (1.0, 2.0)


async def test_rejected_move_keeps_previous_setpoint(uni_core):
    """Raise the adapter's error and keep the previous setpoint."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
    await z.set(5.0)
    with pytest.raises(ValueError):
        await z.set(5000.0)
    assert (await z.locate())["setpoint"] == 5.0


async def test_stop_fails_the_move(uni_core):
    """Fail an ongoing move when stop is called."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
    status = z.set(100.0)
    started = asyncio.Event()
    status.watch(lambda *_, **__: started.set())
    await asyncio.wait_for(started.wait(), 5)
    await z.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await status


async def test_halt_leaves_the_other_xy_axis_where_it_stopped(uni_core, monkeypatch):
    """Keep a halted y axis where it stopped when x moves next."""
    async with init_devices():
        xy = MMXYStage("XY", uni_core)
    started = threading.Event()
    set_xy_position = uni_core.setXYPosition

    def signalling_set_xy_position(*args) -> None:
        set_xy_position(*args)
        started.set()

    monkeypatch.setattr(uni_core, "setXYPosition", signalling_set_xy_position)
    status = xy.y.set(100.0)
    await asyncio.to_thread(started.wait, 5)
    await xy.y.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await status
    assert uni_core.getYPosition("XY") == 0.0
    await xy.x.set(5.0)
    assert tuple(uni_core.getXYPosition("XY")) == (5.0, 0.0)
    assert (await xy.y.locate())["setpoint"] == 0.0


async def test_mock_stage_moves_instantly(bare_core):
    """Move a mock stage without a core."""
    async with init_devices(mock=True):
        z = MMZStage("Z", bare_core)
    await z.set(3.0)
    assert await z.position.get_value() == 3.0


async def test_move_longer_than_move_timeout_fails(uni_core):
    """Fail a move that takes longer than move_timeout."""
    async with init_devices():
        z = MMZStage("Z", uni_core, move_timeout=0.001)
    with pytest.raises(TimeoutError):
        await z.set(1.0)


async def test_position_follows_the_hardware_and_setpoint_keeps_the_target(uni_core):
    """Read the hardware position after an outside move while the setpoint stays."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
    await z.set(5.0)
    uni_core.setPosition("Z", 77.0)
    assert await z.position.get_value() == 77.0
    assert await z.locate() == {"setpoint": 5.0, "readback": 77.0}


async def test_rejected_move_reads_hardware_and_previous_target(uni_core):
    """Report the hardware as the readback and the previous target as the setpoint."""
    async with init_devices():
        z = MMZStage("Z", uni_core)
    await z.set(5.0)
    uni_core.setPosition("Z", 77.0)
    with pytest.raises(ValueError):
        await z.set(5000.0)
    assert await z.locate() == {"setpoint": 5.0, "readback": 77.0}
