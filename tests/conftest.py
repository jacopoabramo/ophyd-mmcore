"""Shared fixtures: Python device doubles loaded into a `UniMMCore`."""

import time
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import DTypeLike
from pymmcore_plus.experimental.unicore import (
    ShutterDevice,
    SimpleCameraDevice,
    StageDevice,
    StateDevice,
    UniMMCore,
    XYStageDevice,
    pymm_property,
)

from ophyd_mmcore import MMController


class FakeCamera(SimpleCameraDevice):
    """Camera with a 16 x 24 uint16 sensor whose frames hold their number."""

    def __init__(self) -> None:
        super().__init__()
        self._exposure = 10.0
        self._frame = 0
        self._gain = 1
        self._sequence_running = False

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure

    def sensor_shape(self) -> tuple[int, int]:
        return (16, 24)

    def dtype(self) -> DTypeLike:
        return np.uint16

    def snap(self, buffer: np.ndarray) -> Mapping:
        buffer[...] = self._frame
        self._frame += 1
        return {}

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:
        self._sequence_running = True
        try:
            yield from super().start_sequence(n, get_buffer)
        finally:
            self._sequence_running = False

    @pymm_property
    def Gain(self) -> int:  # pyright: ignore[reportRedeclaration]
        return self._gain

    @Gain.setter
    def Gain(self, value: int) -> None:
        if self._sequence_running:
            raise RuntimeError("busy")
        self._gain = value


class FakeStage(StageDevice):
    """Z stage limited to -1000..1000 um that stays busy for 50 ms per move."""

    def __init__(self) -> None:
        super().__init__()
        self._target = 0.0
        self._busy_until = 0.0

    def set_position_um(self, val: float) -> None:
        if not -1000 <= val <= 1000:
            raise ValueError(f"{val} um is outside the stage limits")
        self._target = val
        self._busy_until = time.monotonic() + 0.05
        self.core.events.stagePositionChanged.emit(self.get_label(), val)

    def get_position_um(self) -> float:
        return self._target

    def busy(self) -> bool:
        return time.monotonic() < self._busy_until

    def home(self) -> None:
        self.set_position_um(0.0)

    def stop(self) -> None:
        self._busy_until = 0.0

    def set_origin(self) -> None:
        self._target = 0.0


class FakeXYStage(XYStageDevice):
    """XY stage that moves instantly."""

    def __init__(self) -> None:
        super().__init__()
        self._position = (0.0, 0.0)

    def set_position_um(self, x: float, y: float) -> None:
        self._position = (x, y)
        self.core.events.XYStagePositionChanged.emit(self.get_label(), x, y)

    def get_position_um(self) -> tuple[float, float]:
        return self._position

    def home(self) -> None:
        self._position = (0.0, 0.0)

    def stop(self) -> None:
        pass

    def set_origin_x(self) -> None:
        self._position = (0.0, self._position[1])

    def set_origin_y(self) -> None:
        self._position = (self._position[0], 0.0)


class FakeShutter(ShutterDevice):
    """Shutter that starts closed."""

    def __init__(self) -> None:
        super().__init__()
        self._open = False

    def get_open(self) -> bool:
        return self._open

    def set_open(self, open: bool) -> None:
        self._open = open


class FakeWheel(StateDevice):
    """Four-position wheel with labels A to D."""

    def __init__(self) -> None:
        super().__init__(dict(enumerate("ABCD")))
        self._state = 0

    def get_state(self) -> int:
        return self._state

    def set_state(self, position: int) -> None:
        self._state = position


@pytest.fixture
def uni_core() -> Iterator[UniMMCore]:
    """Return a core loaded with the fake devices and a `Channel` group."""
    core = UniMMCore()
    for label, device in (
        ("Cam", FakeCamera()),
        ("Cam2", FakeCamera()),
        ("Z", FakeStage()),
        ("XY", FakeXYStage()),
        ("Shutter", FakeShutter()),
        ("Wheel", FakeWheel()),
    ):
        core.loadPyDevice(label, device)
        core.initializeDevice(label)
    core.setCameraDevice("Cam")
    core.setShutterDevice("Shutter")
    core.setAutoShutter(True)
    for preset in ("A", "B"):
        core.defineConfig("Channel", preset, "Wheel", "Label", preset)
    yield core
    core.unloadAllDevices()


@pytest.fixture
def ctrl(uni_core: UniMMCore) -> Iterator[MMController]:
    """Return a controller on the `uni_core` fixture, closed after the test."""
    controller = MMController(uni_core)
    yield controller
    controller.close()


@pytest.fixture
def bare_ctrl() -> Iterator[MMController]:
    """Return a controller on a core with no devices."""
    controller = MMController(UniMMCore())
    yield controller
    controller.close()
