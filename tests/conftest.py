"""Shared fixtures for ophyd-mmcore tests.

All tests run without a real Micro-Manager installation by patching
the low-level CMMCorePlus methods.  The psygnal event system is kept
real so that callback wiring (set_callback, devicePropertyChanged) can
be exercised properly.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core._constants import PropertyType

from ophyd_mmcore._worker import MMCoreWorker


@pytest.fixture
def core() -> Iterator[CMMCorePlus]:
    """Yield a CMMCorePlus instance with MM device calls patched out.

    Provides a Camera device with:
    - Exposure: float, rw, range 0-1000, current value 10.0
    - Binning:  str,   rw, allowed values ('1', '2', '4'), current '1'
    - ReadMode: str,   ro, current 'Standard'
    """
    instance = CMMCorePlus()

    def _get_property_type(device: str, prop: str) -> PropertyType:
        return {
            ("Camera", "Exposure"): PropertyType.Float,
            ("Camera", "Binning"): PropertyType.String,
            ("Camera", "ReadMode"): PropertyType.String,
        }.get((device, prop), PropertyType.String)

    def _get_property(device: str, prop: str) -> str:
        return {
            ("Camera", "Exposure"): "10.0",
            ("Camera", "Binning"): "1",
            ("Camera", "ReadMode"): "Standard",
        }.get((device, prop), "")

    def _is_read_only(device: str, prop: str) -> bool:
        return (device, prop) == ("Camera", "ReadMode")

    def _has_limits(device: str, prop: str) -> bool:
        return (device, prop) == ("Camera", "Exposure")

    def _lower_limit(device: str, prop: str) -> float:
        return 0.0 if (device, prop) == ("Camera", "Exposure") else 0.0

    def _upper_limit(device: str, prop: str) -> float:
        return 1000.0 if (device, prop) == ("Camera", "Exposure") else 0.0

    def _allowed_values(device: str, prop: str) -> tuple[str, ...]:
        return ("1", "2", "4") if (device, prop) == ("Camera", "Binning") else ()

    with (
        patch.object(instance, "getPropertyType", side_effect=_get_property_type),
        patch.object(instance, "getProperty", side_effect=_get_property),
        patch.object(instance, "setProperty"),
        patch.object(instance, "isPropertyReadOnly", side_effect=_is_read_only),
        patch.object(instance, "hasPropertyLimits", side_effect=_has_limits),
        patch.object(instance, "getPropertyLowerLimit", side_effect=_lower_limit),
        patch.object(instance, "getPropertyUpperLimit", side_effect=_upper_limit),
        patch.object(instance, "getAllowedPropertyValues", side_effect=_allowed_values),
    ):
        yield instance


@pytest.fixture
def worker(core: CMMCorePlus) -> Iterator[MMCoreWorker]:
    """Yield an MMCoreWorker wrapping the patched core."""
    w = MMCoreWorker(core)
    yield w
    w.stop()
