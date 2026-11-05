# ophyd-mmcore

`ophyd-mmcore` wraps the devices of Micro-Manager, loaded through `pymmcore-plus`, as `ophyd-async` devices, so that `bluesky` plans can drive them. Python 3.11 is the lowest supported version.

## Layout

```
src/ophyd_mmcore/
  __init__.py      public names and __all__
  _controller.py   MMController: calls, events, stage targets, XY lock
  _camera_io.py    controller side of a camera: flags, pause, cache, store
  _frames.py       the frame router thread
  _backend.py      signal and command backends, the source type, CameraState
  _connector.py    MMConnector, the markers, name cleaning
  _devices.py      MMDevice, MMSystem, MMMoveLogic, MMShutter, MMStateDevice,
                   MMConfigGroup
  _stage.py        MMZStage, MMAxis, MMXYStage
  _camera.py       MMCamera, MMCameraOutput and the camera logics
  _config.py       make_devices, make_config_groups
tests/
  conftest.py      fixtures: a UniMMCore with Python devices, the demo configuration
  test_*.py        one module per source module, and test_plans.py end to end
```

## Commands

```bash
uv sync                               # install the project and its tools
uv run mmcore install --test-adapters # Micro-Manager demo adapters, for the demo tests
uv run pytest -q
uv run ruff check
uv run pyright
uv run prek run --all-files           # ruff check, ruff format --check and pyright
```

Add or change dependencies only with `uv add` and `uv remove`. `ruff` runs the rules `D`, `I`, `TC`, `UP`, `PLC0415` and `E402`, and `pyright` runs in `standard` mode over `src` and `tests`.

## Architecture rule

Only the controller side calls `CMMCorePlus`: `MMController` in `_controller.py`, together with `_camera_io.py`, `_frames.py` and `_config.py`. A device reaches the core through the sources and backends of `_backend.py`, which `MMConnector` binds while it builds the device. A device never calls a controller method and never imports `pymmcore_plus`, which is what lets it run with `connect(mock=True)` and no core.

## Where a new test goes

- The logic of a device and the documents it emits: a device in mock mode, with `connect(mock=True)` and no core.
- A rule the controller enforces, such as locks, pausing, stores, AutoShutter or faults: the controller on a `UniMMCore`, through the `ctrl` fixture.
- A plan from start to finish against Micro-Manager's own adapters: `tests/test_plans.py`, marked `demo`, which is skipped without the demo adapters.
