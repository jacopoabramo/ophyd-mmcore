# ophyd-mmcore

`ophyd-mmcore` lets `bluesky` plans drive Micro-Manager hardware. It wraps the devices that `pymmcore-plus` loads from a Micro-Manager configuration file as `ophyd-async` devices: cameras, Z and XY stages, shutters, state devices such as filter wheels, and configuration groups such as Channel. A camera writes its frames to a `.zarr` store named by an `ophyd-async` path provider, so a step scan or a fly scan produces the documents and files a `bluesky` script expects.

> [!WARNING]
> Not to use for production yet.

## Installing

The package needs Python 3.11 or later.

```bash
pip install ophyd-mmcore
```

Micro-Manager itself comes from `pymmcore-plus`. If you have no installation yet, `mmcore install` downloads one. Its `--test-adapters` option adds the demo adapters that the examples and the test suite use.

## Building the devices

Every device reads and drives a Micro-Manager core, a `CMMCorePlus` from `pymmcore-plus`. `MMSystem` is the usual way in: given a configuration file, it loads the file into a core at `connect()` and then adds a child for each device and configuration group the core holds.

```python
from pathlib import Path

import bluesky.plans as bp
from bluesky import RunEngine
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from ophyd_async.plan_stubs import ensure_connected

from ophyd_mmcore import MMSystem

provider = StaticPathProvider(UUIDFilenameProvider(), Path("data").resolve())
scope = MMSystem("MMConfig_demo.cfg", path_provider=provider, name="scope")


def plan():
    yield from ensure_connected(scope)
    yield from bp.count([scope.camera], num=3)


RunEngine()(plan())
```

A child is keyed by its cleaned label, so the device labelled `Camera` is `scope.camera`. A configuration group gets its cleaned name followed by `_group`, such as `scope.channel_group`, so that it never shares a key with a device. Each camera writes into the directory that the path provider gives it, which must therefore be absolute. Replace `MMConfig_demo.cfg` with your own file: a relative path that is not in the working directory is looked up in Micro-Manager's adapter folders.

`MMSystem` also takes a core whose devices are already loaded, and then loads nothing. Given a file together with `core=`, it loads the file into that core but keeps any device the core already holds under the same label and adapter. A label that the core holds with another adapter makes `connect()` raise `ValueError` before any device is loaded.

```python
scope = MMSystem(core, path_provider=provider, name="scope")
scope = MMSystem("MMConfig_demo.cfg", core=core, path_provider=provider, name="scope")
```

A plan that needs some children typed declares them in a subclass with `Label`. A declared child exists from construction under any attribute name, and the root adds the remaining devices and groups at `connect()`:

```python
from typing import Annotated as A

from ophyd_mmcore import Label, MMCamera, MMConfigGroup, MMSystem


class Scope(MMSystem):
    camera: A[MMCamera, Label("Camera")]
    channel: A[MMConfigGroup, Label("Channel")]
```

A single device can also stand alone, built from its label and a core:

```python
from ophyd_mmcore import MMCamera, MMZStage

z = MMZStage("Z", core, name="z")
camera = MMCamera("Camera", core, path_provider=provider, name="camera")
```

## AutoShutter

A configuration file often turns on Micro-Manager's AutoShutter, which opens the shutter whenever a camera runs a sequence. A camera here starts its sequence when it first acquires and stops it at `unstage`, so AutoShutter would keep the sample lit for the rest of the run. The controller therefore turns AutoShutter off when a sequence starts and restores it when the sequence ends. In your plan, open and close the shutter yourself by moving its `MMShutter` device.

## Development

Install the project and its tools with `uv sync`. The checks are `uv run prek run --all-files`, which runs `ruff` and `pyright`, and `uv run pytest`. The end-to-end tests need the demo adapters from `uv run mmcore install --test-adapters` and are skipped without them.

### Changelog entries

Each change a user will notice gets a file in `changelog.d/`, named after the pull request and the kind of change, such as `412.fixed.md`. Open the pull request first, so you know its number. A release collects the files into the [changelog](CHANGELOG.md), one entry each, and links each entry to its pull request.

| kind | use it when |
| --- | --- |
| `breaking` | code that uses `ophyd-mmcore` has to change |
| `added` | something new: a class, a function, a keyword, an option |
| `changed` | something that already existed behaves differently |
| `deprecated` | something still works but warns, and goes in a later release |
| `removed` | something is gone |
| `fixed` | something now works as its documentation says |
| `security` | a vulnerability is closed |

`towncrier` writes the file for you. Keep the text in single quotes, so the shell leaves its backticks alone:

```bash
uv run towncrier create 412.fixed.md --content 'A timeout given to `read` is applied.'
```

A second change of the same kind in one pull request goes in `412.fixed.2.md`.

An entry says what changed, in a line or two, and names what a user would type to reach it: a name the package exports as it is, and any other with its module. It doesn't say why: that belongs in the pull request.

```markdown
Good: Add `timeout=` to `read`, which gives up when no reply comes in time.
Bad:  Improve read.
```

A change to something not released yet gets no entry of its own. Edit the entry that added it instead, so the release says what users get.

A pull request nobody using `ophyd-mmcore` would notice, such as tests, CI, a refactor or docs alone, needs no entry. A maintainer gives it the `skip-changelog` label, and CI accepts it without one.
