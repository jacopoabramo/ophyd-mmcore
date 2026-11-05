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

## Counting with a camera

`MMController.from_config` loads a Micro-Manager configuration file and `make_devices` builds one device for each device it loaded. Each camera writes into the directory that the path provider gives it, so that directory must be absolute.

```python
from pathlib import Path

import bluesky.plans as bp
from bluesky import RunEngine
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from ophyd_async.plan_stubs import ensure_connected

from ophyd_mmcore import MMController, make_devices

ctrl = MMController.from_config("MMConfig_demo.cfg")
provider = StaticPathProvider(UUIDFilenameProvider(), Path("data").resolve())
camera = make_devices(ctrl, path_provider=provider)["camera"]


def plan():
    yield from ensure_connected(camera)
    yield from bp.count([camera], num=3)


RunEngine()(plan())
ctrl.close()
```

Replace `MMConfig_demo.cfg` with your own configuration file. `make_devices` keys each device by its cleaned label, and `make_config_groups` does the same for configuration groups.

## AutoShutter

A configuration file often turns on Micro-Manager's AutoShutter, which opens the shutter whenever a camera runs a sequence. A camera here runs a sequence from `stage` to `unstage`, so AutoShutter would keep the sample lit for the whole run. The controller therefore turns AutoShutter off when a sequence starts and restores it when the sequence ends. In your plan, open and close the shutter yourself by moving its `MMShutter` device.

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
