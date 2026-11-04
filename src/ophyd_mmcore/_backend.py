import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import (
    NO_ARG_VOID_SIGNATURE,
    Callback,
    CommandBackend,
    SignalBackend,
    SignalDatatypeT,
    SignalMetadata,
    make_datakey,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MMSource(Generic[T]):
    """How a signal reads, writes and monitors one value of the core.

    `subscribe` takes a callback that receives each new value and the time it
    arrived, and returns the function that cancels the subscription.
    """

    uri: str
    read: Callable[[], Awaitable[T]]
    write: Callable[[T], Awaitable[None]] | None = None
    setpoint: Callable[[], Awaitable[T]] | None = None
    subscribe: Callable[[Callable[[T, float], None]], Callable[[], None]] | None = None
    infer: Callable[[], Awaitable[type[T]]] | None = None
    metadata: Callable[[], Awaitable[SignalMetadata]] | None = None


class MMSignalBackend(SignalBackend[SignalDatatypeT]):
    """Signal backend that reads and writes through an [`MMSource`][ophyd_mmcore._backend.MMSource].

    The backend can be created empty and bound later with `bind`. Connecting
    an unbound backend raises `RuntimeError`.
    """

    def __init__(
        self,
        datatype: type[SignalDatatypeT] | None,
        source: MMSource[SignalDatatypeT] | None = None,
    ) -> None:
        super().__init__(datatype)
        self._source = source
        self._name = ""
        self._unsubscribe: Callable[[], None] | None = None
        self._initial: asyncio.Task[None] | None = None

    def bind(self, source: MMSource[Any]) -> None:
        """Set the source the backend reads and writes."""
        self._source = source

    def _bound(self) -> MMSource[SignalDatatypeT]:
        if self._source is None:
            raise RuntimeError(f"Signal {self._name!r} is not bound to a source")
        return self._source

    def source(self, name: str, read: bool) -> str:
        self._name = name
        return f"mmcore://unbound/{name}" if self._source is None else self._source.uri

    async def connect(self, timeout: float) -> None:
        source = self._bound()
        if self.datatype is None:
            if source.infer is None:
                raise RuntimeError(f"{source.uri} has no datatype and cannot infer one")
            self.datatype = await source.infer()
        await self.get_value()

    def _convert(self, value: Any) -> Any:
        datatype: Any = self.datatype
        if not isinstance(datatype, type) or isinstance(value, datatype):
            return value
        if datatype is bool and isinstance(value, str):
            return value.strip().lower() in ("1", "true")
        return datatype(value)

    async def put(self, value: SignalDatatypeT | None) -> None:
        source = self._bound()
        if source.write is None:
            raise RuntimeError(f"{source.uri} is read-only")
        if value is None:
            raise RuntimeError(f"{source.uri} has no default value to write")
        await source.write(value)

    async def get_datakey(self, source: str) -> DataKey:
        bound = self._bound()
        metadata = (
            SignalMetadata() if bound.metadata is None else await bound.metadata()
        )
        value = await self.get_value()
        return make_datakey(self.datatype or type(value), value, source, metadata)

    async def get_reading(self) -> Reading[SignalDatatypeT]:
        return Reading(
            value=await self.get_value(), timestamp=time.time(), alarm_severity=0
        )

    async def get_value(self) -> SignalDatatypeT:
        return self._convert(await self._bound().read())

    async def get_setpoint(self) -> SignalDatatypeT:
        source = self._bound()
        if source.setpoint is None:
            return await self.get_value()
        return self._convert(await source.setpoint())

    def set_callback(self, callback: Callback[Reading[SignalDatatypeT]] | None) -> None:
        if callback is None:
            self._stop()
            return
        if self._unsubscribe is not None:
            raise RuntimeError("Cannot set a callback when one is already set")
        source = self._bound()
        event_seen = False

        def deliver(value: Any, timestamp: float) -> None:
            nonlocal event_seen
            event_seen = True
            callback(
                Reading(
                    value=self._convert(value), timestamp=timestamp, alarm_severity=0
                )
            )

        async def deliver_current() -> None:
            try:
                value = await source.read()
            except Exception:
                logger.exception("Initial read of %s failed", source.uri)
                return
            if not event_seen:
                deliver(value, time.time())

        subscribe = source.subscribe or (lambda _: lambda: None)
        self._unsubscribe = subscribe(deliver)
        self._initial = asyncio.get_running_loop().create_task(deliver_current())

    def _stop(self) -> None:
        if self._initial is not None:
            self._initial.cancel()
            self._initial = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None


class MMCommandBackend(CommandBackend[[], None]):
    """Command backend that runs a bound call with no arguments."""

    def __init__(self) -> None:
        super().__init__(NO_ARG_VOID_SIGNATURE)
        self._execute: Callable[[], Awaitable[None]] | None = None
        self._uri: str | None = None

    def bind(self, execute: Callable[[], Awaitable[None]], uri: str) -> None:
        """Set the call the command runs and its source."""
        self._execute = execute
        self._uri = uri

    def source(self, name: str) -> str:
        return f"mmcore://unbound/{name}" if self._uri is None else self._uri

    async def connect(self, timeout: float) -> None:
        if self._execute is None:
            raise RuntimeError("Command backend is not bound to a call")

    async def execute(self) -> None:
        if self._execute is None:
            raise RuntimeError("Command backend is not bound to a call")
        await self._execute()
