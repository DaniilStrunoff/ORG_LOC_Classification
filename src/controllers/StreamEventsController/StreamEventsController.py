import asyncio, json, threading
from src.events.Events import Event
from typing import AsyncIterator, Dict, Any, Set
from src.controllers.IController import IController


class StreamEventsController(IController):
    def __init__(self) -> None:
        self._subs: Set[asyncio.Queue[Dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _notify(self, evt_name: str, payload: Dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            for q in list(self._subs):
                loop.call_soon_threadsafe(
                    q.put_nowait, {"_evt": evt_name, "_data": payload}
                )  # pyright: ignore[reportUnknownArgumentType]

    def emit(self, ev: Event) -> None:
        self._notify(ev.type, ev.dump())

    async def event_stream(self) -> AsyncIterator[bytes]:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subs.add(q)
        try:
            yield b": open\n\n"
            while True:
                msg = await q.get()
                evt = msg["_evt"]
                data = json.dumps(msg["_data"], ensure_ascii=False)
                yield f"event: {evt}\n".encode() + f"data: {data}\n\n".encode()
        finally:
            self._subs.discard(q)
