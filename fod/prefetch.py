"""Конвейер в два потока: генератор крутится в фоне на `depth` элементов вперёд.

numpy, torch и small_gicp отпускают GIL на тяжёлых операциях, поэтому чтение и
одометрия следующего кадра идут одновременно с осью и детектором текущего.

`depth=0` — по запросу и точно ко времени: следующий элемент начинается не
раньше, чем потребитель забрал предыдущий, и с таким отставанием, чтобы быть
готовым к его следующему запросу (по средним временам работы обеих сторон).
В живом режиме так в конвейере не лежат устаревшие кадры, а из ячейки
последнего кадра (`fod/latest.py`) берётся самый свежий: задержка — сумма
этапов, а не число этапов × время самого медленного.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")
_END = object()
_EMA = 0.2
# Запас к оценке, когда поток должен начать: сколько-то мс плюс доля от времени потребителя на разброс.
_LEAD_S, _LEAD_SHARE = 0.005, 0.15


def prefetch(items: Iterable[T], depth: int = 2) -> Iterator[T]:
    """Те же элементы в том же порядке; исключение генератора поднимается у потребителя."""
    q: queue.Queue = queue.Queue(maxsize=max(depth, 1))
    ask = threading.Semaphore(1) if depth == 0 else None
    stop = threading.Event()
    # Средние: сколько потребитель работает между «забрал» и «просит следующий» и сколько считается элемент.
    busy = {"consumer": 0.0, "work": 0.0}

    def ema(key: str, value: float) -> None:
        busy[key] = value if busy[key] == 0.0 else busy[key] + _EMA * (value - busy[key])

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def wait_ask() -> bool:
        while not stop.is_set():
            if ask.acquire(timeout=0.1):
                return True
        return False

    def worker() -> None:
        try:
            it = iter(items)
            while True:
                if ask is not None:
                    if not wait_ask():
                        return
                    lag = busy["consumer"] * (1.0 - _LEAD_SHARE) - busy["work"] - _LEAD_S
                    if lag > 0.0 and stop.wait(lag):
                        return
                t0 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    put((_END, None))
                    return
                ema("work", time.perf_counter() - t0)
                if not put((item, None)):
                    return
        except BaseException as exc:  # noqa: BLE001
            put((_END, exc))

    thread = threading.Thread(target=worker, name="prefetch", daemon=True)
    thread.start()
    taken = None
    try:
        while True:
            if taken is not None:
                ema("consumer", time.perf_counter() - taken)
            item, exc = q.get()
            if item is _END:
                if exc is not None:
                    raise exc
                return
            taken = time.perf_counter()
            if ask is not None:
                ask.release()
            yield item
    finally:
        stop.set()
        thread.join(timeout=5.0)
