"""Ячейка «последний кадр» для живого режима.

Колбэк подписки кладёт каждое облако через `put`, обработка забирает через
`get` / итерацию. Пока обработка занята, новый кадр перезаписывает старый,
поэтому отставание от лидара не копится: если тракт медленнее 100 мс, часть
кадров пропускается сама, ровно столько, сколько нужно. Тракт между кадрами
берёт интервал из меток времени, так что пропуск для него — просто кадр с
бо́льшим dt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Latest(Generic[T]):
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._item: T | None = None
        self._has = False
        self._closed = False
        self.received = 0
        self.dropped = 0

    def put(self, item: T) -> None:
        """Из любого потока; не блокирует."""
        with self._cv:
            if self._has:
                self.dropped += 1
            self._item, self._has = item, True
            self.received += 1
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, timeout: float | None = None) -> T | None:
        """Самый свежий кадр; ждёт, если его ещё нет. None — закрыта и пуста или истёк `timeout`."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while not self._has:
                if self._closed:
                    return None
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0.0:
                    return None
                self._cv.wait(left)
            item, self._item, self._has = self._item, None, False
            return item

    def __iter__(self) -> Iterator[T]:
        while (item := self.get()) is not None:
            yield item
