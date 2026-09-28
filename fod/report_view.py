"""Кадр видео для заказчика: range image + вид сверху (`fod/obstacle_view.py`) и панель телеметрии.

Панель: уровень, скорость и рекомендуемая скорость, дальность детекции габарита, тормозной
путь, препятствия; справа графики за последние `HISTORY_S` секунд.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from fod.obstacle_view import BG, COL_AXIS, COL_CONFIRMED, COL_DIM, COL_SUSPECT, WIDTH, _put, render_obstacle_frame
from fod.telemetry import ATTENTION, CLEAR, OBSTACLE, BrakingModel, telemetry

PANEL_H = 210
TEXT_W = 560
HISTORY_S = 60.0
# Цвета графиков не повторяют цвета сцены (рельсы, габарит, ось): у каждого цвета одно значение.
COL_SPEED = (245, 245, 245)       # белый
COL_SAFE = (230, 230, 0)          # бирюзовый
COL_REACH = (255, 110, 170)       # фиолетовый
COL_STOP = (160, 160, 160)        # серый
COL_OVERSPEED = (180, 105, 255)   # розовый
COL_DIST = COL_CONFIRMED          # красный, как препятствие на сцене
COL_SUSP = COL_SUSPECT            # оранжевый, как подозрение на сцене

LEVEL_TEXT = {
    "UNKNOWN": ("NOT READY - track not found yet", COL_DIM),
    CLEAR: ("CLEAR", COL_AXIS),
    ATTENTION: ("ATTENTION", COL_SUSPECT),
    OBSTACLE: ("OBSTACLE", COL_CONFIRMED),
}


class History:
    """Ряды для графиков; при разрыве времени начинаются заново."""

    def __init__(self, span: float = HISTORY_S) -> None:
        self.span = span
        self.rows: deque[tuple[float, ...]] = deque()

    def push(self, stamp: float, *values: float) -> None:
        if self.rows and stamp < self.rows[-1][0]:
            self.rows.clear()
        self.rows.append((stamp, *values))
        while self.rows and stamp - self.rows[0][0] > self.span:
            self.rows.popleft()

    def column(self, k: int) -> np.ndarray:
        return np.array([r[k] for r in self.rows], dtype=np.float64)


class ReportRenderer:
    def __init__(self, model: BrakingModel | None = None) -> None:
        self.model = model or BrakingModel()
        self.history = History()
        self.t0: float | None = None

    def render(self, res, bag_name: str) -> np.ndarray:
        """`res` — `FrameResult` с `keep_view=True`; вставленный стендом объект обводится кружком."""
        from fod.pipeline import AXIS_GRID
        from fod.range_image import build_range_image, crop_elevation, crop_forward_sector

        xyz, intensity, cloud, rail, motion, cr_frame, det, detector = res.view
        tel = telemetry(res, self.model)
        self.history.push(res.stamp, tel.speed_kmh, tel.safe_kmh, tel.reach, tel.distance, tel.stopping_m, tel.suspect)
        if self.t0 is None or res.stamp < self.t0:
            self.t0 = res.stamp
        title = f"{bag_name}   frame {res.index:05d}   t = {res.stamp - self.t0:.1f} s"
        front = crop_elevation(crop_forward_sector(build_range_image(cloud), 60.0), -16.0, 8.0)
        axis = head = None
        if res.axis_n.size:
            axis = lambda q, a=res.axis_n: np.interp(q, AXIS_GRID, a)  # noqa: E731
            if res.axis_z.size:
                head = lambda q, a=res.axis_z: np.interp(q, AXIS_GRID, a)  # noqa: E731
        obj = None
        if res.injected:
            p = min(res.injected, key=lambda q: q.s)
            obj = (p.type, p.s, p.n)
            title += f"   test object: {p.type} at {p.s:.0f} m"
        top = render_obstacle_frame(xyz, intensity, rail, front, det, detector, obj, title=title, cr=cr_frame, axis=axis, head=head)
        panel = self._panel(res, tel)
        return np.vstack([top, panel])

    # --- панель ---------------------------------------------------------------

    def _panel(self, res, tel) -> np.ndarray:
        img = np.full((PANEL_H, WIDTH, 3), BG, np.uint8)
        cv2.line(img, (0, 0), (WIDTH, 0), (70, 70, 70), 1)
        text, color = LEVEL_TEXT.get(tel.level, LEVEL_TEXT["UNKNOWN"])
        if tel.level == OBSTACLE and math.isfinite(tel.distance):
            text += f" at {tel.distance:.1f} m"
            first = res.obstacles[0].first_seen if res.obstacles else float("nan")
            if math.isfinite(first):
                text += f" (first seen at {first:.0f} m)"
        elif tel.level == ATTENTION and math.isfinite(tel.suspect):
            text += f" - suspect at {tel.suspect:.1f} m"
        _put(img, text, (12, 34), 0.75, color, 2)
        safe = "--" if not math.isfinite(tel.safe_kmh) else f"{tel.safe_kmh:.0f}"
        reach = "--" if tel.reach <= 0 else f"{tel.reach:.0f} m"
        lines = [
            (f"speed {tel.speed_kmh:.1f} km/h" + (" - too fast" if tel.overspeed else ""),
             COL_OVERSPEED if tel.overspeed else COL_SPEED),
            (f"recommended {safe} km/h", COL_SAFE),
            (f"gauge checked to {reach}", COL_REACH),
            (f"stopping distance {tel.stopping_m:.0f} m", COL_STOP),
        ]
        for k, (t, c) in enumerate(lines):
            _put(img, t, (12 + (k % 2) * 280, 64 + (k // 2) * 24), 0.52, c)
        y = 122
        rows = [(o, COL_DIST, "obstacle") for o in res.obstacles] + [(o, COL_SUSP, "suspect ") for o in res.suspects]
        for o, c, kind in rows[:3]:
            first = f"first seen {o.first_seen:5.1f} m" if math.isfinite(o.first_seen) else ""
            _put(img, f"{kind} #{o.track_id:<4d} {o.distance:5.1f} m  {first}  offset {o.offset:+.2f} m  h {o.height:.2f} m",
                 (12, y), 0.45, c)
            y += 20
        if not rows:
            _put(img, "no obstacles in the gauge", (12, y), 0.45, COL_DIM)
        _put(img, f"{res.ms:.0f} ms per frame", (12, PANEL_H - 10), 0.42, COL_DIM)

        w = (WIDTH - TEXT_W - 30) // 2
        self._chart(img, (TEXT_W, 10, w, PANEL_H - 20), "km/h", [(1, COL_SPEED, "speed"), (2, COL_SAFE, "recommended")], 90.0)
        self._chart(img, (TEXT_W + w + 20, 10, w, PANEL_H - 20), "m",
                    [(3, COL_REACH, "gauge checked"), (5, COL_STOP, "stopping"), (6, COL_SUSP, "suspect"),
                     (4, COL_DIST, "obstacle")], 160.0)
        return img

    def _chart(self, img, box, unit: str, series, y_max: float) -> None:
        x0, y0, w, h = box
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
        for frac in (0.25, 0.5, 0.75):
            yy = int(y0 + h - frac * h)
            cv2.line(img, (x0, yy), (x0 + w, yy), (40, 40, 40), 1)
            _put(img, f"{frac * y_max:.0f}", (x0 + 3, yy - 3), 0.35, COL_DIM)
        _put(img, unit, (x0 + 3, y0 + 14), 0.4, COL_DIM)
        _put(img, f"-{HISTORY_S:.0f} s", (x0 + 3, y0 + h - 4), 0.35, COL_DIM)
        rows = self.history.rows
        if len(rows) < 2:
            return
        t = self.history.column(0)
        px = x0 + w - (t[-1] - t) / HISTORY_S * w
        for k, (col, color, label) in enumerate(series):
            v = self.history.column(col)
            py = y0 + h - np.clip(v, 0.0, y_max) / y_max * h
            ok = np.isfinite(py)
            for a in range(len(px) - 1):
                if ok[a] and ok[a + 1]:
                    cv2.line(img, (int(px[a]), int(py[a])), (int(px[a + 1]), int(py[a + 1])), color, 2, cv2.LINE_AA)
            _put(img, label, (x0 + w - 118, y0 + 16 + 16 * k), 0.4, color)
