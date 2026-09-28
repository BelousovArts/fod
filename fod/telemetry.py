"""Телеметрия кадра: рекомендуемая скорость и уровень тревоги.

Поезд должен успеть остановиться до конца видимого габарита, а если впереди
подтверждённое препятствие — до него:

    v·t_р + v²/(2a) = D − запас,  D = min(дальность габарита, расстояние до препятствия)

    v = a·(√(t_р² + 2·(D − запас)/a) − t_р)

Уровни: НЕ ГОТОВ (путь не найден), СВОБОДНО, ВНИМАНИЕ (в габарите кандидат, ещё не
подтверждённый), ПРЕПЯТСТВИЕ. Отдельно — `overspeed`: едем быстрее рекомендуемой.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

UNKNOWN, CLEAR, ATTENTION, OBSTACLE = "UNKNOWN", "CLEAR", "ATTENTION", "OBSTACLE"


@dataclass
class BrakingModel:
    decel: float = 1.3          # замедление при экстренном торможении, м/с²
    reaction: float = 1.0       # реакция системы и привода тормозов, с
    margin: float = 5.0         # остановиться не ближе, м
    v_max_kmh: float = 80.0     # потолок рекомендуемой скорости
    tolerance_kmh: float = 2.0  # превышение, с которого `overspeed`

    def stopping_distance(self, kmh: float) -> float:
        v = kmh / 3.6
        return v * self.reaction + v * v / (2.0 * self.decel) + self.margin

    def safe_kmh(self, distance: float) -> float:
        d = distance - self.margin
        if not math.isfinite(d) or d <= 0.0:
            return 0.0
        a, t = self.decel, self.reaction
        v = a * (math.sqrt(t * t + 2.0 * d / a) - t)
        return min(3.6 * v, self.v_max_kmh)


@dataclass
class Telemetry:
    level: str
    speed_kmh: float
    safe_kmh: float             # NaN — путь не найден
    reach: float                # дальность детекции габарита, м
    distance: float             # до ближайшего препятствия, м; NaN — нет
    stopping_m: float           # тормозной путь на текущей скорости, м
    overspeed: bool = False
    suspect: float = float("nan")   # до ближайшего подозрения, м

    def as_dict(self) -> dict:
        r = lambda v, k=1: None if not math.isfinite(v) else round(v, k)  # noqa: E731
        return {
            "level": self.level,
            "speed_kmh": r(self.speed_kmh),
            "safe_kmh": r(self.safe_kmh),
            "reach_m": r(self.reach),
            "distance_m": r(self.distance, 2),
            "stopping_m": r(self.stopping_m),
            "overspeed": self.overspeed,
            "suspect_m": r(self.suspect, 2),
        }


def telemetry(res, model: BrakingModel | None = None) -> Telemetry:
    """`res` — `fod.pipeline.FrameResult`."""
    m = model or BrakingModel()
    speed = float(res.speed_kmh)
    stopping = m.stopping_distance(speed)
    if res.status == UNKNOWN:
        return Telemetry(UNKNOWN, speed, float("nan"), 0.0, float("nan"), stopping)
    d = float(res.reach)
    if res.status == OBSTACLE and math.isfinite(res.distance):
        d = min(d, float(res.distance))
    safe = m.safe_kmh(d)
    suspects = getattr(res, "suspects", [])
    suspect = min((o.distance for o in suspects), default=float("nan"))
    level = OBSTACLE if res.status == OBSTACLE else ATTENTION if suspects else CLEAR
    return Telemetry(level, speed, safe, float(res.reach), float(res.distance), stopping,
                     speed > safe + m.tolerance_kmh, suspect)
