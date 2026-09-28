"""Установка лидара на поезде: из системы лидара — в систему тракта.

Тракт считает, что вперёд по ходу поезда — ось −y, вверх — +z, влево — +x (так стоят
лидары в записях хакатона). Если лидар стоит иначе, облако сразу на входе поворачивается
в эту систему, а результаты для RViz и систем поезда (маркеры, `/fod/obstacles`)
переводятся обратно в систему лидара.

Установка задаётся так:

- `forward` — какая ось лидара смотрит вперёд по ходу: `-y` (по умолчанию), `+x`, `-x`, `+y`;
- `up` — какая ось смотрит вверх: `+z` (по умолчанию) или `-z` (лидар вверх ногами);
- `rpy` — малые поправки после этого, градусы: крен (+ — верх лидара наклонён вправо),
  тангаж (+ — лидар смотрит вниз), рыскание (+ — лидар повёрнут влево);
- `xyz` — где лидар относительно точки отсчёта тракта, м. Обычно не нужно: тракт сам
  находит головки рельсов и ось пути относительно лидара.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def _axis(name: str) -> np.ndarray:
    key = name.strip().lower()
    key = key if key[0] in "+-" else "+" + key
    if key not in AXES:
        raise ValueError(f"ось «{name}»: нужна одна из {', '.join(AXES)}")
    return np.array(AXES[key], dtype=np.float64)


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    x, y, z = axis
    c, s, t = np.cos(a), np.sin(a), 1.0 - np.cos(a)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


@dataclass
class Mount:
    forward: str = "-y"
    up: str = "+z"
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    R: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        f, u = _axis(self.forward), _axis(self.up)
        if abs(float(f @ u)) > 1e-9:
            raise ValueError(f"forward {self.forward} и up {self.up} должны быть разными осями")
        left = np.cross(u, f)
        base = np.stack([left, -f, u])               # строки — оси тракта в системе лидара
        roll, pitch, yaw = self.rpy
        fix = _rot(np.array([0.0, 0.0, 1.0]), yaw) @ _rot(np.array([1.0, 0.0, 0.0]), pitch) @ _rot(np.array([0.0, -1.0, 0.0]), roll)
        self.R = fix @ base
        self.t = -self.R @ np.asarray(self.xyz, dtype=np.float64)

    @property
    def identity(self) -> bool:
        return bool(np.allclose(self.R, np.eye(3)) and np.allclose(self.t, 0.0))

    def to_track(self, xyz: np.ndarray) -> np.ndarray:
        """Точки лидара → система тракта (тот же dtype)."""
        if self.identity:
            return xyz
        out = xyz.astype(np.float64) @ self.R.T + self.t
        return out.astype(xyz.dtype, copy=False)

    def apply(self, cloud):
        """Кадр лидара (упорядоченный, колонка × 128 колец) → кадр в системе тракта.

        Лидар вверх ногами: кольца в колонке идут в обратном порядке, чтобы развёртка
        для сети рельсов осталась «небо сверху».
        """
        from dataclasses import replace

        from fod.cloud import N_RINGS

        if self.identity:
            return cloud
        xyz = self.to_track(cloud.xyz)
        arrays = {"xyz": xyz, "intensity": cloud.intensity, "ring": cloud.ring, "timestamp": cloud.timestamp}
        if _axis(self.up)[2] < 0 and xyz.shape[0] % N_RINGS == 0:
            arrays = {k: v.reshape(-1, N_RINGS, *v.shape[1:])[:, ::-1].reshape(v.shape) for k, v in arrays.items()}
        return replace(cloud, **arrays)

    def to_sensor(self, xyz: np.ndarray) -> np.ndarray:
        """Точки тракта → система лидара."""
        xyz = np.asarray(xyz, dtype=np.float64)
        if self.identity:
            return xyz
        return (xyz - self.t) @ self.R

    def quaternion(self) -> tuple[float, float, float, float]:
        """Поворот тракт → лидар как (x, y, z, w): для ориентации коробок в RViz."""
        m = self.R.T
        w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
        x = np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
        y = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
        z = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
        return float(x), float(y), float(z), float(w)

    def describe(self) -> str:
        return f"вперёд {self.forward}, вверх {self.up}, поправки {tuple(self.rpy)}°, смещение {tuple(self.xyz)} м"


def add_mount_args(parser) -> None:
    g = parser.add_argument_group("установка лидара (по умолчанию как в записях хакатона: вперёд −y, вверх +z)")
    g.add_argument("--forward", default="-y", help="Ось лидара, смотрящая вперёд по ходу поезда: -y, +y, +x, -x.")
    g.add_argument("--up", default="+z", help="Ось лидара, смотрящая вверх: +z или -z.")
    g.add_argument("--rpy", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("КРЕН", "ТАНГАЖ", "РЫСКАНИЕ"),
                   help="Поправки углов установки, градусы.")


def mount_from_args(args) -> Mount:
    return Mount(forward=args.forward, up=args.up, rpy=tuple(args.rpy))
