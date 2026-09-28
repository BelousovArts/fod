"""Карта оси пути в мировых координатах: ось не строится заново в каждом кадре, а уточняется.

Узлы оси стоят в мире через `step` м вдоль пути. У каждого — дисперсия бокового
положения. Кадр даёт измерение: ось слияния этого кадра n(s) с σ(s) до его дальности.
Узлы переносятся в кадр по позе, и каждый обновляется скалярным Калманом вдоль n:
вблизи узел набрал десятки кадров и почти не двигается, вдали — едет за измерениями
со своей σ. Узлы за дальностью кадра не трогаются, пока не устареют, а дальность
оси отступает к дальности кадра плавно: если КР пропал в одном кадре, ось остаётся.
Шум процесса на кадр растёт с дальностью — дрейф поз и то, что прошлые измерения
кадра коррелированы.

За последним узлом ось продолжается дугой постоянной кривизны по последним `fit_span` м
(не прямой: на R = 400 м прямая через 50 м уходит на 3 м).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackMapConfig:
    step: float = 1.0
    back: float = 5.0
    # Шум процесса на кадр: q0 + q1·s, м.
    q0: float = 0.005
    q1: float = 0.0002
    # σ измерения кадра × r_scale: соседние кадры делят точки КР и состояние фильтра рельсов.
    r_scale: float = 1.5
    # Дисперсия узла не ниже rho · дисперсии измерения: ошибка кадров на одной дальности
    # общая и усреднением не уходит. Подъехал — ближнее точное измерение перебивает дальнее.
    rho: float = 0.3
    # Невязка больше gate·σ — узел неверен (стрелка, сбой), берётся измерение.
    gate: float = 4.0
    # Последние `edge` м до дальности кадра — конец линии КР, там она чаще неверна:
    # σ измерения растёт до ×1/edge_min к самому краю.
    edge: float = 15.0
    edge_min: float = 0.2
    # Узел за дальностью кадра живёт столько кадров без обновлений.
    max_age: int = 15
    # Дальность оси падает не быстрее retreat м за кадр сверх проезда: длина линии КР
    # от кадра к кадру гуляет на десятки метров. Растёт сразу.
    retreat: float = 2.0
    # Дальность оси — до последнего узла подряд с σ не больше trust_sigma: свежий узел
    # с конца линии КР входит в габарит, когда его подтвердят ещё кадры.
    trust_sigma: float = 0.20
    fit_span: float = 30.0
    max_curvature: float = 1.0 / 300.0
    s_cap: float = 150.0


class TrackMap:
    def __init__(self, config: TrackMapConfig | None = None) -> None:
        self.cfg = config or TrackMapConfig()
        self.reset()

    def reset(self) -> None:
        self.world = np.zeros((0, 3))
        self.var = np.zeros(0)
        self.age = np.zeros(0, dtype=np.int64)
        self.end: np.ndarray | None = None

    def update(
        self, pose: np.ndarray | None, grid: np.ndarray, n: np.ndarray, z: np.ndarray, sigma: np.ndarray, reach: float,
        floor: float = 0.0,
    ):
        """Измерение кадра на `grid` → ось из карты на `grid` и дальность, до которой есть узлы.

        `floor` — дальность последней пары рельсов этого кадра: ось не короче неё.
        """
        c = self.cfg
        if pose is None:
            self.reset()
            return n, z, reach
        reach = min(float(reach), c.s_cap)
        ss, ns, zs, var, age = self._to_sensor(pose)
        if ss.size:
            var = var + np.square(c.q0 + c.q1 * np.maximum(ss, 0.0))
            obs = (ss >= grid[0]) & (ss <= reach)
            m_n = np.interp(ss[obs], grid, n)
            m_z = np.interp(ss[obs], grid, z)
            r = self._meas_var(ss[obs], grid, sigma, reach, floor)
            p = var[obs]
            innov = m_n - ns[obs]
            k = np.where(innov * innov > c.gate**2 * (p + r), 1.0, p / (p + r))
            ns[obs] += k * innov
            zs[obs] += k * (m_z - zs[obs])
            var[obs] = np.where(k == 1.0, r, np.maximum((1.0 - k) * p, c.rho * r))
            age[obs] = 0
            age[~obs] += 1
            stale = np.flatnonzero(age <= c.max_age)
            last = int(stale[-1]) + 1 if stale.size else 0
            ss, ns, zs, var, age = ss[:last], ns[:last], zs[:last], var[:last], age[:last]
        start = float(ss[-1]) + c.step if ss.size else float(grid[0])
        new = np.arange(start, reach + 1e-6, c.step)
        if new.size:
            ss = np.concatenate([ss, new])
            ns = np.concatenate([ns, np.interp(new, grid, n)])
            zs = np.concatenate([zs, np.interp(new, grid, z)])
            var = np.concatenate([var, self._meas_var(new, grid, sigma, reach, floor)])
            age = np.concatenate([age, np.zeros(new.size, dtype=np.int64)])
        self.var, self.age = var, age
        self.world = np.column_stack([ns, -ss, zs]) @ pose[:3, :3].T + pose[:3, 3]
        if ss.size < 2:
            return n, z, reach
        bad = np.flatnonzero((var > c.trust_sigma**2) & (ss > 0.0))
        shown = float(ss[bad[0] - 1]) if bad.size and bad[0] > 0 else float(ss[-1])
        if self.end is not None:
            inv = np.linalg.inv(pose)
            prev = float(-(inv[1, :3] @ self.end + inv[1, 3]))
            shown = min(max(shown, prev - c.retreat), float(ss[-1]))
        shown = min(max(shown, min(floor, float(ss[-1]))), c.s_cap)
        last = int(np.searchsorted(ss, shown, side="right"))
        self.end = pose[:3, :3] @ np.array([0.0, -shown, 0.0]) + pose[:3, 3]
        if last < 2:
            return n, z, reach
        n_out, z_out = self._sample(grid, ss[:last], ns[:last], zs[:last], n, z)
        # Высота головок кадра точнее памяти: подгонка по парам этого кадра видит и качку кузова.
        z_out[grid <= reach] = z[grid <= reach]
        return n_out, z_out, float(ss[last - 1])

    def _meas_var(self, s: np.ndarray, grid: np.ndarray, sigma: np.ndarray, reach: float, floor: float = 0.0) -> np.ndarray:
        c = self.cfg
        # Где есть пары рельсов, край дальности — не конец линии КР: вес не снижается.
        w = np.clip((max(reach, floor + c.edge) - s) / c.edge, c.edge_min, 1.0)
        return np.square(c.r_scale * np.interp(s, grid, sigma) / w)

    def _to_sensor(self, pose: np.ndarray):
        if not self.world.shape[0]:
            return np.zeros(0), np.zeros(0), np.zeros(0), self.var, self.age
        inv = np.linalg.inv(pose)
        q = self.world @ inv[:3, :3].T + inv[:3, 3]
        s = -q[:, 1]
        # Узлы идут вдоль пути: s растёт. Позади поезда и после первого излома — отбросить.
        ok = np.concatenate([[True], np.diff(s) > 0.1 * self.cfg.step])
        last = int(np.argmin(ok)) if not ok.all() else s.size
        first = int(np.searchsorted(s[:last], -self.cfg.back))
        sl = slice(first, last)
        return s[sl], q[sl, 0].copy(), q[sl, 2].copy(), self.var[sl].copy(), self.age[sl].copy()

    def _sample(self, grid, ss, ns, zs, n_frame, z_frame):
        c = self.cfg
        n_out = np.interp(grid, ss, ns)
        z_out = np.interp(grid, ss, zs)
        before = grid < ss[0]
        n_out[before] = n_frame[before]
        z_out[before] = z_frame[before]
        beyond = grid > ss[-1]
        if np.any(beyond):
            d = grid[beyond] - ss[-1]
            fit = ss > ss[-1] - c.fit_span
            x = ss[fit] - ss[-1]
            if x.size >= 5 and -x[0] > 0.5 * c.fit_span:
                _cc, b, _a = np.polyfit(x, ns[fit], 2)
                cc = float(np.clip(_cc, -0.5 * c.max_curvature, 0.5 * c.max_curvature))
                bz = float(np.polyfit(x, zs[fit], 1)[0])
            else:
                b, cc, bz = 0.0, 0.0, 0.0
            n_out[beyond] = ns[-1] + b * d + cc * d * d
            z_out[beyond] = zs[-1] + bz * d
        return n_out, z_out
