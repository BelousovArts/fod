"""Препятствия в габарите пути, ближняя зона 4…60 м.

Опора — ось фильтра рельсов и плоскость головок рельсов:

1. Точка переводится в координаты пути: `s` вдоль, `n` поперёк оси, `u` — высота
   над головками рельсов на этой дальности (подгонка по высотам найденных пар,
   так уклон и перелом профиля не дают ложной «ступеньки»).
2. Нормальный поперечный профиль полотна `u_norm(n)` — верхняя огибающая (p90)
   ближней зоны 4…25 м, медиана по последним кадрам. Рельсы, шпалы, лоток
   попадают в норму сами, без шаблона.
3. Кандидат — точка в габарите выше нормы на `margin`. Кандидаты склеиваются
   в ячейках `(s, n)`, ячейка вдоль пути растёт с дальностью.
4. Подтверждение — кластер держится в одном месте пути: положение хранится в
   абсолютной путевой координате `S = s_одометрии + s`, объект неподвижен, поезд едет.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from fod.cloud import MIN_RANGE


@dataclass
class ObstacleConfig:
    s_min: float = 4.0
    s_max: float = 60.0
    # Габарит поезда 2.1 × 3.0 м.
    half_width: float = 1.05
    height_max: float = 3.00
    # Дальше этого ошибка оси по n сравнима с расстоянием между рельсом и лотком:
    # кандидат должен быть выше головок рельсов, иначе головка сама «препятствие».
    rail_level_s: float = 30.0
    # Выше нормы полотна на `margin + margin_per_m * s` — кандидат: ошибка
    # подгонки головок и шум дальности растут с расстоянием.
    margin: float = 0.06
    margin_per_m: float = 0.002
    # Между рельсами порог выше на столько: там почти все ложные (шпалы, крепления, мусор на полотне).
    # Цена: объект ниже 10 см между рельсами не виден (0.05 — ложных в 1.3 раза больше, «min» на 20 м).
    inner_raise: float = 0.10
    inner_half: float = 0.85
    # Между рельсами верх кандидата не ниже головок на столько (на всех дальностях): ниже
    # головок предмет под поездом проходит. Таблички по центру пути — верх на 5 см ниже головок.
    inner_top: float = -0.03
    # Дальше `far_s` ошибка оси 5…20 см: края габарита задевают стены, короб КР и свод.
    # Для детекции габарит сужается и опускается на столько за метр: на 120 м ±0.69 × 2.1 м, на 150 м ±0.51 × 1.65 м.
    far_s: float = 60.0
    far_shrink_n: float = 0.006
    far_shrink_h: float = 0.015
    profile_bin: float = 0.05
    profile_s: tuple[float, float] = (4.0, 25.0)
    profile_frames: int = 60
    profile_min_frames: int = 5
    profile_q: float = 97.0
    # Норма сравнивается с максимумом по соседям ±(dilate + dilate_per_m * s):
    # у лотка и бордюра ступенька, и ошибка оси по n, растущая с дальностью,
    # иначе даёт «подъём» на всю её высоту.
    # На стоянке (за `still_frames` кадров проехали меньше `still_m`) готовая норма не
    # обновляется: иначе предмет, появившийся в полосе нормы, через ~3 с становится нормой.
    still_m: float = 0.3
    still_frames: int = 10
    profile_dilate: float = 0.10
    profile_dilate_per_m: float = 0.004
    # Норма не ниже уровня полотна между рельсами минус столько: дно лотка
    # видно только вблизи, издалека видны его стенки.
    bed_floor_drop: float = 0.03
    # Ячейка кластеризации: поперёк фиксированная, вдоль растёт с дальностью.
    cell_n: float = 0.30
    cell_s_min: float = 0.40
    cell_s_per_m: float = 0.03
    # Сколько точек нужно кластеру: на дальности объект даёт одну-две.
    min_points: tuple[tuple[float, int], ...] = ((20.0, 4), (40.0, 3), (1e9, 2))
    # Сопоставление с треками в абсолютной координате пути.
    track_ds: float = 1.2
    track_ds_per_m: float = 0.03
    track_dn: float = 0.6
    # Счёт в кадрах, а не во времени: при пропуске кадров (5 Гц) пересчёт окна во время
    # давал куб на 4 м дальше, но ложных в 1.5–2.5 раза больше.
    confirm_hits: int = 4
    confirm_window: int = 5
    drop_misses: int = 6
    head_fit_min_marks: int = 4
    # С позами (ICP) трек — точка в мире: в кадр переносится точно, и на кривой тоже.
    world_tracks: bool = True
    # Промах — только если место трека в кадре просмотрено: внутри габарита и дальности,
    # и в окне ±(seen_ds + seen_ds_per_m·s) × ±seen_dn есть хоть одна точка. Иначе трек
    # держится как есть, но не дольше hold_frames кадров подряд.
    # Выключено: ложных в 1.15 раза больше, непрерывность объекта не лучше.
    hold_unseen: bool = False
    hold_confirmed_only: bool = False
    hold_frames: int = 20
    seen_ds: float = 0.5
    seen_ds_per_m: float = 0.01
    seen_dn: float = 0.3
    # С позами: горячие точки последних `temporal_k` кадров переносятся в текущий и
    # участвуют в кластеризации; кластер — из точек не менее `temporal_min_frames` кадров.
    # Выключено: k=5 без обязательных точек текущего кадра — ящик с 60 до 76 м,
    # но ложных в 1.7 раза больше; с обязательными — дальность не растёт.
    temporal_k: int = 1
    temporal_min_frames: int = 2
    temporal_need_now: bool = True


@dataclass
class Candidate:
    s: float
    n: float
    height: float
    n_points: int
    s_len: float
    n_abs: float = float("nan")    # n в кадре сенсора (без вычета оси)
    z: float = float("nan")


@dataclass
class ObstacleTrack:
    track_id: int
    S: float                       # координата пути по `s_odom`
    n: float                       # от оси в текущем кадре
    height: float
    n_points: int
    hits: deque = field(default_factory=lambda: deque(maxlen=8))
    misses: int = 0
    confirmed: bool = False
    P: np.ndarray | None = None    # точка в мире, если есть позы
    s: float = float("nan")        # дальность в текущем кадре
    unseen: int = 0                # кадров подряд, где место трека не просмотрено

    def s_now(self, s_odom: float | None = None) -> float:
        return self.s


@dataclass
class ObstacleFrame:
    ready: bool
    status: str                    # UNKNOWN / CLEAR / OBSTACLE
    distance: float                # до ближайшего подтверждённого, м; NaN, если нет
    candidates: list[Candidate]
    tracks: list[ObstacleTrack]
    s_odom: float
    # Точки-кандидаты в координатах сенсора `(s, n, z)` — для картинки.
    cand_s: np.ndarray
    cand_n: np.ndarray
    cand_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ms: float = 0.0
    s_max: float = float("nan")    # до какой дальности смотрели в этом кадре

    @property
    def confirmed(self) -> list[ObstacleTrack]:
        return [t for t in self.tracks if t.confirmed]


class ObstacleDetector:
    def __init__(self, config: ObstacleConfig | None = None) -> None:
        self.cfg = config or ObstacleConfig()
        c = self.cfg
        self.bins = int(round(2.0 * c.half_width / c.profile_bin))
        self._profile_rows: deque = deque(maxlen=c.profile_frames)
        self.profile = np.full(self.bins, np.nan)
        self._dilated: np.ndarray | None = None
        self.head_coeff: np.ndarray | None = None
        self.head_s_max = 0.0
        self.tracks: list[ObstacleTrack] = []
        self._next_id = 1
        self._hot_world: deque = deque(maxlen=c.temporal_k)
        self._odom: deque = deque(maxlen=c.still_frames)
        self.frozen = False

    def reset(self) -> None:
        self.__init__(self.cfg)

    # --- геометрия пути ----------------------------------------------------

    def head_z(self, s: np.ndarray | float) -> np.ndarray:
        """Высота головок рельсов на дальности `s` в системе сенсора.

        За последней парой — по касательной: парабола за краем данных уходит.
        """
        assert self.head_coeff is not None
        s = np.asarray(s, dtype=np.float64)
        s_in = np.minimum(s, self.head_s_max)
        slope = np.polyval(np.polyder(self.head_coeff), self.head_s_max)
        return np.polyval(self.head_coeff, s_in) + slope * (s - s_in)

    def _fit_head(self, marks) -> None:
        c = self.cfg
        if len(marks) < c.head_fit_min_marks:
            return
        s = np.array([m.s for m in marks], dtype=np.float64)
        z = np.array([m.z for m in marks], dtype=np.float64)
        span = float(s.max() - s.min())
        deg = 2 if span > 30.0 and s.size >= 8 else 1 if span > 8.0 else 0
        keep = np.ones(s.size, dtype=bool)
        coeff = np.polyfit(s, z, deg)
        # Два прохода с отбросом: пара, закрытая объектом, врёт по высоте.
        for _ in range(2):
            resid = np.abs(z - np.polyval(coeff, s))
            keep = resid < max(0.04, 2.5 * float(np.median(resid)))
            if int(keep.sum()) < deg + 2:
                break
            coeff = np.polyfit(s[keep], z[keep], deg)
        if deg < 2:
            coeff = np.concatenate([np.zeros(2 - deg), coeff])
        self.head_coeff = coeff
        self.head_s_max = float(s[keep].max()) if np.any(keep) else float(s.max())

    def _bin(self, n_rel: np.ndarray) -> np.ndarray:
        idx = np.floor((n_rel + self.cfg.half_width) / self.cfg.profile_bin).astype(np.int64)
        return np.clip(idx, 0, self.bins - 1)

    def _update_profile(self, s: np.ndarray, n_rel: np.ndarray, u: np.ndarray) -> None:
        c = self.cfg
        sel = (s >= c.profile_s[0]) & (s < c.profile_s[1]) & (u < 0.6)
        row = np.full(self.bins, np.nan)
        if int(sel.sum()) >= 50:
            idx = self._bin(n_rel[sel])
            u_sel = u[sel]
            order = np.lexsort((u_sel, idx))
            u_sorted = u_sel[order]
            counts = np.bincount(idx, minlength=self.bins)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            pick = starts + np.floor((counts - 1) * c.profile_q / 100.0).astype(np.int64)
            ok = counts >= 4
            row[ok] = u_sorted[pick[ok]]
        self._profile_rows.append(row)
        rows = np.stack(self._profile_rows)
        enough = np.sum(np.isfinite(rows), axis=0) >= min(c.profile_min_frames, len(self._profile_rows))
        prof = np.full(self.bins, np.nan)
        if np.any(enough):
            prof[enough] = np.nanmedian(rows[:, enough], axis=0)
        # Дыры в профиле — соседним значением, чтобы не терять кандидатов на краю.
        good = np.isfinite(prof)
        if np.any(good) and not np.all(good):
            x = np.arange(self.bins)
            prof[~good] = np.interp(x[~good], x[good], prof[good])
        self.profile = prof
        self._dilated = None
        if np.all(np.isfinite(prof)):
            centers = (np.arange(self.bins) + 0.5) * c.profile_bin - c.half_width
            inner = np.abs(centers) < 0.6
            floor = float(np.median(prof[inner])) - c.bed_floor_drop if np.any(inner) else -np.inf
            base = np.maximum(prof, floor)
            k_max = int(np.ceil((c.profile_dilate + c.profile_dilate_per_m * c.s_max) / c.profile_bin))
            self._dilated = np.stack(
                [ndimage.maximum_filter1d(base, size=2 * k + 1, mode="nearest") for k in range(k_max + 1)]
            )

    def _base(self, s: np.ndarray, n_rel: np.ndarray) -> np.ndarray:
        """Норма полотна для точек: расширенный профиль, ширина окна растёт с дальностью."""
        c = self.cfg
        k = np.round((c.profile_dilate + c.profile_dilate_per_m * s) / c.profile_bin).astype(np.int64)
        k = np.clip(k, 0, self._dilated.shape[0] - 1)
        base = self._dilated[k, self._bin(n_rel)]
        return np.where(s > c.rail_level_s, np.maximum(base, 0.0), base)

    def _s_cell(self, s: np.ndarray) -> np.ndarray:
        c = self.cfg
        s0 = c.cell_s_min / c.cell_s_per_m
        g = np.where(s < s0, s / c.cell_s_min, s0 / c.cell_s_min + np.log(np.maximum(s, 1e-3) / s0) / c.cell_s_per_m)
        return np.floor(g).astype(np.int64)

    def _min_points(self, s: float) -> int:
        for s_hi, count in self.cfg.min_points:
            if s < s_hi:
                return count
        return 1

    # --- кадр --------------------------------------------------------------

    def step(self, xyz: np.ndarray, rail_frame, axis, s_odom: float, pose: np.ndarray | None = None) -> ObstacleFrame:
        """`rail_frame` — RailFrame кадра, `axis` — запертый фильтр оси (`n`, `psi`) или None.

        У оси могут быть `reach` (до какой дальности она известна, смотрим не дальше)
        и `head(s)` — высота головок, если своя подгонка по парам не дотягивает.
        `pose` — сенсор→мир (онлайн-ICP): треки ведутся точками в мире,
        горячие точки прошлых кадров переносятся в текущий.
        """
        import time

        started = time.perf_counter()
        c = self.cfg
        empty = np.zeros(0)
        locked = axis is not None and getattr(axis, "locked", False)
        if locked:
            self._fit_head(rail_frame.marks)
        if not locked or self.head_coeff is None:
            self._hot_world.clear()
            for t in self.tracks:
                t.s = t.S - s_odom
            return ObstacleFrame(False, "UNKNOWN", float("nan"), [], self.tracks, s_odom, empty, empty)

        s_hi = max(min(c.s_max, float(getattr(axis, "reach", c.s_max))), c.s_min + 1.0)
        xyz = np.asarray(xyz)
        # Грубый отбор по дальности до перевода в float64 — с запасом, точная проверка ниже та же.
        s_raw = -xyz[:, 1]
        xyz = xyz[(s_raw > c.s_min - 0.5) & (s_raw < s_hi + 0.5)].astype(np.float64)
        s = -xyz[:, 1]
        n = xyz[:, 0]
        keep = (s > c.s_min) & (s < s_hi) & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
        s, n, z = s[keep], n[keep], xyz[keep, 2]
        s_axis = np.linspace(c.s_min, s_hi, 1 + int(np.ceil(s_hi - c.s_min)))
        n_axis = np.asarray(axis.n(s_axis), dtype=np.float64)
        slope = np.gradient(n_axis, s_axis)
        n_rel = (n - np.interp(s, s_axis, n_axis)) * np.cos(np.arctan(np.interp(s, s_axis, slope)))
        head = getattr(axis, "head", None) or self.head_z
        u = z - head(s)
        inside = (np.abs(n_rel) < c.half_width) & (u < c.height_max)
        s, n_rel, u, z = s[inside], n_rel[inside], u[inside], z[inside]
        self._odom.append(float(s_odom))
        still = len(self._odom) == self._odom.maxlen and abs(self._odom[-1] - self._odom[0]) < c.still_m
        self.frozen = still and self._dilated is not None and len(self._profile_rows) >= c.profile_min_frames
        if not self.frozen:
            self._update_profile(s, n_rel, u)
        beyond = np.maximum(s - c.far_s, 0.0)
        core = (np.abs(n_rel) < c.half_width - c.far_shrink_n * beyond) & (u < c.height_max - c.far_shrink_h * beyond)
        s, n_rel, u, z = s[core], n_rel[core], u[core], z[core]
        ready = len(self._profile_rows) >= c.profile_min_frames and self._dilated is not None
        candidates: list[Candidate] = []
        cand_s = cand_n = cand_z = empty
        if ready:
            excess = u - self._base(s, n_rel)
            hot = (excess > self._margin(s, n_rel)) & self._above_inner(n_rel, u)
            cand_s, cand_n, cand_u = s[hot], n_rel[hot], excess[hot]
            cand_z = z[hot]
            n_abs = cand_n + np.interp(cand_s, s_axis, n_axis)
            age = None
            if pose is not None and c.temporal_k > 1:
                cand_s, cand_n, cand_u, n_abs, cand_z, age = self._temporal(
                    pose, s_axis, n_axis, slope, head, s_hi,
                    cand_s, cand_n, cand_u, n_abs, cand_z,
                )
            else:
                self._hot_world.clear()
            candidates = self._cluster(cand_s, cand_n, cand_u, n_abs, cand_z, age)
            cand_n = n_abs
        else:
            self._hot_world.clear()
        pose = pose if c.world_tracks else None
        self._track(candidates, s_odom, ready, pose, axis, (s, n_rel), s_hi)
        confirmed = [t for t in self.tracks if t.confirmed and c.s_min * 0.5 < t.s < c.s_max]
        if not ready:
            status, distance = "UNKNOWN", float("nan")
        elif confirmed:
            status, distance = "OBSTACLE", float(min(t.s for t in confirmed))
        else:
            status, distance = "CLEAR", float("nan")
        return ObstacleFrame(
            ready=ready,
            status=status,
            distance=distance,
            candidates=candidates,
            tracks=list(self.tracks),
            s_odom=float(s_odom),
            cand_s=cand_s,
            cand_n=cand_n,
            cand_z=cand_z,
            ms=(time.perf_counter() - started) * 1e3,
            s_max=s_hi,
        )

    def _margin(self, s: np.ndarray, n_rel: np.ndarray) -> np.ndarray:
        c = self.cfg
        return c.margin + c.margin_per_m * s + np.where(np.abs(n_rel) < c.inner_half, c.inner_raise, 0.0)

    def _above_inner(self, n_rel: np.ndarray, u: np.ndarray) -> np.ndarray:
        return (np.abs(n_rel) >= self.cfg.inner_half) | (u > self.cfg.inner_top)

    def _n_rel(self, s: np.ndarray, n: np.ndarray, s_axis: np.ndarray, n_axis: np.ndarray, slope: np.ndarray) -> np.ndarray:
        return (n - np.interp(s, s_axis, n_axis)) * np.cos(np.arctan(np.interp(s, s_axis, slope)))

    def _temporal(
        self,
        pose: np.ndarray,
        s_axis: np.ndarray,
        n_axis: np.ndarray,
        slope: np.ndarray,
        head,
        s_hi: float,
        s: np.ndarray,
        n_rel: np.ndarray,
        excess: np.ndarray,
        n_abs: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Горячие точки прошлых кадров → текущий; текущие кладём в буфер мира.

        Возвращает точки и их возраст в кадрах (0 — текущий).
        """
        c = self.cfg
        inv = np.linalg.inv(pose)
        now_age = np.zeros(s.size, np.int64)
        extra, ages = [], []
        for k, W in enumerate(self._hot_world):
            if W.size == 0:
                continue
            extra.append(W @ inv[:3, :3].T + inv[:3, 3])
            ages.append(np.full(W.shape[0], len(self._hot_world) - k, np.int64))
        now = np.column_stack([n_abs, -s, z]) if s.size else np.zeros((0, 3))
        self._hot_world.append((now @ pose[:3, :3].T + pose[:3, 3]).astype(np.float64) if now.size else now)
        if not extra:
            return s, n_rel, excess, n_abs, z, now_age
        q = np.concatenate(extra)
        a_e = np.concatenate(ages)
        s_e, n_e, z_e = -q[:, 1], q[:, 0], q[:, 2]
        keep = (s_e > c.s_min) & (s_e < s_hi)
        s_e, n_e, z_e, a_e = s_e[keep], n_e[keep], z_e[keep], a_e[keep]
        n_rel_e = self._n_rel(s_e, n_e, s_axis, n_axis, slope)
        u_e = z_e - head(s_e) if s_e.size else s_e
        beyond = np.maximum(s_e - c.far_s, 0.0)
        core = (np.abs(n_rel_e) < c.half_width - c.far_shrink_n * beyond) & (u_e < c.height_max - c.far_shrink_h * beyond)
        s_e, n_e, n_rel_e, u_e, z_e, a_e = s_e[core], n_e[core], n_rel_e[core], u_e[core], z_e[core], a_e[core]
        if s_e.size == 0:
            return s, n_rel, excess, n_abs, z, now_age
        excess_e = u_e - self._base(s_e, n_rel_e)
        still = (excess_e > self._margin(s_e, n_rel_e)) & self._above_inner(n_rel_e, u_e)
        return (
            np.concatenate([s, s_e[still]]),
            np.concatenate([n_rel, n_rel_e[still]]),
            np.concatenate([excess, excess_e[still]]),
            np.concatenate([n_abs, n_e[still]]),
            np.concatenate([z, z_e[still]]),
            np.concatenate([now_age, a_e[still]]),
        )

    def _cluster(
        self, s: np.ndarray, n_rel: np.ndarray, excess: np.ndarray, n_abs: np.ndarray, z: np.ndarray,
        age: np.ndarray | None = None,
    ) -> list[Candidate]:
        """`age` — возраст точек в кадрах: кластер нужен с точками текущего кадра
        и хотя бы из `temporal_min_frames` разных кадров (шум в одном месте не повторяется)."""
        c = self.cfg
        if s.size == 0:
            return []
        multi = age is not None and bool(np.any(age > 0))
        i_s = self._s_cell(s)
        i_n = np.floor((n_rel + c.half_width) / c.cell_n).astype(np.int64)
        i_s0 = int(i_s.min())
        grid = np.zeros((int(i_s.max()) - i_s0 + 1, int(i_n.max()) + 1), dtype=bool)
        grid[i_s - i_s0, i_n] = True
        labels, count = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
        point_label = labels[i_s - i_s0, i_n]
        out: list[Candidate] = []
        for lab in range(1, count + 1):
            sel = point_label == lab
            k = int(sel.sum())
            s_mid = float(np.median(s[sel]))
            if k < self._min_points(s_mid):
                continue
            if multi:
                a = age[sel]
                if (c.temporal_need_now and not np.any(a == 0)) or np.unique(a).size < c.temporal_min_frames:
                    continue
            out.append(
                Candidate(
                    s=float(s[sel].min()),
                    n=float(np.median(n_rel[sel])),
                    height=float(excess[sel].max()),
                    n_points=k,
                    s_len=float(s[sel].max() - s[sel].min()),
                    n_abs=float(np.median(n_abs[sel])),
                    z=float(np.median(z[sel])),
                )
            )
        return out

    def _predict(self, track: ObstacleTrack, s_odom: float, inv: np.ndarray | None, axis) -> None:
        """Дальность и n трека в текущем кадре: по точке в мире или по координате пути."""
        if inv is not None and track.P is not None:
            q = inv[:3, :3] @ track.P + inv[:3, 3]
            track.s = float(-q[1])
            track.n = float(q[0] - axis.n(max(track.s, 0.0)))
        else:
            track.s = track.S - s_odom

    def _seen(self, track: ObstacleTrack, seen: tuple[np.ndarray, np.ndarray], s_hi: float) -> bool:
        c = self.cfg
        s_t = track.s
        if not (c.s_min < s_t < s_hi):
            return False
        beyond = max(s_t - c.far_s, 0.0)
        if abs(track.n) > c.half_width - c.far_shrink_n * beyond:
            return False
        s, n_rel = seen
        ds = c.seen_ds + c.seen_ds_per_m * s_t
        return bool(np.any((np.abs(s - s_t) < ds) & (np.abs(n_rel - track.n) < c.seen_dn)))

    def _track(
        self, candidates: list[Candidate], s_odom: float, ready: bool, pose, axis, seen, s_hi: float
    ) -> None:
        c = self.cfg
        inv = np.linalg.inv(pose) if pose is not None else None
        def world(cand: Candidate) -> np.ndarray | None:
            if pose is None:
                return None
            return pose[:3, :3] @ np.array([cand.n_abs, -cand.s, cand.z]) + pose[:3, 3]

        used = np.zeros(len(candidates), dtype=bool)
        for track in self.tracks:
            self._predict(track, s_odom, inv, axis)
            s_now = track.s
            best, best_d = -1, np.inf
            for j, cand in enumerate(candidates):
                if used[j]:
                    continue
                ds = abs(cand.s - s_now)
                dn = abs(cand.n - track.n)
                if ds < c.track_ds + c.track_ds_per_m * s_now and dn < c.track_dn:
                    d = ds + dn
                    if d < best_d:
                        best, best_d = j, d
            if best >= 0:
                cand = candidates[best]
                used[best] = True
                track.S = 0.5 * (track.S + s_odom + cand.s)
                track.s = 0.5 * (track.s + cand.s)
                track.n = 0.5 * (track.n + cand.n)
                p = world(cand)
                if p is not None:
                    track.P = p if track.P is None else 0.5 * (track.P + p)
                track.height = max(track.height, cand.height)
                track.n_points = cand.n_points
                track.hits.append(1)
                track.misses = 0
                track.unseen = 0
            elif ready:
                hold = c.hold_unseen and (track.confirmed or not c.hold_confirmed_only)
                if hold and not self._seen(track, seen, s_hi):
                    track.unseen += 1
                else:
                    track.hits.append(0)
                    track.misses += 1
                    track.unseen = 0
            window = list(track.hits)[-c.confirm_window :]
            if sum(window) >= c.confirm_hits:
                track.confirmed = True
        for j, cand in enumerate(candidates):
            if used[j]:
                continue
            track = ObstacleTrack(self._next_id, s_odom + cand.s, cand.n, cand.height, cand.n_points, P=world(cand), s=cand.s)
            track.hits.append(1)
            self._next_id += 1
            self.tracks.append(track)
        self.tracks = [
            t for t in self.tracks if t.misses < c.drop_misses and t.unseen <= c.hold_frames and t.s > -2.0
        ]
