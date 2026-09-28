"""Продольная одометрия: развёртка тоннеля + фазовая корреляция.

PLAN, раздел 5.2 / задача 1.1. Point-to-plane ICP в однородном тоннеле
вырожден вдоль пути и на бэгах срывает знак скорости при fitness ≈ 1.
Полная 6-DoF здесь не нужна: для deskew рельсов, накопителя и TTC достаточно
продольной скорости.

Основной канал — текстура стен в координатах `(s, θ)`: кабели, стыки колец,
кронштейны, светильники. Сдвиг по `s` между кадрами — 1D/2D фазовая
корреляция, затем фильтр с моделью «только вперёд или стоп» и ограничением
ускорения. Измерено на хакатонных бэгах: стоячий `doubleT_obstacle` даёт
≈ 0 км/ч, едущие — гладкие 44…56 км/ч без срыва знака после фильтра.

Стены фиксируют ещё roll/pitch/yaw и боковое положение, но это уже делает
трекер рельсов (`fod.track_filter`). Здесь только `s` и `v`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fod.cloud import MIN_RANGE, estimate_floor_z

# Вперёд = −Y, поперёк = X, вверх = Z. s = −y.

MIN_YAW_TRAVEL = 0.30   # м за кадр; ниже кривизна из поворота не считается


@dataclass
class OdomConfig:
    s_min: float = 8.0
    s_max: float = 48.0
    ds: float = 0.20
    n_theta: int = 48
    u_lo: float = 0.40
    u_hi: float = 5.60
    n_half: float = 12.0
    z_center_off: float = 2.40

    v_max: float = 28.0          # м/с ≈ 100 км/ч, выше заявленных 85
    a_max: float = 1.3           # м/с², служебное торможение метро
    meas_sigma: float = 0.80     # м/с, шум сырой корреляции

    # Сопоставление не только с предыдущим кадром. За кадр поезд проезжает
    # 1.2…1.7 м, а шум одного замера — 0.09…0.32 м, то есть 6…20% базы. На базе
    # в 5 кадров та же ошибка в метрах даёт уже 2.4% (scripts/check_odom_scale.py),
    # и путь перестаёт копить эту разницу. Цена — ещё две корреляции на кадр.
    multi_lags: tuple[int, ...] = (3, 6)
    multi_sigma_m: float = 0.10      # м, шум одного сопоставления пары развёрток
    multi_accel_sigma: float = 0.30  # м/с², неизвестность ускорения внутри базы
    multi_prominence: float = 0.05   # на длинной базе кольца обделки врут охотнее
    multi_max_dt: float = 1.2        # с, дальше стена успевает уйти из развёртки
    gate_sigma: float = 4.0      # отсев измерения, если дальше N σ от прогноза
    min_prominence: float = 0.025
    agree_m: float = 0.18        # 1D и 2D «согласны», если |Δs| меньше этого
    lock_frames: int = 4
    gap_s: float = 0.55          # разрыв записи → сброс развёртки
    v_reverse_tol: float = 0.35  # м/с, ниже считаем нулём, не задним ходом
    process_a: float = 0.45      # белый шум ускорения для Q фильтра

    # База профиля стен. Коротая база не отделяет поворот от бокового сноса
    # кузова на рессорах: на 10…45 м кривизна из поворота выходила 0.65 от
    # измеренной по рельсам, на 12…95 м — 0.92…1.10, то есть совпадает.
    yaw_s_min: float = 12.0
    yaw_s_max: float = 95.0
    yaw_ds: float = 1.5
    yaw_max: float = 0.06        # рад/кадр; 0.06 при 2 м/кадр это R = 33 м
    yaw_min_points: int = 14


@dataclass
class Unroll:
    """Развёртка стены: строки = θ, столбцы = s, значение = интенсивность."""

    intensity: np.ndarray
    s_min: float
    ds: float
    fill: float
    wall: "WallProfile | None" = None

    @property
    def n_s(self) -> int:
        return int(self.intensity.shape[1])


@dataclass
class WallProfile:
    """Боковое положение стен по дальности: `s` → левая и правая кромки."""

    s: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass
class MotionEstimate:
    s: float
    v: float
    dt: float
    ds_raw: float
    v_raw: float
    confidence: float
    source: str
    locked: bool
    accepted: bool
    fill: float
    ms: float
    n_multi: int = 0                # сколько длинных баз подтвердили скорость
    ds_multi: float = float("nan")  # путь по самой длинной принятой базе, м
    span_multi: float = float("nan")  # длина этой базы по накопленному `s`, м
    dyaw: float = float("nan")      # поворот за кадр, рад (вправо > 0)
    dyaw_sigma: float = float("nan")
    dlat: float = float("nan")      # боковой снос за кадр, м
    yaw_ok: bool = False
    unroll: Unroll | None = None

    @property
    def kmh(self) -> float:
        return float(self.v * 3.6)

    @property
    def travelled(self) -> float:
        return float(max(self.v * self.dt, 0.0))

    def curvature(self) -> tuple[float, float]:
        """Кривизна пути под поездом и её σ: поезд стоит на рельсах, значит κ = dψ/ds.

        На малом ходу делить на путь нельзя — шум поворота взрывается, поэтому
        ниже порога канал молчит, а не выдаёт случайное число.
        """
        if not self.yaw_ok or self.travelled < MIN_YAW_TRAVEL:
            return float("nan"), float("nan")
        return float(self.dyaw / self.travelled), float(self.dyaw_sigma / self.travelled)


@dataclass
class EgoMotion:
    """Кадр за кадром держит `s` и `v` и не даёт знаку скорости срываться."""

    def __init__(self, config: OdomConfig | None = None) -> None:
        self.cfg = config or OdomConfig()
        self.reset()

    def reset(self) -> None:
        self.s = 0.0
        self.v = 0.0
        self.a = 0.0
        self.yaw = 0.0
        self.P = np.diag([1.0, 20.0 ** 2])
        self.locked = False
        self._prev_t: float | None = None
        self._buf: list[float] = []
        # История развёрток для длинных баз: (развёртка, время, путь на тот момент).
        self._hist: list[tuple[Unroll, float, float]] = []
        self.frames = 0

    @property
    def _prev(self) -> "Unroll | None":
        return self._hist[-1][0] if self._hist else None

    def step(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        t: float,
        z_floor: float | None = None,
    ) -> MotionEstimate:
        started = time.perf_counter()
        cfg = self.cfg
        # Развёртка, профиль стен и пол смотрят только вперёд: точки заранее, втрое быстрее, тот же ответ.
        s32 = -xyz[:, 1]
        ahead = (s32 > min(4.0, cfg.s_min, cfg.yaw_s_min) - 0.1) & (s32 < max(40.0, cfg.s_max, cfg.yaw_s_max) + 0.1)
        ahead &= np.abs(xyz[:, 0]) < max(2.5, cfg.n_half) + 0.1
        xyz_a = xyz[ahead]
        if z_floor is None:
            z_floor = estimate_floor_z(xyz, xyz_a)
        unroll = unroll_tunnel(xyz_a, intensity[ahead], z_floor, cfg)

        dt = 0.1
        ds_raw = float("nan")
        v_raw = float("nan")
        conf = 0.0
        source = "none"
        accepted = False

        if self._prev_t is not None:
            dt = float(t - self._prev_t)
            if dt < 0.02:
                dt = 0.02
            if dt > cfg.gap_s:
                self._hist = []
                self._buf = []
                self.locked = False
                self.P[1, 1] = max(float(self.P[1, 1]), 12.0 ** 2)

        dyaw = float("nan")
        dyaw_sigma = float("nan")
        dlat = float("nan")
        yaw_ok = False

        n_multi = 0
        ds_multi = float("nan")
        span_multi = float("nan")

        if self._prev is not None and dt <= cfg.gap_s:
            v_before = self.v
            self._predict(dt)
            ds_raw, conf, source = measure_ds(unroll, self._prev, self.v * dt, dt, cfg)
            if np.isfinite(ds_raw):
                v_raw = ds_raw / dt
                accepted = self._update(ds_raw, dt, conf, source)
            else:
                v_raw = float("nan")
            n_multi, ds_multi, span_multi = self._update_multi(unroll, t)
            # Ускорение — для пересчёта средней скорости базы в мгновенную.
            self.a += 0.4 * ((self.v - v_before) / max(dt, 1e-3) - self.a)
            self.a = float(np.clip(self.a, -cfg.a_max, cfg.a_max))
            if unroll.wall is not None and self._prev.wall is not None:
                dyaw, dyaw_sigma, dlat, used = measure_yaw(unroll.wall, self._prev.wall, self.v * dt, cfg)
                yaw_ok = np.isfinite(dyaw) and np.isfinite(dyaw_sigma) and used >= cfg.yaw_min_points
                if yaw_ok and self.v * dt >= MIN_YAW_TRAVEL:
                    self.yaw += dyaw

        self.s = max(0.0, float(self.s))
        self.v = float(np.clip(self.v, 0.0, cfg.v_max))
        self._hist.append((unroll, float(t), float(self.s)))
        keep = max(cfg.multi_lags) + 1 if cfg.multi_lags else 1
        if len(self._hist) > keep:
            self._hist = self._hist[-keep:]
        self._prev_t = float(t)
        self.frames += 1

        return MotionEstimate(
            s=float(self.s),
            v=float(self.v),
            dt=float(dt),
            ds_raw=float(ds_raw) if np.isfinite(ds_raw) else float("nan"),
            v_raw=float(v_raw) if np.isfinite(v_raw) else float("nan"),
            confidence=float(conf),
            source=source,
            locked=self.locked,
            accepted=accepted,
            fill=float(unroll.fill),
            ms=(time.perf_counter() - started) * 1e3,
            n_multi=int(n_multi),
            ds_multi=float(ds_multi),
            span_multi=float(span_multi),
            dyaw=float(dyaw),
            dyaw_sigma=float(dyaw_sigma),
            dlat=float(dlat),
            yaw_ok=bool(yaw_ok),
            unroll=unroll,
        )

    def _update_multi(self, unroll: Unroll, t: float) -> tuple[int, float, float]:
        """Сопоставить кадр с более старыми и влить это в скорость.

        Ошибка одного сопоставления почти не зависит от базы — это шум пика
        корреляции, десяток сантиметров. Значит, на базе в пять кадров та же
        ошибка весит впятеро меньше, а накопленный путь перестаёт её копить.
        Средняя скорость базы приводится к мгновенной через текущее ускорение,
        и неизвестность этого ускорения честно входит в σ измерения.
        """
        cfg = self.cfg
        if not cfg.multi_lags or len(self._hist) < 2:
            return 0, float("nan"), float("nan")
        used = 0
        best_ds = float("nan")
        best_span = float("nan")
        sigma_v = float(np.sqrt(max(self.P[1, 1], 1e-6)))
        for lag in sorted(cfg.multi_lags):
            if len(self._hist) < lag:
                continue
            past, past_t, past_s = self._hist[-lag]
            span = float(t - past_t)
            if not (0.0 < span <= cfg.multi_max_dt):
                continue
            if past.intensity.shape != unroll.intensity.shape:
                continue
            expect = self.s - past_s
            if expect < 0.3:
                continue
            # Окно поиска — вокруг ожидания: свободный поиск на длинной базе
            # цепляется за кольца обделки, они повторяются каждый метр.
            half = max(0.60, 3.0 * sigma_v * span + 0.5 * cfg.a_max * span * span)
            ds, conf, source = measure_ds(unroll, past, expect, span, cfg, search=half)
            if not np.isfinite(ds) or source == "none" or conf < cfg.multi_prominence:
                continue
            v_meas = ds / span + 0.5 * self.a * span
            sigma = float(
                np.hypot(
                    cfg.multi_sigma_m / max(conf, 0.15) / span,
                    cfg.multi_accel_sigma * span * 0.5,
                )
            )
            if self._update_velocity(v_meas, sigma):
                used += 1
                best_ds, best_span = float(ds), float(expect)
        return used, best_ds, best_span

    def _predict(self, dt: float) -> None:
        dt = max(dt, 1e-3)
        f = np.array([[1.0, dt], [0.0, 1.0]])
        qa = self.cfg.process_a ** 2
        q = qa * np.array(
            [
                [dt ** 4 / 4.0, dt ** 3 / 2.0],
                [dt ** 3 / 2.0, dt ** 2],
            ]
        )
        self.s = self.s + self.v * dt
        x = np.array([self.s, self.v])
        # s already updated; keep v
        self.P = f @ self.P @ f.T + q
        self.s = float(x[0])
        self.v = float(x[1])

    def _update(self, ds: float, dt: float, conf: float, source: str) -> bool:
        cfg = self.cfg
        sigma = cfg.meas_sigma / max(conf, 0.15)
        if source == "both":
            sigma *= 0.70
        return self._update_velocity(ds / dt, sigma)

    def _update_velocity(self, v_meas: float, sigma: float) -> bool:
        cfg = self.cfg
        if not np.isfinite(v_meas) or abs(v_meas) > cfg.v_max * 1.15:
            return False
        if v_meas < -cfg.v_reverse_tol:
            # Задний ход в данных не ожидается; это срыв корреляции.
            if self.locked and self.v > 1.0:
                return False
            if not self.locked:
                v_meas = 0.0

        var = float(self.P[1, 1])
        gate = cfg.gate_sigma * np.sqrt(max(var, sigma * sigma))
        if not self.locked:
            gate = max(gate, cfg.v_max)
        if abs(v_meas - self.v) > gate:
            return False

        # Измерение скорости, s уже предсказан.
        h = np.array([0.0, 1.0])
        s_innov = float(h @ self.P @ h) + sigma * sigma
        innov = v_meas - self.v
        gain = (self.P @ h) / s_innov
        x = np.array([self.s, self.v]) + gain * innov
        self.P = self.P - np.outer(gain, h @ self.P)
        self.P = 0.5 * (self.P + self.P.T)
        self.s = float(x[0])
        self.v = float(np.clip(x[1], 0.0, cfg.v_max))

        self._buf.append(self.v)
        if len(self._buf) > 12:
            self._buf = self._buf[-12:]
        if not self.locked and len(self._buf) >= cfg.lock_frames:
            recent = np.asarray(self._buf[-cfg.lock_frames :])
            if float(np.std(recent)) < 2.5:
                self.locked = True
        elif self.locked:
            self.locked = True
        return True


def unroll_tunnel(
    xyz: np.ndarray,
    intensity: np.ndarray,
    z_floor: float,
    config: OdomConfig | None = None,
) -> Unroll:
    """Стена тоннеля → картинка `(θ, s)`. Пол отсекается по высоте над УГР."""
    cfg = config or OdomConfig()
    n_s = int(round((cfg.s_max - cfg.s_min) / cfg.ds))
    image = np.full((cfg.n_theta, n_s), np.nan, dtype=np.float64)
    if xyz.size == 0:
        return Unroll(image, cfg.s_min, cfg.ds, 0.0, None)

    s = -xyz[:, 1].astype(np.float64, copy=False)
    n = xyz[:, 0].astype(np.float64, copy=False)
    z = xyz[:, 2].astype(np.float64, copy=False)
    u = z - z_floor
    rng2 = n * n + xyz[:, 1].astype(np.float64) ** 2 + z * z
    keep = (
        (s >= cfg.s_min)
        & (s < cfg.s_max)
        & (u > cfg.u_lo)
        & (u < cfg.u_hi)
        & (np.abs(n) < cfg.n_half)
        & (rng2 > MIN_RANGE * MIN_RANGE)
    )
    if int(keep.sum()) < 200:
        return Unroll(image, cfg.s_min, cfg.ds, 0.0, wall_profile(xyz, z_floor, cfg))

    n_k = n[keep]
    z_k = z[keep]
    wall = keep & (u > 0.90) & (np.abs(n) > 1.10)
    if int(wall.sum()) > 150:
        z_c = float(np.median(z[wall]))
    else:
        z_c = z_floor + cfg.z_center_off

    theta = np.arctan2(n_k, z_k - z_c)
    s_k = s[keep]
    inten = intensity[keep].astype(np.float64, copy=False)

    s_idx = np.clip(((s_k - cfg.s_min) / cfg.ds).astype(np.int64), 0, n_s - 1)
    th_idx = np.clip(((theta + np.pi) / (2.0 * np.pi) * cfg.n_theta).astype(np.int64), 0, cfg.n_theta - 1)
    lin = th_idx * n_s + s_idx
    n_bins = cfg.n_theta * n_s
    cnt = np.bincount(lin, minlength=n_bins).astype(np.float64)
    total = np.bincount(lin, weights=inten, minlength=n_bins)
    ok = cnt > 0
    flat = np.full(n_bins, np.nan, dtype=np.float64)
    flat[ok] = total[ok] / cnt[ok]
    image = flat.reshape(cfg.n_theta, n_s)
    fill = float(ok.mean())
    return Unroll(image, cfg.s_min, cfg.ds, fill, wall_profile(xyz, z_floor, cfg))


def wall_profile(
    xyz: np.ndarray,
    z_floor: float,
    config: OdomConfig | None = None,
) -> WallProfile:
    """Кромки сечения по полосам дальности — опора для поворота и сноса."""
    cfg = config or OdomConfig()
    edges = np.arange(cfg.yaw_s_min, cfg.yaw_s_max + 1e-6, cfg.yaw_ds)
    centers = 0.5 * (edges[:-1] + edges[1:])
    left = np.full(centers.size, np.nan)
    right = np.full(centers.size, np.nan)

    s = -xyz[:, 1].astype(np.float64, copy=False)
    n = xyz[:, 0].astype(np.float64, copy=False)
    u = xyz[:, 2].astype(np.float64, copy=False) - z_floor
    keep = (s >= edges[0]) & (s < edges[-1]) & (u > 0.50) & (u < 3.20) & (np.abs(n) < cfg.n_half)
    keep &= np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE
    if int(keep.sum()) < 200:
        return WallProfile(centers, left, right)

    idx = np.clip(((s[keep] - edges[0]) / cfg.yaw_ds).astype(np.int64), 0, centers.size - 1)
    n_k = n[keep]
    order = np.argsort(idx, kind="stable")
    idx, n_k = idx[order], n_k[order]
    bounds = np.searchsorted(idx, np.arange(centers.size + 1))
    for b in range(centers.size):
        lo, hi = bounds[b], bounds[b + 1]
        if hi - lo < 25:
            continue
        chunk = np.sort(n_k[lo:hi])
        left[b] = float(chunk[int(0.03 * chunk.size)])
        right[b] = float(chunk[int(0.97 * (chunk.size - 1))])
    return WallProfile(centers, left, right)


def measure_yaw(
    current: WallProfile,
    previous: WallProfile,
    ds: float,
    config: OdomConfig | None = None,
) -> tuple[float, float, float, int]:
    """Поворот, его σ и боковой снос из совмещения профилей стен.

    Проехав `ds` и повернув на `dψ`, стенка из прошлого кадра встаёт в
    `n'(s) = n_prev(s + ds) − dψ·s + dn`. Продольная координата уже известна из
    фазовой корреляции, поэтому остаются две хорошо наблюдаемые величины —
    именно те, что вырождены у ICP вдоль тоннеля.
    """
    cfg = config or OdomConfig()
    s = current.s
    rows_s: list[np.ndarray] = []
    rows_r: list[np.ndarray] = []
    for cur, prev in ((current.left, previous.left), (current.right, previous.right)):
        ok_prev = np.isfinite(prev)
        if int(ok_prev.sum()) < 6:
            continue
        shifted = np.interp(s + ds, previous.s[ok_prev], prev[ok_prev], left=np.nan, right=np.nan)
        residual = cur - shifted
        good = np.isfinite(residual)
        if int(good.sum()) < 6:
            continue
        rows_s.append(s[good])
        rows_r.append(residual[good])
    if not rows_s:
        return float("nan"), float("nan"), float("nan"), 0

    s_all = np.concatenate(rows_s)
    r_all = np.concatenate(rows_r)

    def fit(s: np.ndarray, r: np.ndarray):
        design = np.stack([np.ones_like(s), -s], axis=1)
        solution, *_ = np.linalg.lstsq(design, r, rcond=None)
        return design, solution, r - design @ solution

    design, solution, resid = fit(s_all, r_all)
    if r_all.size > 8:
        keep = np.abs(resid) <= max(float(np.percentile(np.abs(resid), 80)), 0.02)
        if int(keep.sum()) >= 8:
            s_all, r_all = s_all[keep], r_all[keep]
            design, solution, resid = fit(s_all, r_all)

    dlat, dyaw = float(solution[0]), float(solution[1])
    if abs(dyaw) > cfg.yaw_max or abs(dlat) > 0.60:
        return float("nan"), float("nan"), float("nan"), int(s_all.size)
    # σ наклона из ковариации подгонки: честная цена шума профиля стен.
    var = float(resid @ resid) / max(r_all.size - 2, 1)
    try:
        cov = var * np.linalg.inv(design.T @ design)
        sigma = float(np.sqrt(max(cov[1, 1], 1e-12)))
    except np.linalg.LinAlgError:
        sigma = float("nan")
    return dyaw, sigma, dlat, int(s_all.size)


def measure_ds(
    current: Unroll,
    previous: Unroll,
    ds_prior: float,
    dt: float,
    config: OdomConfig | None = None,
    search: float | None = None,
) -> tuple[float, float, str]:
    """Сдвиг текущего кадра относительно предыдущего, вперёд > 0.

    `search` — полуокно поиска вокруг `ds_prior`; по умолчанию считается из
    предела скорости и ускорения. Задавать его руками нужно на длинных базах,
    где свободный поиск уходит на соседнее кольцо обделки.
    """
    cfg = config or OdomConfig()
    if current.intensity.shape != previous.intensity.shape:
        return float("nan"), 0.0, "none"
    if current.fill < 0.04 or previous.fill < 0.04:
        return float("nan"), 0.0, "none"

    if search is not None:
        window = float(search)
        lo, hi = ds_prior - window, ds_prior + window
        search = min(max(abs(lo), abs(hi)), cfg.s_max - cfg.s_min - 1.0)
    else:
        lo, hi = -np.inf, np.inf
        search = min(cfg.v_max * dt + 0.35, cfg.s_max - cfg.s_min - 1.0)
        if np.isfinite(ds_prior) and abs(ds_prior) > 0.05:
            search = min(search, abs(ds_prior) + max(0.45, 6.0 * cfg.a_max * dt))

    # Сырой сдвиг обратен по знаку движению, поэтому окно переворачивается.
    window = None if not np.isfinite(lo) else (-hi, -lo)
    ds1, p1 = _corr_1d(current.intensity, previous.intensity, current.ds, search, window)
    ds2, p2 = _corr_2d(current.intensity, previous.intensity, current.ds, search, window)

    # FFT даёт сдвиг картинки previous относительно current. Поезд вперёд →
    # та же шпала ближе → current(s) ≈ previous(s + Δs) → сырой сдвиг отрицательный.
    ds1 = -ds1
    ds2 = -ds2

    cands: list[tuple[float, float, str]] = []
    if np.isfinite(ds1) and p1 >= cfg.min_prominence and _plausible(ds1, dt, cfg):
        cands.append((ds1, p1, "1d"))
    if np.isfinite(ds2) and p2 >= cfg.min_prominence and _plausible(ds2, dt, cfg):
        cands.append((ds2, p2, "2d"))
    if not cands:
        return float("nan"), 0.0, "none"

    if len(cands) == 2 and abs(cands[0][0] - cands[1][0]) <= cfg.agree_m:
        w = np.array([cands[0][1], cands[1][1]], dtype=np.float64)
        ds = float(np.dot(w, [cands[0][0], cands[1][0]]) / w.sum())
        return ds, float(w.max() + 0.05), "both"

    if len(cands) == 2 and np.isfinite(ds_prior):
        i = int(np.argmin([abs(c[0] - ds_prior) for c in cands]))
        return cands[i][0], cands[i][1], cands[i][2]

    i = int(np.argmax([c[1] for c in cands]))
    return cands[i][0], cands[i][1], cands[i][2]


def deskew_xyz(
    xyz: np.ndarray,
    timestamps: np.ndarray,
    velocity: float,
    t_ref: float | None = None,
) -> np.ndarray:
    """Свести кадр к одному моменту: поезд едет вперёд (−Y) со скоростью `velocity`.

    Точка, снятая раньше `t_ref`, в системе `t_ref` должна быть дальше позади,
    потому что начало координат с тех пор уехало вперёд.
    """
    if abs(velocity) < 0.05 or timestamps.size == 0:
        return xyz
    t = timestamps.astype(np.float64, copy=False)
    finite = np.isfinite(t)
    if not np.any(finite):
        return xyz
    if t_ref is None:
        t_ref = float(np.median(t[finite]))
    dt = t_ref - t
    dt = np.where(finite, dt, 0.0)
    out = np.array(xyz, copy=True, dtype=np.float32)
    # p(t_ref) = p(t) − V·(t_ref − t), V = (0, −v, 0) → y += v·dt
    out[:, 1] = out[:, 1] + np.float32(velocity) * dt.astype(np.float32)
    return out


def stamp_of(cloud) -> float:
    return float(cloud.stamp_sec) + 1e-9 * float(cloud.stamp_nsec)


def _plausible(ds: float, dt: float, cfg: OdomConfig) -> bool:
    v = ds / max(dt, 1e-3)
    return -cfg.v_reverse_tol <= v <= cfg.v_max * 1.05


def _fill_rows(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    n_s = out.shape[1]
    x = np.arange(n_s, dtype=np.float64)
    for i in range(out.shape[0]):
        row = out[i]
        ok = np.isfinite(row)
        if int(ok.sum()) < 6:
            out[i] = 0.0
            continue
        if not bool(ok.all()):
            out[i] = np.interp(x, x[ok], row[ok])
    return out


def _windowed(img: np.ndarray) -> np.ndarray:
    filled = _fill_rows(img)
    filled -= float(np.mean(filled))
    return filled * np.hanning(filled.shape[1])[None, :]


def _peak_1d(
    corr: np.ndarray,
    ds: float,
    search: float,
    window: tuple[float, float] | None = None,
) -> tuple[float, float]:
    n = corr.size
    idx = np.arange(n)
    dx = np.where(idx <= n // 2, idx, idx - n).astype(np.float64)
    max_bin = max(int(np.ceil(search / ds)), 1)
    mask = np.abs(dx) <= max_bin
    if window is not None:
        mask &= (dx * ds >= window[0] - 1e-9) & (dx * ds <= window[1] + 1e-9)
    if not np.any(mask):
        return float("nan"), 0.0
    masked = np.where(mask, corr, -np.inf)
    px = int(np.argmax(masked))
    peak = float(corr[px])
    rival = np.where(mask & (np.abs(idx - px) > 1), corr, -np.inf)
    second = float(np.max(rival))
    prominence = peak - second if np.isfinite(second) else peak
    c0 = float(corr[(px - 1) % n])
    c2 = float(corr[(px + 1) % n])
    denom = c0 - 2.0 * peak + c2
    frac = 0.5 * (c0 - c2) / denom if abs(denom) > 1e-12 else 0.0
    frac = float(np.clip(frac, -0.5, 0.5))
    return (dx[px] + frac) * ds, float(prominence)


def _mean_s(img: np.ndarray) -> np.ndarray:
    """Среднее по θ без предупреждения на полностью пустых столбцах."""
    finite = np.isfinite(img)
    count = finite.sum(axis=0)
    acc = np.where(finite, img, 0.0).sum(axis=0)
    out = np.full(img.shape[1], np.nan, dtype=np.float64)
    ok = count > 0
    out[ok] = acc[ok] / count[ok]
    return out


def _corr_1d(
    a: np.ndarray,
    b: np.ndarray,
    ds: float,
    search: float,
    window: tuple[float, float] | None = None,
) -> tuple[float, float]:
    sa = _mean_s(a)
    sb = _mean_s(b)
    ok = np.isfinite(sa) & np.isfinite(sb)
    if int(ok.sum()) < 0.40 * sa.size:
        return float("nan"), 0.0
    x = np.arange(sa.size, dtype=np.float64)
    sa = np.interp(x, x[ok], sa[ok])
    sb = np.interp(x, x[ok], sb[ok])
    win = np.hanning(sa.size)
    sa = (sa - sa.mean()) * win
    sb = (sb - sb.mean()) * win
    fa = np.fft.rfft(sa)
    fb = np.fft.rfft(sb)
    cross = fa * np.conj(fb)
    mag = np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft(cross / mag, n=sa.size)
    return _peak_1d(corr, ds, search, window)


def _corr_2d(
    a: np.ndarray,
    b: np.ndarray,
    ds: float,
    search: float,
    window: tuple[float, float] | None = None,
) -> tuple[float, float]:
    wa = _windowed(a)
    wb = _windowed(b)
    fa = np.fft.rfft2(wa)
    fb = np.fft.rfft2(wb)
    cross = fa * np.conj(fb)
    mag = np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft2(cross / mag, s=wa.shape)
    n_th, n_s = corr.shape
    yy = np.arange(n_th)
    xx = np.arange(n_s)
    dy = np.where(yy <= n_th // 2, yy, yy - n_th)
    dx = np.where(xx <= n_s // 2, xx, xx - n_s)
    max_bin = max(int(np.ceil(search / ds)), 1)
    along = np.abs(dx) <= max_bin
    if window is not None:
        along &= (dx * ds >= window[0] - 1e-9) & (dx * ds <= window[1] + 1e-9)
    mask = along[None, :] & (np.abs(dy)[:, None] <= 4)
    masked = np.where(mask, corr, -np.inf)
    py, px = np.unravel_index(int(np.argmax(masked)), corr.shape)
    peak = float(corr[py, px])
    rival = masked.copy()
    rival[py, px] = -np.inf
    # соседние по s не считаем вторым пиком
    for dpx in (-1, 0, 1):
        rival[py, (px + dpx) % n_s] = -np.inf
    second = float(np.max(rival))
    prominence = peak - second if np.isfinite(second) else peak
    c0 = float(corr[py, (px - 1) % n_s])
    c2 = float(corr[py, (px + 1) % n_s])
    denom = c0 - 2.0 * peak + c2
    frac = 0.5 * (c0 - c2) / denom if abs(denom) > 1e-12 else 0.0
    frac = float(np.clip(frac, -0.5, 0.5))
    return (float(dx[px]) + frac) * ds, float(prominence)
