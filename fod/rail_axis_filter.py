"""Фильтр оси между кадрами и сложение дальних оборотов.

Состояние — кубика в системе сенсора, `n(s) = c0 + c1 s + c2 s² + c3 s³`.
Лидар стоит на поезде, поэтому `c0` и `c1` почти константы. Из одометрии
в прогноз входит только пройденный путь: кривизна набегает как `c2 ← c2 + 3 c3 ds`.
Поворот и боковой снос в `c0`/`c1` не подмешиваются: на старом трекере это
раздувало дрожание `n(0)` с 3.5 мм до 25 мм.

Числа возврата к среднему и шума процесса те же, что уже измерены в
`fod.track_filter.TrackConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fod.rail_template import S_MAX, S_MIN, RailFrame, detect_rails

CHI2 = 9.0
STACK_KEEP = 4
STACK_S_FROM = 40.0
STACK_DS_MAX = 25.0


@dataclass
class RailAxisConfig:
    # Курс под поездом почти не тянется к нулю: 5 %/кадр пилой дёргали ось.
    revert: tuple[float, float, float, float] = (0.020, 0.010, 0.002, 0.040)
    # Смещение и курс кузова относительно пути почти константы, кривизна живёт
    # вдвое свободнее: её меняет путь, а не качка.
    process_std: tuple[float, float, float, float] = (0.005, 0.0010, 1.5e-5, 2.0e-7)
    curvature_per_m: float = 0.8e-5
    rate_per_m: float = 1.5e-7
    init_std: tuple[float, float, float, float] = (0.40, 0.030, 3.0e-4, 4.0e-6)
    # σ измерения n, метры. Больше — кадр слабее двигает ось, фильтр держит прогноз.
    meas_base: float = 0.05
    meas_slope: float = 0.0030
    # Дальше этой дальности все пары кадра сжимаются в одно измерение,
    # а σ растёт квадратом. 80 м — за горизонтом детектора: каждая пара идёт
    # своим измерением, и кривизну дуги фильтр набирает сам, без подтяжки формы.
    meas_far_s: float = 80.0
    meas_far_quad: float = 0.5
    # Когда конец уже измерен (σ оси < far_known_sigma), одиночный скачок больше
    # этого отбрасывается.
    far_known_sigma: float = 0.45
    far_jump_m: float = 0.5
    # Доля, на которую кривизна за кадр едет к полиному пар кадра.
    # 1.0 — прежний режим: форма кадра целиком, память о кривизне пропадает.
    shape_alpha: float = 0.0
    max_c2: float = 3.3e-3
    max_c3: float = 2.0e-5
    lock_updates: int = 6
    lock_sigma: float = 0.20


# Параметры до перехода на ось фильтра — для сравнения в режиме `frame`.
LEGACY_CONFIG = RailAxisConfig(
    revert=(0.020, 0.050, 0.002, 0.040),
    process_std=(0.020, 0.0020, 0.6e-5, 1.0e-7),
    meas_far_s=45.0,
    meas_far_quad=3.0,
    far_jump_m=0.18,
    shape_alpha=1.0,
)


class RailAxisFilter:
    """Одна гипотеза об оси. Пара с соседнего пути гейтом не проходит."""

    def __init__(self, config: RailAxisConfig | None = None) -> None:
        self.cfg = config or RailAxisConfig()
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag(np.square(self.cfg.init_std))
        self.accepted = 0
        self.rejected = 0
        self.miss_frames = 0
        self.locked = False

    def predict(self, ds: float, dt: float) -> None:
        """Проезд `ds`. На стоянке состояние не блуждает, только чуть тянется к нулю."""
        k = max(float(dt), 1e-3) / 0.1
        ds = max(float(ds), 0.0)
        if not np.isfinite(ds):
            ds = 0.0
        move = np.eye(4)
        move[2, 3] = 3.0 * ds
        decay = np.clip(1.0 - np.asarray(self.cfg.revert) * k, 0.0, 1.0)
        q = np.square(np.asarray(self.cfg.process_std)) * k
        q[2] += (self.cfg.curvature_per_m * ds) ** 2
        q[3] += (self.cfg.rate_per_m * ds) ** 2
        transition = decay[:, None] * move
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + np.diag(q)
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()

    def n(self, s: np.ndarray | float) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        c0, c1, c2, c3 = self.x
        return c0 + s * (c1 + s * (c2 + s * c3))

    def psi(self, s: float) -> float:
        s = float(s)
        slope = self.x[1] + s * (2.0 * self.x[2] + 3.0 * self.x[3] * s)
        return float(np.arctan(np.clip(slope, -1.2, 1.2)))

    def sigma(self, s: float) -> float:
        s = float(s)
        h = np.array([1.0, s, s * s, s * s * s], dtype=np.float64)
        var = float(h @ self.P @ h)
        return float(np.sqrt(max(var, 1e-8)))

    def meas_sigma(self, s: float) -> float:
        s = abs(float(s))
        sigma = self.cfg.meas_base + self.cfg.meas_slope * s
        if s > self.cfg.meas_far_s:
            t = (s - self.cfg.meas_far_s) / self.cfg.meas_far_s
            sigma *= 1.0 + self.cfg.meas_far_quad * t * t
        return sigma

    def update_marks(self, marks) -> int:
        """Центры пар. Возвращает, сколько прошло гейт.

        Ближние пары обновляют ось по одной: там σ около 5 см. Дальние за кадр
        дают одно измерение по медиане. Иначе десяток шумных точек с плечом s³
        двигает конец оси, хотя это один и тот же кадр.
        """
        accepted = 0
        near = [mark for mark in marks if float(mark.s) < self.cfg.meas_far_s]
        far = [mark for mark in marks if float(mark.s) >= self.cfg.meas_far_s]
        for mark in near:
            if self._update_at(float(mark.s), float(mark.n)):
                accepted += 1
            else:
                self.rejected += 1
        if far:
            s = float(np.median([float(mark.s) for mark in far]))
            n = float(np.median([float(mark.n) for mark in far]))
            if self._update_at(s, n):
                accepted += 1
            else:
                self.rejected += 1
        self.accepted += accepted
        if accepted:
            self.miss_frames = 0
        elif self.locked:
            self.miss_frames += 1
        self.locked = self.accepted >= self.cfg.lock_updates and self.sigma(8.0) < self.cfg.lock_sigma
        return accepted

    def update_curvature(self, kappa: float, sigma: float) -> bool:
        """`κ = dyaw/ds` под поездом — это `2 c2` при `s = 0`."""
        if not np.isfinite(kappa) or not np.isfinite(sigma) or sigma <= 0.0:
            return False
        h = np.array([0.0, 0.0, 2.0, 0.0], dtype=np.float64)
        return self._update(h, float(kappa), float(sigma))

    def follow_marks(self, marks, alpha: float = 1.0) -> None:
        """Кривизна тянется к полиному пар кадра на долю `alpha`. Смещение и курс не трогаются.

        Иначе после дуги кубика ещё долго рисует поворот, хотя пары уже на прямой.
        `alpha = 1` — форма кадра целиком, память фильтра о кривизне пропадает.
        """
        if len(marks) < 3:
            return
        s = np.array([float(mark.s) for mark in marks], dtype=np.float64)
        n = np.array([float(mark.n) for mark in marks], dtype=np.float64)
        if float(s.max() - s.min()) < 15.0:
            return
        degree = 3 if s.size >= 6 else 2
        coeff = np.polyfit(s, n, degree)
        c2 = float(coeff[-3])
        c3 = float(coeff[0]) if degree >= 3 else 0.0
        self.pull_shape(c2, c3, alpha)

    def samples(self, s0: float = 2.0, s1: float = S_MAX, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        s = np.arange(s0, s1 + 1e-9, step, dtype=np.float64)
        return s, np.asarray(self.n(s), dtype=np.float64)

    def pull_shape(self, c2: float, c3: float, alpha: float) -> None:
        """Сдвинуть только кривизну. Смещение и курс под поездом не меняются."""
        if not np.isfinite(c2) or not np.isfinite(c3) or not np.isfinite(alpha):
            return
        share = float(np.clip(alpha, 0.0, 1.0))
        self.x[2] = (1.0 - share) * self.x[2] + share * float(c2)
        self.x[3] = (1.0 - share) * self.x[3] + share * float(c3)
        self._clamp()

    def update_n(self, s: float, n: float, sigma: float) -> bool:
        """Измерение `n(s)` с заданной σ. Дальний скачок больше порога отбрасывается."""
        if not np.isfinite(s) or not np.isfinite(n) or not np.isfinite(sigma) or sigma <= 0.0:
            return False
        if s >= self.cfg.meas_far_s and self.sigma(s) < self.cfg.far_known_sigma:
            predicted = float(np.asarray(self.n(s)).reshape(-1)[0])
            if abs(float(n) - predicted) > self.cfg.far_jump_m:
                return False
        h = np.array([1.0, float(s), float(s) ** 2, float(s) ** 3], dtype=np.float64)
        return self._update(h, float(n), float(sigma))

    def _update_at(self, s: float, n: float) -> bool:
        if s >= self.cfg.meas_far_s and self.sigma(s) < self.cfg.far_known_sigma:
            predicted = float(np.asarray(self.n(s)).reshape(-1)[0])
            if abs(n - predicted) > self.cfg.far_jump_m:
                return False
        h = np.array([1.0, s, s * s, s * s * s], dtype=np.float64)
        return self._update(h, n, self.meas_sigma(s))

    def _update(self, h: np.ndarray, z: float, sigma: float) -> bool:
        innov = float(z - h @ self.x)
        innovation_var = float(h @ self.P @ h) + sigma * sigma
        if innovation_var <= 0.0 or innov * innov > CHI2 * innovation_var:
            return False
        gain = (self.P @ h) / innovation_var
        self.x = self.x + gain * innov
        self.P = self.P - np.outer(gain, h @ self.P)
        self.P = 0.5 * (self.P + self.P.T)
        self._clamp()
        return True

    def _clamp(self) -> None:
        self.x[2] = float(np.clip(self.x[2], -self.cfg.max_c2, self.cfg.max_c2))
        self.x[3] = float(np.clip(self.x[3], -self.cfg.max_c3, self.cfg.max_c3))


def shift_into_current(xyz: np.ndarray, ds: float, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Перенос облака на проезд `ds` вдоль касательной в текущий кадр.

    Неподвижная точка после проезда оказывается ближе на `ds` и сдвигается
    поперёк на `ds sin ψ`.
    """
    cos_psi = float(np.cos(psi))
    sin_psi = float(np.sin(psi))
    n = xyz[:, 0] - ds * sin_psi
    s = -xyz[:, 1] - ds * cos_psi
    shifted = np.column_stack([n, -s, xyz[:, 2]])
    return shifted, s


def stack_far(
    history: list[tuple[np.ndarray, np.ndarray, float]],
    xyz: np.ndarray,
    intensity: np.ndarray,
    s_now: float,
    psi: float,
    *,
    locked: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Текущий кадр целиком плюс дальние точки 3–4 прошлых оборотов."""
    if not locked or not history:
        return xyz, intensity
    parts_xyz = [xyz]
    parts_i = [np.asarray(intensity)]
    for past_xyz, past_i, past_s in history:
        ds = float(s_now) - float(past_s)
        if ds < 0.05 or ds > STACK_DS_MAX:
            continue
        shifted, s = shift_into_current(past_xyz, ds, psi)
        keep = (s >= STACK_S_FROM) & (s <= S_MAX)
        if int(keep.sum()) < 12:
            continue
        parts_xyz.append(shifted[keep])
        parts_i.append(np.asarray(past_i)[keep])
    if len(parts_xyz) == 1:
        return xyz, intensity
    return np.concatenate(parts_xyz, axis=0), np.concatenate(parts_i, axis=0)


JITTER_S = (10.0, 30.0, 50.0)


class RailTracker:
    """Кадр за кадром: прогноз по `ds`, поиск шаблоном, обновление по парам.

    `axis_mode`:

    * `filter` — ось кадра = кубика фильтра;
    * `frame` — прежнее поведение для сравнения: ось кадра = полином его пар,
      фильтр на `LEGACY_CONFIG`, кривизна каждый кадр заменяется формой пар.
    """

    def __init__(
        self,
        config: RailAxisConfig | None = None,
        guide=None,
        axis_mode: str = "filter",
        predict_m: float = S_MAX,
        detector=None,
    ) -> None:
        if axis_mode not in ("filter", "frame"):
            raise ValueError(f"axis_mode: filter или frame, не {axis_mode!r}")
        if config is None and axis_mode == "frame":
            config = LEGACY_CONFIG
        if not np.isfinite(predict_m) or float(predict_m) < S_MIN:
            raise ValueError(f"predict_m должен быть не короче {S_MIN:.0f} м")
        self.filter = RailAxisFilter(config)
        self.guide = guide
        self.axis_mode = axis_mode
        self.predict_m = float(predict_m)
        # Иной источник пар (например, SegRailDetector): (xyz, intensity, prior, predict_m) → RailFrame.
        self.detector = detector
        self._hist: list[tuple[np.ndarray, np.ndarray, float]] = []
        # n показанной оси на JITTER_S по кадрам; NaN, если ось туда не дотянулась.
        self.n_shown: list[np.ndarray] = []

    def step(self, xyz: np.ndarray, intensity: np.ndarray, motion, pre=None) -> RailFrame:
        """`pre` — заранее посчитанный выход сети для `detector` (см. `SegRailDetector.classify_only`)."""
        ds = float(getattr(motion, "travelled", 0.0))
        dt = float(getattr(motion, "dt", 0.1))
        s_now = float(getattr(motion, "s", 0.0))
        if not np.isfinite(ds) or ds < 0.0:
            ds = 0.0
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        if not np.isfinite(s_now):
            s_now = 0.0
        self.filter.predict(ds, dt)
        curve = motion.curvature() if hasattr(motion, "curvature") else (float("nan"), float("nan"))
        if curve is not None and len(curve) == 2:
            self.filter.update_curvature(float(curve[0]), float(curve[1]))
        prior = self.filter if self.filter.locked else None
        if self.detector is not None:
            extra = {} if pre is None else {"pre": pre}
            frame = self.detector(xyz, intensity, prior=prior, predict_m=self.predict_m, **extra)
        else:
            search_xyz, search_i = stack_far(self._hist, xyz, intensity, s_now, self.filter.psi(0.0), locked=self.filter.locked)
            frame = detect_rails(search_xyz, search_i, prior=prior, predict_m=self.predict_m)
        self.filter.update_marks(frame.marks)
        if self.filter.cfg.shape_alpha > 0.0:
            self.filter.follow_marks(frame.marks, self.filter.cfg.shape_alpha)
        if self.guide is not None and self.filter.locked:
            self.guide.apply(xyz, intensity, self.filter, frame.marks)
        if self.axis_mode == "filter" and self.filter.locked:
            for mark in frame.marks:
                mark.psi = self.filter.psi(mark.s)
            frame.axis_s, frame.axis_n = self.filter.samples(s1=self.predict_m)
            frame.axis_filtered = True
        self.n_shown.append(_axis_at(frame, JITTER_S))
        self._hist.append((np.asarray(xyz), np.asarray(intensity), s_now))
        if len(self._hist) > STACK_KEEP:
            self._hist = self._hist[-STACK_KEEP:]
        return frame

    def jitter(self) -> dict[float, tuple[float, float]]:
        """|Δn| показанной оси между соседними кадрами на JITTER_S: (медиана, p95), метры."""
        out: dict[float, tuple[float, float]] = {}
        if len(self.n_shown) < 2:
            return {s: (float("nan"), float("nan")) for s in JITTER_S}
        n = np.stack(self.n_shown, axis=0)
        step = np.abs(np.diff(n, axis=0))
        for j, s in enumerate(JITTER_S):
            col = step[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                out[s] = (float("nan"), float("nan"))
            else:
                out[s] = (float(np.median(col)), float(np.percentile(col, 95)))
        return out


def _axis_at(frame: RailFrame, stations: tuple[float, ...]) -> np.ndarray:
    out = np.full(len(stations), np.nan, dtype=np.float64)
    if frame.axis_s is None or frame.axis_n is None or frame.axis_s.size < 2:
        return out
    s = np.asarray(stations, dtype=np.float64)
    inside = (s >= frame.axis_s[0]) & (s <= frame.axis_s[-1])
    out[inside] = np.interp(s[inside], frame.axis_s, frame.axis_n)
    return out
