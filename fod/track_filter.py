"""Трекер оси пути: три зоны улик + фильтр Калмана на коэффициентах оси.

Состояние — квадратичная ось в текущей системе сенсора: `n(s) = c0 + c1·s + c2·s²`
(`s` = −y вперёд, `n` = x вбок). Такой выбор даёт даровой временной сшив: лидар
прикручен к поезду, а поезд стоит на рельсах, поэтому `c0` (боковое смещение
установки) и `c1` (рыскание) физически почти константы, а `c2` = 1/(2R) меняется
медленно — кривизна набегает на длине переходной кривой, десятки метров.

Поэтому модель процесса — не «случайное блуждание», а возврат к среднему
(Орнштейна — Уленбека): без измерений оценка сползает к «прямо по центру», а не
замирает на последней экстраполяции. Это же лечит документированный в PLAN срыв
«ось уползла на соседний путь и самоподтверждается».

Измерения трёх зон (PLAN, раздел 5.3):

* 4…78 м — пара головок рельсов, согласованный фильтр (`fod.rails`);
* 20…175 м — центр сечения тоннеля плюс перенос привязки «ось тоннеля → ось пути»,
  измеренной там, где видны обе;
* дальше `s_known` — кривизна затухает с длиной `EXTRAP_LENGTH`, а не обрывается:
  жёсткая заморозка давала на картинке излом, за которым путь вдруг шёл прямо.

Побочно измеряются база рельсов и высота головки над полом: база «1.52 м» —
это колея по внутренним граням, а лидар видит верх головок, и разница в 8 см
заметна глазом при наложении на range image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from fod.cloud import MIN_RANGE
from fod.rails import (
    GAUGE_HALF,
    RAIL_BANDS,
    RAIL_SPACING,
    SPACING_LIMITS,
    BandEvidence,
    axis_at,
    best_peak,
    rail_evidence,
    tunnel_hits,
)

EXTRAP_LENGTH = 45.0
RAIL_HEAD_U = 0.14


@dataclass
class TrackConfig:
    # Возврат к среднему за кадр 0.1 с и шум процесса (см. docstring).
    # Кривизна к нулю почти не возвращается: это свойство пути, а не установки.
    # Темп кривизны, наоборот, возвращается быстро: почти везде путь либо прямой,
    # либо круговой, и только на переходной кривой темп заметно отличен от нуля.
    revert: tuple[float, float, float, float] = (0.020, 0.050, 0.002, 0.040)
    process_std: tuple[float, float, float, float] = (0.020, 0.0020, 0.6e-5, 1.0e-7)
    # Кривизна и её темп меняются с пройденным путём, а не со временем: на
    # стоянке они не должны «плыть» вовсе.
    curvature_per_m: float = 0.8e-5
    rate_per_m: float = 1.5e-7
    init_std: tuple[float, float, float, float] = (0.40, 0.030, 3.0e-4, 4.0e-6)

    gate_chi2: float = 9.0
    min_radius_m: float = 150.0
    gate_min: float = 0.25
    gate_locked: float = 0.75
    gate_far: float = 0.55
    gate_search: float = 1.50

    near_split: float = 30.0
    score_min: float = 0.34
    score_min_far: float = 0.42
    prominence_min: float = 0.06
    prominence_min_far: float = 0.12
    lock_bands: int = 2
    lost_after: int = 8

    rail_sigma_base: float = 0.05
    rail_sigma_slope: float = 0.0030
    tunnel_sigma_base: float = 0.25
    tunnel_sigma_slope: float = 0.006
    binding_tau: float = 0.15
    binding_min_hits: int = 3
    wall_min_frac: float = 0.75   # кромка ближе этой доли полуширины — не стенка, а край обзора
    # σ подгонки профиля стен (1.3e-4) слишком оптимистична: остатки коррелированы,
    # а межкадровый разброс κ отвечает 6e-4. Берём измеренное, а не заявленное.
    yaw_sigma_min: float = 5.0e-4
    yaw_sigma_max: float = 3.0e-3
    geometry_tau: float = 0.10
    max_grade: float = 0.03
    max_c2: float = 3.3e-3        # R не меньше 150 м
    max_c3: float = 2.0e-5        # переходная кривая не короче 40 м
    fan_steps: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)   # в σ кривизны
    fan_max_step: float = 6.0e-5
    fan_penalty: float = 0.05
    # Авторазметка оси: резать дальность рельсов и выключать канал тоннеля.
    # По умолчанию поведение как раньше — все полосы и стены в деле.
    rail_s_max: float = 200.0
    use_tunnel: bool = True

    # Станционные наблюдения сети. Ближе 40 м их не берём: там рельсы измеряют
    # ось с σ 0.05 м против замеренных 0.15 м у сети на 50 м, и сеть способна
    # только внести смещение. Дальше 150 м сеть не обучалась.
    axis_s_min: float = 40.0
    axis_s_max: float = 150.0
    axis_sigma_min: float = 0.10
    axis_sigma_max: float = 1.50   # выше — станция отбрасывается, а не зажимается


@dataclass
class BandReport:
    s_mid: float
    source: str          # rail / tunnel / weak / gated / narrow / none
    n_meas: float | None
    n_pred: float
    score: float
    rise: float
    dark: float
    sigma: float
    accepted: bool


@dataclass
class TrackEstimate:
    coeff: np.ndarray
    cov: np.ndarray
    z_floor: float
    s_known: float
    binding: float | None
    radius_m: float
    locked: bool
    misses: int
    n_rail_bands: int
    n_tunnel_bands: int
    yaw_used: bool
    spacing: float
    head_u0: float
    grade: float
    ms: float
    bands: list[BandReport] = field(default_factory=list)
    evidence: list[BandEvidence] = field(default_factory=list)
    axis_used: bool = False
    n_axis_stations: int = 0

    def n_at(self, s: np.ndarray | float) -> np.ndarray:
        """Кубическая ось (дуга + переходная кривая) до `s_known`, дальше — затухание.

        Кривизна за пределами измерений гасится как `exp(-d/EXTRAP_LENGTH)`:
        первая и вторая производные непрерывны, поэтому излома нет, а на бесконечности
        путь всё-таки становится прямым — экстраполировать дугу вечно нечестно.
        """
        s = np.asarray(s, dtype=np.float64)
        c0, c1, c2, c3 = self.coeff
        s_in = np.minimum(s, self.s_known)
        # Член c2³·s⁴ — это точная дуга до четвёртого порядка: для окружности
        # n = s²/(2R) + s⁴/(8R³), а 1/R = 2c2. Полином без него занижает уход на
        # 0.10 м при R = 500 и 0.23 м при R = 380 на сотне метров, то есть на
        # уровне σ. В измерительной зоне (до 78 м) поправка 4 см, поэтому линейная
        # модель измерений в фильтре остаётся корректной и трогать её не нужно.
        arc = c2 ** 3 * s_in ** 4
        base = c0 + s_in * (c1 + s_in * (c2 + s_in * c3)) + arc
        d = np.maximum(s - self.s_known, 0.0)
        if not np.any(d > 0.0):
            return base
        slope = c1 + s_in * (2.0 * c2 + 3.0 * c3 * s_in) + 4.0 * c2 ** 3 * s_in ** 3
        curvature = 2.0 * c2 + 6.0 * c3 * s_in + 12.0 * c2 ** 3 * s_in * s_in
        length = EXTRAP_LENGTH
        turn = curvature * length * (d - length * (1.0 - np.exp(-d / length)))
        return base + slope * d + turn

    def sigma_at(self, s: np.ndarray | float) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        basis = np.stack([np.ones_like(s), s, s * s, s * s * s], axis=-1)
        var = np.einsum("...i,ij,...j->...", basis, self.cov, basis)
        grow = np.maximum(s - self.s_known, 0.0) * 0.012
        return np.sqrt(np.maximum(var, 1e-6)) + grow

    def rails_at(self, s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_at(s)
        half = 0.5 * self.spacing
        return n - half, n + half

    def rail_z(self, s: np.ndarray | float) -> np.ndarray:
        """Высота головки: измеренный продольный профиль, а не константа."""
        s = np.asarray(s, dtype=np.float64)
        return self.z_floor + self.head_u0 + self.grade * s

    def corridor_half(self, s: np.ndarray | float, k: float = 2.0) -> np.ndarray:
        return GAUGE_HALF + k * self.sigma_at(s)


def _basis(s: float) -> np.ndarray:
    return np.array([1.0, s, s * s, s * s * s], dtype=np.float64)


class TrackTracker:
    """Кадр за кадром держит одну гипотезу об оси и не даёт ей скакать."""

    def __init__(self, config: TrackConfig | None = None) -> None:
        self.cfg = config or TrackConfig()
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag(np.square(self.cfg.init_std))
        self.z_floor = -1.40
        self.binding: float | None = None
        self.off_left: float | None = None
        self.off_right: float | None = None
        self.binding_hits = 0
        self.spacing = RAIL_SPACING
        self.head_u0 = RAIL_HEAD_U
        self.grade = 0.0
        self.locked = False
        self.yaw_used = False
        self.axis_used = False
        self.misses = 0
        self.frames = 0

    # --- фильтр -------------------------------------------------------------

    def _predict(self, dt: float, ds: float) -> None:
        """Проезд `ds` по переходной кривой: кривизна набегает как 3·c3·ds.

        Полный перенос оси (`c0' = c0 + c1·ds + … − dn`, `c1' = c1 + 2c2·ds − dψ`)
        математически точнее, но на деле хуже: он впрыскивает в оценку шум
        измеренных поворота и сноса каждый кадр, и дрожание `n(0)` выросло с
        3.5 до 25 мм. Лидар стоит на поезде, поезд — на рельсах, поэтому боковое
        смещение и рыскание физически почти константы, и модель «почти константа»
        точнее шумного измерения. Из движения берётся только набег кривизны.
        """
        k = max(dt, 1e-3) / 0.1
        ds = max(ds, 0.0)
        move = np.eye(4)
        move[2, 3] = 3.0 * ds
        decay = np.clip(1.0 - np.asarray(self.cfg.revert) * k, 0.0, 1.0)
        q = np.square(np.asarray(self.cfg.process_std)) * k
        q[2] += (self.cfg.curvature_per_m * ds) ** 2
        q[3] += (self.cfg.rate_per_m * ds) ** 2
        transition = decay[:, None] * move
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + np.diag(q)

    def _update(self, h: np.ndarray, z: float, sigma: float) -> bool:
        innovation = float(z - h @ self.x)
        s = float(h @ self.P @ h) + sigma * sigma
        if innovation * innovation > self.cfg.gate_chi2 * s:
            return False
        gain = (self.P @ h) / s
        self.x = self.x + gain * innovation
        self.P = self.P - np.outer(gain, h @ self.P)
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def update_axis(self, stations) -> int:
        """Наблюдения сети как независимые скалярные `n(s)`. Возвращает число принятых.

        Раньше сюда приходил весь вектор состояния с ковариацией 4×4, собранной
        из σ якорей через обратный Вандермонд. Замер на измеренных σ: cond(Σ_c)
        ≈ 3·10¹³, корреляции коэффициентов 0.92…0.99, σ(c0) = 1.72 м при том, что
        рельсы измеряют боковое смещение с точностью 0.05 м. Фильтр делал
        четырёхмерный прыжок в вырожденном направлении каждый кадр — это и было
        основным источником скачков итоговой оси.

        Скалярная форма совпадает с той, которой уже пользуются рельсовые полосы:
        гейт χ²(1) применяется к каждой станции отдельно, поэтому мусор с 146 м
        не портит оценку на 62 м. Физический конус проверяется один раз в конце.

        Сквозной замер на roundT_doubleT: MAE `n_at` на 90 м 0.508 м против
        1.834 м без сети, на 130 м 2.173 против 4.704 м; принимается в среднем
        5.6 станции из 8. Обрезка станций дальше 120 м пробовалась и делает хуже
        (2.32 м на 130 м), хотя σ там занижена в 1.4× — дальние станции всё равно
        несут форму, которую ближним полосам взять негде.
        """
        if stations is None:
            return 0
        cfg = self.cfg
        used = 0
        for item in stations:
            s, n, sigma = (float(v) for v in item)
            if not (np.isfinite(s) and np.isfinite(n) and np.isfinite(sigma)):
                continue
            if sigma <= 0.0 or s < cfg.axis_s_min or s > cfg.axis_s_max:
                continue
            # σ сверху не зажимается, а отбрасывается: зажатая «не знаю» на 10 м
            # превратилась бы в наблюдение с весом σ = 3 м и всё равно потянула
            # бы оценку. Снизу зажимается — сеть систематически переуверена.
            if sigma > cfg.axis_sigma_max:
                continue
            sigma = max(sigma, cfg.axis_sigma_min)
            used += int(self._update(_basis(s), n, sigma))
        if used:
            self._clamp_geometry()
        return used

    def _focus_curvature(self, xyz, intensity, anchor, windows, bands):
        """Веер гипотез по кривизне: наводка на резкость для дальних полос.

        Точки разгибаются по предсказанной оси, и ошибка кривизны размазывает
        головку вдоль полосы: при σ(c2) = 5e-5 на полосе 62…78 м это 11 см, то
        есть два бина. Перебор нескольких c2 вокруг оценки возвращает резкость.
        Отклонение штрафуется, иначе перебор всегда найдёт кривизну, под которую
        подстроится шум разреженной полосы.
        """
        cfg = self.cfg
        spread = float(np.clip(np.sqrt(max(self.P[2, 2], 1e-12)), 1e-6, cfg.fan_max_step))
        # Полосы узкие, поэтому облако режем один раз на всю дальнюю зону.
        s = -xyz[:, 1]
        near = (s >= bands[0][0] - 1.0) & (s < bands[-1][1] + 1.0)
        sub_xyz, sub_int = xyz[near], intensity[near]

        best_total = -np.inf
        best: tuple[np.ndarray, list] | None = None
        for step in cfg.fan_steps:
            candidate = anchor.copy()
            candidate[2] += step * spread
            stage = rail_evidence(sub_xyz, sub_int, self.z_floor, candidate, windows, bands)
            total = 0.0
            for band, window in zip(stage, windows):
                center = float(axis_at(candidate, np.array(band.s_mid)))
                peak = best_peak(band, center, float(window))
                if peak is not None and peak.score >= cfg.score_min_far:
                    total += peak.score
            total -= cfg.fan_penalty * step * step
            if total > best_total:
                best_total, best = total, (candidate, stage)
        if best is None:
            return anchor, rail_evidence(sub_xyz, sub_int, self.z_floor, anchor, windows, bands)
        return best

    def _clamp_geometry(self) -> None:
        """Рельсы — не свободная кривая: радиус и длина переходной кривой нормированы.

        Минимальный радиус метро принят 150 м, темп набора кривизны ограничен
        переходной кривой не короче 40 м. Ограничение почти всегда не работает,
        но не даёт дальней зоне уехать в геометрически невозможную дугу.
        """
        self.x[2] = float(np.clip(self.x[2], -self.cfg.max_c2, self.cfg.max_c2))
        self.x[3] = float(np.clip(self.x[3], -self.cfg.max_c3, self.cfg.max_c3))

    def _sigma_pred(self, s: float) -> float:
        h = _basis(s)
        return float(np.sqrt(max(h @ self.P @ h, 1e-6)))

    # --- измерения ----------------------------------------------------------

    def _floor(self, xyz: np.ndarray) -> float:
        s = -xyz[:, 1]
        n = xyz[:, 0] - axis_at(self.x, s)
        m = (s > 6.0) & (s < 25.0) & (np.abs(n) < 1.6)
        m &= np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE
        if int(m.sum()) < 200:
            m = (s > 4.0) & (s < 40.0) & (np.abs(n) < 2.5)
            m &= np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE
        if int(m.sum()) < 50:
            return self.z_floor
        return float(np.percentile(xyz[m, 2], 12))

    def step(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        dt: float = 0.1,
        ds: float | None = None,
        curvature: tuple[float, float] | None = None,
        axis: list[tuple[float, float, float]] | None = None,
    ) -> TrackEstimate:
        """`ds` — пройденный за кадр путь, `curvature` — (κ, σ) из поворота одометрии.

        Кривизна из поворота — независимый и главное **безынерционный** канал:
        поезд стоит на рельсах, поэтому его угловая скорость, делённая на
        скорость, и есть кривизна пути под ним. Форму оси впереди она не знает,
        зато мгновенно фиксирует кривизну в нуле, куда полиному дотянуться нечем.

        `axis` — необязательный список `(s, n, σ)` от сети; без него путь тот же,
        что раньше.
        """
        started = time.perf_counter()
        cfg = self.cfg
        self._predict(dt, 2.0 * dt / 0.1 if ds is None else ds)
        self.yaw_used = False
        if curvature is not None:
            kappa, sigma = curvature
            if np.isfinite(kappa) and np.isfinite(sigma):
                sigma = float(np.clip(sigma, cfg.yaw_sigma_min, cfg.yaw_sigma_max))
                self.yaw_used = self._update(np.array([0.0, 0.0, 2.0, 0.0]), kappa, sigma)
        n_axis = self.update_axis(axis)
        self.axis_used = n_axis > 0

        floor_now = self._floor(xyz)
        self.z_floor = floor_now if self.frames == 0 else 0.7 * self.z_floor + 0.3 * floor_now

        reports: list[BandReport] = []
        evidence: list[BandEvidence] = []
        geometry: list[tuple[float, float, float]] = []   # s_mid, база, высота головки
        s_known = 8.0
        n_rail = 0

        # Два прохода: сначала ближние полосы правят ось, потом по уточнённой оси
        # разгибаются дальние. Один проход означал бы, что дальние полосы биннятся
        # по одной оси, а гейтятся по другой — уехавшей после ближних обновлений.
        for far in (False, True):
            bands = tuple(
                b
                for b in RAIL_BANDS
                if 0.5 * (b[0] + b[1]) <= cfg.rail_s_max
                and (0.5 * (b[0] + b[1]) > cfg.near_split) == far
            )
            if not bands:
                continue
            anchor = self.x.copy()
            gate_cap = (cfg.gate_far if far else cfg.gate_locked) if self.locked else cfg.gate_search
            windows = np.array(
                [
                    float(np.clip(3.0 * self._sigma_pred(0.5 * (a + b)) + 0.20, cfg.gate_min, gate_cap))
                    for a, b in bands
                ]
            )
            if far:
                anchor, stage = self._focus_curvature(xyz, intensity, anchor, windows, bands)
            else:
                stage = rail_evidence(xyz, intensity, self.z_floor, anchor, windows, bands)
            evidence.extend(stage)
            score_min = cfg.score_min_far if far else cfg.score_min
            prominence_min = cfg.prominence_min_far if far else cfg.prominence_min

            for band, window in zip(stage, windows):
                center = float(axis_at(anchor, np.array(band.s_mid)))
                peak = best_peak(band, center, float(window))
                if peak is None:
                    reports.append(BandReport(band.s_mid, "none", None, center, 0.0, 0.0, 0.0, 0.0, False))
                    continue
                strong = peak.score >= score_min and peak.prominence >= prominence_min
                sigma = float(
                    np.clip(
                        (cfg.rail_sigma_base + cfg.rail_sigma_slope * band.s_mid) / max(peak.score, 0.3),
                        0.05,
                        0.60,
                    )
                )
                accepted = self._update(_basis(band.s_mid), peak.n, sigma) if strong else False
                if accepted:
                    n_rail += 1
                    s_known = max(s_known, band.s_mid)
                    if SPACING_LIMITS[0] < peak.spacing < SPACING_LIMITS[1] and np.isfinite(peak.u_head):
                        geometry.append((band.s_mid, peak.spacing, peak.u_head))
                reports.append(
                    BandReport(
                        s_mid=band.s_mid,
                        source="rail" if accepted else ("weak" if not strong else "gated"),
                        n_meas=peak.n,
                        n_pred=center,
                        score=peak.score,
                        rise=peak.rise,
                        dark=peak.dark,
                        sigma=sigma,
                        accepted=accepted,
                    )
                )

        self.locked = n_rail >= cfg.lock_bands
        self.misses = 0 if self.locked else self.misses + 1
        if self.misses == cfg.lost_after:
            # Долго без рельсов — вернуть готовность к поиску, не теряя оценку.
            self.P = self.P + np.diag(np.square(self.cfg.init_std)) * 0.25

        self._update_geometry(geometry)
        n_tunnel = 0
        if cfg.use_tunnel:
            n_tunnel = self._tunnel_stage(xyz, reports, s_known, n_rail)
            for report in reports:
                if report.source == "tunnel" and report.accepted:
                    s_known = max(s_known, min(report.s_mid, 120.0))

        self._clamp_geometry()
        c2 = float(self.x[2])
        radius = float(1.0 / abs(2.0 * c2)) if abs(c2) > 1e-6 else float("inf")

        self.frames += 1

        return TrackEstimate(
            coeff=self.x.copy(),
            cov=self.P.copy(),
            z_floor=self.z_floor,
            s_known=float(s_known),
            binding=self.binding,
            radius_m=radius,
            locked=self.locked,
            misses=self.misses,
            n_rail_bands=n_rail,
            n_tunnel_bands=n_tunnel,
            yaw_used=self.yaw_used,
            spacing=self.spacing,
            head_u0=self.head_u0,
            grade=self.grade,
            ms=(time.perf_counter() - started) * 1e3,
            bands=reports,
            evidence=evidence,
            axis_used=self.axis_used,
            n_axis_stations=n_axis,
        )

    def _update_geometry(self, hits: list[tuple[float, float, float]]) -> None:
        """База рельсов и продольный профиль головки — медленные величины."""
        if len(hits) < 2:
            return
        tau = self.cfg.geometry_tau
        data = np.array(hits, dtype=np.float64)
        self.spacing += tau * (float(np.median(data[:, 1])) - self.spacing)

        s, u = data[:, 0], data[:, 2]
        if s.size >= 3 and float(s.max() - s.min()) > 12.0:
            grade, intercept = np.polyfit(s, u, 1)
            grade = float(np.clip(grade, -self.cfg.max_grade, self.cfg.max_grade))
        else:
            grade, intercept = self.grade, float(np.mean(u) - self.grade * float(np.mean(s)))
        self.grade += tau * (grade - self.grade)
        self.head_u0 += tau * (float(np.clip(intercept, -0.1, 0.5)) - self.head_u0)

    def _tunnel_stage(
        self,
        xyz: np.ndarray,
        reports: list[BandReport],
        s_rail_max: float,
        n_rail: int,
    ) -> int:
        cfg = self.cfg
        anchor = self.x.copy()
        hits = tunnel_hits(xyz, self.z_floor, anchor)

        # Привязка «ось → кромка» измеряется отдельно для каждой стенки и только
        # там, где в этом же кадре найдены рельсы.
        if n_rail >= cfg.lock_bands:
            near = [h for h in hits if h.s_mid <= min(s_rail_max + 6.0, 55.0)]
            self._learn_offsets(anchor, near)

        n_used = 0
        for hit in hits:
            pred = float(axis_at(anchor, np.array(hit.s_mid)))
            source, z = self._axis_from_walls(hit, pred)
            if z is None or hit.s_mid <= s_rail_max:
                reports.append(BandReport(hit.s_mid, source, None, pred, 0.0, 0.0, 0.0, 0.0, False))
                continue
            sigma = cfg.tunnel_sigma_base + cfg.tunnel_sigma_slope * max(hit.s_mid - 50.0, 0.0)
            if source == "wall":
                sigma *= 1.6   # одна кромка знает ось хуже, чем обе
            accepted = self._update(_basis(hit.s_mid), z, sigma)
            n_used += int(accepted)
            reports.append(
                BandReport(
                    s_mid=hit.s_mid,
                    source=source if accepted else "gated",
                    n_meas=z,
                    n_pred=pred,
                    score=float(hit.width),
                    rise=0.0,
                    dark=0.0,
                    sigma=sigma,
                    accepted=accepted,
                )
            )
        return n_used

    def _learn_offsets(self, anchor: np.ndarray, hits: list) -> None:
        left, right = [], []
        for hit in hits:
            if hit.left is None or hit.right is None:
                continue
            axis = float(axis_at(anchor, np.array(hit.s_mid)))
            left.append(axis - hit.left)
            right.append(axis - hit.right)
        if not left:
            return
        tau = self.cfg.binding_tau
        for name, values in (("off_left", left), ("off_right", right)):
            measured = float(np.median(values))
            current = getattr(self, name)
            setattr(self, name, measured if current is None else current + tau * (measured - current))
        self.binding_hits = min(self.binding_hits + 1, 50)
        self.binding = 0.5 * (self.off_left + self.off_right)

    def _axis_from_walls(self, hit, pred: float) -> tuple[str, float | None]:
        """Ось по сечению: обе кромки точнее, но одной видимой уже достаточно."""
        if self.binding_hits < self.cfg.binding_min_hits or self.off_left is None:
            return "none", None
        both = hit.left is not None and hit.right is not None
        if both:
            return "tunnel", hit.center + 0.5 * (self.off_left + self.off_right)

        # Перекрытая стенка выглядит как кромка видимой области — она заметно
        # ближе к оси, чем настоящая. Отбрасываем её по выученной полуширине.
        candidates = []
        for edge, offset in ((hit.left, self.off_left), (hit.right, self.off_right)):
            if edge is None:
                continue
            if abs(edge - pred) < self.cfg.wall_min_frac * abs(offset):
                continue
            candidates.append(edge + offset)
        if not candidates:
            return "narrow", None
        return "wall", float(np.mean(candidates))
