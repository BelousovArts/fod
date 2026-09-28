"""Последовательный поиск рельсов шаблоном пары головок.

Внутри кадра шаблоны идут от поезда вдаль. Шаг между станциями растёт
медленно: 2 м у поезда, 4 м к 60 м. Срез вдоль рельса длиннее шага на дальней
дистанции, чтобы на головку хватало точек.

Ось внутри кадра — один полином по всем уже найденным парам:

* две точки — прямая;
* три–пять — парабола `n = c0 + c1 s + c2 s²`, кривизна постоянна;
* шесть и дальше — кубика `n = c0 + c1 s + c2 s² + c3 s³`. Если кубика круче
  радиуса 150 м или переходная короче ~40 м, остаётся парабола по тем же точкам.

Между кадрами эту ось держит `RailTracker`: поиск следующего кадра идёт от
его прогноза. Шаблон ставится по нормали к касательной, база 1.60 м меряется
поперёк рельса.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fod.cloud import estimate_floor_z
from fod.rails import RISE_REF, SPACING_LIMITS, BandEvidence, best_peak, rail_evidence

S_MIN = 2.0
S_MAX = 60.0
STEP_NEAR = 2.0
STEP_FAR = 4.0
# Срез вдоль касательной. Вблизи он совпадает с шагом, к концу горизонта дорастает до 7 м:
# возвратов на головку меньше, и короткий срез перестаёт набирать порог.
SLICE_NEAR = 2.0
SLICE_FAR = 7.0
# Первый шаблон: прямо перед поездом, не на весь коридор ±12 м.
FIRST_HALF = 1.4
# Дальше этого, так и не найдя первую пару, цепочку не начинаем.
START_S_MAX = 12.0
# Допуск вокруг экстраполяции. Не растёт углом: 0.5 м в сторону — это уже
# другой рельс, а не продолжение прямой из десяти предыдущих шаблонов.
GATE_HALF = 0.22
GATE_PER_M = 0.012
# На каждый метр пустого пути поперёк оси добавляется 2 см. Потолок выше только
# пока дыра не закрыта: после попадания окно снова от σ фильтра.
GAP_GROW_PER_M = 0.02
GATE_CAP = 0.45
GATE_CAP_GAP = 0.75
# Пустые срезы не обрывают цепочку, пока суммарный пропуск короче этого.
GAP_BUDGET = 15.0
# Дальше этой дальности пара проходит, только если есть и подъём, и темнота.
FAR_BOTH_S = 40.0
RISE_FLOOR = 0.025
DARK_FLOOR = 0.04
GEO_FLOOR = 0.45
# Три точки — парабола, шесть — кубика. Считаются все найденные пары.
AXIS_FROM = 3
CUBIC_FROM = 6
MAX_KAPPA = 1.0 / 150.0
# |c3| в n = … + c3 s³. 2e-5 — тот же предел, что у трекера: переходная не короче ~40 м.
MAX_C3 = 2.0e-5
SCORE_MIN = 0.34
PROMINENCE_MIN = 0.06
# |ψ| больше ~40° на метро не бывает; cos режем, чтобы не делить на ноль.
COS_MIN = 0.75


def _lerp(s: float, near: float, far: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    span = max(s_max - s_min, 1.0)
    t = float(np.clip((float(s) - s_min) / span, 0.0, 1.0))
    return near + (far - near) * t


def step_at(s: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    """Шаг до следующей станции: 2 м у поезда, 4 м к концу горизонта."""
    return _lerp(s, STEP_NEAR, STEP_FAR, s_min, s_max)


def slice_at(s: float, s_min: float = S_MIN, s_max: float = S_MAX) -> float:
    """Длина среза вдоль рельса: 2 м у поезда, 7 м к концу горизонта."""
    return _lerp(s, SLICE_NEAR, SLICE_FAR, s_min, s_max)


def count_limits(s_mid: float) -> tuple[float, float]:
    """Сколько точек требовать в головке и в кольце. К дальнему концу порог падает до 1 и 3."""
    t = float(np.clip((float(s_mid) - FAR_BOTH_S) / FAR_BOTH_S, 0.0, 1.0))
    return 2.0 - t, 6.0 - 3.0 * t


def template_bands(
    s_min: float = S_MIN,
    s_max: float = S_MAX,
) -> tuple[tuple[float, float, float], ...]:
    """Станции `(s0, s1, step)`. Срез может быть длиннее шага и перекрывать соседний."""
    bands: list[tuple[float, float, float]] = []
    s = float(s_min)
    while s < s_max - 0.5:
        step = step_at(s, s_min, s_max)
        mid = min(s + 0.5 * step, float(s_max))
        length = slice_at(mid, s_min, s_max)
        b0 = max(0.5, mid - 0.5 * length)
        b1 = min(float(s_max), mid + 0.5 * length)
        if b1 > b0 + 0.8:
            bands.append((float(b0), float(b1), float(step)))
        nxt = s + step
        if nxt <= s + 1e-6:
            break
        s = nxt
    if len(bands) < 2:
        raise ValueError(f"слишком короткий диапазон {s_min}…{s_max}")
    return tuple(bands)


TEMPLATE_BANDS = template_bands()


@dataclass
class RailMark:
    """Одна пара головок в одной полосе дальности."""

    s0: float
    s1: float
    s: float
    n: float
    n_left: float
    n_right: float
    z: float
    score: float
    rise: float
    dark: float
    prominence: float
    spacing: float
    psi: float


@dataclass
class RailFrame:
    z_floor: float
    marks: list[RailMark]
    evidence: list[BandEvidence]
    ms: float
    axis_s: np.ndarray | None = None
    axis_n: np.ndarray | None = None
    axis_filtered: bool = False


class _Axis:
    """Ось n(s) по всем найденным парам. `u = (s - s_ref) / scale`."""

    def __init__(self, coeff_u: np.ndarray, s_ref: float, scale: float) -> None:
        self.coeff_u = np.asarray(coeff_u, dtype=np.float64)
        self.s_ref = float(s_ref)
        self.scale = float(scale)

    def n(self, s: np.ndarray | float) -> np.ndarray:
        u = (np.asarray(s, dtype=np.float64) - self.s_ref) / self.scale
        return np.polyval(self.coeff_u, u)

    def slope(self, s: float) -> float:
        if self.coeff_u.size < 2:
            return 0.0
        u = (float(s) - self.s_ref) / self.scale
        return float(np.polyval(np.polyder(self.coeff_u), u) / self.scale)

    def psi(self, s: float) -> float:
        return float(np.arctan(np.clip(self.slope(s), -1.2, 1.2)))


def _kappa_peak(coeff: np.ndarray, u: np.ndarray, scale: float) -> float:
    """Максимум |n''(s)| на отрезке данных."""
    if coeff.size < 3:
        return 0.0
    # n'' = d²n/du² / scale², d²/du² (c3 u³ + c2 u² + …) = 6 c3 u + 2 c2.
    c3 = float(coeff[0]) if coeff.size >= 4 else 0.0
    c2 = float(coeff[-3])
    uu = np.asarray(u, dtype=np.float64)
    return float(np.max(np.abs(6.0 * c3 * uu + 2.0 * c2)) / (scale * scale))


def _fit_coeff(u: np.ndarray, n: np.ndarray, deg: int, scale: float) -> np.ndarray:
    """Старшая степень, которая не нарушает радиус и длину переходной."""
    while deg >= 1:
        coeff = np.polyfit(u, n, deg)
        if deg >= 3:
            c3_phys = float(coeff[0]) / scale**3
            if abs(c3_phys) <= MAX_C3 and _kappa_peak(coeff, u, scale) <= MAX_KAPPA:
                return coeff
            deg = 2
            continue
        if deg == 2:
            if _kappa_peak(coeff, u, scale) <= MAX_KAPPA:
                return coeff
            deg = 1
            continue
        return coeff
    return np.polyfit(u, n, 1)


def _fit_axis(marks: list[RailMark]) -> _Axis:
    """МНК по всем парам. Степень растёт с числом точек, точки не выбрасываются."""
    s = np.array([mark.s for mark in marks], dtype=np.float64)
    n = np.array([mark.n for mark in marks], dtype=np.float64)
    s_ref = float(np.mean(s))
    scale = float(np.std(s))
    if scale < 1.0:
        scale = 1.0
    u = (s - s_ref) / scale
    if len(marks) >= CUBIC_FROM:
        deg = 3
    elif len(marks) >= AXIS_FROM:
        deg = 2
    else:
        deg = 1
    return _Axis(_fit_coeff(u, n, deg, scale), s_ref, scale)


def _constant_axis(n_center: float, s_ref: float) -> _Axis:
    return _Axis(np.array([n_center], dtype=np.float64), s_ref, 1.0)


def _map_band(band: BandEvidence, n_mid: float, cos_psi: float) -> BandEvidence:
    """Сетка отклика из координат «поперёк рельса» обратно в n сенсора."""
    return BandEvidence(
        s0=band.s0,
        s1=band.s1,
        s_mid=band.s_mid,
        grid=n_mid + band.grid * cos_psi,
        score=band.score,
        head=band.head,
        rise=band.rise,
        dark=band.dark,
        u_abs=band.u_abs,
        n_points=band.n_points,
    )


def _search_band(
    xyz: np.ndarray,
    intensity: np.ndarray,
    z_floor: float,
    s0: float,
    s1: float,
    axis: _Axis,
    psi: float,
    half: float,
    by_s: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[RailMark | None, BandEvidence | None]:
    """Один шаблон в осях рельса: вдоль касательной и поперёк неё.

    `by_s` — (целые метры s по возрастанию, порядок точек): тогда считается только полоса по s,
    куда точки шаблона заведомо попадают (|Δs| ≤ |вдоль| + |поперёк|), в прежнем порядке.
    """
    s_mid = 0.5 * (s0 + s1)
    half_len = 0.5 * (s1 - s0)
    cos_psi = max(float(np.cos(psi)), COS_MIN)
    sin_psi = float(np.sin(psi))
    reach = half + 2.2
    n_mid = float(np.asarray(axis.n(s_mid)).reshape(-1)[0])

    def rotate(pts: np.ndarray):
        ds = -pts[:, 1] - s_mid
        dn = pts[:, 0] - n_mid
        # Поворот в касательную: v — поперёк рельса, along — вдоль.
        perp = dn * cos_psi - ds * sin_psi
        along = ds * cos_psi + dn * sin_psi
        return perp, along, (np.abs(along) <= half_len + 0.4) & (np.abs(perp) < reach)

    if by_s is not None:
        bound = half_len + 0.4 + reach + 0.5
        s_bin, order = by_s
        lo = np.searchsorted(s_bin, np.floor(s_mid - bound), side="left")
        hi = np.searchsorted(s_bin, np.floor(s_mid + bound), side="right")
        window = order[lo:hi]
        _, _, sel_w = rotate(xyz[window])
        idx = np.sort(window[sel_w])
        xyz, intensity = xyz[idx], intensity[idx]
    perp, along, sel = rotate(xyz)
    if int(sel.sum()) < 12:
        return None, None
    xyz_l = np.column_stack([perp[sel], -(s_mid + along[sel]), xyz[sel, 2]])
    head_min, ring_min = count_limits(s_mid)
    evidence = rail_evidence(
        xyz_l,
        intensity[sel],
        z_floor,
        np.zeros(3, dtype=np.float64),
        np.array([half], dtype=np.float64),
        ((s_mid - half_len, s_mid + half_len),),
        head_min=head_min,
        ring_min=ring_min,
        geo_floor=GEO_FLOOR,
    )
    band = evidence[0]
    mapped = _map_band(band, n_mid, cos_psi)
    peak = best_peak(band, 0.0, half)
    if peak is None or peak.score < SCORE_MIN or peak.prominence < PROMINENCE_MIN:
        return None, mapped
    # База проверяется поперёк рельса: в координатах сенсора она растёт как 1/cos(ψ).
    if not (SPACING_LIMITS[0] < peak.spacing < SPACING_LIMITS[1]):
        return None, mapped
    if s_mid >= FAR_BOTH_S and (peak.rise < RISE_FLOOR or peak.dark < DARK_FLOOR):
        return None, mapped
    z_head = z_floor + (peak.u_head if np.isfinite(peak.u_head) else RISE_REF)
    mark = RailMark(
        s0=float(s0),
        s1=float(s1),
        s=float(s_mid - peak.n * sin_psi),
        n=float(n_mid + peak.n * cos_psi),
        n_left=float(n_mid + peak.n_left * cos_psi),
        n_right=float(n_mid + peak.n_right * cos_psi),
        z=float(z_head),
        score=float(peak.score),
        rise=float(peak.rise),
        dark=float(peak.dark),
        prominence=float(peak.prominence),
        spacing=float(peak.spacing),
        psi=float(psi),
    )
    return mark, mapped


def _half_width(ds: float, gap_m: float, sigma: float | None) -> float:
    """Полуширина окна поперёк оси, в метрах.

    Основа — допуск либо 2σ фильтра, что больше. Дыра расширяет окно на 2 см
    за каждый пустой метр и поднимает потолок, пока пара снова не найдена.
    """
    base = GATE_HALF
    cap = GATE_CAP
    if sigma is not None and np.isfinite(sigma):
        base = max(base, min(2.0 * float(sigma), GATE_CAP))
    if gap_m > 0.0:
        cap = GATE_CAP_GAP
    return min(cap, base + GATE_PER_M * abs(ds) + GAP_GROW_PER_M * gap_m)


def _gate(
    marks: list[RailMark],
    s_mid: float,
    gap_m: float,
    prior: object | None,
) -> tuple[_Axis, float, float]:
    """Ось, курс шаблона и полуширина окна.

    Пока в кадре меньше двух пар, запертый фильтр задаёт и центр, и курс.
    Дальше пары этого кадра собирают свой полином, а ширина окна остаётся от σ.
    """
    locked = prior is not None and bool(getattr(prior, "locked", False))
    sigma = float(prior.sigma(s_mid)) if locked else None
    if not marks:
        if locked:
            half = _half_width(0.0, gap_m, sigma)
            return _constant_axis(float(prior.n(s_mid)), s_mid), float(prior.psi(s_mid)), half
        return _constant_axis(0.0, s_mid), 0.0, FIRST_HALF
    prev = marks[-1]
    half = _half_width(s_mid - prev.s, gap_m, sigma)
    if len(marks) == 1:
        psi = float(prior.psi(s_mid)) if locked else 0.0
        return _constant_axis(prev.n, s_mid), psi, half
    axis = _fit_axis(marks)
    return axis, axis.psi(s_mid), half


def detect_rails(
    xyz: np.ndarray,
    intensity: np.ndarray,
    prior: object | None = None,
    predict_m: float | None = None,
) -> RailFrame:
    """Цепочка пар от поезда до `S_MAX`.

    `prior` — запертый фильтр прошлого кадра (`n`, `psi`, `sigma`, `locked`).
    Без него кадр ищется сам по себе, как раньше.
    `predict_m` продлевает нарисованную ось полиномом пар до этой дальности.
    Поиск пар от него не зависит.
    """
    started = time.perf_counter()
    xyz = np.asarray(xyz, dtype=np.float64)
    intensity = np.asarray(intensity)
    if xyz.ndim != 2 or xyz.shape[0] < 30 or xyz.shape[1] < 3:
        return RailFrame(z_floor=-1.4, marks=[], evidence=[], ms=0.0)
    z_floor = estimate_floor_z(xyz)
    # Точки, которые могут попасть хоть в одну полосу (см. `_search_band`), по возрастанию s.
    # Пустые лучи (0, 0, 0) не берём: у Hesai без отражения их больше половины кадра.
    s_all = -xyz[:, 1]
    reach_max = max(0.5 * (b1 - b0) for b0, b1, _ in TEMPLATE_BANDS) + 0.4 + max(FIRST_HALF, GATE_CAP_GAP) + 2.2 + 0.5
    near = np.flatnonzero(
        (s_all > TEMPLATE_BANDS[0][0] - reach_max) & (s_all < TEMPLATE_BANDS[-1][1] + reach_max) & xyz.any(axis=1)
    )
    # Порядок по целым метрам: поразрядная сортировка int16, окно полосы — надмножество точного.
    s_bin = np.floor(s_all[near]).astype(np.int16)
    order = near[np.argsort(s_bin, kind="stable")]
    by_s = (np.sort(s_bin, kind="stable"), order)
    marks: list[RailMark] = []
    evidence: list[BandEvidence] = []
    gap = 0.0
    locked = prior is not None and bool(getattr(prior, "locked", False))
    for s0, s1, step in TEMPLATE_BANDS:
        s_mid = 0.5 * (s0 + s1)
        if not marks and s_mid > START_S_MAX and not locked:
            break
        axis, psi, half = _gate(marks, s_mid, gap, prior if locked else None)
        mark, band = _search_band(xyz, intensity, z_floor, s0, s1, axis, psi, half, by_s)
        if band is not None:
            evidence.append(band)
        if mark is None:
            if marks or locked:
                gap += step
                if gap > GAP_BUDGET:
                    break
            continue
        gap = 0.0
        marks.append(mark)
    axis_s = None
    axis_n = None
    if len(marks) >= 2:
        axis = _fit_axis(marks)
        for mark in marks:
            mark.psi = axis.psi(mark.s)
        s_lo = float(marks[0].s)
        s_hi = float(marks[-1].s)
        if predict_m is not None and np.isfinite(predict_m):
            s_hi = max(float(predict_m), s_lo)
        axis_s = np.linspace(s_lo, s_hi, max(int(round((s_hi - s_lo) / 0.5)), 2))
        axis_n = np.asarray(axis.n(axis_s), dtype=np.float64)
    return RailFrame(
        z_floor=float(z_floor),
        marks=marks,
        evidence=evidence,
        ms=(time.perf_counter() - started) * 1e3,
        axis_s=axis_s,
        axis_n=axis_n,
    )
