"""Улики о рельсах: согласованный фильтр «пара головок» по полосам дальности.

Измерено `scripts/probe_rails.py` на 4 бэгах × 3 кадра:

* возвраты от головок стоят на **1.56…1.66 м** центр-в-центр (колея 1.52 м по
  внутренним граням + ширина головки) → шаблон 1.60 м, точное значение
  доизмеряется на кадре (`PeakHit.n_left/n_right`);
* головка поднята над окружением на **0.09…0.18 м**, сигнал слабеет после 30 м;
* интенсивность на головке ниже медианы полосы на **0.5…1.0** — признак сильнее
  геометрии и доживает до 50…65 м, где подъём уже не разрешается.

Отсюда веса шаблона: вблизи решает геометрия, дальше — темнота головок.

**Профиль строится в координатах, привязанных к предсказанной оси.** Иначе на
кривой ось уходит вбок внутри самой полосы (при R = 600 м это 0.72 м на полосе
38…48 м и 1.08 м на 48…60 м), головка шириной 7 см размазывается по два десятка
бинов и отклик разваливается — ровно там, где кривизну и надо измерять.

Модуль только считает отклик; решение о том, какой пик принять, за трекером
(`fod/track_filter.py`), потому что без временного гейта на двухпутном участке
фильтр с равным успехом ловит соседний путь.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fod.cloud import MIN_RANGE

GAUGE = 1.52
RAIL_SPACING = 1.60
HALF_SPACING = RAIL_SPACING / 2.0
SPACING_LIMITS = (1.45, 1.80)
GAUGE_HALF = 1.30

BIN_N = 0.05
RISE_REF = 0.14
DARK_REF = 0.60
SINGLE_HEAD_PENALTY = 0.75   # видна одна головка из двух — улик вдвое меньше
U_LO = -0.40
U_HI = 0.60

RAIL_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 17.0),
    (17.0, 23.0),
    (23.0, 30.0),
    (30.0, 38.0),
    (38.0, 48.0),
    (48.0, 62.0),
    (62.0, 78.0),
)

TUNNEL_BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 30.0),
    (30.0, 45.0),
    (45.0, 60.0),
    (60.0, 80.0),
    (80.0, 105.0),
    (105.0, 135.0),
    (135.0, 175.0),
)

MIN_TUNNEL_WIDTH = 2.8
MAX_TUNNEL_WIDTH = 26.0


@dataclass
class BandEvidence:
    """Отклик шаблона «две головки» по поперечному смещению в одной полосе."""

    s0: float
    s1: float
    s_mid: float
    grid: np.ndarray       # (K,) положение оси на s_mid, м
    score: np.ndarray      # (K,) парный отклик 0…1.2, NaN там, где нет данных
    head: np.ndarray       # (K,) отклик одной головки — для доводки её положения
    rise: np.ndarray       # (K,) подъём слабейшей из головок, м
    dark: np.ndarray       # (K,) относительный провал интенсивности слабейшей
    u_abs: np.ndarray      # (K,) высота поверхности над полом, м
    n_points: int


@dataclass
class TunnelHit:
    """Сечение тоннеля в полосе: обе кромки по отдельности, а не только центр.

    На кривой внутренняя стенка перекрывает обзор уже через десятки метров, а
    внешняя видна далеко. Центр «того, что видно» уезжает наружу поворота и
    распрямляет путь — измерено: объявленный радиус 1269 м против реальных 580.
    Поэтому кромки хранятся раздельно, и трекер решает, какой из них верить.
    """

    s0: float
    s1: float
    s_mid: float
    left: float | None
    right: float | None
    width: float
    n_points: int

    @property
    def center(self) -> float | None:
        if self.left is None or self.right is None:
            return None
        return 0.5 * (self.left + self.right)


@dataclass
class PeakHit:
    n: float
    score: float
    rise: float
    dark: float
    prominence: float
    n_left: float
    n_right: float
    u_head: float

    @property
    def spacing(self) -> float:
        return self.n_right - self.n_left


def _ring_kernel(inner: int, outer: int) -> np.ndarray:
    kernel = np.zeros(2 * outer + 1, dtype=np.float64)
    kernel[: outer - inner] = 1.0
    kernel[outer + inner + 1 :] = 1.0
    return kernel


_RING_OUTER = 8
_HEAD_KERNEL = np.ones(3, dtype=np.float64)                    # ±0.05 м — ширина головки с запасом
_RING_KERNEL = _ring_kernel(inner=3, outer=_RING_OUTER)        # 0.20…0.40 м по обе стороны


def axis_at(coeff: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Полином оси произвольной степени по схеме Горнера."""
    out = np.zeros_like(np.asarray(s, dtype=np.float64))
    for c in reversed(np.asarray(coeff, dtype=np.float64)):
        out = out * s + c
    return out


def _pair_score(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Отклик пары головок, различающий «не видно» и «видно, но не рельс».

    Минимум по паре — правильная защита от одиночной тёмной линии: кромки лотка,
    основания стены, кабельного канала. Но он же обнулял гипотезу, когда во вторую
    головку просто не попал ни один луч — а с дальностью это происходит всё чаще.
    Неподтверждённая сторона теперь нейтральна (с платой за половину улик),
    противоречащая по-прежнему убивает гипотезу.
    """
    seen_left = np.isfinite(left)
    seen_right = np.isfinite(right)
    both = seen_left & seen_right
    single = seen_left ^ seen_right
    out = np.full(left.shape, np.nan)
    out[both] = np.minimum(left[both], right[both])
    if np.any(single):
        alone = np.where(seen_left, left, right)
        out[single] = SINGLE_HEAD_PENALTY * alone[single]
    return out


def weights_for(s_mid: float) -> tuple[float, float]:
    """Вблизи решает геометрия, дальше — интенсивность (см. замеры в docstring)."""
    t = float(np.clip((s_mid - 22.0) / 16.0, 0.0, 1.0))
    w_geo = 0.62 - 0.32 * t
    return w_geo, 1.0 - w_geo


def rail_evidence(
    xyz: np.ndarray,
    intensity: np.ndarray,
    z_floor: float,
    coeff: np.ndarray,
    half_window: np.ndarray,
    bands: tuple[tuple[float, float], ...] = RAIL_BANDS,
    *,
    head_min: float = 2.0,
    ring_min: float = 6.0,
    geo_floor: float | None = None,
) -> list[BandEvidence]:
    """Отклик шаблона по каждой полосе в окне вокруг предсказанной оси `coeff`.

    Точки разгибаются по предсказанию: биннится не `n`, а «где эта точка была бы
    на `s_mid`, если бы шла вдоль предсказанной оси». `half_window` задаёт трекер
    из своей ковариации — по одному значению на полосу.
    """
    s_all = -xyz[:, 1]
    n_all = xyz[:, 0]
    u_all = xyz[:, 2] - z_floor
    margin = HALF_SPACING + (_RING_OUTER + 1) * BIN_N
    s_lo = float(bands[0][0])
    s_hi = float(bands[-1][1])

    keep = (
        (s_all >= s_lo)
        & (s_all < s_hi)
        & (u_all > U_LO)
        & (u_all < U_HI)
        & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
    )
    s = s_all[keep]
    n = n_all[keep]
    u = u_all[keep]
    it = intensity[keep].astype(np.float64, copy=False)
    n_flat = n - axis_at(coeff, s)

    band_edges = np.array([b[0] for b in bands] + [bands[-1][1]], dtype=np.float64)
    band_idx = np.searchsorted(band_edges, s, side="right") - 1

    out: list[BandEvidence] = []
    for b, (s0, s1) in enumerate(bands):
        s_mid = 0.5 * (s0 + s1)
        n_pred = float(axis_at(coeff, np.array(s_mid)))
        lo = -float(half_window[b]) - margin
        hi = +float(half_window[b]) + margin
        edges = np.arange(lo, hi + BIN_N * 0.5, BIN_N)
        grid = 0.5 * (edges[:-1] + edges[1:]) + n_pred
        k = grid.size
        head = int(round(HALF_SPACING / BIN_N))
        if k < 2 * (head + _RING_OUTER) + 3:
            out.append(_empty_band(s0, s1, s_mid, grid))
            continue

        sel = band_idx == b
        if int(sel.sum()) < 12:
            out.append(_empty_band(s0, s1, s_mid, grid))
            continue
        n_b, u_b, i_b = n_flat[sel], u[sel], it[sel]
        inside = (n_b >= lo) & (n_b < hi)
        if int(inside.sum()) < 12:
            out.append(_empty_band(s0, s1, s_mid, grid))
            continue
        n_b, u_b, i_b = n_b[inside], u_b[inside], i_b[inside]

        idx = np.clip(((n_b - lo) / BIN_N).astype(np.int64), 0, k - 1)
        cnt = np.bincount(idx, minlength=k).astype(np.float64)
        u_sum = np.bincount(idx, weights=u_b, minlength=k)
        i_sum = np.bincount(idx, weights=i_b, minlength=k)
        i_ref = max(float(np.median(i_b)), 1.0)

        c_head = np.convolve(cnt, _HEAD_KERNEL, mode="same")
        c_ring = np.convolve(cnt, _RING_KERNEL, mode="same")
        ok = (c_head >= head_min) & (c_ring >= ring_min)
        blank = np.full(k, np.nan)
        u_head = np.divide(np.convolve(u_sum, _HEAD_KERNEL, "same"), c_head, out=blank.copy(), where=ok)
        u_ring = np.divide(np.convolve(u_sum, _RING_KERNEL, "same"), c_ring, out=blank.copy(), where=ok)
        i_head = np.divide(np.convolve(i_sum, _HEAD_KERNEL, "same"), c_head, out=blank.copy(), where=ok)
        i_ring = np.divide(np.convolve(i_sum, _RING_KERNEL, "same"), c_ring, out=blank.copy(), where=ok)

        rise = u_head - u_ring
        dark = (i_ring - i_head) / i_ref

        # Верх головки, а не среднее по бину: в окно 0.15 м попадают и шпалы, и
        # балласт, из-за чего средняя высота занижена на 3…7 см — при наложении
        # на range image это уводит линию заметно ниже рельса.
        above = u_b > (np.nan_to_num(u_ring[idx], nan=-np.inf) + 0.5 * RISE_REF)
        cnt_top = np.bincount(idx[above], minlength=k).astype(np.float64)
        sum_top = np.bincount(idx[above], weights=u_b[above], minlength=k)
        c_top = np.convolve(cnt_top, _HEAD_KERNEL, mode="same")
        u_top = np.divide(
            np.convolve(sum_top, _HEAD_KERNEL, "same"), c_top, out=u_head.copy(), where=c_top >= 2.0
        )
        w_geo, w_dark = weights_for(s_mid)
        if geo_floor is not None:
            w_geo = max(w_geo, float(geo_floor))
            w_dark = 1.0 - w_geo
        head_score = w_geo * np.clip(rise / RISE_REF, 0.0, 1.2) + w_dark * np.clip(dark / DARK_REF, 0.0, 1.2)

        score = np.full(k, np.nan)
        pair_rise = np.full(k, np.nan)
        pair_dark = np.full(k, np.nan)
        mid = slice(head, k - head)
        left = slice(0, k - 2 * head)
        right = slice(2 * head, k)
        score[mid] = _pair_score(head_score[left], head_score[right])
        pair_rise[mid] = np.fmin(rise[left], rise[right])
        pair_dark[mid] = np.fmin(dark[left], dark[right])

        out.append(
            BandEvidence(
                s0=float(s0),
                s1=float(s1),
                s_mid=float(s_mid),
                grid=grid,
                score=score,
                head=head_score,
                rise=pair_rise,
                dark=pair_dark,
                u_abs=u_top,
                n_points=int(n_b.size),
            )
        )
    return out


def _empty_band(s0: float, s1: float, s_mid: float, grid: np.ndarray) -> BandEvidence:
    nan = np.full(grid.size, np.nan)
    return BandEvidence(
        s0=float(s0),
        s1=float(s1),
        s_mid=float(s_mid),
        grid=grid,
        score=nan.copy(),
        head=nan.copy(),
        rise=nan.copy(),
        dark=nan.copy(),
        u_abs=nan.copy(),
        n_points=0,
    )


def _refine_head(band: BandEvidence, index: int, radius: int = 3) -> tuple[float, float]:
    """Центроид отклика одной головки: истинная база отличается от шаблонной."""
    lo = max(index - radius, 0)
    hi = min(index + radius + 1, band.grid.size)
    weight = np.nan_to_num(band.head[lo:hi], nan=0.0)
    peak = float(weight.max()) if weight.size else 0.0
    if peak <= 0.0:
        return float(band.grid[index]), float("nan")
    weight = np.maximum(weight - 0.5 * peak, 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return float(band.grid[index]), float("nan")
    position = float(np.dot(weight, band.grid[lo:hi]) / total)
    u = band.u_abs[lo:hi]
    good = np.isfinite(u) & (weight > 0.0)
    height = float(np.dot(weight[good], u[good]) / weight[good].sum()) if np.any(good) else float("nan")
    return position, height


def best_peak(band: BandEvidence, n_center: float, half_window: float) -> PeakHit | None:
    """Максимум отклика в окне + запас над ближайшим конкурентом вне ±0.35 м."""
    score = band.score
    if not np.any(np.isfinite(score)):
        return None
    gate = (band.grid >= n_center - half_window) & (band.grid <= n_center + half_window)
    gate &= np.isfinite(score)
    if not np.any(gate):
        return None
    j = int(np.argmax(np.where(gate, score, -np.inf)))
    best = float(score[j])
    if not np.isfinite(best):
        return None
    rival = np.where(gate & (np.abs(band.grid - band.grid[j]) > 0.35), score, -np.inf)
    second = float(np.max(rival))
    prominence = best - second if np.isfinite(second) else best

    offset = int(round(HALF_SPACING / BIN_N))
    n_left, u_left = _refine_head(band, max(j - offset, 0))
    n_right, u_right = _refine_head(band, min(j + offset, band.grid.size - 1))
    heights = [h for h in (u_left, u_right) if np.isfinite(h)]
    return PeakHit(
        n=float(band.grid[j]),
        score=best,
        rise=float(band.rise[j]) if np.isfinite(band.rise[j]) else 0.0,
        dark=float(band.dark[j]) if np.isfinite(band.dark[j]) else 0.0,
        prominence=float(prominence),
        n_left=n_left,
        n_right=n_right,
        u_head=float(np.mean(heights)) if heights else float("nan"),
    )


def _peak_at(band: BandEvidence, j: int, prominence: float) -> PeakHit:
    offset = int(round(HALF_SPACING / BIN_N))
    n_left, u_left = _refine_head(band, max(j - offset, 0))
    n_right, u_right = _refine_head(band, min(j + offset, band.grid.size - 1))
    heights = [h for h in (u_left, u_right) if np.isfinite(h)]
    return PeakHit(
        n=float(band.grid[j]),
        score=float(band.score[j]),
        rise=float(band.rise[j]) if np.isfinite(band.rise[j]) else 0.0,
        dark=float(band.dark[j]) if np.isfinite(band.dark[j]) else 0.0,
        prominence=float(prominence),
        n_left=n_left,
        n_right=n_right,
        u_head=float(np.mean(heights)) if heights else float("nan"),
    )


def find_peaks(
    band: BandEvidence,
    *,
    score_min: float,
    prominence_min: float,
    rival_m: float = 0.35,
    local_m: float = 1.6,
    nms_m: float = 2.0,
) -> list[PeakHit]:
    """Все локальные максимумы пары в полосе, без окна вокруг прогноза оси.

    Запас пика считается по соседству `rival_m…local_m`, а не по всему коридору.
    Второй путь в нескольких метрах — отдельная пара, а не конкурент, которого
    надо задавить одним победителем.
    """
    score = band.score
    finite = np.isfinite(score)
    if int(finite.sum()) < 3:
        return []
    is_peak = np.zeros(score.shape, dtype=bool)
    is_peak[1:-1] = (
        finite[1:-1]
        & finite[:-2]
        & finite[2:]
        & (score[1:-1] >= score[:-2])
        & (score[1:-1] > score[2:])
        & (score[1:-1] >= score_min)
    )
    candidates = np.flatnonzero(is_peak)
    if candidates.size == 0:
        return []
    grid = band.grid
    scored: list[tuple[float, int, float]] = []
    for j in candidates:
        j = int(j)
        dn = np.abs(grid - grid[j])
        rival = finite & (dn > rival_m) & (dn < local_m)
        if np.any(rival):
            prominence = float(score[j] - np.max(score[rival]))
        else:
            prominence = float(score[j])
        if prominence < prominence_min:
            continue
        scored.append((float(score[j]), j, prominence))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[int, float]] = []
    for _value, j, prominence in scored:
        if any(abs(float(grid[j] - grid[k])) < nms_m for k, _p in kept):
            continue
        kept.append((j, prominence))
    return [_peak_at(band, j, prominence) for j, prominence in kept]


def tunnel_hits(
    xyz: np.ndarray,
    z_floor: float,
    coeff: np.ndarray,
    bands: tuple[tuple[float, float], ...] = TUNNEL_BANDS,
) -> list[TunnelHit]:
    """Центр сечения тоннеля по 2-му и 98-му процентилю поперечной координаты.

    Так же, как и рельсы, считается по разогнутым координатам: на кривой полоса
    105…135 м иначе размазывается на шесть метров и «ширина сечения» врёт.
    """
    s_all = -xyz[:, 1]
    u_all = xyz[:, 2] - z_floor
    n_all = xyz[:, 0]
    keep = (
        (s_all >= bands[0][0])
        & (s_all < bands[-1][1])
        & (u_all > 0.30)
        & (u_all < 3.60)
        & (np.abs(n_all) < 14.0)
        & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
    )
    s = s_all[keep]
    n = n_all[keep] - axis_at(coeff, s_all[keep])
    out: list[TunnelHit] = []
    for s0, s1 in bands:
        s_mid = 0.5 * (float(s0) + float(s1))
        n_pred = float(axis_at(coeff, np.array(s_mid)))
        sel = (s >= s0) & (s < s1)
        count = int(sel.sum())
        if count < 80:
            out.append(TunnelHit(float(s0), float(s1), s_mid, None, None, 0.0, count))
            continue
        band_n = n[sel]
        lo, hi = np.percentile(band_n, [2.0, 98.0])
        width = float(hi - lo)
        if not MIN_TUNNEL_WIDTH < width < MAX_TUNNEL_WIDTH:
            out.append(TunnelHit(float(s0), float(s1), s_mid, None, None, width, count))
            continue
        # Кромка засчитывается, только если это действительно стенка, а не край
        # видимой области: у стенки рядом с процентилем набирается масса точек.
        left = float(lo) + n_pred if int(np.sum(np.abs(band_n - lo) < 0.6)) >= 12 else None
        right = float(hi) + n_pred if int(np.sum(np.abs(band_n - hi) < 0.6)) >= 12 else None
        out.append(
            TunnelHit(
                s0=float(s0),
                s1=float(s1),
                s_mid=s_mid,
                left=left,
                right=right,
                width=width,
                n_points=count,
            )
        )
    return out
