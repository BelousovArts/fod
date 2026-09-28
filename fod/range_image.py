"""Сборка range image Pandar128: dual-return, ректификация, угловая сетка."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from fod.cloud import MIN_RANGE, N_RINGS, LidarCloud
from fod.pandar128 import (
    AZ_OFFSET_DEG,
    AZ_STEP_DEG,
    DENSE_ELEV_STEP_DEG,
    ELEVATION_DEG,
    HIGH_RES_RING_FIRST,
    HIGH_RES_RING_LAST,
    elevation_grid_deg,
)

# Максимальный зазор по элевации, который ещё заполняем (чуть больше 0.5°
# разреженной зоны и 1° крайних каналов). Дыры шире — настоящие пропуски.
MAX_ELEV_GAP_DEG = 0.65


@dataclass
class RangeImage:
    range: np.ndarray          # (H, W) метры, NaN = нет возврата
    intensity: np.ndarray      # (H, W)
    elevation_deg: np.ndarray  # (H,)
    azimuth_deg: np.ndarray    # (W,)  номинальный азимут колонки
    wrapped: bool              # 360° (циклический сдвиг) vs сектор


def _as_ring_grid(cloud: LidarCloud) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cloud.n_points
    if n % N_RINGS != 0:
        raise ValueError(f"Число точек {n} не кратно {N_RINGS}")
    n_cols = n // N_RINGS
    rng = cloud.range.reshape(n_cols, N_RINGS).T
    intensity = cloud.intensity.reshape(n_cols, N_RINGS).T.astype(np.float64)
    valid = rng > MIN_RANGE
    rng = np.where(valid, rng.astype(np.float64), np.nan)
    intensity = np.where(valid, intensity, np.nan)
    return rng, intensity, valid


def _is_dual_return(n_cols: int, rng: np.ndarray) -> bool:
    if n_cols % 2 != 0:
        return False
    # Соседние колонки — пара возвратов одного луча: валидность почти совпадает.
    a = np.isfinite(rng[:, 0::2])
    b = np.isfinite(rng[:, 1::2])
    return float(np.mean(a == b)) > 0.9


def collapse_dual_return(rng: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ближний возврат из пары колонок с одним азимутом."""
    near = rng[:, 0::2]
    far = rng[:, 1::2]
    i_near = intensity[:, 0::2]
    i_far = intensity[:, 1::2]
    near_ok = np.isfinite(near)
    far_ok = np.isfinite(far)
    both = near_ok & far_ok
    pick_far = (far_ok & ~near_ok) | (both & (far < near))
    out_r = np.where(pick_far, far, near)
    out_i = np.where(pick_far, i_far, i_near)
    return out_r, out_i


def rectify_azimuth(
    rng: np.ndarray,
    intensity: np.ndarray,
    wrap: bool,
    az_step_deg: float = AZ_STEP_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Сдвигает каждую строку на офсет блока лазеров, чтобы колонка = один азимут.

    Сканирование по часовой: индекс колонки растёт, азимут убывает. Положительный
    офсет Hesai (по часовой) → сдвиг строки вправо.
    """
    n_rings, n_az = rng.shape
    shifts = np.rint(AZ_OFFSET_DEG[:n_rings] / az_step_deg).astype(np.int32)
    out_r = np.full_like(rng, np.nan)
    out_i = np.full_like(intensity, np.nan)
    cols = np.arange(n_az)
    for ring, shift in enumerate(shifts):
        if wrap:
            src = (cols - shift) % n_az
            out_r[ring] = rng[ring, src]
            out_i[ring] = intensity[ring, src]
        else:
            src = cols - shift
            ok = (src >= 0) & (src < n_az)
            out_r[ring, ok] = rng[ring, src[ok]]
            out_i[ring, ok] = intensity[ring, src[ok]]
    return out_r, out_i


def _resample_elevation(
    values: np.ndarray,
    el_src: np.ndarray,
    el_grid: np.ndarray,
    max_gap_deg: float,
    interpolate: bool = True,
) -> np.ndarray:
    """Раскладка колец на равномерную сетку элевации.

    interpolate=True: линейно заполняет зазор между соседними кольцами.
    interpolate=False: только измеренные лучи, пустые строки остаются NaN.
    """
    height = int(el_grid.shape[0])
    n_az = int(values.shape[1])
    step = float(el_grid[0] - el_grid[1])
    rows = np.rint((el_grid[0] - el_src) / step).astype(np.int32)
    rows = np.clip(rows, 0, height - 1)
    out = np.full((height, n_az), np.nan, dtype=np.float64)

    finite = np.isfinite(values)
    for ring in range(values.shape[0]):
        ok = finite[ring]
        if not np.any(ok):
            continue
        out[rows[ring], ok] = values[ring, ok]

    if not interpolate:
        return out

    for i in range(values.shape[0] - 1):
        gap = abs(float(el_src[i] - el_src[i + 1]))
        if gap > max_gap_deg * 2:
            continue
        y0 = int(rows[i])
        y1 = int(rows[i + 1])
        if y1 == y0:
            continue
        if y1 < y0:
            y0, y1 = y1, y0
            v0, v1 = values[i + 1], values[i]
        else:
            v0, v1 = values[i], values[i + 1]
        both = np.isfinite(v0) & np.isfinite(v1)
        if not np.any(both):
            continue
        n = y1 - y0
        t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)[:, None]
        seg = v0[None, :] * (1.0 - t) + v1[None, :] * t
        out[y0 : y1 + 1, both] = seg[:, both]
    return out


def _fill_single_pixel_holes(values: np.ndarray, wrap: bool) -> np.ndarray:
    """Зазор 0.2° по азимуту у разреженных колец — одна пустая колонка."""
    out = values.copy()
    left = np.roll(out, 1, axis=1)
    right = np.roll(out, -1, axis=1)
    hole = ~np.isfinite(out) & np.isfinite(left) & np.isfinite(right)
    if not wrap:
        hole[:, 0] = False
        hole[:, -1] = False
    out[hole] = 0.5 * (left[hole] + right[hole])
    return out


def _azimuth_grid(cloud: LidarCloud, dual: bool) -> np.ndarray:
    n_cols = cloud.n_points // N_RINGS
    xyz = cloud.xyz.reshape(n_cols, N_RINGS, 3)
    valid = (cloud.range.reshape(n_cols, N_RINGS) > MIN_RANGE).T
    az = np.rad2deg(np.arctan2(xyz[:, :, 1], xyz[:, :, 0])).T.astype(np.float64)
    az = np.where(valid, az, np.nan)
    if dual:
        az = az[:, 0::2]
    return az


def crop_forward_sector(image: RangeImage, half_width_deg: float = 60.0) -> RangeImage:
    """Оставить передний сектор вокруг −90° (вперёд), как с камеры: слева — левая сторона пути.

    Облако правое (лидар крутится по часовой сверху, азимут со временем убывает),
    вперёд — −y, значит слева +x, азимут ближе к 0°: колонки по убыванию азимута.
    """
    az = image.azimuth_deg
    center = -90.0
    delta = np.abs(((az - center + 180.0) % 360.0) - 180.0)
    sel = delta <= half_width_deg
    if int(sel.sum()) < 32:
        return image
    order = np.flatnonzero(sel)
    if az[order[0]] < az[order[-1]]:
        order = order[::-1]
    return RangeImage(
        range=image.range[:, order],
        intensity=image.intensity[:, order],
        elevation_deg=image.elevation_deg,
        azimuth_deg=az[order],
        wrapped=False,
    )


def crop_elevation(image: RangeImage, el_min_deg: float, el_max_deg: float) -> RangeImage:
    """Оставить полосу элевации `[el_min, el_max]`, градусы."""
    el = np.asarray(image.elevation_deg, dtype=np.float64)
    sel = (el >= el_min_deg) & (el <= el_max_deg)
    if int(sel.sum()) < 4:
        return image
    return RangeImage(
        range=image.range[sel],
        intensity=image.intensity[sel],
        elevation_deg=el[sel],
        azimuth_deg=image.azimuth_deg,
        wrapped=image.wrapped,
    )


def crop_high_res_band(image: RangeImage) -> RangeImage:
    """Плотная зона Pandar128: rings 25…89, шаг 0.125°."""
    return crop_elevation(
        image,
        float(ELEVATION_DEG[HIGH_RES_RING_LAST]),
        float(ELEVATION_DEG[HIGH_RES_RING_FIRST]),
    )


def _fill_gaps(img: np.ndarray, axis: int, max_gap: int) -> np.ndarray:
    """Дыры до `max_gap` пикселей вдоль `axis` — средним соседей, по одному слою за проход."""
    out = img.copy()
    for _ in range(max_gap):
        a = np.roll(out, 1, axis=axis)
        b = np.roll(out, -1, axis=axis)
        hole = ~np.isfinite(out) & (np.isfinite(a) | np.isfinite(b))
        if not np.any(hole):
            break
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmean(np.stack([a, b]), axis=0)
        out[hole] = fill[hole]
    return out


def range_image_from_points(cloud: LidarCloud, interpolate: bool = True) -> RangeImage:
    """Range image по углам точек, без сетки колец (облака с дописанными/удалёнными точками).

    Строка — ближайшая элевация плотной сетки, колонка — азимут с шагом `AZ_STEP_DEG`
    по убыванию (как у лидара); в ячейке ближняя точка.
    """
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(rng) & (rng > MIN_RANGE)
    xyz, rng, inten = xyz[ok], rng[ok], np.asarray(cloud.intensity, dtype=np.float64)[ok]
    az = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.rad2deg(np.arcsin(np.clip(xyz[:, 2] / rng, -1.0, 1.0)))
    el_grid = elevation_grid_deg(DENSE_ELEV_STEP_DEG)
    step = float(el_grid[0] - el_grid[1])
    rows = np.clip(np.rint((el_grid[0] - el) / step).astype(np.int64), 0, el_grid.size - 1)
    wrap = bool(np.ptp(az) > 300.0) if az.size else False
    az_hi = 180.0 if wrap else (float(az.max()) if az.size else 0.0)
    cols = np.rint((az_hi - az) / AZ_STEP_DEG).astype(np.int64)
    n_az = int(cols.max()) + 1 if cols.size else 1
    rng_img = np.full((el_grid.size, n_az), np.nan)
    int_img = np.full((el_grid.size, n_az), np.nan)
    order = np.argsort(-rng)
    rng_img[rows[order], cols[order]] = rng[order]
    int_img[rows[order], cols[order]] = inten[order]
    if interpolate:
        for img in (rng_img, int_img):
            img[:] = _fill_gaps(_fill_gaps(img, 0, 4), 1, 1)
    return RangeImage(
        range=rng_img,
        intensity=int_img,
        elevation_deg=el_grid,
        azimuth_deg=az_hi - AZ_STEP_DEG * np.arange(n_az, dtype=np.float64),
        wrapped=wrap,
    )


def build_range_image(
    cloud: LidarCloud,
    wrap: bool | None = None,
    interpolate: bool = True,
) -> RangeImage:
    if cloud.n_points % N_RINGS != 0 or not np.any(cloud.ring):
        return range_image_from_points(cloud, interpolate)
    rng, intensity, _valid = _as_ring_grid(cloud)
    n_cols = rng.shape[1]
    dual = _is_dual_return(n_cols, rng)
    if dual:
        rng, intensity = collapse_dual_return(rng, intensity)
    if wrap is None:
        wrap = rng.shape[1] >= 3000
    az = _azimuth_grid(cloud, dual)
    rng, intensity = rectify_azimuth(rng, intensity, wrap=wrap)
    az, _ = rectify_azimuth(az, az, wrap=wrap)
    if interpolate:
        rng = _fill_single_pixel_holes(rng, wrap)
        intensity = _fill_single_pixel_holes(intensity, wrap)
        az = _fill_single_pixel_holes(az, wrap)
    el_grid = elevation_grid_deg(DENSE_ELEV_STEP_DEG)
    rng_img = _resample_elevation(rng, ELEVATION_DEG, el_grid, MAX_ELEV_GAP_DEG, interpolate)
    int_img = _resample_elevation(intensity, ELEVATION_DEG, el_grid, MAX_ELEV_GAP_DEG, interpolate)
    if interpolate:
        rng_img = _fill_single_pixel_holes(rng_img, wrap)
        int_img = _fill_single_pixel_holes(int_img, wrap)
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            az_col = np.nanmedian(az, axis=0)
    if not np.any(np.isfinite(az_col)):
        az_col = _column_azimuth_deg(rng_img.shape[1], wrap)
    else:
        idx = np.arange(az_col.shape[0])
        good = np.isfinite(az_col)
        if not np.all(good):
            az_col = az_col.copy()
            az_col[~good] = np.interp(idx[~good], idx[good], az_col[good])
    return RangeImage(
        range=rng_img,
        intensity=int_img,
        elevation_deg=el_grid,
        azimuth_deg=az_col,
        wrapped=wrap,
    )


def _column_azimuth_deg(n_az: int, wrap: bool) -> np.ndarray:
    if wrap:
        return -AZ_STEP_DEG * np.arange(n_az, dtype=np.float64)
    center = (n_az - 1) / 2.0
    return -90.0 + AZ_STEP_DEG * (np.arange(n_az, dtype=np.float64) - center)
