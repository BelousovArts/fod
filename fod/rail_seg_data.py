"""Данные для сегментации рельсов по range image: изображение кадра и разметка по эталону пути.

Изображение — родные кольца Pandar128 (128 строк, сверху вниз) × передний сектор по
азимуту с шагом 0.1° (два возврата луча сведены к ближнему, строки выровнены по
азимутальному офсету кольца). В пикселе — дальность и интенсивность точки после
компенсации движения; координаты точки восстанавливаются по углам кольца и колонки.

Разметка не зависит от детектора. Эталон оси — будущая траектория лидара по позам
карты в пределах участка записи без обрыва плюс сдвиг лидара от оси (по КР).
Головки — по обе стороны оси на `half` м, высота — траектория плюс `zh` (уточняется
в каждом кадре по ближней зоне, там же — боковой сдвиг эталона `delta`).

Классы: 0 — фон, 1 — левая головка (+n), 2 — правая (−n), 3 — КР, 255 — не учитывать
(полоса неуверенности вокруг головок и всё дальше, чем известен эталон).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RINGS = 128
AZ_STEP = 0.1
SECTOR = (-150.0, -30.0)
WIDTH = int(round((SECTOR[1] - SECTOR[0]) / AZ_STEP))
IGNORE = 255
BG, LEFT, RIGHT, CR = 0, 1, 2, 3
# Центры головок при колее 1520 мм и головке 72 мм.
HALF = 0.796


@dataclass
class RingImage:
    range: np.ndarray       # (128, W) float32, 0 — нет возврата
    intensity: np.ndarray   # (128, W) uint8
    index: np.ndarray       # (128, W) int32 — индекс точки в облаке, −1 — нет


def _nanmedian_rows(a: np.ndarray) -> np.ndarray:
    """`np.nanmedian(a, axis=1)` без предупреждений и втрое быстрее; пустая строка — NaN."""
    srt = np.sort(a, axis=1)
    k = np.sum(np.isfinite(a), axis=1)
    rows = np.arange(a.shape[0])
    med = 0.5 * (srt[rows, np.maximum((k - 1) // 2, 0)] + srt[rows, k // 2])
    return np.where(k > 0, med, np.nan)


def ring_image(xyz: np.ndarray, intensity: np.ndarray) -> RingImage:
    """Организованное облако (колонка·128 + кольцо, пары колонок — два возврата) → изображение сектора."""
    from fod.pandar128 import AZ_OFFSET_DEG

    n = xyz.shape[0]
    cols = n // N_RINGS
    rng = np.sqrt(np.einsum("ij,ij->i", xyz, xyz)).reshape(cols, N_RINGS)
    ok = rng > 0.3
    a, b = rng[0::2], rng[1::2]
    ok_a, ok_b = ok[0::2], ok[1::2]
    pick_b = (ok_b & ~ok_a) | (ok_a & ok_b & (b < a))
    first = np.arange(cols // 2)[:, None] * (2 * N_RINGS) + np.arange(N_RINGS)[None, :]
    sel = first + np.where(pick_b, N_RINGS, 0)
    sel = np.where(ok_a | ok_b, sel, -1)                              # (cols/2, 128)
    p = xyz[np.maximum(sel, 0)]
    az = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    # Азимут колонки — по кольцам с возвратом, с поправкой на офсет кольца (Hesai: по часовой).
    az_col = np.where(sel >= 0, az + AZ_OFFSET_DEG[None, :], np.nan)
    az_c = _nanmedian_rows(az_col)
    good = np.isfinite(az_c)
    j = np.arange(az_c.size)
    az_c = np.interp(j, j[good], az_c[good]) if good.any() else np.zeros(az_c.size)
    c = np.rint((SECTOR[1] - (az_c[:, None] - AZ_OFFSET_DEG[None, :])) / AZ_STEP).astype(np.int64)
    col_i, ring_i = np.nonzero((c >= 0) & (c < WIDTH) & (sel >= 0))
    out_idx = np.full((N_RINGS, WIDTH), -1, np.int32)
    out_idx[ring_i, c[col_i, ring_i]] = sel[col_i, ring_i]
    # Разреженные кольца стреляют через колонку: одиночную дыру между двумя точками — соседом.
    left, right = np.roll(out_idx, 1, axis=1), np.roll(out_idx, -1, axis=1)
    hole = (out_idx < 0) & (left >= 0) & (right >= 0)
    hole[:, 0] = hole[:, -1] = False
    out_idx[hole] = left[hole]
    have = out_idx >= 0
    r_img = np.zeros((N_RINGS, WIDTH), np.float32)
    r_img[have] = rng.reshape(-1)[out_idx[have]]
    i_img = np.zeros((N_RINGS, WIDTH), np.uint8)
    i_img[have] = np.clip(intensity[out_idx[have]], 0, 255).astype(np.uint8)
    return RingImage(r_img, i_img, out_idx)


@dataclass
class Truth:
    s: np.ndarray
    n: np.ndarray           # ось пути
    z: np.ndarray           # высота лидара по пути; головки — z + zh
    end: float


def truth_at(poses: np.ndarray, i: int, seg_end: int, offset: float, horizon: int = 600) -> Truth | None:
    """Будущая траектория лидара в кадре i, не дальше конца участка без обрыва."""
    inv = np.linalg.inv(poses[i])
    q = poses[i : min(seg_end, i + horizon), :3, 3] @ inv[:3, :3].T + inv[:3, 3]
    s, n, z = -q[:, 1], q[:, 0], q[:, 2]
    keep = np.concatenate([[True], np.diff(s) > 0.05])
    s, n, z = s[keep], n[keep], z[keep]
    if s.size < 5 or s[-1] < 20.0 or np.any(np.diff(s) <= 0):
        return None
    return Truth(s, n + offset, z, float(s[-1]))


BANDS = ((5.0, 25.0), (20.0, 40.0), (35.0, 55.0), (50.0, 70.0), (65.0, 85.0))


def _search(dn, dz, half, lo, hi, d_max, s_mid):
    """Лучшие (delta, zh): окна вокруг обеих головок, вблизи ±2.5 см поперёк и ±3 см по высоте.

    Вдали окно шире: колонки на 0.1° расходятся на 10 см к 60 м.
    """
    cell = 0.01
    ne = np.arange(-1.2, 1.2 + 1e-9, cell)
    ze = np.arange(lo, hi + 1e-9, cell)
    h, _, _ = np.histogram2d(dn, dz, bins=(ne, ze))
    cs = np.pad(h.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def box(i0, i1, j0, j1):
        return cs[i1, j1] - cs[i0, j1] - cs[i1, j0] + cs[i0, j0]

    wn = max(3, int(round(0.0009 * s_mid / cell)))
    wz = max(3, int(round(0.0006 * s_mid / cell)))
    best = (-1.0, 0.0, 0.0, 0.0, 0.0)
    zc = np.arange(wz, ze.size - 1 - wz)
    if not zc.size:
        return best
    for d in np.arange(-d_max, d_max + 1e-6, cell):
        il = int(round((half + d + 1.2) / cell))
        ir = int(round((-half + d + 1.2) / cell))
        cl = box(il - wn, il + wn, zc - wz, zc + wz)
        cr = box(ir - wn, ir + wn, zc - wz, zc + wz)
        tot = cl + cr
        k = int(np.argmax(tot))
        if tot[k] > best[0]:
            best = (float(tot[k]), float(d), float(ze[zc[k]] + 0.5 * cell), float(cl[k]), float(cr[k]))
    return best


def refine_bands(xyz: np.ndarray, truth: Truth, half: float = HALF, zh0: float | None = None,
                 d_max: float = 0.15, min_pts: int = 8):
    """Сдвиг эталона поперёк и высота головок над траекторией по полосам дальности.

    Первая полоса — обязательная (иначе None), дальние берутся, если на обеих головках
    вместе не меньше 2·`min_pts` точек; поиск в полосе — вокруг значений прошлой полосы.
    Возвращает массив строк (s центра, delta, zh, точек слева, точек справа).
    """
    s = -xyz[:, 1]
    slope = np.gradient(truth.n, truth.s)
    rows = []
    prev_d, prev_z = 0.0, zh0
    for k, (a, b) in enumerate(BANDS):
        b = min(b, truth.end)
        if b - a < 10.0:
            break
        m = (s > a) & (s < b)
        sm = s[m]
        cos = 1.0 / np.sqrt(1.0 + np.interp(sm, truth.s, slope) ** 2)
        dn = (xyz[m, 0] - np.interp(sm, truth.s, truth.n)) * cos
        dz = xyz[m, 2] - np.interp(sm, truth.s, truth.z)
        if prev_z is None:
            lo, hi = -3.0, 0.0
        else:
            lo, hi = prev_z - (0.3 if k == 0 else 0.12), prev_z + (0.3 if k == 0 else 0.12)
        sel = (np.abs(np.abs(dn - prev_d) - half) < 0.25) & (dz > lo) & (dz < hi)
        if int(sel.sum()) < 2 * min_pts:
            if k == 0:
                return None
            continue
        dm = d_max if k == 0 else 0.06
        _, d, zh, cl, cr = _search(dn[sel] - prev_d, dz[sel], half, lo, hi, dm, 0.5 * (a + b))
        d += prev_d
        # Вдали точки часто только на одной головке — её хватает, колея известна.
        weak = min(cl, cr) < MIN_NEAR if k == 0 else cl + cr < 2 * min_pts
        if weak or abs(d - prev_d) >= dm - 1e-6:
            if k == 0:
                return np.array([[0.5 * (a + b), d, zh, cl, cr]])
            continue
        rows.append((0.5 * (a + b), d, zh, cl, cr))
        prev_d, prev_z = d, zh
    return np.array(rows) if rows else None


MIN_NEAR = 20


def label_image(img: RingImage, xyz: np.ndarray, truth: Truth, bands: np.ndarray,
                cr_n: np.ndarray | None = None, cr_z: np.ndarray | None = None, cr_grid: np.ndarray | None = None,
                half: float = HALF) -> np.ndarray:
    """Классы пикселей по эталону с поправками полос `bands` (из `refine_bands`).

    `cr_n`, `cr_z` — (2, G) линии КР кадра на `cr_grid`, NaN — нет.
    """
    lab = np.zeros(img.index.shape, np.uint8)
    have = img.index >= 0
    p = xyz[img.index[have]].astype(np.float64)
    s = -p[:, 1]
    out = np.zeros(p.shape[0], np.uint8)
    inside = (s > 1.0) & (s <= truth.end)
    slope = np.gradient(truth.n, truth.s)
    sc = np.clip(s, truth.s[0], truth.s[-1])
    cos = 1.0 / np.sqrt(1.0 + np.interp(sc, truth.s, slope) ** 2)
    delta = np.interp(s, bands[:, 0], bands[:, 1])
    zh = np.interp(s, bands[:, 0], bands[:, 2])
    dn = (p[:, 0] - np.interp(sc, truth.s, truth.n) - delta) * cos
    dz = p[:, 2] - (np.interp(sc, truth.s, truth.z) + zh)
    sp = np.maximum(s, 0.0)
    tol_n = 0.04 + 0.001 * sp
    tol_z = 0.04 + 0.0006 * sp
    for side, cls in ((1.0, LEFT), (-1.0, RIGHT)):
        en = np.abs(dn - side * half) / tol_n
        ez = np.abs(dz) / tol_z
        e = np.maximum(en, ez)
        out[inside & (e <= 1.0)] = cls
        out[inside & (e > 1.0) & (e <= 2.0) & (out == 0)] = IGNORE
    if cr_n is not None:
        for k in range(2):
            v = np.isfinite(cr_n[k])
            if v.sum() < 5:
                continue
            g = cr_grid[v]
            on = (s >= g[0]) & (s <= min(g[-1], truth.end))
            en = np.abs(p[:, 0] - np.interp(s, g, cr_n[k][v])) / (0.10 + 0.001 * sp)
            ez = np.abs(p[:, 2] - np.interp(s, g, cr_z[k][v])) / (0.12 + 0.001 * sp)
            e = np.maximum(en, ez)
            out[on & (e <= 1.0) & (out == 0)] = CR
            out[on & (e > 1.0) & (e <= 1.6) & (out == 0)] = IGNORE
    out[s > truth.end] = IGNORE
    lab[have] = out
    return lab
