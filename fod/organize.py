"""Неупорядоченное облако Pandar128 → сетка колонок × 128 колец, как у драйвера.

Тракт ждёт организованное облако: индекс = колонка·128 + кольцо, колонки парами — два
возврата одного луча. Бывает иначе: драйвер выбрасывает пустые лучи, генератор дописывает
точки в конец (`cloud_with_fake_obj`), нет полей `ring` и `timestamp`. Тогда:

* кольцо — по элевации точки. У кольца она постоянна, поэтому элевации колец берутся
  с самого облака (128 самых населённых значений); паспорт (`fod/pandar128.py`)
  отличается от них до 0,12° при шаге колец 0,125° и годится только как запасной;
* колонка — по азимуту с поправкой на офсет кольца, шаг азимута — по данным, отсчёт
  от первой точки сообщения (лидар крутится по часовой, азимут со временем убывает);
  пустые азимуты сетки выбрасываются;
* два возврата в одной ячейке: ближний — в чётную колонку пары, дальний — в нечётную;
* время точки — из поля `timestamp`, если оно есть, иначе по азимуту за оборот 0,1 с.
"""

from __future__ import annotations

import numpy as np

from fod.cloud import MIN_RANGE, N_RINGS, LidarCloud

SWEEP_S = 0.1
DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2"), ("timestamp", "<f8")])


def _elevation(xyz: np.ndarray, rng: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(rng, 1e-9), -1.0, 1.0)))


def is_organized(cloud: LidarCloud) -> bool:
    n = cloud.n_points
    if n == 0 or n % N_RINGS:
        return False
    if "ring" in cloud.field_offsets:
        head = cloud.ring[: 4 * N_RINGS]
        return bool(np.all(head == np.tile(np.arange(N_RINGS), head.size // N_RINGS)))
    # Без поля ring: у организованного облака элевация в каждом кольце почти одна.
    xyz = cloud.xyz.reshape(-1, N_RINGS, 3)[:: max(1, n // N_RINGS // 200)].astype(np.float64)
    r = np.linalg.norm(xyz, axis=2)
    el = np.degrees(np.arcsin(np.clip(xyz[..., 2] / np.maximum(r, 1e-9), -1.0, 1.0)))
    el[r < 1.0] = np.nan
    if np.sum(np.isfinite(el)) < 1000:
        return True
    seen = np.isfinite(el).any(axis=0)
    el = el[:, seen]
    dev = np.abs(el - np.nanmedian(el, axis=0))
    return float(np.mean(dev[np.isfinite(el)] < 0.05)) > 0.95


def calibrate_rings(el: np.ndarray) -> np.ndarray | None:
    """Элевации 128 колец по точкам кадра, сверху вниз; None — не похоже на 128 колец."""
    srt = np.sort(el)
    cut = np.flatnonzero(np.diff(srt) > 0.02) + 1
    groups = np.split(srt, cut)
    if len(groups) < N_RINGS:
        return None
    sizes = np.array([g.size for g in groups])
    top = np.sort(np.argsort(-sizes)[:N_RINGS])
    if sizes[top].min() < 20:
        return None
    return np.array([float(np.median(groups[k])) for k in top])[::-1]


class Organizer:
    """Состояние — калибровка колец и шаг азимута, найденные по первому подходящему кадру."""

    def __init__(self) -> None:
        self.elev: np.ndarray | None = None
        self.az_step: float | None = None

    def __call__(self, cloud: LidarCloud) -> LidarCloud:
        if is_organized(cloud):
            return cloud
        return self.organize(cloud)

    def organize(self, cloud: LidarCloud) -> LidarCloud:
        from fod.pandar128 import AZ_OFFSET_DEG, ELEVATION_DEG

        xyz = cloud.xyz.astype(np.float64)
        rng = np.linalg.norm(xyz, axis=1)
        keep = np.flatnonzero(np.isfinite(rng) & (rng > MIN_RANGE))
        xyz, rng = xyz[keep], rng[keep]
        el = _elevation(xyz, rng)
        if self.elev is None:
            self.elev = calibrate_rings(el[rng > 1.0])
        table = self.elev if self.elev is not None else ELEVATION_DEG
        order = np.argsort(table)
        srt = table[order]
        j = np.clip(np.searchsorted(srt, el), 1, srt.size - 1)
        j -= (el - srt[j - 1]) < (srt[j] - el)
        ring = order[j]

        nominal = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) + AZ_OFFSET_DEG[ring]
        if self.az_step is None:
            self.az_step = _azimuth_step(nominal, ring)
        step = self.az_step
        n_az = int(round(360.0 / step))
        a0 = nominal[0] if nominal.size else 0.0
        col = np.rint(((a0 - nominal) % 360.0) / step).astype(np.int64) % n_az
        # Только занятые азимуты: лидар не обязательно стреляет на каждом шаге сетки.
        used, col = np.unique(col, return_inverse=True)
        n_az_used = used.size

        cell = col * N_RINGS + ring
        by = np.lexsort((rng, cell))
        cell_s = cell[by]
        first = np.concatenate([[True], cell_s[1:] != cell_s[:-1]])
        start = np.maximum.accumulate(np.where(first, np.arange(cell_s.size), 0))
        rank = np.arange(cell_s.size) - start
        use = rank < 2
        src = by[use]
        dst = (2 * col[src] + rank[use]) * N_RINGS + ring[src]

        total = 2 * n_az_used * N_RINGS
        out = np.zeros(total, dtype=DTYPE)
        idx = keep[src]
        out["x"][dst], out["y"][dst], out["z"][dst] = cloud.xyz[idx, 0], cloud.xyz[idx, 1], cloud.xyz[idx, 2]
        out["intensity"][dst] = cloud.intensity[idx]
        out["ring"] = np.tile(np.arange(N_RINGS, dtype=np.uint16), total // N_RINGS)
        stamp = float(cloud.stamp_sec) + 1e-9 * float(cloud.stamp_nsec)
        cols = np.arange(total) // N_RINGS // 2
        out["timestamp"] = stamp + SWEEP_S * used[cols] / n_az
        if "timestamp" in cloud.field_offsets and np.any(cloud.timestamp):
            out["timestamp"][dst] = cloud.timestamp[idx]
        raw = out.view(np.uint8).reshape(total, DTYPE.itemsize)
        return LidarCloud(
            xyz=np.stack([out["x"], out["y"], out["z"]], axis=1),
            intensity=out["intensity"].copy(),
            ring=out["ring"].copy(),
            timestamp=out["timestamp"].copy(),
            width=total,
            height=1,
            is_dense=False,
            point_step=DTYPE.itemsize,
            frame_id=cloud.frame_id,
            stamp_sec=cloud.stamp_sec,
            stamp_nsec=cloud.stamp_nsec,
            raw=raw,
            field_offsets={name: (DTYPE.fields[name][1], DTYPE.fields[name][0]) for name in DTYPE.names},
        )


def _azimuth_step(nominal: np.ndarray, ring: np.ndarray) -> float:
    """Шаг азимута по самому населённому кольцу; округление до 0,01°."""
    if ring.size == 0:
        return 0.2
    k = int(np.bincount(ring, minlength=N_RINGS).argmax())
    a = np.unique(np.round(np.sort(nominal[ring == k] % 360.0), 3))
    d = np.diff(a)
    d = d[d > 0.005]
    if d.size < 100:
        return 0.2
    return float(max(0.05, round(float(np.median(d)), 2)))
