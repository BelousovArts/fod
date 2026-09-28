"""Разбор и сборка PointCloud2 без зависимости от ROS в вычислительной части."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RINGS = 128
MIN_RANGE = 0.3

_PCL_DTYPE = {
    1: np.dtype("<i1"),
    2: np.dtype("<u1"),
    3: np.dtype("<i2"),
    4: np.dtype("<u2"),
    5: np.dtype("<i4"),
    6: np.dtype("<u4"),
    7: np.dtype("<f4"),
    8: np.dtype("<f8"),
}


@dataclass
class LidarCloud:
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    timestamp: np.ndarray
    width: int
    height: int
    is_dense: bool
    point_step: int
    frame_id: str
    stamp_sec: int
    stamp_nsec: int
    raw: np.ndarray
    field_offsets: dict[str, tuple[int, np.dtype]]

    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def range(self) -> np.ndarray:
        return np.linalg.norm(self.xyz, axis=1)

    @property
    def valid(self) -> np.ndarray:
        return self.range > MIN_RANGE

    def copy(self) -> "LidarCloud":
        return LidarCloud(
            xyz=self.xyz.copy(),
            intensity=self.intensity.copy(),
            ring=self.ring.copy(),
            timestamp=self.timestamp.copy(),
            width=self.width,
            height=self.height,
            is_dense=self.is_dense,
            point_step=self.point_step,
            frame_id=self.frame_id,
            stamp_sec=self.stamp_sec,
            stamp_nsec=self.stamp_nsec,
            raw=self.raw.copy(),
            field_offsets=dict(self.field_offsets),
        )


def from_msg(msg) -> LidarCloud:
    """`raw` — только для чтения, представление буфера сообщения (менять — через копию, как `apply_to_raw`)."""
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    offsets: dict[str, tuple[int, np.dtype]] = {}
    for field in msg.fields:
        dtype = _PCL_DTYPE.get(int(field.datatype))
        if dtype is None:
            continue
        offsets[field.name] = (int(field.offset), dtype)
    rec = np.frombuffer(
        msg.data,
        dtype=np.dtype({"names": list(offsets), "formats": [d for _o, d in offsets.values()],
                        "offsets": [o for o, _d in offsets.values()], "itemsize": step}),
        count=n,
    )

    def take(name: str, out_dtype, target: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros(n, dtype=out_dtype) if target is None else target
        if name in offsets:
            out[...] = rec[name]
        return out

    xyz = np.empty((n, 3), dtype=np.float32)
    for k, name in enumerate(("x", "y", "z")):
        take(name, np.float32, xyz[:, k])
    intensity = take("intensity", np.float32)
    ring = take("ring", np.uint16)
    timestamp = take("timestamp", np.float64)
    return LidarCloud(
        xyz=xyz,
        intensity=intensity,
        ring=ring,
        timestamp=timestamp,
        width=int(msg.width),
        height=int(msg.height),
        is_dense=bool(msg.is_dense),
        point_step=int(msg.point_step),
        frame_id=str(msg.header.frame_id),
        stamp_sec=int(msg.header.stamp.sec),
        stamp_nsec=int(msg.header.stamp.nanosec),
        raw=raw,
        field_offsets=offsets,
    )


def _write_field(raw: np.ndarray, offset: int, values: np.ndarray) -> None:
    blob = np.ascontiguousarray(values).view(np.uint8).reshape(values.shape[0], -1)
    raw[:, offset : offset + blob.shape[1]] = blob


def apply_to_raw(cloud: LidarCloud) -> np.ndarray:
    raw = cloud.raw.copy()
    for name, array in (
        ("x", cloud.xyz[:, 0]),
        ("y", cloud.xyz[:, 1]),
        ("z", cloud.xyz[:, 2]),
        ("intensity", cloud.intensity),
        ("ring", cloud.ring),
        ("timestamp", cloud.timestamp),
    ):
        if name not in cloud.field_offsets:
            continue
        offset, dtype = cloud.field_offsets[name]
        _write_field(raw, offset, array.astype(dtype, copy=False))
    return raw


def estimate_floor_z(xyz: np.ndarray, ahead: np.ndarray | None = None) -> float:
    """`ahead` — точки `xyz` в прежнем порядке, среди которых все с 4 < s < 40 и |n| < 2.5 (быстрее, тот же ответ)."""
    full = xyz
    xyz = full if ahead is None else ahead
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rng = np.linalg.norm(xyz, axis=1)
    s = -y
    n = x
    mask = (rng > MIN_RANGE) & (s > 6.0) & (s < 25.0) & (np.abs(n) < 1.6)
    if int(mask.sum()) < 200:
        mask = (rng > MIN_RANGE) & (s > 4.0) & (s < 40.0) & (np.abs(n) < 2.5)
    if int(mask.sum()) < 50:
        valid = np.linalg.norm(full, axis=1) > MIN_RANGE
        if not np.any(valid):
            return -1.4
        return float(np.percentile(full[valid, 2], 8))
    return float(np.percentile(z[mask], 12))


def default_elevations_rad() -> np.ndarray:
    el = np.empty(N_RINGS, dtype=np.float64)
    el[25:89] = np.linspace(np.deg2rad(1.97), np.deg2rad(-6.07), 64)
    el[0:25] = np.linspace(np.deg2rad(14.40), np.deg2rad(1.97 + 0.487), 25)
    el[89:128] = np.linspace(np.deg2rad(-6.07 - 0.487), np.deg2rad(-25.12), 39)
    return el


def _grouped_mean(values: np.ndarray, groups: np.ndarray, n_groups: int, valid: np.ndarray) -> np.ndarray:
    counts = np.bincount(groups[valid], minlength=n_groups).astype(np.float64)
    totals = np.bincount(groups[valid], weights=values[valid], minlength=n_groups)
    out = np.full(n_groups, np.nan, dtype=np.float64)
    ok = counts > 0
    out[ok] = totals[ok] / counts[ok]
    return out


def ray_directions(cloud: LidarCloud) -> tuple[np.ndarray, np.ndarray]:
    """Единичные направления и дальность. Пустые лучи получают направление по сетке."""
    rng = cloud.range
    valid = rng > MIN_RANGE
    dirs = np.zeros_like(cloud.xyz, dtype=np.float64)
    dirs[valid] = cloud.xyz[valid] / rng[valid, None]
    measured = np.where(valid, rng.astype(np.float64), np.inf)

    n = cloud.n_points
    if n % N_RINGS != 0:
        return dirs, measured

    n_cols = n // N_RINGS
    index = np.arange(n)
    rings = index % N_RINGS
    cols = index // N_RINGS
    az = np.arctan2(cloud.xyz[:, 1], cloud.xyz[:, 0])
    el = np.arctan2(cloud.xyz[:, 2], np.hypot(cloud.xyz[:, 0], cloud.xyz[:, 1]))

    el_ring = _grouped_mean(el, rings, N_RINGS, valid)
    fallback = default_elevations_rad()
    missing = np.isnan(el_ring)
    el_ring[missing] = fallback[missing]

    az_col = _grouped_mean(az, cols, n_cols, valid)
    idx = np.arange(n_cols)
    good = np.isfinite(az_col)
    if np.any(good) and not np.all(good):
        az_col[~good] = np.interp(idx[~good], idx[good], az_col[good])
    elif not np.any(good):
        az_col = np.deg2rad(-0.100000 * (idx // 2) - 30.0035)

    empty = ~valid
    if np.any(empty):
        el_e = el_ring[rings[empty]]
        az_e = az_col[cols[empty]]
        cel = np.cos(el_e)
        dirs[empty, 0] = cel * np.cos(az_e)
        dirs[empty, 1] = cel * np.sin(az_e)
        dirs[empty, 2] = np.sin(el_e)
    return dirs, measured
