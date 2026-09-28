"""Лучевая инъекция объектов в реальное облако лидара."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fod.cloud import N_RINGS, LidarCloud, apply_to_raw, estimate_floor_z, ray_directions
from fod.objects import SceneObject, hit_object, incidence_cos, object_intensity, tunnel_intensity


@dataclass
class InjectionResult:
    cloud: LidarCloud
    z_floor: float
    hits: dict[str, int] = field(default_factory=dict)
    hit_mask: np.ndarray | None = None


def inject(
    cloud: LidarCloud,
    objects: list[SceneObject],
    z_floor: float | None = None,
    range_noise: float = 0.02,
    rng: np.random.Generator | None = None,
    dual_return: bool = True,
    reflect: float | None = None,
) -> InjectionResult:
    """`reflect` — отражательная способность объекта: интенсивность как у поверхностей тоннеля
    (`tunnel_intensity`) вместо яркости из каталога."""
    if not objects:
        return InjectionResult(cloud=cloud.copy(), z_floor=z_floor or estimate_floor_z(cloud.xyz))

    out = cloud.copy()
    floor = estimate_floor_z(out.xyz) if z_floor is None else z_floor
    dirs, measured = ray_directions(out)
    t_best = np.full(out.n_points, np.inf, dtype=np.float64)
    winner = np.full(out.n_points, -1, dtype=np.int32)
    hits = {obj.name: 0 for obj in objects}

    for idx, obj in enumerate(objects):
        t_hit = hit_object(obj, dirs, floor)
        better = t_hit < t_best
        t_best[better] = t_hit[better]
        winner[better] = idx

    replace = (winner >= 0) & (t_best < measured)
    if not np.any(replace):
        return InjectionResult(cloud=out, z_floor=floor, hits=hits, hit_mask=replace)

    engine = rng or np.random.default_rng(0)
    t_use = t_best[replace]
    if range_noise > 0:
        t_use = np.maximum(t_use + engine.normal(0.0, range_noise, size=t_use.shape), 0.05)

    orig_xyz = out.xyz.copy()
    orig_int = out.intensity.copy()
    orig_ts = out.timestamp.copy()

    dirs_r = dirs[replace]
    out.xyz[replace] = (t_use[:, None] * dirs_r).astype(np.float32)
    intensity = np.empty(out.n_points, dtype=np.float32)
    edge = _edge_rays(replace) if reflect is not None else None
    for idx, obj in enumerate(objects):
        sel = replace & (winner == idx)
        count = int(sel.sum())
        hits[obj.name] = count
        if count == 0:
            continue
        if reflect is None:
            intensity[sel] = object_intensity(obj, t_best[sel], dirs[sel]).astype(np.float32)
        else:
            cos_inc = incidence_cos(obj, t_best[sel], dirs[sel], floor)
            intensity[sel] = tunnel_intensity(reflect, cos_inc, edge[sel], engine).astype(np.float32)
    out.intensity[replace] = intensity[replace]

    if dual_return and out.n_points % N_RINGS == 0:
        _write_second_return(out, replace, orig_xyz, orig_int, orig_ts, measured)

    out.raw = apply_to_raw(out)
    return InjectionResult(cloud=out, z_floor=floor, hits=hits, hit_mask=replace)


def _edge_rays(hit: np.ndarray) -> np.ndarray:
    """Лучи объекта, у которых соседний по кольцу или по колонке (пары двойного возврата) луч мимо."""
    if hit.size % N_RINGS:
        return np.zeros_like(hit)
    grid = hit.reshape(-1, N_RINGS)
    pad = np.pad(grid, ((2, 2), (1, 1)), constant_values=False)
    inner = pad[2:-2, :-2] & pad[2:-2, 2:] & pad[:-4, 1:-1] & pad[4:, 1:-1]
    return (grid & ~inner).reshape(-1)


def _write_second_return(
    cloud: LidarCloud,
    replace: np.ndarray,
    orig_xyz: np.ndarray,
    orig_int: np.ndarray,
    orig_ts: np.ndarray,
    measured: np.ndarray,
) -> None:
    n_cols = cloud.n_points // N_RINGS
    rings = np.arange(cloud.n_points) % N_RINGS
    cols = np.arange(cloud.n_points) // N_RINGS
    pair_cols = cols ^ 1
    valid_pair = pair_cols < n_cols
    pair_idx = pair_cols * N_RINGS + rings
    src = np.flatnonzero(replace & valid_pair)
    if src.size == 0:
        return
    dst = pair_idx[src]
    can_write = (~replace[dst]) & np.isfinite(measured[src])
    src = src[can_write]
    dst = dst[can_write]
    if src.size == 0:
        return
    cloud.xyz[dst] = orig_xyz[src]
    cloud.intensity[dst] = orig_int[src]
    cloud.timestamp[dst] = orig_ts[src]
    cloud.ring[dst] = cloud.ring[src]
