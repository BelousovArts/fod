#!/usr/bin/env python3
"""Карта по записи: облако всей записи в мире и картинка сверху.

Позы — те же, что в тракте (small_gicp, `fod/pipeline.py`). Каждый кадр даёт точки
ближе `--range` м (вблизи облако плотнее и точнее), всё копится в вокселях `--voxel` м
(среднее положение и интенсивность). В `<out>/<запись>/map/`:

- `map.ply`, `map.pcd` — облако: x, y, z, intensity (float32), система — мир первого кадра;
- `map_top.png` — вид сверху всей записи: полотно по интенсивности, стены приглушённо,
  траектория (зелёная), ось пути (жёлтая), подтверждённые препятствия (красные, с номером);
- `tiles/*.png` — то же плитками по 100 м пути, 3 см на пиксель, каждая выпрямлена по участку;
- `trajectory.csv` — позы по кадрам: положение, курс, скорость, статус;
- `obstacles.csv` — препятствия в мире: где впервые подтверждено и с какого расстояния.

  python3 scripts/make_map.py roundT_doubleT
  python3 scripts/make_map.py /путь/к/записи --voxel 0.1 --until 120 --out results
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEY_OFF = 1 << 20          # ключ вокселя: по 21 биту на ось, ±52 км при 5 см
MERGE_EVERY = 40           # кадров между слияниями накопленного


class VoxelMap:
    """Сумма положений и интенсивности по вокселям; сливается пачками."""

    def __init__(self, size: float) -> None:
        self.size = size
        self.keys = np.zeros(0, np.int64)
        self.sums = np.zeros((0, 4))
        self.counts = np.zeros(0, np.int64)
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def _key(self, xyz: np.ndarray) -> np.ndarray:
        q = np.floor(xyz / self.size).astype(np.int64) + KEY_OFF
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    @staticmethod
    def _reduce(keys, sums, counts):
        uniq, inv = np.unique(keys, return_inverse=True)
        s = np.zeros((uniq.size, sums.shape[1]))
        np.add.at(s, inv, sums)
        c = np.bincount(inv, weights=counts, minlength=uniq.size).astype(np.int64)
        return uniq, s, c

    def add(self, xyz: np.ndarray, intensity: np.ndarray) -> None:
        if not xyz.shape[0]:
            return
        sums = np.concatenate([xyz, intensity[:, None]], axis=1)
        self._pending.append(self._reduce(self._key(xyz), sums, np.ones(xyz.shape[0], np.int64)))
        if len(self._pending) >= MERGE_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        keys = np.concatenate([self.keys] + [p[0] for p in self._pending])
        sums = np.concatenate([self.sums] + [p[1] for p in self._pending])
        counts = np.concatenate([self.counts] + [p[2] for p in self._pending])
        self._pending = []
        self.keys, self.sums, self.counts = self._reduce(keys, sums, counts)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        self.flush()
        mean = self.sums / self.counts[:, None]
        return mean[:, :3], mean[:, 3]


def write_ply(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    data = _records(xyz, intensity)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {data.size}\n"
        "property float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def write_pcd(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    data = _records(xyz, intensity)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {data.size}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {data.size}\nDATA binary\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def _records(xyz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    data = np.empty(xyz.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["intensity"] = intensity
    return data


def _put(img, text, org, scale=0.6, color=(235, 235, 235)) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class TopView:
    """Растр сверху: полотно по интенсивности, стены приглушённо, траектория, ось, препятствия.

    Полотно — точки не выше 0,5 м над нижним уровнем у ближайшей позы: потолок и стены сверху
    закрыли бы путь.
    """

    def __init__(self, xyz, intensity, traj: np.ndarray, axis_pts: np.ndarray, obstacles: list[dict]) -> None:
        from scipy.spatial import cKDTree

        self.xyz, self.intensity = xyz, intensity
        self.traj, self.axis_pts, self.obstacles = traj, axis_pts, obstacles
        _, j = cKDTree(traj[:, :2]).query(xyz[:, :2], workers=-1)
        rel = xyz[:, 2] - traj[j, 2]
        low = np.percentile(rel, 20) if rel.size else 0.0
        self.floor = rel < low + 0.5
        self.wall = ~self.floor & (rel < 1.0)
        self.path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))])
        self.s_point = self.path[j]

    def render(self, center, rot, u_lim, v_lim, px: float, sel=None) -> np.ndarray:
        import cv2

        from fod.colormaps import colorize_intensity

        (u0, u1), (v0, v1) = u_lim, v_lim
        w, h = int(np.ceil((u1 - u0) / px)), int(np.ceil((v1 - v0) / px))

        def to_px(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            r = (q[:, :2] - center) @ rot.T
            return ((r[:, 0] - u0) / px).astype(np.int64), ((v1 - r[:, 1]) / px).astype(np.int64)

        idx = np.arange(self.xyz.shape[0]) if sel is None else np.flatnonzero(sel)
        iu, iv = to_px(self.xyz[idx])
        ok = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
        idx, iu, iv = idx[ok], iu[ok], iv[ok]
        img = np.full((h, w, 3), 10, np.uint8)
        wall = self.wall[idx]
        img[iv[wall], iu[wall]] = (90, 70, 50)
        floor = self.floor[idx]
        # По возрастанию интенсивности: яркие (рельсы, метки) не затираются тёмными.
        order = np.argsort(self.intensity[idx][floor], kind="stable")
        fu, fv = iu[floor][order], iv[floor][order]
        img[fv, fu] = colorize_intensity(self.intensity[idx][floor][order].astype(np.float32)[None, :], vmax=40.0)[0]
        tu, tv = to_px(self.traj)
        cv2.polylines(img, [np.stack([tu, tv], axis=1).astype(np.int32)], False, (70, 170, 90), 3, cv2.LINE_AA)
        if self.axis_pts.shape[0] >= 2:
            au, av = to_px(self.axis_pts)
            cv2.polylines(img, [np.stack([au, av], axis=1).astype(np.int32)], False, (40, 230, 255), 1, cv2.LINE_AA)
        for o in self.obstacles:
            ou, ov = to_px(np.array([[o["x"], o["y"], o["z"]]]))
            cv2.circle(img, (int(ou[0]), int(ov[0])), max(8, int(0.8 / px)), (60, 60, 255), 2, cv2.LINE_AA)
            _put(img, f"#{o['track_id']} confirmed at {o['distance_m']:.0f} m", (int(ou[0]) + 10, int(ov[0]) - 10), 0.5, (60, 60, 255))
        return img

    def overview(self, path: Path, title: str, max_px: int = 8000) -> None:
        """Вся запись, повёрнута по главной оси траектории и обрезана по занятой области."""
        import cv2

        traj = self.traj
        c = traj[:, :2].mean(axis=0)
        rot = np.eye(2)
        if traj.shape[0] >= 2 and np.ptp(traj[:, :2], axis=0).max() > 1.0:
            _, _, vt = np.linalg.svd(traj[:, :2] - c, full_matrices=False)
            ax = vt[0] if vt[0] @ (traj[-1, :2] - traj[0, :2]) >= 0 else -vt[0]
            rot = np.array([[ax[0], ax[1]], [-ax[1], ax[0]]])
        uv = (self.xyz[:, :2] - c) @ rot.T
        lo, hi = np.percentile(uv, 0.1, axis=0) - 2.0, np.percentile(uv, 99.9, axis=0) + 2.0
        px = max(0.05, float(max(hi - lo)) / max_px)
        img = self.render(c, rot, (lo[0], hi[0]), (lo[1], hi[1]), px)
        rows = np.flatnonzero((img != 10).any(axis=(1, 2)))
        if rows.size:
            img = img[max(rows[0] - 10, 0) : rows[-1] + 10]
        cv2.imwrite(str(path), np.vstack([_caption(img.shape[1], f"{title}, {px * 100:.0f} cm/px"), img]))

    def tiles(self, out: Path, title: str, length: float = 100.0, half: float = 12.0, px: float = 0.03) -> int:
        """Плитки вдоль пути, каждая повёрнута по своей хорде."""
        import cv2

        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob("*.png"):
            old.unlink()
        traj, path = self.traj, self.path
        n = 0
        for a in np.arange(0.0, max(path[-1], 1e-3), length):
            i0 = int(np.searchsorted(path, a))
            i1 = min(int(np.searchsorted(path, a + length)), len(path) - 1)
            if i1 > i0 and np.linalg.norm(traj[i1, :2] - traj[i0, :2]) > 5.0:
                chord = traj[i1, :2] - traj[i0, :2]
                fwd = chord / np.linalg.norm(chord)
                center = 0.5 * (traj[i0, :2] + traj[i1, :2])
                span = 0.5 * float(np.linalg.norm(chord))
            else:
                # Стоянка: плитка вокруг поезда, по его курсу (вперёд — −y сенсора).
                fwd = self.forward[i0]
                center = traj[i0, :2] + fwd * 0.5 * length
                span = 0.5 * length
            rot = np.array([[fwd[0], fwd[1]], [-fwd[1], fwd[0]]])
            sel = (self.s_point > a - 60.0) & (self.s_point < a + length + 60.0)
            img = self.render(center, rot, (-span - 2.0, span + 2.0), (-half, half), px, sel)
            cap = f"{title}  {a:.0f}-{min(a + length, path[-1]):.0f} m, {px * 100:.0f} cm/px, train moves right"
            cv2.imwrite(str(out / f"{n:03d}_{int(a):05d}m.png"), np.vstack([_caption(img.shape[1], cap), img]))
            n += 1
        return n


def _caption(width: int, text: str) -> np.ndarray:
    top = np.full((34, width, 3), 10, np.uint8)
    _put(top, text + ";  green: train path,  yellow: track centreline,  red: obstacles (where first confirmed)", (10, 23))
    return top


def main() -> int:
    from fod.mount import add_mount_args, mount_from_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="Папка записи, файл .db3/.mcap или имя записи в FOD_DATA.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--topic", help="Топик облака; по умолчанию первый PointCloud2.")
    parser.add_argument("--start", type=float, help="С какой секунды записи.")
    parser.add_argument("--until", type=float, help="До какой секунды записи.")
    parser.add_argument("--voxel", type=float, default=0.05, help="Размер вокселя карты, м.")
    parser.add_argument("--range", type=float, default=60.0, help="Точки кадра не дальше, м.")
    parser.add_argument("--device", default="cuda")
    add_mount_args(parser)
    args = parser.parse_args()

    from fod.bags import bag_time_range, iter_pointclouds, read_bag_topic, resolve_bag
    from fod.pipeline import Pipeline

    bag_dir = resolve_bag(args.bag)
    topic = args.topic or read_bag_topic(bag_dir)
    t0 = bag_time_range(bag_dir)[0]
    begin = t0 + int(args.start * 1e9) if args.start else None
    stop = t0 + int(args.until * 1e9) if args.until else None
    out = args.out / bag_dir.name / "map"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Запись {bag_dir}, топик {topic} → {out}", flush=True)

    pipe = Pipeline(device=args.device, keep_view=True, mount=mount_from_args(args))
    vmap = VoxelMap(args.voxel)
    items = ((i, msg, 0.0) for i, _ts, msg in iter_pointclouds(bag_dir, topic, begin, stop))
    traj, forward, axis_pts, rows = [], [], [], []
    last_axis = np.zeros((0, 3))
    obstacles: dict[int, dict] = {}
    for res in pipe.run(items):
        if res.pose is None:
            continue
        T = res.pose
        xyz, intensity = res.view[0], res.view[1]
        near = np.einsum("ij,ij->i", xyz, xyz) < args.range ** 2
        vmap.add(xyz[near] @ T[:3, :3].T + T[:3, 3], intensity[near].astype(np.float64))
        traj.append(T[:3, 3].copy())
        f = -T[:2, 1]
        forward.append(f / max(float(np.linalg.norm(f)), 1e-9))
        yaw = math.degrees(math.atan2(-T[1, 1], -T[0, 1]))   # вперёд — −y сенсора
        rows.append([res.index, f"{res.stamp:.6f}", *(f"{v:.3f}" for v in T[:3, 3]), f"{yaw:.2f}",
                     f"{res.speed_kmh:.1f}", res.status])
        if res.axis_n.size:
            s = 5.0
            p = np.array([np.interp(s, res.axis_s, res.axis_n), -s,
                          np.interp(s, res.axis_s, res.axis_z) if res.axis_z.size else -1.2])
            axis_pts.append(T[:3, :3] @ p + T[:3, 3])
            keep = (res.axis_s >= 5.0) & (res.axis_s <= res.reach)
            z = res.axis_z[keep] if res.axis_z.size else np.full(int(keep.sum()), -1.2)
            last_axis = np.stack([res.axis_n[keep], -res.axis_s[keep], z], axis=1) @ T[:3, :3].T + T[:3, 3]
        for o in res.obstacles:
            if o.track_id not in obstacles:
                w = T[:3, :3] @ np.array([o.x, o.y, o.z]) + T[:3, 3]
                obstacles[o.track_id] = {"track_id": o.track_id, "frame": res.index, "stamp": res.stamp,
                                         "distance_m": o.distance, "x": w[0], "y": w[1], "z": w[2], "height_m": o.height}
        if len(traj) % 100 == 0:
            print(f"  кадр {res.index}: вокселей {vmap.keys.size}", flush=True)
    if not traj:
        print("Нет ни одной позы — карта не построена.")
        return 1

    xyz, intensity = vmap.points()
    traj_a = np.array(traj)
    # Ось по кадрам в 5 м перед поездом, дальше — вся ось последнего кадра (на стоянке — только она).
    axis_a = np.concatenate([np.array(axis_pts).reshape(-1, 3), last_axis])
    write_ply(out / "map.ply", xyz, intensity)
    write_pcd(out / "map.pcd", xyz, intensity)
    length = float(np.linalg.norm(np.diff(traj_a, axis=0), axis=1).sum()) if len(traj_a) > 1 else 0.0
    top = TopView(xyz, intensity, traj_a, axis_a, list(obstacles.values()))
    top.forward = np.array(forward)
    top.overview(out / "map_top.png", f"{bag_dir.name}: {length:.0f} m, {len(traj)} frames")
    n_tiles = top.tiles(out / "tiles", bag_dir.name)
    with (out / "trajectory.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "stamp", "x", "y", "z", "yaw_deg", "speed_kmh", "status"])
        wr.writerows(rows)
    with (out / "obstacles.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["track_id", "frame", "stamp", "distance_m", "x", "y", "z", "height_m"])
        for o in obstacles.values():
            wr.writerow([o["track_id"], o["frame"], f"{o['stamp']:.6f}", f"{o['distance_m']:.2f}",
                         f"{o['x']:.2f}", f"{o['y']:.2f}", f"{o['z']:.2f}", f"{o['height_m']:.2f}"])
    print(f"Карта: {xyz.shape[0]} точек, путь {length:.0f} м, препятствий {len(obstacles)}, плиток {n_tiles} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
