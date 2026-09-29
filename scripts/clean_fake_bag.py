#!/usr/bin/env python3
"""Запись `cloud_with_fake_obj` без синтетических препятствий, в формате исходных записей.

Генератор дописал точки объектов в конец облака (интенсивность ровно 1) и удалил
настоящие точки в их тени, а поля `ring` и `timestamp` выбросил. Здесь:

* хвост синтетики отрезается;
* облако возвращается в сетку 128 × 2400 (индекс = колонка·128 + кольцо): удалённые
  точки тени находятся по сдвигу номера кольца и становятся пустыми лучами, как у
  лидара без возврата. Кольцо — по элевации точки; элевации колец берутся из кадра
  этой записи без объектов (от паспорта отличаются до 0.12° при шаге колец 0.09°);
* `ring` — по индексу, `timestamp` — время кадра плюс шаблон времени точек из
  исходной записи: он одинаков во всех кадрах.

  python3 scripts/clean_fake_bag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from obstacle_bench import ensure_ros_env  # noqa: E402

N_RINGS = 128
N_POINTS = 128 * 2400
DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("ring", "<u2"), ("timestamp", "<f8")])


def synthetic_tail(intensity: np.ndarray) -> int:
    """Начало хвоста синтетики: точки после последней с интенсивностью не 1."""
    other = np.flatnonzero(intensity != 1.0)
    return int(other[-1]) + 1 if other.size else 0


def elevation(xyz: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(xyz, axis=1)
    return np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0)))


def ring_elevations(xyz: np.ndarray) -> np.ndarray:
    """Элевация каждого кольца по кадру в исходном порядке точек."""
    el = elevation(xyz.astype(np.float64)).reshape(-1, N_RINGS)
    el[np.linalg.norm(xyz, axis=1).reshape(-1, N_RINGS) < 0.3] = np.nan
    return np.nanmedian(el, axis=0)


def ring_of(xyz: np.ndarray, elev: np.ndarray) -> np.ndarray:
    el = elevation(xyz)
    order = np.argsort(elev)
    srt = elev[order]
    j = np.clip(np.searchsorted(srt, el), 1, srt.size - 1)
    j -= (el - srt[j - 1]) < (srt[j] - el)
    return order[j]


def restore_index(xyz: np.ndarray, elev: np.ndarray) -> np.ndarray | None:
    """Индексы точек в сетке 128 × 2400 после удаления части точек."""
    m = xyz.shape[0]
    k = N_POINTS - m
    if k == 0:
        return np.arange(m)
    if k < 0:
        return None
    nz = np.flatnonzero(np.linalg.norm(xyz, axis=1) > 0.3)
    if nz.size < 1000:
        return None
    off = np.unwrap(((ring_of(xyz[nz], elev) - nz) % N_RINGS).astype(np.float64), period=N_RINGS)
    if off[0] > N_RINGS / 2:
        off -= N_RINGS
    off = np.clip(np.maximum.accumulate(off), 0, k)
    full = np.zeros(m)
    full[nz] = off
    # Пустые лучи между точками — со сдвигом предыдущей точки с возвратом.
    have = np.zeros(m, dtype=bool)
    have[nz] = True
    last = np.maximum.accumulate(np.where(have, np.arange(m), -1))
    full = np.where(last >= 0, full[np.maximum(last, 0)], 0.0)
    idx = np.arange(m) + full.astype(np.int64)
    if np.any(np.diff(idx) <= 0) or idx[-1] >= N_POINTS:
        return None
    return idx


def main() -> int:
    ensure_ros_env()
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import PointCloud2, PointField

    from fod.bags import DATA_ROOT, iter_pointclouds, read_bag_topic, resolve_bag
    from fod.cloud import from_msg
    from fod.odometry import stamp_of

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="cloud_with_fake_obj")
    parser.add_argument("--dst", type=Path, default=DATA_ROOT / "cloud_clean", help="Куда писать (по умолчанию FOD_DATA/cloud_clean).")
    parser.add_argument("--ref", default="roundT_doubleT", help="Запись, откуда шаблон времени точек.")
    parser.add_argument("--limit", type=int, help="Только столько кадров (проверка).")
    args = parser.parse_args()

    ref_dir = resolve_bag(args.ref)
    for _i, _ts, msg in iter_pointclouds(ref_dir, read_bag_topic(ref_dir)):
        ref = from_msg(msg)
        pattern = ref.timestamp - stamp_of(ref)
        fields = [PointField(name=f.name, offset=f.offset, datatype=f.datatype, count=f.count) for f in msg.fields]
        break

    src = resolve_bag(args.src)
    topic = read_bag_topic(src)
    for _i, _ts, msg in iter_pointclouds(src, topic):
        cloud = from_msg(msg)
        if cloud.n_points == N_POINTS and synthetic_tail(cloud.intensity) == N_POINTS:
            elev = ring_elevations(cloud.xyz)
            break
    if args.dst.exists():
        raise SystemExit(f"{args.dst} уже есть")
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.dst), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=topic, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"))

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(src), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    stats = []
    rings = np.tile(np.arange(N_RINGS, dtype=np.uint16), N_POINTS // N_RINGS)
    k = 0
    while reader.has_next():
        name, data, t_bag = reader.read_next()
        if name != topic:
            continue
        if args.limit is not None and k >= args.limit:
            break
        msg = deserialize_message(data, PointCloud2)
        cloud = from_msg(msg)
        cut = synthetic_tail(cloud.intensity)
        xyz, inten = cloud.xyz[:cut], cloud.intensity[:cut]
        idx = restore_index(xyz, elev)
        if idx is None:
            stats.append((k, cloud.n_points - cut, N_POINTS - cut, -1.0))
            print(f"  кадр {k}: не восстановлен, пропущен")
            k += 1
            continue
        out = np.zeros(N_POINTS, dtype=DTYPE)
        out["x"][idx], out["y"][idx], out["z"][idx] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        out["intensity"][idx] = inten
        out["ring"] = rings
        out["timestamp"] = stamp_of(cloud) + pattern
        nz = idx[np.linalg.norm(xyz, axis=1) > 0.3]
        p = np.stack([out["x"][nz], out["y"][nz], out["z"][nz]], axis=1).astype(np.float64)
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / np.linalg.norm(p, axis=1), -1.0, 1.0)))
        bad = float(np.mean(np.abs(el - elev[nz % N_RINGS]) > 0.03))
        stats.append((k, cloud.n_points - cut, N_POINTS - cut, bad))
        if bad > 1e-3:
            print(f"  кадр {k}: кольцо не сходится у {100 * bad:.2f}% точек, пропущен")
            k += 1
            continue
        new = PointCloud2()
        new.header = msg.header
        new.height, new.width = 1, N_POINTS
        new.fields = fields
        new.is_bigendian = False
        new.point_step = DTYPE.itemsize
        new.row_step = DTYPE.itemsize * N_POINTS
        new.data = out.tobytes()
        new.is_dense = False
        writer.write(topic, serialize_message(new), t_bag)
        if k % 100 == 0 or (args.limit and cut < cloud.n_points):
            print(f"  кадр {k}: синтетики {cloud.n_points - cut}, удалено тенью {N_POINTS - cut}, кольцо не сходится у {100 * bad:.2f}% точек", flush=True)
        k += 1
    del writer
    st = np.array(stats)
    np.save(ROOT / "out" / "tmp" / "clean_fake_stats.npy", st)
    obj = st[:, 1] > 0
    print(f"\nкадров {k}, с синтетикой {int(obj.sum())}, пропущено {int(np.sum(st[:, 3] < 0))}")
    print(f"удалено тенью: медиана {np.median(st[obj, 2]):.0f}, максимум {st[:, 2].max():.0f} точек")
    print(f"кольцо не сходится с элевацией: медиана {100 * np.median(st[st[:, 3] >= 0, 3]):.3f}%, максимум {100 * st[:, 3].max():.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
