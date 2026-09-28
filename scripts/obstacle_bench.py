#!/usr/bin/env python3
"""Стенд детектора препятствий: дальность на инъекциях и ложные тревоги на чистых записях.

Дальность. Неподвижный объект ставится на ось пути (по фильтру рельсов) в 80 м
впереди, поезд к нему подъезжает. Для эпизода пишутся три дальности: первое
попадание луча в объект, первый кандидат («подозрение») и подтверждение.
Когда объект проехан, следующий ставится снова в 80 м. Одометрия считается по
чистому облаку, всё остальное — по облаку с объектом.

Ложные тревоги. Те же записи без инъекций: доля кадров со статусом OBSTACLE
и число подтверждённых объектов в минуту. `--new-data K` добавляет 20 минут
`new_data`, порезанные на K кусков. В `doubleT_obstacle` препятствие настоящее —
она показана отдельно и в итог ложных не входит.

Кэш. `--cache` сохраняет по кадрам точки коридора, ось, марки и одометрию;
`--replay` гоняет по кэшу только детектор (секунды вместо 20 минут),
`--set margin=0.1 confirm_hits=4` меняет его настройки.

  python3 scripts/obstacle_bench.py --new-data 8 --cache
  python3 scripts/obstacle_bench.py --replay --new-data 8 --set margin=0.1
  python3 scripts/obstacle_bench.py --video roundT_doubleT --preset person

`--contact-rail`: ось детектора — слияние оси путевых рельсов с осью по
контактному рельсу (`fod/contact_rail.py`, позы — small_gicp в цикле), которое
уточняет карту оси пути в мире (`fod/track_map.py`; `--axis hold` — прежнее
смешивание с прошлой осью). Детектор смотрит до 150 м, но не дальше, чем известна ось;
объект ставится в 155 м. Кэш и видео с суффиксом `_cr`.

`--fake`: вместо четырёх ящиков — десять объектов записи `cloud_with_fake_obj` (те же
форма, поворот, смещение от оси и высота над головками). Стоящие объекты — на пол под
ними по облаку; интенсивность — как у поверхностей тоннеля (отражательная способность
на эпизод, угол падения, шум, ослабление на краях), тень — лучами.

  python3 scripts/obstacle_bench.py --only inject --fake --contact-rail --axis far --rails seg
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import sys
import time
import zlib
from dataclasses import dataclass, field, replace
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROS_SETUP = Path("/opt/ros/jazzy/setup.bash")
SHORT_BAGS = (
    "doubleT_obstacle",
    "doubleT_platform",
    "roundT_doubleT",
    "roundT_pressureGate_roundT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
)
# На стоящем поезде объект не приближается — для дальности не годится.
STATIC_BAGS = frozenset({"doubleT_obstacle"})
REAL_OBSTACLE_BAGS = frozenset({"doubleT_obstacle"})
PRESETS = ("min", "cube", "crate", "person")
# Объекты записи cloud_with_fake_obj (fod/objects.py: FAKE_OBJECTS) с интенсивностью как у тоннеля.
FAKE_PRESETS = (
    "f1_shield", "f2_cube_air", "f3_cube_rail", "f4_cube_air_r", "f5_cube_air_l",
    "f6_big_r", "f7_big_l", "f8_over", "f9_slab", "f10_rod",
)
ALL_PRESETS = PRESETS + FAKE_PRESETS
# Пол под объектом, если своих точек мало: ниже головок.
FLOOR_BELOW_HEAD = 0.2
START_S = 80.0
START_S_CR = 155.0
S_MAX_CR = 150.0
END_S = 3.0
GAP_S = 0.55
# Низ объекта относительно головок рельсов: полотно между рельсами ниже головки на 0.09…0.18 м.
BASE_BELOW_HEAD = 0.12
REPORT_S = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0)
OUT_DIR = ROOT / "out" / "obstacles"
CACHE_DIR = OUT_DIR / "cache"
CACHE_S = (2.0, 155.0)
CACHE_HALF = 1.8
AXIS_GRID = np.arange(0.0, 160.5, 0.5)
RAIL_REACH = 60.0
RAIL_REACH_SEG = 80.0


def ensure_ros_env() -> None:
    if os.environ.get("ROS_DISTRO"):
        try:
            import rosbag2_py  # noqa: F401

            return
        except ImportError:
            pass
    if not ROS_SETUP.is_file():
        raise SystemExit("Не найден ROS 2 в /opt/ros/jazzy.")
    quoted = " ".join("'" + p.replace("'", "'\"'\"'") + "'" for p in [sys.executable, *sys.argv])
    os.execvp("bash", ["bash", "-lc", f'source "{ROS_SETUP}" && exec {quoted}'])


@dataclass
class Job:
    bag: str
    kind: str                      # inject / fp
    preset: str | None = None
    n_off: float = 0.0
    start_ns: int | None = None
    stop_ns: int | None = None
    part: int = 0
    video: Path | None = None
    cr: bool = False
    poses: str = "online"          # online / none / map / kiss / gicp
    axis: str = "map"              # map — карта пути в мире, tunnel — карта + профиль тоннеля, hold — AxisHold, ml — только сеть, far — tunnel + дальняя сеть
    rails: str = "template"        # template — шаблон пары головок, seg — сеть по range image
    every: int = 1                 # брать каждый K-й кадр записи
    realtime: bool = False         # проигрывать с частотой записи, обработка берёт последний кадр (fod/latest.py)
    delay_ms: float = 0.0          # лишняя задержка на кадр — как на медленной машине

    @property
    def name(self) -> str:
        tail = self.preset if self.kind == "inject" else "fp"
        cr = f"_cr_{self.axis}" if self.cr else ""
        seg = "_seg" if self.rails == "seg" else ""
        rate = (f"_e{self.every}" if self.every > 1 else "") + ("_rt" if self.realtime else "")
        rate += f"_d{self.delay_ms:g}" if self.delay_ms > 0 else ""
        return f"{self.bag}_{tail}" + (f"_{self.part}" if self.start_ns is not None else "") + cr + seg + f"_{self.poses}" + rate


@dataclass
class Frame:
    """Всё, что нужно детектору и оценке в одном кадре."""

    pts: np.ndarray                # (N, 3) после компенсации движения
    marks_s: np.ndarray
    marks_z: np.ndarray
    axis_n: np.ndarray             # ось на AXIS_GRID; пусто, если фильтр не заперт
    s_odom: float
    reset: bool
    ep: int = -1                   # номер эпизода инъекции, -1 — объекта нет
    obj_s: float = float("nan")
    obj_n: float = float("nan")
    obj_hits: int = 0
    view: tuple | None = field(default=None, repr=False)  # для видео, в кэш не идёт
    axis_z: np.ndarray = field(default_factory=lambda: np.zeros(0))  # головки на AXIS_GRID; пусто — своя подгонка
    reach: float = RAIL_REACH
    pose: np.ndarray | None = None  # сенсор→мир; None — треки по координате пути
    index: int = -1                # номер кадра в записи
    dt: float = 0.1                # с прошлого обработанного кадра, с
    t_in: float = float("nan")     # perf_counter, когда кадр пришёл


class AxisSamples:
    def __init__(self, axis_n: np.ndarray, axis_z: np.ndarray | None = None, reach: float = RAIL_REACH) -> None:
        self.axis_n = axis_n
        self.locked = axis_n.size > 0
        self.reach = reach
        self.head = (lambda s: np.interp(s, AXIS_GRID, axis_z)) if axis_z is not None and axis_z.size else None

    def n(self, s):
        return np.interp(s, AXIS_GRID, self.axis_n)


def bag_messages(job: Job):
    """Сообщения записи: (номер, сообщение, perf_counter прихода). С `job.realtime` запись играется в
    своём потоке с частотой записи, а отдаётся всегда последнее пришедшее — как у узла на живом потоке."""
    import threading

    from fod.bags import iter_pointclouds, read_bag_topic, resolve_bag
    from fod.latest import Latest

    bag_dir = resolve_bag(job.bag)
    items = iter_pointclouds(bag_dir, read_bag_topic(bag_dir), job.start_ns, job.stop_ns)
    if job.every > 1:
        items = (it for it in items if it[0] % job.every == 0)
    if not job.realtime:
        for index, _ts, msg in items:
            yield index, msg, time.perf_counter()
        return
    slot: Latest = Latest()

    def play() -> None:
        try:
            origin = None
            for index, ts, msg in items:
                if slot.closed:
                    return
                if origin is None:
                    origin = (time.perf_counter(), ts)
                wait = origin[0] + (ts - origin[1]) * 1e-9 - time.perf_counter()
                if wait > 0.0:
                    time.sleep(wait)
                slot.put((index, msg, time.perf_counter()))
        finally:
            slot.close()

    thread = threading.Thread(target=play, name="player", daemon=True)
    thread.start()
    try:
        yield from slot
    finally:
        slot.close()
        thread.join(timeout=5.0)


def motion_frames(job: Job):
    """Кадры записи с одометрией по чистому облаку: (номер, облако, штамп, сброс, движение, облако после
    компенсации, время прихода)."""
    from fod.cloud import from_msg
    from fod.odometry import EgoMotion, deskew_xyz, stamp_of

    odom = EgoMotion()
    last_stamp = None
    for index, msg, t_in in bag_messages(job):
        cloud = from_msg(msg)
        stamp = stamp_of(cloud)
        reset = last_stamp is not None and stamp - last_stamp > GAP_S
        if reset:
            odom = EgoMotion()
        last_stamp = stamp
        motion = odom.step(cloud.xyz, cloud.intensity, stamp)
        yield index, cloud, stamp, reset, motion, deskew_xyz(cloud.xyz, cloud.timestamp, motion.v), t_in


def prepared_frames(frames, classify=None, icp=None):
    """К кадрам `motion_frames` — выход сети рельсов `classify(xyz, intensity)` и поза `icp` (small_gicp в своём
    потоке, одновременно с сетью); None, если их нет. Всё — по чистому облаку, от трекера не зависит."""
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(1, thread_name_prefix="icp")
    for index, cloud, stamp, reset, motion, xyz, t_in in frames:
        icp_job = None
        if icp is not None:
            if reset:
                icp.reset()
            icp_job = pool.submit(icp.step, xyz[np.isfinite(xyz).all(axis=1)], motion.travelled)
        pre = (xyz, *classify(xyz, cloud.intensity)) if classify is not None else None
        pose = icp_job.result() if icp_job is not None else None
        yield index, cloud, stamp, reset, motion, xyz, pre, pose, t_in


def live_frames(job: Job, keep_view: bool = False):
    from fod.cloud import estimate_floor_z
    from fod.inject import inject
    from fod.objects import FAKE_OBJECTS, REFLECT_MEDIAN, REFLECT_SIGMA, SceneObject
    from fod.objects import PRESETS as OBJ_PRESETS
    from fod.obstacles import ObstacleDetector
    from fod.odometry import deskew_xyz
    from fod.prefetch import prefetch
    from fod.rail_axis_filter import RailTracker

    detector = None
    if job.rails == "seg":
        from fod.rail_seg_detect import SegRailDetector

        detector = SegRailDetector()
    rail_cap = RAIL_REACH_SEG if job.rails == "seg" else RAIL_REACH
    tracker, placer = RailTracker(detector=detector, predict_m=rail_cap), ObstacleDetector()
    cr = None
    start_s = START_S
    icp = None
    if job.poses == "online":
        from fod.lidar_odometry import GicpOdometry

        icp = GicpOdometry()
    elif job.poses != "none":
        sys.path.insert(0, str(ROOT / "scripts"))
        from fast_odometry import load_poses

        pose_stamps, poses = load_poses(job.bag, job.poses)
    if job.cr:
        from fod.contact_rail import AxisHold, ContactRailConfig, ContactRailTracker, axis_state, extend_linear

        from fod.track_map import TrackMap, TrackMapConfig

        cr_cfg = ContactRailConfig()
        from fod.tunnel_axis import TunnelAxis, fuse_tunnel

        use_ml = job.axis == "ml"
        use_hold = job.axis in ("hold", "tunnel_hold")
        cr = hold = tunnel = far = None
        if not use_ml:
            cr, hold = ContactRailTracker(cr_cfg, device="cuda"), (AxisHold() if use_hold else TrackMap())
            tunnel = TunnelAxis(device="cuda") if job.axis.startswith("tunnel") or job.axis == "far" else None
        if job.axis == "far":
            from fod.far_bev import FarAxis, FrameBuffer

            far, far_buf, far_map = FarAxis(), FrameBuffer(device="cuda"), TrackMap(TrackMapConfig(s_cap=160.0))
        start_s = START_S_CR
    axis_prev = z_prev = np.zeros(0)
    S_obj: float | None = None
    ep = -1
    fake_ep, floor_off, ep_rng, reflect = -1, None, None, REFLECT_MEDIAN
    # Три потока: чтение и EgoMotion → сеть рельсов и small_gicp → ось, КР, тоннель, детектор.
    classify = detector.classify_only if detector is not None else None
    # В реальном времени — по запросу, иначе в очередях лежат устаревшие кадры.
    depth = 0 if job.realtime else 2
    frames = prefetch(prepared_frames(prefetch(motion_frames(job), depth), classify, icp), depth)
    for _index, cloud, stamp, reset, motion, xyz_clean, pre, icp_pose, t_in in frames:
        if job.delay_ms > 0:
            time.sleep(1e-3 * job.delay_ms)
        if reset:
            tracker, placer = RailTracker(detector=detector, predict_m=rail_cap), ObstacleDetector()
            axis_prev = z_prev = np.zeros(0)
            if cr is not None:
                cr.reset()
                hold.reset()
                if tunnel is not None:
                    tunnel.reset()
                if far is not None:
                    far_buf.reset()
                    far_map.reset()
            S_obj = None
        s_odom = float(motion.s)
        obj = None
        hits = 0
        if job.kind == "inject":
            armed = tracker.filter.locked and placer.head_coeff is not None
            if not armed:
                S_obj = None
            elif S_obj is None and motion.v > 1.0:
                S_obj = s_odom + start_s
                ep += 1
            if S_obj is not None and S_obj - s_odom < END_S:
                S_obj = None
            if S_obj is not None:
                s_obj = S_obj - s_odom
                z_floor = estimate_floor_z(cloud.xyz)
                if z_prev.size:
                    head = float(np.interp(s_obj, AXIS_GRID, z_prev))
                else:
                    head = float(placer.head_z(float(np.clip(s_obj, 4.0, 60.0))))
                n_axis = float(np.interp(s_obj, AXIS_GRID, axis_prev)) if axis_prev.size else float(tracker.filter.n(s_obj))
                if job.preset in FAKE_OBJECTS:
                    spec = FAKE_OBJECTS[job.preset]
                    if ep != fake_ep:
                        fake_ep, floor_off = ep, None
                        ep_rng = np.random.default_rng(zlib.crc32(f"{job.bag}/{job.preset}/{ep}".encode()))
                        reflect = float(ep_rng.lognormal(np.log(REFLECT_MEDIAN), REFLECT_SIGMA))
                    n_c = n_axis + spec["n"] + job.n_off
                    if spec["rest"] == "floor":
                        off = _floor_below(cloud.xyz, s_obj, n_c, spec["size"], head)
                        if off is not None:
                            floor_off = off if floor_off is None else 0.7 * floor_off + 0.3 * off
                        base = head + (floor_off if floor_off is not None else -FLOOR_BELOW_HEAD)
                    else:
                        base = head + spec["u"]
                    obj = SceneObject(
                        kind=spec["kind"], name=job.preset, s=s_obj, n=n_c, u=base - z_floor,
                        size=tuple(spec["size"]), extra={"yaw": spec.get("yaw", 0.0)},
                    )
                    injected = inject(cloud, [obj], z_floor=z_floor, rng=ep_rng, reflect=reflect)
                else:
                    preset = OBJ_PRESETS[job.preset]
                    obj = SceneObject(
                        kind=preset["kind"],
                        name=job.preset,
                        s=s_obj,
                        n=n_axis + job.n_off,
                        u=head - BASE_BELOW_HEAD - z_floor,
                        size=tuple(preset["size"]),
                        intensity=float(preset["intensity"]),
                    )
                    injected = inject(cloud, [obj], z_floor=z_floor)
                cloud = injected.cloud
                hits = int(injected.hits.get(job.preset, 0))
        xyz = deskew_xyz(cloud.xyz, cloud.timestamp, motion.v) if obj is not None else xyz_clean
        if icp is not None:
            pose = icp_pose
        elif job.poses != "none":
            k = int(np.searchsorted(pose_stamps, stamp))
            pose = poses[k] if k < pose_stamps.size and abs(pose_stamps[k] - stamp) < 1e-3 else None
        else:
            pose = None
        rail = tracker.step(xyz, cloud.intensity, motion, pre=pre)
        placer._fit_head(rail.marks)
        if far is not None and pose is not None:
            far_buf.push(xyz_clean, cloud.intensity, pose)
        locked = tracker.filter.locked
        axis_n = np.asarray(tracker.filter.n(AXIS_GRID), dtype=np.float64) if locked else np.zeros(0)
        axis_z = np.zeros(0)
        reach = rail_cap
        cr_frame = None
        if cr is not None or job.axis == "ml":
            ok = locked and len(rail.marks) >= 4 and placer.head_coeff is not None
            if job.axis == "ml":
                cr_frame = detector.contact_rail(xyz, tracker.filter if locked else None, placer if ok else None)
            else:
                cr_frame = cr.step(xyz, tracker.filter if ok else None, placer if ok else None, pose)
            if ok:
                rail_reach = max(float(m.s) for m in rail.marks)
                axis_n, axis_z, reach, sigma = axis_state(tracker.filter, placer, cr_frame, rail_reach, AXIS_GRID, cr_cfg, S_MAX_CR, rail_cap)
                if tunnel is not None:
                    slices = tunnel.step(xyz_clean, AXIS_GRID, axis_n, axis_z, reach)
                    axis_n, sigma, reach = fuse_tunnel(AXIS_GRID, axis_n, sigma, reach, slices)
                if job.axis == "ml":
                    pass
                elif not use_hold:
                    t_reach, t_sigma = reach, sigma
                    axis_n, axis_z, reach = hold.update(pose, AXIS_GRID, axis_n, axis_z, sigma, reach, min(rail_reach, rail_cap))
                    if far is not None and len(far_buf):
                        axis_n, axis_z, reach, f_sigma = far(far_buf, pose, AXIS_GRID, axis_n, axis_z, reach)
                        sigma = np.where(AXIS_GRID <= t_reach, t_sigma, f_sigma)
                        axis_n, axis_z, reach = far_map.update(pose, AXIS_GRID, axis_n, axis_z, sigma, reach, min(rail_reach, rail_cap))
                else:
                    axis_n, axis_z, reach = hold.update(pose, AXIS_GRID, axis_n, axis_z, reach)
                    axis_n, axis_z = extend_linear(AXIS_GRID, axis_n, reach), extend_linear(AXIS_GRID, axis_z, reach)
            elif locked and hold is not None:
                hold.reset()
                if tunnel is not None:
                    tunnel.reset()
                if far is not None:
                    far_map.reset()
            axis_prev, z_prev = axis_n, axis_z
        yield Frame(
            pts=xyz,
            marks_s=np.array([m.s for m in rail.marks], dtype=np.float64),
            marks_z=np.array([m.z for m in rail.marks], dtype=np.float64),
            axis_n=axis_n,
            s_odom=s_odom,
            reset=reset,
            ep=ep if obj is not None else -1,
            obj_s=obj.s if obj is not None else float("nan"),
            obj_n=obj.n if obj is not None else float("nan"),
            obj_hits=hits,
            view=(xyz, cloud, rail, motion, _index, cr_frame) if keep_view else None,
            axis_z=axis_z,
            reach=reach,
            pose=pose,
            index=int(_index),
            dt=float(motion.dt),
            t_in=t_in,
        )


def _half_extent(spec: dict) -> tuple[float, float]:
    """Полуразмеры объекта по n и по s с учётом поворота."""
    w, length = spec["size"][0], spec["size"][1]
    yaw = np.deg2rad(spec.get("yaw", 0.0))
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return 0.5 * (w * c + length * s), 0.5 * (w * s + length * c)


def _floor_below(xyz: np.ndarray, s_obj: float, n_c: float, size, head: float) -> float | None:
    """Пол под объектом относительно головок по точкам кадра; None — точек мало."""
    hn, hs = _half_extent({"size": size})
    s = -xyz[:, 1]
    near = (np.abs(s - s_obj) < max(hs + 0.5, 0.02 * s_obj)) & (np.abs(xyz[:, 0] - n_c) < hn + 0.2)
    z = xyz[near, 2]
    z = z[z < head + 0.3]
    if z.size < 6:
        return None
    return float(np.percentile(z, 10)) - head


def crop(frame: Frame) -> Frame:
    if frame.axis_n.size == 0:
        pts = np.zeros((0, 3), np.float32)
    else:
        s = -frame.pts[:, 1]
        keep = (s > CACHE_S[0]) & (s < min(CACHE_S[1], frame.reach + 2.0))
        n_rel = frame.pts[keep, 0] - np.interp(s[keep], AXIS_GRID, frame.axis_n)
        pts = frame.pts[keep][np.abs(n_rel) < CACHE_HALF].astype(np.float32)
    return replace(frame, pts=pts, view=None)


def evaluate(job: Job, frames, cfg, writer=None) -> dict:
    from fod.obstacles import ObstacleDetector

    detector = ObstacleDetector(cfg)
    episodes: list[dict] = []
    current: dict | None = None
    frames_n = ready = alarm = 0
    fp_ids: dict[int, tuple[float, float, float]] = {}
    det_ms: list[float] = []
    lat_ms: list[float] = []
    ready_s = 0.0
    first_index = last_index = -1

    def close() -> None:
        nonlocal current
        if current is not None and current["last_s"] < 10.0:
            episodes.append({k: current[k] for k in ("first_ray", "suspect", "confirmed", "after", "shown", "drops")})
        current = None

    from fod.objects import FAKE_OBJECTS

    spec = FAKE_OBJECTS.get(job.preset)
    hn, hs = _half_extent(spec) if spec else (0.0, 0.0)
    n_obj = job.n_off + (spec["n"] if spec else 0.0)

    def matches(s_det: float, n_det: float, frame: Frame) -> bool:
        return abs(s_det - frame.obj_s) < 2.0 + 0.03 * frame.obj_s + hs and abs(n_det - n_obj) < 0.8 + hn

    for frame in frames:
        if frame.reset:
            detector = ObstacleDetector(cfg)
            close()
        if current is not None and frame.ep != current["ep"]:
            close()
        if frame.ep >= 0 and current is None:
            current = {
                "ep": frame.ep, "first_ray": np.nan, "suspect": np.nan, "confirmed": np.nan, "last_s": np.inf,
                "after": 0, "shown": 0, "drops": 0, "was_shown": False,
            }
        axis = AxisSamples(frame.axis_n, getattr(frame, "axis_z", None), frame.reach)
        marks = [SimpleNamespace(s=s, z=z) for s, z in zip(frame.marks_s, frame.marks_z)]
        det = detector.step(
            frame.pts, SimpleNamespace(marks=marks), axis if axis.locked else None, frame.s_odom, getattr(frame, "pose", None)
        )
        frames_n += 1
        det_ms.append(det.ms)
        t_in = getattr(frame, "t_in", float("nan"))
        if np.isfinite(t_in):
            lat_ms.append(1e3 * (time.perf_counter() - t_in))
        index = getattr(frame, "index", -1)
        if index >= 0:
            first_index = index if first_index < 0 else first_index
            last_index = index
        ready += int(det.ready)
        if det.ready:
            ready_s += 0.1 if frame.reset else min(float(getattr(frame, "dt", 0.1)), GAP_S)
        alarm += int(det.status == "OBSTACLE")
        has_obj = frame.ep >= 0
        for track in det.confirmed:
            s_t = track.s_now(frame.s_odom)
            if not (0.0 < s_t < detector.cfg.s_max):
                continue
            if has_obj and matches(s_t, track.n, frame):
                continue
            fp_ids.setdefault(track.track_id, (float(s_t), float(track.n), float(frame.reach)))
        if has_obj:
            current["last_s"] = frame.obj_s
            if frame.obj_hits > 0 and np.isnan(current["first_ray"]):
                current["first_ray"] = frame.obj_s
            if np.isnan(current["suspect"]) and any(matches(c.s, c.n, frame) for c in det.candidates):
                current["suspect"] = frame.obj_s
            shown = any(matches(t.s, t.n, frame) for t in det.confirmed)
            if np.isnan(current["confirmed"]) and shown:
                current["confirmed"] = frame.obj_s
            # Непрерывность после первого подтверждения, пока объект дальше s_min.
            if np.isfinite(current["confirmed"]) and frame.obj_s > detector.cfg.s_min:
                current["after"] += 1
                current["shown"] += int(shown)
                current["drops"] += int(current["was_shown"] and not shown)
                current["was_shown"] = shown
        if writer is not None:
            writer.add(frame, det, detector, job)
    close()
    return {
        "job": job.name,
        "bag": job.bag,
        "kind": job.kind,
        "preset": job.preset,
        "part": job.part,
        "frames": frames_n,
        "ready": ready,
        "alarm": alarm,
        "fp_objects": len(fp_ids),
        "fp_at": list(fp_ids.values()),
        "minutes": ready_s / 60.0,
        "det_ms": float(np.median(det_ms)) if det_ms else float("nan"),
        # Кадров записи на отрезке и сколько из них не дошло до обработки (прореживание и пропуски).
        "source_frames": last_index - first_index + 1 if first_index >= 0 else frames_n,
        "lat_ms": [float(np.percentile(lat_ms, q)) for q in (50, 95, 100)] if lat_ms else None,
        "episodes": episodes,
    }


def run_job(task: tuple) -> dict:
    job, overrides, mode = task
    from fod.obstacles import load_detector_config

    cfg = replace(load_detector_config(), **overrides)
    path = CACHE_DIR / f"{job.name}.pkl"
    if mode == "replay":
        with path.open("rb") as fh:
            frames = pickle.load(fh)
        return evaluate(job, frames, cfg)
    writer = _VideoWriter(job.video) if job.video else None
    stored: list[Frame] = []

    def source():
        for frame in live_frames(job, keep_view=writer is not None):
            if mode == "cache":
                stored.append(crop(frame))
            yield frame

    try:
        result = evaluate(job, source(), cfg, writer)
    finally:
        if writer is not None:
            writer.close()
    if mode == "cache":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(stored, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


class _VideoWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.ffmpeg = None
        self.shape = None

    def add(self, frame: Frame, det, detector, job: Job) -> None:
        import cv2

        from fod.obstacle_view import render_obstacle_frame
        from fod.range_image import build_range_image, crop_elevation, crop_forward_sector
        from fod.video import even_dims, open_ffmpeg

        xyz, cloud, rail, motion, index, cr_frame = frame.view
        title = f"{job.bag}  kadr {index:05d}  {motion.kmh:4.1f} km/h  det {det.ms:.1f} ms"
        if cr_frame is not None:
            title += f"  KR do {cr_frame.reach:3.0f} m" if cr_frame.lines else "  KR net"
            if det.ready:
                title += f"  gauge to {det.s_max:3.0f} m"
        front = crop_elevation(crop_forward_sector(build_range_image(cloud), 60.0), -16.0, 8.0)
        obj = (job.preset, frame.obj_s, frame.obj_n) if frame.ep >= 0 else None
        axis = head = None
        if job.cr and frame.axis_n.size:
            axis = lambda q, a=frame.axis_n: np.interp(q, AXIS_GRID, a)  # noqa: E731
            if frame.axis_z.size:
                head = lambda q, a=frame.axis_z: np.interp(q, AXIS_GRID, a)  # noqa: E731
        img = render_obstacle_frame(
            xyz, cloud.intensity, rail, front, det, detector, obj, title=title, cr=cr_frame, axis=axis, head=head
        )
        img = even_dims(img)
        if self.shape is None:
            self.shape = (img.shape[1], img.shape[0])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.ffmpeg = open_ffmpeg(self.path, self.shape[0], self.shape[1], 10.0)
        elif (img.shape[1], img.shape[0]) != self.shape:
            img = cv2.resize(img, self.shape, interpolation=cv2.INTER_AREA)
        self.ffmpeg.stdin.write(img.tobytes())

    def close(self) -> None:
        if self.ffmpeg is not None and self.ffmpeg.stdin is not None:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()


def build_jobs(args) -> list[Job]:
    from fod.bags import bag_time_range, resolve_bag

    bags = args.bags or list(SHORT_BAGS)
    jobs: list[Job] = []
    if args.only in (None, "inject"):
        for bag in bags:
            if bag in STATIC_BAGS:
                continue
            for preset in args.presets:
                jobs.append(Job(bag, "inject", preset, n_off=args.n_off, cr=args.contact_rail, poses=args.poses, axis=args.axis, rails=args.rails))
    if args.only in (None, "fp"):
        jobs.extend(Job(bag, "fp", cr=args.contact_rail, poses=args.poses, axis=args.axis, rails=args.rails) for bag in bags)
        if args.new_data:
            start, stop = bag_time_range(resolve_bag("new_data"))
            if args.until:
                stop = start + int(args.until * 1e9)
            edges = np.linspace(start, stop + 1, args.new_data + 1).astype(np.int64)
            for part in range(args.new_data):
                jobs.append(
                    Job("new_data", "fp", start_ns=int(edges[part]), stop_ns=int(edges[part + 1]), part=part, cr=args.contact_rail, poses=args.poses, axis=args.axis, rails=args.rails)
                )
    for job in jobs:
        job.every, job.realtime, job.delay_ms = args.every, args.realtime, args.delay
    return jobs


def _fmt(values) -> str:
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    return "  —  " if not np.any(ok) else f"{np.median(v[ok]):5.1f}"


def report(results: list[dict]) -> dict:
    summary: dict = {}
    inj = [r for r in results if r["kind"] == "inject"]
    presets = [p for p in ALL_PRESETS if any(r["preset"] == p for r in inj)]
    if inj:
        print(f"\nДальность (объект на оси, подъезд с {START_S:.0f} м, с --contact-rail с {START_S_CR:.0f}): медиана по эпизодам, м")
        head = "  ".join(f">{s:.0f}м" for s in REPORT_S)
        print(f"  {'объект':13} {'эпиз':>4}  {'луч':>5}  {'подозр':>6}  {'подтв':>5}   подтверждено дальше: {head}  пропуск")
        for preset in presets:
            eps = [e for r in inj if r["preset"] == preset for e in r["episodes"]]
            if not eps:
                continue
            conf = np.array([e["confirmed"] for e in eps], dtype=np.float64)
            shares = "  ".join(f"{100.0 * np.mean(np.nan_to_num(conf, nan=-1.0) > s):4.0f}%" for s in REPORT_S)
            missed = 100.0 * np.mean(np.isnan(conf))
            summary[preset] = {"median": float(np.nanmedian(conf)) if np.any(np.isfinite(conf)) else None, "missed": missed}
            print(
                f"  {preset:13} {len(eps):4d}  {_fmt([e['first_ray'] for e in eps])}  "
                f"{_fmt([e['suspect'] for e in eps]):>6}  {_fmt(conf)}   {' ' * 21}{shares}  {missed:5.0f}%"
            )
        print("  непрерывность после подтверждения: доля кадров с объектом, пропаданий на эпизод (среднее)")
        for preset in presets:
            eps = [e for r in inj if r["preset"] == preset for e in r["episodes"] if e.get("after")]
            if eps:
                shown = sum(e["shown"] for e in eps) / sum(e["after"] for e in eps)
                print(f"    {preset:13} {100 * shown:5.1f}%   {np.mean([e['drops'] for e in eps]):4.1f}")
        print("  по записям, подтверждение медиана (эпизодов):")
        for bag in sorted({r["bag"] for r in inj}):
            cells = []
            for preset in presets:
                eps = [e for r in inj if r["bag"] == bag and r["preset"] == preset for e in r["episodes"]]
                if eps:
                    cells.append(f"{preset} {_fmt([e['confirmed'] for e in eps])} ({len(eps)})")
            print(f"    {bag:38} " + "   ".join(cells))
    fp = [r for r in results if r["kind"] == "fp"]
    if fp:
        print("\nЛожные тревоги на чистых записях")
        print(f"  {'запись':38} {'кадров':>6} {'готов':>6} {'OBSTACLE':>9} {'объектов':>9} {'в мин':>6} {'мс':>5}")
        merged: dict[str, dict] = {}
        for r in fp:
            m = merged.setdefault(r["bag"], {"frames": 0, "ready": 0, "alarm": 0, "fp_objects": 0, "minutes": 0.0, "ms": []})
            for key in ("frames", "ready", "alarm", "fp_objects", "minutes"):
                m[key] += r[key]
            m["ms"].append(r["det_ms"])
        total = {"ready": 0, "alarm": 0, "fp_objects": 0, "minutes": 0.0}
        for bag in sorted(merged, key=lambda b: (b in REAL_OBSTACLE_BAGS, b)):
            m = merged[bag]
            if bag not in REAL_OBSTACLE_BAGS:
                for key in total:
                    total[key] += m[key]
            rate = m["fp_objects"] / m["minutes"] if m["minutes"] > 0 else float("nan")
            ready_pct = 100.0 * m["ready"] / max(m["frames"], 1)
            alarm_pct = 100.0 * m["alarm"] / max(m["ready"], 1)
            label = bag + (" (настоящее)" if bag in REAL_OBSTACLE_BAGS else "")
            print(
                f"  {label:38} {m['frames']:6d} {ready_pct:5.0f}% {alarm_pct:8.1f}% {m['fp_objects']:9d} {rate:6.1f} {np.nanmedian(m['ms']):5.1f}"
            )
        rate = total["fp_objects"] / total["minutes"] if total["minutes"] > 0 else float("nan")
        alarm_pct = 100.0 * total["alarm"] / max(total["ready"], 1)
        summary["fp_per_min"] = rate
        summary["alarm_pct"] = alarm_pct
        print(f"  {'ИТОГО без настоящего':38} {'':6} {'':6} {alarm_pct:8.2f}% {total['fp_objects']:9d} {rate:6.2f}   за {total['minutes']:.1f} мин")
        at = np.array([a for r in fp if r["bag"] not in REAL_OBSTACLE_BAGS for a in r.get("fp_at", [])]).reshape(-1, 3)
        if at.size:
            edges = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 160.0)
            cells = "  ".join(f"{int(a)}–{int(b)} м: {int(np.sum((at[:, 0] >= a) & (at[:, 0] < b)))}" for a, b in zip(edges[:-1], edges[1:]))
            print(f"  ложные по дальности подтверждения: {cells};  в последних 15 м оси: {int(np.sum(at[:, 0] > at[:, 2] - 15.0))}")
    timed = [r for r in results if "_rt" in r["job"] or r.get("source_frames", r["frames"]) > r["frames"]]
    if timed:
        print("\nОбработано кадров записи и задержка от прихода кадра до ответа детектора (медиана / p95 / макс), мс")
        for r in sorted(timed, key=lambda r: r["job"]):
            share = 100.0 * r["frames"] / max(r.get("source_frames", r["frames"]), 1)
            lat = r.get("lat_ms")
            lat_s = " / ".join(f"{v:4.0f}" for v in lat) if lat and "_rt" in r["job"] else "—"
            print(f"  {r['job']:60} {r['frames']:5d} из {r.get('source_frames', r['frames']):5d} ({share:3.0f}%)   {lat_s}")
    return summary


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = ast.literal_eval(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bags", nargs="*", help="Записи. По умолчанию все короткие.")
    parser.add_argument("--only", choices=("inject", "fp"), help="Только дальность или только ложные тревоги.")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=ALL_PRESETS)
    parser.add_argument("--fake", action="store_true", help="Объекты записи cloud_with_fake_obj вместо --presets.")
    parser.add_argument("--n-off", type=float, default=0.0, help="Смещение объекта от оси пути, м.")
    parser.add_argument("--new-data", type=int, default=0, help="Добавить new_data в ложные тревоги, на K кусков.")
    parser.add_argument("--jobs", type=int, default=8, help="Параллельных процессов.")
    parser.add_argument("--cache", action="store_true", help="Сохранить кадры для --replay.")
    parser.add_argument("--replay", action="store_true", help="Только детектор по сохранённым кадрам.")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE", help="Настройки ObstacleConfig.")
    parser.add_argument("--tag", default="bench", help="Имя JSON с результатами.")
    parser.add_argument("--video", metavar="BAG", help="Одна запись с видео вместо стенда.")
    parser.add_argument("--preset", default="person", choices=ALL_PRESETS, help="Объект для --video.")
    parser.add_argument("--no-inject", action="store_true", help="В --video не вставлять объект.")
    parser.add_argument("--until", type=float, help="Только первые столько секунд записи (--video и new_data).")
    parser.add_argument("--every", type=int, default=1, help="Брать каждый K-й кадр записи (проверка на 10/K Гц).")
    parser.add_argument("--realtime", action="store_true",
                        help="Играть запись с её частотой, обработка берёт последний пришедший кадр, лишние пропускает. "
                             "Нагружает машину — лучше с --jobs 1.")
    parser.add_argument("--delay", type=float, default=0.0, metavar="MS",
                        help="Лишняя задержка на кадр в основном потоке: с --realtime изображает медленную машину.")
    parser.add_argument("--contact-rail", action="store_true", help="Ось детектора — слияние с осью по контактному рельсу.")
    parser.add_argument("--poses", default="online", choices=("online", "none", "map", "kiss", "gicp"),
                        help="Позы для трекера КР и треков препятствий: online — small_gicp в цикле; "
                        "none — без поз (треки по пути); map/kiss/gicp — из файлов.")
    parser.add_argument("--rails", default="template", choices=("template", "seg"), help="Источник пар головок: шаблон или сеть.")
    parser.add_argument("--axis", default="map", choices=("map", "tunnel", "tunnel_hold", "hold", "ml", "far"),
                        help="С --contact-rail: map — карта пути в мире, tunnel — карта и ось по профилю тоннеля, "
                             "tunnel_hold — AxisHold и профиль тоннеля, hold — прежний AxisHold, "
                             "ml — рельсы, КР и габарит только по сети, far — тоннель и карта, дальше поправка "
                             "дальней сетью по виду сверху (fod/far_bev.py) и вторая карта пути.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fake:
        args.presets = list(FAKE_PRESETS)
    if args.axis == "ml":
        args.rails = "seg"
        args.contact_rail = True
    if not args.replay:
        ensure_ros_env()
    overrides = parse_overrides(args.set)
    if args.contact_rail:
        overrides.setdefault("s_max", S_MAX_CR)
    started = time.perf_counter()
    if args.video:
        from fod.bags import bag_time_range, resolve_bag

        kind = "fp" if args.no_inject else "inject"
        name = f"{args.video}_{'clean' if args.no_inject else args.preset}{f'_cr_{args.axis}' if args.contact_rail else ''}{'_seg' if args.rails == 'seg' else ''}.mp4"
        job = Job(
            args.video, kind, None if args.no_inject else args.preset, n_off=args.n_off, video=OUT_DIR / name, cr=args.contact_rail,
            poses=args.poses, axis=args.axis, rails=args.rails, every=args.every, realtime=args.realtime, delay_ms=args.delay,
        )
        if args.until:
            start, _stop = bag_time_range(resolve_bag(args.video))
            job.stop_ns = start + int(args.until * 1e9)
        print(f"Видео: {job.video}")
        report([run_job((job, overrides, "live"))])
        return 0
    jobs = build_jobs(args)
    mode = "replay" if args.replay else "cache" if args.cache else "live"
    print(f"Задач: {len(jobs)}, процессов: {args.jobs}, режим: {mode}, настройки: {overrides or 'по умолчанию'}")
    results = []
    # Готовые задачи — по файлу на задачу: после перезапуска машины они не пересчитываются.
    done_dir = OUT_DIR / "jobs" / args.tag
    done_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for job in jobs:
        path = done_dir / f"{job.name}.json"
        if mode != "replay" and path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            todo.append(job)
    if len(todo) < len(jobs):
        print(f"Уже готово {len(jobs) - len(todo)} задач из {done_dir}")
    # Длинные задачи вперёд, чтобы хвост не ждал одну.
    todo.sort(key=lambda j: (j.bag != "new_data", j.bag))
    with Pool(processes=max(1, min(args.jobs, len(todo)))) as pool:
        for result in pool.imap_unordered(run_job, [(job, overrides, mode) for job in todo]):
            results.append(result)
            if mode != "replay":
                (done_dir / f"{result['job']}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                label = result["preset"] or f"fp {result['part']}"
                print(f"  готово {result['bag']} {label}: {result['frames']} кадров", flush=True)
    summary = report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({"overrides": overrides, "summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nЗа {(time.perf_counter() - started) / 60.0:.1f} мин → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
