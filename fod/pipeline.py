"""Весь тракт на потоке облаков: одометрия, рельсы, ось до 150 м, габарит, препятствия.

То же, что стенд с `--contact-rail --axis far --rails seg` и позами small_gicp
(`scripts/obstacle_bench.py`), без вставки объектов. Три потока: чтение и
развёртка тоннеля → сеть рельсов и small_gicp → ось, контактный рельс, тоннель,
детектор. `live=True` — живой режим: этапы берут кадр по запросу и точно ко
времени (`fod/prefetch.py`), источник обычно — ячейка последнего кадра
(`fod/latest.py`), так что при нехватке времени старые кадры пропускаются.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

AXIS_GRID = np.arange(0.0, 160.5, 0.5)
S_MAX = 150.0                 # дальше детектор не смотрит
RAIL_CAP = 80.0               # до куда фильтр рельсов предсказывает пары
GAP_S = 0.55                  # разрыв потока → всё с нуля
SUSPECT_HITS = 2              # неподтверждённый трек с таким числом попаданий — подозрение


@dataclass
class Obstacle:
    track_id: int
    distance: float           # вдоль пути до ближней грани, м
    offset: float             # от оси пути, м (+ влево, по x тракта)
    height: float             # над нормой полотна, м
    n_points: int
    x: float                  # в системе тракта (вперёд −y, влево +x, вверх +z)
    y: float
    z: float
    first_seen: float = float("nan")       # на каком расстоянии трек впервые появился, м
    first_confirmed: float = float("nan")  # на каком расстоянии впервые подтверждён, м


@dataclass
class FrameResult:
    index: int
    stamp: float
    status: str               # UNKNOWN / CLEAR / OBSTACLE
    distance: float           # до ближайшего подтверждённого, м; NaN — нет
    obstacles: list[Obstacle]
    reach: float              # до куда известна ось (и смотрит детектор), м
    speed_kmh: float
    ms: float                 # время основного этапа
    latency_ms: float         # от прихода сообщения до ответа
    frame_id: str
    axis_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    axis_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    axis_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    half_width: float = 0.0
    suspects: list[Obstacle] = field(default_factory=list)   # кандидаты, ещё не подтверждённые
    injected: list | None = None                              # что вставил стенд (`fod/stand.py`)
    pose: np.ndarray | None = field(default=None, repr=False)  # сенсор → мир, 4×4
    view: tuple | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "stamp": self.stamp,
            "status": self.status,
            "distance": None if not np.isfinite(self.distance) else round(self.distance, 2),
            "reach": round(self.reach, 1),
            "speed_kmh": round(self.speed_kmh, 1),
            "ms": round(self.ms, 1),
            "latency_ms": round(self.latency_ms, 1),
            "obstacles": [_rounded(o) for o in self.obstacles],
            "suspects": [_rounded(o) for o in self.suspects],
        }


def _rounded(o: Obstacle) -> dict:
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else round(v, 3) if isinstance(v, float) else v)
            for k, v in o.__dict__.items()}


class AxisSamples:
    """Ось на `AXIS_GRID` в виде, который ждёт `ObstacleDetector`."""

    def __init__(self, axis_n: np.ndarray, axis_z: np.ndarray, reach: float) -> None:
        self.axis_n = axis_n
        self.locked = axis_n.size > 0
        self.reach = reach
        self.head = (lambda s: np.interp(s, AXIS_GRID, axis_z)) if axis_z.size else None

    def n(self, s):
        return np.interp(s, AXIS_GRID, self.axis_n)


class Pipeline:
    """`injector(pipeline, cloud, motion)` → (облако, сведения) — стенд вставляет объекты перед
    основным этапом; одометрия, small_gicp и сеть рельсов считаются по чистому облаку.
    `mount` — установка лидара (`fod/mount.py`): облако сразу переводится в систему тракта
    (вперёд −y, вверх +z), все результаты — в ней."""

    def __init__(self, device: str = "cuda", keep_view: bool = False, obstacle_config=None, injector=None, mount=None) -> None:
        from fod.contact_rail import ContactRailConfig
        from fod.mount import Mount
        from fod.far_bev import FarAxis
        from fod.lidar_odometry import GicpOdometry
        from fod.obstacles import load_detector_config
        from fod.rail_seg_detect import SegRailDetector

        self.device = device
        self.keep_view = keep_view
        self.injector = injector
        self.mount = mount or Mount()
        self.obstacle_cfg = obstacle_config if obstacle_config is not None else load_detector_config()
        self.cr_cfg = ContactRailConfig()
        self.rails = SegRailDetector(device=device)
        self.far = FarAxis(device=device)
        self.icp = GicpOdometry()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="icp")
        self._reset_motion()
        self._reset_main()

    # --- состояние ------------------------------------------------------------

    def _reset_motion(self) -> None:
        from fod.odometry import EgoMotion

        self.odom = EgoMotion()
        self._last_stamp: float | None = None
        from fod.organize import Organizer

        self.organizer = Organizer()

    def _reset_main(self) -> None:
        from fod.contact_rail import ContactRailTracker
        from fod.far_bev import FrameBuffer
        from fod.obstacles import ObstacleDetector
        from fod.rail_axis_filter import RailTracker
        from fod.track_map import TrackMap, TrackMapConfig
        from fod.tunnel_axis import TunnelAxis

        gpu = self.device if self.device.startswith("cuda") else None
        self.tracker = RailTracker(detector=self.rails, predict_m=RAIL_CAP)
        self.placer = ObstacleDetector()          # только подгонка высоты головок
        self.detector = ObstacleDetector(self.obstacle_cfg)
        self.cr = ContactRailTracker(self.cr_cfg, device=gpu)
        self.hold = TrackMap()
        self.tunnel = TunnelAxis(device=gpu)
        self.far_buf = FrameBuffer(device=gpu)
        self.far_map = TrackMap(TrackMapConfig(s_cap=160.0))
        self.last_axis = (np.zeros(0), np.zeros(0))   # ось прошлого кадра (n, z) на AXIS_GRID
        self._first: dict[int, list[float]] = {}      # трек → [впервые виден, впервые подтверждён], м

    # --- этапы ----------------------------------------------------------------

    def _motion(self, items: Iterable[tuple]) -> Iterator[tuple]:
        """Этап 1: сообщение → облако, одометрия по развёртке, компенсация движения."""
        from fod.cloud import from_msg
        from fod.odometry import deskew_xyz, stamp_of

        for index, msg, t_in in items:
            cloud = self.mount.apply(self.organizer(from_msg(msg)))
            stamp = stamp_of(cloud)
            last = self._last_stamp
            # Разрыв или время назад (запись пошла по кругу) — всё с нуля.
            reset = last is not None and (stamp - last > GAP_S or stamp < last)
            if reset:
                self._reset_motion()
            self._last_stamp = stamp
            motion = self.odom.step(cloud.xyz, cloud.intensity, stamp)
            xyz = deskew_xyz(cloud.xyz, cloud.timestamp, motion.v)
            yield index, msg.header.frame_id, cloud, stamp, reset, motion, xyz, t_in

    def _prepare(self, frames: Iterable[tuple]) -> Iterator[tuple]:
        """Этап 2: сеть рельсов и small_gicp одновременно."""
        for item in frames:
            _index, _fid, cloud, _stamp, reset, motion, xyz, _t_in = item
            if reset:
                self.icp.reset()
            icp_job = self._pool.submit(self.icp.step, xyz[np.isfinite(xyz).all(axis=1)], motion.travelled)
            pre = (xyz, *self.rails.classify_only(xyz, cloud.intensity))
            yield (*item, pre, icp_job.result())

    def _main(self, item: tuple) -> FrameResult:
        """Этап 3: ось, контактный рельс, тоннель, дальняя ось, детектор."""
        from fod.contact_rail import axis_state
        from fod.tunnel_axis import fuse_tunnel

        index, frame_id, cloud, stamp, reset, motion, xyz, t_in, pre, pose = item
        started = time.perf_counter()
        if reset:
            self._reset_main()
        grid = AXIS_GRID
        clean = None
        injected = None
        if self.injector is not None:
            from fod.odometry import deskew_xyz

            cloud_in, injected = self.injector(self, cloud, motion)
            if cloud_in is not cloud:
                clean, cloud = (xyz, cloud.intensity), cloud_in
                xyz = deskew_xyz(cloud.xyz, cloud.timestamp, motion.v)
        rail = self.tracker.step(xyz, cloud.intensity, motion, pre=pre)
        self.placer._fit_head(rail.marks)
        # Сети рельсов нужен порядок колец, дальше — нет. Пустые лучи (0, 0, 0) дальше никто не
        # берёт (все отборы — s > 4…25 м), а у Hesai их больше половины кадра.
        valid = xyz.any(axis=1)
        intensity = cloud.intensity
        if not valid.all():
            xyz, intensity = xyz[valid], intensity[valid]
        if pose is not None:
            if clean is None:
                self.far_buf.push(xyz, intensity, pose)
            else:
                ok = clean[0].any(axis=1)
                self.far_buf.push(clean[0][ok], clean[1][ok], pose)
        filt = self.tracker.filter
        locked = filt.locked
        axis_n = np.asarray(filt.n(grid), dtype=np.float64) if locked else np.zeros(0)
        axis_z = np.zeros(0)
        reach = RAIL_CAP
        ok = locked and len(rail.marks) >= 4 and self.placer.head_coeff is not None
        cr_frame = self.cr.step(xyz, filt if ok else None, self.placer if ok else None, pose)
        if ok:
            rail_reach = max(float(m.s) for m in rail.marks)
            axis_n, axis_z, reach, sigma = axis_state(filt, self.placer, cr_frame, rail_reach, grid, self.cr_cfg, S_MAX, RAIL_CAP)
            slices = self.tunnel.step(xyz, grid, axis_n, axis_z, reach)
            axis_n, sigma, reach = fuse_tunnel(grid, axis_n, sigma, reach, slices)
            t_reach, t_sigma = reach, sigma
            near = min(rail_reach, RAIL_CAP)
            axis_n, axis_z, reach = self.hold.update(pose, grid, axis_n, axis_z, sigma, reach, near)
            if len(self.far_buf):
                axis_n, axis_z, reach, f_sigma = self.far(self.far_buf, pose, grid, axis_n, axis_z, reach)
                sigma = np.where(grid <= t_reach, t_sigma, f_sigma)
                axis_n, axis_z, reach = self.far_map.update(pose, grid, axis_n, axis_z, sigma, reach, near)
        elif locked:
            self.hold.reset()
            self.tunnel.reset()
            self.far_map.reset()
        axis = AxisSamples(axis_n, axis_z, reach)
        det = self.detector.step(xyz, rail, axis if axis.locked else None, float(motion.s), pose)
        alive = set()
        for t in self.detector.tracks:
            first = self._first.setdefault(int(t.track_id), [float(t.s), float("nan")])
            if t.confirmed and not np.isfinite(first[1]):
                first[1] = float(t.s)
            alive.add(int(t.track_id))
        for k in [k for k in self._first if k not in alive]:
            del self._first[k]
        obstacles = [self._obstacle(t, axis) for t in det.confirmed if self.detector.cfg.s_min * 0.5 < t.s < self.detector.cfg.s_max]
        obstacles.sort(key=lambda o: o.distance)
        lo, hi = self.detector.cfg.s_min * 0.5, min(self.detector.cfg.s_max, det.s_max if det.ready else 0.0)
        suspects = [self._obstacle(t, axis) for t in self.detector.tracks
                    if not t.confirmed and sum(t.hits) >= SUSPECT_HITS and lo < t.s < hi] if det.ready else []
        suspects.sort(key=lambda o: o.distance)
        done = time.perf_counter()
        keep = axis_n.size > 0
        self.last_axis = (axis_n, axis_z) if keep else (np.zeros(0), np.zeros(0))
        return FrameResult(
            index=int(index),
            stamp=float(stamp),
            status=det.status,
            distance=float(det.distance),
            obstacles=obstacles,
            reach=float(det.s_max) if det.ready else 0.0,
            speed_kmh=float(motion.kmh),
            ms=1e3 * (done - started),
            latency_ms=1e3 * (done - t_in),
            frame_id=frame_id,
            axis_s=grid if keep else np.zeros(0),
            axis_n=axis_n if keep else np.zeros(0),
            axis_z=axis_z if keep and axis_z.size else np.zeros(0),
            half_width=float(self.detector.cfg.half_width),
            suspects=suspects,
            injected=injected,
            pose=None if pose is None else np.asarray(pose, dtype=np.float64),
            view=(xyz, intensity, cloud, rail, motion, cr_frame, det, self.detector) if self.keep_view else None,
        )

    def _obstacle(self, track, axis: AxisSamples) -> Obstacle:
        s = float(track.s)
        x = float(track.n + axis.n(max(s, 0.0))) if axis.locked else float(track.n)
        head = axis.head(s) if axis.head is not None else self.detector.head_z(s)
        first = self._first.get(int(track.track_id), [float("nan"), float("nan")])
        return Obstacle(
            track_id=int(track.track_id),
            distance=s,
            offset=float(track.n),
            height=float(track.height),
            n_points=int(track.n_points),
            x=x,
            y=-s,
            z=float(head + 0.5 * track.height),
            first_seen=first[0],
            first_confirmed=first[1],
        )

    # --- запуск ---------------------------------------------------------------

    def run(self, items: Iterable[tuple], live: bool = False) -> Iterator[FrameResult]:
        """`items` — (номер, PointCloud2, perf_counter прихода)."""
        from fod.prefetch import prefetch

        depth = 0 if live else 2
        frames = prefetch(self._prepare(prefetch(self._motion(items), depth)), depth)
        for item in frames:
            yield self._main(item)
