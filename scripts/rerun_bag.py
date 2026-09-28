#!/usr/bin/env python3
"""Прогон записи в rerun: 3D, range image, вид сверху и показатели по времени.

- `train` — 3D в системе поезда (лидар в начале координат, вперёд — −y): облако в коридоре
  пути, путевые и контактный рельсы, габарит, ось, препятствия и подозрения, кузов поезда.
  Сцена не уезжает, вращать и масштабировать удобно. Облако — в нескольких раскрасках
  (`--colors`): `train/lidar_points/<раскраска>`, видна первая, остальные включаются
  глазком в левой панели rerun. Подписей в 3D нет — названия слоёв в той же панели;
- `world` — карта, которая копится по ходу (прореженная), траектория и препятствия в мире;
- `views/front` — передний сектор развёртки с габаритом и рельсами (как в видео),
  `views/range` — дальность на 360°, `views/top` — вид сверху до 150 м;
- `metrics/*` — скорость и рекомендуемая, до куда проверен габарит, тормозной путь,
  расстояние до препятствия и подозрения; `log` — смена уровня.

Подписи на английском, цвета — как в видео (`fod/obstacle_view.py`, `fod/report_view.py`).

  python3 scripts/rerun_bag.py roundT_doubleT                  # открыть окно rerun
  python3 scripts/rerun_bag.py doubleT_obstacle --save run.rrd # в файл: rerun run.rrd
  python3 scripts/rerun_bag.py /путь/к/записи --start 30 --until 90 --stride 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAP_EVERY = 20             # кадров на кусок накопленной карты
MAP_VOXEL = 0.3
CORRIDOR = (-25.0, 160.0, 12.0, -4.0, 6.0)   # s от, s до, |x|, z от, z до — облако в 3D поезда
TRAIN = (2.7, 20.0, 3.6)   # кузов: ширина, длина позади лидара, высота над головками


def rgb(bgr) -> tuple[int, int, int]:
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def palette() -> dict:
    from fod import obstacle_view as ov
    from fod import report_view as rv

    return {
        "rails": rgb(ov.COL_RAIL), "contact": rgb(ov.COL_CR), "gauge": rgb(ov.COL_GAUGE), "axis": rgb(ov.COL_AXIS),
        "obstacle": rgb(ov.COL_CONFIRMED), "suspect": rgb(ov.COL_SUSPECT), "object": rgb(ov.COL_OBJECT),
        "speed": rgb(rv.COL_SPEED), "safe": rgb(rv.COL_SAFE), "reach": rgb(rv.COL_REACH), "stop": rgb(rv.COL_STOP),
        "train": (150, 150, 150), "path": (90, 220, 120),
        "UNKNOWN": (150, 150, 150), "CLEAR": rgb(ov.COL_AXIS), "ATTENTION": rgb(ov.COL_SUSPECT),
        "OBSTACLE": rgb(ov.COL_CONFIRMED),
    }


def blueprint(colors: list[str]):
    import rerun.blueprint as rrb
    from rerun.blueprint.components import Visible

    hidden = {f"train/lidar_points/{m}": [Visible(False)] for m in colors[1:]}
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="train", name="3D - train frame (lidar at origin, forward = -y)",
                                  overrides=hidden),
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world", name="World - map and train path"),
                    rrb.TextLogView(origin="log", name="Level changes"),
                ),
                row_shares=[3, 1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="views/front", name="Front view (range image)"),
                rrb.Spatial2DView(origin="views/range", name="Range, 360 deg"),
                rrb.Spatial2DView(origin="views/top", name="Top view"),
                rrb.TimeSeriesView(origin="metrics/speed", name="Speed, km/h"),
                rrb.TimeSeriesView(origin="metrics/distance", name="Distances, m"),
                row_shares=[2, 1, 4, 2, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )


def setup_static(pal: dict) -> None:
    import rerun as rr

    series = {
        "metrics/speed/actual": ("speed", pal["speed"]),
        "metrics/speed/recommended": ("recommended speed", pal["safe"]),
        "metrics/distance/gauge_checked": ("gauge checked to", pal["reach"]),
        "metrics/distance/stopping": ("stopping distance", pal["stop"]),
        "metrics/other/ms": ("ms per frame", (180, 180, 180)),
    }
    for path, (name, color) in series.items():
        rr.log(path, rr.SeriesLine(name=name, color=color, width=2), static=True)
    rr.log("metrics/distance/obstacle", rr.SeriesPoint(name="obstacle", color=pal["obstacle"], marker_size=3), static=True)
    rr.log("metrics/distance/suspect", rr.SeriesPoint(name="suspect", color=pal["suspect"], marker_size=3), static=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("train", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def intensity_rgb(intensity: np.ndarray, vmax: float = 80.0, gamma: float = 0.5) -> np.ndarray:
    """Яркая шкала для тёмного фона 3D: слабые отражения — синие, сильные — жёлтые и красные."""
    import cv2

    t = np.clip(np.nan_to_num(intensity, nan=0.0) / vmax, 0.0, 1.0) ** gamma
    u8 = (40 + 215 * t).astype(np.uint8).reshape(1, -1)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]


COLOR_MODES = {
    "intensity": "по интенсивности отражения",
    "height": "по высоте над головками рельсов: синий — полотно, красный — 4 м и выше",
    "range": "по дальности от лидара",
    "gauge": "внутри габарита — белые, остальное — тёмное",
}


def point_colors(mode: str, pts: np.ndarray, inten: np.ndarray, res, head_z, cfg) -> np.ndarray:
    import cv2

    def turbo(t: np.ndarray) -> np.ndarray:
        u8 = (255 * np.clip(t, 0.0, 1.0)).astype(np.uint8).reshape(1, -1)
        return np.ascontiguousarray(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1])

    if mode == "intensity":
        return intensity_rgb(inten)
    s = np.clip(-pts[:, 1], 0.0, None)
    if mode == "range":
        r = np.sqrt(np.einsum("ij,ij->i", pts, pts))
        return turbo(np.log1p(r) / np.log1p(160.0))
    base = head_z(s) if head_z is not None else np.full(s.size, -1.2)
    u = pts[:, 2] - base
    if mode == "height":
        return turbo((u + 0.5) / 4.5)
    inside = np.zeros(s.size, bool)
    if res.axis_n.size and res.reach > 0:
        from fod.pipeline import AXIS_GRID

        n_rel = pts[:, 0] - np.interp(s, AXIS_GRID, res.axis_n)
        inside = (np.abs(n_rel) < res.half_width) & (u > 0.05) & (u < cfg.height_max) & (s < res.reach)
    out = np.tile(np.array([[70, 80, 100]], np.uint8), (s.size, 1))
    out[inside] = (255, 255, 255)
    return out


def voxel_pick(xyz: np.ndarray, size: float) -> np.ndarray:
    """По одной точке на воксель: вблизи лидара облако в разы плотнее, чем вдали."""
    keys = np.floor(xyz / size).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    return np.unique(flat, return_index=True)[1]


def track_lines(res, rail, cr_frame, head_z) -> tuple[list, list]:
    """Путевые рельсы (до последней найденной пары) и контактный рельс в системе сенсора."""
    from fod.obstacle_view import _found_tracks

    rails = []
    tracks = _found_tracks(rail) if head_z is not None else None
    for c in tracks or ():
        c = c[c[:, 0] > 2.0]
        if len(c) >= 2:
            rails.append(np.stack([c[:, 1], -c[:, 0], head_z(c[:, 0])], axis=1))
    contact = []
    if cr_frame is not None:
        for line in cr_frame.lines.values():
            if len(line) >= 2:
                contact.append(np.stack([line[:, 1], -line[:, 0], line[:, 2]], axis=1))
    return rails, contact


def gauge_lines(res, height: float) -> tuple[list, np.ndarray | None]:
    """Рёбра габарита (низ и верх с обеих сторон, поперечины через 10 м) и ось пути."""
    keep = res.axis_s <= res.reach
    s, n = res.axis_s[keep], res.axis_n[keep]
    z = res.axis_z[keep] if res.axis_z.size else np.full(s.size, -1.2)
    if s.size < 2:
        return [], None
    w = res.half_width
    edge = lambda off, dz: np.stack([n + off, -s, z + dz], axis=1)  # noqa: E731
    lines = [edge(-w, 0.0), edge(w, 0.0), edge(-w, height), edge(w, height)]
    for k in range(0, s.size, 20):
        a = np.array([[n[k] - w, -s[k], z[k]], [n[k] - w, -s[k], z[k] + height],
                      [n[k] + w, -s[k], z[k] + height], [n[k] + w, -s[k], z[k]]])
        lines.append(a)
    return lines, edge(0.0, 0.0)


def log_boxes(path: str, items, color) -> None:
    import rerun as rr

    if not items:
        rr.log(path, rr.Clear(recursive=False))
        return
    rr.log(path, rr.Boxes3D(
        centers=[(o.x, o.y, o.z) for o in items],
        half_sizes=[(0.3, 0.3, max(o.height, 0.2) / 2) for o in items],
        colors=[color] * len(items), radii=0.05,
    ))


def log_frame(res, tel, rng: np.random.Generator, args, state: dict) -> None:
    import rerun as rr

    from fod.obstacle_view import FRONT_H, TITLE_H
    from fod.pipeline import AXIS_GRID

    pal = state["pal"]
    xyz, intensity, cloud, rail, motion, cr_frame, det, detector = res.view
    rr.set_time_sequence("frame", res.index)
    rr.set_time_seconds("time", res.stamp - state["t0"])

    # --- 3D в системе поезда -----------------------------------------------------
    s0, s1, half, z0, z1 = CORRIDOR
    s_pt = -xyz[:, 1]
    sel = (s_pt > s0) & (s_pt < s1) & (np.abs(xyz[:, 0]) < half) & (xyz[:, 2] > z0) & (xyz[:, 2] < z1)
    pts, inten = xyz[sel], intensity[sel]
    if args.voxel > 0 and pts.shape[0]:
        keep = voxel_pick(pts, args.voxel)
        pts, inten = pts[keep], inten[keep]
    if pts.shape[0] > args.points:
        pick = rng.choice(pts.shape[0], args.points, replace=False)
        pts, inten = pts[pick], inten[pick]
    head_z = (lambda q, a=res.axis_z: np.interp(q, AXIS_GRID, a)) if res.axis_z.size else None  # noqa: E731
    pts32 = pts.astype(np.float32)
    for mode in args.colors:
        colors = point_colors(mode, pts, inten, res, head_z, detector.cfg)
        rr.log(f"train/lidar_points/{mode}", rr.Points3D(pts32, colors=colors, radii=rr.Radius.ui_points(1.5)))

    rails, contact = track_lines(res, rail, cr_frame, head_z)
    for name, lines, color in (("running_rails", rails, pal["rails"]), ("contact_rail", contact, pal["contact"])):
        if lines:
            rr.log(f"train/{name}", rr.LineStrips3D(lines, colors=[color] * len(lines), radii=0.04))
        else:
            rr.log(f"train/{name}", rr.Clear(recursive=False))
    gauge, centre = gauge_lines(res, detector.cfg.height_max) if res.axis_n.size and res.reach > 0 else ([], None)
    if gauge:
        rr.log("train/clearance_gauge", rr.LineStrips3D(gauge, colors=[pal["gauge"]] * len(gauge), radii=0.03))
        rr.log("train/centreline", rr.LineStrips3D([centre], colors=[pal["axis"]], radii=0.03))
    else:
        rr.log("train/clearance_gauge", rr.Clear(recursive=False))
        rr.log("train/centreline", rr.Clear(recursive=False))
    log_boxes("train/obstacles", res.obstacles, pal["obstacle"])
    log_boxes("train/suspects", res.suspects, pal["suspect"])
    if res.injected:
        rr.log("train/test_objects", rr.Points3D([(p.n, -p.s, 0.0) for p in res.injected], colors=pal["object"], radii=0.3))
    base = float(res.axis_z[0]) if res.axis_z.size else -1.2
    w, length, h = TRAIN
    rr.log("train/train_body", rr.Boxes3D(centers=[(0.0, length / 2, base + h / 2)], half_sizes=[(w / 2, length / 2, h / 2)],
                                          colors=[pal["train"]]))

    # --- мир ------------------------------------------------------------------------
    if res.pose is not None:
        T = res.pose
        rr.log("world/train", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        rr.log("world/train/body", rr.Boxes3D(centers=[(0.0, length / 2, base + h / 2)], half_sizes=[(w / 2, length / 2, h / 2)],
                                              colors=[pal["train"]]))
        state["track"].append(T[:3, 3].copy())
        if len(state["track"]) >= 2:
            rr.log("world/train_path", rr.LineStrips3D([np.array(state["track"])], colors=[pal["path"]], radii=0.15))
        near = np.einsum("ij,ij->i", xyz, xyz) < 60.0 ** 2
        state["chunk"].append(xyz[near][:: 4] @ T[:3, :3].T + T[:3, 3])
        state["chunk_i"].append(intensity[near][:: 4])
        if len(state["chunk"]) >= MAP_EVERY:
            flush_map(state)
        for o in res.obstacles:
            if o.track_id not in state["world_obstacles"]:
                state["world_obstacles"][o.track_id] = (T[:3, :3] @ np.array([o.x, o.y, o.z]) + T[:3, 3], o.distance)
        if state["world_obstacles"]:
            items = list(state["world_obstacles"].items())
            rr.log("world/obstacles", rr.Points3D([p for _k, (p, _d) in items], colors=pal["obstacle"], radii=0.6))

    # --- картинки ------------------------------------------------------------------
    frame = state["renderer"].render(res, state["bag"])
    front = frame[TITLE_H : TITLE_H + FRONT_H, :, ::-1]
    top = frame[TITLE_H + FRONT_H : -state["panel_h"], :, ::-1]
    rr.log("views/front", rr.Image(np.ascontiguousarray(front)).compress(jpeg_quality=85))
    rr.log("views/top", rr.Image(np.ascontiguousarray(top)).compress(jpeg_quality=85))
    if args.raw_range:
        from fod.colormaps import colorize_range
        from fod.range_image import build_range_image

        img = colorize_range(build_range_image(cloud).range, 80.0)[:, :, ::-1]
        rr.log("views/range", rr.Image(np.ascontiguousarray(img)).compress(jpeg_quality=85))

    # --- показатели ------------------------------------------------------------------
    rr.log("metrics/speed/actual", rr.Scalar(tel.speed_kmh))
    if math.isfinite(tel.safe_kmh):
        rr.log("metrics/speed/recommended", rr.Scalar(tel.safe_kmh))
    rr.log("metrics/distance/gauge_checked", rr.Scalar(tel.reach))
    rr.log("metrics/distance/stopping", rr.Scalar(tel.stopping_m))
    if math.isfinite(tel.distance):
        rr.log("metrics/distance/obstacle", rr.Scalar(tel.distance))
    if math.isfinite(tel.suspect):
        rr.log("metrics/distance/suspect", rr.Scalar(tel.suspect))
    rr.log("metrics/other/ms", rr.Scalar(res.ms))
    if tel.level != state["level"]:
        text = tel.level
        if tel.level == "OBSTACLE" and math.isfinite(tel.distance):
            text += f" at {tel.distance:.1f} m"
            if res.obstacles and math.isfinite(res.obstacles[0].first_seen):
                text += f" (first seen at {res.obstacles[0].first_seen:.0f} m)"
        elif tel.level == "ATTENTION" and math.isfinite(tel.suspect):
            text += f" - suspect at {tel.suspect:.1f} m"
        lvl = {"OBSTACLE": "ERROR", "ATTENTION": "WARN", "UNKNOWN": "DEBUG"}.get(tel.level, "INFO")
        rr.log("log", rr.TextLog(text, level=lvl))
        state["level"] = tel.level


def flush_map(state: dict) -> None:
    import rerun as rr

    if not state["chunk"]:
        return
    pts = np.concatenate(state["chunk"])
    inten = np.concatenate(state["chunk_i"])
    keep = voxel_pick(pts, MAP_VOXEL) if pts.shape[0] else np.zeros(0, np.int64)
    k = state["chunks"]
    # Кусок карты — отдельная сущность: уже показанные остаются на экране и дальше по времени.
    rr.log(f"world/map/part_{k:04d}", rr.Points3D(pts[keep].astype(np.float32), colors=intensity_rgb(inten[keep]),
                                                   radii=rr.Radius.ui_points(1.0)))
    state["chunks"] += 1
    state["chunk"], state["chunk_i"] = [], []


def main() -> int:
    from fod.mount import add_mount_args, mount_from_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="Папка записи, файл .db3/.mcap или имя записи в FOD_DATA.")
    parser.add_argument("--save", type=Path, help="Писать в файл .rrd вместо окна.")
    parser.add_argument("--topic", help="Топик облака; по умолчанию первый PointCloud2.")
    parser.add_argument("--start", type=float, help="С какой секунды записи.")
    parser.add_argument("--until", type=float, help="До какой секунды записи.")
    parser.add_argument("--stride", type=int, default=1, help="Показывать каждый N-й кадр (тракт считает все).")
    parser.add_argument("--points", type=int, default=100_000, help="Не больше стольких точек облака на кадр в 3D.")
    parser.add_argument("--voxel", type=float, default=0.1, help="Прореживание облака в 3D, м (0 — нет).")
    parser.add_argument("--no-raw-range", dest="raw_range", action="store_false", help="Без развёртки дальности 360°.")
    parser.add_argument("--colors", default="intensity,height,range,gauge",
                        help="Раскраски облака в 3D через запятую, первая видна сразу: " + "; ".join(f"{k} — {v}" for k, v in COLOR_MODES.items()))
    parser.add_argument("--device", default="cuda")
    add_mount_args(parser)
    args = parser.parse_args()
    args.colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    bad = [c for c in args.colors if c not in COLOR_MODES]
    if bad or not args.colors:
        parser.error(f"неизвестные раскраски: {', '.join(bad) or '—'}; есть: {', '.join(COLOR_MODES)}")

    import rerun as rr

    from fod.bags import bag_time_range, iter_pointclouds, read_bag_topic, resolve_bag
    from fod.pipeline import Pipeline
    from fod.report_view import PANEL_H, ReportRenderer
    from fod.telemetry import telemetry

    bag_dir = resolve_bag(args.bag)
    topic = args.topic or read_bag_topic(bag_dir)
    t0 = bag_time_range(bag_dir)[0]
    begin = t0 + int(args.start * 1e9) if args.start else None
    stop = t0 + int(args.until * 1e9) if args.until else None

    rr.init(f"fod {bag_dir.name}", spawn=args.save is None)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save)
    rr.send_blueprint(blueprint(args.colors), make_active=True, make_default=True)
    pal = palette()
    setup_static(pal)

    print(f"Запись {bag_dir}, топик {topic}", flush=True)
    pipe = Pipeline(device=args.device, keep_view=True, mount=mount_from_args(args))
    items = ((i, msg, 0.0) for i, _ts, msg in iter_pointclouds(bag_dir, topic, begin, stop))
    rng = np.random.default_rng(0)
    state = {"t0": None, "track": [], "level": None, "renderer": ReportRenderer(), "bag": bag_dir.name,
             "panel_h": PANEL_H, "pal": pal, "chunk": [], "chunk_i": [], "chunks": 0, "world_obstacles": {}}
    n = 0
    for res in pipe.run(items):
        tel = telemetry(res)
        if state["t0"] is None:
            state["t0"] = res.stamp
        if n % args.stride == 0:
            log_frame(res, tel, rng, args, state)
        else:
            state["renderer"].history.push(res.stamp, tel.speed_kmh, tel.safe_kmh, tel.reach, tel.distance,
                                           tel.stopping_m, tel.suspect)
        n += 1
        if n % 100 == 0:
            print(f"  кадр {res.index}: {tel.level}", flush=True)
    flush_map(state)
    print(f"Готово: {n} кадров" + (f", файл {args.save} (открыть: rerun {args.save})" if args.save else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
