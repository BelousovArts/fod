#!/usr/bin/env python3
"""Прогнать запись ROS 2 целиком и сохранить результат по кадрам.

Читаются все кадры подряд, без пропусков (быстрее или медленнее реального
времени — неважно). В `<out>/<имя записи>/`:

- `detections.csv` — по кадру: статус (UNKNOWN / CLEAR / OBSTACLE), расстояние до
  ближайшего препятствия, до куда известна ось, скорость, время обработки, уровень
  (с ВНИМАНИЕ), рекомендуемая скорость и тормозной путь (`fod/telemetry.py`);
- `obstacles.csv` — подтверждённые препятствия по кадрам: номер трека, расстояние
  вдоль пути, смещение от оси, высота над полотном, координаты в системе лидара;
- `summary.json` — итог: доля кадров по статусам, каждое препятствие — на каком
  расстоянии впервые подтверждено и сколько кадров держалось;
- `video.mp4` с `--video` — range image и вид сверху с габаритом и препятствиями,
  панель скорости, рекомендуемой скорости и дальности габарита (`fod/report_view.py`).

  python3 -m fod_ros.run_bag /путь/к/записи [--out results] [--video] [--start 10] [--until 60]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class VideoWriter:
    def __init__(self, path: Path, fps: float = 10.0) -> None:
        from fod.report_view import ReportRenderer

        self.path, self.fps = path, fps
        self.ffmpeg = None
        self.shape = None
        self.renderer = ReportRenderer()

    def add(self, res, bag_name: str) -> None:
        import cv2

        from fod.video import even_dims, open_ffmpeg

        img = even_dims(self.renderer.render(res, bag_name))
        if self.shape is None:
            self.shape = (img.shape[1], img.shape[0])
            self.ffmpeg = open_ffmpeg(self.path, self.shape[0], self.shape[1], self.fps)
        elif (img.shape[1], img.shape[0]) != self.shape:
            img = cv2.resize(img, self.shape, interpolation=cv2.INTER_AREA)
        self.ffmpeg.stdin.write(img.tobytes())

    def close(self) -> None:
        if self.ffmpeg is not None and self.ffmpeg.stdin is not None:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()


from fod.mount import add_mount_args, mount_from_args  # noqa: E402


def run(bag: str, out_root: Path, topic: str | None, video: bool, until: float | None, device: str,
        start: float | None = None, fps: float = 10.0, mount=None) -> dict:
    from fod.bags import bag_time_range, iter_pointclouds, read_bag_topic, resolve_bag
    from fod.pipeline import Pipeline
    from fod.telemetry import telemetry

    bag_dir = resolve_bag(bag)
    topic = topic or read_bag_topic(bag_dir)
    out = out_root / bag_dir.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = bag_time_range(bag_dir)[0]
    begin = t0 + int(start * 1e9) if start else None
    stop = t0 + int(until * 1e9) if until else None
    print(f"Запись {bag_dir}, топик {topic} → {out}", flush=True)

    pipe = Pipeline(device=device, keep_view=video, mount=mount)
    writer = VideoWriter(out / "video.mp4", fps) if video else None
    items = ((i, msg, time.perf_counter()) for i, _ts, msg in iter_pointclouds(bag_dir, topic, begin, stop))
    counts = {"UNKNOWN": 0, "CLEAR": 0, "OBSTACLE": 0}
    levels = {"UNKNOWN": 0, "CLEAR": 0, "ATTENTION": 0, "OBSTACLE": 0}
    tracks: dict[int, dict] = {}
    ms: list[float] = []
    first_stamp = last_stamp = None
    started = time.perf_counter()
    with (out / "detections.csv").open("w", newline="", encoding="utf-8") as f_det, \
         (out / "obstacles.csv").open("w", newline="", encoding="utf-8") as f_obj:
        det_csv, obj_csv = csv.writer(f_det), csv.writer(f_obj)
        det_csv.writerow(["frame", "stamp", "status", "distance_m", "obstacles", "reach_m", "speed_kmh", "ms",
                          "level", "safe_kmh", "stopping_m", "overspeed", "suspect_m"])
        obj_csv.writerow(["frame", "stamp", "track_id", "distance_m", "offset_m", "height_m", "points", "x", "y", "z",
                          "first_seen_m", "first_confirmed_m"])
        try:
            for res in pipe.run(items):
                first_stamp = res.stamp if first_stamp is None else first_stamp
                last_stamp = res.stamp
                counts[res.status] = counts.get(res.status, 0) + 1
                tel = telemetry(res)
                levels[tel.level] += 1
                ms.append(res.ms)
                dist = f"{res.distance:.2f}" if np.isfinite(res.distance) else ""
                det_csv.writerow([res.index, f"{res.stamp:.6f}", res.status, dist, len(res.obstacles),
                                  f"{res.reach:.1f}", f"{res.speed_kmh:.1f}", f"{res.ms:.1f}", tel.level,
                                  "" if not np.isfinite(tel.safe_kmh) else f"{tel.safe_kmh:.1f}", f"{tel.stopping_m:.1f}", int(tel.overspeed),
                                  "" if not np.isfinite(tel.suspect) else f"{tel.suspect:.2f}"])
                for o in res.obstacles:
                    obj_csv.writerow([res.index, f"{res.stamp:.6f}", o.track_id, f"{o.distance:.2f}", f"{o.offset:.2f}",
                                      f"{o.height:.2f}", o.n_points, f"{o.x:.2f}", f"{o.y:.2f}", f"{o.z:.2f}",
                                      f"{o.first_seen:.1f}", f"{o.first_confirmed:.1f}"])
                    t = tracks.setdefault(o.track_id, {"first_frame": res.index, "first_stamp": res.stamp,
                                                       "first_seen_m": round(o.first_seen, 1),
                                                       "first_distance_m": round(o.distance, 1), "frames": 0})
                    t["frames"] += 1
                    t["last_frame"], t["last_distance_m"] = res.index, round(o.distance, 1)
                if writer is not None:
                    writer.add(res, bag_dir.name)
                if len(ms) % 100 == 0:
                    print(f"  кадр {res.index}: {res.status}, {np.median(ms[-100:]):.0f} мс на кадр", flush=True)
        finally:
            if writer is not None:
                writer.close()
    wall = time.perf_counter() - started
    frames = sum(counts.values())
    summary = {
        "bag": bag_dir.name,
        "topic": topic,
        "frames": frames,
        "duration_s": round((last_stamp - first_stamp), 1) if frames else 0.0,
        "status_frames": counts,
        "level_frames": levels,
        "ready_share": round(1.0 - counts["UNKNOWN"] / max(frames, 1), 3),
        "obstacles": [{"track_id": k, **v} for k, v in sorted(tracks.items(), key=lambda kv: kv[1]["first_frame"])],
        "ms_per_frame": {"median": round(float(np.median(ms)), 1), "p95": round(float(np.percentile(ms, 95)), 1)} if ms else None,
        "wall_s": round(wall, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="Папка записи (с metadata.yaml), файл .db3/.mcap или имя записи в FOD_DATA.")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Куда писать результат.")
    parser.add_argument("--topic", help="Топик облака; по умолчанию первый PointCloud2 в записи.")
    parser.add_argument("--video", action="store_true", help="Сохранить video.mp4.")
    parser.add_argument("--start", type=float, help="Начать с этой секунды записи.")
    parser.add_argument("--until", type=float, help="Закончить на этой секунде записи.")
    parser.add_argument("--fps", type=float, default=10.0, help="Частота видео.")
    parser.add_argument("--device", default="cuda", help="cuda или cpu.")
    add_mount_args(parser)
    args = parser.parse_args()
    s = run(args.bag, args.out, args.topic, args.video, args.until, args.device, args.start, args.fps, mount_from_args(args))
    c = s["status_frames"]
    print(f"\nКадров {s['frames']} за {s['duration_s']} с записи: готов {100 * s['ready_share']:.0f} %, "
          f"СВОБОДНО {c['CLEAR']}, ПРЕПЯТСТВИЕ {c['OBSTACLE']}, не готов {c['UNKNOWN']}")
    for o in s["obstacles"]:
        print(f"  препятствие {o['track_id']}: впервые замечено на {o['first_seen_m']} м, "
              f"подтверждено на {o['first_distance_m']} м (кадр {o['first_frame']}), "
              f"держалось {o['frames']} кадров")
    if s["ms_per_frame"]:
        print(f"Время на кадр: медиана {s['ms_per_frame']['median']} мс, p95 {s['ms_per_frame']['p95']} мс; всего {s['wall_s']} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
