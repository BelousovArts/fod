#!/usr/bin/env python3
"""Стенд: своя запись + вставленные препятствия → как их видит детектор.

Объекты — из сценария YAML (`--scenario`, формат в `fod/stand.py`) или коротко
`--object ТИП@ДАЛЬНОСТЬ` (ставить за столько метров впереди, после проезда — снова).
В `<out>/<запись>/stand_<сценарий>/`: `report.json` (по типам и по подъездам: первое
попадание лучей, подозрение, подтверждение, пропуски; ложные тревоги), `episodes.csv`,
`frames.csv` и с `--video` — `video.mp4`.

  python3 scripts/stand.py roundT_doubleT --object person@155 --object crate@155
  python3 scripts/stand.py doubleT_platform --scenario config/stand_example.yaml --video
  python3 scripts/stand.py doubleT_obstacle --object cube@40     # на стоянке — один объект
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from fod.mount import add_mount_args, mount_from_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", nargs="?", help="Папка записи, файл .db3/.mcap или имя записи в FOD_DATA.")
    parser.add_argument("--scenario", help="YAML со списком объектов.")
    parser.add_argument("--object", action="append", default=[], metavar="ТИП@М",
                        help="Объект за столько метров впереди, повторять после проезда (можно несколько).")
    parser.add_argument("--list", action="store_true", help="Показать типы объектов.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--topic")
    parser.add_argument("--start", type=float)
    parser.add_argument("--until", type=float)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", default="cuda")
    add_mount_args(parser)
    args = parser.parse_args()

    from fod.objects import FAKE_OBJECTS, PRESETS

    if args.list:
        for k, v in PRESETS.items():
            print(f"  {k:10s} {v['kind']:7s} {'x'.join(f'{x:g}' for x in v['size'])} м")
        for k, v in FAKE_OBJECTS.items():
            print(f"  {k:14s} {v['kind']:4s} {'x'.join(f'{x:g}' for x in v['size'])} м, от оси {v['n']:+.2f} м "
                  f"({'на полу' if v['rest'] == 'floor' else f'над головками {v.get(chr(117), 0):.2f} м'})")
        return 0
    if not args.bag:
        parser.error("нужен бэг (или --list)")

    from fod.bags import bag_time_range, iter_pointclouds, read_bag_topic, resolve_bag
    from fod.pipeline import Pipeline
    from fod.stand import Injector, StandReport, load_scenario

    specs = load_scenario(args.scenario) if args.scenario else []
    specs += load_scenario(args.object)
    if not specs:
        parser.error("нужен --scenario или хотя бы один --object")
    tag = Path(args.scenario).stem if args.scenario else "_".join(o.replace("@", "") for o in args.object)

    bag_dir = resolve_bag(args.bag)
    topic = args.topic or read_bag_topic(bag_dir)
    t0 = bag_time_range(bag_dir)[0]
    begin = t0 + int(args.start * 1e9) if args.start else None
    stop = t0 + int(args.until * 1e9) if args.until else None
    out = args.out / bag_dir.name / f"stand_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Запись {bag_dir}, объекты: {', '.join(s.name for s in specs)} → {out}", flush=True)

    injector = Injector(specs, seed=bag_dir.name)
    pipe = Pipeline(device=args.device, keep_view=args.video, injector=injector, mount=mount_from_args(args))
    writer = None
    if args.video:
        from fod_ros.run_bag import VideoWriter

        writer = VideoWriter(out / "video.mp4")
    report = StandReport()
    items = ((i, msg, 0.0) for i, _ts, msg in iter_pointclouds(bag_dir, topic, begin, stop))
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "stamp", "status", "distance_m", "objects", "nearest_object", "object_s_m", "object_hits"])
        try:
            for res in pipe.run(items):
                report.step(res)
                near = min(res.injected or [], key=lambda p: p.s, default=None)
                dist = f"{res.distance:.2f}" if np.isfinite(res.distance) else ""
                wr.writerow([res.index, f"{res.stamp:.6f}", res.status, dist, len(res.injected or []),
                             near.name if near else "", f"{near.s:.2f}" if near else "", near.hits if near else ""])
                if writer is not None:
                    writer.add(res, bag_dir.name)
                if report.frames % 100 == 0:
                    print(f"  кадр {res.index}: подъездов {len(report.episodes)}", flush=True)
        finally:
            if writer is not None:
                writer.close()

    s = report.summary()
    s["bag"] = bag_dir.name
    s["objects"] = [spec.__dict__ for spec in specs]
    (out / "report.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        if s["episodes"]:
            wr = csv.DictWriter(fh, fieldnames=list(s["episodes"][0]))
            wr.writeheader()
            wr.writerows(s["episodes"])

    fmt = lambda v: "   —" if v is None else f"{v:5.1f}"  # noqa: E731
    print(f"\n{'объект':16s} {'подъезд':>7s} {'лучи':>6s} {'подозр.':>7s} {'подтв.':>6s}  держалось")
    for e in s["episodes"]:
        held = "" if e["held"] is None else f"{100 * e['held']:.0f} %"
        print(f"{e['name']:16s} {e['episode']:7d} {fmt(e['first_ray_m']):>6s} {fmt(e['suspect_m']):>7s} "
              f"{fmt(e['confirmed_m']):>6s}  {held}")
    for t, v in s["by_type"].items():
        print(f"{t}: подъездов {v['episodes']}, подтверждено — медиана {v['confirmed_median_m']} м, "
              f"ближайшее {v['confirmed_min_m']} м, пропущено {v['missed']}")
    print(f"Ложные: {len(s['false_alarms'])} треков, {s['false_frames']} кадров из {s['frames']}")
    if writer is not None:
        print(f"Видео: {out / 'video.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
