#!/usr/bin/env python3
"""Видео по записи: range image + вид сверху + панель скорости, рекомендуемой скорости,
дальности габарита и препятствий. Рядом кладутся те же CSV и summary.json, что у run_bag.

  python3 scripts/make_video.py roundT_doubleT
  python3 scripts/make_video.py /путь/к/записи --start 20 --until 80 --out results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from fod.mount import add_mount_args, mount_from_args
    from fod_ros.run_bag import run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="Папка записи, файл .db3/.mcap или имя записи в FOD_DATA.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--topic", help="Топик облака; по умолчанию первый PointCloud2.")
    parser.add_argument("--start", type=float, help="С какой секунды записи.")
    parser.add_argument("--until", type=float, help="До какой секунды записи.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    add_mount_args(parser)
    args = parser.parse_args()
    s = run(args.bag, args.out, args.topic, True, args.until, args.device, args.start, args.fps, mount_from_args(args))
    print(f"Видео: {args.out / s['bag'] / 'video.mp4'}  ({s['frames']} кадров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
