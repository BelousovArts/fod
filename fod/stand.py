"""Стенд препятствий: объекты из сценария вставляются в облако записи, детектор — весь тракт.

Объект в сценарии (YAML, список `objects`):

    - type: person          # каталог fod/objects.py: min, cube, crate, suitcase, person, barrel,
                            #   cone, wire, sphere; или объекты записи заказчика f1_shield … f10_rod
      at: 120               # стоит на 120-м метре пути от начала записи (по одометрии)
      n: 0.0                # от оси пути, м (+ вправо по ходу)
      u: 0.0                # поднять над полом, м (для f*: над головками рельсов, как в каталоге)
      size: [0.5, 0.5, 1.75]  # необязательно: поперёк, вдоль, высота, м
      yaw: 0                # необязательно: поворот, градусы
      appear: 2             # появиться через столько секунд после того, как найден путь
    - type: crate
      ahead: 155            # вместо `at`: ставить за 155 м впереди, после проезда — снова

Объект появляется через `appear` с после того, как найден путь (ось есть): `at` — если к
этому моменту он ещё впереди, `ahead` — и заново после каждого проезда (на стоянке — один раз).
Норма полотна копится по 4–25 м перед поездом, поэтому на стоящем поезде предмет, который
лежал там с самого начала (`appear: 0`), детектор примет за полотно — как и настоящий.

Вставка лучевая (`fod/inject.py`): объект закрывает то, что за ним, и даёт тень; дальность
и интенсивность — как у настоящих поверхностей тоннеля. Одометрия, small_gicp и сеть
рельсов считаются по чистому облаку, всё остальное — по облаку с объектом.

По каждому подъезду: с какого расстояния лучи впервые попали в объект, когда детектор
заподозрил (неподтверждённый трек) и когда подтвердил. Ложные — подтверждённые
препятствия, которые не совпали ни с одним объектом.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

END_S = 3.0                 # объект ближе — проехали, подъезд окончен
FLOOR_BELOW_HEAD = 0.2      # пол ниже головок, если по точкам не найти
BASE_BELOW_HEAD = 0.12      # объекты каталога стоят на шпалах/полу чуть ниже головок


@dataclass
class ObjectSpec:
    type: str
    at: float | None = None
    ahead: float | None = None
    n: float = 0.0
    u: float | None = None
    size: tuple[float, float, float] | None = None
    yaw: float | None = None
    name: str = ""
    appear: float = 2.0

    def resolved(self) -> dict:
        """kind, size, yaw, rest (floor / head), u, n — с учётом каталога."""
        from fod.objects import FAKE_OBJECTS, PRESETS

        if self.type in FAKE_OBJECTS:
            spec = dict(FAKE_OBJECTS[self.type])
            base = {"kind": spec["kind"], "size": tuple(spec["size"]), "yaw": spec.get("yaw", 0.0),
                    "rest": spec["rest"], "u": spec.get("u", 0.0), "n": spec["n"], "reflect": True}
        elif self.type in PRESETS:
            spec = PRESETS[self.type]
            base = {"kind": spec["kind"], "size": tuple(spec["size"]), "yaw": 0.0, "rest": "floor",
                    "u": float(spec.get("u", 0.0)), "n": 0.0, "reflect": False, "intensity": spec["intensity"]}
        else:
            known = ", ".join(list(PRESETS) + list(FAKE_OBJECTS))
            raise ValueError(f"Неизвестный объект «{self.type}». Есть: {known}")
        if self.size is not None:
            base["size"] = tuple(float(v) for v in self.size)
        if self.yaw is not None:
            base["yaw"] = float(self.yaw)
        if self.u is not None:
            base["u"] = float(self.u)
        base["n"] += float(self.n)
        return base


def load_scenario(path_or_text: str | Path | list) -> list[ObjectSpec]:
    import yaml

    if isinstance(path_or_text, list):
        items = path_or_text
    else:
        p = Path(path_or_text)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else yaml.safe_load(str(path_or_text))
        items = data.get("objects", data) if isinstance(data, dict) else data
    specs = []
    for k, item in enumerate(items or []):
        if isinstance(item, str):
            kind, _, dist = item.partition("@")
            item = {"type": kind, "ahead": float(dist or 155.0)}
        spec = ObjectSpec(
            type=str(item["type"]),
            at=None if item.get("at") is None else float(item["at"]),
            ahead=None if item.get("ahead") is None else float(item["ahead"]),
            n=float(item.get("n", 0.0)),
            u=None if item.get("u") is None else float(item["u"]),
            size=None if item.get("size") is None else tuple(item["size"]),
            yaw=None if item.get("yaw") is None else float(item["yaw"]),
            name=str(item.get("name", f"{item['type']}_{k}")),
            appear=float(item.get("appear", 2.0)),
        )
        if (spec.at is None) == (spec.ahead is None):
            raise ValueError(f"У объекта {spec.name} нужно ровно одно из: at, ahead")
        spec.resolved()
        specs.append(spec)
    return specs


@dataclass
class _Live:
    spec: ObjectSpec
    geo: dict
    S: float                       # положение в одометрии (motion.s), м
    episode: int
    rng: np.random.Generator
    reflect: float
    floor_off: float | None = None


@dataclass
class Placement:
    """Что вставлено в этом кадре; `s` — вдоль пути до центра, `n` — от оси сенсора."""
    name: str
    type: str
    episode: int
    s: float
    n: float
    half_s: float
    half_n: float
    hits: int


class Injector:
    """Вызывается `Pipeline` на каждом кадре перед основным этапом."""

    def __init__(self, specs: list[ObjectSpec], seed: str = "") -> None:
        self.specs = specs
        self.seed = seed
        self.live: dict[str, _Live] = {}
        self.episodes: dict[str, int] = {s.name: -1 for s in specs}
        self.used: set[str] = set()      # `at` — только один подъезд
        self._last_s: float | None = None
        self._armed_at: float | None = None

    def __call__(self, pipe, cloud, motion):
        from fod.inject import estimate_floor_z, inject
        from fod.objects import REFLECT_MEDIAN, REFLECT_SIGMA, SceneObject
        from fod.odometry import stamp_of
        from fod.pipeline import AXIS_GRID

        s_odom = float(motion.s)
        if self._last_s is not None and s_odom < self._last_s - 1.0:
            self.live.clear()            # одометрия с нуля (разрыв записи)
        self._last_s = s_odom
        axis_n, axis_z = pipe.last_axis
        armed = axis_n.size > 0
        stamp = stamp_of(cloud)
        if not armed or (self._armed_at is not None and stamp < self._armed_at):
            self.live.clear()
            self._armed_at = None
            if not armed:
                return cloud, []
        if self._armed_at is None:
            self._armed_at = stamp
        for spec in self.specs:
            if spec.name in self.live or stamp - self._armed_at < spec.appear:
                continue
            if spec.at is not None:
                if spec.name in self.used or spec.at - s_odom < END_S + 1.0:
                    continue
                S = spec.at
                self.used.add(spec.name)
            else:
                if spec.name in self.used and motion.v < 1.0:
                    continue             # на стоянке ставим один раз
                S = s_odom + spec.ahead
                self.used.add(spec.name)
            self.episodes[spec.name] += 1
            ep = self.episodes[spec.name]
            rng = np.random.default_rng(zlib.crc32(f"{self.seed}/{spec.name}/{ep}".encode()))
            reflect = float(rng.lognormal(np.log(REFLECT_MEDIAN), REFLECT_SIGMA))
            self.live[spec.name] = _Live(spec, spec.resolved(), S, ep, rng, reflect)

        objects, meta = [], []
        z_floor = None
        for name, lv in list(self.live.items()):
            s_obj = lv.S - s_odom
            if s_obj < END_S:
                del self.live[name]
                continue
            if s_obj > AXIS_GRID[-1]:
                continue
            if z_floor is None:
                z_floor = estimate_floor_z(cloud.xyz)
            g = lv.geo
            head = float(np.interp(s_obj, AXIS_GRID, axis_z)) if axis_z.size else float(
                pipe.placer.head_z(float(np.clip(s_obj, 4.0, 60.0))))
            n_c = float(np.interp(s_obj, AXIS_GRID, axis_n)) + g["n"]
            if g["rest"] == "floor":
                off = _floor_below(cloud.xyz, s_obj, n_c, g["size"], g["yaw"], head)
                if off is not None:
                    lv.floor_off = off if lv.floor_off is None else 0.7 * lv.floor_off + 0.3 * off
                below = lv.floor_off if lv.floor_off is not None else -FLOOR_BELOW_HEAD
                base = head + (below if g["reflect"] else -BASE_BELOW_HEAD) + g["u"]
            else:
                base = head + g["u"]
            obj = SceneObject(kind=g["kind"], name=name, s=s_obj, n=n_c, u=base - z_floor, size=g["size"],
                              intensity=float(g.get("intensity", 70.0)), extra={"yaw": g["yaw"]})
            objects.append(obj)
            hn, hs = _half_extent(g["size"], g["yaw"])
            meta.append((lv, s_obj, n_c, hs, hn))
        if not objects:
            return cloud, []
        reflect = meta[0][0].reflect if all(m[0].spec.resolved()["reflect"] for m in meta) else None
        res = inject(cloud, objects, z_floor=z_floor, rng=meta[0][0].rng, reflect=reflect)
        placed = [Placement(lv.spec.name, lv.spec.type, lv.episode, s, n, hs, hn, int(res.hits.get(lv.spec.name, 0)))
                  for lv, s, n, hs, hn in meta]
        return res.cloud, placed


def _half_extent(size, yaw_deg: float) -> tuple[float, float]:
    """Полуразмеры по n и по s с учётом поворота."""
    w, length = size[0], size[1]
    yaw = np.deg2rad(yaw_deg)
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return 0.5 * (w * c + length * s), 0.5 * (w * s + length * c)


def _floor_below(xyz: np.ndarray, s_obj: float, n_c: float, size, yaw: float, head: float) -> float | None:
    """Пол под объектом относительно головок по точкам кадра; None — точек мало."""
    hn, hs = _half_extent(size, yaw)
    s = -xyz[:, 1]
    near = (np.abs(s - s_obj) < max(hs + 0.5, 0.02 * s_obj)) & (np.abs(xyz[:, 0] - n_c) < hn + 0.2)
    z = xyz[near, 2]
    z = z[z < head + 0.3]
    if z.size < 6:
        return None
    return float(np.percentile(z, 10)) - head


# --- оценка ----------------------------------------------------------------------


def matches(det_s: float, det_x: float, p: Placement) -> bool:
    """Как в `scripts/obstacle_bench.py`: допуск растёт с дальностью."""
    return abs(det_s - p.s) < 2.0 + 0.03 * p.s + p.half_s and abs(det_x - p.n) < 0.8 + p.half_n


def assign(dets, placed: list[Placement]) -> dict[int, list]:
    """Каждое обнаружение — только ближайшему по дальности подходящему объекту."""
    out: dict[int, list] = {k: [] for k in range(len(placed))}
    for o in dets:
        best = [(abs(o.distance - p.s), k) for k, p in enumerate(placed) if matches(o.distance, o.x, p)]
        if best:
            out[min(best)[1]].append(o)
    return out


@dataclass
class Episode:
    name: str
    type: str
    episode: int
    first_ray: float = float("nan")    # расстояние, м
    suspect: float = float("nan")
    confirmed: float = float("nan")
    frames: int = 0
    shown: int = 0                     # кадров с подтверждением после первого подтверждения
    after: int = 0                     # кадров после первого подтверждения
    last_s: float = float("inf")

    def as_dict(self) -> dict:
        r = lambda v: None if not np.isfinite(v) else round(float(v), 1)  # noqa: E731
        return {
            "name": self.name, "type": self.type, "episode": self.episode,
            "first_ray_m": r(self.first_ray), "suspect_m": r(self.suspect), "confirmed_m": r(self.confirmed),
            "missed": not np.isfinite(self.confirmed), "frames": self.frames,
            "held": round(self.shown / self.after, 3) if self.after else None,
            "closest_m": r(self.last_s),
        }


@dataclass
class StandReport:
    episodes: dict[tuple[str, int], Episode] = field(default_factory=dict)
    false_tracks: dict[int, dict] = field(default_factory=dict)
    frames: int = 0
    alarm_frames: int = 0
    false_frames: int = 0

    def step(self, res) -> None:
        placed: list[Placement] = res.injected or []
        self.frames += 1
        seen_by = assign(res.suspects + res.obstacles, placed)
        shown_by = assign(res.obstacles, placed)
        for k, p in enumerate(placed):
            ep = self.episodes.setdefault((p.name, p.episode), Episode(p.name, p.type, p.episode))
            ep.frames += 1
            ep.last_s = min(ep.last_s, p.s)
            if p.hits > 0 and not np.isfinite(ep.first_ray):
                ep.first_ray = p.s
            seen = bool(seen_by[k])
            if seen and not np.isfinite(ep.suspect):
                ep.suspect = p.s
            shown = bool(shown_by[k])
            if shown and not np.isfinite(ep.confirmed):
                ep.confirmed = p.s
                if not np.isfinite(ep.suspect):
                    ep.suspect = p.s
            if np.isfinite(ep.confirmed):
                ep.after += 1
                ep.shown += int(shown)
        if res.obstacles:
            self.alarm_frames += 1
        false = [o for o in res.obstacles if not any(matches(o.distance, o.x, p) for p in placed)]
        if false:
            self.false_frames += 1
        for o in false:
            t = self.false_tracks.setdefault(o.track_id, {"track_id": o.track_id, "first_frame": res.index,
                                                           "distance_m": round(o.distance, 1), "frames": 0})
            t["frames"] += 1

    def summary(self) -> dict:
        eps = [e.as_dict() for e in self.episodes.values()]
        by_type: dict[str, dict] = {}
        for e in eps:
            t = by_type.setdefault(e["type"], {"episodes": 0, "confirmed": [], "missed": 0})
            t["episodes"] += 1
            if e["missed"]:
                t["missed"] += 1
            else:
                t["confirmed"].append(e["confirmed_m"])
        for t in by_type.values():
            c = t.pop("confirmed")
            t["confirmed_median_m"] = round(float(np.median(c)), 1) if c else None
            t["confirmed_min_m"] = round(float(np.min(c)), 1) if c else None
        return {
            "frames": self.frames,
            "by_type": by_type,
            "episodes": eps,
            "false_alarms": list(self.false_tracks.values()),
            "false_frames": self.false_frames,
        }
