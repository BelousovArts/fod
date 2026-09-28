"""Каталог синтетических объектов и пересечение с лучами лидара."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Путевые координаты: s вперёд (−Y), n вправо (+X), u вверх от пола.


@dataclass
class SceneObject:
    kind: str
    name: str
    s: float
    n: float
    u: float
    size: tuple[float, float, float]
    intensity: float = 70.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def color_rgba(self) -> tuple[float, float, float, float]:
        return KIND_COLORS.get(self.kind, (1.0, 0.3, 0.1, 0.85))


PRESETS: dict[str, dict] = {
    "box": {"kind": "box", "size": (0.3, 0.3, 0.1), "intensity": 55.0},
    "min": {"kind": "box", "size": (0.3, 0.3, 0.1), "intensity": 55.0},
    "cube": {"kind": "box", "size": (0.3, 0.3, 0.3), "intensity": 70.0},
    "crate": {"kind": "box", "size": (0.5, 0.5, 0.5), "intensity": 80.0},
    "suitcase": {"kind": "box", "size": (0.70, 0.40, 0.25), "intensity": 65.0},
    "person": {"kind": "person", "size": (0.50, 0.50, 1.75), "intensity": 90.0},
    "barrel": {"kind": "barrel", "size": (0.60, 0.60, 0.90), "intensity": 75.0},
    "cone": {"kind": "cone", "size": (0.36, 0.36, 0.75), "intensity": 110.0},
    "wire": {"kind": "wire", "size": (2.10, 0.02, 0.02), "intensity": 40.0, "u": 2.0},
    "sphere": {"kind": "sphere", "size": (0.40, 0.40, 0.40), "intensity": 85.0},
}

# Объекты записи `cloud_with_fake_obj` (по кадрам ближе 15 м). size — вдоль n, вдоль s, высота;
# n — центр от оси; rest: head — низ на `u` над головками, floor — на пол под объектом;
# yaw — поворот в плоскости (n, s), градусы.
FAKE_OBJECTS: dict[str, dict] = {
    "f1_shield": {"kind": "box", "size": (2.0, 0.05, 1.86), "n": 0.02, "rest": "floor"},
    "f2_cube_air": {"kind": "box", "size": (0.3, 0.3, 0.3), "n": 0.19, "rest": "head", "u": 1.04},
    "f3_cube_rail": {"kind": "box", "size": (0.3, 0.3, 0.3), "n": 0.76, "rest": "head", "u": 0.08},
    "f4_cube_air_r": {"kind": "box", "size": (0.3, 0.3, 0.3), "n": 1.07, "rest": "head", "u": 1.04},
    "f5_cube_air_l": {"kind": "box", "size": (0.3, 0.3, 0.3), "n": -1.26, "rest": "head", "u": 1.04},
    "f6_big_r": {"kind": "box", "size": (1.9, 1.9, 1.9), "n": 2.12, "rest": "floor", "yaw": -13.0},
    "f7_big_l": {"kind": "box", "size": (1.95, 1.95, 1.9), "n": -2.40, "rest": "floor", "yaw": -12.0},
    "f8_over": {"kind": "box", "size": (2.3, 2.2, 1.9), "n": 0.05, "rest": "head", "u": 2.87, "yaw": -10.0},
    "f9_slab": {"kind": "box", "size": (2.0, 0.6, 0.2), "n": 0.07, "rest": "head", "u": 0.0},
    "f10_rod": {"kind": "rod", "size": (0.05, 0.05, 1.95), "n": 0.04, "rest": "head", "u": 2.70},
}

KIND_COLORS = {
    "box": (1.0, 0.15, 0.10, 0.85),
    "rod": (0.20, 0.95, 0.95, 0.90),
    "person": (1.0, 0.55, 0.10, 0.85),
    "barrel": (0.20, 0.45, 1.00, 0.85),
    "cone": (1.0, 0.40, 0.05, 0.85),
    "wire": (0.20, 0.95, 0.95, 0.90),
    "sphere": (0.80, 0.20, 0.90, 0.85),
}

ALIASES = {
    "min_obstacle": "min",
    "human": "person",
    "man": "person",
    "cylinder": "barrel",
    "cable": "wire",
}


def _parse_size(text: str) -> tuple[float, float, float]:
    parts = [float(p) for p in text.replace("x", ",").replace(";", ",").split(",") if p]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) != 3:
        raise ValueError(f"Размер должен быть W,L,H — получено «{text}»")
    return (parts[0], parts[1], parts[2])


def parse_object_spec(spec: str, index: int = 0) -> SceneObject:
    """Разбор 'person:s=40,n=0' или 'box:s=20,size=0.3x0.3x0.1'."""
    raw = spec.strip()
    if ":" in raw:
        kind_key, payload = raw.split(":", 1)
    else:
        kind_key, payload = raw, ""
    kind_key = ALIASES.get(kind_key, kind_key)
    if kind_key not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(f"Неизвестный объект «{kind_key}». Доступны: {known}")

    preset = PRESETS[kind_key]
    fields: dict[str, str] = {}
    if payload:
        for chunk in payload.split(","):
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Ожидалось key=value в «{chunk}»")
            key, value = chunk.split("=", 1)
            fields[key.strip()] = value.strip()

    size = _parse_size(fields["size"]) if "size" in fields else tuple(preset["size"])
    u_default = float(preset.get("u", 0.0))
    obj = SceneObject(
        kind=str(preset["kind"]),
        name=fields.get("name", f"{kind_key}_{index}"),
        s=float(fields.get("s", 30.0)),
        n=float(fields.get("n", 0.0)),
        u=float(fields.get("u", u_default)),
        size=(float(size[0]), float(size[1]), float(size[2])),
        intensity=float(fields.get("intensity", preset["intensity"])),
    )
    if "diameter" in fields:
        obj.extra["diameter"] = float(fields["diameter"])
        d = float(fields["diameter"])
        obj.size = (obj.size[0], d, d)
    if "length" in fields:
        obj.size = (float(fields["length"]), obj.size[1], obj.size[2])
    return obj


def object_from_dict(item: dict, index: int = 0) -> SceneObject:
    kind = str(item.get("type", item.get("kind", "box")))
    kind = ALIASES.get(kind, kind)
    if kind not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(f"Неизвестный объект «{kind}». Доступны: {known}")
    preset = PRESETS[kind]
    size = item.get("size", preset["size"])
    if isinstance(size, str):
        size = _parse_size(size)
    else:
        size = (float(size[0]), float(size[1]), float(size[2]))
    obj = SceneObject(
        kind=str(preset["kind"]),
        name=str(item.get("name", f"{kind}_{index}")),
        s=float(item.get("s", 30.0)),
        n=float(item.get("n", 0.0)),
        u=float(item.get("u", preset.get("u", 0.0))),
        size=size,
        intensity=float(item.get("intensity", preset["intensity"])),
    )
    if "diameter" in item:
        obj.extra["diameter"] = float(item["diameter"])
        obj.size = (obj.size[0], float(item["diameter"]), float(item["diameter"]))
    if "length" in item:
        obj.size = (float(item["length"]), obj.size[1], obj.size[2])
    return obj


def objects_from_yaml(data: dict, start_index: int = 0) -> list[SceneObject]:
    items = data.get("objects", data if isinstance(data, list) else [])
    return [object_from_dict(item, start_index + i) for i, item in enumerate(items)]


def lidar_center(obj: SceneObject, z_floor: float) -> np.ndarray:
    height = obj.size[2]
    if obj.kind == "wire":
        z = z_floor + obj.u
    else:
        z = z_floor + obj.u + 0.5 * height
    return np.array([obj.n, -obj.s, z], dtype=np.float64)


def ray_aabb(dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, np.copysign(1e-12, dirs + 1e-32), dirs)
    t1 = bmin * inv
    t2 = bmax * inv
    t_enter = np.max(np.minimum(t1, t2), axis=1)
    t_exit = np.min(np.maximum(t1, t2), axis=1)
    hit = (t_exit >= t_enter) & (t_exit > 1e-4)
    t = np.where(t_enter > 1e-4, t_enter, t_exit)
    return np.where(hit & (t > 1e-4), t, np.inf)


def ray_sphere(dirs: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    oc = -center
    b = 2.0 * dirs @ oc
    c = float(oc @ oc) - radius * radius
    disc = b * b - 4.0 * c
    out = np.full(dirs.shape[0], np.inf)
    ok = disc >= 0.0
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_d) * 0.5
    t2 = (-b + sqrt_d) * 0.5
    t = np.where(t1 > 1e-4, t1, t2)
    return np.where(ok & (t > 1e-4), t, out)


def ray_finite_cylinder(
    dirs: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    half_len: float,
) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    oc = -center
    da = dirs @ axis
    oca = float(oc @ axis)
    d_perp = dirs - da[:, None] * axis
    oc_perp = oc - oca * axis
    a = np.sum(d_perp * d_perp, axis=1)
    b = 2.0 * (d_perp @ oc_perp)
    c = float(oc_perp @ oc_perp) - radius * radius
    disc = b * b - 4.0 * a * c
    t_hit = np.full(dirs.shape[0], np.inf)
    ok = (disc >= 0.0) & (a > 1e-10)
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    denom = 2.0 * np.where(ok, a, 1.0)
    for sign in (-1.0, 1.0):
        t = (-b + sign * sqrt_d) / denom
        axial = oca + t * da
        good = ok & (t > 1e-4) & (np.abs(axial) <= half_len)
        t_hit = np.where(good, np.minimum(t_hit, t), t_hit)

    for cap_sign in (-1.0, 1.0):
        cap_center = center + cap_sign * half_len * axis
        denom = dirs @ axis
        finite = np.abs(denom) > 1e-9
        t = np.full(dirs.shape[0], np.inf)
        t[finite] = (cap_center @ axis) / denom[finite]
        point = t[:, None] * dirs
        radial = point - cap_center
        axial = radial @ axis
        radial = radial - axial[:, None] * axis
        good = finite & (t > 1e-4) & (np.sum(radial * radial, axis=1) <= radius * radius)
        t_hit = np.where(good, np.minimum(t_hit, t), t_hit)
    return t_hit


def _box_frame(obj: SceneObject) -> np.ndarray:
    """Строки — оси ящика в координатах лидара (x = n, y = −s)."""
    yaw = np.deg2rad(obj.extra.get("yaw", 0.0))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hit_object(obj: SceneObject, dirs: np.ndarray, z_floor: float) -> np.ndarray:
    center = lidar_center(obj, z_floor)
    w, length, h = obj.size
    if obj.kind == "box":
        half = np.array([0.5 * w, 0.5 * length, 0.5 * h], dtype=np.float64)
        if obj.extra.get("yaw", 0.0) == 0.0:
            return ray_aabb(dirs, center - half, center + half)
        rot = _box_frame(obj)
        return ray_aabb(dirs @ rot.T, rot @ center - half, rot @ center + half)
    if obj.kind == "rod":
        return ray_finite_cylinder(dirs, center, np.array([0.0, 0.0, 1.0]), 0.5 * w, 0.5 * h)
    if obj.kind == "sphere":
        return ray_sphere(dirs, center, 0.5 * max(w, length, h))
    if obj.kind == "wire":
        # Тонкий горизонтальный цилиндр легко проскакивает между кольцами.
        # Для стенда берём тонкий AABB того же сечения — плотность точек реалистичнее.
        radius = 0.5 * obj.extra.get("diameter", max(length, h, 0.03))
        half = np.array([0.5 * w, max(radius, 0.015), max(radius, 0.015)], dtype=np.float64)
        return ray_aabb(dirs, center - half, center + half)
    if obj.kind in {"person", "barrel", "cone"}:
        radius = 0.25 * (w + length)
        if obj.kind == "cone":
            radius *= 0.85
        return ray_finite_cylinder(dirs, center, np.array([0.0, 0.0, 1.0]), radius, 0.5 * h)
    raise ValueError(f"Неподдерживаемый тип объекта: {obj.kind}")


def object_intensity(obj: SceneObject, t_hit: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    points = t_hit[:, None] * dirs
    radial = points / np.maximum(t_hit[:, None], 1e-6)
    if obj.kind == "box":
        cos_inc = np.clip(np.abs(radial).max(axis=1), 0.15, 1.0)
    else:
        cos_inc = np.clip(np.abs(radial[:, 2]) * 0.25 + 0.55, 0.15, 1.0)
    falloff = (12.0 / np.maximum(t_hit, 4.0)) ** 0.35
    value = obj.intensity * cos_inc * falloff
    return np.clip(value, 1.0, 255.0)


def incidence_cos(obj: SceneObject, t_hit: np.ndarray, dirs: np.ndarray, z_floor: float) -> np.ndarray:
    """|cos| угла между лучом и нормалью поверхности в точке попадания."""
    center = lidar_center(obj, z_floor)
    local = t_hit[:, None] * dirs - center
    if obj.kind == "box":
        rot = _box_frame(obj)
        half = 0.5 * np.asarray(obj.size, dtype=np.float64)
        face = np.argmax(np.abs(local @ rot.T) / half, axis=1)
        return np.abs(np.take_along_axis(dirs @ rot.T, face[:, None], axis=1)[:, 0])
    radial = local[:, :2] / np.maximum(np.linalg.norm(local[:, :2], axis=1, keepdims=True), 1e-9)
    return np.abs(np.sum(dirs[:, :2] * radial, axis=1))


# Отражательная способность поверхностей тоннеля 5–20 м (0.3 < u < 3 м, |n| < 4 м, три записи):
# медиана 8, p5 1, p95 22 — логнормаль с σ ≈ 0.6.
REFLECT_MEDIAN = 8.0
REFLECT_SIGMA = 0.6


def tunnel_intensity(
    reflect: float, cos_inc: np.ndarray, edge: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Целочисленная интенсивность как у реальных поверхностей: угол, шум по точкам, ослабление на краях."""
    value = reflect * (0.35 + 0.65 * np.clip(cos_inc, 0.0, 1.0)) * rng.lognormal(0.0, 0.3, cos_inc.shape)
    value = np.where(edge, value * rng.uniform(0.25, 1.0, cos_inc.shape), value)
    return np.clip(np.round(value), 0.0, 255.0)
