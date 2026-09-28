"""Дальний вид сверху, выпрямленный вдоль текущей оси и накопленный по позам.

Ось — вариант «тоннель + карта» с парами головок от сети по range image, за её
дальностью — продолжение с постоянной кривизной. Кадры копятся в мире по позам
small_gicp и переносятся в текущий кадр. Точка → (s, d, u): вдоль оси, поперёк неё
по нормали, высота над головками.

Растр: s 10…170 м через 0.5 м (320 строк), d ±10 м через 0.1 м (200 столбцов; сети
идёт ±8 м, запас — под сдвиг оси при обучении). Каналы, uint8: число точек в слоях
высоты, средняя интенсивность, средняя высота низких точек и наибольшая высота в ячейке
(по ним сеть видит, насколько пол и свод съехали от высоты оси). Станции оси — центры
полос 2 м: 11, 13, …, 169 м; ось выдаётся до 160 м, дальше — запас по контексту.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

GRID = np.arange(0.0, 160.5, 0.5)
S0, S1, DS = 10.0, 170.0, 0.5
GRID_BEV = np.arange(0.0, S1 + 0.5, 0.5)
D_STORE, D_NET, DD = 10.0, 8.0, 0.1
N_S = int(round((S1 - S0) / DS))
N_D = int(round(2 * D_STORE / DD))
N_D_NET = int(round(2 * D_NET / DD))
MARGIN = (N_D - N_D_NET) // 2
# Слои высоты над головками, м: лоток, полотно с рельсами, низ стен и КР, стены, верх стен, свод.
LAYERS = ((-1.5, -0.3), (-0.3, 0.25), (0.25, 1.2), (1.2, 2.5), (2.5, 4.0), (4.0, 8.0))
# Средняя высота точек ниже LOW_TOP и наибольшая высота: 0 — пусто, 1…255 — от LAYERS[0][0] до верха.
LOW_TOP = 1.0
N_CH = len(LAYERS) + 3
ST_DS = 2.0
STATIONS = np.arange(S0 + ST_DS / 2, S1, ST_DS)
K_MAX = 10
GAP_S = 0.55


def extend_curved(grid: np.ndarray, values: np.ndarray, reach: float, span: float = 40.0, k_max: float = 1.0 / 250.0):
    """За `reach` — парабола по последним `span` м, кривизна не круче `k_max`."""
    out = values.copy()
    last = np.flatnonzero(grid <= reach)
    if last.size < 2:
        return out
    last = int(last[-1])
    fit = (grid > reach - span) & (grid <= reach)
    x = grid[fit] - grid[last]
    if fit.sum() >= 6 and -x[0] >= 15.0:
        c, b, _a = np.polyfit(x, values[fit], 2)
        c = float(np.clip(c, -0.5 * k_max, 0.5 * k_max))
    elif fit.sum() >= 2:
        (b, _a), c = np.polyfit(x, values[fit], 1), 0.0
    else:
        return out
    far = grid > grid[last]
    d = grid[far] - grid[last]
    out[far] = values[last] + b * d + c * d * d
    return out


def extend_axis(grid: np.ndarray, n: np.ndarray, z: np.ndarray, reach: float):
    """Ось до `reach` → n и z на `GRID_BEV`: вбок парабола, по высоте прямая."""
    reach = min(reach, float(grid[-1]))
    n2, z2 = np.interp(GRID_BEV, grid, n), np.interp(GRID_BEV, grid, z)
    return extend_curved(GRID_BEV, n2, reach), extend_curved(GRID_BEV, z2, reach, k_max=0.0)


def encode_u(u: np.ndarray, top: float) -> np.ndarray:
    lo = LAYERS[0][0]
    return np.clip(np.rint(1.0 + 254.0 * (u - lo) / (top - lo)), 1, 255)


def straighten(pts: np.ndarray, grid: np.ndarray, n_ax: np.ndarray, z_ax: np.ndarray):
    """Точки кадра → (s, d, u)."""
    s = -pts[:, 1]
    slope = np.gradient(n_ax, grid)
    d = (pts[:, 0] - np.interp(s, grid, n_ax)) / np.sqrt(1.0 + np.interp(s, grid, slope) ** 2)
    u = pts[:, 2] - np.interp(s, grid, z_ax)
    return s, d, u


def normal_offset(s: np.ndarray, n: np.ndarray, grid: np.ndarray, n_ax: np.ndarray) -> np.ndarray:
    """Боковое отклонение линии (s, n) от оси по нормали."""
    slope = np.gradient(n_ax, grid)
    return (n - np.interp(s, grid, n_ax)) / np.sqrt(1.0 + np.interp(s, grid, slope) ** 2)


def raster(s: np.ndarray, d: np.ndarray, u: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """(N_CH, N_S, N_D) uint8: число точек по слоям (до 255), средняя интенсивность,
    средняя высота точек ниже `LOW_TOP` и наибольшая высота."""
    r = np.floor((s - S0) / DS).astype(np.int64)
    c = np.floor((d + D_STORE) / DD).astype(np.int64)
    ok = (r >= 0) & (r < N_S) & (c >= 0) & (c < N_D) & (u >= LAYERS[0][0]) & (u < LAYERS[-1][1])
    r, c, u, inten = r[ok], c[ok], u[ok], intensity[ok].astype(np.float64)
    flat = r * N_D + c
    layer = np.searchsorted([lo for lo, _ in LAYERS[1:]], u, side="right")
    cnt = np.bincount(layer * (N_S * N_D) + flat, minlength=len(LAYERS) * N_S * N_D).reshape(len(LAYERS), N_S, N_D)
    tot = cnt.sum(axis=0)
    i_sum = np.bincount(flat, weights=inten, minlength=N_S * N_D).reshape(N_S, N_D)
    low = u < LOW_TOP
    n_low = np.bincount(flat[low], minlength=N_S * N_D)
    u_low = np.bincount(flat[low], weights=u[low], minlength=N_S * N_D) / np.maximum(n_low, 1)
    u_max = np.full(N_S * N_D, -np.inf)
    np.maximum.at(u_max, flat, u)
    out = np.empty((N_CH, N_S, N_D), np.uint8)
    n_l = len(LAYERS)
    out[:n_l] = np.minimum(cnt, 255)
    out[n_l] = np.clip(np.rint(i_sum / np.maximum(tot, 1)), 0, 255)
    out[n_l + 1] = np.where(n_low > 0, encode_u(u_low, LOW_TOP), 0).reshape(N_S, N_D)
    out[n_l + 2] = np.where(np.isfinite(u_max), encode_u(np.where(np.isfinite(u_max), u_max, 0.0), LAYERS[-1][1]), 0).reshape(N_S, N_D)
    return out


def _interp_uniform(x, grid: np.ndarray, values):
    """`np.interp` по равномерной сетке для тензора `x`; `values` — тензор на сетке."""
    import torch

    f = ((x - float(grid[0])) / float(grid[1] - grid[0])).clamp(0.0, grid.size - 1.0)
    i0 = f.floor().long().clamp(max=grid.size - 2)
    w = f - i0
    return torch.lerp(values[i0], values[i0 + 1], w)


def straighten_t(pts, grid: np.ndarray, n_ax: np.ndarray, z_ax: np.ndarray):
    """`straighten` на видеокарте: тензор точек (N, 3) → (s, d, u); сетка равномерная."""
    import torch

    t = lambda a: torch.as_tensor(np.asarray(a, np.float64), device=pts.device, dtype=pts.dtype)  # noqa: E731
    s = -pts[:, 1]
    slope = _interp_uniform(s, grid, t(np.gradient(n_ax, grid)))
    d = (pts[:, 0] - _interp_uniform(s, grid, t(n_ax))) / torch.sqrt(1.0 + slope * slope)
    u = pts[:, 2] - _interp_uniform(s, grid, t(z_ax))
    return s, d, u


def _encode_u_t(u, top: float):
    lo = LAYERS[0][0]
    return (1.0 + 254.0 * (u - lo) / (top - lo)).round().clamp(1, 255)


def raster_t(s, d, u, intensity):
    """`raster` на видеокарте: тензоры точек → (N_CH, N_S, N_D) uint8 тензор."""
    import torch

    dev = s.device
    r = torch.floor((s - S0) / DS).long()
    c = torch.floor((d + D_STORE) / DD).long()
    ok = (r >= 0) & (r < N_S) & (c >= 0) & (c < N_D) & (u >= LAYERS[0][0]) & (u < LAYERS[-1][1])
    r, c, u, inten = r[ok], c[ok], u[ok], intensity[ok].double()
    cells = N_S * N_D
    flat = r * N_D + c
    bounds = torch.tensor([lo for lo, _ in LAYERS[1:]], device=dev, dtype=u.dtype)
    layer = torch.bucketize(u, bounds, right=True)
    n_l = len(LAYERS)
    cnt = torch.bincount(layer * cells + flat, minlength=n_l * cells).view(n_l, N_S, N_D)
    tot = cnt.sum(dim=0)
    i_sum = torch.bincount(flat, weights=inten, minlength=cells).view(N_S, N_D)
    low = u < LOW_TOP
    n_low = torch.bincount(flat[low], minlength=cells)
    u_low = torch.bincount(flat[low], weights=u[low], minlength=cells) / n_low.clamp(min=1)
    u_max = torch.full((cells,), -torch.inf, device=dev, dtype=u.dtype).scatter_reduce_(0, flat, u, "amax")
    has = torch.isfinite(u_max)
    out = torch.empty((N_CH, N_S, N_D), dtype=torch.uint8, device=dev)
    out[:n_l] = cnt.clamp(max=255)
    out[n_l] = (i_sum / tot.clamp(min=1)).round().clamp(0, 255)
    out[n_l + 1] = torch.where(n_low > 0, _encode_u_t(u_low, LOW_TOP), 0).view(N_S, N_D)
    out[n_l + 2] = torch.where(has, _encode_u_t(torch.where(has, u_max, 0.0), LAYERS[-1][1]), 0).view(N_S, N_D)
    return out


class FrameBuffer:
    """Последние кадры в мире: точки перед поездом до 175 м и их интенсивность.

    С `device` — то же на видеокарте с той же арифметикой (мир во float32, перенос во
    float64): растр совпадает с numpy побайтно, а к нему сеть чувствительна.
    """

    def __init__(self, k_max: int = K_MAX, device: str | None = None) -> None:
        self.buf: deque = deque(maxlen=k_max)
        self.device = None
        if device is not None:
            import torch

            if device != "cuda" or torch.cuda.is_available():
                self.device = torch.device(device)

    def reset(self) -> None:
        self.buf.clear()

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, xyz: np.ndarray, intensity: np.ndarray, pose: np.ndarray) -> None:
        if self.device is not None:
            self._push_t(xyz, intensity, pose)
            return
        s = -xyz[:, 1]
        keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
        keep &= (s > S0 - 2.0) & (s < S1 + 15.0) & (np.abs(xyz[:, 0]) < 60.0)
        world = (xyz[keep].astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
        self.buf.append((world, np.clip(intensity[keep], 0, 255).astype(np.uint8)))

    def _push_t(self, xyz: np.ndarray, intensity: np.ndarray, pose: np.ndarray) -> None:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(xyz, dtype=np.float32)).to(self.device)
        i = torch.from_numpy(np.ascontiguousarray(intensity)).to(self.device)
        s = -x[:, 1]
        keep = torch.isfinite(x).all(dim=1) & (x.abs().sum(dim=1) > 0)
        keep &= (s > S0 - 2.0) & (s < S1 + 15.0) & (x[:, 0].abs() < 60.0)
        p = torch.as_tensor(np.asarray(pose, np.float64), device=self.device)
        world = torch.addmm(p[:3, 3], x[keep].double(), p[:3, :3].T).float()
        self.buf.append((world, i[keep].float().clamp(0, 255).to(torch.uint8)))

    def points(self, pose: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Точки последних `k` кадров в координатах кадра `pose` и их интенсивность."""
        if self.device is not None:
            pts, inten = self.points_t(pose, k)
            return pts.double().cpu().numpy(), inten.cpu().numpy()
        items = list(self.buf)[-k:]
        inv = np.linalg.inv(pose)
        pts = np.concatenate([w for w, _ in items]).astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]
        return pts, np.concatenate([i for _, i in items])

    def points_t(self, pose: np.ndarray, k: int):
        """То же тензорами на видеокарте (только при `device`)."""
        import torch

        items = list(self.buf)[-k:]
        inv = torch.as_tensor(np.linalg.inv(pose), device=self.device)
        pts = torch.addmm(inv[:3, 3], torch.cat([w for w, _ in items]).double(), inv[:3, :3].T)
        return pts, torch.cat([i for _, i in items])


# σ сети по абсолютной величине завышена примерно в 5 раз (в ней ширина метки обучения):
# |ошибка| < 0.2·σ на 68 % станций проверочных записей. Порог дальности — по сырой σ.
SIGMA_SCALE = 0.2
SIGMA_REACH = 0.6
# Вблизи ось по рельсам точнее сети: поправка нарастает с 30 до 50 м.
BLEND = (30.0, 50.0)
S_CAP = 160.0


class FarAxis:
    """Поправка оси сетью по выпрямленному виду сверху и дальность, до которой сеть уверена."""

    def __init__(self, ckpt=None, k: int = K_MAX, device: str = "cuda") -> None:
        from pathlib import Path

        from fod.far_bev_net import load

        ckpt = ckpt or Path(__file__).resolve().parents[1] / "models" / "far_bev.pt"
        self.net = load(ckpt, device)
        self.k = k
        self.last = None

    def __call__(self, buf: FrameBuffer, pose: np.ndarray, grid: np.ndarray, n: np.ndarray, z: np.ndarray, reach: float):
        """Ось кадра до `reach` → ось с поправкой сети, высота, дальность и σ оси на `grid`."""
        from fod.far_bev_net import predict

        n_ax, z_ax = extend_axis(grid, n, z, reach)
        if buf.device is not None:
            pts, inten = buf.points_t(pose, min(self.k, len(buf)))
            bev = raster_t(*straighten_t(pts, GRID_BEV, n_ax, z_ax), inten)
        else:
            pts, inten = buf.points(pose, min(self.k, len(buf)))
            bev = raster(*straighten(pts, GRID_BEV, n_ax, z_ax), inten)
        d_hat, sigma, dz, _dz_sig = predict(self.net, bev, reach)
        slope = np.interp(STATIONS, GRID_BEV, np.gradient(n_ax, GRID_BEV))
        w = np.clip((STATIONS - BLEND[0]) / (BLEND[1] - BLEND[0]), 0.0, 1.0)
        dn = w * d_hat * np.sqrt(1.0 + slope * slope)
        bad = np.flatnonzero((sigma >= SIGMA_REACH) & (STATIONS >= BLEND[0]))
        net_reach = float(STATIONS[bad[0] - 1]) if bad.size and bad[0] > 0 else float(STATIONS[-1])
        out_reach = min(max(reach, net_reach), S_CAP, float(grid[-1]))
        n_out = np.interp(grid, GRID_BEV, n_ax) + np.interp(grid, STATIONS, dn, left=0.0)
        z_out = np.interp(grid, GRID_BEV, z_ax) + np.interp(grid, STATIONS, w * dz, left=0.0)
        sig_out = np.interp(grid, STATIONS, np.maximum(SIGMA_SCALE * sigma, 0.03))
        self.last = (bev, d_hat, sigma, n_ax, dz)
        return n_out, z_out, out_reach, sig_out


@dataclass
class LiveFrame:
    stamp: float
    xyz: np.ndarray
    intensity: np.ndarray
    pose: np.ndarray
    travelled: float
    n_ax: np.ndarray | None = None  # на GRID_BEV
    z_ax: np.ndarray | None = None
    reach: float = 0.0


class LiveAxis:
    """Цепочка оси как в детекторе (`--rails seg`, тоннель + карта) и буфер кадров в мире."""

    def __init__(self, k_max: int = K_MAX, cap: float = 80.0, device: str = "cuda") -> None:
        from fod.contact_rail import ContactRailConfig
        from fod.rail_seg_detect import SegRailDetector

        self.k_max, self.cap = k_max, cap
        self.detector = SegRailDetector(device=device)
        self.cr_cfg = ContactRailConfig()
        self.last = None
        self.reset()

    def reset(self) -> None:
        from fod.contact_rail import ContactRailTracker
        from fod.lidar_odometry import GicpOdometry
        from fod.obstacles import ObstacleDetector
        from fod.odometry import EgoMotion
        from fod.rail_axis_filter import RailTracker
        from fod.track_map import TrackMap
        from fod.tunnel_axis import TunnelAxis

        self.odom, self.icp = EgoMotion(), GicpOdometry()
        self.tracker = RailTracker(detector=self.detector, predict_m=self.cap)
        self.placer, self.cr = ObstacleDetector(), ContactRailTracker(self.cr_cfg)
        self.tmap, self.tun = TrackMap(), TunnelAxis()
        self.buf = FrameBuffer(self.k_max)

    def step(self, cloud) -> LiveFrame:
        from fod.contact_rail import axis_state
        from fod.odometry import deskew_xyz, stamp_of
        from fod.tunnel_axis import fuse_tunnel

        stamp = stamp_of(cloud)
        if self.last is not None and stamp - self.last > GAP_S:
            self.reset()
        self.last = stamp
        motion = self.odom.step(cloud.xyz, cloud.intensity, stamp)
        xyz = deskew_xyz(cloud.xyz, cloud.timestamp, motion.v)
        fin = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
        pose = self.icp.step(xyz[fin], motion.travelled)
        self.buf.push(xyz, cloud.intensity, pose)
        frame = LiveFrame(stamp, xyz, cloud.intensity, pose, float(motion.travelled))
        rail = self.tracker.step(xyz, cloud.intensity, motion)
        self.placer._fit_head(rail.marks)
        locked = self.tracker.filter.locked and len(rail.marks) >= 4 and self.placer.head_coeff is not None
        cr_frame = self.cr.step(xyz, self.tracker.filter if locked else None, self.placer if locked else None, pose)
        if not locked:
            self.tmap.reset()
            self.tun.reset()
            return frame
        rail_reach = max(float(m.s) for m in rail.marks)
        n, z, r, sig = axis_state(self.tracker.filter, self.placer, cr_frame, rail_reach, GRID, self.cr_cfg, 150.0, self.cap)
        sl = self.tun.step(xyz, GRID, n, z, r)
        tn, tsig, tr_ = fuse_tunnel(GRID, n, sig, r, sl)
        gn, gz, gr = self.tmap.update(pose, GRID, tn, z, tsig, tr_, min(rail_reach, self.cap))
        frame.n_ax, frame.z_ax = extend_axis(GRID, gn, gz, gr)
        frame.reach = float(gr)
        return frame

    def accumulated(self, frame: LiveFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Точки последних `k` кадров в координатах текущего кадра и их интенсивность."""
        return self.buf.points(frame.pose, k)

    def bev(self, frame: LiveFrame, k: int) -> np.ndarray:
        pts, inten = self.accumulated(frame, k)
        return raster(*straighten(pts, GRID_BEV, frame.n_ax, frame.z_ax), inten)
