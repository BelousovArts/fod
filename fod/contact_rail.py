"""Контактный рельс: геометрический трекер, ось пути по нему и слияние с осью путевых рельсов.

Короб стоит на 1.41 м от оси (±2 см по всем записям) и на 0.3–0.45 м выше
головок. Затравка — короб вблизи у оси фильтра путевых рельсов, дальше линия
наращивается бинами вперёд по экстраполяции последних 40 м. Точки копятся за K
кадров по позам.

Ось по контактному рельсу — сдвиг линии на 1.41 м по нормали. Слияние с осью
фильтра — обратными дисперсиями: σ фильтра растёт с дальностью (кубика по
ближним парам), σ оси КР почти постоянна до конца линии и бесконечна за ним.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

OFFSET = 1.41
BOX_N = (1.30, 1.55)
BOX_U = (0.25, 0.55)
NEAR_S = (6.0, 30.0)
MIN_HOOK = 20


@dataclass
class ContactRailConfig:
    k: int = 10
    far_s: tuple[float, float] = (25.0, 210.0)
    s_max: float = 200.0
    fit_span: float = 40.0
    max_gap: float = 20.0
    tail_gap: float = 10.0
    min_bin: int = 2
    gate_n: float = 0.15
    gate_z: float = 0.15
    mad_n: float = 0.10
    sigma_base: float = 0.04
    sigma_per_m: float = 0.0004
    # σ фильтра путевых рельсов на дальности занижена в разы; снизу — по измеренной
    # ошибке против будущей траектории: ~10 см на 40 м, ~19 на 60, ~48 на 100.
    rail_sigma_base: float = 0.03
    rail_sigma_quad: float = 0.45


@dataclass
class ContactRailFrame:
    lines: dict[int, np.ndarray] = field(default_factory=dict)  # сторона (+1 слева, −1 справа) → (s, n, z)
    axis_s: np.ndarray | None = None
    axis_n: np.ndarray | None = None
    side: int = 0

    @property
    def reach(self) -> float:
        return max((float(a[-1, 0]) for a in self.lines.values()), default=0.0)

    def axis_sigma(self, s: np.ndarray, cfg: ContactRailConfig) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        out = np.full(s.shape, np.inf)
        if self.axis_s is not None:
            inside = (s >= self.axis_s[0]) & (s <= self.axis_s[-1])
            out[inside] = cfg.sigma_base + cfg.sigma_per_m * s[inside]
        return out


def _bin(s: float) -> float:
    return max(2.5, 0.05 * s)


def seed(xyz: np.ndarray, axis, head, side: int) -> np.ndarray | None:
    """Ближние опоры (s, n, z) по 2 м из короба у оси фильтра."""
    s = -xyz[:, 1]
    near = (s > NEAR_S[0]) & (s < NEAR_S[1])
    pts = xyz[near]
    sn = -pts[:, 1]
    n_rel = side * (pts[:, 0] - axis.n(sn))
    u = pts[:, 2] - head.head_z(sn)
    q = (n_rel > BOX_N[0]) & (n_rel < BOX_N[1]) & (u > BOX_U[0]) & (u < BOX_U[1])
    if int(q.sum()) < MIN_HOOK:
        return None
    ps, pn, pz = sn[q], pts[q, 0], pts[q, 2]
    rows = []
    for b in np.arange(NEAR_S[0], NEAR_S[1], 2.0):
        m = (ps >= b) & (ps < b + 2.0)
        if int(m.sum()) >= 3:
            rows.append((float(np.median(ps[m])), float(np.median(pn[m])), float(np.median(pz[m]))))
    return np.array(rows) if len(rows) >= 4 else None


def grow(anchors: np.ndarray, s: np.ndarray, n: np.ndarray, z: np.ndarray, cfg: ContactRailConfig) -> np.ndarray:
    """Наращивание линии вперёд по отсортированным `s`."""
    A = [tuple(r) for r in anchors]
    b = A[-1][0] + 0.5
    while b < cfg.s_max:
        w = _bin(b)
        last = A[-1][0]
        if b + 0.5 * w - last > cfg.max_gap:
            break
        c = b + 0.5 * w
        gap = c - last
        R = np.array([r for r in A if r[0] > last - cfg.fit_span])
        if R.shape[0] < 3:
            R = np.array(A[-3:])
        x0 = R[-1, 0]
        deg = 2 if (R.shape[0] >= 6 and R[-1, 0] - R[0, 0] > 20.0) else 1
        pn = np.polyval(np.polyfit(R[:, 0] - x0, R[:, 1], deg), c - x0)
        pz = np.polyval(np.polyfit(R[:, 0] - x0, R[:, 2], 1), c - x0)
        gn, gz = cfg.gate_n + 0.004 * gap, cfg.gate_z + 0.004 * gap
        lo, hi = np.searchsorted(s, b), np.searchsorted(s, b + w)
        q = (np.abs(n[lo:hi] - pn) < gn) & (np.abs(z[lo:hi] - pz) < gz)
        if int(q.sum()) >= cfg.min_bin:
            qn = n[lo:hi][q]
            mn = float(np.median(qn))
            if 1.4826 * np.median(np.abs(qn - mn)) < cfg.mad_n:
                A.append((float(np.median(s[lo:hi][q])), mn, float(np.median(z[lo:hi][q]))))
        b += w
    A = np.array(A)
    # Хвостовая опора после прыжка без продолжения — чаще чужая конструкция.
    while A.shape[0] > anchors.shape[0] and A[-1, 0] - A[-2, 0] > cfg.tail_gap:
        A = A[:-1]
    return A


def offset_axis(line: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Ось пути по линии КР: сдвиг на 1.41 м по нормали к линии в плоскости (s, n)."""
    s, n = line[:, 0], line[:, 1]
    if s.size >= 3:
        k = np.ones(3) / 3.0
        n_s = np.convolve(np.pad(n, 1, mode="edge"), k, mode="valid")
    else:
        n_s = n
    slope = np.gradient(n_s, s) if s.size >= 2 else np.zeros_like(s)
    norm = np.sqrt(1.0 + slope * slope)
    a_s = s + side * OFFSET * slope / norm
    a_n = n_s - side * OFFSET / norm
    order = np.argsort(a_s)
    return a_s[order], a_n[order]


class ContactRailTracker:
    """`device="cuda"` — буфер кадров, перенос и сортировка на видеокарте (та же арифметика, ~10 мс быстрее)."""

    def __init__(self, config: ContactRailConfig | None = None, device: str | None = None) -> None:
        self.cfg = config or ContactRailConfig()
        self.device = None
        if device is not None:
            import torch

            if device != "cuda" or torch.cuda.is_available():
                self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.buf: deque = deque(maxlen=self.cfg.k)

    def _push(self, xyz: np.ndarray, pose: np.ndarray) -> None:
        if self.device is None:
            s_all = -xyz[:, 1]
            far = xyz[(s_all > self.cfg.far_s[0]) & (s_all < self.cfg.far_s[1])].astype(np.float64)
            self.buf.append((far @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
            return
        import torch

        x = torch.from_numpy(np.ascontiguousarray(xyz)).to(self.device)
        s_all = -x[:, 1]
        far = x[(s_all > self.cfg.far_s[0]) & (s_all < self.cfg.far_s[1])].double()
        p = torch.as_tensor(np.asarray(pose, np.float64), device=self.device)
        self.buf.append(torch.addmm(p[:3, 3], far, p[:3, :3].T).float())

    def _sorted(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Точки буфера в кадре `pose`, отсортированные по дальности: s, n, z."""
        inv = np.linalg.inv(pose)
        if self.device is None:
            W = np.concatenate(list(self.buf)).astype(np.float64) @ inv[:3, :3].T + inv[:3, 3]
            ws = -W[:, 1]
            order = np.argsort(ws)
            return ws[order], W[order, 0], W[order, 2]
        import torch

        t = torch.as_tensor(inv, device=self.device)
        W = torch.addmm(t[:3, 3], torch.cat(list(self.buf)).double(), t[:3, :3].T)
        ws = -W[:, 1]
        ws, order = torch.sort(ws)
        W = W[order]
        return ws.cpu().numpy(), W[:, 0].cpu().numpy(), W[:, 2].cpu().numpy()

    def step(self, xyz: np.ndarray, axis, head, pose: np.ndarray | None) -> ContactRailFrame:
        """`axis` — запертый фильтр путевых рельсов (или None), `head` — детектор с подгонкой головок, `pose` — сенсор → мир."""
        out = ContactRailFrame()
        if pose is None:
            self.reset()
            return out
        self._push(xyz, pose)
        if axis is None or head is None:
            return out
        s_all = -xyz[:, 1]
        near = xyz[(s_all > NEAR_S[0]) & (s_all < NEAR_S[1])]
        seeds = {side: sd for side in (1, -1) if (sd := seed(near, axis, head, side)) is not None}
        if not seeds:
            return out
        ws, wn, wz = self._sorted(pose)
        for side, sd in seeds.items():
            out.lines[side] = grow(sd, ws, wn, wz, self.cfg)
        if out.lines:
            side = max(out.lines, key=lambda k: out.lines[k][-1, 0])
            out.side = side
            out.axis_s, out.axis_n = offset_axis(out.lines[side], side)
        return out


def axis_state(filter_, head, cr: ContactRailFrame, rail_reach: float, grid: np.ndarray, cfg: ContactRailConfig, s_cap: float = 150.0,
               rail_cap: float = 60.0):
    """Ось детектора на `grid`: n (слияние), z головок, дальность, до которой ось известна, и σ оси.

    z вблизи — подгонка головок по парам; за последней парой — линия КР минус её высота
    над головками, измеренная в этом же кадре на 8…40 м.
    """
    n, _w, sigma = fuse_axis(filter_, cr, grid, cfg)
    z = np.asarray(head.head_z(grid), dtype=np.float64)
    reach = min(float(rail_reach), rail_cap)
    if cr.axis_s is not None:
        line = cr.lines[cr.side]
        near = (line[:, 0] > 8.0) & (line[:, 0] < min(40.0, head.head_s_max))
        if int(near.sum()) >= 3:
            u_cr = float(np.median(line[near, 2] - head.head_z(line[near, 0])))
            far = (grid > head.head_s_max) & (grid <= line[-1, 0])
            z[far] = np.interp(grid[far], line[:, 0], line[:, 2]) - u_cr
            reach = max(reach, float(cr.axis_s[-1]))
    return n, z, min(reach, s_cap), sigma


def extend_linear(grid: np.ndarray, values: np.ndarray, reach: float, span: float = 10.0) -> np.ndarray:
    """За `reach` — по касательной последних `span` м: кубика фильтра там уходит."""
    out = values.copy()
    fit = (grid > reach - span) & (grid <= reach)
    if int(fit.sum()) >= 2:
        k, _b = np.polyfit(grid[fit], values[fit], 1)
        # Прямая из последнего узла до `reach`, а не подгоночная: иначе ступенька на границе.
        last = int(np.flatnonzero(grid <= reach)[-1])
        far = grid > grid[last]
        out[far] = values[last] + k * (grid[far] - grid[last])
    return out


class AxisHold:
    """Ось во времени по позам: прошлая ось переносится в текущий кадр и
    усредняется с новой; если дальность упала, прошлая держится до `hold_frames` кадров."""

    def __init__(self, hold_frames: int = 10, blend: float = 0.5, blend_from: float = 30.0, taper: float = 10.0) -> None:
        self.hold_frames = hold_frames
        self.blend = blend
        self.blend_from = blend_from
        self.taper = taper
        self.reset()

    def reset(self) -> None:
        self.world: np.ndarray | None = None
        self.age = 0

    def update(self, pose: np.ndarray | None, grid: np.ndarray, n: np.ndarray, z: np.ndarray, reach: float):
        if pose is None:
            self.reset()
            return n, z, reach
        n, z = n.copy(), z.copy()
        if self.world is not None and self.world.shape[0] >= 2:
            inv = np.linalg.inv(pose)
            q = self.world @ inv[:3, :3].T + inv[:3, 3]
            order = np.argsort(-q[:, 1])
            ps, pn, pz = -q[order, 1], q[order, 0], q[order, 2]
            lo, hi = max(self.blend_from, ps[0]), min(reach, ps[-1])
            both = (grid >= lo) & (grid <= hi)
            # Вес прошлой оси спадает к краям зоны: обрыв смешивания — ступенька в полразницы осей.
            g = grid[both]
            b = self.blend * np.clip((g - lo) / self.taper, 0.0, 1.0) * np.clip((hi - g) / self.taper, 0.0, 1.0)
            n[both] = (1.0 - b) * n[both] + b * np.interp(g, ps, pn)
            z[both] = (1.0 - b) * z[both] + b * np.interp(g, ps, pz)
            if ps[-1] > reach + 1.0 and self.age < self.hold_frames and reach >= ps[0]:
                ext = (grid > reach) & (grid <= ps[-1])
                d_n = float(np.interp(reach, grid, n) - np.interp(reach, ps, pn))
                d_z = float(np.interp(reach, grid, z) - np.interp(reach, ps, pz))
                n[ext] = np.interp(grid[ext], ps, pn) + d_n
                z[ext] = np.interp(grid[ext], ps, pz) + d_z
                reach = float(grid[ext][-1]) if np.any(ext) else reach
                self.age += 1
            else:
                self.age = 0
        keep = grid <= reach
        pts = np.column_stack([n[keep], -grid[keep], z[keep]])
        self.world = pts @ pose[:3, :3].T + pose[:3, 3]
        return n, z, reach


def smooth_axis(grid: np.ndarray, values: np.ndarray, sigma: np.ndarray, wavelength: float = 20.0) -> np.ndarray:
    """Сглаживатель Уиттекера со штрафом на вторую разность, веса (σ_min/σ)².

    Там, где σ минимальна, гасятся колебания короче `wavelength` м; с ростом σ
    отсечка растёт как √σ. Прямую и плавную дугу не искажает. Сетка — равномерная.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    m = grid.size
    if m < 5:
        return values.copy()
    h = float(grid[1] - grid[0])
    lam = (wavelength / (2.0 * np.pi * h)) ** 4
    w = np.square(np.min(sigma) / sigma)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m))
    A = sparse.diags(w) + lam * (D.T @ D)
    return spsolve(A.tocsc(), w * values)


def fuse_axis(
    filter_, cr: ContactRailFrame, s: np.ndarray, cfg: ContactRailConfig,
    ramp: float = 5.0, fade: float = 10.0, step: float = 0.5, wavelength: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Слияние осей обратными дисперсиями. Возвращает n(s), вес КР (0…1) и σ оси.

    Слияние — поправка к фильтру рельсов Δ(s) = w·(n_КР − n_рельсы). В начале линии КР
    она нарастает на `ramp` м; за концом продолжается с тем же значением и наклоном,
    наклон затухает на `fade` м. Иначе на краях линии ось ступенькой или изломом
    переходит на фильтр рельсов, который вдали ошибается на десятки см.
    Δ сглаживается (`smooth_axis`): шум линии КР по бинам — несколько см на метрах,
    настоящий путь так не изгибается.
    """
    s = np.asarray(s, dtype=np.float64)
    n_r = np.asarray(filter_.n(s), dtype=np.float64)
    a1 = float(cr.axis_s[-1]) if cr.axis_s is not None else 0.0
    g = np.arange(0.0, max(float(s.max()), a1) + step, step)
    sig_r = np.maximum(
        np.array([filter_.sigma(float(v)) for v in g]),
        cfg.rail_sigma_base + cfg.rail_sigma_quad * np.square(g / 100.0),
    )
    if cr.axis_s is None:
        return n_r, np.zeros_like(s), np.interp(s, g, sig_r)
    a0 = float(cr.axis_s[0])
    nr = np.asarray(filter_.n(g), dtype=np.float64)
    sig_c = cfg.sigma_base + cfg.sigma_per_m * np.minimum(g, a1)
    ins = (g >= a0) & (g <= a1)
    wc = np.where(ins, (1.0 / sig_c**2) / (1.0 / sig_r**2 + 1.0 / sig_c**2), 0.0) * np.clip((g - a0) / ramp, 0.0, 1.0)
    delta = wc * (np.interp(g, cr.axis_s, cr.axis_n) - nr)
    # За концом: σ поправки растёт с удалением от конца линии.
    sig = np.where(ins, 1.0 / np.sqrt(1.0 / sig_r**2 + wc / sig_c**2), sig_r)
    beyond = g > a1
    if np.any(beyond) and np.any(ins):
        k = np.flatnonzero(ins)
        last, back = k[-1], k[max(len(k) - 1 - int(round(5.0 / step)), 0)]
        slope = (delta[last] - delta[back]) / max(g[last] - g[back], step)
        x = g[beyond] - g[last]
        delta[beyond] = delta[last] + slope * fade * (1.0 - np.exp(-x / fade))
        sig[beyond] = np.minimum(sig_r[beyond], sig[last] + 0.02 * x)
    delta = smooth_axis(g, delta, sig, wavelength)
    return n_r + np.interp(s, g, delta), np.interp(s, g, wc), np.interp(s, g, sig)
