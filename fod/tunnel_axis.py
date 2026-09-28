"""Ось пути вдали по поперечному профилю тоннеля.

Вблизи (`near`) ось по рельсам точна до сантиметров. Точки там в координатах оси —
боковое отклонение по нормали к оси и высота над головками — дают шаблон сечения
тоннеля: стены, лотки, кабели, свод. Шаблон — доля кадров, в которых ячейка занята,
с забыванием `decay`: от плотности точек не зависит, стойки и ниши усредняются.

Дальше по срезам через `step` м ищется сдвиг (dn, du), при котором занятые ячейки
среза лучше всего ложатся на шаблон. Центр поиска — ось кадра, пока срез в её
дальности, дальше — продолжение уже найденных срезов параболой. На кривой, где КР
на ближней стороне и уходит из вида, стены видны дальше: одна стена тоже работает,
если она есть в шаблоне.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TunnelAxisConfig:
    near: tuple[float, float] = (8.0, 30.0)
    n_half: float = 7.0
    u_lo: float = -0.6
    u_hi: float = 5.0
    cell: float = 0.1
    decay: float = 0.9
    blur: float = 1.5              # ячеек
    step: float = 5.0
    slice_len: float = 6.0
    search: float = 0.8            # поиск по n, ± м
    search_du: float = 0.3         # поиск по высоте, ± м
    min_cells: int = 25
    min_contrast: float = 0.35
    max_gap: float = 20.0
    fit_span: float = 40.0
    max_curvature: float = 1.0 / 300.0
    s_max: float = 150.0
    # σ среза: sigma0 + sigma_s·s — по ошибке против эталона за дальностью кадра; от контраста не зависит.
    sigma0: float = 0.04
    sigma_s: float = 0.0015
    min_frames: int = 5


@dataclass
class TunnelSlices:
    s: np.ndarray
    n: np.ndarray
    z: np.ndarray
    contrast: np.ndarray
    sigma: np.ndarray


EMPTY = TunnelSlices(*(np.zeros(0) for _ in range(5)))


class TunnelAxis:
    """`device="cuda"` — сортировка точек по дальности на видеокарте (порядок равных `s` на результат не влияет)."""

    def __init__(self, config: TunnelAxisConfig | None = None, device: str | None = None) -> None:
        self.cfg = config or TunnelAxisConfig()
        c = self.cfg
        self.nn = int(round(2 * c.n_half / c.cell))
        self.nu = int(round((c.u_hi - c.u_lo) / c.cell))
        self.device = None
        if device is not None:
            import torch

            if device != "cuda" or torch.cuda.is_available():
                self.device = torch.device(device)
        self.reset()

    def _argsort(self, s: np.ndarray) -> np.ndarray:
        if self.device is None:
            return np.argsort(s)
        import torch

        return torch.argsort(torch.from_numpy(np.ascontiguousarray(s)).to(self.device)).cpu().numpy()

    def reset(self) -> None:
        self.occ = np.zeros((self.nn, self.nu))
        self.frames = 0

    def _cells(self, dn: np.ndarray, u: np.ndarray) -> np.ndarray:
        c = self.cfg
        i = np.floor((dn + c.n_half) / c.cell).astype(np.int64)
        j = np.floor((u - c.u_lo) / c.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.nn) & (j >= 0) & (j < self.nu)
        return np.unique(i[ok] * self.nu + j[ok])

    def step(self, xyz: np.ndarray, grid: np.ndarray, n: np.ndarray, z: np.ndarray, reach: float) -> TunnelSlices:
        from scipy.ndimage import gaussian_filter

        c = self.cfg
        pts = xyz[np.isfinite(xyz).all(axis=1)]
        s = -pts[:, 1]
        keep = (s > c.near[0]) & (s < c.s_max + c.slice_len)
        pts, s = pts[keep], s[keep]
        order = self._argsort(s)
        pts, s = pts[order], s[order]
        slope = np.gradient(n, grid)

        near_hi = min(c.near[1], reach)
        if near_hi - c.near[0] < 10.0:
            return EMPTY
        a, b = np.searchsorted(s, [c.near[0], near_hi])
        sn = s[a:b]
        dn = (pts[a:b, 0] - np.interp(sn, grid, n)) / np.sqrt(1.0 + np.interp(sn, grid, slope) ** 2)
        cells = self._cells(dn, pts[a:b, 2] - np.interp(sn, grid, z))
        frame = np.zeros(self.nn * self.nu)
        frame[cells] = 1.0
        self.occ = c.decay * self.occ + (1.0 - c.decay) * frame.reshape(self.nn, self.nu)
        self.frames += 1
        if self.frames < c.min_frames:
            return EMPTY
        tmpl = gaussian_filter(self.occ, c.blur)
        tmpl /= max(float(tmpl.max()), 1e-9)

        k_n = int(round(c.search / c.cell))
        k_u = int(round(c.search_du / c.cell))
        sh_n = np.arange(-k_n, k_n + 1)
        sh_u = np.arange(-k_u, k_u + 1)
        acc_s = list(np.arange(c.near[0], near_hi + 1e-6, 2.0))
        acc_n = list(np.interp(acc_s, grid, n))
        acc_z = list(np.interp(acc_s, grid, z))
        out = []
        last_ok = near_hi
        for sc in np.arange(near_hi + c.step / 2.0, c.s_max + 1e-6, c.step):
            if sc - last_ok > c.max_gap:
                break
            lo, hi = sc - c.slice_len / 2.0, sc + c.slice_len / 2.0
            a, b = np.searchsorted(s, [lo, hi])
            if b - a < c.min_cells:
                continue
            ss, p = s[a:b], pts[a:b]
            if hi <= reach:
                pn, pk = np.interp(ss, grid, n), np.interp(ss, grid, slope)
                pc = float(np.interp(sc, grid, n))
            else:
                pn, pk, pc = self._predict(np.asarray(acc_s), np.asarray(acc_n), ss, sc)
            az = np.asarray(acc_z)[np.asarray(acc_s) > sc - c.fit_span]
            az_s = np.asarray(acc_s)[np.asarray(acc_s) > sc - c.fit_span]
            kz, bz = np.polyfit(az_s, az, 1) if az_s.size >= 3 else (0.0, float(acc_z[-1]))
            cos = 1.0 / np.sqrt(1.0 + pk * pk)
            cells = self._cells((p[:, 0] - pn) * cos, p[:, 2] - (kz * ss + bz))
            if cells.size < c.min_cells:
                continue
            ci, cj = cells // self.nu, cells % self.nu
            ii = ci[None, :] - sh_n[:, None]
            jj = cj[None, :] - sh_u[:, None]
            ok = ((ii >= 0) & (ii < self.nn))[:, None, :] & ((jj >= 0) & (jj < self.nu))[None, :, :]
            vals = tmpl[np.clip(ii, 0, self.nn - 1)[:, None, :], np.clip(jj, 0, self.nu - 1)[None, :, :]]
            score = np.where(ok, vals, 0.0).sum(axis=2) / cells.size
            bi, bu = np.unravel_index(int(np.argmax(score)), score.shape)
            best = float(score[bi, bu])
            prof = score[:, bu]
            contrast = (best - float(np.median(prof))) / max(best, 1e-9)
            if contrast < c.min_contrast or bi in (0, sh_n.size - 1):
                continue
            y0, y1, y2 = prof[bi - 1], prof[bi], prof[bi + 1]
            den = y0 - 2.0 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            d_n = (sh_n[bi] + float(np.clip(frac, -0.5, 0.5))) * c.cell
            cc = float(1.0 / np.sqrt(1.0 + np.interp(sc, ss, pk) ** 2))
            n_sc = pc + d_n / cc
            z_sc = kz * sc + bz + sh_u[bu] * c.cell
            acc_s.append(sc)
            acc_n.append(n_sc)
            acc_z.append(z_sc)
            last_ok = sc
            sig = c.sigma0 + c.sigma_s * sc
            out.append((sc, n_sc, z_sc, contrast, sig))
        if not out:
            return EMPTY
        return TunnelSlices(*(np.array(v) for v in zip(*out)))

    def _predict(self, acc_s, acc_n, ss, sc):
        c = self.cfg
        fit = acc_s > acc_s[-1] - c.fit_span
        x = acc_s[fit] - acc_s[-1]
        if x.size >= 4 and -x[0] > 15.0:
            cc, bb, aa = np.polyfit(x, acc_n[fit], 2)
            cc = float(np.clip(cc, -0.5 * c.max_curvature, 0.5 * c.max_curvature))
        elif x.size >= 2:
            (bb, aa), cc = np.polyfit(x, acc_n[fit], 1), 0.0
        else:
            aa, bb, cc = float(acc_n[-1]), 0.0, 0.0
        d = ss - acc_s[-1]
        dc = sc - acc_s[-1]
        return aa + bb * d + cc * d * d, bb + 2.0 * cc * d, float(aa + bb * dc + cc * dc * dc)


def fuse_tunnel(grid: np.ndarray, n: np.ndarray, sigma: np.ndarray, reach: float, tun: TunnelSlices, grow: float = 0.03):
    """Поправка оси кадра по срезам тоннеля обратными дисперсиями.

    За дальностью кадра его σ растёт на `grow` м на метр: там ось кадра — продолжение
    кубики или линии КР. Возвращает n, σ и дальность — до последнего среза.
    """
    from fod.contact_rail import smooth_axis

    if not tun.s.size:
        return n, sigma, reach
    sig_f = np.sqrt(np.square(np.interp(tun.s, grid, sigma)) + np.square(grow * np.maximum(tun.s - reach, 0.0)))
    w = sig_f**2 / (sig_f**2 + tun.sigma**2)
    d = w * (tun.n - np.interp(tun.s, grid, n))
    sig_c = 1.0 / np.sqrt(1.0 / sig_f**2 + 1.0 / tun.sigma**2)
    s0 = float(tun.s[0]) - 5.0
    ks = np.concatenate([[s0], tun.s])
    delta = np.interp(grid, ks, np.concatenate([[0.0], d]))
    sig_all = np.sqrt(np.square(sigma) + np.square(grow * np.maximum(grid - reach, 0.0)))
    sig_new = np.where(grid >= s0, np.minimum(sig_all, np.interp(grid, tun.s, sig_c)), sigma)
    sig_new = np.where(grid > tun.s[-1], sig_all, sig_new)
    delta = smooth_axis(grid, delta, np.maximum(sig_new, 1e-3))
    return n + delta, sig_new, max(reach, float(tun.s[-1]))
