"""Онлайн-одометрия small_gicp: кадр к локальной воксельной карте, ~29 мс на кадр в один поток, ~9 мс в четыре
(позы те же до долей миллиметра).

Прогноз шага — поворот и боковой снос от прошлого шага, путь вперёд (−y) — от `EgoMotion`:
в однородном тоннеле ICP вдоль пути вырожден и со старта «залипает» на нулевой скорости.
Позы — сенсор→мир, как `pose_map` в `out/maps/<bag>/poses.npz`.
"""

from __future__ import annotations

import numpy as np

MAX_RANGE = 150.0


def with_forward(delta: np.ndarray, ds: float) -> np.ndarray:
    out = delta.copy()
    if np.isfinite(ds):
        out[1, 3] = -ds
    return out


class GicpOdometry:
    def __init__(self, voxel: float = 1.0, down: float = 0.5, max_range: float = MAX_RANGE, threads: int = 4) -> None:
        import small_gicp

        self.sg = small_gicp
        self.voxel, self.down, self.max_range, self.threads = voxel, down, max_range, threads
        self.reset()

    def reset(self) -> None:
        self.map = self.sg.GaussianVoxelMap(self.voxel)
        self.map.set_lru(horizon=100, clear_cycle=10)
        self.T = np.eye(4)
        self.delta = np.eye(4)
        self.first = True

    def step(self, xyz: np.ndarray, ds: float) -> np.ndarray:
        """`xyz` — облако после компенсации движения, `ds` — путь за кадр по `EgoMotion`."""
        r = np.linalg.norm(xyz, axis=1)
        pts = xyz[np.isfinite(r) & (r > 1.0) & (r < self.max_range)].astype(np.float64)
        src, _tree = self.sg.preprocess_points(pts, self.down, num_threads=self.threads)
        if not self.first:
            res = self.sg.align(
                self.map, src, self.T @ with_forward(self.delta, ds),
                max_correspondence_distance=1.0, num_threads=self.threads, max_iterations=20,
            )
            T = res.T_target_source
            self.delta = np.linalg.inv(self.T) @ T
            self.T = T
        self.first = False
        self.map.insert(src, self.T)
        return self.T.copy()
