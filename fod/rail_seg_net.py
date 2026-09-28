"""Сеть сегментации рельсов по range image (кольца Pandar128 × передний сектор).

Лёгкий энкодер-декодер в духе SalsaNext: остаточные блоки с расширенными свёртками,
по вертикали сжатие только вдвое дважды (колец всего 128), по горизонтали — в 16 раз;
skip-связи сохраняют тонкие головки (1–3 пикселя). Выход — логиты 4 классов
(фон, левая головка, правая, КР).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_CLASSES = 4
IN_CH = 6
MAX_RANGE = 150.0


def features(rng_cm: torch.Tensor, z_cm: torch.Tensor, inten: torch.Tensor, col0: int | torch.Tensor = 0,
             width_total: int = 1200) -> torch.Tensor:
    """(B, H, W) сырые каналы → (B, 6, H, W): дальность, высота, интенсивность, маска, строка, колонка."""
    r = rng_cm.float() / 100.0
    valid = (r > 0).float()
    b, h, w = r.shape
    f_r = torch.log1p(r) / math.log1p(MAX_RANGE)
    f_z = z_cm.float() / 300.0 * valid
    f_i = inten.float() / 255.0 * valid
    rows = torch.linspace(-1.0, 1.0, h, device=r.device).view(1, h, 1).expand(b, h, w)
    col0 = torch.as_tensor(col0, device=r.device).view(-1, 1, 1).float()
    cols = (torch.arange(w, device=r.device).view(1, 1, w) + col0) / (width_total - 1) * 2.0 - 1.0
    return torch.stack([f_r, f_z, f_i, valid, rows, cols.expand(b, h, w)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.skip = nn.Conv2d(c_in, c_out, 1, bias=False) if c_in != c_out else nn.Identity()
        self.a = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.1))
        self.b = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c_out),
                               nn.LeakyReLU(0.1))
        self.c = nn.Sequential(nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return F.leaky_relu(self.c(self.b(self.a(x))) + self.skip(x), 0.1)


class RailSegNet(nn.Module):
    CH = (32, 64, 96, 128, 160)
    STRIDES = ((2, 2), (2, 2), (1, 2), (1, 2))

    def __init__(self, in_ch: int = IN_CH, n_classes: int = N_CLASSES):
        super().__init__()
        ch = self.CH
        self.stem = ResBlock(in_ch, ch[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i, s in enumerate(self.STRIDES):
            self.down.append(nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, stride=s, padding=1, bias=False),
                                           nn.BatchNorm2d(ch[i + 1]), nn.LeakyReLU(0.1)))
            self.enc.append(ResBlock(ch[i + 1], ch[i + 1]))
        self.dec = nn.ModuleList(ResBlock(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(self.STRIDES))))
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = [self.stem(x)]
        y = skips[0]
        for down, enc in zip(self.down, self.enc):
            y = enc(down(y))
            skips.append(y)
        for dec, skip in zip(self.dec, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([y, skip], dim=1))
        return self.head(y)


def load(path, device: str = "cuda") -> RailSegNet:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net = RailSegNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return net.to(device).eval()


@torch.no_grad()
def predict(net: RailSegNet, rng_cm: np.ndarray, z_cm: np.ndarray, inten: np.ndarray) -> np.ndarray:
    """Один кадр (128, W) → вероятности классов (4, 128, W) float32."""
    dev = next(net.parameters()).device
    t = [torch.from_numpy(np.ascontiguousarray(a))[None].to(dev) for a in (rng_cm, z_cm, inten)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = net(features(*t))
    return torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
