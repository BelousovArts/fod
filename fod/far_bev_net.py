"""Дальняя сеть оси по выпрямленному виду сверху (`fod/far_bev.py`).

Вход — растр ±8 м вокруг текущей оси, s 10…170 м. Выход — на каждой станции через 2 м
распределение по 160 боковым ячейкам 0.1 м: где ось относительно оси выпрямления.
Поправка — среднее в окне ±0.8 м вокруг максимума (две ветки стрелки не усредняются),
σ — разброс всего распределения: он растёт и там, где не видно ничего, и там, где
кандидатов два.

Вторая голова — поправка высоты оси на станциях и её масштаб (распределение Лапласа):
признаки декодера собираются поперёк пути с весами бокового распределения (детали там,
где ось) и средним по всей ширине (стены и свод).

Энкодер-декодер из тех же блоков, что `RailSegNet`: вдоль s сжатие в 32 раза, поперёк
в 8; в узком месте — свёртки с расширением вдоль s, чтобы перекрывать провалы данных
на десятки метров. Декодер возвращается к строкам станций (каждые 4 строки растра)
при полном боковом разрешении.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fod.far_bev import D_NET, DD, DS, LAYERS, LOW_TOP, N_D_NET, N_S, S0, S1, STATIONS
from fod.rail_seg_net import ResBlock

IN_CH = len(LAYERS) + 8
WINDOW = 8
SIGMA_MIN = 0.03
DZ_SCALE_MIN = 0.01


def centers(device=None) -> torch.Tensor:
    return -D_NET + (torch.arange(N_D_NET, device=device, dtype=torch.float32) + 0.5) * DD


def _decode_u(x: torch.Tensor, top: float) -> tuple[torch.Tensor, torch.Tensor]:
    lo = LAYERS[0][0]
    hit = (x > 0).float()
    return hit * (lo + (x - 1.0) / 254.0 * (top - lo)), hit


def features(bev: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:
    """(B, 9, N_S, 160) uint8 и дальность оси (B,) → (B, 14, N_S, 160)."""
    b, _, h, w = bev.shape
    x = bev.float()
    n_l = len(LAYERS)
    cnt = torch.log1p(x[:, :n_l]) / math.log(256.0)
    inten = x[:, n_l : n_l + 1] / 255.0
    hit = (x[:, :n_l].sum(1, keepdim=True) > 0).float()
    u_low, low_hit = _decode_u(x[:, n_l + 1 : n_l + 2], LOW_TOP)
    u_max, _ = _decode_u(x[:, n_l + 2 : n_l + 3], LAYERS[-1][1])
    s = S0 + (torch.arange(h, device=bev.device, dtype=torch.float32) + 0.5) * DS
    rows = (2.0 * s / S1 - 1.0).view(1, 1, h, 1).expand(b, 1, h, w)
    cols = torch.linspace(-1.0, 1.0, w, device=bev.device).view(1, 1, 1, w).expand(b, 1, h, w)
    measured = (s.view(1, 1, h, 1) <= reach.view(b, 1, 1, 1).float()).float().expand(b, 1, h, w)
    return torch.cat([cnt, inten * hit, hit, u_low, low_hit, u_max / 4.0, rows, cols, measured], dim=1)


def _down(c_in: int, c_out: int, stride) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.LeakyReLU(0.1), ResBlock(c_out, c_out))


class _AlongS(nn.Module):
    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c, c, (3, 3), padding=(dilation, 1), dilation=(dilation, 1), bias=False),
                                  nn.BatchNorm2d(c), nn.LeakyReLU(0.1))

    def forward(self, x):
        return x + self.conv(x)


class FarBevNet(nn.Module):
    CH = (32, 48, 64, 96, 128, 192)
    STRIDES = ((2, 1), (2, 1), (2, 2), (2, 2), (2, 2))

    def __init__(self, in_ch: int = IN_CH):
        super().__init__()
        ch = self.CH
        self.stem = ResBlock(in_ch, ch[0])
        self.down = nn.ModuleList(_down(ch[i], ch[i + 1], s) for i, s in enumerate(self.STRIDES))
        self.mid = nn.Sequential(*(_AlongS(ch[-1], d) for d in (1, 2, 4, 8)))
        # Декодер до уровня станций: N_S/4 строк (после двух сжатий вдоль s) × 160.
        self.up = nn.ModuleList(ResBlock(ch[i + 1] + ch[i], ch[i]) for i in (4, 3, 2))
        self.head = nn.Sequential(nn.Conv2d(ch[2], 32, 3, padding=1), nn.LeakyReLU(0.1), nn.Conv2d(32, 1, 1))
        self.z_head = nn.Sequential(nn.Conv1d(2 * ch[2], 64, 5, padding=2), nn.LeakyReLU(0.1),
                                    nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.LeakyReLU(0.1), nn.Conv1d(64, 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 14, N_S, 160) → логиты (B, S, 160) и высота (B, 2, S): поправка и сырой масштаб."""
        skips = [self.stem(x)]
        y = skips[0]
        for down in self.down:
            y = down(y)
            skips.append(y)
        y = self.mid(y)
        for up, skip in zip(self.up, (skips[4], skips[3], skips[2])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(torch.cat([y, skip], dim=1))
        logits = self.head(y).squeeze(1)
        p = torch.softmax(logits.float().detach(), dim=-1).to(y.dtype)
        pooled = torch.cat([torch.einsum("bcsd,bsd->bcs", y, p), y.mean(-1)], dim=1)
        return logits, self.z_head(pooled)


def decode(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Логиты (B, S, D) → поправка, σ (B, S) и вероятности."""
    p = torch.softmax(logits.float(), dim=-1)
    dc = centers(logits.device)
    k = p.argmax(-1, keepdim=True)
    j = torch.arange(dc.numel(), device=logits.device)
    win = ((j - k).abs() <= WINDOW).float()
    pw = p * win
    mean = (pw * dc).sum(-1) / pw.sum(-1).clamp(min=1e-6)
    sigma = ((p * (dc - mean.unsqueeze(-1)) ** 2).sum(-1)).sqrt().clamp(min=SIGMA_MIN)
    return mean, sigma, p


def decode_z(zh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 2, S) → поправка высоты и масштаб Лапласа, м."""
    zh = zh.float()
    return zh[:, 0], F.softplus(zh[:, 1]) + DZ_SCALE_MIN


def target_sigma(device=None) -> torch.Tensor:
    return 0.10 + 0.0015 * torch.as_tensor(STATIONS, device=device, dtype=torch.float32)


def _smooth(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m3 = m[:, 2:] * m[:, 1:-1] * m[:, :-2]
    d2 = (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).abs()
    return (d2 * m3).sum() / m3.sum().clamp(min=1.0)


def loss_fn(out, dt: torch.Tensor, valid: torch.Tensor, dz: torch.Tensor, valid_z: torch.Tensor) -> dict[str, torch.Tensor]:
    logits, zh = out
    logp = torch.log_softmax(logits.float(), dim=-1)
    dc = centers(logits.device)
    st = target_sigma(logits.device).view(1, -1, 1)
    tgt = torch.exp(-0.5 * ((dc.view(1, 1, -1) - torch.nan_to_num(dt).unsqueeze(-1)) / st) ** 2)
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-9)
    m = valid.float()
    denom = m.sum().clamp(min=1.0)
    ce = (-(tgt * logp).sum(-1) * m).sum() / denom
    mean, _sigma, _p = decode(logits)
    l1 = (F.smooth_l1_loss(mean, torch.nan_to_num(dt), reduction="none", beta=0.2) * m).sum() / denom
    smooth = _smooth(mean, m)
    mz = valid_z.float()
    dz_hat, scale = decode_z(zh)
    ez = (dz_hat - torch.nan_to_num(dz)).abs()
    nll_z = ((ez / scale + scale.log()) * mz).sum() / mz.sum().clamp(min=1.0)
    l1_z = (F.smooth_l1_loss(dz_hat, torch.nan_to_num(dz), reduction="none", beta=0.05) * mz).sum() / mz.sum().clamp(min=1.0)
    smooth_z = _smooth(dz_hat, mz)
    loss = ce + l1 + 0.5 * smooth + 0.2 * nll_z + l1_z + 0.5 * smooth_z
    return {"loss": loss, "ce": ce, "l1": l1, "smooth": smooth, "nll_z": nll_z, "l1_z": l1_z}


def load(path, device: str = "cuda") -> FarBevNet:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net = FarBevNet()
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return net.to(device).eval()


@torch.no_grad()
def predict(net: FarBevNet, bev, reach: float):
    """Растр кадра (9, N_S, 200) uint8, массив или тензор → поправка, σ, поправка высоты и её масштаб на `STATIONS`, м."""
    from fod.far_bev import MARGIN

    dev = next(net.parameters()).device
    if isinstance(bev, torch.Tensor):
        x = bev[None, :, :, MARGIN : MARGIN + N_D_NET].to(dev)
    else:
        x = torch.from_numpy(np.ascontiguousarray(bev[:, :, MARGIN : MARGIN + N_D_NET]))[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits, zh = net(features(x, torch.tensor([reach], device=dev)))
    mean, sigma, _p = decode(logits)
    dz, scale = decode_z(zh)
    return mean[0].cpu().numpy(), sigma[0].cpu().numpy(), dz[0].cpu().numpy(), scale[0].cpu().numpy()


assert N_S == 4 * STATIONS.size
