"""RTMoSR + EA: the teacher-ablation-derived film-tier student (2026-08-20).

Ablation of 4xNomosWebPhoto_RealPLKSR (results/teacher-ablation.json)
showed its quality lives in (a) per-pixel EA gating — removal is the most
catastrophic single ablation, 14.1 dB survival — and (b) collective depth,
NOT in the 17px large kernels (1x1-cropped kernels survive at 29.6 dB).
So: keep RTMoSR's fast small-kernel backbone, graft the teacher's EA gate
(3x3 conv + sigmoid + multiply — cheap, bandwidth-light) onto every block
output, and prefer depth over width when spending latency.
"""
import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

from userdevice_rtus.rtmosr_vendored import RTMoSR, GatedCNNBlock


class EA(nn.Module):
    """Element-wise (per-pixel) attention gate, taken from RealPLKSR.

    The definition it was grafted from is vendored alongside this file, in
    `realplksr_vendored.py` (see its `PLKBlock`), so the graft can be diffed
    against its source. That module is reference only -- nothing imports it,
    and the teacher itself is loaded through spandrel.

    This gate is why the student exists in this shape: ablating the teacher
    showed removing EA is its most catastrophic single ablation (14.1 dB
    survival), while 1x1-cropping its 17px kernels survives at 29.6 dB. So
    the student keeps the cheap small-kernel backbone and buys this instead.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.f(x)


class GatedCNNBlockEA(GatedCNNBlock):
    def __init__(self, dim: int = 64, expansion_ratio: float = 8 / 3,
                 conv_ratio: float = 1.0, dccm: bool = True,
                 se: bool = False) -> None:
        super().__init__(dim, expansion_ratio, conv_ratio, dccm, se)
        self.ea = EA(dim)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        x = self.ea(x)
        return x + shortcut


class RTMoSREA(RTMoSR):
    """RTMoSR with EA-gated blocks. Same constructor semantics."""

    def __init__(self, scale: int = 2, dim: int = 32,
                 ffn_expansion: float = 2, n_blocks: int = 2,
                 unshuffle_mod: bool = False, dccm: bool = True,
                 se: bool = True) -> None:
        super().__init__(scale=scale, dim=dim, ffn_expansion=ffn_expansion,
                         n_blocks=n_blocks, unshuffle_mod=unshuffle_mod,
                         dccm=dccm, se=se)
        self.body = nn.Sequential(*[
            GatedCNNBlockEA(dim, ffn_expansion, dccm=dccm, se=se)
            for _ in range(n_blocks)
        ])
        self.apply(self._init_weights)
