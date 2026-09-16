"""Block-grid loss: penalise codec-block-aligned gradient energy the GT does not have.

WHY (talos-cluster-28or, 2026-09-13/14): on the consolidation blind A/B the
production d64 (iter145000) beat the best faithful checkpoint 6:1 on look while
losing to it on every metric in the recipe (foliage DISTS 0.191 vs 0.228, LPIPS,
hf). What the eye rejected, measured after the fact, was x264 macroblock
structure carried through from the LR and sharpened: the mean |horizontal luma
gradient| on columns that sit on a block boundary, over the mean on all other
columns, reads 1.00 on GT and 1.05-1.26 on both students' foliage crops. The
loss stack (msssim-l1, perceptual, DISTS, HSLuv, LDL, FF, GAN) scores that grid
energy as texture. This term scores it as the artefact it is.

WHAT: for prediction and target, luma horizontal- and vertical-gradient
magnitude; for each period P (8 = the LR macroblock edge, 16 = the same edge at
2x output) and each phase k in 0..P-1, the mean gradient on the columns (rows)
x % P == k over the mean on all columns (rows). The maximum over k is the grid
energy at whatever phase the grid actually sits -- PHASE-AGNOSTIC, so the term
needs no crop offset (traiNNer's paired_random_crop draws an arbitrary integer
LQ offset, so the grid's phase inside a training crop is uniform). The loss is
relu(max_k pred_ratio - max_k target_ratio), averaged over the batch, axes and
periods: only grid energy in excess of the GT's is penalised, and a GT that
itself carries a grid (some do, at 1.03) is not chased below its own level.

traiNNer calls a generic loss as ``loss(output, gt)`` and multiplies by
``loss_weight`` itself, so ``forward`` returns the UNWEIGHTED term. Registered
under LOSS_REGISTRY as ``blockgridloss`` (traiNNer lower-cases the class name);
in a config: ``- {type: blockgridloss, loss_weight: <w>, periods: [8, 16]}``.
Also runnable as a script for calibration: ``python -m
userdevice_rtus.blockgrid_loss <pred.png> <gt.png> [...]`` prints the raw term.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:  # the real registry when traiNNer is importable, see _registry.py
    from traiNNer.utils.registry import LOSS_REGISTRY  # type: ignore
except ImportError:  # pragma: no cover - eval/export environments
    from userdevice_rtus._registry import ARCH_REGISTRY as LOSS_REGISTRY  # type: ignore

_LUMA = (0.299, 0.587, 0.114)


def luma(x: Tensor) -> Tensor:
    """N,3,H,W (any range) -> N,1,H,W Rec.601 luma. A 1-channel input passes through."""
    if x.shape[1] == 1:
        return x
    w = x.new_tensor(_LUMA).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def grid_ratios(x: Tensor, period: int, eps: float = 1e-6) -> Tensor:
    """Per-phase grid ratio along both axes.

    Returns N,2,period: [:,0,k] is the mean |dx| on columns x % period == k over
    the mean |dx| on all columns; [:,1,k] the same for rows. ``dx[..., j]`` is
    the edge between column j and j+1, i.e. the boundary at column j+1, so the
    boundary column index is j+1 -- that is what ``x % period`` is taken on.
    """
    y = luma(x)
    dx = (y[..., :, 1:] - y[..., :, :-1]).abs()  # N,1,H,W-1
    dy = (y[..., 1:, :] - y[..., :-1, :]).abs()  # N,1,H-1,W
    out = []
    for d, axis_len in ((dx, dx.shape[-1]), (dy, dy.shape[-2])):
        idx = torch.arange(1, axis_len + 1, device=x.device) % period
        mean_all = d.mean(dim=(1, 2, 3)) + eps  # N
        per_phase = []
        for k in range(period):
            sel = idx == k
            if d is dx:
                on = d[..., :, sel].mean(dim=(1, 2, 3))
            else:
                on = d[..., sel, :].mean(dim=(1, 2, 3))
            per_phase.append(on / mean_all)
        out.append(torch.stack(per_phase, dim=1))  # N,period
    return torch.stack(out, dim=1)  # N,2,period


@LOSS_REGISTRY.register()
class BlockGridLoss(nn.Module):
    """relu(max-phase grid ratio of pred - that of target), per period and axis."""

    def __init__(
        self, loss_weight: float = 1.0, periods: tuple[int, ...] | list[int] = (8, 16)
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.periods = tuple(int(p) for p in periods)
        if not self.periods or any(p < 2 for p in self.periods):
            raise ValueError(f"periods must be >= 2, got {self.periods}")

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        terms = []
        for p in self.periods:
            rp = grid_ratios(pred, p).amax(dim=2)  # N,2
            rt = grid_ratios(target, p).amax(dim=2)
            terms.append(torch.relu(rp - rt).mean())
        return torch.stack(terms).mean()


def _main(argv: list[str]) -> int:
    """Calibration: raw term for (pred, gt) PNG pairs, whole and on 128 px crops."""
    import numpy as np
    from PIL import Image

    if len(argv) < 2 or len(argv) % 2:
        print("usage: python -m userdevice_rtus.blockgrid_loss pred.png gt.png [...]")
        return 2
    loss = BlockGridLoss(periods=(8, 16))
    g = torch.Generator().manual_seed(1024)
    for i in range(0, len(argv), 2):
        imgs = []
        for path in argv[i : i + 2]:
            a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(a).permute(2, 0, 1)[None])
        pred, gt = imgs
        h, w = pred.shape[-2:]
        whole = float(loss(pred, gt))
        crops = []
        if h >= 128 and w >= 128:
            for _ in range(32):
                t = int(torch.randint(0, h - 128 + 1, (1,), generator=g))
                x = int(torch.randint(0, w - 128 + 1, (1,), generator=g))
                win = (..., slice(t, t + 128), slice(x, x + 128))
                crops.append(float(loss(pred[win], gt[win])))
        r8 = grid_ratios(pred, 8).amax(dim=2)[0].tolist()
        r16 = grid_ratios(pred, 16).amax(dim=2)[0].tolist()
        g8 = grid_ratios(gt, 8).amax(dim=2)[0].tolist()
        g16 = grid_ratios(gt, 16).amax(dim=2)[0].tolist()
        if crops:
            crop_s = f"crop128 mean {sum(crops) / len(crops):.4f} max {max(crops):.4f}"
        else:
            crop_s = "crop128 n/a"
        print(
            f"{argv[i]}: term whole {whole:.4f} {crop_s}  "
            f"pred max-ratio P8 x/y {r8[0]:.3f}/{r8[1]:.3f} "
            f"P16 {r16[0]:.3f}/{r16[1]:.3f}  "
            f"gt P8 {g8[0]:.3f}/{g8[1]:.3f} P16 {g16[0]:.3f}/{g16[1]:.3f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
