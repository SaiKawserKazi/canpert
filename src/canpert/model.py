"""CASTER: Context-Adaptive Set Transformer for pErturbation Responses.

CASTER maps a *set* of S unperturbed cells from a context (cell line) plus a perturbation
label to a set of S predicted perturbed cells. The context is never given as a label: the
model infers it from the control set through self-attention, so it can be applied to
contexts with few (or no) observed perturbations.

Architecture (all dimensions h):
    d_src  = transfer prior: the perturbation's mean observed shift in *other* contexts
    H_cell = f_cell(X_ctrl)                          4-layer GELU MLP, per cell
    H_pert = f_prior(d_src) [+ f_pert(onehot(p)) in the ``pert_id`` ablation]
    c      = f_ctx(mean_s H_cell)                    context summary of the control set
    H      = H_cell + H_pert + FiLM(c) (x) H_pert    context-modulated perturbation signal
    O      = H_cell + f_ST(H)                        bidirectional transformer over the set
    X_hat  = X_ctrl + (1 + g(c)) (.) d_src + f_out(O)

g(c) is a per-gene, context-dependent rescaling of the transferred effect (initialised at
zero, so training starts from the PerturbMean solution); f_out(O) adds a cell-resolved,
context-specific correction learned by attention over the control set. The prior never
contains the target context (see ``train.PriorBank``). The loss is the squared energy-distance MMD between the
predicted and observed perturbed sets, plus a small pseudobulk-mean term.
Ablations (Figure 1): ``attention=False`` replaces the transformer by a per-cell MLP
(no cell-cell interaction); ``film=False`` removes context modulation; set size S=1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CasterConfig:
    """Hyperparameters (defaults used for all main results)."""

    n_genes: int
    n_perts: int
    hidden: int = 256
    layers: int = 4
    heads: int = 8
    attention: bool = True
    film: bool = True
    prior: bool = True
    pert_id: bool = False
    mean_weight: float = 1.0


def _mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int = 4) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = d_in
    for _ in range(n_layers - 1):
        mods += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class Caster(nn.Module):
    """Set-to-set perturbation response model."""

    def __init__(self, cfg: CasterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.f_cell = _mlp(cfg.n_genes, h, h)
        self.f_pert = nn.Sequential(nn.Embedding(cfg.n_perts, h), nn.GELU(), nn.Linear(h, h))
        self.f_ctx = _mlp(h, h, 2 * h, n_layers=2)
        if cfg.attention:
            layer = nn.TransformerEncoderLayer(
                h, cfg.heads, dim_feedforward=2 * h, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.f_st: nn.Module = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        else:
            self.f_st = _mlp(h, 2 * h, h, n_layers=cfg.layers)
        self.f_out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.n_genes))
        if cfg.prior:
            # start exactly at the transfer-prior solution: the correction head is zero
            nn.init.zeros_(self.f_out[1].weight)
            nn.init.zeros_(self.f_out[1].bias)
            self.f_prior = _mlp(cfg.n_genes, h, h, n_layers=2)
            self.g_scale = nn.Linear(h, cfg.n_genes)
            nn.init.zeros_(self.g_scale.weight)
            nn.init.zeros_(self.g_scale.bias)

    def forward(self, x_ctrl: torch.Tensor, pert: torch.Tensor,
                prior: torch.Tensor | None = None) -> torch.Tensor:
        """x_ctrl: (B, S, G); pert: (B,) long; prior: (B, G) -> predicted set (B, S, G)."""
        h_cell = self.f_cell(x_ctrl)
        if self.cfg.pert_id or not self.cfg.prior:
            h_pert = self.f_pert(pert)
        else:
            h_pert = torch.zeros(x_ctrl.shape[0], self.cfg.hidden, dtype=x_ctrl.dtype)
        if self.cfg.prior:
            assert prior is not None
            h_pert = h_pert + self.f_prior(prior)
        h_pert = h_pert[:, None, :]
        hidden = h_cell + h_pert
        ctx = h_cell.mean(dim=1)
        if self.cfg.film:
            gamma, beta = self.f_ctx(ctx).chunk(2, dim=-1)
            hidden = hidden + gamma[:, None, :] * h_pert + beta[:, None, :]
        out = h_cell + self.f_st(hidden)
        pred = x_ctrl + self.f_out(out)
        if self.cfg.prior:
            assert prior is not None
            pred = pred + ((1 + self.g_scale(ctx)) * prior)[:, None, :]
        return pred


def energy_mmd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared MMD with the energy kernel k(u,v) = -||u-v||, averaged over the batch."""
    def mean_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(a, b).mean(dim=(1, 2))

    mmd = 2 * mean_dist(pred, target) - mean_dist(pred, pred) - mean_dist(target, target)
    return mmd.mean()


def caster_loss(pred: torch.Tensor, target: torch.Tensor, mean_weight: float) -> torch.Tensor:
    """Energy MMD over sets + mean-squared error of set means (pseudobulk anchor)."""
    loss = energy_mmd(pred, target)
    if mean_weight > 0:
        loss = loss + mean_weight * ((pred.mean(1) - target.mean(1)) ** 2).sum(-1).mean()
    return loss
