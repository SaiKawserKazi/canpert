"""PertBench: the evaluation suite (expression, differential expression, effect size).

For each test perturbation t we have observed perturbed cells X_t, predicted cells
X_hat_t (same count), and a fixed pool of reference control cells R (disjoint from the
control cells given to the models). Metrics:

Expression
- ``pearson_delta``: Pearson r between (mean X_hat_t - mean R) and (mean X_t - mean R).
- ``mae``: mean absolute error of pseudobulks.
- ``discrimination``: 1 - 2 r_t / (T-1), where r_t counts other test perturbations t'
  whose observed pseudobulk is closer (L1) to the prediction than the true one. 1 is
  perfect, 0 is chance.

Differential expression (Wilcoxon rank-sum vs R, Benjamini-Hochberg FDR < 0.05)
- ``de_auprc``: area under the precision-recall curve for recovering true DEGs from
  predicted -log10 adjusted p values.
- ``de_overlap``: |top-N true DEGs & top-N predicted DEGs| / N with N = number of true
  DEGs, both ranked by |log2 fold change| among significant genes.
- ``lfc_spearman``: Spearman rho of log2 fold changes over the true DEGs.
- ``direction``: fraction of true DEGs whose predicted fold change has the right sign.

Effect size (per fold)
- ``effect_spearman``: Spearman rho across perturbations between the number of predicted
  and observed DEGs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

FDR = 0.05


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (1-D)."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def de_test(cells: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wilcoxon rank-sum (asymptotic, tie-corrected) per gene -> (padj, log2FC)."""
    _, p = stats.mannwhitneyu(cells, ref, axis=0, method="asymptotic")
    p = np.where(np.isfinite(p), p, 1.0)
    a = np.expm1(np.clip(cells, 0, None)).mean(0)
    b = np.expm1(np.clip(ref, 0, None)).mean(0)
    lfc = np.log2((a + 1e-2) / (b + 1e-2))
    return bh(p), lfc


@dataclass
class DE:
    """DE result for one cell set."""

    padj: np.ndarray
    lfc: np.ndarray

    @property
    def sig(self) -> np.ndarray:
        """Boolean mask of significant genes."""
        return self.padj < FDR


def _top(de: DE, n: int) -> set[int]:
    sig = np.where(de.sig)[0]
    return set(sig[np.argsort(-np.abs(de.lfc[sig]))[:n]].tolist())


def per_perturbation(true: DE, pred: DE, d_true: np.ndarray, d_pred: np.ndarray) -> dict[str, float]:
    """Metrics for one perturbation given DE results and pseudobulk deltas."""
    out: dict[str, float] = {}
    out["pearson_delta"] = float(stats.pearsonr(d_pred, d_true)[0]) if np.std(d_pred) > 0 else 0.0
    out["mae"] = float(np.mean(np.abs(d_pred - d_true)))
    n_true = int(true.sig.sum())
    out["n_de_true"] = n_true
    out["n_de_pred"] = int(pred.sig.sum())
    if 0 < n_true < true.sig.size:
        score = -np.log10(np.clip(pred.padj, 1e-300, 1.0))
        score = score + 1e-6 * np.abs(pred.lfc)  # break ties among saturated p values
        out["de_auprc"] = float(average_precision_score(true.sig, score))
        out["de_auprc_base"] = n_true / true.sig.size
        out["de_overlap"] = len(_top(true, n_true) & _top(pred, n_true)) / n_true
        idx = np.where(true.sig)[0]
        out["direction"] = float(np.mean(np.sign(pred.lfc[idx]) == np.sign(true.lfc[idx])))
        out["lfc_spearman"] = (
            float(stats.spearmanr(pred.lfc[idx], true.lfc[idx])[0]) if idx.size >= 3 else np.nan
        )
    else:
        for k in ("de_auprc", "de_auprc_base", "de_overlap", "direction", "lfc_spearman"):
            out[k] = np.nan
    return out


def discrimination(pred_pb: np.ndarray, true_pb: np.ndarray) -> np.ndarray:
    """Normalised L1 discrimination score per perturbation (rows aligned)."""
    t = pred_pb.shape[0]
    own = np.abs(pred_pb - true_pb).sum(-1)
    closer = np.zeros(t, dtype=np.int64)
    for i in range(0, t, 64):  # chunked: (64, T, G) at a time keeps memory bounded
        d = np.abs(pred_pb[i:i + 64, None, :] - true_pb[None, :, :]).sum(-1)
        closer[i:i + 64] = (d < own[i:i + 64, None]).sum(1)
    return 1.0 - 2.0 * closer / max(t - 1, 1)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and s.e.m. of every metric grouped by ``by``."""
    metrics = [c for c in df.select_dtypes("number").columns if c not in {*by, "fold", "seed"}]
    g = df.groupby(by)[metrics]
    mean = g.mean().add_suffix("_mean")
    sem = g.sem().add_suffix("_sem")
    return pd.concat([mean, sem], axis=1).reset_index()
