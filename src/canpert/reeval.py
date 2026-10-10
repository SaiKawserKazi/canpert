"""AMENDMENT-002: valid predicted-side differential expression.

Problem (found while checking results, after the pre-registered analysis): the Wilcoxon
rank-sum test applied to *predicted* cells is invalid. Predictions are continuous, and
shift-based ones turn every exact zero of a control cell into a small positive value. Against
zero-inflated reference cells this breaks thousands of ties in the same direction, and the
test calls ~1,300 of 2,000 genes significant for almost any shift. Observed cells keep their
exact zeros, so the observed (ground-truth) DE is unaffected.

Fix: predicted DE is a moment-matched z-test on the predicted pseudobulk shift. The variance
is that of the reference control cells, and the sample sizes are those of the observed
experiment (n perturbed cells vs m = 500 reference controls):

    z_g = delta_g / (s_g * sqrt(1/n + 1/m)),  p_g = 2 * Phi(-|z_g|),  BH-adjusted

Predicted fold-change direction and ranks use delta_g. Ground truth (observed Wilcoxon DEGs and
log2 FC) is unchanged. Pearson delta and discrimination are pseudobulk metrics and are
unaffected. Everything is recomputed from stored arrays; no model is retrained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from canpert.evaluate import FDR, bh
from canpert.experiments import N_REF
from canpert.train import index_line

DE_COLS = ("de_auprc", "de_overlap", "lfc_spearman", "direction", "n_de_pred")


def ref_sd(proc: Path) -> dict[str, np.ndarray]:
    """Per-gene SD of the fixed reference control pool of each line (same pool as evaluation)."""
    from canpert.data import LINES, LineData

    out = {}
    for n in LINES:  # one line at a time to bound memory
        zz = np.load(proc / f"{n}.npz", allow_pickle=False)
        line = LineData(n, zz["x"], zz["pert"], zz["gem"])
        ix = index_line(line)
        out[n] = line.x[ix.ctrl_ref[:N_REF]].std(0, ddof=1).astype(np.float64)
        del line, zz
    return out


def _top(sig: np.ndarray, mag: np.ndarray, k: int) -> set[int]:
    idx = np.where(sig)[0]
    return set(idx[np.argsort(-mag[idx])[:k]].tolist())


def reeval_file(pred_path: Path, metrics_path: Path, sd: np.ndarray, n_cells: dict[str, int]) -> pd.DataFrame:
    """Recompute predicted-side DE metrics for one fold; keeps Wilcoxon versions as *_wilcoxon."""
    z = np.load(pred_path)
    df = pd.read_parquet(metrics_path)
    if "de_auprc_wilcoxon" in df:
        return df  # already amended
    perts = list(z["perts"])
    true_sig = z["padj___true"] < FDR
    true_lfc = z["lfc___true"].astype(np.float64)
    skipped = "pred_wilcoxon_skipped" in df and bool(df["pred_wilcoxon_skipped"].fillna(False).any())
    for c in DE_COLS:
        # review-round runs skipped the (invalid) predicted-side Wilcoxon test: record it as missing
        df[f"{c}_wilcoxon"] = np.nan if skipped else df[c]
    side: dict[str, np.ndarray] = {}
    for method in df.method.unique():
        delta = z[method].astype(np.float64)
        side[f"padj__{method}"] = np.ones(delta.shape, dtype=np.float32)
        for i, p in enumerate(perts):
            n = n_cells[p]
            se = sd * np.sqrt(1.0 / n + 1.0 / N_REF)
            zs = np.where(se > 0, delta[i] / np.where(se > 0, se, 1), 0.0)
            padj = bh(2 * stats.norm.sf(np.abs(zs)))
            sig = padj < FDR
            side[f"padj__{method}"][i] = padj
            row = (df.method == method) & (df.perturbation == p)
            ts = true_sig[i]
            n_true = int(ts.sum())
            vals: dict[str, float] = {"n_de_pred": int(sig.sum())}
            if 0 < n_true < ts.size:
                score = -np.log10(np.clip(padj, 1e-300, 1)) + 1e-9 * np.abs(delta[i])
                vals["de_auprc"] = average_precision_score(ts, score)
                vals["de_overlap"] = len(_top(ts, np.abs(true_lfc[i]), n_true)
                                         & _top(sig, np.abs(delta[i]), n_true)) / n_true
                idx = np.where(ts)[0]
                vals["direction"] = float(np.mean(np.sign(delta[i][idx]) == np.sign(true_lfc[i][idx])))
                vals["lfc_spearman"] = (float(stats.spearmanr(delta[i][idx], true_lfc[i][idx])[0])
                                        if idx.size >= 3 else np.nan)
            else:
                vals.update(de_auprc=np.nan, de_overlap=np.nan, direction=np.nan, lfc_spearman=np.nan)
            for k, val in vals.items():
                df.loc[row, k] = val
    np.savez_compressed(pred_path.with_name(pred_path.name.replace("pred_", "deamended_")), **side)
    df.to_parquet(metrics_path)
    return df


def run(art: Path, proc: Path) -> None:
    """Amend every metrics file under artifacts (t1, t2, ablation, transfer)."""
    sd = ref_sd(proc)
    from canpert.analysis import load_observed

    obs = load_observed(art / "observed")
    ncell = {n: dict(zip(obs[n]["perts"], obs[n]["n_cells"], strict=True)) for n in obs}
    for task in ("t1", "t2", "ablation", "transfer"):
        for mp in sorted((art / task).glob("metrics_*.parquet")):
            tag = mp.stem.removeprefix("metrics_")
            pp = art / task / f"pred_{tag}.npz"
            target = pd.read_parquet(mp, columns=["target"]).target.iloc[0]
            reeval_file(pp, mp, sd[target], ncell[target])
            print(f"amended {task}/{tag}", flush=True)
