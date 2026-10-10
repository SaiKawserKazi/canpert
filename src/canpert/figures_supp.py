"""Supplementary figures S7-S12 (reviewer-facing analyses)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from canpert import figstyle as S
from canpert.data import CANCER_LINES, LINES
from canpert.figures import METRIC_LABEL, _csv, _methods, _save


def s7_amendment(rep: Path, out: Path) -> None:
    """AMENDMENT-002 diagnostics: predicted DEG counts under the two tests."""
    t1 = pd.read_parquet(rep / "tables" / "t1_per_perturbation.parquet")
    t1 = t1[t1.seed == 0]
    fig = plt.figure(figsize=(S.FULL_W, 62 * S.MM))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45, width_ratios=[1.2, 1, 1])
    a = fig.add_subplot(gs[0])
    meths = _methods(set(t1.method))
    w = 0.38
    x = np.arange(len(meths))
    wil = [t1[t1.method == m].n_de_pred_wilcoxon.mean() for m in meths]
    zt = [t1[t1.method == m].n_de_pred.mean() for m in meths]
    a.bar(x - w / 2, wil, w, color="#d9d8d3", label="Wilcoxon on predicted cells (pre-registered)")
    a.bar(x + w / 2, zt, w, color=[S.COLORS[m] for m in meths], label="moment-matched z-test (amended)")
    a.axhline(t1[t1.method == "CASTER"].n_de_true.mean(), color=S.INK, lw=0.7, ls=(0, (2, 2)),
              label="observed (mean)")
    a.set_yscale("symlog", linthresh=10)
    a.set_xticks(x, meths, rotation=45, ha="right", rotation_mode="anchor")
    a.set_ylabel("Mean predicted # DEGs per perturbation")
    a.legend(fontsize=4.5, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    S.hgrid(a)
    for k, (col, title) in enumerate((("n_de_pred_wilcoxon", "Pre-registered Wilcoxon"),
                                      ("n_de_pred", "Amended z-test"))):
        a = fig.add_subplot(gs[1 + k])
        d = t1[t1.method == "CASTER"]
        a.scatter(np.log10(d.n_de_true + 1), np.log10(d[col] + 1), s=1.5, lw=0, alpha=0.4,
                  color=S.COLORS["CASTER"], rasterized=True)
        a.plot([0, 3.3], [0, 3.3], color=S.INK3, lw=0.5, ls=(0, (2, 2)))
        r = stats.spearmanr(d.n_de_true, d[col]).statistic
        a.set_title(f"{title}: CASTER, rho = {r:.2f}", loc="left", fontsize=5.5)
        a.set_xlabel("log10(1 + observed # DEGs)")
        a.set_ylabel("log10(1 + predicted # DEGs)")
        a.set_xlim(0, 3.3)
        a.set_ylim(0, 3.4)
        S.hgrid(a)
    _csv(pd.DataFrame({"method": meths, "wilcoxon": wil, "ztest": zt}), out, "figS7_data")
    _save(fig, out, "FigureS7")


def s8_gain_strength(rep: Path, out: Path) -> None:
    """CASTER - PerturbMean gain vs observed response strength, T1 and T2."""
    fig, axs = plt.subplots(3, 4, figsize=(S.FULL_W, 120 * S.MM), sharey="row")
    bins = [0, 1, 10, 50, 200, 5000]
    lab = ["0", "1-9", "10-49", "50-199", "200+"]
    rows = []
    for r_, task in enumerate(("T1", "T2", "T1lin")):
        p = rep / "tables" / f"gain_vs_strength_{task[:2]}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if task == "T1lin":
            if "Linear" not in d:
                continue
            d = d.assign(gain=d.CASTER - d.Linear)
        for c, n in enumerate(LINES):
            a = axs[r_, c]
            g = d[d.target == n]
            b = np.digitize(g.n_de_true, bins) - 1
            data = [g.gain[b == i].to_numpy() for i in range(len(lab))]
            bp = a.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                           medianprops={"color": S.INK, "lw": 0.8}, boxprops={"lw": 0.4},
                           whiskerprops={"lw": 0.4}, capprops={"lw": 0})
            for box in bp["boxes"]:
                box.set_facecolor({"T1": "#9ec5f4", "T2": "#d9d8d3", "T1lin": "#f7d48a"}[task])
            a.axhline(0, color=S.INK2, lw=0.5)
            a.set_xticks(range(1, len(lab) + 1), lab, rotation=45, ha="right", rotation_mode="anchor")
            a.set_title({"T1": "Under-represented", "T2": "Zero-shot", "T1lin": "Under-repr., vs Linear"}[task]
                        + f": {S.LINE_LABEL[n]}", loc="left", fontsize=5.0)
            if c == 0:
                a.set_ylabel("CASTER - " + ("Linear" if task == "T1lin" else "PerturbMean") + "\n(Pearson delta)")
            if r_ == 2:
                a.set_xlabel("Observed # DEGs")
            S.hgrid(a)
            for i, v in enumerate(data):
                rows.append({"task": task, "line": n, "bin": lab[i], "median_gain": float(np.median(v))
                             if len(v) else np.nan, "n": len(v)})
    _csv(pd.DataFrame(rows), out, "figS8_data")
    fig.tight_layout()
    _save(fig, out, "FigureS8")


def s9_programs(art: Path, rep: Path, out: Path, line: str = "k562") -> None:
    """Program-activity heatmaps (programs x strongest held-out perturbations)."""
    z = np.load(art / "t1" / f"pred_{line}_s0.npz")
    labels = np.load(rep / "tables" / f"programs_{line}.npy")
    t1 = pd.read_parquet(rep / "tables" / "t1_per_perturbation.parquet")
    d = t1[(t1.target == line) & (t1.method == "CASTER") & (t1.seed == 0)].set_index("perturbation")
    perts = z["perts"]
    strong = np.argsort(-d.loc[perts, "n_de_true"].to_numpy())[:80]
    progs = np.unique(labels)

    def act(x: np.ndarray) -> np.ndarray:
        return np.stack([x[strong][:, labels == k].mean(1) for k in progs])

    true = act(z["_true"])
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = leaves_list(linkage(true.T, "average", metric="correlation"))
    fig, axs = plt.subplots(1, 3, figsize=(S.FULL_W, 62 * S.MM))
    vmax = np.percentile(np.abs(true), 98)
    for a, me in zip(axs, ["_true", "CASTER", "PerturbMean"], strict=True):
        m = act(z[me])[:, order]
        im = a.imshow(m, cmap=S.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        r = np.corrcoef(m.ravel(), true[:, order].ravel())[0, 1]
        a.set_title(("Observed" if me == "_true" else f"{me} (r = {r:.2f})"), loc="left", fontsize=5.5)
        a.set_xlabel(f"80 strongest held-out {S.LINE_LABEL[line]} perturbations")
        a.set_yticks(range(len(progs)), [f"P{k}" for k in progs], fontsize=4.5)
        a.set_xticks([])
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
    cb.set_label("Program activity (mean shift)", fontsize=5)
    cb.ax.tick_params(labelsize=4.5)
    _save(fig, out, "FigureS9")


def s10_examples(art: Path, rep: Path, out: Path) -> None:
    """Best and worst example knockdowns per cancer line (largest and smallest CASTER gain over
    PerturbMean among knockdowns with >= 30 DEGs): CASTER, PerturbMean and Linear."""
    t1 = pd.read_parquet(rep / "tables" / "t1_per_perturbation.parquet")
    t1 = t1[t1.seed == 0]
    meths = ("CASTER", "PerturbMean", "Linear")
    fig, axs = plt.subplots(6, 3, figsize=(S.FULL_W, 230 * S.MM))
    rows = []
    r_ = 0
    for n in CANCER_LINES:
        pv = t1[t1.target == n].pivot_table(index="perturbation", columns="method", values="pearson_delta")
        nd = t1[(t1.target == n) & (t1.method == "CASTER")].set_index("perturbation").n_de_true
        cand = (pv.CASTER - pv.PerturbMean)[nd >= 30].sort_values(ascending=False)
        z = np.load(art / "t1" / f"pred_{n}_s0.npz")
        perts = list(z["perts"])
        for kind, pert in (("best", cand.index[0]), ("worst", cand.index[-1])):
            i = perts.index(pert)
            tl = z["lfc___true"][i].astype(np.float32)
            sig = z["padj___true"][i] < 0.05
            for j, me in enumerate(meths):
                a = axs[r_, j]
                pl = z[me][i]
                a.scatter(tl[~sig], pl[~sig], s=0.6, color="#d9d8d3", lw=0, rasterized=True)
                a.scatter(tl[sig], pl[sig], s=1.2, color=S.COLORS[me], lw=0, rasterized=True)
                a.axhline(0, color=S.INK3, lw=0.4)
                a.axvline(0, color=S.INK3, lw=0.4)
                rr = stats.pearsonr(tl[sig], pl[sig])[0] if sig.sum() > 2 else np.nan
                a.set_title(f"{pert} in {S.LINE_LABEL[n]} ({kind}): {me}, r(DEGs) = {rr:.2f}", loc="left",
                            fontsize=4.6)
                a.tick_params(labelsize=4.3)
                if r_ == 5:
                    a.set_xlabel("Observed log2 FC", fontsize=5)
                if j == 0:
                    a.set_ylabel("Predicted shift", fontsize=5)
                rows.append({"line": n, "perturbation": pert, "case": kind, "method": me, "r_degs": rr,
                             "n_degs": int(sig.sum())})
            r_ += 1
    _csv(pd.DataFrame(rows), out, "figS10_data")
    fig.tight_layout()
    _save(fig, out, "FigureS10")


def s11_zeroshot_metrics(rep: Path, out: Path) -> None:
    """All metrics per line, zero-shot task."""
    ft = pd.read_csv(rep / "tables" / "t2_fold_means.csv")
    metrics = ["pearson_delta", "discrimination", "de_auprc", "de_overlap", "lfc_spearman", "effect_spearman"]
    fig, axs = plt.subplots(len(LINES), len(metrics), figsize=(S.FULL_W, 135 * S.MM))
    for i, n in enumerate(LINES):
        for j, m in enumerate(metrics):
            a = axs[i, j]
            d = ft[ft.target == n]
            meths = _methods(set(d.method))
            table = {me: (float(d[d.method == me][m].mean()), 0.0, []) for me in meths}
            S.bars_with_points(a, table, meths)
            if i < len(LINES) - 1:
                a.set_xticklabels([])
            if j == 0:
                a.set_ylabel(S.LINE_LABEL[n], fontweight="bold")
            if i == 0:
                a.set_title(METRIC_LABEL[m], fontsize=5.5)
    fig.tight_layout()
    _save(fig, out, "FigureS11")


def s12_predicted_h4(rep: Path, out: Path) -> None:
    """Dependency-response coupling (Spearman rho with Chronos) for observed and predicted responses."""
    d = pd.read_csv(rep / "tables" / "H4_predicted.csv")
    fig, a = plt.subplots(figsize=(S.FULL_W * 0.6, 60 * S.MM))
    meths = ["_observed", "CASTER", "PerturbMean", "Linear", "ContextMean"]
    meths = [m for m in meths if m in set(d.method)]
    w = 0.8 / len(meths)
    for j, me in enumerate(meths):
        v = [d[(d.line == n) & (d.method == me)].rho.mean() for n in CANCER_LINES]
        a.bar(np.arange(3) + (j - (len(meths) - 1) / 2) * w, v, w * 0.92,
              color=S.INK if me == "_observed" else S.COLORS[me], label="Observed" if me == "_observed" else me)
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_xticks(range(3), [S.LINE_LABEL[n] for n in CANCER_LINES])
    a.set_ylabel("Spearman rho, log(1 + # DEGs)\nvs Chronos gene effect")
    a.legend(fontsize=4.6, ncol=3, loc="lower left")
    S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS12")


def render_supp(art: Path, rep: Path) -> None:
    """Render S7-S12."""
    S.setup()
    out = rep / "figures"
    s7_amendment(rep, out)
    s8_gain_strength(rep, out)
    s9_programs(art, rep, out)
    s10_examples(art, rep, out)
    s11_zeroshot_metrics(rep, out)
    s12_predicted_h4(rep, out)
