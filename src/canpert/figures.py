"""Main and supplementary figures (Cell layout, 600 dpi PNG + vector PDF).

Every figure reads only ``artifacts/`` and ``reports/tables`` and writes the data it plots
as CSV next to the image (``<fig>_data_<panel>.csv``).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats
from sklearn.metrics import precision_recall_curve

from canpert import figstyle as S
from canpert.data import CANCER_LINES, LINES

METRIC_LABEL = {
    "pearson_delta": "Pearson delta",
    "discrimination": "Perturbation\ndiscrimination",
    "de_auprc": "DE AUPRC",
    "de_overlap": "DE overlap",
    "lfc_spearman": "Fold-change\nSpearman",
    "effect_spearman": "Effect-size\nSpearman",
    "direction": "Direction\nagreement",
    "mae": "MAE",
}


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _csv(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)


def _methods(present: set[str]) -> list[str]:
    return [m for m in S.METHODS if m in present]


# ============================================================================ Figure 1
def draw_context_schematic(ax: plt.Axes) -> None:
    """Same knockdown, different contexts, different responses."""
    S.schematic_axes(ax)
    ax.text(0.5, 0.97, "One perturbation, four cellular contexts", ha="center", va="top",
            fontsize=6.5, fontweight="bold")
    xs = [0.14, 0.38, 0.62, 0.86]
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 12)
    for i, (n, x) in enumerate(zip(LINES, xs, strict=True)):
        S.cell_population(ax, x, 0.72, 0.075, S.CTX_FILL[n], n=7, seed=i, ec=S.INK3)
        ax.text(x, 0.585, f"{S.LINE_LABEL[n]}", ha="center", fontsize=6, fontweight="bold")
        ax.text(x, 0.535, S.LINE_DESC[n], ha="center", fontsize=5, color=S.INK2)
        S.arrow(ax, x, 0.50, x, 0.44, color=S.INK3)
        # response barcode: shared core + context-specific part
        eff = base * (0.55 if n == "rpe1" else 1.0) + rng.normal(0, 0.55, 12)
        for j, v in enumerate(eff):
            c = S.DIVERGING(0.5 + np.clip(v, -2, 2) / 4)
            ax.add_patch(Rectangle((x - 0.09 + j * 0.015, 0.30), 0.013, 0.11, fc=c, ec="none"))
        ax.text(x, 0.265, "response", ha="center", fontsize=4.8, color=S.INK3)
    S.lightning(ax, 0.02, 0.80, 0.05)
    ax.text(0.075, 0.845, "CRISPRi\nknockdown", fontsize=5, color=S.ACCENT_PERT, va="center")
    S.box(ax, 0.03, 0.03, 0.94, 0.17,
          "response = shared core + context-specific part\n"
          "Can the context-specific part be predicted?",
          fc="#f4f3f0", ec=S.GRID, fs=5.2)


def draw_caster_schematic(ax: plt.Axes) -> None:
    """CASTER architecture."""
    S.schematic_axes(ax)
    # input control set
    S.box(ax, 0.0, 0.52, 0.17, 0.40, "", fc="#eef4fc", ec="#9ec5f4", r=0.03)
    ax.text(0.085, 0.95, "Control cell set\n(target context)", ha="center", va="bottom", fontsize=5.2)
    for k in range(5):
        S.cell(ax, 0.045 + (k % 2) * 0.08, 0.84 - k * 0.07, 0.022, "#cde2fb")
    ax.text(0.085, 0.47, "S cells x G genes", ha="center", fontsize=4.8, color=S.INK2)
    # prior
    S.box(ax, 0.0, 0.05, 0.17, 0.30, "Transfer prior $d_{src}$\nshift of p in\nother contexts",
          fc="#fdebe3", ec="#f3a3a2", fs=4.8, r=0.03)
    S.lightning(ax, 0.015, 0.27, 0.035)
    # encoders
    S.box(ax, 0.23, 0.68, 0.14, 0.14, "$f_{cell}$\nMLP", fc="white", ec=S.INK3, fs=5)
    S.box(ax, 0.23, 0.15, 0.14, 0.14, "$f_{prior}$\nMLP", fc="white", ec=S.INK3, fs=5)
    S.box(ax, 0.23, 0.42, 0.14, 0.14, "context\nsummary c", fc="white", ec=S.INK3, fs=5)
    S.arrow(ax, 0.17, 0.75, 0.23, 0.75)
    S.arrow(ax, 0.17, 0.22, 0.23, 0.22)
    S.arrow(ax, 0.30, 0.68, 0.30, 0.56)
    # film + transformer
    S.box(ax, 0.43, 0.30, 0.20, 0.55, "", fc="#e7f6ee", ec="#1baf7a", r=0.03)
    ax.text(0.53, 0.80, "Set transformer", ha="center", fontsize=5.4, fontweight="bold", color="#0d6b4a")
    ax.text(0.53, 0.74, "self-attention over cells", ha="center", fontsize=4.7, color=S.INK2)
    for k in range(4):
        y = 0.40 + k * 0.075
        ax.add_patch(Rectangle((0.46, y), 0.14, 0.05, fc="white", ec="#1baf7a", lw=0.5))
        for q in range(4):
            ax.add_patch(plt.Circle((0.48 + q * 0.033, y + 0.025), 0.009, fc="#86b6ef", ec="none"))
    S.arrow(ax, 0.37, 0.75, 0.43, 0.65)
    S.arrow(ax, 0.37, 0.22, 0.43, 0.40)
    S.arrow(ax, 0.37, 0.49, 0.43, 0.55)
    ax.text(0.385, 0.525, "FiLM", fontsize=4.4, color=S.INK2, ha="center", va="bottom")
    # outputs
    S.box(ax, 0.69, 0.55, 0.13, 0.17, "$f_{out}$\ncorrection", fc="white", ec=S.INK3, fs=5)
    S.box(ax, 0.69, 0.20, 0.13, 0.17, "$1+g(c)$\ngene-wise\nrescaling", fc="white", ec=S.INK3, fs=4.8)
    S.arrow(ax, 0.63, 0.62, 0.69, 0.63)
    S.arrow(ax, 0.17, 0.12, 0.755, 0.12, ls="--", color="#f3a3a2")
    S.arrow(ax, 0.755, 0.12, 0.755, 0.20, color="#f3a3a2")
    S.box(ax, 0.86, 0.36, 0.14, 0.30, "", fc="#fde3d6", ec="#eb6834", r=0.03)
    ax.text(0.93, 0.70, "Predicted\nperturbed set", ha="center", va="bottom", fontsize=5.2)
    for k in range(4):
        S.cell(ax, 0.90 + (k % 2) * 0.06, 0.60 - k * 0.065, 0.02, "#f7c1a6")
    S.arrow(ax, 0.82, 0.63, 0.86, 0.56)
    S.arrow(ax, 0.82, 0.28, 0.86, 0.44)
    ax.text(0.93, 0.27, "loss: energy MMD\n+ pseudobulk MSE", ha="center", va="top", fontsize=4.6,
            color=S.INK2)


def figure1(art: Path, rep: Path, out: Path) -> None:
    res = json.loads((rep / "results.json").read_text())
    fig = plt.figure(figsize=(S.FULL_W, 175 * S.MM))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.05, 1.0, 1.0], hspace=0.62, wspace=1.0)
    # A
    ax = fig.add_subplot(gs[0, :3])
    draw_context_schematic(ax)
    S.fig_letter(fig, "A", 0.07, 0.905)
    # B data summary
    axb = fig.add_subplot(gs[0, 3:5])
    audit = res.get("audit", {})
    obs_n = res["observed_n_de"]
    cells = [audit.get(f"{n}_cells", np.nan) for n in LINES]
    perts = [obs_n[n]["n_perturbations"] for n in LINES]
    y = np.arange(len(LINES))
    axb.barh(y, perts, color=[S.CTX_FILL[n] for n in LINES], ec=S.INK3, lw=0.4, height=0.62)
    for i, n in enumerate(LINES):
        axb.text(perts[i] * 1.03, i, f"{S.LINE_DESC[n]}\n{perts[i]:,} perts, {int(cells[i]) if np.isfinite(cells[i]) else 0:,} cells",
                 va="center", fontsize=4.8, color=S.INK2)
    axb.set_yticks(y)
    axb.set_yticklabels([S.LINE_LABEL[n] for n in LINES], fontsize=5.5)
    axb.invert_yaxis()
    axb.set_xlabel("Perturbations profiled")
    axb.set_xlim(0, max(perts) * 2.0)
    axb.set_title("CRISPRi Perturb-seq contexts", loc="left", pad=8)
    S.fig_letter(fig, "B", 0.515, 0.905)
    _csv(pd.DataFrame({"line": LINES, "perturbations": perts, "cells": cells}), out, "fig1_data_B")
    # B' overlap
    axo = fig.add_subplot(gs[0, 5])
    per = json.loads((art.parent / "data/processed/perturbations.json").read_text()) \
        if (art.parent / "data/processed/perturbations.json").exists() else None
    if per is None:
        per = {n: list(np.load(art / "observed" / f"observed_{n}.npz")["perts"]) for n in LINES}
    sets = {n: set(v) for n, v in per.items()}
    k_counts = pd.Series([sum(p in sets[n] for n in LINES) for p in set().union(*sets.values())]).value_counts()
    ks = [1, 2, 3, 4]
    axo.bar(ks, [k_counts.get(k, 0) for k in ks], color=S.SEQ_BLUE([0.3, 0.5, 0.7, 0.9]), width=0.7)
    axo.set_xticks(ks)
    axo.set_xlabel("Contexts observing\nthe perturbation")
    axo.set_ylabel("Perturbations")
    S.hgrid(axo)
    _csv(pd.DataFrame({"n_contexts": ks, "n_perturbations": [k_counts.get(k, 0) for k in ks]}), out, "fig1_data_B2")
    # C architecture
    axc = fig.add_subplot(gs[1, :4])
    draw_caster_schematic(axc)
    axc.set_title("CASTER: Context-Adaptive Set Transformer for pErturbation Responses",
                  loc="left", fontsize=6.5, fontweight="bold", pad=16)
    S.fig_letter(fig, "C", 0.07, 0.61)
    # D cross-context similarity heatmap
    sim = pd.read_csv(rep / "tables" / "cross_context_similarity.csv")
    axd = fig.add_subplot(gs[1, 4:])
    mat = np.full((4, 4), np.nan)
    for (a, b), g in sim.groupby(["a", "b"]):
        i, j = LINES.index(a), LINES.index(b)
        mat[i, j] = mat[j, i] = g["pearson"].median()
    np.fill_diagonal(mat, 1.0)
    im = axd.imshow(mat, cmap=S.SEQ_BLUE, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            axd.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5,
                     color="white" if mat[i, j] > 0.6 else S.INK)
    lab = [S.LINE_LABEL[n] for n in LINES]
    axd.set_xticks(range(4), lab, rotation=45, ha="right", rotation_mode="anchor")
    axd.set_yticks(range(4), lab)
    axd.set_title("Median response similarity\n(Pearson r of shifts, >=10 DEGs)", loc="left")
    cb = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5)
    S.fig_letter(fig, "D", 0.66, 0.61)
    # E distributions by pair type
    axe = fig.add_subplot(gs[2, :2])
    order = sorted(sim.groupby(["a", "b"]).groups.keys(), key=lambda t: -sim[(sim.a == t[0]) & (sim.b == t[1])].pearson.median())
    data = [sim[(sim.a == a) & (sim.b == b)]["pearson"].to_numpy() for a, b in order]
    pos = np.arange(len(order))
    vp = axe.violinplot(data, positions=pos, widths=0.8, showextrema=False)
    for body, (a, b) in zip(vp["bodies"], order, strict=True):
        body.set_facecolor("#9ec5f4" if "rpe1" not in (a, b) else "#d9d8d3")
        body.set_edgecolor("none")
        body.set_alpha(1)
    axe.boxplot(data, positions=pos, widths=0.18, showfliers=False, medianprops={"color": S.INK, "lw": 0.8},
                boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5}, capprops={"lw": 0})
    axe.set_xticks(pos, [f"{S.LINE_LABEL[a]}-{S.LINE_LABEL[b]}" for a, b in order], rotation=45, ha="right",
                   rotation_mode="anchor", fontsize=5)
    axe.set_ylabel("Response similarity (r)")
    axe.legend(handles=[Patch(fc="#9ec5f4", label="cancer-cancer"), Patch(fc="#d9d8d3", label="with RPE1")],
               loc="lower left", fontsize=5, handlelength=1)
    S.hgrid(axe)
    S.fig_letter(fig, "E", 0.07, 0.33)
    # F set-size ablation: validation curves
    hist = res.get("ablation_history", {})
    axf = fig.add_subplot(gs[2, 2:4])
    ss = [("S1", 1), ("S4", 4), ("S8", 8), ("S16", 16), ("S32", 32)]
    cols = S.SEQ_BLUE(np.linspace(0.35, 1.0, len(ss)))
    rows = []
    for (tag, sz), c in zip(ss, cols, strict=True):
        if tag not in hist:
            continue
        h = pd.DataFrame(hist[tag])
        axf.plot(h["epoch"] + 1, h["val"] / h["val"].iloc[0], color=c, lw=1.0, label=f"S = {sz}")
        rows.append(h.assign(variant=tag))
    axf.set_xlabel("Epoch")
    axf.set_ylabel("Validation loss\n(relative to epoch 1)")
    axf.legend(fontsize=4.8, ncol=1, handlelength=1.2, loc="upper right")
    S.hgrid(axf)
    if rows:
        _csv(pd.concat(rows), out, "fig1_data_F")
    S.fig_letter(fig, "F", 0.37, 0.33)
    # G ablations (20 epochs; dots = seeds) with PerturbMean and Linear on the same split
    axg = fig.add_subplot(gs[2, 4:])
    ab20 = res.get("review", {}).get("ablation20", {}).get("means")
    if ab20:
        ab = pd.DataFrame(ab20)
        order_ab = [t for t in ["S1", "S8", "S32", "S32_noattn", "S32_nofilm", "S32_noprior", "S32_pertid"]
                    if t in set(ab.method)]
        mu = ab.groupby("method").pearson_delta.mean().reindex(order_ab)
        colors = [S.COLORS["CASTER"] if t == "S32" else "#9ec5f4" if "_" not in t else "#d9d8d3" for t in order_ab]
        axg.bar(range(len(order_ab)), mu, color=colors, width=0.7)
        for i_, t in enumerate(order_ab):
            pts = ab[ab.method == t].pearson_delta
            axg.scatter(np.full(len(pts), i_), pts, s=4, color=S.INK, zorder=3, lw=0)
        for ref, ls in (("PerturbMean", (0, (2, 2))), ("Linear", (0, (5, 2)))):
            v = ab[ab.method == ref].pearson_delta.mean()
            if np.isfinite(v):
                axg.axhline(v, color=S.COLORS[ref], lw=0.8, ls=ls, label=ref)
        axg.set_xticks(range(len(order_ab)),
                       [t.replace("S32_pertid", "+ ID embedding").replace("S32_no", "no ")
                        .replace("attn", "attention") for t in order_ab],
                       rotation=45, ha="right", rotation_mode="anchor", fontsize=5)
        axg.set_ylim(0, max(mu.max(), ab.pearson_delta.max()) * 1.15)
        axg.set_ylabel("Pearson delta (K562 test)")
        axg.legend(fontsize=4.5, loc="upper right", handlelength=1.6)
        S.hgrid(axg)
        _csv(ab, out, "fig1_data_G")
    S.fig_letter(fig, "G", 0.66, 0.33)
    _save(fig, out, "Figure1")


# ============================================================================ Figure 2
def draw_task_schematic(ax: plt.Axes, zero_shot: bool = False) -> None:
    """Contexts x perturbations grid with train / test cells."""
    S.schematic_axes(ax)
    n_c, n_p = 4, 10
    x0, y0, w, h = 0.16, 0.33, 0.70, 0.36
    cw, ch = w / n_c, h / n_p
    rng = np.random.default_rng(1)
    for c in range(n_c):
        for p in range(n_p):
            held = c == 1
            if held:
                train = (not zero_shot) and p < 4
                fc = "#9ec5f4" if train else "#2a78d6"
            else:
                fc = "#d9d8d3" if rng.random() > 0.15 else "white"
            ax.add_patch(Rectangle((x0 + c * cw + 0.004, y0 + (n_p - 1 - p) * ch + 0.003), cw - 0.008, ch - 0.006,
                                   fc=fc, ec="white", lw=0.3))
    for c, n in enumerate(LINES):
        ax.text(x0 + (c + 0.5) * cw, y0 + h + 0.02, S.LINE_LABEL[n], ha="left", va="bottom", fontsize=4.8,
                rotation=40, rotation_mode="anchor")
    ax.text(x0 - 0.02, y0 + h / 2, "Perturbations", rotation=90, ha="right", va="center", fontsize=5.2)
    items = [("#d9d8d3", "train (other contexts)"),
             ("#2a78d6", "test (held-out context)")]
    if not zero_shot:
        items.insert(1, ("#9ec5f4", "train + val (25% + 5% of held-out)"))
    for i, (c, t) in enumerate(items):
        ax.add_patch(Rectangle((x0, 0.21 - i * 0.085), 0.035, 0.055, fc=c, ec="none"))
        ax.text(x0 + 0.05, 0.237 - i * 0.085, t, fontsize=4.8, va="center")
    ax.text(0.5, 0.99, "Zero-shot context" if zero_shot else "Under-represented context",
            ha="center", va="top", fontsize=6, fontweight="bold")


def figure2(art: Path, rep: Path, out: Path) -> None:
    ft_all = pd.read_csv(rep / "tables" / "t1_fold_means.csv")
    ft = ft_all[ft_all.seed == 0]
    t1 = pd.read_parquet(rep / "tables" / "t1_per_perturbation.parquet")
    t1 = t1[t1.seed == 0]
    methods = _methods(set(ft.method))
    metrics = ["pearson_delta", "discrimination", "de_auprc", "de_overlap", "lfc_spearman", "effect_spearman"]
    fig = plt.figure(figsize=(S.FULL_W, 205 * S.MM))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.8, 2.6, 1.15], hspace=0.42)
    top = GridSpecFromSubplotSpec(1, 3, outer[0], wspace=0.35, width_ratios=[1.1, 1, 1.1])
    ax = fig.add_subplot(top[0])
    draw_task_schematic(ax)
    S.fig_letter(fig, "A", 0.08, 0.905)
    # B: pooled bars for primary metrics
    axb = fig.add_subplot(top[1:])
    S.schematic_axes(axb)
    inner = GridSpecFromSubplotSpec(1, 3, top[1:], wspace=0.55)
    for k, m in enumerate(["pearson_delta", "discrimination", "de_auprc"]):
        a = fig.add_subplot(inner[k])
        g = ft_all.groupby("method")[m]
        table = {me: (g.mean()[me], g.std()[me], ft_all[ft_all.method == me][m].tolist()) for me in methods}
        S.bars_with_points(a, table, methods, ylabel=METRIC_LABEL[m].replace("\n", " "))
        a.set_title(f"4 contexts x {ft_all.seed.nunique()} seeds", fontsize=5.5, loc="left", color=S.INK2)
    S.fig_letter(fig, "B", 0.37, 0.905)
    # C: rows = lines, cols = metrics
    grid = GridSpecFromSubplotSpec(len(LINES), len(metrics), outer[1], hspace=0.75, wspace=0.55)
    t1_stats = []
    for i, n in enumerate(LINES):
        for j, m in enumerate(metrics):
            a = fig.add_subplot(grid[i, j])
            d = ft[ft.target == n]
            sub = t1[(t1.target == n)]
            table = {}
            for me in methods:
                v = d[d.method == me][m]
                if m == "effect_spearman":
                    table[me] = (float(v.mean()), 0.0, [])
                else:
                    per = sub[sub.method == me][m].dropna()
                    table[me] = (float(per.mean()), float(per.std() / np.sqrt(max(len(per), 1))), [])
                t1_stats.append({"line": n, "metric": m, "method": me, "value": table[me][0]})
            S.bars_with_points(a, table, methods, annotate=True)
            if i < len(LINES) - 1:
                a.set_xticklabels([])
            if j == 0:
                a.set_ylabel(f"{S.LINE_LABEL[n]}", fontweight="bold")
            if i == 0:
                a.set_title(METRIC_LABEL[m], fontsize=5.8)
    _csv(pd.DataFrame(t1_stats), out, "fig2_data_C")
    S.fig_letter(fig, "C", 0.08, 0.715)
    # D: PR curves (pooled over lines), E: effect-size confusion
    bot = GridSpecFromSubplotSpec(1, 4, outer[2], wspace=0.55, width_ratios=[1, 1, 1, 1])
    for k, n in enumerate(["k562", "hepg2"]):
        a = fig.add_subplot(bot[k])
        z = np.load(art / "t1" / f"pred_{n}_s0.npz")
        amended = art / "t1" / f"deamended_{n}_s0.npz"
        zp = np.load(amended) if amended.exists() else z
        truth = (z["padj___true"] < 0.05).ravel()
        for me in methods:
            if me == "NoChange" or f"padj__{me}" not in zp.files:
                continue
            sc = (-np.log10(np.clip(zp[f"padj__{me}"].astype(np.float64).ravel(), 1e-300, 1))
                  + 1e-9 * np.abs(z[me].astype(np.float64).ravel()))
            pr, rc, _ = precision_recall_curve(truth, sc)
            step = max(1, len(pr) // 400)
            a.plot(rc[::step], pr[::step], color=S.COLORS[me], lw=1.0 if me == "CASTER" else 0.8, label=me)
        a.axhline(truth.mean(), color=S.INK3, lw=0.5, ls=(0, (2, 2)))
        a.set_xlabel("Recall")
        a.set_ylabel("Precision")
        a.set_title(f"DEG recovery, {S.LINE_LABEL[n]}", loc="left")
        a.set_xlim(0, 1)
        a.set_ylim(0, None)
        S.hgrid(a)
        if k == 0:
            a.legend(fontsize=4.6, loc="upper right", handlelength=1.2)
    S.fig_letter(fig, "D", 0.08, 0.305)
    bins = [0, 1, 10, 50, 100, 200, 400, 800, 5000]
    blab = ["0", "1-9", "10-49", "50-99", "100-199", "200-399", "400-799", "800+"]
    for k, me in enumerate(["CASTER", "PerturbMean"]):
        a = fig.add_subplot(bot[2 + k])
        d = t1[t1.method == me]
        tb = np.digitize(d.n_de_true, bins) - 1
        pb = np.digitize(d.n_de_pred, bins) - 1
        cm = np.zeros((len(blab), len(blab)))
        for x_, y_ in zip(pb, tb, strict=True):
            cm[min(y_, len(blab) - 1), min(x_, len(blab) - 1)] += 1
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(cmn, cmap=S.SEQ_BLUE, vmin=0, vmax=1, origin="lower")
        a.set_xticks(range(len(blab)), blab, rotation=60, ha="right", rotation_mode="anchor", fontsize=4.5)
        a.set_yticks(range(len(blab)), blab, fontsize=4.5)
        a.set_xlabel(f"{me}-predicted # DEGs")
        if k == 0:
            a.set_ylabel("Observed # DEGs")
        rho = stats.spearmanr(d.n_de_pred, d.n_de_true).statistic
        a.set_title(f"{me}  rho = {rho:.2f}", loc="left", fontsize=5.5)
        _csv(pd.DataFrame(cm, index=blab, columns=blab).reset_index(), out, f"fig2_data_E_{me}")
    S.fig_letter(fig, "E", 0.50, 0.305)
    _save(fig, out, "Figure2")


# ============================================================================ Figure 3
def figure3(art: Path, rep: Path, out: Path) -> None:
    res = json.loads((rep / "results.json").read_text())
    fig = plt.figure(figsize=(S.FULL_W, 150 * S.MM))
    gs = GridSpec(2, 4, figure=fig, hspace=0.75, wspace=0.95, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    draw_task_schematic(ax, zero_shot=True)
    S.fig_letter(fig, "A", 0.07, 0.91)
    ft2 = pd.read_csv(rep / "tables" / "t2_fold_means.csv")
    methods = _methods(set(ft2.method))
    for k, m in enumerate(["pearson_delta", "discrimination", "de_auprc"]):
        a = fig.add_subplot(gs[0, 1 + k])
        g = ft2.groupby("method")[m]
        table = {me: (g.mean()[me], g.std()[me], ft2[ft2.method == me][m].tolist()) for me in methods}
        S.bars_with_points(a, table, methods, ylabel=METRIC_LABEL[m].replace("\n", " "))
        a.set_title("Zero-shot, 4 contexts", loc="left", fontsize=5.5, color=S.INK2)
    S.fig_letter(fig, "B", 0.30, 0.91)
    # C transfer matrices
    tm = pd.read_csv(rep / "tables" / "transfer_matrix.csv")
    for k, me in enumerate(["PerturbMean", "CASTER"]):
        a = fig.add_subplot(gs[1, k])
        mat = np.full((4, 4), np.nan)
        for _, r in tm[tm.method == me].iterrows():
            mat[LINES.index(r.source), LINES.index(r.target)] = r.pearson_delta
        lo_, hi_ = np.nanmin(tm.pearson_delta), np.nanmax(tm.pearson_delta)
        lo_ = lo_ - 0.15 * (hi_ - lo_ + 1e-9)
        im = a.imshow(mat, cmap=S.SEQ_BLUE, vmin=lo_, vmax=hi_)
        for i in range(4):
            for j in range(4):
                if np.isfinite(mat[i, j]):
                    a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5,
                           color="white" if (mat[i, j] - lo_) > 0.6 * (hi_ - lo_) else S.INK)
                elif i == j:
                    a.text(j, i, "-", ha="center", va="center", fontsize=5, color=S.INK3)
        if k == 1:
            cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(labelsize=5)
        lab = [S.LINE_LABEL[n] for n in LINES]
        a.set_xticks(range(4), lab, rotation=45, ha="right", rotation_mode="anchor")
        a.set_yticks(range(4), lab)
        a.set_xlabel("Target context")
        if k == 0:
            a.set_ylabel("Single source context")
        a.set_title(f"{me}: Pearson delta", loc="left")
    S.fig_letter(fig, "C", 0.07, 0.47)
    # D cancer-cancer vs RPE1 pairs
    a = fig.add_subplot(gs[1, 2])
    tr = pd.concat([pd.read_parquet(p) for p in sorted((art / "transfer").glob("metrics_*.parquet"))])
    tr["kind"] = np.where((tr.source == "rpe1") | (tr.target == "rpe1"), "with RPE1", "cancer-cancer")
    rows = []
    for i, me in enumerate(["PerturbMean", "CASTER"]):
        for j, kind in enumerate(["cancer-cancer", "with RPE1"]):
            v = tr[(tr.method == me) & (tr.kind == kind)].groupby("perturbation")["pearson_delta"].mean()
            x = i * 2.6 + j
            bp = a.boxplot([v.dropna()], positions=[x], widths=0.7, showfliers=False, patch_artist=True,
                           medianprops={"color": S.INK, "lw": 0.8}, boxprops={"lw": 0.4},
                           whiskerprops={"lw": 0.4}, capprops={"lw": 0})
            bp["boxes"][0].set_facecolor(S.COLORS[me] if kind == "cancer-cancer" else "#d9d8d3")
            rows.append({"method": me, "kind": kind, "median": float(v.median()), "n": int(v.size)})
    a.set_xticks([0, 1, 2.6, 3.6], ["cancer-\ncancer", "with\nRPE1"] * 2, fontsize=4.8)
    for x, me in ((0.5, "PerturbMean"), (3.1, "CASTER")):
        a.text(x, -0.22, me, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=5.5,
               color=S.COLORS[me], fontweight="bold")
    a.set_ylabel("Per-perturbation Pearson delta")
    h3 = res.get("H3", {}).get("PerturbMean", {})
    if h3:
        a.set_title(f"Transfer by pair type\nWilcoxon p = {h3['p_two_sided']:.1e}", loc="left", fontsize=5.5)
    S.hgrid(a)
    _csv(pd.DataFrame(rows), out, "fig3_data_D")
    S.fig_letter(fig, "D", 0.52, 0.47)
    # E per-line zero-shot vs under-represented gain
    a = fig.add_subplot(gs[1, 3])
    ft1 = pd.read_csv(rep / "tables" / "t1_fold_means.csv")
    ft1 = ft1[ft1.seed == 0]
    xs = np.arange(len(LINES))
    for off, (df, lab_, hatch) in enumerate([(ft2, "zero-shot", "////"), (ft1, "25% + 5% observed", "")]):
        c = df[df.method == "CASTER"].set_index("target").pearson_delta
        p = df[df.method == "PerturbMean"].set_index("target").pearson_delta
        gain = [(c[n] - p[n]) for n in LINES]
        a.bar(xs + (off - 0.5) * 0.38, gain, 0.36, color=S.COLORS["CASTER"], alpha=0.55 if off == 0 else 1,
              hatch=hatch, ec="white", lw=0.3, label=lab_)
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_xticks(xs, [S.LINE_LABEL[n] for n in LINES], rotation=45, ha="right", rotation_mode="anchor")
    a.set_ylabel("CASTER - PerturbMean\n(Pearson delta)")
    a.legend(fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, handlelength=1)
    S.hgrid(a)
    S.fig_letter(fig, "E", 0.73, 0.47)
    _save(fig, out, "Figure3")


# ============================================================================ Figure 4
def figure4(art: Path, rep: Path, out: Path) -> None:
    t1 = pd.read_parquet(rep / "tables" / "t1_per_perturbation.parquet")
    t1 = t1[t1.seed == 0]
    fig = plt.figure(figsize=(S.FULL_W, 185 * S.MM))
    gs = GridSpec(3, 6, figure=fig, hspace=0.75, wspace=0.75, height_ratios=[1.0, 1, 1])
    piv = t1.pivot_table(index=["target", "perturbation"], columns="method",
                         values=["de_overlap", "lfc_spearman", "pearson_delta"])
    # A/B: CASTER (y) vs each baseline (x), one point per held-out perturbation
    for row, (metric, letter) in enumerate([("de_overlap", "A"), ("lfc_spearman", "B")]):
        d = piv[metric].dropna(subset=["CASTER"])
        for k, base in enumerate(["ContextMean", "PerturbMean", "Linear"]):
            a = fig.add_subplot(gs[0, row * 3 + k])
            if base not in d:
                a.axis("off")
                continue
            dd = d[["CASTER", base]].dropna()
            diff = dd["CASTER"] - dd[base]
            a.scatter(dd[base], dd["CASTER"], c=np.clip(diff, -0.5, 0.5), cmap=S.DIVERGING.reversed(),
                      vmin=-0.5, vmax=0.5, s=1.2, lw=0, rasterized=True)
            lo = min(dd.min().min(), 0)
            a.plot([lo, 1], [lo, 1], color=S.INK3, lw=0.5, ls=(0, (2, 2)))
            a.set_xlim(lo, 1)
            a.set_ylim(lo, 1)
            win = (diff > 0).mean() * 100
            tie = (diff == 0).mean() * 100
            a.set_title(f"{win:.0f}% above" + (f", {tie:.0f}% tied" if tie >= 5 else ""),
                        fontsize=4.6, color=S.INK2, loc="left")
            a.set_xlabel(base, fontsize=5.5)
            if k == 0:
                a.set_ylabel(f"CASTER\n{METRIC_LABEL[metric].lower()}", fontsize=5.5)
                a.text(0, 1.22, METRIC_LABEL[metric].replace(chr(10), " "), transform=a.transAxes,
                       fontsize=6, fontweight="bold")
        S.fig_letter(fig, letter, 0.04 + row * 0.455, 0.905)
    # C: example perturbation with the largest CASTER gain over PerturbMean in a cancer line
    pd_ = piv["pearson_delta"].dropna()
    pd_ = pd_[pd_.index.get_level_values(0).isin(CANCER_LINES)]
    t1c = t1[(t1.method == "CASTER")].set_index(["target", "perturbation"])
    strong = t1c.loc[pd_.index, "n_de_true"] >= 30
    cand = (pd_["CASTER"] - pd_["PerturbMean"])[strong.to_numpy()]
    line, pert = cand.idxmax()
    z = np.load(art / "t1" / f"pred_{line}_s0.npz")
    i = list(z["perts"]).index(pert)
    true_l = z["lfc___true"][i].astype(np.float32)
    sig = z["padj___true"][i] < 0.05
    ex_rows = []
    for k, me in enumerate(["CASTER", "PerturbMean", "Linear"]):
        a = fig.add_subplot(gs[1, k * 2: k * 2 + 2])
        pl = z[me][i].astype(np.float32)  # predicted pseudobulk shift (log1p units; AMENDMENT-002)
        a.scatter(true_l[~sig], pl[~sig], s=1, color="#d9d8d3", lw=0, rasterized=True)
        a.scatter(true_l[sig], pl[sig], s=2, color=S.COLORS[me], lw=0, rasterized=True)
        lx = np.percentile(np.abs(true_l), 99.8)
        ly = np.percentile(np.abs(pl), 99.8)
        a.axhline(0, color=S.INK3, lw=0.4)
        a.axvline(0, color=S.INK3, lw=0.4)
        a.set_xlim(-lx, lx)
        a.set_ylim(-ly, ly)
        r = stats.pearsonr(true_l[sig], pl[sig])[0] if sig.sum() > 2 else np.nan
        a.text(0.04, 0.93, f"r(DEGs) = {r:.2f}\nn = {int(sig.sum())} DEGs", transform=a.transAxes, fontsize=5,
               va="top")
        a.set_xlabel("Observed log2 FC")
        a.set_ylabel(f"{me} predicted shift")
        a.set_title(f"{pert} knockdown in {S.LINE_LABEL[line]}", loc="left", fontsize=5.5)
        ex_rows.append(pd.DataFrame({"gene_idx": np.arange(true_l.size), "observed": true_l, "pred": pl,
                                     "sig": sig, "method": me}))
    _csv(pd.concat(ex_rows), out, "fig4_data_C")
    S.fig_letter(fig, "C", 0.07, 0.605)
    # D: cell-cycle error
    cc_path = rep / "tables" / "cell_cycle.csv"
    try:
        cc = pd.read_csv(cc_path)
    except pd.errors.EmptyDataError:
        cc = pd.DataFrame()
    if "phase" not in cc:
        cc = pd.DataFrame(columns=["phase", "method", "abs_error"])
    for k, ph in enumerate(["S", "G2M"]):
        a = fig.add_subplot(gs[2, k])
        if not (cc.phase == ph).any():
            a.text(0.5, 0.5, "no signature genes\nin gene space", ha="center", va="center",
                   transform=a.transAxes)
            continue
        for me in ["NoChange", "PerturbMean", "Linear", "CASTER"]:
            v = cc[(cc.phase == ph) & (cc.method == me)].abs_error
            if not len(v):
                continue
            a.hist(v, bins=40, range=(0, np.percentile(cc[cc.phase == ph].abs_error, 98)), color=S.COLORS[me],
                   histtype="step", lw=1.0, label=f"{me}  mu = {v.mean():.3f}")
        a.set_xlabel(f"|error|, {ph} score shift", fontsize=5.2)
        a.set_ylabel("Perturbations")
        a.legend(fontsize=4.0, loc="upper right", handlelength=0.8, frameon=False)
        a.set_title(f"{ph} phase", loc="left")
    S.fig_letter(fig, "D", 0.07, 0.30)
    # E: similarity heatmaps (one cancer line) + ARI curve
    n = "k562"
    z = np.load(art / "t1" / f"pred_{n}_s0.npz")
    d = t1[(t1.target == n) & (t1.method == "CASTER")].set_index("perturbation")
    strong = d.loc[z["perts"], "n_de_true"].to_numpy() >= 10
    from scipy.cluster.hierarchy import leaves_list, linkage
    xt = z["_true"][strong]
    zt = xt / np.linalg.norm(xt, axis=1, keepdims=True)
    order = leaves_list(linkage(zt, "average", metric="cosine"))
    for k, me in enumerate(["_true", "CASTER", "PerturbMean"]):
        a = fig.add_subplot(gs[2, 2 + k])
        x = z[me][strong][order]
        zz = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        a.imshow(zz @ zz.T, cmap=S.SEQ_BLUE, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("Observed" if me == "_true" else me, fontsize=5.5)
        if k == 0:
            a.set_ylabel(f"{S.LINE_LABEL[n]} perturbations\n(>=10 DEGs)")
    ari = pd.read_csv(rep / "tables" / "ari.csv")
    a = fig.add_subplot(gs[2, 5])
    for me in _methods(set(ari.method)):
        if me == "NoChange":
            continue
        v = ari[ari.method == me].groupby("k").ari.mean()
        a.plot(v.index, v.values, color=S.COLORS[me], lw=1, marker="o", ms=1.8, label=me)
    a.set_xlabel("Number of clusters")
    a.set_ylabel("ARI vs observed")
    a.legend(fontsize=4.4, handlelength=1, loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=2)
    S.hgrid(a)
    S.fig_letter(fig, "E", 0.37, 0.30)
    _save(fig, out, "Figure4")


# ============================================================================ Figure 5
def figure5(art: Path, rep: Path, out: Path) -> None:
    res = json.loads((rep / "results.json").read_text())
    fig = plt.figure(figsize=(S.FULL_W, 185 * S.MM))
    gs = GridSpec(3, 6, figure=fig, hspace=0.95, wspace=1.2)
    dep = pd.read_csv(rep / "tables" / "dependency_table.csv", index_col=0)
    obs_nde = pd.read_csv(rep / "tables" / "observed_n_de.csv", index_col=0)
    chronos = pd.read_csv(rep / "tables" / "chronos_cancer_lines.csv", index_col=0)
    h4 = pd.DataFrame(res["H4"]["tests"])
    for k, n in enumerate(CANCER_LINES):
        a = fig.add_subplot(gs[0, k * 2: k * 2 + 2])
        v = obs_nde[n].dropna()
        idx = v.index.intersection(chronos.index)
        x = chronos.loc[idx, n].dropna()  # Chronos is missing for some genes
        y = np.log1p(v.loc[x.index])
        a.hexbin(x, y, gridsize=35, cmap=S.SEQ_BLUE, mincnt=1, lw=0, bins="log", rasterized=True)
        # binned medians make the trend readable despite the many zero-response knockdowns
        edges = np.quantile(x, np.linspace(0, 1, 11))
        mids, meds = [], []
        for lo_, hi_ in zip(edges[:-1], edges[1:], strict=True):
            sel = (x >= lo_) & (x <= hi_)
            if sel.sum() > 10:
                mids.append(float(x[sel].median()))
                meds.append(float(y[sel].mean()))
        a.plot(mids, meds, color=S.COLORS["PerturbMean"], lw=1.2, marker="o", ms=2.5, zorder=5,
               label="decile mean")
        a.legend(fontsize=4.5, loc="upper right")
        r = h4[(h4.test == "H4") & (h4.line == n)].iloc[0]
        a.text(0.03, 0.95, f"rho = {r.rho:.2f}\nHolm p = {r.p_holm:.1e}\nn = {int(r.n)}", transform=a.transAxes,
               fontsize=5, va="top")
        a.set_xlabel("DepMap Chronos gene effect")
        a.set_ylabel("log(1 + # DEGs)")
        a.set_title(f"{S.LINE_LABEL[n]}: dependency vs response", loc="left")
        a.axvline(-1, color=S.INK3, lw=0.5, ls=(0, (2, 2)))
    S.fig_letter(fig, "A", 0.07, 0.905)
    # B context-specific
    for k, n in enumerate(CANCER_LINES):
        a = fig.add_subplot(gs[1, k * 2: k * 2 + 2])
        others = [o for o in CANCER_LINES if o != n]
        dn = dep[f"log1p_nde_{n}"] - dep[[f"log1p_nde_{o}" for o in others]].mean(1)
        dc = dep[f"chronos_{n}"] - dep[[f"chronos_{o}" for o in others]].mean(1)
        a.scatter(dc, dn, s=2, color=S.COLORS["CASTER"], alpha=0.5, lw=0, rasterized=True)
        top = (dc.abs() / dc.abs().max() + dn.abs() / dn.abs().max()).nlargest(3).index
        for g in top:
            a.annotate(g, (dc[g], dn[g]), fontsize=4.5, xytext=(4, -6), textcoords="offset points",
                       color=S.INK, arrowprops={"arrowstyle": "-", "lw": 0.3, "color": S.INK3})
        r = h4[(h4.test == "H4b") & (h4.line == n)].iloc[0]
        a.text(0.03, 0.04, f"rho = {r.rho:.2f}\nHolm p = {r.p_holm:.1e}", transform=a.transAxes, fontsize=5,
               va="bottom")
        a.axhline(0, color=S.INK3, lw=0.4)
        a.axvline(0, color=S.INK3, lw=0.4)
        a.set_xlabel(f"Chronos: {S.LINE_LABEL[n]} - other cancer lines")
        a.set_ylabel("Response: line - others")
        a.set_title("Context-specific dependency vs response", loc="left", fontsize=5.5)
    S.fig_letter(fig, "B", 0.07, 0.60)
    # C Chronos prediction from predicted transcriptomes
    h5 = pd.DataFrame(res["H5"])
    a = fig.add_subplot(gs[2, :2])
    meths = ["_true"] + [m for m in _methods(set(h5.method)) if m != "NoChange"]
    w = 0.8 / len(meths)
    for j, me in enumerate(meths):
        v = [h5[(h5.line == n) & (h5.method == me)].pearson.mean() for n in CANCER_LINES]
        a.bar(np.arange(3) + (j - (len(meths) - 1) / 2) * w, v, w * 0.92,
              color=S.INK if me == "_true" else S.COLORS[me], label="Observed" if me == "_true" else me)
    a.set_xticks(range(3), [S.LINE_LABEL[n] for n in CANCER_LINES])
    a.set_ylabel("Pearson r, predicted vs\nmeasured Chronos")
    a.legend(fontsize=4.3, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), handlelength=1,
             columnspacing=0.8)
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_title("Dependency from predicted transcriptome", loc="left", fontsize=5.5, pad=24)
    S.hgrid(a)
    S.fig_letter(fig, "C", 0.07, 0.30)
    # D gene programs
    pc = pd.read_csv(rep / "tables" / "programs.csv")
    ann = pd.read_csv(rep / "tables" / "program_annotation.csv")
    a = fig.add_subplot(gs[2, 2:4])
    n = "k562"
    d = pc[pc.line == n].pivot_table(index="program", columns="method", values="pearson")
    an = ann[ann.line == n].set_index("program")
    d = d.sort_values("CASTER", ascending=False)
    x = np.arange(len(d))
    for j, me in enumerate(["CASTER", "Linear", "PerturbMean", "ContextMean"]):
        if me in d:
            a.bar(x + (j - 1.5) * 0.2, d[me], 0.19, color=S.COLORS[me], label=me)
    labels = [(str(an.loc[p, "top_term"])[:24] + ("..." if len(str(an.loc[p, "top_term"])) > 24 else ""))
              if an.loc[p, "p"] < 1e-3 else f"program {p}" for p in d.index]
    a.set_xticks(x, labels, rotation=60, ha="right", rotation_mode="anchor", fontsize=4.2)
    a.set_ylabel("Program activity r\n(pred vs observed)")
    a.set_title(f"Gene programs, {S.LINE_LABEL[n]}", loc="left", fontsize=5.5, pad=16)
    a.legend(fontsize=4.3, handlelength=1, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02),
             columnspacing=0.6)
    S.hgrid(a)
    S.fig_letter(fig, "D", 0.37, 0.30)
    # E inverse design + F STRING
    inv = pd.read_csv(rep / "tables" / "inverse_design.csv")
    a = fig.add_subplot(gs[2, 4])
    for me in _methods(set(inv.method)):
        if me == "NoChange":
            continue
        v = inv[inv.method == me].groupby("k").accuracy.mean()
        a.plot(v.index, v.values, color=S.COLORS[me], lw=1, marker="o", ms=1.8, label=me)
    a.set_xscale("log")
    a.set_xlabel("k")
    a.set_ylabel("Top-k accuracy")
    a.legend(fontsize=4.2, handlelength=1, loc="upper left")
    a.set_title("Inverse design", loc="left", fontsize=5.5)
    S.hgrid(a)
    S.fig_letter(fig, "E", 0.655, 0.30)
    st = pd.read_csv(rep / "tables" / "string.csv")
    a = fig.add_subplot(gs[2, 5])
    meths = ["_true"] + [m for m in _methods(set(st.method)) if m not in ("NoChange", "ContextMean")]
    v = [st[st.method == me].auroc.mean() for me in meths]
    a.bar(range(len(meths)), v, color=[S.INK if m == "_true" else S.COLORS[m] for m in meths], width=0.7)
    a.axhline(0.5, color=S.INK3, lw=0.5, ls=(0, (2, 2)))
    a.set_xticks(range(len(meths)), ["Observed" if m == "_true" else m for m in meths], rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=4.8)
    a.set_ylim(min(0.45, min(v) - 0.02), max(v) + 0.02)
    a.set_ylabel("AUROC, STRING pairs")
    a.set_title("Complex recovery", loc="left", fontsize=5.5)
    S.hgrid(a)
    S.fig_letter(fig, "F", 0.82, 0.30)
    _save(fig, out, "Figure5")


# ============================================================================ supplementary
def figure_s_training(art: Path, out: Path) -> None:
    """Figure S2: training curves for every T1/T2 fold."""
    fig, axs = plt.subplots(2, 4, figsize=(S.FULL_W, 80 * S.MM), sharex=False)
    for r, task in enumerate(["t1", "t2"]):
        for c, n in enumerate(LINES):
            a = axs[r, c]
            p = art / task / f"history_{n}_s0.json"
            if not p.exists():
                a.axis("off")
                continue
            h = pd.DataFrame(json.loads(p.read_text())["history"])
            a.plot(h.epoch + 1, h.train, color=S.INK3, lw=0.8, label="train")
            a.plot(h.epoch + 1, h.val, color=S.COLORS["CASTER"], lw=1.0, label="validation")
            best = h.val.idxmin()
            a.scatter(h.epoch[best] + 1, h.val[best], s=8, color=S.COLORS["PerturbMean"], zorder=3,
                      label="selected")
            a.set_title(f"{'Under-represented' if task == 't1' else 'Zero-shot'}: {S.LINE_LABEL[n]}", loc="left",
                        fontsize=5.5)
            a.set_xlabel("Epoch")
            if c == 0:
                a.set_ylabel("Loss")
            S.hgrid(a)
            if r == 0 and c == 0:
                a.legend(fontsize=4.5)
    fig.tight_layout()
    _save(fig, out, "FigureS2")


def figure_s_metrics(rep: Path, out: Path) -> None:
    """Figure S3: secondary metrics per line (T1) and zero-shot per line (T2)."""
    ft = pd.read_csv(rep / "tables" / "t1_fold_means.csv")
    ft = ft[ft.seed == 0]
    ft2 = pd.read_csv(rep / "tables" / "t2_fold_means.csv")
    metrics = ["mae", "direction"]
    fig = plt.figure(figsize=(S.FULL_W, 120 * S.MM))
    gs = GridSpec(3, 4, figure=fig, hspace=0.9, wspace=0.55)
    for i, m in enumerate(metrics):
        for j, n in enumerate(LINES):
            a = fig.add_subplot(gs[i, j])
            d = ft[ft.target == n]
            meths = _methods(set(d.method))
            table = {me: (float(d[d.method == me][m].mean()), 0.0, []) for me in meths}
            S.bars_with_points(a, table, meths, annotate=m != "mae", best_line=m != "mae")
            a.set_title(f"{S.LINE_LABEL[n]}: {METRIC_LABEL[m].replace(chr(10), ' ')}", loc="left", fontsize=5.3)
    for j, n in enumerate(LINES):
        a = fig.add_subplot(gs[2, j])
        d = ft2[ft2.target == n]
        meths = _methods(set(d.method))
        table = {me: (float(d[d.method == me]["de_overlap"].mean()), 0.0, []) for me in meths}
        S.bars_with_points(a, table, meths)
        a.set_title(f"Zero-shot {S.LINE_LABEL[n]}: DE overlap", loc="left", fontsize=5.3)
    _save(fig, out, "FigureS3")


def figure_s_qc(art: Path, rep: Path, out: Path) -> None:
    """Figure S1: response-size distributions and cells per perturbation."""
    nde = pd.read_csv(rep / "tables" / "observed_n_de.csv", index_col=0)
    fig, axs = plt.subplots(1, 4, figsize=(S.FULL_W, 50 * S.MM))
    for n in LINES:
        v = nde[n].dropna()
        axs[0].hist(np.log10(v + 1), bins=40, histtype="step", lw=1, color=S.INK if n == "rpe1" else None,
                    label=S.LINE_LABEL[n])
    axs[0].set_xlabel("log10(1 + # DEGs)")
    axs[0].set_ylabel("Perturbations")
    axs[0].legend(fontsize=5)
    axs[0].set_title("Response size per context", loc="left")
    for n in LINES:
        z = np.load(art / "observed" / f"observed_{n}.npz")
        axs[1].hist(z["n_cells"], bins=np.arange(14, 34), histtype="step", lw=1, label=S.LINE_LABEL[n])
    axs[1].set_xlabel("Cells per perturbation (after subsampling)")
    axs[1].set_title("Cells per perturbation", loc="left")
    sim = pd.read_csv(rep / "tables" / "cross_context_similarity.csv")
    for kind, c in [("cancer-cancer", "#2a78d6"), ("cancer-RPE1", "#8a8984")]:
        v = sim[sim.pair_type == kind].jaccard
        axs[2].hist(v, bins=30, color=c, alpha=0.6, label=f"{kind} (median {v.median():.2f})")
    axs[2].set_xlabel("Jaccard index of DEG sets")
    axs[2].legend(fontsize=5)
    axs[2].set_title("DEG overlap between contexts", loc="left")
    st = rep / "tables" / "similarity_by_strength.csv"
    if st.exists():
        d = pd.read_csv(st)
        for kind, c in [("cancer-cancer", "#2a78d6"), ("cancer-RPE1", "#8a8984")]:
            g = d[d.pair_type == kind]
            axs[3].plot(g.min_de, g.median_pearson, color=c, marker="o", ms=2.5, lw=1, label=kind)
        axs[3].set_xscale("log")
        axs[3].set_xlabel("Minimum # DEGs in both contexts")
        axs[3].set_ylabel("Median response similarity (r)")
        axs[3].legend(fontsize=5)
        axs[3].set_title("Similarity vs response strength", loc="left")
    for a in axs:
        S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS1")


def figure_s_positive_control(rep: Path, out: Path) -> None:
    """Figure S4: synthetic positive control (learnable vs unlearnable context)."""
    p = rep / "tables" / "positive_control.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axs = plt.subplots(1, 2, figsize=(HALF := S.FULL_W * 0.7, 55 * S.MM))
    _ = HALF
    for a, scen in zip(axs, ["unlearnable", "learnable"], strict=True):
        sub = d[d.scenario == scen].set_index("method").reindex([m for m in S.METHODS if m in set(d.method)])
        a.bar(range(len(sub)), sub.pearson_delta, color=[S.COLORS[m] for m in sub.index], width=0.7)
        a.set_xticks(range(len(sub)), sub.index, rotation=45, ha="right", rotation_mode="anchor")
        a.set_ylabel("Pearson delta")
        a.set_title({"unlearnable": "Context effect = random noise",
                     "learnable": "Context effect = f(baseline expression)"}[scen], loc="left", fontsize=5.5)
        S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS4")


def figure_s_h5(rep: Path, out: Path) -> None:
    """Figure S5: measured vs predicted Chronos per cancer line (observed and CASTER inputs)."""
    fig, axs = plt.subplots(2, 3, figsize=(S.FULL_W, 95 * S.MM))
    for j, n in enumerate(CANCER_LINES):
        d = pd.read_csv(rep / "tables" / f"H5_predictions_{n}.csv", index_col=0)
        for i, me in enumerate(["_true", "CASTER"]):
            a = axs[i, j]
            a.scatter(d["measured"], d[me], s=2, color=S.INK if me == "_true" else S.COLORS["CASTER"], lw=0,
                      alpha=0.6, rasterized=True)
            r = stats.pearsonr(d["measured"], d[me])[0]
            a.set_title(f"{S.LINE_LABEL[n]}, input: {'observed' if me == '_true' else 'CASTER'}  r = {r:.2f}",
                        loc="left", fontsize=5.3)
            a.set_xlabel("Measured Chronos")
            a.set_ylabel("Predicted Chronos")
            S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS5")


def figure_s_transfer_all(rep: Path, out: Path) -> None:
    """Figure S6: transfer matrices for DE AUPRC and discrimination."""
    tm = pd.read_csv(rep / "tables" / "transfer_matrix.csv")
    fig, axs = plt.subplots(2, 2, figsize=(S.FULL_W * 0.75, 110 * S.MM))
    for i, m in enumerate(["de_auprc", "discrimination"]):
        for j, me in enumerate(["PerturbMean", "CASTER"]):
            a = axs[i, j]
            mat = np.full((4, 4), np.nan)
            for _, r in tm[tm.method == me].iterrows():
                mat[LINES.index(r.source), LINES.index(r.target)] = r[m]
            vmax = np.nanmax(tm[m])
            a.imshow(mat, cmap=S.SEQ_BLUE, vmin=0, vmax=vmax)
            for x in range(4):
                for y in range(4):
                    if np.isfinite(mat[y, x]):
                        a.text(x, y, f"{mat[y, x]:.2f}", ha="center", va="center", fontsize=5,
                               color="white" if mat[y, x] > 0.6 * vmax else S.INK)
            lab = [S.LINE_LABEL[n] for n in LINES]
            a.set_xticks(range(4), lab, rotation=45, ha="right", rotation_mode="anchor")
            a.set_yticks(range(4), lab)
            a.set_title(f"{me}: {METRIC_LABEL[m].replace(chr(10), ' ')}", loc="left", fontsize=5.5)
    fig.tight_layout()
    _save(fig, out, "FigureS6")


def graphical_abstract(out: Path) -> None:
    """Graphical abstract: contexts -> CASTER -> evaluation and dependency validation."""
    fig = plt.figure(figsize=(S.FULL_W * 0.8, 105 * S.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.05], hspace=0.15)
    draw_context_schematic(fig.add_subplot(gs[0]))
    ax = fig.add_subplot(gs[1])
    draw_caster_schematic(ax)
    fig.text(0.5, 0.005, "PertBench: discrimination | Pearson delta | DE AUPRC | DE overlap | fold change | "
             "effect size  +  DepMap dependency validation", ha="center", fontsize=5.5, color=S.INK2)
    _save(fig, out, "GraphicalAbstract")


def render_all(art: Path, rep: Path) -> None:
    """Render every figure."""
    S.setup()
    out = rep / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure1(art, rep, out)
    figure2(art, rep, out)
    figure3(art, rep, out)
    figure4(art, rep, out)
    figure5(art, rep, out)
    figure_s_qc(art, rep, out)
    figure_s_training(art, out)
    figure_s_metrics(rep, out)
    figure_s_positive_control(rep, out)
    figure_s_h5(rep, out)
    figure_s_transfer_all(rep, out)
    graphical_abstract(out)
    from canpert.figures_supp import render_supp

    render_supp(art, rep)
    from canpert.figures_review import render_review

    render_review(art, rep)
    _ = (Line2D,)
