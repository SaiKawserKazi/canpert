"""Review-round figures: Figure 6 (calibration, noise ceiling, controls, seeds) and S13-S15."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from canpert import figstyle as S
from canpert.data import CANCER_LINES, LINES
from canpert.figures import _csv, _save

CAL_LABEL = {"pearson_deltactrl": "Pearson delta\n(vs control)", "pearson_deltapert_top100": "Pearson delta\n(top 100, vs pert. mean)",
             "mse": "MSE", "mae": "MAE", "wmse": "WMSE", "wr2_deltactrl": "Weighted R2\n(vs control)",
             "wr2_deltapert": "Weighted R2\n(vs pert. mean)", "nir": "NIR", "pds": "PDS"}
CAL_ORDER = ["pearson_deltactrl", "pearson_deltapert_top100", "wr2_deltactrl", "wr2_deltapert", "wmse", "mse",
             "mae", "nir", "pds"]


def figure6(art: Path, rep: Path, out: Path) -> None:
    res = json.loads((rep / "results.json").read_text())
    rv = res.get("review", {})
    tab = rep / "tables"
    fig = plt.figure(figsize=(S.FULL_W, 240 * S.MM))
    gs = GridSpec(4, 6, figure=fig, hspace=1.0, wspace=1.1, height_ratios=[1, 1.05, 1, 1])
    # A: noise ceiling vs achieved Pearson delta (T1, all seeds)
    a = fig.add_subplot(gs[0, :3])
    ceil = pd.read_csv(tab / "noise_ceiling.csv")
    ceil = ceil[ceil.task == "t1"].groupby("target").ceiling.median()
    t1 = pd.read_parquet(tab / "t1_per_perturbation.parquet")
    fm = t1.groupby(["target", "method"]).pearson_delta.mean().unstack()
    meths = [m for m in ("CASTER", "Linear", "LineOffset", "PerturbMean", "ContextMean") if m in fm]
    x = np.arange(len(LINES))
    w = 0.8 / (len(meths) + 1)
    a.bar(x - 0.4 + w / 2, [ceil.get(n, np.nan) for n in LINES], w * 0.92, color="#ffffff", ec=S.INK, lw=0.6,
          hatch="////", label="Noise ceiling")
    for j, me in enumerate(meths):
        a.bar(x - 0.4 + w * (j + 1.5), [fm.loc[n, me] for n in LINES], w * 0.92, color=S.COLORS[me], label=me)
    a.set_xticks(x, [S.LINE_LABEL[n] for n in LINES])
    a.set_ylabel("Pearson delta")
    a.set_title("Achieved accuracy vs split-half noise ceiling", loc="left", fontsize=5.8)
    a.legend(fontsize=4.4, ncol=3, loc="upper left", handlelength=1)
    S.hgrid(a)
    _csv(pd.DataFrame({"line": LINES, "ceiling": [ceil.get(n, np.nan) for n in LINES]}), out, "fig6_data_A")
    S.fig_letter(fig, "A", 0.06, 0.91)
    # B: DRF heatmap (metric x line), T1
    a = fig.add_subplot(gs[0, 3:])
    drf = pd.read_csv(tab / "drf.csv")
    d = drf[drf.task == "t1"].pivot_table(index="metric", columns="target", values="drf").reindex(CAL_ORDER)[list(LINES)]
    im = a.imshow(d.to_numpy(), cmap=S.SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iloc[i, j]
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.6, color="white" if v > 0.6 else S.INK)
    short = {"pearson_deltactrl": "Pearson delta", "pearson_deltapert_top100": "Pearson d, top 100 (vs PM)",
             "wr2_deltactrl": "Weighted R2", "wr2_deltapert": "Weighted R2 (vs PM)", "wmse": "WMSE", "mse": "MSE",
             "mae": "MAE", "nir": "NIR", "pds": "PDS"}
    a.set_yticks(range(len(d)), [short[m] for m in d.index], fontsize=4.6)
    a.set_xticks(range(4), [S.LINE_LABEL[n] for n in LINES])
    a.set_title("Dynamic range fraction (DRF) per metric", loc="left", fontsize=5.8)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=4.5)
    S.fig_letter(fig, "B", 0.50, 0.91)
    # C: calibrated metrics, normalised between negative control (0) and technical duplicate (1)
    a = fig.add_subplot(gs[1, :4])
    nm = pd.read_csv(tab / "calibrated_normalised.csv")
    nm = nm[nm.task == "t1"]
    show = ["InterpDup", "CASTER", "Linear", "LineOffset", "PerturbMean", "ContextMean"]
    cal_ok = nm.groupby("metric").calibrated.mean()
    mets = [m for m in CAL_ORDER if cal_ok.get(m, 0) >= 0.75]
    dropped = [m for m in CAL_ORDER if m not in mets]
    w = 0.8 / len(show)
    rows = []
    for j, me in enumerate(show):
        v = [nm[(nm.method == me) & (nm.metric == m)].fraction_of_range.mean() for m in mets]
        rows += [{"method": me, "metric": m, "value": vv} for m, vv in zip(mets, v, strict=True)]
        col = S.INK if me == "InterpDup" else S.COLORS.get(me, S.INK3)
        a.bar(np.arange(len(mets)) - 0.4 + w * (j + 0.5), v, w * 0.92, color=col, label=me)
    a.axhline(1, color=S.INK, lw=0.6, ls=(0, (2, 2)))
    a.axhline(0, color=S.INK2, lw=0.5)
    a.text(len(mets) - 0.5, 1.02, "technical duplicate", fontsize=4.5, ha="right", va="bottom")
    a.set_xticks(range(len(mets)), [CAL_LABEL[m] for m in mets], fontsize=4.6)
    a.set_ylabel("Score, scaled: negative control = 0,\ntechnical duplicate = 1")
    a.set_title("Calibrated metrics, under-represented task (mean over lines and seeds)", loc="left", fontsize=5.8)
    if dropped:
        a.text(0.0, -0.30, "Not shown (technical duplicate not better than negative control): "
               + ", ".join(CAL_LABEL[m].replace("\n", " ") for m in dropped), transform=a.transAxes, fontsize=4.4,
               color=S.INK2)
    a.set_ylim(top=1.55)
    a.legend(fontsize=4.4, ncol=3, loc="upper left", handlelength=1, frameon=False)
    S.hgrid(a)
    _csv(pd.DataFrame(rows), out, "fig6_data_C")
    S.fig_letter(fig, "C", 0.06, 0.70)
    # D: seeds — Pearson delta per line and seed
    a = fig.add_subplot(gs[1, 4:])
    sm = t1.groupby(["seed", "target", "method"]).pearson_delta.mean().reset_index()
    for k, me in enumerate(("CASTER", "Linear", "PerturbMean")):
        for i, n in enumerate(LINES):
            v = sm[(sm.method == me) & (sm.target == n)].pearson_delta
            a.scatter(np.full(len(v), i + (k - 1) * 0.22), v, s=7, color=S.COLORS[me], lw=0,
                      label=me if i == 0 else None, zorder=3)
            a.plot([i + (k - 1) * 0.22 - 0.08, i + (k - 1) * 0.22 + 0.08], [v.mean()] * 2, color=S.INK, lw=0.6)
    a.set_xticks(range(4), [S.LINE_LABEL[n] for n in LINES])
    a.set_ylabel("Pearson delta")
    a.set_title(f"Seeds / splits (n = {sm.seed.nunique()})", loc="left", fontsize=5.8)
    a.legend(fontsize=4.4, loc="upper left", handlelength=1)
    S.hgrid(a)
    S.fig_letter(fig, "D", 0.66, 0.70)
    # E: controls — CASTER minus control (Pearson delta) per line
    a = fig.add_subplot(gs[2, :3])
    ctr = rv.get("controls", {})
    comps = [("LineOffset", "vs LineOffset\n(PerturbMean + line offset)"),
             ("idonly", "vs line-identity-only\n(no control cells)"),
             ("wrongctrl", "vs wrong-line\ncontrol cells")]
    rows = []
    for j, (k, lab) in enumerate(comps):
        if k == "LineOffset":
            pv = t1.pivot_table(index=["target", "seed", "perturbation"], columns="method", values="pearson_delta")
            vals = [(pv.CASTER - pv.LineOffset).xs(n, level=0).median() if "LineOffset" in pv else np.nan for n in LINES]
        else:
            per = ctr.get(f"{k}_per_line", {})
            vals = [per.get(n, {}).get("pearson_delta", {}).get("median_diff", np.nan) for n in LINES]
        rows += [{"comparison": k, "line": n, "median_diff": v} for n, v in zip(LINES, vals, strict=True)]
        a.bar(np.arange(4) + (j - 1) * 0.27, vals, 0.25, color=["#2a78d6", "#86b6ef", "#cde2fb"][j], label=lab,
              ec=S.INK3, lw=0.3)
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_xticks(range(4), [S.LINE_LABEL[n] for n in LINES])
    a.set_ylabel("Median per-perturbation gain of\nCASTER (Pearson delta)")
    a.set_title("Does CASTER learn more than one shift per line?", loc="left", fontsize=5.8)
    a.legend(fontsize=4.3, loc="upper left", handlelength=1)
    S.hgrid(a)
    _csv(pd.DataFrame(rows), out, "fig6_data_E")
    S.fig_letter(fig, "E", 0.06, 0.48)
    # F: zero-shot calibrated (fraction of range), CASTER vs PerturbMean
    a = fig.add_subplot(gs[2, 3:])
    nm2 = pd.read_csv(tab / "calibrated_normalised.csv")
    nm2 = nm2[nm2.task == "t2"]
    if len(nm2):
        w = 0.8 / 3
        for j, me in enumerate(("InterpDup", "CASTER", "PerturbMean")):
            v = [nm2[(nm2.method == me) & (nm2.metric == m)].fraction_of_range.mean() for m in mets]
            a.bar(np.arange(len(mets)) - 0.4 + w * (j + 0.5), v, w * 0.92,
                  color=S.INK if me == "InterpDup" else S.COLORS[me], label=me)
        a.axhline(1, color=S.INK, lw=0.6, ls=(0, (2, 2)))
        a.axhline(0, color=S.INK2, lw=0.5)
        a.set_xticks(range(len(mets)), [CAL_LABEL[m] for m in mets], fontsize=4.4, rotation=45, ha="right",
                     rotation_mode="anchor")
        a.set_ylabel("Scaled score")
        a.set_title("Unseen-line task, calibrated metrics", loc="left", fontsize=5.8)
        a.set_ylim(top=1.3)
        a.legend(fontsize=4.3, loc="upper right", ncol=3, handlelength=1, frameon=False)
        S.hgrid(a)
    S.fig_letter(fig, "F", 0.50, 0.48)
    # G: noise-free rendering � each method re-scored without the sampling noise of its input control cells
    a = fig.add_subplot(gs[3, :4])
    cl = t1[t1.method.str.endswith("_clean")]
    if len(cl):
        fmc = cl.groupby(["target", "method"]).pearson_delta.mean().unstack()
        mm = [m for m in ("CASTER", "Linear", "LineOffset", "PerturbMean", "ContextMean") if f"{m}_clean" in fmc]
        w = 0.8 / (len(mm) + 1)
        a.bar(x - 0.4 + w / 2, [ceil.get(n, np.nan) for n in LINES], w * 0.92, color="#ffffff", ec=S.INK, lw=0.6,
              hatch="////", label="Noise ceiling")
        rows = []
        for j, me in enumerate(mm):
            vals = [fmc.loc[n, f"{me}_clean"] for n in LINES]
            raw = [fm.loc[n, me] for n in LINES]
            a.bar(x - 0.4 + w * (j + 1.5), vals, w * 0.92, color=S.COLORS[me], label=me)
            a.scatter(x - 0.4 + w * (j + 1.5), raw, s=6, color=S.INK, zorder=3, marker="_", lw=0.8)
            rows += [{"line": n, "method": me, "noise_free": v_, "as_rendered": r_} for n, v_, r_ in zip(LINES, vals, raw, strict=True)]
        a.set_xticks(x, [S.LINE_LABEL[n] for n in LINES])
        a.set_ylabel("Pearson delta")
        a.set_title("Noise-free rendering (bars) vs rendering on 32 sampled control cells (ticks)", loc="left",
                    fontsize=5.8)
        a.legend(fontsize=4.4, ncol=3, loc="upper left", handlelength=1)
        S.hgrid(a)
        _csv(pd.DataFrame(rows), out, "fig6_data_G")
    S.fig_letter(fig, "G", 0.06, 0.255)
    _save(fig, out, "Figure6")


def figure_s13_p53(rep: Path, out: Path) -> None:
    tab = rep / "tables"
    p53 = pd.read_csv(tab / "p53_scores.csv")
    ex = json.loads((tab / "review_extra.json").read_text())["p53"]
    sim = pd.read_csv(tab / "cross_context_similarity.csv")
    fig, axs = plt.subplots(1, 4, figsize=(S.FULL_W, 52 * S.MM))
    a = axs[0]
    fr = [ex["per_line"][n]["frac_induced"] * 100 for n in LINES]
    a.bar(range(4), fr, color=["#86b6ef" if "wild" in ex["per_line"][n]["tp53"] else "#d9d8d3" for n in LINES])
    a.set_xticks(range(4), [f"{S.LINE_LABEL[n]}\n{ex['per_line'][n]['tp53'].replace('TP53 ', '')}" for n in LINES],
                 fontsize=4.6)
    a.set_ylabel("% knockdowns inducing\np53 targets (FDR < 0.05)")
    a.set_title("p53 activation", loc="left", fontsize=5.5)
    S.hgrid(a)
    a = axs[1]
    mat = np.eye(4)
    for k, v in ex["pair_spearman_p53_shift"].items():
        i, j = (LINES.index(x) for x in k.split("-"))
        mat[i, j] = mat[j, i] = v
    a.imshow(mat, cmap=S.SEQ_BLUE, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=4.6, color="white" if mat[i, j] > 0.6 else S.INK)
    a.set_xticks(range(4), [S.LINE_LABEL[n] for n in LINES], rotation=45, ha="right", rotation_mode="anchor", fontsize=4.6)
    a.set_yticks(range(4), [S.LINE_LABEL[n] for n in LINES], fontsize=4.6)
    a.set_title("p53 score shift, rank corr.", loc="left", fontsize=5.5)
    a = axs[2]
    ind = p53.pivot_table(index="perturbation", columns="line", values="induced")
    hr = sim[(sim.a == "hepg2") & (sim.b == "rpe1")].set_index("perturbation")
    both = ind.reindex(hr.index)[["hepg2", "rpe1"]].fillna(False).astype(bool).all(1)
    a.boxplot([hr.pearson[both], hr.pearson[~both]], widths=0.55, showfliers=False,
              medianprops={"color": S.INK}, boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5}, capprops={"lw": 0})
    a.set_xticks([1, 2], ["p53 induced\nin both", "other"], fontsize=4.8)
    a.set_ylabel("HepG2-RPE1 response similarity (r)")
    a.set_title(f"p = {ex['hepg2_rpe1_by_p53']['mannwhitney_p']:.1e}", loc="left", fontsize=5.5)
    S.hgrid(a)
    a = axs[3]
    ne = ex["similarity_p53_neither"]
    keys = sorted(ne, key=lambda k: -ne[k]["median_r"])
    a.barh(range(len(keys)), [ne[k]["median_r"] for k in keys], color=["#2a78d6" if k == "hepg2-rpe1" else "#d9d8d3" for k in keys])
    a.set_yticks(range(len(keys)), ["-".join(S.LINE_LABEL[x] for x in k.split("-")) for k in keys], fontsize=4.6)
    a.invert_yaxis()
    a.set_xlabel("Median response similarity (r)")
    a.set_title("Knockdowns without p53 induction", loc="left", fontsize=5.5)
    fig.tight_layout()
    _save(fig, out, "FigureS13")


def figure_s14_depmap(rep: Path, out: Path) -> None:
    ex = json.loads((rep / "tables" / "review_extra.json").read_text())["depmap_controls"]
    cr = pd.DataFrame(ex["cross"]).pivot_table(index="response_line", columns="chronos_line", values="rho")
    cr = cr.reindex(index=list(CANCER_LINES), columns=list(CANCER_LINES))
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL_W * 0.7, 55 * S.MM))
    a = axs[0]
    a.imshow(-cr.to_numpy(), cmap=S.SEQ_BLUE, vmin=0, vmax=0.6)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5,
                   color="white" if -cr.iloc[i, j] > 0.4 else S.INK, fontweight="bold" if i == j else "normal")
    a.set_xticks(range(3), [S.LINE_LABEL[n] for n in CANCER_LINES])
    a.set_yticks(range(3), [S.LINE_LABEL[n] for n in CANCER_LINES])
    a.set_xlabel("Chronos gene effect of line")
    a.set_ylabel("Response size in line")
    a.set_title("Spearman rho, response vs dependency", loc="left", fontsize=5.5)
    a = axs[1]
    g = pd.DataFrame(ex["generality"])
    a.bar(range(3), g.pearson_other_lines_mean_vs_own, color="#86b6ef")
    a.set_xticks(range(3), [S.LINE_LABEL[n] for n in g.line])
    a.set_ylim(0, 1)
    a.set_ylabel("Pearson r, own Chronos vs\nmean of the other two lines")
    a.set_title("Dependency is largely shared", loc="left", fontsize=5.5)
    S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS14")


def figure_s15_ribosome(rep: Path, out: Path) -> None:
    tab = rep / "tables"
    rg = pd.read_csv(tab / "ribosome_gain.csv")
    res = json.loads((rep / "results.json").read_text()).get("review", {})
    fig, axs = plt.subplots(1, 3, figsize=(S.FULL_W, 55 * S.MM))
    a = axs[0]
    d0 = rg[rg.seed == 0]
    data, labels = [], []
    for n in LINES:
        g = d0[d0.target == n]
        data += [g[g.ribosome_any].gain, g[~g.ribosome_any].gain]
        labels += [f"{S.LINE_LABEL[n]}\nribo", "other"]
    bp = a.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, medianprops={"color": S.INK},
                   boxprops={"lw": 0.4}, whiskerprops={"lw": 0.4}, capprops={"lw": 0})
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("#2a78d6" if k % 2 == 0 else "#d9d8d3")
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_xticks(range(1, len(labels) + 1), labels, fontsize=4.2)
    a.set_ylabel("CASTER - PerturbMean (Pearson delta)")
    a.set_title("Random split", loc="left", fontsize=5.5)
    S.hgrid(a)
    a = axs[1]
    gr = res.get("grouped", {}).get("ribosome_gain_grouped", {})
    if gr:
        x = np.arange(4)
        a.bar(x - 0.2, [gr.get(n, {}).get("median_gain_ribosome", np.nan) for n in LINES], 0.38, color="#2a78d6",
              label="ribosome genes")
        a.bar(x + 0.2, [gr.get(n, {}).get("median_gain_other", np.nan) for n in LINES], 0.38, color="#d9d8d3",
              label="other")
        a.axhline(0, color=S.INK2, lw=0.5)
        a.set_xticks(x, [S.LINE_LABEL[n] for n in LINES])
        a.set_ylabel("Median gain (Pearson delta)")
        a.legend(fontsize=4.5)
    a.set_title("Complex-grouped split (no sibling in training)", loc="left", fontsize=5.5)
    S.hgrid(a)
    a = axs[2]
    rb = d0[d0.ribosome_any]
    for n in LINES:
        g = rb[rb.target == n]
        a.scatter(g.siblings_in_train, g.gain, s=4, lw=0, alpha=0.7, label=S.LINE_LABEL[n])
    a.axhline(0, color=S.INK2, lw=0.5)
    a.set_xlabel("Same-complex genes in training")
    a.set_ylabel("Gain (Pearson delta)")
    a.set_title("Ribosome knockdowns, random split", loc="left", fontsize=5.5)
    a.legend(fontsize=4.3)
    S.hgrid(a)
    fig.tight_layout()
    _save(fig, out, "FigureS15")


def render_review(art: Path, rep: Path) -> None:
    S.setup()
    out = rep / "figures"
    for fn in (lambda: figure6(art, rep, out), lambda: figure_s13_p53(rep, out),
               lambda: figure_s14_depmap(rep, out), lambda: figure_s15_ribosome(rep, out)):
        fn()
