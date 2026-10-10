"""Review round aggregation (V.1): adds review results to results.json and writes tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from canpert import analysis as A
from canpert.data import CANCER_LINES, LINES

PRIM = ("pearson_delta", "discrimination", "de_auprc")
DE_AFFECTED = ("de_auprc", "de_overlap", "lfc_spearman", "direction")


def _paired(df: pd.DataFrame, a: str, b: str, metric: str, keys=("target", "seed", "perturbation")) -> dict:
    x = df[df.method == a].set_index(list(keys))[metric]
    y = df[df.method == b].set_index(list(keys))[metric]
    j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 5:
        return {"n": len(j)}
    d = j.a - j.b
    if metric in A.LOWER_IS_BETTER:
        d = -d
    w = stats.wilcoxon(d, zero_method="zsplit")
    return {"n": int(len(j)), "mean_a": float(j.a.mean()), "mean_b": float(j.b.mean()),
            "median_diff": float(d.median()), "frac_a_better": float((d > 0).mean()), "p_two_sided": float(w.pvalue)}


def seeds_summary(t1: pd.DataFrame) -> dict:
    """Per seed x line means, mean +- SD over seeds, cancer-only averages, per-seed paired tests."""
    fm = t1.groupby(["seed", "target", "method"])[list(PRIM) + ["mae"]].mean().reset_index()
    per_line = fm.groupby(["target", "method"])[list(PRIM) + ["mae"]].agg(["mean", "std"]).reset_index()
    per_line.columns = ["_".join(c).strip("_") for c in per_line.columns]
    allc = fm.groupby(["seed", "method"])[list(PRIM)].mean().reset_index()
    canc = fm[fm.target.isin(CANCER_LINES)].groupby(["seed", "method"])[list(PRIM)].mean().reset_index()
    agg_all = allc.groupby("method")[list(PRIM)].agg(["mean", "std"])
    agg_canc = canc.groupby("method")[list(PRIM)].agg(["mean", "std"])
    tests = []
    for seed, g in t1.groupby("seed"):
        for b in ("PerturbMean", "ContextMean", "Linear", "LineOffset"):
            if b not in set(g.method):
                continue
            for m in PRIM:
                r = _paired(g, "CASTER", b, m)
                r.update(seed=int(seed), baseline=b, metric=m)
                tests.append(r)
    tdf = pd.DataFrame(tests)
    return {"seeds": sorted(int(s) for s in t1.seed.unique()),
            "per_line": per_line.to_dict("records"),
            "all_lines": {m: {k: {"mean": float(agg_all.loc[m, (k, "mean")]), "sd": float(agg_all.loc[m, (k, "std")])}
                              for k in PRIM} for m in agg_all.index},
            "cancer_only": {m: {k: {"mean": float(agg_canc.loc[m, (k, "mean")]), "sd": float(agg_canc.loc[m, (k, "std")])}
                                for k in PRIM} for m in agg_canc.index},
            "per_seed_tests": tdf.to_dict("records")}


def controls_summary(art: Path, t1: pd.DataFrame) -> dict:
    """LineOffset, line-identity-only CASTER and wrong-line control cells."""
    out: dict = {}
    for b in ("LineOffset",):
        out[b] = {m: _paired(t1, "CASTER", b, m) for m in PRIM}
    wc = t1[t1.method.isin(["CASTER", "CASTER_wrongctrl"])]
    if "CASTER_wrongctrl" in set(wc.method):
        out["wrongctrl"] = {m: _paired(wc, "CASTER", "CASTER_wrongctrl", m) for m in PRIM}
        out["wrongctrl_per_line"] = {n: {m: _paired(wc[wc.target == n], "CASTER", "CASTER_wrongctrl", m) for m in PRIM}
                                     for n in LINES}
    mc = A.load_metrics(art / "t1_meanctrl")
    if len(mc):
        mc = mc[mc.method == "CASTER"].assign(method="CASTER_idonly")
        both = pd.concat([t1[(t1.seed == 0) & (t1.method == "CASTER")], mc], ignore_index=True)
        out["idonly"] = {m: _paired(both, "CASTER", "CASTER_idonly", m) for m in PRIM}
        out["idonly_per_line"] = {n: {m: _paired(both[both.target == n], "CASTER", "CASTER_idonly", m) for m in PRIM}
                                  for n in LINES}
        po = pd.concat([mc, t1[(t1.seed == 0) & (t1.method.isin(["PerturbMean", "LineOffset", "Linear"]))]])
        out["idonly_vs"] = {b: {m: _paired(po, "CASTER_idonly", b, m) for m in PRIM} for b in ("PerturbMean", "LineOffset", "Linear")}
    return out


def grouped_summary(art: Path, rep: Path) -> dict:
    g = A.load_metrics(art / "t1_grouped")
    if not len(g):
        return {}
    out = {"means": g.groupby(["target", "method"])[list(PRIM)].mean().reset_index().to_dict("records"),
           "tests": {b: {m: _paired(g, "CASTER", b, m) for m in PRIM} for b in ("PerturbMean", "Linear", "LineOffset")
                     if b in set(g.method)}}
    rg = pd.read_csv(rep / "tables" / "ribosome_gain.csv") if (rep / "tables" / "ribosome_gain.csv").exists() else None
    pv = g.pivot_table(index=["target", "perturbation"], columns="method", values="pearson_delta")
    pv["gain"] = pv.CASTER - pv.PerturbMean
    rib = {p for p in pv.index.get_level_values(1) if p.startswith(("RPL", "RPS"))}
    if rg is not None:
        rib |= set(rg[rg.ribosome_any].perturbation)
    pv = pv.reset_index()
    pv["ribosome_any"] = pv.perturbation.isin(rib)
    out["ribosome_gain_grouped"] = {n: {"median_gain_ribosome": float(sub[sub.ribosome_any].gain.median()),
                                        "median_gain_other": float(sub[~sub.ribosome_any].gain.median()),
                                        "n_ribosome_test": int(sub.ribosome_any.sum())}
                                    for n, sub in pv.groupby("target")}
    return out


def ablation20_summary(art: Path, t1: pd.DataFrame) -> dict:
    ab = A.load_metrics(art / "ablation20")
    if not len(ab):
        return {}
    # S32 = the main CASTER run on the same K562 split (seed 0 and 1)
    full = t1[(t1.target == "k562") & (t1.method.isin(["CASTER", "PerturbMean", "Linear", "LineOffset"])) &
              (t1.seed.isin([0, 1]))].copy()
    full["method"] = full.method.replace({"CASTER": "S32"})
    ab = ab.copy()
    ab["seed"] = [int(s.split("_s")[-1]) if "_s" in s else 0 for s in ab.method]
    ab["method"] = [s.split("_s")[0] if s.split("_s")[-1].isdigit() else s for s in ab.method]
    allab = pd.concat([ab, full], ignore_index=True)
    means = allab.groupby(["method", "seed"])[list(PRIM)].mean().reset_index()
    tests = []
    for v in sorted(set(ab.method)):
        for m in PRIM:
            r = _paired(allab, "S32", v, m)
            r.update(variant=v, metric=m)
            tests.append(r)
    t = pd.DataFrame(tests)
    if len(t) and "p_two_sided" in t:
        t["p_holm"] = A.holm(t.p_two_sided.fillna(1).tolist())
    epochs = {}
    for hp in sorted((art / "ablation20").glob("history_*.json")):
        h = json.loads(hp.read_text())
        epochs[hp.stem.removeprefix("history_")] = len(h["history"])
    return {"means": means.to_dict("records"), "tests": t.to_dict("records"), "epochs": epochs}


def calibration_summary(art: Path, rep: Path) -> dict:
    cdir = rep / "calibration"
    files = sorted(cdir.glob("calib_*.parquet"))
    if not files:
        return {}
    from canpert.calib import HIGHER, drf

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rel = pd.concat([pd.read_parquet(f) for f in sorted(cdir.glob("reliability_*.parquet"))], ignore_index=True)
    means = df.groupby(["task", "target", "method"])[list(HIGHER)].mean().reset_index()
    means.to_csv(rep / "tables" / "calibrated_metrics.csv", index=False)
    drows = []
    for (task, tgt, seed), g in df.groupby(["task", "target", "seed"]):
        d = drf(g)
        d["task"], d["target"], d["seed"] = task, tgt, seed
        drows.append(d)
    dd = pd.concat(drows, ignore_index=True)
    drf_tab = dd.groupby(["task", "target", "metric"]).drf.mean().reset_index()
    drf_tab.to_csv(rep / "tables" / "drf.csv", index=False)
    # normalised score of each method between MeanBaseline (0) and TechDup (1)
    norm = []
    from canpert.calib import DELTAPERT

    for (task, tgt), g in means.groupby(["task", "target"]):
        gi = g.set_index("method")
        for m, hb in HIGHER.items():
            # same negative control as the DRF: zero-shift prediction for delta-from-perturbed-mean metrics
            neg = "NoChange" if m in DELTAPERT and "NoChange" in gi.index else "MeanBaseline"
            lo, hi = gi.loc[neg, m], gi.loc["TechDup", m]
            gap = (hi - lo) if hb else (lo - hi)
            # a metric whose positive control is not clearly better than its negative control cannot be
            # rescaled meaningfully (division by ~0): mark it not calibrated (NaN)
            ok = np.isfinite(gap) and gap > 0.05 * max(abs(lo), abs(hi), 1e-9)
            for me in gi.index:
                v = gi.loc[me, m]
                norm.append({"task": task, "target": tgt, "method": me, "metric": m,
                             "fraction_of_range": float((v - lo) / (hi - lo)) if ok else np.nan,
                             "calibrated": bool(ok)})
    pd.DataFrame(norm).to_csv(rep / "tables" / "calibrated_normalised.csv", index=False)
    ceil = rel.groupby(["task", "target"]).agg(split_half_r=("split_half_r", "median"),
                                               ceiling=("ceiling_pearson_delta", "median")).reset_index()
    ceil.to_csv(rep / "tables" / "noise_ceiling.csv", index=False)
    # paired tests on the calibrated metrics, T1 seed-pooled
    tests = []
    t1c = df[df.task == "t1"]
    for b in ("PerturbMean", "Linear", "LineOffset", "ContextMean"):
        if b not in set(t1c.method):
            continue
        for m in HIGHER:
            r = _paired(t1c, "CASTER", b, m)
            r.update(baseline=b, metric=m)
            tests.append(r)
    tt = pd.DataFrame(tests)
    tt.to_csv(rep / "tables" / "calibrated_tests.csv", index=False)
    return {"means": means.to_dict("records"), "drf": drf_tab.to_dict("records"), "ceiling": ceil.to_dict("records"),
            "tests": tt.to_dict("records")}


def noise_free_summary(t1: pd.DataFrame) -> dict:
    """Noise-free rendering: every method re-scored after removing the sampling noise of the control cells
    it was rendered on ('<method>_clean')."""
    cl = t1[t1.method.str.endswith("_clean")]
    if not len(cl):
        return {}
    fm = cl.groupby(["seed", "target", "method"])[list(PRIM)].mean().reset_index()
    allc = fm.groupby(["seed", "method"])[list(PRIM)].mean().reset_index()
    canc = fm[fm.target.isin(CANCER_LINES)].groupby(["seed", "method"])[list(PRIM)].mean().reset_index()
    out = {"per_line": fm.groupby(["target", "method"])[list(PRIM)].mean().reset_index().to_dict("records"),
           "all_lines": allc.groupby("method")[list(PRIM)].agg(["mean", "std"]).pipe(
               lambda d: {m: {k: {"mean": float(d.loc[m, (k, "mean")]), "sd": float(d.loc[m, (k, "std")])}
                              for k in PRIM} for m in d.index}),
           "cancer_only": canc.groupby("method")[list(PRIM)].mean().to_dict("index"),
           "tests": []}
    for b in ("Linear_clean", "PerturbMean_clean", "LineOffset_clean", "ContextMean_clean"):
        for m in PRIM:
            r = _paired(cl, "CASTER_clean", b, m)
            r.update(baseline=b, metric=m)
            out["tests"].append(r)
            for seed, g in cl.groupby("seed"):
                r2 = _paired(g, "CASTER_clean", b, m)
                r2.update(baseline=b, metric=m, seed=int(seed))
                out.setdefault("per_seed_tests", []).append(r2)
    return out


def preregistered_de_tests(t1: pd.DataFrame) -> list[dict]:
    """Pre-registered (Wilcoxon on predicted cells) versions of every DE-affected test, seed 0."""
    g = t1[t1.seed == 0]
    rows = []
    for m in DE_AFFECTED:
        col = f"{m}_wilcoxon"
        if col not in g:
            continue
        sub = g.drop(columns=[m]).rename(columns={col: m})
        r = A.paired_tests(sub, "CASTER", A.BASELINES, (m,))
        rows += r.to_dict("records")
    return rows


def run(art: Path, rep: Path, res: dict) -> dict:
    tab = rep / "tables"
    t1 = A.load_metrics(art / "t1")
    rv: dict = {}
    rv["seeds"] = seeds_summary(t1)
    rv["controls"] = controls_summary(art, t1)
    rv["grouped"] = grouped_summary(art, rep)
    rv["ablation20"] = ablation20_summary(art, t1)
    rv["calibration"] = calibration_summary(art, rep)
    rv["preregistered_de"] = preregistered_de_tests(t1)
    rv["noise_free"] = noise_free_summary(t1)
    t2 = A.load_metrics(art / "t2")
    if len(t2) and (t2.method.str.endswith("_clean")).any():
        c2 = t2[t2.method.str.endswith("_clean")]
        rv["noise_free_t2"] = {"means": c2.groupby(["target", "method"])[list(PRIM)].mean().reset_index().to_dict("records"),
                               "tests": {m: _paired(c2, "CASTER_clean", "PerturbMean_clean", m) for m in PRIM}}
    tr = A.load_metrics(art / "transfer")
    if len(tr) and (tr.method.str.endswith("_clean")).any():
        ct = tr[tr.method.str.endswith("_clean")].copy()
        ct["kind"] = np.where((ct.source == "rpe1") | (ct.target == "rpe1"), "rpe1", "cancer")
        piv = ct[ct.method == "PerturbMean_clean"].groupby(["perturbation", "kind"]).pearson_delta.mean().unstack().dropna()
        w = stats.wilcoxon(piv["cancer"], piv["rpe1"])
        rv["noise_free_h3"] = {"mean_cancer_cancer": float(piv["cancer"].mean()), "mean_rpe1_pairs": float(piv["rpe1"].mean()),
                               "p_two_sided": float(w.pvalue), "n": int(len(piv))}
    if (tab / "review_extra.json").exists():
        rv["extra"] = json.loads((tab / "review_extra.json").read_text())
    tr = pd.read_csv(tab / "transfer_matrix.csv")
    p = tr.pivot_table(index=["source", "target"], columns="method", values=list(PRIM) + ["de_overlap"])
    rv["transfer_worse_counts"] = {m: int(((p[m]["CASTER"] - p[m]["PerturbMean"]) < 0).sum())
                                   for m in list(PRIM) + ["de_overlap"]}
    f2 = pd.read_csv(tab / "t2_fold_means.csv")
    mets = ["pearson_delta", "discrimination", "de_auprc", "de_overlap", "lfc_spearman", "effect_spearman"]
    worse = sum(int(f2[(f2.target == n) & (f2.method == "CASTER")][m].iloc[0]
                    < f2[(f2.target == n) & (f2.method == "PerturbMean")][m].iloc[0]) for m in mets for n in f2.target.unique())
    rv["t2_worse_panels"] = {"worse": worse, "total": len(mets) * f2.target.nunique()}
    eff = {}
    for me in ("CASTER", "PerturbMean", "Linear"):
        d = t1[(t1.seed == 0) & (t1.method == me)]
        eff[me] = float(stats.spearmanr(d.n_de_pred, d.n_de_true).statistic)
    rv["effect_rho_pooled"] = eff
    s9 = tab / "splits.csv"
    if s9.exists():
        s = pd.read_csv(s9)
        rv["minutes_range"] = [float(s.minutes.min()), float(s.minutes.max())]
    res["review"] = rv
    return res
