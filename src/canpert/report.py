"""Run every pre-declared test and exploratory analysis; write results.json + tables.

The manuscript builder and figure module read only what this module writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from canpert import analysis as A
from canpert.data import CANCER_LINES, LINES
from canpert.evaluate import summarise

METRICS = ("pearson_delta", "discrimination", "mae", "de_auprc", "de_overlap", "lfc_spearman",
           "direction")


def _jsonable(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        return None if not np.isfinite(x) else float(x)
    return x


def effect_size_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho across perturbations between predicted and observed #DEGs, per fold."""
    from scipy import stats

    rows = []
    for (t, s, m), g in df.groupby(["target", "seed", "method"]):
        if g["n_de_pred"].nunique() < 2:
            rho = 0.0
        else:
            rho = stats.spearmanr(g["n_de_pred"], g["n_de_true"]).statistic
        rows.append({"target": t, "seed": s, "method": m, "effect_spearman": rho})
    return pd.DataFrame(rows)


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per fold (target, seed) x method means of every metric + effect-size Spearman."""
    per = df.groupby(["target", "seed", "method"])[list(METRICS)].mean().reset_index()
    return per.merge(effect_size_corr(df), on=["target", "seed", "method"])


def _pred_dict(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Pseudobulk shifts only (drops the stored per-gene DE arrays)."""
    return {k: z[k] for k in z.files if k != "perts" and "__" not in k}


def run_analysis(art: Path, out: Path, raw: Path) -> dict[str, object]:
    """Compute everything and persist it."""
    out.mkdir(parents=True, exist_ok=True)
    tab = out / "tables"
    tab.mkdir(exist_ok=True)
    res: dict[str, object] = {}

    # ---------------------------------------------------------------- data audit
    audit_p = art.parent / "data" / "processed" / "audit.json"
    if audit_p.exists() and "dry" not in art.name:
        res["audit"] = json.loads(audit_p.read_text())

    # ---------------------------------------------------------------- AMENDMENT-002
    proc = art.parent / "data" / "processed"
    if (proc / "audit.json").exists() and "dry" not in art.name:
        from canpert.reeval import run as amend

        amend(art, proc)

    # ---------------------------------------------------------------- T1 (H1)
    t1 = A.load_metrics(art / "t1")
    t1.to_parquet(tab / "t1_per_perturbation.parquet")
    ft1 = fold_table(t1)
    ft1.to_csv(tab / "t1_fold_means.csv", index=False)
    h1 = A.paired_tests(t1[t1.seed == 0], "CASTER", A.BASELINES, A.PRIMARY)
    h1.to_csv(tab / "H1_tests.csv", index=False)
    n_sig = int(((h1.p_holm < 0.05)).sum())
    n_opp = int(((h1.p_two_sided < 0.05) & (h1.median_diff < 0)).sum())
    res["H1"] = {"tests": h1.to_dict("records"), "n_significant": n_sig, "n_opposite": n_opp,
                 "supported": bool(n_sig >= 6 and n_opp == 0)}
    if "de_auprc_wilcoxon" in t1:
        pre = t1[t1.seed == 0].drop(columns=["de_auprc"]).rename(columns={"de_auprc_wilcoxon": "de_auprc"})
        h1p = A.paired_tests(pre, "CASTER", A.BASELINES, A.PRIMARY)
        h1p.to_csv(tab / "H1_tests_preregistered_wilcoxon.csv", index=False)
        n_sig_p = int((h1p.p_holm < 0.05).sum())
        n_opp_p = int(((h1p.p_two_sided < 0.05) & (h1p.median_diff < 0)).sum())
        res["H1_preregistered"] = {"tests": h1p.to_dict("records"), "n_significant": n_sig_p,
                                   "n_opposite": n_opp_p, "supported": bool(n_sig_p >= 6 and n_opp_p == 0)}
        nd = t1[t1.seed == 0].groupby("method")[["n_de_pred", "n_de_pred_wilcoxon", "n_de_true"]].mean()
        res["amendment2_de_counts"] = nd.reset_index().to_dict("records")
    res["T1_summary"] = summarise(ft1, ["method"]).to_dict("records")
    res["T1_per_line"] = summarise(ft1, ["target", "method"]).to_dict("records")
    sec = A.paired_tests(t1, "CASTER", A.BASELINES, tuple(m for m in METRICS if m not in A.PRIMARY))
    sec.to_csv(tab / "T1_secondary_tests.csv", index=False)
    res["T1_secondary"] = sec.to_dict("records")
    res["T1_seeds"] = sorted(t1.seed.unique().tolist())

    # ---------------------------------------------------------------- T2 (H2)
    t2 = A.load_metrics(art / "t2")
    if len(t2):
        t2.to_parquet(tab / "t2_per_perturbation.parquet")
        ft2 = fold_table(t2)
        ft2.to_csv(tab / "t2_fold_means.csv", index=False)
        h2 = A.paired_tests(t2, "CASTER", ("PerturbMean",), A.PRIMARY)
        h2.to_csv(tab / "H2_tests.csv", index=False)
        res["H2"] = {"tests": h2.to_dict("records"),
                     "supported": bool((h2.p_holm < 0.05).all())}
        res["T2_summary"] = summarise(ft2, ["method"]).to_dict("records")
        res["T2_per_line"] = summarise(ft2, ["target", "method"]).to_dict("records")

    # ---------------------------------------------------------------- transfer (H3)
    tr = A.load_metrics(art / "transfer")
    if len(tr):
        tr["source"] = tr["source"].astype(str)
        mat = tr.groupby(["method", "source", "target"])[list(METRICS)].mean().reset_index()
        mat.to_csv(tab / "transfer_matrix.csv", index=False)
        res["transfer_matrix"] = mat.to_dict("records")
        from scipy import stats
        h3 = {}
        for method in ("PerturbMean", "CASTER"):
            g = tr[tr.method == method]
            g = g.assign(kind=np.where((g.source == "rpe1") | (g.target == "rpe1"), "rpe1", "cancer"))
            piv = g.groupby(["perturbation", "kind"])["pearson_delta"].mean().unstack().dropna()
            w = stats.wilcoxon(piv["cancer"], piv["rpe1"])
            h3[method] = {"mean_cancer_cancer": piv["cancer"].mean(), "mean_rpe1_pairs": piv["rpe1"].mean(),
                          "median_diff": (piv["cancer"] - piv["rpe1"]).median(), "n": len(piv),
                          "p_two_sided": w.pvalue}
        res["H3"] = h3
        res["H3"]["supported"] = bool(h3["PerturbMean"]["p_two_sided"] < 0.05
                                      and h3["PerturbMean"]["median_diff"] > 0)

    # ---------------------------------------------------------------- ablation
    ab = A.load_metrics(art / "ablation")
    if len(ab):
        abm = ab.groupby("method")[list(METRICS)].mean().reset_index()
        abm = abm.merge(effect_size_corr(ab).groupby("method")["effect_spearman"].mean().reset_index(), on="method")
        abm.to_csv(tab / "ablation.csv", index=False)
        res["ablation"] = abm.to_dict("records")
        hist = {}
        for p in sorted((art / "ablation").glob("history_*.json")):
            hist[p.stem.removeprefix("history_")] = json.loads(p.read_text())["history"]
        # review round: 20-epoch ablations replace the 10-epoch curves (S32 = main K562 run, seed 0)
        if (art / "ablation20").exists():
            hist = {}
            for p in sorted((art / "ablation20").glob("history_S*.json")):
                tag = p.stem.removeprefix("history_")
                if "_s" not in tag:
                    hist[tag] = json.loads(p.read_text())["history"]
            k = art / "t1" / "history_k562_s0.json"
            if k.exists():
                hist["S32"] = json.loads(k.read_text())["history"]
        res["ablation_history"] = hist
        # paired: S32 (full) vs each ablation, primary metrics
        rows = []
        for other in sorted(set(ab.method) - {"S32"}):
            sub = ab[ab.method.isin(["S32", other])].replace({"S32": "CASTER"})
            rows.append(A.paired_tests(sub.assign(target="k562"), "CASTER", (other,), A.PRIMARY))
        abt = pd.concat(rows, ignore_index=True)
        abt["p_holm"] = A.holm(abt["p_greater"].tolist())
        abt.to_csv(tab / "ablation_tests.csv", index=False)
        res["ablation_tests"] = abt.to_dict("records")

    # ---------------------------------------------------------------- observed effects
    obs = A.load_observed(art / "observed")
    genes = np.load(art / "observed" / "genes.npy", allow_pickle=True).astype(str)
    sim = A.cross_context_similarity(obs)
    sim.to_csv(tab / "cross_context_similarity.csv", index=False)
    nde_map = {n: dict(zip(obs[n]["perts"], (obs[n]["padj"] < 0.05).sum(1), strict=True)) for n in LINES}
    sim["min_nde"] = [min(nde_map[a][p], nde_map[b][p]) for a, b, p in zip(sim.a, sim.b, sim.perturbation,
                                                                          strict=True)]
    strength = []
    for thr in (10, 25, 50, 100, 200):
        t = sim[sim.min_nde >= thr]
        for kind, g in t.groupby("pair_type"):
            strength.append({"min_de": thr, "pair_type": kind, "median_pearson": float(g.pearson.median()),
                             "n": len(g)})
    pd.DataFrame(strength).to_csv(tab / "similarity_by_strength.csv", index=False)
    sim.to_csv(tab / "cross_context_similarity.csv", index=False)
    res["cross_context"] = {
        "by_strength": strength,
        "by_pair": sim.groupby(["a", "b"])[["pearson", "jaccard"]].median().reset_index().to_dict("records"),
        "by_type": sim.groupby("pair_type")[["pearson", "jaccard"]].median().reset_index().to_dict("records"),
        "n_perturbations_min10de": int(sim.perturbation.nunique()),
    }
    nde = pd.DataFrame({n: pd.Series((obs[n]["padj"] < 0.05).sum(1), index=obs[n]["perts"]) for n in LINES})
    nde.to_csv(tab / "observed_n_de.csv")
    res["observed_n_de"] = {n: {"n_perturbations": int(nde[n].notna().sum()),
                                "median": float(nde[n].median()),
                                "frac_ge10": float((nde[n].dropna() >= 10).mean()),
                                "max": int(nde[n].max())} for n in LINES}

    # ---------------------------------------------------------------- DepMap (H4, H5)
    chronos = A.load_chronos(raw / "depmap" / "CRISPRGeneEffect.parquet")
    chronos.to_csv(tab / "chronos_cancer_lines.csv")
    h4, dep_table = A.dependency_tests(obs, chronos)
    h4.to_csv(tab / "H4_tests.csv", index=False)
    dep_table.to_csv(tab / "dependency_table.csv")
    res["H4"] = {"tests": h4.to_dict("records"),
                 "H4_supported": bool(((h4.test == "H4") & (h4.p_holm < 0.05) & (h4.rho < 0)).sum() >= 2),
                 "H4b_supported": bool(((h4.test == "H4b") & (h4.p_holm < 0.05) & (h4.rho < 0)).sum() >= 2)}
    h5 = []
    for n in CANCER_LINES:
        z = np.load(art / "t1" / f"pred_{n}_s0.npz")
        perts = z["perts"]
        measured = chronos[n].dropna().index  # Chronos is missing for some genes
        keep = np.isin(perts, measured)
        pred = {k: v[keep] for k, v in _pred_dict(z).items()}
        r, preds = A.chronos_from_shifts(pred, chronos.loc[perts[keep], n])
        r["line"] = n
        h5.append(r)
        preds.to_csv(tab / f"H5_predictions_{n}.csv")
    h5d = pd.concat(h5, ignore_index=True)
    h5d.to_csv(tab / "H5_chronos_prediction.csv", index=False)
    res["H5"] = h5d.to_dict("records")

    # ---------------------------------------------------------------- exploratory (per T1 target)
    pairs = A.load_string_pairs(raw / "network")
    reactome = A.read_reactome(raw / "network")
    ari, strg, inv, prog, cc, annot = [], [], [], [], [], []
    for n in LINES:
        z = np.load(art / "t1" / f"pred_{n}_s0.npz")
        pred = _pred_dict(z)
        perts = z["perts"]
        # restrict similarity analyses to perturbations with a detectable observed effect
        d = t1[(t1.target == n) & (t1.seed == 0) & (t1.method == "CASTER")].set_index("perturbation")
        n_true = d.loc[perts, "n_de_true"].to_numpy()
        strong = n_true >= 10
        a = A.clustering_ari({k: v[strong] for k, v in pred.items()})
        a["line"] = n
        ari.append(a)
        s = A.string_recovery({k: v[strong] for k, v in pred.items()}, perts[strong], pairs)
        s["line"] = n
        strg.append(s)
        i = A.inverse_design(pred, n_true)
        i["line"] = n
        inv.append(i)
        labels = A.gene_programs(obs[n]["delta"], genes)
        an = A.annotate_programs(labels, genes, reactome)
        an["line"] = n
        annot.append(an)
        pc = A.program_concordance(pred, labels)
        pc["line"] = n
        prog.append(pc)
        c = A.cell_cycle_shift(pred, genes)
        c["line"] = n
        cc.append(c)
        np.save(tab / f"programs_{n}.npy", labels)
    for name, frames in (("ari", ari), ("string", strg), ("inverse_design", inv), ("programs", prog),
                         ("cell_cycle", cc), ("program_annotation", annot)):
        f = pd.concat(frames, ignore_index=True)
        f.to_csv(tab / f"{name}.csv", index=False)
    res["ari_mean"] = pd.concat(ari).groupby("method")["ari"].mean().to_dict()
    res["string_auroc"] = pd.concat(strg).groupby("method")["auroc"].mean().to_dict()
    inv_all = pd.concat(inv)
    res["inverse_top10"] = inv_all[inv_all.k == 10].groupby("method")["accuracy"].mean().to_dict()
    res["program_pearson"] = pd.concat(prog).groupby("method")["pearson"].mean().to_dict()
    cc_all = pd.concat(cc)
    if len(cc_all):
        res["cell_cycle_mae"] = cc_all.groupby(["phase", "method"])["abs_error"].mean().unstack().to_dict()
        res["cell_cycle_n_genes"] = cc_all.groupby("phase")["n_genes"].first().to_dict()

    # ---------------------------------------------------------------- supplementary analyses
    from scipy import stats as _st

    t1p = t1[t1.seed == 0]
    # (a) does CASTER's gain over PerturbMean depend on response strength?
    gain_rows = []
    for task, df_ in (("T1", t1p), ("T2", t2 if len(t2) else None)):
        if df_ is None:
            continue
        pv = df_.pivot_table(index=["target", "perturbation"], columns="method",
                             values="pearson_delta").dropna(subset=["CASTER", "PerturbMean"])
        nd = df_[df_.method == "CASTER"].set_index(["target", "perturbation"]).n_de_true
        pv = pv.join(nd)
        pv["gain"] = pv["CASTER"] - pv["PerturbMean"]
        if "Linear" not in pv:
            pv["Linear"] = np.nan
        for tgt, g in pv.groupby(level=0):
            r = _st.spearmanr(np.log1p(g.n_de_true), g.gain)
            gain_rows.append({"task": task, "target": tgt, "rho": r.statistic, "p": r.pvalue,
                              "median_gain_weak": g[g.n_de_true < 10].gain.median(),
                              "median_gain_strong": g[g.n_de_true >= 50].gain.median(), "n": len(g)})
        pv.reset_index().to_csv(tab / f"gain_vs_strength_{task}.csv", index=False)
    res["gain_vs_strength"] = gain_rows
    # (b) do predicted responses preserve the dependency-response coupling (H4)?
    h4p = []
    for n in CANCER_LINES:
        d_ = t1p[t1p.target == n]
        v_ = chronos[n].dropna()
        for me in ("_observed", "CASTER", "PerturbMean", "Linear", "ContextMean"):
            col = "n_de_true" if me == "_observed" else "n_de_pred"
            src = d_[d_.method == ("CASTER" if me == "_observed" else me)].set_index("perturbation")[col]
            src = src[src.index.isin(v_.index)]
            if len(src) < 10 or src.nunique() < 2:
                continue
            r = _st.spearmanr(np.log1p(src), v_.loc[src.index])
            h4p.append({"line": n, "method": me, "rho": r.statistic, "p": r.pvalue, "n": len(src)})
    pd.DataFrame(h4p).to_csv(tab / "H4_predicted.csv", index=False)
    res["H4_predicted"] = h4p

    pc_path = tab / "positive_control.csv"
    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        res["positive_control"] = {s: dict(zip(g.method, g.pearson_delta, strict=True))
                                   for s, g in pc.groupby("scenario")}

    # ---------------------------------------------------------------- fold/timing table (Table S9)
    srows = []
    for task in ("t1", "t2", "t1_meanctrl", "t1_grouped"):
        for hp in sorted((art / task).glob("history_*.json")):
            h = json.loads(hp.read_text())
            tag = hp.stem.removeprefix("history_")
            m_ = pd.read_parquet(art / task / f"metrics_{tag}.parquet", columns=["perturbation"])
            hist = h["history"]
            srows.append({"task": task, "fold": tag, "train": len(h["train"]), "val": len(h["val"]),
                          "test": int(m_.perturbation.nunique()), "epochs": len(hist),
                          "best_epoch": int(np.argmin([x["val"] for x in hist])),
                          "minutes": round(h["seconds"] / 60, 1)})
    if srows:
        pd.DataFrame(srows).to_csv(tab / "splits.csv", index=False)

    # ---------------------------------------------------------------- review round (V.1)
    if "dry" not in art.name:
        from canpert.review_report import run as review_run

        res = review_run(art, out, res)

    (out / "results.json").write_text(json.dumps(_jsonable(res), indent=1))
    return res
