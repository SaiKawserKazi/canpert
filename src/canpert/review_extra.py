"""Review round analyses that need no model training: p53 activity, DepMap controls, ribosome
enrichment of CASTER's gain (with a complex-leakage check)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from canpert.analysis import load_chronos, load_observed, read_reactome
from canpert.data import CANCER_LINES, LINES, stream_rows
from canpert.ingest import CONTROL_LABEL

# Fischer (2017) consensus p53 targets measured in all four lines of this dataset
P53_TARGETS = ["MDM2", "BAX", "TP53I3", "RRM2B", "ZMAT3", "FDXR", "AEN", "TRIAP1", "XPC", "RPS27L",
               "CCNG1", "GADD45A", "PPM1D", "APAF1"]
TP53_STATUS = {"k562": "TP53 null (frameshift)", "hepg2": "TP53 wild type", "jurkat": "TP53 mutant",
               "rpe1": "TP53 wild type"}


def p53_scores(raw: Path, out: Path) -> pd.DataFrame:
    """Per line and perturbation: p53 activity shift (mean z over the target genes, z relative to
    the line's controls) and a Welch t statistic against controls."""
    d = raw / "hf"
    obs = pd.read_parquet(d / "obs.parquet")
    var = pd.read_parquet(d / "var.parquet")
    genes = var.gene_name.to_numpy().astype(str)
    cols = np.array([np.where(genes == g)[0][0] for g in P53_TARGETS])
    rows_all = np.sort(np.load(d / "selected_rows.npy"))
    got, x = stream_rows(d / "shards", rows_all, cols)
    x = x.toarray()
    o = obs.iloc[got].reset_index(drop=True)
    res = []
    for n in LINES:
        m = (o.cell_line == n).to_numpy()
        lab = o.gene.to_numpy()[m]
        xm = x[m]
        ctrl = xm[lab == CONTROL_LABEL]
        mu, sd = ctrl.mean(0), ctrl.std(0) + 1e-6
        score = ((xm - mu) / sd).mean(1)
        cs = score[lab == CONTROL_LABEL]
        for p in np.unique(lab):
            if p == CONTROL_LABEL:
                continue
            s = score[lab == p]
            if s.size < 16:
                continue
            t = stats.ttest_ind(s, cs, equal_var=False)
            res.append({"line": n, "perturbation": p, "p53_shift": float(s.mean() - cs.mean()),
                        "t": float(t.statistic), "p": float(t.pvalue), "n": int(s.size)})
    df = pd.DataFrame(res)
    df["padj"] = np.nan
    for n, g in df.groupby("line"):
        from canpert.evaluate import bh

        df.loc[g.index, "padj"] = bh(g.p.to_numpy())
    df["induced"] = (df.padj < 0.05) & (df.p53_shift > 0)
    df.to_csv(out / "p53_scores.csv", index=False)
    return df


def p53_summary(p53: pd.DataFrame, sim: pd.DataFrame) -> dict:
    """Does p53 activation explain the HepG2-RPE1 similarity?"""
    summ = {}
    for n in LINES:
        g = p53[p53.line == n]
        summ[n] = {"tp53": TP53_STATUS[n], "frac_induced": float(g.induced.mean()),
                   "median_shift": float(g.p53_shift.median()),
                   "shift_q90": float(g.p53_shift.quantile(0.9)), "n": int(len(g))}
    piv = p53.pivot_table(index="perturbation", columns="line", values="p53_shift")
    pair = {}
    for i, a in enumerate(LINES):
        for b in LINES[i + 1:]:
            c = piv[[a, b]].dropna()
            pair[f"{a}-{b}"] = float(stats.spearmanr(c[a], c[b]).statistic)
    # stratify HepG2-RPE1 response similarity by p53 induction in both lines
    ind = p53.pivot_table(index="perturbation", columns="line", values="induced")
    hr = sim[(sim.a == "hepg2") & (sim.b == "rpe1")].set_index("perturbation")
    both = ind.reindex(hr.index)[["hepg2", "rpe1"]].fillna(False).astype(bool).all(1)
    strat = {"both_induced_median_r": float(hr.pearson[both].median()), "n_both": int(both.sum()),
             "other_median_r": float(hr.pearson[~both].median()), "n_other": int((~both).sum()),
             "mannwhitney_p": float(stats.mannwhitneyu(hr.pearson[both], hr.pearson[~both]).pvalue)
             if both.sum() > 5 else np.nan}
    # Is HepG2-RPE1 similarity still the highest pair among knockdowns that induce p53 in neither line?
    neither = {}
    for (a, b), g in sim.groupby(["a", "b"]):
        gi = g.set_index("perturbation")
        indn = ind.reindex(gi.index)
        mask = ~(indn.get(a, pd.Series(False, index=gi.index)).fillna(False).astype(bool)
                 | indn.get(b, pd.Series(False, index=gi.index)).fillna(False).astype(bool))
        neither[f"{a}-{b}"] = {"median_r": float(gi.pearson[mask].median()), "n": int(mask.sum())}
    return {"per_line": summ, "pair_spearman_p53_shift": pair, "hepg2_rpe1_by_p53": strat,
            "similarity_p53_neither": neither}


def depmap_controls(obs: dict, chronos: pd.DataFrame) -> dict:
    """Review item 6: is the dependency-response link line-specific?"""
    nde = {n: pd.Series(np.log1p((obs[n]["padj"] < 0.05).sum(1)), index=obs[n]["perts"]) for n in LINES}
    cross = []
    for n in CANCER_LINES:
        for m in CANCER_LINES:
            v = chronos[m].dropna()
            idx = nde[n].index.intersection(v.index)
            cross.append({"response_line": n, "chronos_line": m, "rho": float(stats.spearmanr(nde[n][idx], v[idx]).statistic),
                          "n": len(idx)})
    cross_df = pd.DataFrame(cross)
    # how well does the other lines' Chronos predict this line's Chronos (generality of dependency)?
    gen = []
    for n in CANCER_LINES:
        others = [m for m in CANCER_LINES if m != n]
        c = chronos[[n] + others].dropna()
        panel = c.index.intersection(nde[n].index)
        c = c.loc[panel]
        gen.append({"line": n, "pearson_other_lines_mean_vs_own": float(np.corrcoef(c[others].mean(1), c[n])[0, 1]),
                    "n": len(c)})
    return {"cross": cross_df.to_dict("records"), "generality": gen}


RIBO_TERMS = ("Major pathway of rRNA processing in the nucleolus and cytosol",
              "rRNA processing in the nucleus and cytosol")


def ribosome_sets(net: Path) -> dict[str, set[str]]:
    reac = read_reactome(net)
    rb = set()
    for t in RIBO_TERMS:
        rb |= reac.get(t, set())
    return {"ribosome_biogenesis": rb}


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept; returns coefficients and two-sided t-test p values."""
    xx = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(xx, y, rcond=None)
    resid = y - xx @ beta
    dof = len(y) - xx.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(xx.T @ xx)))
    return beta, 2 * stats.t.sf(np.abs(beta / se), dof)


def ribosome_enrichment(t1: pd.DataFrame, net: Path, groups: dict[str, int], art: Path) -> dict:
    """Review item 7: is CASTER's gain enriched in ribosome biogenesis / ribosomal-protein
    knockdowns after adjusting for response strength, and does it depend on sibling genes
    (same STRING complex group) being in training?"""
    sets = ribosome_sets(net)
    out = {}
    rows = []
    for (tgt, seed), g in t1.groupby(["target", "seed"]):
        pv = g.pivot_table(index="perturbation", columns="method", values="pearson_delta")
        if not {"CASTER", "PerturbMean"} <= set(pv.columns):
            continue
        nd = g[g.method == "CASTER"].set_index("perturbation").n_de_true
        h = json.loads((art / "t1" / f"history_{tgt}_s{seed}.json").read_text())
        train = set(h["train"]) | set(h["val"])
        for p in pv.index:
            gid = groups.get(p)
            sib = sum(1 for q in train if groups.get(q) == gid) if gid is not None else 0
            rows.append({"target": tgt, "seed": seed, "perturbation": p,
                         "gain": pv.loc[p, "CASTER"] - pv.loc[p, "PerturbMean"],
                         "gain_vs_linear": pv.loc[p, "CASTER"] - pv.loc[p, "Linear"] if "Linear" in pv else np.nan,
                         "log_nde": float(np.log1p(nd.get(p, 0))), "siblings_in_train": sib,
                         "ribosomal_protein": p.startswith(("RPL", "RPS")) and not p.startswith(("RPS6K", "RPL3L")),
                         "ribosome_biogenesis": p in sets["ribosome_biogenesis"]})
    d = pd.DataFrame(rows)
    d["ribosome_any"] = d.ribosomal_protein | d.ribosome_biogenesis
    fits = {}
    for (tgt, seed), g in d.groupby(["target", "seed"]):
        beta, pvals = _ols(g.gain.to_numpy(), np.column_stack([g.log_nde, g.ribosome_any.astype(float)]))
        fits[f"{tgt}_s{seed}"] = {"coef_ribosome": float(beta[2]),
                                  "p_ribosome": float(pvals[2]),
                                  "coef_log_nde": float(beta[1]),
                                  "n_ribosome": int(g.ribosome_any.sum()), "n": int(len(g))}
        sub = g[g.ribosome_any]
        if len(sub) > 5:
            r = stats.spearmanr(sub.siblings_in_train, sub.gain)
            fits[f"{tgt}_s{seed}"].update({"rho_siblings_gain_ribosome": float(r.statistic),
                                           "p_siblings": float(r.pvalue)})
    out["per_fold"] = fits
    d.to_csv(art.parent / "reports" / "tables" / "ribosome_gain.csv", index=False)
    return out


def run(art: Path, raw: Path, rep: Path) -> dict:
    tab = rep / "tables"
    res: dict = {}
    obs = load_observed(art / "observed")
    chronos = load_chronos(raw / "depmap" / "CRISPRGeneEffect.parquet")
    res["depmap_controls"] = depmap_controls(obs, chronos)
    sim = pd.read_csv(tab / "cross_context_similarity.csv")
    p53 = pd.read_csv(tab / "p53_scores.csv") if (tab / "p53_scores.csv").exists() else p53_scores(raw / "perturbseq", tab)
    res["p53"] = p53_summary(p53, sim)
    t1 = pd.read_parquet(tab / "t1_per_perturbation.parquet")
    groups = {k: int(v) for k, v in json.loads((art.parent / "data" / "processed" / "complex_groups.json").read_text()).items()}
    res["ribosome"] = ribosome_enrichment(t1, raw / "network", groups, art)
    (tab / "review_extra.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def allgene_similarity(raw: Path, art: Path, out: Path, min_de: int = 10) -> pd.DataFrame:
    """Review check: cross-line response similarity on all 6,546 measured genes instead of the 2,000
    control-selected genes. Pseudobulk shifts are accumulated shard by shard (low memory)."""
    import scipy.sparse as sp

    d = raw / "perturbseq" / "hf"
    obs = pd.read_parquet(d / "obs.parquet")
    sel = np.sort(np.load(d / "selected_rows.npy"))
    key = obs.cell_line.astype(str) + "|" + obs.gene.astype(str)
    keys = pd.Index(sorted(set(key.iloc[sel])))
    pos = pd.Series(np.arange(len(keys)), index=keys)
    G = int(json.loads((d / "layout.json").read_text())["n_genes"])
    sums = np.zeros((len(keys), G), dtype=np.float64)
    cnt = np.zeros(len(keys))
    seen: set[int] = set()
    want = set(sel.tolist())
    for path in sorted((d / "shards").glob("shard*.npz")):
        if ".tmp" in path.name:
            continue
        z = np.load(path)
        m = sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int32), z["indptr"]), shape=tuple(z["shape"]))
        rows = z["rows"]
        keep = np.array([int(r) in want and int(r) not in seen for r in rows])
        if not keep.any():
            continue
        idx = np.where(keep)[0]
        k = pos[key.iloc[rows[idx]].to_numpy()].to_numpy()
        sub = m[idx].toarray()
        np.add.at(sums, k, sub)
        np.add.at(cnt, k, 1)
        seen.update(int(r) for r in rows[idx])
    means = sums / np.maximum(cnt, 1)[:, None]
    obsd = load_observed(art / "observed")
    nde = {n: dict(zip(obsd[n]["perts"], (obsd[n]["padj"] < 0.05).sum(1), strict=True)) for n in LINES}
    ctrl = {n: means[pos[f"{n}|{CONTROL_LABEL}"]] for n in LINES}
    res = []
    for i, a in enumerate(LINES):
        for b in LINES[i + 1:]:
            for p in sorted(set(nde[a]) & set(nde[b])):
                if nde[a][p] < min_de or nde[b][p] < min_de:
                    continue
                da = means[pos[f"{a}|{p}"]] - ctrl[a]
                db = means[pos[f"{b}|{p}"]] - ctrl[b]
                res.append({"a": a, "b": b, "perturbation": p, "pearson_allgenes": float(np.corrcoef(da, db)[0, 1])})
    df = pd.DataFrame(res)
    df.to_csv(out / "similarity_allgenes.csv", index=False)
    return df
