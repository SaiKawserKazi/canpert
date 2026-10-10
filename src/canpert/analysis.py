"""Statistics and downstream analyses (H1-H5 and exploratory), from artifacts only."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import KFold

from canpert.data import CANCER_LINES, LINES

DEPMAP_IDS = {"k562": "ACH-000551", "hepg2": "ACH-000739", "jurkat": "ACH-000995"}
PRIMARY = ("pearson_delta", "discrimination", "de_auprc")
LOWER_IS_BETTER = {"mae", "mse", "wmse"}
BASELINES = ("PerturbMean", "ContextMean", "Linear")

# Tirosh et al. 2016 / Regev-lab cell-cycle signatures (as distributed with Scanpy tutorials)
S_GENES = """MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP HELLS
RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1
TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8""".split()
G2M_GENES = """HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3
FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 CDC20
TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE
CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA""".split()


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p values."""
    arr = np.asarray(p, dtype=float)
    order = np.argsort(arr)
    m = arr.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * arr[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def load_metrics(folder: Path, pattern: str = "metrics_*.parquet") -> pd.DataFrame:
    """Concatenate all metric files of a task."""
    frames = [pd.read_parquet(p) for p in sorted(folder.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_tests(df: pd.DataFrame, model: str, baselines: tuple[str, ...],
                 metrics: tuple[str, ...]) -> pd.DataFrame:
    """One-sided paired Wilcoxon (model > baseline) per metric, pooled over folds."""
    rows = []
    for b in baselines:
        if b not in set(df["method"]):
            continue
        for m in metrics:
            a = df[df.method == model].set_index(["target", "seed", "perturbation"])[m]
            c = df[df.method == b].set_index(["target", "seed", "perturbation"])[m]
            j = pd.concat([a, c], axis=1, keys=["a", "c"]).dropna()
            diff = j["a"] - j["c"]
            if m in LOWER_IS_BETTER:
                diff = -diff  # orient so that a positive difference always means the model is better
            greater = stats.wilcoxon(diff, alternative="greater", zero_method="zsplit")
            two = stats.wilcoxon(diff, alternative="two-sided", zero_method="zsplit")
            rows.append({"model": model, "baseline": b, "metric": m, "n": len(j),
                         "mean_model": j["a"].mean(), "mean_baseline": j["c"].mean(),
                         "median_diff": diff.median(), "frac_better": float((diff > 0).mean()),
                         "p_greater": greater.pvalue, "p_two_sided": two.pvalue,
                         "relative_improvement": (j["a"].mean() - j["c"].mean()) / abs(j["c"].mean())
                         if j["c"].mean() != 0 else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_greater"].tolist())
    return out


# ----------------------------------------------------------------------------- observed effects
def load_observed(folder: Path) -> dict[str, dict[str, np.ndarray]]:
    """Observed per-line effect arrays written by ``observed_effects``."""
    return {n: dict(np.load(folder / f"observed_{n}.npz")) for n in LINES}


def cross_context_similarity(obs: dict[str, dict[str, np.ndarray]], min_de: int = 10) -> pd.DataFrame:
    """Per perturbation and line pair: Pearson r of observed shifts and Jaccard of DEGs.

    Restricted to perturbations with >= min_de DEGs in both lines, so that noise-only
    responses do not dominate the comparison.
    """
    rows = []
    pos = {n: {p: i for i, p in enumerate(obs[n]["perts"])} for n in LINES}
    for i, a in enumerate(LINES):
        for b in LINES[i + 1:]:
            da, db = obs[a]["delta"], obs[b]["delta"]
            sa, sb = obs[a]["padj"] < 0.05, obs[b]["padj"] < 0.05
            for p in sorted(set(pos[a]) & set(pos[b])):
                ka, kb = pos[a][p], pos[b][p]
                na, nb = int(sa[ka].sum()), int(sb[kb].sum())
                if na < min_de or nb < min_de:
                    continue
                r = stats.pearsonr(da[ka], db[kb])[0]
                jac = (sa[ka] & sb[kb]).sum() / max((sa[ka] | sb[kb]).sum(), 1)
                rows.append({"perturbation": p, "a": a, "b": b, "pearson": r, "jaccard": jac,
                             "pair_type": "cancer-cancer" if b != "rpe1" and a != "rpe1" else "cancer-RPE1"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- DepMap
def load_chronos(path: Path) -> pd.DataFrame:
    """Chronos gene effect for the three cancer lines: index gene symbol, columns lines."""
    g = pd.read_parquet(path).set_index("ModelID")
    g.columns = [c.split(" (")[0] for c in g.columns]
    out = pd.DataFrame({n: g.loc[i] for n, i in DEPMAP_IDS.items()})
    return out[~out.index.duplicated()]


def dependency_tests(obs: dict[str, dict[str, np.ndarray]], chronos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H4 (per line, that line's perturbations) and H4b (context-specific; perturbations
    observed in all three cancer lines) Spearman tests."""
    series = {n: pd.Series((obs[n]["padj"] < 0.05).sum(1), index=obs[n]["perts"]) for n in CANCER_LINES}
    rows = []
    for n in CANCER_LINES:
        idx = series[n].index.intersection(chronos.index)
        v = chronos.loc[idx, n].dropna()
        r = stats.spearmanr(np.log1p(series[n].loc[v.index]), v)
        rows.append({"test": "H4", "line": n, "n": len(v), "rho": r.statistic, "p": r.pvalue})
    nde = pd.DataFrame(series).dropna()
    common = nde.index.intersection(chronos.dropna().index)
    nde, ch = np.log1p(nde.loc[common]), chronos.loc[common, list(CANCER_LINES)]
    for n in CANCER_LINES:
        others = [o for o in CANCER_LINES if o != n]
        dn = nde[n] - nde[others].mean(1)
        dc = ch[n] - ch[others].mean(1)
        r = stats.spearmanr(dn, dc)
        rows.append({"test": "H4b", "line": n, "n": len(common), "rho": r.statistic, "p": r.pvalue})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for t in ("H4", "H4b"):
        m = res.test == t
        res.loc[m, "p_holm"] = holm(res.loc[m, "p"].tolist())
    table = pd.concat([nde.add_prefix("log1p_nde_"), ch.add_prefix("chronos_")], axis=1)
    return res, table


def chronos_from_shifts(pred: dict[str, np.ndarray], chronos: pd.Series, seed: int = 0) -> pd.DataFrame:
    """H5: ridge trained on observed shifts predicts Chronos; applied to each method's shifts.

    ``pred`` maps method -> (perts x genes) shifts aligned with chronos.index; must contain
    '_true'. 10-fold CV by perturbation; the regressor never sees a predicted shift in
    training.
    """
    y = chronos.to_numpy()
    out = {m: np.zeros_like(y) for m in pred}
    for tr, te in KFold(10, shuffle=True, random_state=seed).split(y):
        reg = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(pred["_true"][tr], y[tr])
        for m, x in pred.items():
            out[m][te] = reg.predict(x[te])
    rows = []
    for m, yhat in out.items():
        r = stats.pearsonr(yhat, y) if np.std(yhat) > 0 else (np.nan, np.nan)
        rows.append({"method": m, "pearson": r[0], "p": r[1], "n": len(y)})
    return pd.DataFrame(rows), pd.DataFrame(out, index=chronos.index).assign(measured=y)


# ----------------------------------------------------------------------------- similarity
def _cos(x: np.ndarray) -> np.ndarray:
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    return z @ z.T


def clustering_ari(pred: dict[str, np.ndarray], ks: range = range(2, 21)) -> pd.DataFrame:
    """ARI between clusterings of predicted and observed perturbation-similarity maps."""
    rows = []
    true_d = 1 - _cos(pred["_true"])
    for k in ks:
        lab_t = AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(true_d)
        for m, x in pred.items():
            if m == "_true":
                continue
            d = 1 - _cos(x)
            if np.allclose(d, d[0, 0]):
                ari = 0.0
            else:
                lab = AgglomerativeClustering(k, metric="precomputed", linkage="average").fit_predict(d)
                ari = adjusted_rand_score(lab_t, lab)
            rows.append({"k": k, "method": m, "ari": ari})
    return pd.DataFrame(rows)


def load_string_pairs(net_dir: Path, min_score: int = 700) -> set[frozenset[str]]:
    """High-confidence STRING v12 pairs as symbol pairs."""
    with gzip.open(net_dir / "9606.protein.info.v12.0.txt.gz", "rt") as f:
        info = pd.read_csv(f, sep="\t", usecols=[0, 1])
    sym = dict(zip(info.iloc[:, 0], info.iloc[:, 1], strict=True))
    pairs: set[frozenset[str]] = set()
    with gzip.open(net_dir / "9606.protein.links.v12.0.txt.gz", "rt") as f:
        for chunk in pd.read_csv(f, sep=" ", chunksize=2_000_000):
            hi = chunk[chunk.combined_score >= min_score]
            for a, b in zip(hi.protein1.map(sym), hi.protein2.map(sym), strict=True):
                pairs.add(frozenset((a, b)))
    return pairs


def string_recovery(pred: dict[str, np.ndarray], perts: np.ndarray, pairs: set[frozenset[str]]) -> pd.DataFrame:
    """AUROC for separating STRING-linked perturbation pairs by effect cosine similarity."""
    iu = np.triu_indices(len(perts), 1)
    lab = np.array([frozenset((perts[i], perts[j])) in pairs for i, j in zip(*iu, strict=True)])
    rows = []
    for m, x in pred.items():
        s = _cos(x)[iu]
        auc = roc_auc_score(lab, s) if lab.any() and np.std(s) > 0 else 0.5
        rows.append({"method": m, "auroc": auc, "n_pairs": int(lab.size), "n_linked": int(lab.sum())})
    return pd.DataFrame(rows)


def inverse_design(pred: dict[str, np.ndarray], nde_true: np.ndarray, top: int = 100,
                   ks: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    """Top-k retrieval: for the `top` strongest perturbations, is the correct perturbation
    among the k predictions nearest (L1) to its observed shift?"""
    strong = np.argsort(-nde_true)[:top]
    rows = []
    true = pred["_true"]
    for m, x in pred.items():
        if m == "_true":
            continue
        d = np.abs(true[strong][:, None, :] - x[None, :, :]).sum(-1)
        rank = (d < d[np.arange(len(strong)), strong][:, None]).sum(1)
        for k in ks:
            rows.append({"method": m, "k": k, "accuracy": float((rank < k).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- programs
def read_reactome(net_dir: Path) -> dict[str, set[str]]:
    """Reactome pathway -> gene set."""
    with zipfile.ZipFile(net_dir / "ReactomePathways.gmt.zip") as z:
        name = [n for n in z.namelist() if n.endswith(".gmt")][0]
        text = io.TextIOWrapper(z.open(name), encoding="utf-8").read()
    out = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) > 3:
            out[parts[0]] = set(parts[2:])
    return out


def gene_programs(obs_delta: np.ndarray, genes: np.ndarray, n_programs: int = 15,
                  seed: int = 0) -> np.ndarray:
    """Cluster genes by their response profile across perturbations (K-means on z-scores)."""
    z = (obs_delta - obs_delta.mean(0)) / (obs_delta.std(0) + 1e-9)
    return KMeans(n_programs, n_init=10, random_state=seed).fit_predict(z.T)


def annotate_programs(labels: np.ndarray, genes: np.ndarray, reactome: dict[str, set[str]]) -> pd.DataFrame:
    """Best Reactome term per program (hypergeometric, background = modelled genes)."""
    universe = set(genes)
    rows = []
    for k in np.unique(labels):
        members = set(genes[labels == k])
        tested = []
        for term, gs in reactome.items():
            g = gs & universe
            if not 10 <= len(g) <= 500:  # avoid tiny, idiosyncratic terms and very broad ones
                continue
            hit = len(members & g)
            p = stats.hypergeom.sf(hit - 1, len(universe), len(g), len(members)) if hit >= 3 else 1.0
            tested.append((term, p, hit))
        from canpert.evaluate import bh

        padj = bh(np.array([t[1] for t in tested])) if tested else np.array([])
        i = int(np.argmin(padj)) if len(padj) else -1
        best = (tested[i][0], tested[i][1], tested[i][2], padj[i]) if i >= 0 and padj[i] < 0.05 else ("(none)", 1.0, 0, 1.0)
        rows.append({"program": int(k), "size": len(members), "top_term": best[0], "p": best[1], "hits": best[2],
                     "p_adj": best[3], "n_terms_tested": len(tested)})
    return pd.DataFrame(rows)


def program_concordance(pred: dict[str, np.ndarray], labels: np.ndarray) -> pd.DataFrame:
    """Pearson r (across perturbations) of program activity, predicted vs observed."""
    rows = []
    for k in np.unique(labels):
        sel = labels == k
        t = pred["_true"][:, sel].mean(1)
        for m, x in pred.items():
            if m == "_true":
                continue
            a = x[:, sel].mean(1)
            r = stats.pearsonr(a, t)[0] if np.std(a) > 0 else 0.0
            rows.append({"program": int(k), "method": m, "pearson": r})
    return pd.DataFrame(rows)


def cell_cycle_shift(pred: dict[str, np.ndarray], genes: np.ndarray) -> pd.DataFrame:
    """Absolute error of predicted S / G2M signature shifts (mean over signature genes)."""
    rows = []
    for phase, sig in (("S", S_GENES), ("G2M", G2M_GENES)):
        sel = np.isin(genes, sig)
        if sel.sum() < 3:
            continue
        t = pred["_true"][:, sel].mean(1)
        for m, x in pred.items():
            if m == "_true":
                continue
            e = np.abs(x[:, sel].mean(1) - t)
            rows += [{"phase": phase, "method": m, "abs_error": v, "n_genes": int(sel.sum())} for v in e]
    return pd.DataFrame(rows)
