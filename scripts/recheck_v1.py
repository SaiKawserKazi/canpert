"""Independent recheck of the V.1 review numbers (does not import canpert.report/review_report).

Recomputes from raw artifacts and compares with reports/results.json; prints PASS/FAIL per item."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "reports" / "results.json").read_text())
rv = res["review"]
ok_all = True


def check(name: str, mine: float, reported: float, tol: float = 1e-6) -> None:
    global ok_all
    good = (np.isnan(mine) and np.isnan(reported)) or abs(mine - reported) <= tol * max(1, abs(reported))
    ok_all &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: recomputed {mine:.6g} vs reported {reported:.6g}")


# 1. per-seed means (all lines) from raw metric files
t1 = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT / "artifacts/t1/metrics_*.parquet")))])
t1 = t1[t1.seed.isin(rv["seeds"]["seeds"])]  # exactly the folds the report used
for me in ("CASTER", "Linear", "PerturbMean", "LineOffset"):
    per_seed = t1[t1.method == me].groupby(["seed", "target"]).pearson_delta.mean().groupby("seed").mean()
    check(f"seeds mean Pearson delta {me}", float(per_seed.mean()), rv["seeds"]["all_lines"][me]["pearson_delta"]["mean"])
    check(f"seeds SD Pearson delta {me}", float(per_seed.std()), rv["seeds"]["all_lines"][me]["pearson_delta"]["sd"])

# 2. LineOffset and wrong-line control paired tests (two-sided Wilcoxon, zsplit)
def paired(a: str, b: str, df: pd.DataFrame) -> tuple[float, float]:
    x = df[df.method == a].set_index(["target", "seed", "perturbation"]).pearson_delta
    y = df[df.method == b].set_index(["target", "seed", "perturbation"]).pearson_delta
    j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna()
    d = j.a - j.b
    return float(j.b.mean()), float(stats.wilcoxon(d, zero_method="zsplit").pvalue)

for b, key in (("LineOffset", "LineOffset"), ("CASTER_wrongctrl", "wrongctrl")):
    mb, p = paired("CASTER", b, t1)
    rep = rv["controls"][key]["pearson_delta"]
    check(f"control {b}: mean", mb, rep["mean_b"])
    lp = lambda v: np.log10(max(v, 1e-300))  # noqa: E731
    check(f"control {b}: log10 p", lp(p), lp(rep["p_two_sided"]), tol=1e-3)

# 3. noise-free means
for me in ("CASTER", "Linear", "PerturbMean"):
    c = t1[t1.method == f"{me}_clean"].groupby(["seed", "target"]).pearson_delta.mean().groupby("seed").mean()
    check(f"noise-free mean {me}", float(c.mean()), rv["noise_free"]["all_lines"][f"{me}_clean"]["pearson_delta"]["mean"])

# 4. noise ceiling: median of sqrt(2r/(1+r)) per line from reliability files (T1)
rel = pd.concat([pd.read_parquet(p) for p in glob.glob(str(ROOT / "reports/calibration/reliability_t1_*.parquet"))])
rel = rel[rel.task == "t1"]  # main folds only (excludes ID-only / grouped controls)
ceil_rep = {r["target"]: r["ceiling"] for r in rv["calibration"]["ceiling"] if r["task"] == "t1"}
for n, g in rel.groupby("target"):
    r = g.split_half_r
    mine = float(np.median(np.sqrt(np.clip(2 * r / (1 + r), 0, None))))
    check(f"noise ceiling {n}", mine, ceil_rep[n], tol=1e-6)

# 5. p53 fraction induced (from the per-knockdown table) and DepMap cross-line rho (from raw files)
p53 = pd.read_csv(ROOT / "reports/tables/p53_scores.csv")
for n, g in p53.groupby("line"):
    mine = float(((g.padj < 0.05) & (g.p53_shift > 0)).mean())
    check(f"p53 frac induced {n}", mine, rv["extra"]["p53"]["per_line"][n]["frac_induced"])
gene = pd.read_parquet(ROOT / "data/raw/depmap/CRISPRGeneEffect.parquet").set_index("ModelID")
gene.columns = [c.split(" (")[0] for c in gene.columns]
gene = gene.loc[:, ~gene.columns.duplicated()]
ids = {"k562": "ACH-000551", "hepg2": "ACH-000739", "jurkat": "ACH-000995"}
cross = {(r["response_line"], r["chronos_line"]): r["rho"] for r in rv["extra"]["depmap_controls"]["cross"]}
for n in ids:
    o = np.load(ROOT / f"artifacts/observed/observed_{n}.npz")
    nde = pd.Series(np.log1p((o["padj"] < 0.05).sum(1)), index=o["perts"])
    for m, mid in ids.items():
        ch = gene.loc[mid].dropna()
        idx = nde.index.intersection(ch.index)
        check(f"DepMap rho response {n} vs Chronos {m}", float(stats.spearmanr(nde[idx], ch[idx]).statistic),
              cross[(n, m)], tol=1e-6)

# 6. pre-specified H1 (seed 0) still unchanged by the review round
h1 = pd.DataFrame(res["H1"]["tests"])
print(f"{'PASS' if res['H1']['n_significant'] == 6 and res['H1']['n_opposite'] == 0 else 'CHECK'}  "
      f"H1 seed 0: {res['H1']['n_significant']}/9 significant, {res['H1']['n_opposite']} opposite")
print("ALL PASS" if ok_all else "SOME FAILED")
