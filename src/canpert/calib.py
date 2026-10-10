"""Review round (V.0.0 -> V.1): calibrated metrics, noise ceiling and the LineOffset baseline.

Calibrated evaluation follows Miller et al. (2026) / Mejia et al. (2025): each held-out
perturbation's cells are split into two random halves A and B (seeded per perturbation).

- truth  = pseudobulk shift of half A (vs the fixed reference controls);
- TechDup (technical duplicate, positive control) = shift of half B;
- MeanBaseline (negative control) = mean shift of the target line's support perturbations;
- InterpDup (interpolated duplicate) = alpha * TechDup + (1 - alpha) * MeanBaseline with
  alpha_g = 1 - p_g (BH-adjusted Welch t-test p of half A vs the rest of the line);
- DEG weights w_g = (min-max normalised |t_g|)^2 from the same half-A test.

Metrics (per perturbation): pearson_deltactrl; pearson_deltapert_top100 (centred on the
perturbed mean = MeanBaseline, top-100 genes by |t|); mse; mae; wmse; weighted r2 on
delta-from-control and delta-from-perturbed-mean (clipped at -1); NIR (Euclidean) and PDS
(Manhattan) = fraction of other test perturbations whose truth is farther from the prediction
than the correct truth. DRF = (TechDup - baseline) / (perfect - baseline) for higher-is-better
metrics and (baseline - TechDup) / baseline for lower-is-better, clipped to [-1, 1], baseline =
MeanBaseline (zero shift for the delta-from-perturbed-mean metrics).

Noise ceiling for the main (full-cell) evaluation: split-half Pearson r_h between the two
half-shifts gives the reliability of the full-cell shift by Spearman-Brown,
rho = 2 r_h / (1 + r_h); the best achievable Pearson delta is sqrt(rho).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from canpert.data import LINES, LineData
from canpert.evaluate import bh
from canpert.experiments import N_REF
from canpert.train import eligible, index_line, make_split, pert_mean

HIGHER = {"pearson_deltactrl": True, "pearson_deltapert_top100": True, "mse": False, "mae": False,
          "wmse": False, "wr2_deltactrl": True, "wr2_deltapert": True, "nir": True, "pds": True}
DELTAPERT = {"pearson_deltapert_top100", "wr2_deltapert"}


def load_line(proc: Path, n: str) -> LineData:
    z = np.load(proc / f"{n}.npz", allow_pickle=False)
    return LineData(n, z["x"], z["pert"], z["gem"])


def _halves(cells: np.ndarray, p: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(zlib.crc32(f"half|{p}".encode()))
    c = rng.permutation(cells)
    h = c.size // 2
    return np.sort(c[:h]), np.sort(c[h:])


def _welch_vs_rest(xa: np.ndarray, tot_sum: np.ndarray, tot_sq: np.ndarray, n_tot: int) -> tuple[np.ndarray, np.ndarray]:
    na = xa.shape[0]
    ma = xa.mean(0)
    va = xa.var(0, ddof=1)
    nr = n_tot - na
    mr = (tot_sum - xa.sum(0)) / nr
    vr = np.maximum(((tot_sq - (xa ** 2).sum(0)) - nr * mr ** 2) / (nr - 1), 0)
    se = np.sqrt(va / na + vr / nr)
    t = np.where(se > 0, (ma - mr) / np.where(se > 0, se, 1), 0.0)
    dof = np.where(se > 0, (va / na + vr / nr) ** 2 / ((va / na) ** 2 / (na - 1) + (vr / nr) ** 2 / (nr - 1) + 1e-30), 1)
    p = 2 * stats.t.sf(np.abs(t), np.maximum(dof, 1))
    return t, p


def _r2w(truth: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    if w.sum() <= 0 or np.allclose(truth, truth.mean()):
        return np.nan
    return max(r2_score(truth, pred, sample_weight=w), -1.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def score_fold(line: LineData, perts: list[str], preds: dict[str, np.ndarray], split_support: list[str],
               seed_ctrl: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated metrics for one fold. ``preds``: method -> (T, G) shifts aligned to ``perts``
    (relative to the reference-control mean, as stored in pred_*.npz)."""
    ix = index_line(line, seed_ctrl)
    ref_mean = line.x[ix.ctrl_ref[:N_REF]].mean(0)
    x = line.x
    tot_sum = x.sum(0, dtype=np.float64)
    tot_sq = (x.astype(np.float64) ** 2).sum(0)
    n_tot = x.shape[0]
    mb = np.mean([pert_mean(ix, q) for q in split_support], axis=0) - ref_mean
    truth, tech, interp, weights, top = [], [], [], [], []
    rel = []
    for p in perts:
        a, b = _halves(ix.by_pert[p], p)
        ta = x[a].mean(0) - ref_mean
        tb = x[b].mean(0) - ref_mean
        t, pv = _welch_vs_rest(x[a].astype(np.float64), tot_sum, tot_sq, n_tot)
        alpha = 1 - bh(pv)
        at = np.abs(t)
        w = ((at - at.min()) / (at.max() - at.min() + 1e-12)) ** 2
        truth.append(ta)
        tech.append(tb)
        interp.append(alpha * tb + (1 - alpha) * mb)
        weights.append(w)
        top.append(np.argsort(-at)[:100])
        rel.append(_pearson(ta, tb))
    truth_m = np.stack(truth)
    allp = dict(preds)
    allp["TechDup"] = np.stack(tech)
    allp["InterpDup"] = np.stack(interp)
    allp["MeanBaseline"] = np.repeat(mb[None], len(perts), 0)
    rows = []
    tsq = (truth_m ** 2).sum(1)
    for name, pr in allp.items():
        # squared Euclidean distances via |a|^2 + |b|^2 - 2ab (no T x T x G temporary)
        d_e = (pr ** 2).sum(1)[:, None] + tsq[None, :] - 2 * pr @ truth_m.T
        for i, p in enumerate(perts):
            yp, yt, w, tp = pr[i], truth_m[i], weights[i], top[i]
            de_row = d_e[i]
            dm_row = np.abs(pr[i][None] - truth_m).sum(-1)
            others = np.arange(len(perts)) != i
            rows.append({
                "method": name, "perturbation": p,
                "pearson_deltactrl": _pearson(yp, yt),
                "pearson_deltapert_top100": _pearson((yp - mb)[tp], (yt - mb)[tp]),
                "mse": float(np.mean((yp - yt) ** 2)),
                "mae": float(np.mean(np.abs(yp - yt))),
                "wmse": float(np.sum(w / w.sum() * (yp - yt) ** 2)) if w.sum() > 0 else np.nan,
                "wr2_deltactrl": _r2w(yt, yp, w),
                "wr2_deltapert": _r2w(yt - mb, yp - mb, w),
                "nir": float(np.mean(de_row[i] < de_row[others])),
                "pds": float(np.mean(dm_row[i] < dm_row[others])),
            })
    df = pd.DataFrame(rows)
    rel_df = pd.DataFrame({"perturbation": perts, "split_half_r": rel,
                           "n_cells": [ix.by_pert[p].size for p in perts]})
    rel_df["spearman_brown"] = 2 * rel_df.split_half_r / (1 + rel_df.split_half_r)
    rel_df["ceiling_pearson_delta"] = np.sqrt(rel_df.spearman_brown.clip(lower=0))
    return df, rel_df


def drf(df: pd.DataFrame) -> pd.DataFrame:
    """Per-perturbation DRF of each metric (TechDup ceiling vs MeanBaseline/zero-shift floor)."""
    piv = {m: df.pivot_table(index="perturbation", columns="method", values=m) for m in HIGHER}
    out = []
    for m, hb in HIGHER.items():
        t = piv[m]["TechDup"]
        if m in DELTAPERT:
            # negative control for delta-from-perturbed-mean metrics: the mean baseline has zero
            # variance there, so use the control (zero-shift) prediction, as in Miller et al.
            base = piv[m]["NoChange"] if "NoChange" in piv[m] else piv[m]["MeanBaseline"]
        else:
            base = piv[m]["MeanBaseline"]
        if hb:
            v = (t - base) / ((1.0 if "pearson" in m or "r2" in m else 1.0) - base + 1e-6)
        else:
            v = (base - t) / (base + 1e-6)
        out.append(pd.DataFrame({"perturbation": t.index, "metric": m, "drf": np.clip(v.to_numpy(), -1, 1)}))
    return pd.concat(out, ignore_index=True)


def calibrate_task(art: Path, proc: Path, task_dir: str, out: Path, frac_train: float = 0.25,
                   frac_val: float = 0.05) -> None:
    """Run calibrated scoring for every fold file of a task directory."""
    out.mkdir(parents=True, exist_ok=True)
    folds = sorted((art / task_dir).glob("pred_*_s*.npz"))
    by_line: dict[str, list[Path]] = {}
    for f in folds:
        by_line.setdefault(f.stem.removeprefix("pred_").split("_")[0], []).append(f)
    for n, files in by_line.items():
        line = load_line(proc, n)
        idx_all = None
        for f in files:
            tag = f.stem.removeprefix("pred_")
            dest = out / f"calib_{task_dir}_{tag}.parquet"
            if dest.exists():
                continue
            seed = int(tag.split("_s")[-1])
            z = np.load(f)
            perts = list(z["perts"])
            preds = {k: z[k].astype(np.float64) for k in z.files if k != "perts" and "__" not in k and k != "_true"}
            hist = art / task_dir / f"history_{tag}.json"
            import json

            if hist.exists():
                h = json.loads(hist.read_text())
                support = list(h.get("train", [])) + list(h.get("val", []))
            else:
                support = []
            if not support:
                # zero-shot: no target support; use all non-test perturbations of the line only to
                # define the negative control (never seen by any model)
                if idx_all is None:
                    idx_all = index_line(line)
                support = [q for q in idx_all.by_pert if q not in set(perts)]
            df, rel = score_fold(line, perts, preds, support)
            df["target"], df["seed"], df["task"] = n, seed, task_dir
            rel["target"], rel["seed"], rel["task"] = n, seed, task_dir
            df.to_parquet(dest)
            rel.to_parquet(out / f"reliability_{task_dir}_{tag}.parquet")
            print(f"calibrated {task_dir}/{tag}", flush=True)
        del line


def backfill_lineoffset(art: Path, proc: Path, task_dir: str = "t1") -> None:
    """Add the LineOffset baseline to folds trained before it existed (seed 0), scored with the
    same pseudobulk metrics (Pearson delta, discrimination, MAE; DE metrics via AMENDMENT-002)."""
    import json

    from canpert.evaluate import discrimination
    from canpert.reeval import _top
    from canpert.analysis import load_observed

    obs = load_observed(art / "observed")
    for f in sorted((art / task_dir).glob("pred_*_s*.npz")):
        tag = f.stem.removeprefix("pred_")
        z = dict(np.load(f))
        if "LineOffset" in z:
            continue
        n = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        h = json.loads((art / task_dir / f"history_{tag}.json").read_text())
        support = h["train"] + h["val"]
        line = load_line(proc, n)
        ix = index_line(line)
        ref = line.x[ix.ctrl_ref[:N_REF]]
        ref_mean, sd = ref.mean(0), ref.std(0, ddof=1).astype(np.float64)
        perts = list(z["perts"])
        sources = tuple(h["sources"])
        src = {}
        for s in sources:
            sl = load_line(proc, s)
            sx = index_line(sl)
            src[s] = {p: pert_mean(sx, p) - sx.ctrl_mean for p in set(perts) | set(support) if p in sx.by_pert}
            del sl, sx

        def pm(p: str) -> np.ndarray:
            return np.mean([src[s][p] for s in sources if p in src[s]], axis=0)

        offset = np.mean([pert_mean(ix, q) - ix.ctrl_mean - pm(q) for q in support], axis=0)
        # rendered as cells: control-input cells + shift; pseudobulk relative to ref mean
        from canpert.train import input_controls

        shifts = np.stack([pm(p) + offset + input_controls(ix, p, ix.by_pert[p].size, seed).mean(0) - ref_mean
                           for p in perts]).astype(np.float32)
        true = z["_true"].astype(np.float64)
        tsig = z["padj___true"] < 0.05
        tl = z["lfc___true"].astype(np.float64)
        ncell = dict(zip(obs[n]["perts"], obs[n]["n_cells"], strict=True))
        rows = []
        disc = discrimination(shifts.astype(np.float64) + ref_mean, true + ref_mean)
        for i, p in enumerate(perts):
            d = shifts[i].astype(np.float64)
            se = sd * np.sqrt(1.0 / ncell[p] + 1.0 / N_REF)
            zs = np.where(se > 0, d / np.where(se > 0, se, 1), 0.0)
            padj = bh(2 * stats.norm.sf(np.abs(zs)))
            sig = padj < 0.05
            ts = tsig[i]
            nt = int(ts.sum())
            r = {"method": "LineOffset", "perturbation": p, "pearson_delta": _pearson(d, true[i]),
                 "mae": float(np.mean(np.abs(d - true[i]))), "n_de_true": nt, "n_de_pred": int(sig.sum()),
                 "discrimination": float(disc[i]), "target": n, "seed": seed}
            if 0 < nt < ts.size:
                from sklearn.metrics import average_precision_score

                sc = -np.log10(np.clip(padj, 1e-300, 1)) + 1e-9 * np.abs(d)
                r["de_auprc"] = average_precision_score(ts, sc)
                r["de_auprc_base"] = nt / ts.size
                r["de_overlap"] = len(_top(ts, np.abs(tl[i]), nt) & _top(sig, np.abs(d), nt)) / nt
                idx = np.where(ts)[0]
                r["direction"] = float(np.mean(np.sign(d[idx]) == np.sign(tl[i][idx])))
                r["lfc_spearman"] = float(stats.spearmanr(d[idx], tl[i][idx])[0]) if idx.size >= 3 else np.nan
            rows.append(r)
        mpath = art / task_dir / f"metrics_{tag}.parquet"
        m = pd.read_parquet(mpath)
        m = pd.concat([m, pd.DataFrame(rows)], ignore_index=True)
        m.to_parquet(mpath)
        z["LineOffset"] = shifts
        np.savez_compressed(f, **z)
        print(f"LineOffset added to {task_dir}/{tag}", flush=True)
        del line


def run(art: Path, proc: Path, out: Path) -> None:
    backfill_lineoffset(art, proc, "t1")
    for task in ("t1", "t1_meanctrl", "t1_grouped", "t2"):
        if (art / task).exists():
            calibrate_task(art, proc, task, out)
    _ = (LINES, eligible, make_split)


def noise_cancel_check(art: Path, proc: Path, out: Path) -> pd.DataFrame:
    """Review check: do baselines lose accuracy because they inherit the sampling noise of the control
    cells they are rendered on (which CASTER might learn to remove)? Re-render PerturbMean and Linear on
    the noise-free mean of the control-input pool and compare Pearson delta (seed 0, T1)."""
    from canpert.train import input_controls

    rows = []
    for f in sorted((art / "t1").glob("pred_*_s0.npz")):
        n = f.stem.removeprefix("pred_").split("_")[0]
        z = np.load(f)
        line = load_line(proc, n)
        ix = index_line(line)
        ref_mean = line.x[ix.ctrl_ref[:N_REF]].mean(0)
        true = z["_true"].astype(np.float64)
        for i, p in enumerate(z["perts"]):
            m_in = input_controls(ix, str(p), ix.by_pert[str(p)].size, 0).mean(0)
            r = {"target": n, "perturbation": str(p)}
            for me in ("PerturbMean", "Linear", "CASTER", "LineOffset"):
                if me not in z.files:
                    continue
                s_ = z[me][i].astype(np.float64)
                r[me] = _pearson(s_, true[i])
                if me != "CASTER":
                    clean = s_ - (m_in - ref_mean) + (ix.ctrl_mean - ref_mean)
                    r[f"{me}_clean"] = _pearson(clean, true[i])
            rows.append(r)
        del line
    df = pd.DataFrame(rows)
    df.to_csv(out / "noise_cancel_check.csv", index=False)
    return df


def _score_rows(shifts: np.ndarray, z: dict, perts: list[str], sd: np.ndarray, ncell: dict, name: str,
                target: str, seed: int, ref_mean: np.ndarray) -> list[dict]:
    """Pseudobulk metrics (Pearson delta, discrimination, MAE, AMENDMENT-002 DE metrics) for one method."""
    from sklearn.metrics import average_precision_score

    from canpert.evaluate import discrimination
    from canpert.reeval import _top

    true = z["_true"].astype(np.float64)
    tsig = z["padj___true"] < 0.05
    tl = z["lfc___true"].astype(np.float64)
    disc = discrimination(shifts + ref_mean, true + ref_mean)
    rows = []
    for i, p in enumerate(perts):
        d = shifts[i]
        se = sd * np.sqrt(1.0 / ncell[p] + 1.0 / N_REF)
        zs = np.where(se > 0, d / np.where(se > 0, se, 1), 0.0)
        padj = bh(2 * stats.norm.sf(np.abs(zs)))
        sig = padj < 0.05
        ts = tsig[i]
        nt = int(ts.sum())
        r = {"method": name, "perturbation": p, "pearson_delta": _pearson(d, true[i]),
             "mae": float(np.mean(np.abs(d - true[i]))), "n_de_true": nt, "n_de_pred": int(sig.sum()),
             "discrimination": float(disc[i]), "target": target, "seed": seed}
        if 0 < nt < ts.size:
            sc = -np.log10(np.clip(padj, 1e-300, 1)) + 1e-9 * np.abs(d)
            r["de_auprc"] = average_precision_score(ts, sc)
            r["de_auprc_base"] = nt / ts.size
            r["de_overlap"] = len(_top(ts, np.abs(tl[i]), nt) & _top(sig, np.abs(d), nt)) / nt
            idx = np.where(ts)[0]
            r["direction"] = float(np.mean(np.sign(d[idx]) == np.sign(tl[i][idx])))
            r["lfc_spearman"] = float(stats.spearmanr(d[idx], tl[i][idx])[0]) if idx.size >= 3 else np.nan
        rows.append(r)
    return rows


CLEAN_BASES = ("PerturbMean", "Linear", "LineOffset", "ContextMean", "CASTER", "CASTER_wrongctrl")


def add_clean_variants(art: Path, proc: Path, task_dir: str) -> None:
    """Noise-free rendering (review check): every method's predicted pseudobulk shift contains the sampling
    noise of the control cells it was rendered on (mean of n input cells minus the reference mean). Remove
    it: clean = stored - (mean(input) - ref_mean) + (mean(input pool) - ref_mean). For baselines this is
    their shift on noise-free controls; for CASTER it treats the model's output relative to its own input
    as the effect. Adds methods '<name>_clean' to pred_*.npz and metrics_*.parquet."""
    from canpert.analysis import load_observed
    from canpert.train import input_controls

    obs = load_observed(art / "observed")
    files = sorted((art / task_dir).glob("pred_*.npz"))
    by_line: dict[str, list[Path]] = {}
    for f in files:
        by_line.setdefault(f.stem.removeprefix("pred_").split("_")[0], []).append(f)
    for n, fl in by_line.items():
        line = load_line(proc, n)
        ix = index_line(line)
        ref = line.x[ix.ctrl_ref[:N_REF]]
        ref_mean, sd = ref.mean(0).astype(np.float64), ref.std(0, ddof=1).astype(np.float64)
        ncell = dict(zip(obs[n]["perts"], obs[n]["n_cells"], strict=True))
        for f in fl:
            tag = f.stem.removeprefix("pred_")
            z = dict(np.load(f))
            todo = [b for b in CLEAN_BASES if b in z and f"{b}_clean" not in z]
            if not todo:
                continue
            seed = int(tag.split("_s")[-1]) if "_s" in tag and tag.split("_s")[-1].isdigit() else 0
            perts = [str(p) for p in z["perts"]]
            noise = np.stack([input_controls(ix, p, ix.by_pert[p].size, seed).mean(0) - ix.ctrl_mean
                              for p in perts]).astype(np.float64)
            rows = []
            for b in todo:
                clean = z[b].astype(np.float64) - noise
                z[f"{b}_clean"] = clean.astype(np.float32)
                rows += _score_rows(clean, z, perts, sd, ncell, f"{b}_clean", n, seed, ref_mean)
            mpath = art / task_dir / f"metrics_{tag}.parquet"
            m = pd.read_parquet(mpath)
            extra = pd.DataFrame(rows)
            for col in ("source",):
                if col in m:
                    extra[col] = m[col].iloc[0]
            m = pd.concat([m, extra], ignore_index=True)
            m.to_parquet(mpath)
            np.savez_compressed(f, **z)
            print(f"clean variants added: {task_dir}/{tag} ({', '.join(todo)})", flush=True)
        del line
