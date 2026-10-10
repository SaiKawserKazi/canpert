"""Build the analysis dataset from fetched single cells (Arc Replogle-Nadig concatenation).

Source: ``arcinstitute/State-Replogle-Filtered/replogle_concat.h5ad`` = Replogle et al.
2022 (K562 essential, RPE1) + Nadig et al. 2025 (HepG2, Jurkat) CRISPRi Perturb-seq,
library-size normalised and log1p-transformed, restricted to the 6,546 genes measured in all
four lines, and knockdown-filtered by Arc (perturbation median residual <= 0.30, cell
residual <= 0.50, >= 30 cells) exactly as described in the State paper's methods.

Our steps:
1. load the stratified subsample (<= 32 cells per perturbation, 3,000 controls per line);
2. choose the modelled gene space: the top N_HVG genes by bin-normalised dispersion computed
   on **control cells only**, pooled over lines, after removing every gene that is a
   perturbation target in any line. No perturbed cell informs gene selection, and the
   trivial on-target knockdown signal cannot inflate any metric;
3. save one array per line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from canpert.ingest import CONTROL_LABEL

LINES: tuple[str, ...] = ("k562", "hepg2", "jurkat", "rpe1")
CANCER_LINES: tuple[str, ...] = ("k562", "hepg2", "jurkat")
N_HVG = 2000
MIN_CELLS = 16


@dataclass
class LineData:
    """Log-normalised expression for one cell line on the modelled gene space."""

    name: str
    x: np.ndarray  # (cells, genes) float32 log1p-normalised
    pert: np.ndarray  # (cells,) perturbation label; CONTROL_LABEL for controls
    gem: np.ndarray  # (cells,) gem group (batch)


def _shard_paths(d: Path) -> list[Path]:
    # first pass (shard_) + exact-row pass (shardB_)
    return [p for p in sorted(d.glob("shard*.npz")) if ".tmp" not in p.name]


def _read_shard(path: Path) -> tuple[np.ndarray, sp.csr_matrix]:
    z = np.load(path)
    m = sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int32), z["indptr"]),
                      shape=tuple(z["shape"]))
    return z["rows"], m


def stream_rows(d: Path, wanted: np.ndarray, cols: np.ndarray | None) -> tuple[np.ndarray, sp.csr_matrix]:
    """Collect the wanted file rows (each once) from all shards, keeping only ``cols``.

    Memory-bounded: one shard is expanded at a time. The two fetch passes overlap; a row fetched
    twice is byte-identical (verified), so the first copy is kept.
    """
    want = set(wanted.tolist())
    seen: set[int] = set()
    ids, mats = [], []
    for path in _shard_paths(d):
        rows, m = _read_shard(path)
        keep = np.array([int(r) in want and int(r) not in seen for r in rows])
        if not keep.any():
            continue
        sub = m[np.where(keep)[0]]
        if cols is not None:
            sub = sub[:, cols]
        ids.append(rows[keep])
        mats.append(sub.tocsr())
        seen.update(int(r) for r in rows[keep])
    allrows = np.concatenate(ids)
    order = np.argsort(allrows)
    return allrows[order], sp.vstack(mats).tocsr()[order]


def control_hvg(x_ctrl: sp.csr_matrix, exclude: np.ndarray, n: int) -> np.ndarray:
    """Seurat-v1-style bin-normalised dispersion on control cells; returns sorted indices."""
    mean = np.asarray(x_ctrl.mean(axis=0)).ravel()
    sq = np.asarray(x_ctrl.multiply(x_ctrl).mean(axis=0)).ravel()
    var = np.maximum(sq - mean**2, 0)
    ok = (mean > 0.02) & ~exclude
    disp = np.full(mean.shape, -np.inf)
    disp[ok] = np.log(var[ok] / mean[ok] + 1e-12)
    fin = np.where(ok)[0]
    bins = pd.qcut(mean[fin], 20, labels=False, duplicates="drop")
    z = np.full(mean.shape, -np.inf)
    for b in np.unique(bins):
        sel = fin[bins == b]
        d = disp[sel]
        z[sel] = (d - d.mean()) / (d.std() + 1e-9)
    return np.sort(np.argsort(-z)[:n])


def build(root: Path, out: Path) -> dict[str, object]:
    """Build processed per-line arrays and a manifest; returns the audit summary."""
    out.mkdir(parents=True, exist_ok=True)
    d = root / "hf"
    obs = pd.read_parquet(d / "obs.parquet")
    var = pd.read_parquet(d / "var.parquet")
    expected = np.sort(np.load(d / "selected_rows.npy"))
    genes = var["gene_name"].to_numpy().astype(str)
    sel_obs = obs.iloc[expected]
    targets = set(sel_obs["gene"]) - {CONTROL_LABEL}
    exclude = np.isin(genes, list(targets))
    # pass 1: control cells only, all genes -> gene selection
    ctrl_rows = expected[(sel_obs["gene"] == CONTROL_LABEL).to_numpy()]
    got_ctrl, x_ctrl = stream_rows(d / "shards", ctrl_rows, None)
    assert np.array_equal(got_ctrl, ctrl_rows), "control rows incomplete"
    hvg = control_hvg(x_ctrl, exclude, N_HVG)
    del x_ctrl
    # pass 2: every selected cell, selected genes only
    rows, x = stream_rows(d / "shards", expected, hvg)
    missing = np.setdiff1d(expected, rows)
    assert missing.size == 0, f"{missing.size} selected rows not fetched"
    assert np.unique(rows).size == rows.size
    o = obs.iloc[rows].reset_index(drop=True)
    audit: dict[str, object] = {"source": "arcinstitute/State-Replogle-Filtered (replogle_concat.h5ad)",
                                "n_cells_fetched": int(rows.size), "n_genes_measured": int(genes.size),
                                "n_target_genes_excluded": int(exclude.sum()), "n_hvg": int(hvg.size)}
    pd.DataFrame({"gene_name": genes[hvg]}).to_csv(out / "genes.csv", index=False)
    perts_by_line: dict[str, list[str]] = {}
    for n in LINES:
        m = (o["cell_line"] == n).to_numpy()
        labels = o.loc[m, "gene"].to_numpy().astype(str)
        counts = pd.Series(labels).value_counts()
        keep_p = [p for p, c in counts.items() if p != CONTROL_LABEL and c >= MIN_CELLS]
        sel = np.isin(labels, keep_p) | (labels == CONTROL_LABEL)
        xm = x[np.where(m)[0][sel]].toarray().astype(np.float32)
        np.savez_compressed(out / f"{n}.npz", x=xm, pert=labels[sel],
                            gem=o.loc[m, "gem_group"].to_numpy()[sel].astype(np.int64))
        perts_by_line[n] = sorted(keep_p)
        audit[f"{n}_cells"] = int(sel.sum())
        audit[f"{n}_controls"] = int((labels[sel] == CONTROL_LABEL).sum())
        audit[f"{n}_perturbations"] = len(keep_p)
        audit[f"{n}_median_cells_per_perturbation"] = float(counts[keep_p].median())
    allp = sorted(set().union(*perts_by_line.values()))
    audit["n_perturbations_union"] = len(allp)
    audit["n_perturbations_all_four"] = len(set.intersection(*map(set, perts_by_line.values())))
    audit["n_perturbations_all_three_cancer"] = len(
        set.intersection(*(set(perts_by_line[n]) for n in CANCER_LINES)))
    (out / "perturbations.json").write_text(json.dumps(perts_by_line))
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def load_processed(out: Path) -> tuple[dict[str, LineData], list[str], pd.DataFrame]:
    """Load processed lines, the union perturbation list and the gene table."""
    lines = {}
    for n in LINES:
        z = np.load(out / f"{n}.npz", allow_pickle=False)
        lines[n] = LineData(n, z["x"], z["pert"], z["gem"])
    per_line = json.loads((out / "perturbations.json").read_text())
    panel = sorted(set().union(*per_line.values()))
    genes = pd.read_csv(out / "genes.csv")
    return lines, panel, genes
