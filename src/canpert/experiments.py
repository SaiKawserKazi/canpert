"""Experiment runner: benchmark tasks, ablations and transfer matrix.

Outputs (under ``artifacts/<task>/``): per-perturbation metrics (``metrics.parquet``),
predicted pseudobulk shifts per method (``pred_<target>_s<seed>.npz``), and training
histories (``history.json``). Everything downstream (figures, statistics, paper numbers)
reads only these files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from canpert.data import LINES, LineData
from canpert.evaluate import DE, de_test, discrimination, per_perturbation
from canpert.train import (
    Indexed, Predictor, Split, TrainResult, baseline_deltas, caster_predictor, eligible,
    index_line, input_controls, make_grouped_split, make_split, shift_predictor, train_caster,
    wrong_ctrl_predictor,
)

N_REF = 500
SKIP_PRED_WILCOXON = os.environ.get("CANPERT_SKIP_PRED_WILCOXON") == "1"


class TrueDECache:
    """Observed DE per (line, perturbation) against the fixed reference controls."""

    def __init__(self, idx: dict[str, Indexed]) -> None:
        self.idx = idx
        self.ref = {n: ix.line.x[ix.ctrl_ref[:N_REF]] for n, ix in idx.items()}
        self.cache: dict[tuple[str, str], DE] = {}

    def get(self, line: str, p: str) -> DE:
        """Return (and memoise) the observed DE of p in line."""
        key = (line, p)
        if key not in self.cache:
            cells = self.idx[line].line.x[self.idx[line].by_pert[p]]
            self.cache[key] = DE(*de_test(cells, self.ref[line]))
        return self.cache[key]


def evaluate_methods(
    idx: dict[str, Indexed], split: Split, methods: dict[str, Predictor], truth: TrueDECache,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-perturbation metrics for every method + predicted pseudobulk shifts."""
    tgt = idx[split.target]
    ref = truth.ref[split.target]
    ref_mean = ref.mean(0)
    true_pb = np.stack([tgt.line.x[tgt.by_pert[p]].mean(0) for p in split.test])
    rows, shifts = [], {}
    for name, predict in methods.items():
        pred_pb, n_pred, lfcs, padjs = [], [], [], []
        for i, p in enumerate(split.test):
            n = tgt.by_pert[p].size
            ctrl = input_controls(tgt, p, n, split.seed)
            cells = predict(p, ctrl)
            if SKIP_PRED_WILCOXON:
                # AMENDMENT-002: predicted-side DE is recomputed by reeval; skip the invalid test
                pd_de = DE(np.ones(ref.shape[1]), np.zeros(ref.shape[1]))
            else:
                pd_de = DE(*de_test(cells, ref))
            lfcs.append(pd_de.lfc)
            padjs.append(pd_de.padj)
            m = per_perturbation(truth.get(split.target, p), pd_de, cells.mean(0) - ref_mean,
                                 true_pb[i] - ref_mean)
            m.update(method=name, perturbation=p)
            rows.append(m)
            pred_pb.append(cells.mean(0))
            n_pred.append(m["n_de_pred"])
        pred_pb_arr = np.stack(pred_pb)
        disc = discrimination(pred_pb_arr, true_pb)
        for j in range(len(split.test)):
            rows[-len(split.test) + j]["discrimination"] = float(disc[j])
        shifts[name] = (pred_pb_arr - ref_mean).astype(np.float32)
        shifts[f"lfc__{name}"] = np.stack(lfcs).astype(np.float16)
        shifts[f"padj__{name}"] = np.stack(padjs).astype(np.float32)
    shifts["_true"] = (true_pb - ref_mean).astype(np.float32)
    shifts["lfc___true"] = np.stack([truth.get(split.target, p).lfc for p in split.test]).astype(np.float16)
    shifts["padj___true"] = np.stack([truth.get(split.target, p).padj for p in split.test]).astype(np.float32)
    df = pd.DataFrame(rows)
    df["target"], df["seed"] = split.target, split.seed
    return df, shifts


def _save(out: Path, split: Split, df: pd.DataFrame, shifts: dict[str, np.ndarray],
          res: TrainResult | None, tag: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if res is not None:
        import torch

        torch.save(res.model.state_dict(), out / f"model_{tag}.pt")
    if SKIP_PRED_WILCOXON:
        df["pred_wilcoxon_skipped"] = True
    df.to_parquet(out / f"metrics_{tag}.parquet")
    np.savez_compressed(out / f"pred_{tag}.npz", perts=np.array(split.test), **shifts)
    if res is not None:
        (out / f"history_{tag}.json").write_text(json.dumps(
            {"history": res.history, "seconds": res.seconds, "train": split.train,
             "val": split.val, "sources": list(split.sources)}))


def run_context_task(
    lines: dict[str, LineData], panel: list[str], out: Path, *, frac_train: float,
    frac_val: float, seeds: tuple[int, ...] = (0,), targets: tuple[str, ...] = LINES,
    epochs: int = 40, log=print,  # noqa: ANN001
    ctrl_mode: str = "cells", groups: dict[str, int] | None = None,
) -> None:
    """Underrepresented (frac_train>0) or zero-shot (frac_train=0) context generalisation.

    ``ctrl_mode="mean"`` trains the line-identity-only variant; ``groups`` makes whole groups
    (complexes) fall on one side of the split."""
    idx = {n: index_line(line) for n, line in lines.items()}
    truth = TrueDECache(idx)
    for seed in seeds:
        for target in targets:
            tag = f"{target}_s{seed}"
            if (out / f"metrics_{tag}.parquet").exists():
                log(f"skip {tag}")
                continue
            sources = tuple(n for n in LINES if n != target)
            el = eligible(idx, target, sources)
            split = (make_grouped_split(el, groups, target, frac_train, frac_val, seed) if groups
                     else make_split(el, target, frac_train, frac_val, seed))
            log(f"== {out.name} {tag}: train {len(split.train)} val {len(split.val)} test {len(split.test)}")
            t0 = time.time()
            res = train_caster(idx, split, panel, epochs=epochs, seed=seed, log=log,
                               ctrl_mode=ctrl_mode)
            methods: dict[str, Predictor] = {"CASTER": caster_predictor(res)}
            if ctrl_mode == "cells":
                # review item 4: the same model given another line's control cells
                other = LINES[(LINES.index(target) + 1) % len(LINES)]
                methods["CASTER_wrongctrl"] = wrong_ctrl_predictor(res, idx, other, seed)
            for name, deltas in baseline_deltas(idx, split).items():
                methods[name] = shift_predictor(deltas)
            df, shifts = evaluate_methods(idx, split, methods, truth)
            _save(out, split, df, shifts, res, tag)
            log(f"   done {tag} in {time.time() - t0:.0f}s\n"
                + df.groupby("method")[["pearson_delta", "discrimination", "de_auprc"]].mean().round(3).to_string())


def run_ablation(
    lines: dict[str, LineData], panel: list[str], out: Path, target: str = "k562",
    epochs: int = 30, log=print,  # noqa: ANN001
    seed: int = 0, only: tuple[str, ...] | None = None,
) -> None:
    """Set size / attention / FiLM ablations on one underrepresented-context fold."""
    idx = {n: index_line(line) for n, line in lines.items()}
    truth = TrueDECache(idx)
    sources = tuple(n for n in LINES if n != target)
    split = make_split(eligible(idx, target, sources), target, 0.25, 0.05, seed)
    # (tag, set size, attention, FiLM, transfer prior, one-hot perturbation ID)
    variants = [("S1", 1, True, True, True, False), ("S8", 8, True, True, True, False),
                ("S32", 32, True, True, True, False), ("S32_noattn", 32, False, True, True, False),
                ("S32_nofilm", 32, True, False, True, False),
                ("S32_noprior", 32, True, True, False, False),
                ("S32_pertid", 32, True, True, True, True)]
    for tag, s, attn, film, prior, pert_id in variants:
        if only is not None and tag not in only:
            continue
        if seed:
            tag = f"{tag}_s{seed}"
        if (out / f"metrics_{tag}.parquet").exists():
            continue
        log(f"== ablation {tag}")
        # keep cells seen per step constant: batch * S = 1024
        res = train_caster(idx, split, panel, set_size=s, batch=max(1024 // s, 32) if s < 32 else 32,
                           epochs=epochs, attention=attn, film=film, prior=prior,
                           pert_id=pert_id, log=log, seed=seed)
        df, shifts = evaluate_methods(idx, split, {tag: caster_predictor(res, s)}, truth)
        _save(out, split, df, shifts, res, tag)


def run_transfer_matrix(
    lines: dict[str, LineData], panel: list[str], out: Path, epochs: int = 25,
    log=print,  # noqa: ANN001
) -> None:
    """Single-source zero-shot transfer for every ordered pair of lines."""
    idx = {n: index_line(line) for n, line in lines.items()}
    truth = TrueDECache(idx)
    part, parts = (int(x) for x in os.environ.get("CANPERT_PART", "0/1").split("/"))
    k = -1
    for src in LINES:
        for tgt in LINES:
            if src == tgt:
                continue
            k += 1
            if k % parts != part:
                continue
            tag = f"{src}_to_{tgt}"
            if (out / f"metrics_{tag}.parquet").exists():
                continue
            split = make_split(eligible(idx, tgt, (src,)), tgt, 0.0, 0.0, 0, sources=(src,))
            log(f"== transfer {tag}")
            res = train_caster(idx, split, panel, epochs=epochs, log=log)
            methods = {"CASTER": caster_predictor(res)}
            for name, deltas in baseline_deltas(idx, split).items():
                methods[name] = shift_predictor(deltas)
            df, shifts = evaluate_methods(idx, split, methods, truth)
            df["source"] = src
            _save(out, split, df, shifts, res, tag)


def observed_effects(lines: dict[str, LineData], panel: list[str], out: Path) -> None:
    """Observed DE for every perturbation in every line (context-specificity analyses)."""
    idx = {n: index_line(line) for n, line in lines.items()}
    truth = TrueDECache(idx)
    out.mkdir(parents=True, exist_ok=True)
    for n in LINES:
        path = out / f"observed_{n}.npz"
        if path.exists():
            continue
        perts = sorted(idx[n].by_pert)
        ref_mean = truth.ref[n].mean(0)
        delta = np.stack([idx[n].line.x[idx[n].by_pert[p]].mean(0) - ref_mean for p in perts])
        des = [truth.get(n, p) for p in perts]
        np.savez_compressed(path, perts=np.array(perts), delta=delta.astype(np.float32),
                            padj=np.stack([d.padj for d in des]).astype(np.float32),
                            lfc=np.stack([d.lfc for d in des]).astype(np.float32),
                            n_cells=np.array([idx[n].by_pert[p].size for p in perts]))
