"""Splits, baselines and CASTER training/prediction.

Every predictor returns, for each test perturbation, a *set of predicted cells* of the same
size as the observed set, so that single-cell DE metrics are computed identically for all
methods. Mean-shift baselines produce cells as (input control cells + predicted shift),
which is the fair single-cell rendering of a pseudobulk prediction.
"""

from __future__ import annotations

import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from canpert.data import LINES, LineData
from canpert.ingest import CONTROL_LABEL
from canpert.model import Caster, CasterConfig, caster_loss

SET_SIZE = 32


@dataclass
class Split:
    """One fold: held-out line, and its support/val/test perturbations."""

    target: str
    train: list[str]
    val: list[str]
    test: list[str]
    seed: int
    sources: tuple[str, ...] = field(default_factory=tuple)


def eligible(idx: dict[str, Indexed], target: str, sources: tuple[str, ...]) -> list[str]:
    """Perturbations observed in the target line and in at least one source line."""
    return sorted(p for p in idx[target].by_pert if any(p in idx[s].by_pert for s in sources))


def make_split(
    panel: list[str], target: str, frac_train: float, frac_val: float, seed: int,
    sources: tuple[str, ...] | None = None,
) -> Split:
    """Shuffle the target's eligible perturbations and cut them into train/val/test."""
    rng = np.random.default_rng(seed)
    perm = [str(x) for x in rng.permutation(sorted(panel))]
    n_tr, n_va = int(round(frac_train * len(perm))), int(round(frac_val * len(perm)))
    src = tuple(n for n in LINES if n != target) if sources is None else sources
    return Split(target, perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:], seed, src)


@dataclass
class Indexed:
    """Per-line index: cell indices by perturbation, and control input/reference pools."""

    line: LineData
    by_pert: dict[str, np.ndarray]
    ctrl_input: np.ndarray
    ctrl_ref: np.ndarray
    ctrl_mean: np.ndarray


def index_line(line: LineData, seed: int = 0) -> Indexed:
    """Split controls 50/50 into model-input and DE-reference pools (disjoint)."""
    by: dict[str, list[int]] = {}
    for i, p in enumerate(line.pert):
        by.setdefault(str(p), []).append(i)
    arr = {k: np.array(v) for k, v in by.items()}
    ctrl = np.random.default_rng(seed).permutation(arr.pop(CONTROL_LABEL))
    half = ctrl.size // 2
    c_in, c_ref = np.sort(ctrl[:half]), np.sort(ctrl[half:])
    return Indexed(line, arr, c_in, c_ref, line.x[c_in].mean(0))


def pert_mean(ix: Indexed, p: str) -> np.ndarray:
    """Observed pseudobulk of perturbation p in a line."""
    return ix.line.x[ix.by_pert[p]].mean(0)


# ----------------------------------------------------------------------------- predictions
Predictor = Callable[[str, np.ndarray], np.ndarray]


def input_controls(ix: Indexed, p: str, n: int, seed: int) -> np.ndarray:
    """Deterministic control input set for perturbation p (shared by all methods)."""
    rng = np.random.default_rng(zlib.crc32(f"{p}|{seed}|{ix.line.name}".encode()))
    return ix.line.x[rng.choice(ix.ctrl_input, size=n, replace=n > ix.ctrl_input.size)]


def shift_predictor(deltas: dict[str, np.ndarray]) -> Predictor:
    """Cells = control input cells + a per-perturbation shift."""
    def predict(p: str, ctrl: np.ndarray) -> np.ndarray:
        return ctrl + deltas[p][None, :]
    return predict


def baseline_deltas(
    idx: dict[str, Indexed], split: Split
) -> dict[str, dict[str, np.ndarray]]:
    """ContextMean, PerturbMean and Linear-transfer shifts for the split's test set."""
    tgt = idx[split.target]
    support = split.train + split.val
    out: dict[str, dict[str, np.ndarray]] = {}
    src_delta = {
        s: {p: pert_mean(idx[s], p) - idx[s].ctrl_mean for p in split.test + support
            if p in idx[s].by_pert}
        for s in split.sources
    }

    def pm(p: str) -> np.ndarray:
        return np.mean([src_delta[s][p] for s in split.sources if p in src_delta[s]], axis=0)

    out["PerturbMean"] = {p: pm(p) for p in split.test}
    out["NoChange"] = {p: np.zeros_like(tgt.ctrl_mean) for p in split.test}
    if support:
        ctx = np.mean([pert_mean(tgt, p) for p in support], axis=0) - tgt.ctrl_mean
        out["ContextMean"] = {p: ctx for p in split.test}
        # Linear transfer: ridge from PCA of source-line effects to target-line effect. A
        # source line that did not observe p contributes the mean over the lines that did.
        feats = {p: np.concatenate([src_delta[s].get(p, pm(p)) for s in split.sources])
                 for p in split.test + support}
        pca = PCA(n_components=min(64, len(support) - 1), random_state=0)
        pca.fit(np.stack([feats[p] for p in split.test + support]))  # source data only
        xs = pca.transform(np.stack([feats[p] for p in support]))
        ys = np.stack([pert_mean(tgt, p) - tgt.ctrl_mean for p in support])
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xs, ys)
        xt = pca.transform(np.stack([feats[p] for p in split.test]))
        pred = ridge.predict(xt)
        out["Linear"] = {p: pred[i] for i, p in enumerate(split.test)}
        # LineOffset: transfer prior plus one constant offset per target line, estimated on the
        # support perturbations (tests whether gains are "one shift per line")
        offset = np.mean([pert_mean(tgt, q) - tgt.ctrl_mean - pm(q) for q in support], axis=0)
        out["LineOffset"] = {p: pm(p) + offset for p in split.test}
    return out


# ----------------------------------------------------------------------------- CASTER
@dataclass
class TrainResult:
    """Trained model plus its training trace."""

    model: Caster
    pert_index: dict[str, int]
    history: list[dict[str, float]]
    seconds: float
    bank: PriorBank


class PriorBank:
    """Transfer priors d_src(line, p) and the cells each item is trained against.

    - target line: mean shift of p over the source lines that observed p (never the target);
    - a source line l: mean shift over the *other* source lines that observed p
      (leave-one-context-out);
    - a source item with no other observing source line (always the case in single-source
      splits): the prior is computed from a random half of p's cells in that line and the
      model is trained against the other half, so a prior and its training target never
      share a cell.
    """

    def __init__(self, idx: dict[str, Indexed], split: Split, seed: int = 0) -> None:
        self.idx, self.split = idx, split
        rng = np.random.default_rng(seed + 777)
        self.delta = {s: {p: pert_mean(idx[s], p) - idx[s].ctrl_mean for p in idx[s].by_pert}
                      for s in split.sources}
        self.half: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for s in split.sources:
            for p in sorted(idx[s].by_pert):
                if not any(p in self.delta[o] for o in split.sources if o != s):
                    c = rng.permutation(idx[s].by_pert[p])
                    self.half[(s, p)] = (c[: c.size // 2], c[c.size // 2:])

    def prior(self, line: str, p: str) -> np.ndarray:
        """Transfer prior for item (line, p)."""
        if line == self.split.target:
            return np.mean([self.delta[s][p] for s in self.split.sources if p in self.delta[s]], axis=0)
        if (line, p) in self.half:
            ix = self.idx[line]
            return ix.line.x[self.half[(line, p)][0]].mean(0) - ix.ctrl_mean
        others = [s for s in self.split.sources if s != line and p in self.delta[s]]
        return np.mean([self.delta[s][p] for s in others], axis=0)

    def target_cells(self, line: str, p: str) -> np.ndarray:
        """Cells an item is trained/validated against."""
        if (line, p) in self.half:
            return self.half[(line, p)][1]
        return self.idx[line].by_pert[p]


def _training_items(idx: dict[str, Indexed], split: Split) -> list[tuple[str, str]]:
    items = [(s, p) for s in split.sources for p in idx[s].by_pert]
    items += [(split.target, p) for p in split.train]
    return items


def _sample_set(ix: Indexed, cells: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return ix.line.x[rng.choice(cells, size=size, replace=cells.size < size)]


def _ctrl_set(ix: Indexed, size: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    """Control input set: real cells, or (``mean``) the line's mean control profile repeated,
    which carries line identity but no cell-to-cell information."""
    if mode == "mean":
        return np.repeat(ix.ctrl_mean[None, :], size, axis=0).astype(np.float32)
    return _sample_set(ix, ix.ctrl_input, size, rng)


def train_caster(
    idx: dict[str, Indexed], split: Split, panel: list[str], *, set_size: int = SET_SIZE,
    epochs: int = 40, batch: int = 32, lr: float = 3e-4, seed: int = 0,
    attention: bool = True, film: bool = True, prior: bool = True, pert_id: bool = False,
    ctrl_mode: str = "cells", max_seconds: float = 3600.0,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train CASTER on the split; select the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pert_index = {p: i for i, p in enumerate(panel)}
    n_genes = idx[split.target].line.x.shape[1]
    cfg = CasterConfig(n_genes=n_genes, n_perts=len(panel), attention=attention, film=film,
                       prior=prior, pert_id=pert_id)
    model = Caster(cfg)
    model.ctrl_mode = ctrl_mode  # type: ignore[assignment]
    bank = PriorBank(idx, split, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    items = _training_items(idx, split)
    val_items = [(split.target, p) for p in split.val]
    if not val_items:
        # zero-shot: select the checkpoint on held-out source-line perturbations (5%), so no
        # target-line perturbed cell is ever used
        hold = np.random.default_rng(seed + 999).permutation(len(items))[: max(1, len(items) // 20)]
        hold_set = set(hold.tolist())
        val_items = [items[i] for i in sorted(hold_set)]
        items = [it for i, it in enumerate(items) if i not in hold_set]
    steps_per_epoch = max(1, len(items) // batch)
    total = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total))

    def lr_lambda(step: int) -> float:
        # linear warm-up (5%) then cosine decay to 5% of the peak
        if step < warm:
            return (step + 1) / warm
        frac = (step - warm) / max(1, total - warm)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(frac, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # the pseudobulk-anchor term scales with set size; for S=1 only the mean term is informative
    best, best_state, history = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(items))
        tot = 0.0
        for b in range(steps_per_epoch):
            chunk = [items[i] for i in order[b * batch:(b + 1) * batch]]
            xc = np.stack([_ctrl_set(idx[s], set_size, rng, ctrl_mode) for s, _ in chunk])
            xt = np.stack([_sample_set(idx[s], bank.target_cells(s, p), set_size, rng)
                           for s, p in chunk])
            pt = torch.tensor([pert_index[p] for _, p in chunk])
            pr = torch.from_numpy(np.stack([bank.prior(s, p) for s, p in chunk]).astype(np.float32))
            pred = model(torch.from_numpy(xc), pt, pr)
            loss = caster_loss(pred, torch.from_numpy(xt), cfg.mean_weight)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        val = evaluate_loss(model, idx, bank, val_items, pert_index, SET_SIZE, seed)
        history.append({"epoch": ep, "train": tot / steps_per_epoch, "val": val,
                        "seconds": time.time() - t0})
        log(f"  ep {ep:02d} train {tot / steps_per_epoch:.4f} val {val:.4f} ({time.time() - t0:.0f}s)")
        if val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if time.time() - t0 > max_seconds:
            log("  time budget reached")
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(model, pert_index, history, time.time() - t0, bank)


@torch.no_grad()
def evaluate_loss(
    model: Caster, idx: dict[str, Indexed], bank: PriorBank, items: list[tuple[str, str]],
    pert_index: dict[str, int], set_size: int, seed: int,
) -> float:
    """Validation loss on fixed control/target draws (energy MMD + mean term)."""
    model.eval()
    rng = np.random.default_rng(seed + 12345)
    losses = []
    for i in range(0, len(items), 64):
        chunk = items[i:i + 64]
        xc = np.stack([_ctrl_set(idx[s], set_size, rng, getattr(model, "ctrl_mode", "cells"))
                       for s, _ in chunk])
        xt = np.stack([_sample_set(idx[s], bank.target_cells(s, p), set_size, rng) for s, p in chunk])
        pr = torch.from_numpy(np.stack([bank.prior(s, p) for s, p in chunk]).astype(np.float32))
        pred = model(torch.from_numpy(xc), torch.tensor([pert_index[p] for _, p in chunk]), pr)
        losses.append(float(caster_loss(pred, torch.from_numpy(xt), model.cfg.mean_weight)) * len(chunk))
    return sum(losses) / len(items)


def caster_predictor(res: TrainResult, set_size: int = SET_SIZE) -> Predictor:
    """Predict any number of cells by chunking the control input into sets."""
    model = res.model.eval()

    mode = getattr(model, "ctrl_mode", "cells")
    tmean = res.bank.idx[res.bank.split.target].ctrl_mean

    @torch.no_grad()
    def predict(p: str, ctrl: np.ndarray) -> np.ndarray:
        n = ctrl.shape[0]
        if mode == "mean":
            base = ctrl
            ctrl = np.repeat(tmean[None, :], n, axis=0).astype(np.float32)
            out_ = _run(p, ctrl, n)
            return base + (out_ - ctrl)
        return _run(p, ctrl, n)

    @torch.no_grad()
    def _run(p: str, ctrl: np.ndarray, n: int) -> np.ndarray:
        reps = int(np.ceil(n / set_size))
        pad = np.concatenate([ctrl, ctrl[: reps * set_size - n]]) if reps * set_size > n else ctrl
        sets = torch.from_numpy(pad.reshape(reps, set_size, -1).astype(np.float32))
        prior = res.bank.prior(res.bank.split.target, p)
        pr = torch.from_numpy(np.repeat(prior[None], reps, 0).astype(np.float32))
        out = model(sets, torch.full((reps,), res.pert_index[p], dtype=torch.long), pr)
        return out.reshape(reps * set_size, -1).numpy()[:n]

    return predict


def wrong_ctrl_predictor(res: TrainResult, idx: dict[str, Indexed], other: str, seed: int) -> Predictor:
    """CASTER fed another line's control cells. The predicted *shift* (prediction minus its input)
    is added to the target's own control cells, so only the model's use of the control set
    changes, not the baseline expression level."""
    base = caster_predictor(res)

    def predict(p: str, ctrl: np.ndarray) -> np.ndarray:
        wc = input_controls(idx[other], p, ctrl.shape[0], seed)
        shift = base(p, wc).mean(0) - wc.mean(0)
        return ctrl + shift[None, :]

    return predict


def make_grouped_split(perts: list[str], groups: dict[str, int], target: str, frac_train: float,
                       frac_val: float, seed: int) -> Split:
    """Assign whole groups (e.g. protein complexes) to train, val or test, so that no group is
    split across training and test (review item 7)."""
    rng = np.random.default_rng(seed)
    by: dict[int, list[str]] = {}
    for p in sorted(perts):
        by.setdefault(groups.get(p, -abs(hash(p)) % 10**9 - 1), []).append(p)
    keys = [k for k in by]
    keys = [keys[i] for i in rng.permutation(len(keys))]
    n = len(perts)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for k in keys:
        if len(train) < frac_train * n:
            train += by[k]
        elif len(val) < frac_val * n:
            val += by[k]
        else:
            test += by[k]
    src = tuple(x for x in LINES if x != target)
    return Split(target, train, val, test, seed, src)
