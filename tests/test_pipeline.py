"""End-to-end and unit tests on synthetic data with a known ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from canpert.data import LINES, LineData
from canpert.evaluate import bh, de_test, discrimination
from canpert.experiments import run_context_task
from canpert.ingest import CONTROL_LABEL, _ranges
from canpert.model import Caster, CasterConfig, energy_mmd
from canpert.train import index_line, input_controls, make_split


def synthetic(n_perts: int = 20, genes: int = 30, seed: int = 0) -> tuple[dict[str, LineData], list[str]]:
    """Four lines; each perturbation has a shared effect plus a line-specific effect."""
    rng = np.random.default_rng(seed)
    panel = [f"G{i}" for i in range(n_perts)]
    shared = rng.normal(0, 0.8, (n_perts, genes)) * (rng.random((n_perts, genes)) < 0.3)
    lines = {}
    for li, n in enumerate(LINES):
        base = rng.uniform(0.5, 2.0, genes) + 0.3 * li
        specific = rng.normal(0, 0.3, (n_perts, genes)) * (rng.random((n_perts, genes)) < 0.1)
        xs, ps = [rng.normal(base, 0.3, (200, genes))], [CONTROL_LABEL] * 200
        for i, p in enumerate(panel):
            if n == "rpe1" and i % 5 == 0:
                continue  # partial overlap: RPE1 misses every 5th perturbation
            xs.append(rng.normal(base + shared[i] + specific[i], 0.3, (24, genes)))
            ps += [p] * 24
        x = np.clip(np.concatenate(xs), 0, None).astype(np.float32)
        lines[n] = LineData(n, x, np.array(ps), np.zeros(len(ps), dtype=np.int64))
    return lines, panel


def test_bh_matches_manual() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    rng = np.random.default_rng(0)
    q = rng.random(200) ** 3
    from scipy.stats import false_discovery_control
    np.testing.assert_allclose(bh(q), false_discovery_control(q, method="bh"))


def test_de_detects_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(1, 0.2, (300, 5))
    cells = rng.normal(1, 0.2, (40, 5))
    cells[:, 0] += 1.0
    padj, lfc = de_test(cells, ref)
    assert padj[0] < 1e-6 and lfc[0] > 0
    assert (padj[1:] > 0.01).all()


def test_discrimination_perfect_and_chance() -> None:
    rng = np.random.default_rng(0)
    true = rng.normal(size=(10, 8))
    assert np.allclose(discrimination(true, true), 1.0)
    rev = discrimination(true[::-1], true)
    assert rev.mean() < 0.5


def test_ranges_merge() -> None:
    assert _ranges(np.array([1, 2, 3, 10, 11, 30]), max_gap=2, max_rows=64) == [(1, 4), (10, 12), (30, 31)]


def test_energy_mmd_zero_for_identical_and_positive_for_shift() -> None:
    a = torch.randn(2, 16, 4)
    assert float(energy_mmd(a, a)) == pytest.approx(0.0, abs=1e-5)
    assert float(energy_mmd(a, a + 2.0)) > 1.0


def test_caster_is_permutation_equivariant() -> None:
    torch.manual_seed(0)
    m = Caster(CasterConfig(n_genes=6, n_perts=3, hidden=16, layers=2, heads=2, prior=False)).eval()
    x = torch.randn(1, 8, 6)
    perm = torch.randperm(8)
    y = m(x, torch.tensor([1]))
    y_perm = m(x[:, perm], torch.tensor([1]))
    torch.testing.assert_close(y[:, perm], y_perm, atol=1e-5, rtol=1e-5)


def test_split_disjoint_and_deterministic() -> None:
    panel = [f"G{i}" for i in range(100)]
    s1 = make_split(panel, "k562", 0.25, 0.05, 0)
    s2 = make_split(panel, "k562", 0.25, 0.05, 0)
    assert s1 == s2
    assert not set(s1.train) & set(s1.test) and not set(s1.val) & set(s1.test)
    assert len(s1.train) + len(s1.val) + len(s1.test) == 100
    assert "k562" not in s1.sources


def test_control_pools_disjoint_and_inputs_deterministic() -> None:
    lines, _ = synthetic()
    ix = index_line(lines["k562"])
    assert not set(ix.ctrl_input) & set(ix.ctrl_ref)
    np.testing.assert_array_equal(input_controls(ix, "G1", 5, 0), input_controls(ix, "G1", 5, 0))


def test_end_to_end_beats_nochange(tmp_path: Path) -> None:
    lines, panel = synthetic()
    run_context_task(lines, panel, tmp_path, frac_train=0.3, frac_val=0.1, targets=("hepg2",),
                     epochs=15, log=lambda _: None)
    df = pd.read_parquet(tmp_path / "metrics_hepg2_s0.parquet")
    means = df.groupby("method")["pearson_delta"].mean()
    assert set(means.index) >= {"CASTER", "PerturbMean", "ContextMean", "Linear", "NoChange"}
    assert means["CASTER"] > 0.5 and means["PerturbMean"] > 0.5
    # held-out perturbations never enter training for the target line
    z = np.load(tmp_path / "pred_hepg2_s0.npz")
    split = make_split(panel, "hepg2", 0.3, 0.1, 0)  # all hepg2 perts are eligible here
    assert list(z["perts"]) == split.test
