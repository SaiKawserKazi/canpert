"""Command-line entry points: ``uv run canpert <step>``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import typer

from canpert.data import LINES, LineData, build, load_processed

app = typer.Typer(add_completion=False, no_args_is_help=True)
if "OMP_NUM_THREADS" in __import__("os").environ:
    import torch as _torch

    _torch.set_num_threads(int(__import__("os").environ["OMP_NUM_THREADS"]))
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "perturbseq"
PROC = ROOT / "data" / "processed"


def _art(dry: bool) -> Path:
    return ROOT / ("artifacts_dry" if dry else "artifacts")


def _log_to(path: Path):  # noqa: ANN202
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    return log


def synthetic_lines(n_perts: int = 160, n_genes: int = 400, cells: int = 24, seed: int = 0,
                    learnable_context: bool = False) -> tuple[dict[str, LineData], list[str], np.ndarray]:
    """Realistic-shaped synthetic data (dry run only): shared + context-specific effects
    drawn from low-rank programs; perturbation/gene names are real symbols so that the
    downstream DepMap/STRING code paths are exercised."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    ch = pd.read_parquet(ROOT / "data/raw/depmap/CRISPRGeneEffect.parquet").columns[1:]
    syms = np.array([c.split(" (")[0] for c in ch])
    rng.shuffle(syms)
    panel = sorted(syms[:n_perts].tolist())
    genes = syms[n_perts:n_perts + n_genes]
    progs = rng.normal(0, 1, (12, n_genes)) * (rng.random((12, n_genes)) < 0.15)
    load = rng.normal(0, 0.5, (n_perts, 12)) * (rng.random((n_perts, 12)) < 0.3)
    shared = load @ progs
    lines = {}
    for li, n in enumerate(LINES):
        base = rng.gamma(2, 0.5, n_genes) + 0.1 * li
        ctx_load = load + rng.normal(0, 0.25, load.shape) * (rng.random(load.shape) < 0.2)
        eff = ctx_load @ progs * (0.6 if n == "rpe1" else 1.0)
        if learnable_context:
            # positive control: a gene responds in proportion to its baseline expression in
            # that line (observable from control cells), and not at all when silent
            base = rng.gamma(1.2, 0.9, n_genes) * (rng.random(n_genes) > 0.25)
            eff = (load @ progs) * base / base.mean()
        xs, ps = [np.clip(rng.normal(base, 0.4, (600, n_genes)), 0, None)], ["non-targeting"] * 600
        observed = rng.random(n_perts) < 0.8
        for i, p in enumerate(panel):
            if not observed[i]:
                continue
            xs.append(np.clip(rng.normal(base + eff[i], 0.4, (cells, n_genes)), 0, None))
            ps += [p] * cells
        lines[n] = LineData(n, np.concatenate(xs).astype(np.float32), np.array(ps),
                            np.zeros(len(ps), dtype=np.int64))
    _ = shared
    return lines, panel, genes


def _load(dry: bool) -> tuple[dict[str, LineData], list[str], np.ndarray]:
    if dry:
        return synthetic_lines()
    lines, panel, genes = load_processed(PROC)
    return lines, panel, genes["gene_name"].to_numpy()


@app.command("build-data")
def build_data() -> None:
    """Build processed arrays from fetched shards (QC, gene space, panel)."""
    audit = build(RAW, PROC)
    print(json.dumps(audit, indent=2))


@app.command()
def observed(dry: bool = False) -> None:
    """Observed DE for every perturbation in every line."""
    from canpert.experiments import observed_effects

    lines, panel, genes = _load(dry)
    observed_effects(lines, panel, _art(dry) / "observed")
    np.save(_art(dry) / "observed" / "genes.npy", genes)


@app.command()
def t1(dry: bool = False, seeds: str = "0", epochs: int = 40, targets: str = "k562,hepg2,jurkat,rpe1",
       variant: str = "main") -> None:
    """Underrepresented-context task (25/5/70).

    variant: main | meanctrl (line identity only, no control cells) | grouped (complex-grouped split)
    """
    from canpert.experiments import run_context_task

    lines, panel, _ = _load(dry)
    out = _art(dry) / ("t1" if variant == "main" else f"t1_{variant}")
    groups = complex_groups(panel) if variant == "grouped" else None
    run_context_task(lines, panel, out, frac_train=0.25, frac_val=0.05,
                     seeds=tuple(int(s) for s in seeds.split(",")), epochs=epochs,
                     targets=tuple(targets.split(",")), log=_log_to(out / "log.txt"),
                     ctrl_mode="mean" if variant == "meanctrl" else "cells", groups=groups)


def complex_groups(panel: list[str], min_score: int = 950) -> dict[str, int]:
    """Protein-complex groups: Louvain communities (resolution 1, seed 0) of the STRING >= 950
    graph among the perturbed genes; keeps e.g. all ribosomal proteins in one group. Genes without
    a high-confidence partner form singleton groups."""
    import json as _json

    cache = ROOT / "data" / "processed" / "complex_groups.json"
    if cache.exists():
        return {k: int(v) for k, v in _json.loads(cache.read_text()).items()}
    import networkx as nx

    from canpert.analysis import load_string_pairs

    pairs = load_string_pairs(ROOT / "data" / "raw" / "network", min_score)
    pset = set(panel)
    g = nx.Graph()
    g.add_nodes_from(panel)
    g.add_edges_from(tuple(e) for e in pairs if len(e) == 2 and set(e) <= pset)
    comms = nx.algorithms.community.louvain_communities(g, resolution=1.0, seed=0)
    groups = {gene: i for i, c in enumerate(comms) for gene in c}
    cache.write_text(_json.dumps(groups))
    return groups


@app.command()
def t2(dry: bool = False, epochs: int = 40) -> None:
    """Zero-shot context task."""
    from canpert.experiments import run_context_task

    lines, panel, _ = _load(dry)
    out = _art(dry) / "t2"
    run_context_task(lines, panel, out, frac_train=0.0, frac_val=0.0, epochs=epochs,
                     log=_log_to(out / "log.txt"))


@app.command()
def ablation(dry: bool = False, epochs: int = 30, seed: int = 0, only: str = "",
             outdir: str = "ablation") -> None:
    """Set-size / attention / FiLM / prior ablations (K562 held out)."""
    from canpert.experiments import run_ablation

    lines, panel, _ = _load(dry)
    out = _art(dry) / outdir
    run_ablation(lines, panel, out, epochs=epochs, log=_log_to(out / "log.txt"), seed=seed,
                 only=tuple(only.split(",")) if only else None)


@app.command()
def transfer(dry: bool = False, epochs: int = 25) -> None:
    """Single-source transfer matrix."""
    from canpert.experiments import run_transfer_matrix

    lines, panel, _ = _load(dry)
    out = _art(dry) / "transfer"
    run_transfer_matrix(lines, panel, out, epochs=epochs, log=_log_to(out / "log.txt"))


@app.command()
def posctl(epochs: int = 60, out_reports: str = "reports") -> None:
    """Synthetic positive control: CASTER vs baselines when the context effect is random
    noise (unlearnable) or a function of baseline expression (learnable from controls)."""
    import pandas as pd

    from canpert.experiments import run_context_task

    rows = []
    for scen, learn in (("unlearnable", False), ("learnable", True)):
        lines, panel, _ = synthetic_lines(learnable_context=learn)
        out = ROOT / "artifacts" / f"posctl_{scen}"
        run_context_task(lines, panel, out, frac_train=0.25, frac_val=0.05, targets=("hepg2",),
                         epochs=epochs, log=_log_to(out / "log.txt"))
        df = pd.read_parquet(out / "metrics_hepg2_s0.parquet")
        g = df.groupby("method")[["pearson_delta", "discrimination", "de_auprc"]].mean().reset_index()
        rows.append(g.assign(scenario=scen))
    tab = ROOT / out_reports / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(tab / "positive_control.csv", index=False)


@app.command()
def analyze(dry: bool = False) -> None:
    """All statistics and downstream analyses -> reports/results.json + tables."""
    from canpert.report import run_analysis

    run_analysis(_art(dry), ROOT / ("reports_dry" if dry else "reports"), ROOT / "data/raw")


@app.command()
def figures(dry: bool = False) -> None:
    """Render all main and supplementary figures."""
    from canpert.figures import render_all

    render_all(_art(dry), ROOT / ("reports_dry" if dry else "reports"))


@app.command()
def paper(dry: bool = False) -> None:
    """Build the manuscript (.docx, and .pdf through MS Word)."""
    from canpert.paper import build_paper

    build_paper(ROOT / ("reports_dry" if dry else "reports"))


if __name__ == "__main__":
    sys.exit(app())
