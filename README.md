# CanPert: cell line context and genetic perturbation responses

[![Repository](https://img.shields.io/badge/GitHub-SaiKawserKazi%2Fcanpert-181717?logo=github)](https://github.com/SaiKawserKazi/canpert)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/manuscript-V.1.0%20(review%20round)-a81f2d)

How much does the cell line context reshape the transcriptional response to the same CRISPRi
knockdown, and can a model transfer responses between lines? CanPert answers this with Perturb-seq
of essential genes in three cancer lines (K562 CML, HepG2 hepatoblastoma, Jurkat T-ALL) and the
non-transformed RPE1 line. It introduces **CASTER** (Context-Adaptive Set Transformer for
pErturbation Responses), benchmarks it against mean, offset and linear baselines with calibrated
metrics and noise ceilings, and relates response size to DepMap gene essentiality.

**Repository:** https://github.com/SaiKawserKazi/canpert

- **Manuscript (V.1.0, current):** [`reports/CanPert_manuscript_V.1.0.pdf`](reports/CanPert_manuscript_V.1.0.pdf)
  (`.docx` alongside), 43 pages in Cell article format, with 6 main figures, 15 supplementary
  figures, 1 main table and 11 supplementary tables.
- **Response to review:** [`reports/CanPert_response_to_review_V.1.0.docx`](reports/CanPert_response_to_review_V.1.0.docx).
- **V.0.0 (first version):** [`reports/CanPert_manuscript.pdf`](reports/CanPert_manuscript.pdf), 35 pages.
- **Protocol and amendments:** [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) (frozen at tag `protocol-v1.0`).
- **Overall analysis (V.1.0):** [`docs/OVERALL_ANALYSIS_V1.0.md`](docs/OVERALL_ANALYSIS_V1.0.md).
- **Topic decision:** [`docs/STEP0_TOPIC_DECISION.md`](docs/STEP0_TOPIC_DECISION.md).
- **All numbers:** `reports/results.json`. Each figure panel writes its CSV to `reports/figures/`.

## Results (all pre-declared tests reported)

| | Outcome |
|---|---|
| Cross-context similarity | Responses only moderately shared (median r 0.31 among cancer lines); similarity rises with response strength |
| H1: under-represented context | **Supported**: CASTER > PerturbMean and ContextMean; tied with a context-adaptive linear model (6/9 Holm-significant, 0 opposite) |
| H2: zero-shot context | **Not supported**: PerturbMean is better without target-context perturbation data |
| H3: cancer-cancer transfer > transfer with RPE1 | **Not supported, opposite direction**, driven by the epithelial pair HepG2-RPE1 |
| H4: response magnitude tracks DepMap dependency | **Supported** in all three cancer lines (rho -0.39 to -0.54) |
| H4b: context-specific response tracks context-specific dependency | Not supported (K562 only, weak) |
| H5: predicted shifts retain dependency information | Yes (CASTER r 0.50 vs 0.57 for observed shifts) |

AMENDMENT-002 documents an invalid pre-registered predicted-side DE test (Wilcoxon on continuous
predictions) and its replacement. Both versions are reported, and H1 holds under both.

## Reproduce

```bash
uv sync
uv run canpert build-data   # after fetching (src/canpert/ingest.py: hf_metadata, hf_fetch)
uv run canpert observed
uv run canpert t1 --epochs 20
uv run canpert t2 --epochs 20
uv run canpert ablation --epochs 10
uv run canpert transfer --epochs 10
uv run canpert posctl
uv run canpert analyze
uv run canpert figures
uv run canpert paper        # DOCX + PDF (PDF export needs MS Word)
```

Every command has `--dry` for a synthetic end-to-end run. Tests: `uv run pytest -q`.

## Verification performed

- Fetched-row completeness checked: 0 of 196,093 selected cells missing.
- 112,310 duplicate rows from overlapping download passes confirmed byte-identical.
- Independent recomputation of H1 and H4 from raw artifacts and raw DepMap matches exactly.
- Every number in the PDF traced to `results.json`; all figure and table cross-references
  and all 85 references resolve.
- Every figure inspected visually and corrected.

**Open items for the authors:** author list and affiliations, lead contact, funding, author
contributions, competing interests, a DOI for the code, and a final check of every reference.

## V.1.0 review round (October 2026)

A review of V.0.0 (`CanPert_review_and_todo.docx`) was addressed point by point; see
`reports/CanPert_response_to_review_V.1.0.docx` and AMENDMENT-003 in `RESEARCH_PROTOCOL.md`.
Additions, all labelled exploratory:
- Seeds 0-2 for the main benchmark.
- New baselines and controls: a LineOffset baseline, a line-identity-only CASTER, wrong-line
  control cells and complex-grouped splits.
- 20-epoch ablations with seeds.
- Calibrated metrics following Miller et al. 2026: technical and interpolated duplicates,
  WMSE, weighted R2, NIR, PDS and DRF. Split-half noise ceilings.
- Noise-free re-scoring of all methods, and p53, DepMap cross-line and ribosome analyses.
- A fix to the direction of the paired test for MAE.

Main V.1 conclusions:
- CASTER outperforms mean baselines when part of the target line is screened, and ties a
  linear transfer model.
- It is worse than the perturbation mean in unseen lines.
- Response size tracks general gene essentiality, not line-specific dependency.

## Citation

See `CITATION.cff`. Author list to be completed.
