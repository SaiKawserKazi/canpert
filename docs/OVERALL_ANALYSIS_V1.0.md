# CanPert V.1.0: overall analysis

Assessment of the V.1.0 manuscript (`reports/CanPert_manuscript_V.1.0.pdf`, 43 pages), written
after the review round. All numbers come from `reports/results.json`, and an independent recheck
(`scripts/recheck_v1.py`) reproduces every one checked.

## Core finding (one sentence)

Cell line context substantially reshapes the response to the same CRISPRi knockdown. Some
target-line data is needed to predict it. Response size tracks how essential the gene is in
general, not line-specific dependency.

## Hypotheses and outcomes

| Hypothesis | Outcome | Key numbers |
|---|---|---|
| Responses are shared across lines | Only moderately | Median r = 0.31 among cancer lines; similarity rises with response strength |
| H1: CASTER helps in an under-represented line | Supported vs mean baselines; **tie with Linear** | Pearson delta over 3 seeds: CASTER 0.276 ± 0.003, Linear 0.282 ± 0.003, LineOffset 0.249, PerturbMean 0.198 ± 0.002. Seed 0: 6/9 tests Holm-significant, 0 opposite |
| H2: zero-shot to an unseen line | **Not supported** (opposite) | PerturbMean better in 21 of 24 line × metric panels |
| H3: cancer-cancer transfer beats transfer with RPE1 | **Not supported** (opposite) | Driven by HepG2-RPE1; partly explained by shared p53 activity |
| H4: response size tracks DepMap dependency | **Supported** in all 3 cancer lines | Spearman rho -0.39 to -0.54 |
| H4b: line-specific response tracks line-specific dependency | Not supported | Weak, K562 only |
| H5: predicted shifts keep dependency information | Yes (exploratory) | Mean r over cancer lines: CASTER 0.50, Linear 0.52, PerturbMean 0.47, observed shifts 0.57 |

## Strengths

1. **Clear biological message.** Context matters a lot, and the paper measures how much.
2. **Consistent essentiality link** across all three cancer lines. The DepMap cross-line control
   shows it reflects general essentiality.
3. **Methodological finding with wide relevance.** Rendering noise lowers every method's Pearson
   delta by about 0.15. With noise-free re-scoring, CASTER reaches 0.423, Linear 0.438 and
   PerturbMean 0.252.
4. **Rigour above the field's norm.**
   - Pre-specified protocol with three documented amendments.
   - Three seeds.
   - Calibrated metrics following Miller et al. 2026, with DRF filtering and split-half noise
     ceilings.
   - Complex-grouped splits as a leakage check.
   - LineOffset, line-identity-only and wrong-line-control controls.
   - All results reported, including the negative ones.
5. **Reproducible.** Open code (https://github.com/SaiKawserKazi/canpert), every number
   traceable, and an independent recheck script.

## Weaknesses (stated in the manuscript)

1. CASTER ties a linear transfer model, so there is no state-of-the-art claim.
2. CASTER fails in unseen lines; PerturbMean is better there.
3. CASTER's gain comes mostly from line identity: the identity-only variant is nearly as good.
4. Complex-grouped splits shrink the advantage of the learned methods, so part of the gain
   comes from related genes.
5. Only 4 cell lines, from two studies, so batch and protocol differences are mixed with
   biology.
6. Essential genes, CRISPRi only, 2,000 genes; no new wet-lab validation.
7. Some review controls use one seed; one ablation stopped at 14/20 epochs.

## Journal suitability

- **Good fit (Q1):**
  - PLOS Computational Biology
  - Bioinformatics
  - Briefings in Bioinformatics
  - NAR Genomics and Bioinformatics
  - BMC Biology
- **Possible but competitive:** Cell Systems and Genome Biology, framed as a benchmark and
  biology study.
- **Unlikely:** Cell, Nature Methods and Nature Biotechnology. They need a method that clearly
  wins, or new experiments.
- **Recommended framing:** a benchmark and biology study ("how much does context change
  perturbation responses, and what can current models transfer?"), not "a new model that wins".

## Reviewer checklist status (V.0.0 review)

All 9 main issues, 9 must-do experiments, the metric requests, the text, number and figure
fixes, and the wording changes are addressed. The point-by-point response is in
`reports/CanPert_response_to_review_V.1.0.docx`, and the additions are recorded as
AMENDMENT-003 in `RESEARCH_PROTOCOL.md`.

## Remaining items for the authors before submission

- Author names and affiliations; corresponding author and e-mail.
- Funding statement, author contributions, competing interests.
- A code DOI (for example a Zenodo release of the GitHub repository).
- A final check of every reference against the journal's style.
- A cover letter that leads with the core finding above.
