# CanPert research protocol (v1.0)

Status: **frozen before any model is fitted on real data** (git tag `protocol-v1.0`).
Changes after freezing go in numbered amendments at the end of this file.

## 1. Question

Does the cellular context (three cancer lines and one non-transformed line) change the
transcriptional response to the same CRISPRi knockdown? Can a set-based model (CASTER)
transfer knockdown responses into an under-sampled or unseen context better than simple
baselines? Do context-specific responses mark context-specific cancer dependencies?

## 2. Data

- Source: the Arc Institute four-line concatenation of the Replogle et al. 2022 (K562, RPE1)
  and Nadig et al. 2025 (HepG2, Jurkat) CRISPRi Perturb-seq screens
  (`arcinstitute/State-Replogle-Filtered`, `replogle_concat.h5ad`; log-normalised, 6,546
  shared genes, knockdown-filtered by Arc). We chose this source over the per-line
  originals for two reasons: it is processed identically for all four lines, and NCBI
  throttled range requests to about 0.04 MB/s.
- Subsample: <= 32 cells per perturbation and line, plus 3,000 non-targeting controls per
  line, chosen at random (seed 0) from metadata before any expression value is read.
  Perturbations keep >= 16 cells.
- Gene space: 2,000 genes chosen by dispersion on **control cells only**, after removing
  every perturbation-target gene.
- DepMap 24Q4 Chronos gene effect for K562, HepG2 and Jurkat. It is used only for
  validation.

## 3. Tasks and splits

- **Eligible perturbations** for a target line are those observed in the target and in at
  least one other line. Arc's knockdown filter acts per line, so only 638 perturbations
  pass in all four.
- **T1 under-represented context.** For each target line (4 folds), the eligible
  perturbations are shuffled (seed 0) and split 25% train, 5% validation (checkpoint
  selection) and 70% test. Everything in the other three lines is training data.
  Seed 0 is primary.
- **T2 zero-shot context.** No perturbed cells of the target line are used. Checkpoint
  selection uses 5% held-out source-line items.
- **T3 single-source transfer matrix.** All 12 ordered pairs: train on one source line,
  test on all perturbations shared with the target.
- **Ablations (T1, target K562, seed 0):** set size S in {1, 4, 8, 16, 32}; S=32
  without attention; without FiLM; without the transfer prior (one-hot embedding
  instead); with an added one-hot ID embedding.

## 4. Methods compared

CASTER (proposed; transfer prior + context-conditioned set transformer, initialised at the PerturbMean solution); PerturbMean; ContextMean (T1 only); Linear transfer, which is ridge
regression from PCA features of source-line effects to target-line effects, fitted on
the support perturbations (T1 only); NoChange (control cells, a floor).

## 5. Metrics (PertBench, `src/canpert/evaluate.py`)

Pearson-delta, discrimination (L1), MAE, DE AUPRC, DE overlap at N, log fold-change
Spearman on true DEGs, direction agreement, and effect-size Spearman (number of DEGs).

## 6. Hypotheses and tests

- **H1 (primary).** On T1, CASTER beats PerturbMean, ContextMean and Linear on the
  three primary metrics: Pearson-delta, discrimination and DE AUPRC. Test: per-
  perturbation paired Wilcoxon signed-rank, pooled over the 4 folds (seed 0). One-sided
  alternative: CASTER is better. Holm correction over 9 tests (3 baselines x 3 metrics).
  Supported if at least 6 of 9 tests have Holm p < 0.05 and none has a significant effect
  in the opposite direction (two-sided check).
- **H2.** On T2, CASTER beats PerturbMean on the same 3 metrics (Holm over 3 tests).
- **H3 (cancer versus normal).** In T3, transfer is better between cancer lines than
  into or out of RPE1. Test: PerturbMean single-source Pearson-delta. For each
  perturbation, the mean over cancer-to-cancer pairs is compared with the mean over
  pairs involving RPE1 (paired Wilcoxon, two-sided).
- **H4 (dependency).** In each cancer line, the observed response magnitude (number of
  DEGs) of a knockdown correlates negatively with the Chronos gene effect of the
  target (stronger dependency means a larger response). Spearman rho per line with
  Holm over 3 lines. **H4b:** the context-specific response magnitude (line value minus
  mean of the other cancer lines, in log1p number of DEGs) correlates negatively with the
  context-specific Chronos effect (line minus mean of the other two).
- **H5 (application, exploratory).** A ridge model trained on observed transcriptional
  shifts to predict Chronos gene effect is applied to shifts predicted by each method for
  held-out perturbations. We compare Pearson r between predicted and measured Chronos
  across methods (10-fold cross-validation by perturbation).
- **Exploratory, no multiplicity claims.** Perturbation-similarity clustering (ARI of
  predicted versus observed), recovery of STRING high-confidence pairs by effect
  similarity (AUROC), gene programs, inverse perturbation design (top-k retrieval), and
  cell-cycle score recovery.

## 7. Reporting rules

Every pre-declared test is reported with its raw and Holm-adjusted p value, whatever the
outcome. Deviations (for example, seeds not run for compute reasons) are reported as
deviations. Every number in the paper is generated from `reports/results.json`.

## AMENDMENT-001 (2026-10-01, before any real-data result was seen)

At about 110 s per epoch on this 4-core CPU, the planned schedule would have taken about
12 h. Before any real-data result existed (the first training epoch was the only output),
we changed the following:
- T1 and T2 are trained for 20 epochs (was unspecified in v1.0; the code default was 40).
- Ablations and the transfer matrix are trained for 10 epochs.
- The set-size ablation uses S in {1, 8, 32} (S = 4 and 16 dropped). All component
  ablations are kept.

No hypothesis, metric, split or decision rule changed.

## AMENDMENT-002 (2026-10-02, after the T1 results were first inspected)

**Defect.** The pre-registered single-cell DE test (Wilcoxon rank-sum, predicted cells vs
reference controls) is invalid for continuous predictions. Shift-based predictions turn
every exact zero of a control cell into a small positive value, so against zero-inflated
reference cells about 1,300 of 2,000 genes are called significant for almost any shift. In
the HepG2 fold, every shifted method had a mean of 1,283-1,494 predicted DEGs, while
unshifted control cells (NoChange) had 0.1 and the observed median was 2-5. The observed
(ground-truth) DE keeps exact zeros and is unaffected.

**Change.** Predicted-side DE now uses a moment-matched z-test on the predicted pseudobulk
shift, with reference-control variance and the observed sample sizes (n perturbed vs 500
reference cells; BH FDR 0.05). Predicted fold-change ranks and signs use the pseudobulk
shift. Affected metrics: DE AUPRC, DE overlap, fold-change Spearman, direction and
effect-size Spearman. Pearson delta and discrimination are unchanged. Ground truth is
unchanged. Nothing is retrained.

**Reporting.** The amended metrics are primary. The pre-registered Wilcoxon versions,
including the full H1 test family, are reported in the supplement and flagged as invalid,
with this rationale. This amendment was made after results were seen. It is disclosed as
such, and the decision rule for H1 is unchanged.

## AMENDMENT-003 (2026-10-08/09, review round; after all pre-specified results were known)

Prompted by an internal review (`CanPert_review_and_todo.docx`). Everything below is exploratory
and labelled as added later in the manuscript; no pre-specified result was altered or re-tested
with a new rule.

- **Seeds:** T1 repeated with seeds 1 and 2. Each seed changes the split, initialisation and
  batch order. 20 epochs.
- **Baselines and controls:**
  - LineOffset: PerturbMean plus one per-line offset estimated on the support set.
  - Line-identity-only CASTER: control sets replaced by the line's mean control profile.
  - Wrong-line control cells at inference.
  - Complex-grouped splits: Louvain communities of the STRING >= 950 graph; whole groups go to
    train, val or test.
- **Ablations:** rerun at 20 epochs (K562 fold), with seed 1 for the component ablations. The
  full model reuses the main T1 K562 runs.
- **Calibrated metrics:** following Miller et al. 2026 and Mejia et al. 2025, add a split-half
  technical duplicate, an interpolated duplicate, the mean-baseline negative control, WMSE,
  weighted R2 (vs control and vs perturbed mean), Pearson delta on the top-100 DEGs vs the
  perturbed mean, MSE, NIR, PDS and DRF. Add Spearman-Brown noise ceilings.
- **Bug fix:** paired tests for lower-is-better metrics (MAE) had been run as
  higher-is-better. They are now oriented correctly.
- **Analyses:**
  - p53 target-gene score per knockdown and line.
  - DepMap cross-line control (other lines' Chronos).
  - Ribosome enrichment of the gain (adjusted for response strength) with a sibling-leakage
    check.
  - Noise-cancellation check (baselines rendered on noise-free controls).
  - All-gene cross-line similarity (gene-choice check).
- **Programs:** the Reactome annotation is restricted to terms with 10-500 modelled genes, and
  BH-adjusted over the terms tested.
- **Compute:** new runs skip the invalid predicted-side Wilcoxon test (AMENDMENT-002). Runs
  are sequential with 4 threads because two parallel jobs exceeded 8 GB RAM.
