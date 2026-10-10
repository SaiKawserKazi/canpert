# Step 0: Topic decision record

Date: 2026-09-30. Status: **decided**.

## Reference paper and what "same type" means here

The reference is Adduri et al., *Predicting cellular responses to perturbation across
diverse contexts with State*, Cell 189, 5914–5931 (2026). It is a 39-page Cell article:
18 main pages, 5 multi-panel figures, and 20 pages of STAR Methods. Its structure is:

1. a **set-based model** that predicts perturbed single-cell populations from control
   populations (a transformer over cell sets trained with an MMD loss);
2. a **context-generalisation benchmark** (underrepresented and zero-shot contexts)
   against mean baselines, a linear model and deep baselines;
3. a **biologically interpretable evaluation suite** (Cell-Eval: discrimination,
   Pearson-delta, DE AUPRC, DE overlap, fold-change Spearman, effect size);
4. **applications**: context-specific DE, perturbation similarity, gene programs,
   inverse perturbation design, and viability/DepMap validation.

We keep this type of study and not its content. The new study has a new question, a new
model, new analyses and a new title. We cannot reproduce the reference's scale
(167 M-cell embedding model, H100 cluster) on this machine (4 CPU cores, 8 GB RAM,
16 GB free disk, about 60 KB/s per connection). The design below is sized to run fully,
on real data, within those limits.

## Topic (perturbation first, cancer second)

**How much does the cancer cell context reshape the transcriptional response to the same
genetic perturbation, can a set-based model transfer knockdown responses into an
under-sampled cancer context, and do the context-specific parts of the response mark
context-specific cancer dependencies?**

Working title: *Cancer cell context reshapes genetic perturbation responses: set-based
transfer of Perturb-seq effects across leukaemia, hepatoma and non-transformed cells.*

Model name: **CASTER** (Context-Adaptive Set Transformer for pErturbation Responses).

## Why this topic

- **Perturbation first.** The study uses the only public set of CRISPRi Perturb-seq
  screens that target the same ~2,000 essential genes in four human cell lines:
  Replogle et al. 2022 (K562, RPE1) and Nadig et al. 2025 (HepG2, Jurkat).
- **Cancer is built in.** Three of the lines are cancers: K562 (chronic myeloid
  leukaemia, BCR-ABL1), HepG2 (hepatoblastoma/hepatocellular carcinoma) and Jurkat
  (T-cell acute lymphoblastic leukaemia). RPE1 (hTERT-immortalised retinal pigment
  epithelium, TP53 wild type) is a non-transformed reference. This gives a natural
  cancer-versus-normal contrast that the reference paper does not analyse.
- **Independent cancer validation.** DepMap CRISPR (Chronos) gene-effect scores exist for
  K562 (ACH-000551), HepG2 (ACH-000739) and Jurkat (ACH-000995). These are measured by a
  different technology (CRISPR-KO fitness) and are never used to train the model.
- **Different from our previous project** (KRAS-Ctx: drug perturbations in KRAS-mutant
  lung lines from Tahoe-100M). This study uses genetic perturbations, single-cell set
  modelling, a different model and different cancers.

## Feasibility decision (data)

The four source h5ad files total about 33 GB, which is too large to download here. Each
stores `X` as a contiguous dense float32 matrix, so we read only obs/var and fetch a
stratified subsample of cells by HTTP byte ranges (`src/canpert/ingest.py`):
**up to 32 cells per perturbation, plus 3,000 non-targeting control cells per line**.
The paper reports this subsampling as a deviation from full-data training.
