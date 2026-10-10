"""STAR Methods (result-independent text, equations and the key resources table)."""

from __future__ import annotations

from typing import Any

from canpert.docxkit import Manuscript


def key_resources(m: Manuscript) -> None:
    m.heading("Key resources table", 2)
    rows = [
        ["Deposited data", "", ""],
        ["K562 and RPE1 CRISPRi Perturb-seq", "Replogle et al. 2022", "Figshare plus article 20029387"],
        ["HepG2 and Jurkat CRISPRi Perturb-seq", "Nadig et al. 2025", "GEO: GSE264667"],
        ["Knockdown-filtered four-line concatenation (log-normalised)", "Arc Institute (State)",
         "huggingface.co/datasets/arcinstitute/State-Replogle-Filtered"],
        ["DepMap CRISPR gene effect (Chronos), release 24Q4", "DepMap / Broad Institute", "depmap.org/portal"],
        ["STRING v12.0 human protein links", "Szklarczyk et al. 2023", "string-db.org"],
        ["Reactome pathways (GMT)", "Milacic et al. 2024", "reactome.org"],
        ["Experimental models: cell lines (data only)", "", ""],
        ["K562 (CML)", "Lozzio and Lozzio 1975", "RRID:CVCL_0004"],
        ["HepG2 (hepatoblastoma)", "Aden et al. 1979", "RRID:CVCL_0027"],
        ["Jurkat (T-ALL)", "Schneider et al. 1977", "RRID:CVCL_0065"],
        ["hTERT RPE-1 (non-transformed)", "Bodnar et al. 1998", "RRID:CVCL_4388"],
        ["Software and algorithms", "", ""],
        ["CASTER, PertBench and all analyses", "This paper", "canpert repository (code availability)"],
        ["PyTorch 2.13 (CPU)", "Paszke et al. 2019", "pytorch.org"],
        ["scikit-learn", "Pedregosa et al. 2011", "scikit-learn.org"],
        ["SciPy / NumPy", "Virtanen et al. 2020; Harris et al. 2020", "scipy.org; numpy.org"],
        ["h5py / anndata format", "Virshup et al. 2024", "anndata.readthedocs.io"],
        ["Matplotlib", "Hunter 2007", "matplotlib.org"],
    ]
    m.table(["REAGENT or RESOURCE", "SOURCE", "IDENTIFIER"], rows, widths=[3.0, 1.9, 2.2])


def method_details(m: Manuscript, res: dict[str, Any]) -> None:  # noqa: PLR0915
    audit = res.get("audit", {})
    m.heading("Experimental model and study participant details", 1)
    m.para("This study generated no new experimental data. All analyses use published CRISPR interference "
           "(CRISPRi) Perturb-seq profiles of four human cell lines: K562 (chronic myeloid leukaemia, "
           "BCR-ABL1-positive)[@lozzio1975], HepG2 (hepatoblastoma-derived, historically described as "
           "hepatocellular carcinoma)[@aden1979,lopezterrada2009], Jurkat (T cell acute lymphoblastic "
           "leukaemia)[@schneider1977,abraham2004] and hTERT RPE-1 (telomerase-immortalised, "
           "non-transformed retinal pigment epithelium)[@bodnar1998]. K562 and RPE1 were profiled by "
           "Replogle et al.[@replogle2022] and HepG2 and Jurkat by Nadig et al.[@nadig2025] with the same "
           "essential-gene CRISPRi sgRNA library. The four screens differ in CRISPRi effector construct, "
           "time point after transduction and sequencing depth, which adds technical differences "
           "between lines to the biological ones.", indent=True)

    m.heading("Method details", 1)
    m.heading("Data acquisition", 2)
    m.para("We used the four-line concatenation released with the State model[@adduri2026] "
           "(replogle_concat.h5ad). It contains library-size-normalised, log1p-transformed expression of "
           f"{audit.get('n_genes_measured', 6546):,} genes measured in all four lines. Perturbations and cells "
           "were already filtered for on-target knockdown, as described by Adduri et al.[@adduri2026]: "
           "perturbation median residual target expression <= 0.30, cell residual <= 0.50, and >= 30 "
           "cells. The file is 30 GB. Because compute and bandwidth were limited, we never downloaded it "
           "whole. We read the cell metadata, chose a stratified random subsample (seed 0) **before any "
           "expression value was read**, and fetched only those rows with HTTP byte-range requests against "
           "the contiguous dense matrix. The subsample is up to 32 cells per perturbation and line, plus "
           "3,000 non-targeting control cells per line. Every fetched row was checked to be finite, "
           "non-negative and in the log1p range, and the set of fetched rows was checked to equal the "
           "selected set exactly. Rows fetched twice by overlapping download passes were byte-identical "
           "and were kept once.", indent=True)
    m.heading("Gene space", 2)
    m.para("We modelled 2,000 genes. They were chosen as the top genes by bin-normalised dispersion (20 "
           "mean bins, Seurat v1 procedure[@satija2015]) computed on the **pooled control cells only**, "
           "after excluding every gene that is a perturbation target in any line "
           f"({audit.get('n_target_genes_excluded', 'n/a')} genes). Because no perturbed cell enters gene "
           "selection, held-out perturbation effects cannot leak through the choice of genes. Because "
           "knocked-down targets are excluded, the trivial on-target signal cannot inflate any metric.",
           indent=True)

    m.heading("Problem formulation", 2)
    m.para("Let $\\mathcal{X}_{\\ell}^{0}$ be the distribution of unperturbed cells in context (cell line) "
           "$\\ell$, and $\\mathcal{X}_{\\ell}^{p}$ the distribution after perturbation $p$. We write the "
           "pseudobulk shift of $p$ in $\\ell$ as", indent=True)
    m.equation(r"\delta_{\ell,p} = \mathbb{E}[\mathcal{X}^{p}_{\ell}] - \mathbb{E}[\mathcal{X}^{0}_{\ell}]"
               r" = \bar\delta_{p} + \epsilon_{\ell,p}")
    m.para("Here $\\bar\\delta_p$ is the shared component (the mean over contexts) and $\\epsilon_{\\ell,p}$ the context-specific component. Context generalisation asks for "
           "$\\mathcal{X}_{t}^{p}$ in a target context $t$, given control cells of $t$, observations of $p$ "
           "in other contexts and, optionally, a small set of other perturbations observed in $t$. "
           "PerturbMean estimates only $\\bar\\delta_p$. Any method that beats it must recover part of "
           "$\\epsilon_{t,p}$.", indent=True)

    m.heading("CASTER architecture", 2)
    m.para("CASTER (Context-Adaptive Set Transformer for pErturbation Responses) maps a set of $S$ "
           "control cells $X^{0} \\in \\mathbb{R}^{S\\times G}$ from the target context, together with a "
           "perturbation, to a set of $S$ predicted perturbed cells. It has five parts.", indent=True)
    m.para("**Transfer prior.** The perturbation is represented by its observed shift in the *other* "
           "contexts. For an item in context $\\ell$ with source contexts $\\mathcal{C}$:", indent=True)
    m.equation(r"d_{\ell,p} = \frac{1}{|\mathcal{C}_{p}\setminus\{\ell\}|}\sum_{c \in \mathcal{C}_{p}\setminus\{\ell\}}"
               r" \hat\delta_{c,p}")
    m.para("$\\mathcal{C}_p$ is the set of source contexts that observed $p$, and $\\hat\\delta_{c,p}$ is the "
           "empirical shift relative to the control mean of $c$. The target context never contributes to "
           "any prior. When no other source context observed $p$ (always the case in single-source "
           "transfer), the prior is computed from a random half of $p$'s cells, and the model is trained "
           "against the disjoint other half.", indent=True)
    m.para("**Encoders.** A 4-layer GELU[@hendrycks2016] MLP $f_{cell}$ embeds each control cell into "
           "$\\mathbb{R}^{h}$ ($h = 256$). A 2-layer MLP $f_{prior}$ embeds the prior. A context summary "
           "$c$ is the mean of the cell embeddings, passed through a 2-layer MLP that outputs FiLM[@perez2018] "
           "parameters $(\\gamma, \\beta)$:", indent=True)
    m.equation(r"H = f_{cell}(X^{0}) + h_{p} + \gamma(c)\odot h_{p} + \beta(c),\quad h_{p} = f_{prior}(d_{t,p})")
    m.para("**Set transformer.** $H$ passes through a 4-layer, 8-head, pre-norm, bidirectional transformer "
           "encoder[@vaswani2017,lee2019] without positional encodings. The model is therefore "
           "permutation-equivariant over cells[@zaheer2017], which our tests verify.", indent=True)
    m.para("**Output.** The prediction adds a context-dependent, gene-wise rescaling of the transferred "
           "shift and a cell-resolved correction:", indent=True)
    m.equation(r"\hat X = X^{0} + \mathbf{1}\,[(1 + g(c))\odot d_{t,p}]^{\top} + f_{out}(f_{cell}(X^{0}) + f_{ST}(H))")
    m.para("Here $g$ is linear in $c$, and $f_{out}$ is a LayerNorm[@ba2016] followed by a linear map. "
           "Both $g$ and the output layer of $f_{out}$ are initialised at zero. **Training therefore starts "
           "exactly at the PerturbMean solution** and leaves it only where the validation objective "
           "improves. The ablations remove the transformer (per-cell MLP), FiLM, or the prior (a learned "
           "one-hot perturbation embedding instead), or add a one-hot ID embedding to the prior.",
           indent=True)
    m.heading("Training objective", 2)
    m.para("For a predicted set $\\hat X$ and an observed set $X$ of the same perturbation and context, the "
           "loss is the squared maximum mean discrepancy[@gretton2012] with the energy-distance "
           "kernel[@szekely2013,feydy2019], plus a pseudobulk term:", indent=True)
    m.equation(r"\mathcal{L} = \frac{2}{S^2}\sum_{i,j}\|\hat x_i - x_j\| - \frac{1}{S^2}\sum_{i,j}\|\hat x_i - \hat x_j\|"
               r" - \frac{1}{S^2}\sum_{i,j}\|x_i - x_j\| + \lambda\,\|\bar{\hat x} - \bar x\|_2^2")
    m.para("We used $\\lambda = 1$ and $S = 32$. Training used AdamW[@loshchilov2019] (learning rate "
           "3e-4, weight decay 1e-4, batch 32 sets), gradient clipping at 1.0, a 5% linear warm-up and "
           "cosine decay. The number of epochs was 20 for the context tasks and 10 for ablations and "
           "single-source transfer. The checkpoint with the lowest validation loss was kept. In the "
           "under-represented task, validation uses the 5% validation perturbations of the target. In "
           "the zero-shot and transfer tasks, it uses 5% of source-context items, so no target perturbed "
           "cell is ever used. All runs used CPU only (PyTorch[@paszke2019]).", indent=True)

    m.heading("Baselines", 2)
    m.para("Every method returns a set of predicted cells, so that single-cell DE metrics are computed "
           "identically for all methods. A pseudobulk prediction $\\hat\\delta$ is rendered as cells by "
           "adding it to the same control cells given to CASTER.", indent=True)
    m.bullets([
        "**NoChange**: $\\hat\\delta = 0$; a floor.",
        "**PerturbMean**: $\\hat\\delta_{t,p} = d_{t,p}$, the mean shift of $p$ over the source contexts "
        "that observed it.",
        "**ContextMean**: the mean shift of the target's training perturbations, the same for every "
        "perturbation[@kernfeld2025].",
        "**LineOffset** (added in the review round): PerturbMean plus one constant shift per target line, "
        "the mean residual $\\frac{1}{|\\mathcal{S}_t|}\\sum_{q\\in\\mathcal{S}_t}(\\hat\\delta_{t,q} - d_{t,q})$ of the "
        "support perturbations. It tests whether a model's gain is a single per-line offset.",
        "**Linear transfer**: ridge regression[@hoerl1970] (penalty chosen by efficient leave-one-out "
        "cross-validation over 13 values) from a 64-component PCA of the concatenated source shifts "
        "of $p$ to its target shift. It is fitted on the target's support perturbations; a missing "
        "source is imputed with the mean over the observing sources. This is a strong, explicitly "
        "context-adaptive linear model of the kind recently shown to match deep "
        "models[@ahlmann2025,kernfeld2025].",
    ])

    m.heading("Tasks and splits", 2)
    m.para("**Under-represented context (T1).** For each target line $t$ (four folds), the eligible "
           "perturbations are those observed in $t$ and in at least one other line. They were shuffled "
           "(seed 0) and split 25% training, 5% validation and 70% test. All perturbations of the three "
           "other lines are training data. **Zero-shot context (T2).** No perturbed cell of $t$ is used. "
           "**Single-source transfer (T3).** For each ordered pair (source $s$, target $t$), training uses "
           "$s$ only and the test set is every perturbation observed in both. Unperturbed cells of every "
           "line, including $t$, are available to all methods, as in Adduri et al.[@adduri2026]. Control "
           "cells of each line were split 50/50 into a model-input pool and a disjoint DE-reference pool.",
           indent=True)
    m.para("**Seeds.** The pre-specified analysis used seed 0. In the review round the under-represented task was "
           "repeated with seeds 1 and 2; each seed changes the perturbation split, the model initialisation and "
           "the mini-batch order. **Controls for per-line effects.** (i) LineOffset (above). (ii) A "
           "line-identity-only CASTER: identical architecture and training, but every control set is replaced by "
           "the line's mean control profile repeated $S$ times, so the model knows which line it is in but sees "
           "no cell-to-cell variation. (iii) Wrong-line control cells: the trained CASTER is given the control "
           "cells of another line (the next line in the order K562, HepG2, Jurkat, RPE1); its predicted shift "
           "(prediction minus its input) is added to the target line's own control cells, so only the model's "
           "use of the control set changes. **Complex-grouped splits.** Perturbations were grouped by Louvain "
           "communities (resolution 1, seed 0) of the STRING v12 graph with combined score >= 950 among the "
           "2,023 targets (500 groups, the largest 157 genes; all 73 ribosomal-protein targets fall in one "
           "group). Whole groups were assigned to training, validation or test, so no held-out gene had a "
           "same-group sibling in training.", indent=True)

    m.heading("Gene programs, similarity and cell-cycle analyses", 2)
    m.para("Gene programs were defined per line by K-means[@lloyd1982] (15 programs, 10 restarts) on the "
           "z-scored observed shift matrix (perturbations x genes), clustering genes by their response "
           "profiles. Programs were annotated with their most enriched Reactome pathway[@milacic2024] "
           "(hypergeometric test, background = modelled genes). Program activity is the mean shift over "
           "program genes. Perturbation-similarity structure used average-linkage hierarchical clustering "
           "of cosine distances between shifts of perturbations with >= 10 observed DEGs, compared with "
           "the adjusted Rand index[@hubert1985]. Recovery of functional relationships used STRING v12 "
           "pairs with combined score >= 700[@szklarczyk2023], scored by effect cosine similarity "
           "(AUROC). Cell-cycle shifts are the mean shift over the Tirosh et al. S-phase and G2/M "
           "signature genes[@tirosh2016] present in the gene space.", indent=True)

    m.heading("p53 activity", 2)
    m.para("p53 activity was scored with 14 consensus p53 target genes[@fischer2017] measured in all four lines "
           "(MDM2, BAX, TP53I3, RRM2B, ZMAT3, FDXR, AEN, TRIAP1, XPC, RPS27L, CCNG1, GADD45A, PPM1D, APAF1; "
           "CDKN1A and GDF15 are not among the shared measured genes). Each gene was z-scored against the line's "
           "control cells and the score was the mean z over the 14 genes. A knockdown was called p53-inducing when "
           "its cells' score exceeded the controls' (Welch t-test, BH FDR < 0.05, positive shift). TP53 status: "
           "K562 TP53 null, Jurkat TP53 mutant (DepMap damaging-mutation calls), HepG2 and RPE1 TP53 wild "
           "type.", indent=True)

    m.heading("Simulation (positive control)", 2)
    m.para("Four synthetic lines shared 160 perturbations and 400 genes. Perturbation effects were sparse "
           "combinations of 12 gene programs (program loadings ~ N(0, 0.5^2) with 30% non-zero; program "
           "weights ~ N(0, 1) with 15% non-zero). Each line observed a random 80% of the perturbations, with 24 "
           "cells per perturbation and 600 controls drawn from N(baseline + effect, 0.4^2), clipped at 0. In the "
           "'unlearnable' scenario each line added independent noise to the loadings (N(0, 0.25^2) on 20% of "
           "entries). In the 'learnable' scenario a gene's effect in a line was the shared effect scaled by its "
           "baseline expression in that line (baselines ~ Gamma(1.2, 0.9) with 25% of genes silent), which "
           "is observable from control cells. All methods were run on the HepG2-like line with the "
           "under-represented split (25% + 5% screened) and 60 epochs.", indent=True)

    m.heading("Dependency analyses", 2)
    m.para("Chronos gene effects[@dempster2021] (DepMap 24Q4[@tsherniak2017,meyers2017]) were taken for "
           "K562 (ACH-000551), HepG2 (ACH-000739) and Jurkat (ACH-000995); RPE1 is not profiled. Response "
           "magnitude is $\\log(1 + n_{DEG})$. The context-specific response of a knockdown in line $\\ell$ is "
           "its magnitude minus the mean over the other two cancer lines, and the context-specific "
           "dependency is defined the same way on Chronos scores. For dependency prediction (H5), a ridge "
           "model was trained, within 10-fold cross-validation over the held-out perturbations of each "
           "cancer line, to map *observed* shifts to Chronos. It was then applied unchanged to the shifts "
           "predicted by each method for the held-out fold. As a control for line specificity, each line's "
           "response magnitude was also correlated with the other lines' Chronos scores, and each line's Chronos "
           "scores were predicted by the mean Chronos score of the other two lines.", indent=True)


def reproducibility(m: Manuscript) -> None:
    m.heading("Software, computation and reproducibility", 2)
    m.para("All analyses were run on a single workstation (4 CPU cores, 8 GB RAM, no GPU) with Python 3.12, "
           "PyTorch 2.13 (CPU build)[@paszke2019], NumPy[@harris2020], SciPy[@virtanen2020], "
           "scikit-learn[@pedregosa2011], pandas, h5py and Matplotlib[@hunter2007]. Under-represented and "
           "zero-shot folds took 19-56 min each to train and evaluate (Table S9); single-source transfer models about "
           "10 min each. Two jobs ran concurrently with two threads each. All randomness (cell subsampling, "
           "control-pool splits, perturbation splits, model initialisation, mini-batch sampling and the control "
           "cells given to each method) is seeded. Control input sets are derived deterministically from a hash "
           "of the perturbation, seed and context, so that every method receives identical inputs. The pipeline "
           "is a sequence of command-line steps (data build, observed DE, T1, T2, ablations, transfer, analysis, "
           "figures and manuscript), and each step writes versioned artefacts (per-perturbation metric tables, "
           "predicted pseudobulk shifts, per-gene DE arrays and training histories). Every number in this paper "
           "is read from a single results file generated from those artefacts, and every figure panel writes "
           "the table it plots.", indent=True)
    m.para("The code base has unit and integration tests. They cover Benjamini-Hochberg correction against "
           "SciPy, the DE test on simulated shifts, perturbation discrimination at its bounds, the energy-MMD "
           "loss (zero for identical sets, positive for shifted ones), permutation equivariance of CASTER, "
           "split disjointness and determinism, control-pool disjointness, and an end-to-end synthetic run "
           "in which CASTER and PerturbMean must recover planted effects. Data integrity was verified at "
           "acquisition: every selected row was fetched exactly once in the final dataset, and 112,310 rows "
           "fetched twice by overlapping download passes were confirmed byte-identical before deduplication.",
           indent=True)
    m.heading("Deviations from the pre-specified protocol", 2)
    m.bullets([
        "**AMENDMENT-001** (before any result was seen): to fit the compute budget, training epochs were "
        "set to 20 for context tasks and 10 for ablations and transfer, and the set-size ablation was "
        "reduced to S in {1, 8, 32}.",
        "**AMENDMENT-002** (after T1 results were first inspected): the predicted-side DE test was replaced "
        "by a moment-matched z-test because the pre-specified Wilcoxon test is invalid for continuous "
        "predictions (STAR Methods; Figure S7). Both versions of every affected test are reported.",
        "**Data source** (before freezing): the knockdown-filtered four-line release[@adduri2026] replaced the "
        "per-line original files, for uniform processing and because the original repository throttled "
        "range requests.",
        "Seeds 1 and 2 of the under-represented task (declared secondary) were run in the review round.",
        "**AMENDMENT-003** (review round, 2026-10-08, after all pre-specified results were known): added the "
        "LineOffset baseline, a line-identity-only CASTER, wrong-line control cells, complex-grouped splits, "
        "ablations at 20 epochs with two seeds, calibrated metrics with technical and interpolated duplicates "
        "and noise ceilings[@miller2026,mejia2025], the p53 and DepMap control analyses, and corrected the "
        "direction of the paired test for lower-is-better metrics (MAE), which had been tested as "
        "higher-is-better. All are exploratory and labelled as added later. Predicted-side Wilcoxon tests were "
        "not recomputed for the new runs because AMENDMENT-002 replaced them.",
    ])


def quantification(m: Manuscript) -> None:
    m.heading("Quantification and statistical analysis", 1)
    m.heading("Evaluation suite (PertBench)", 2)
    m.para("For test perturbation $t$ with observed cells $X_t$, predicted cells $\\hat X_t$ (same "
           "number of cells) and reference controls $R$ (500 cells, disjoint from model inputs):",
           indent=True)
    m.para("**Pearson delta.** The correlation between predicted and observed pseudobulk shifts:", indent=True)
    m.equation(r"\rho_t = \mathrm{corr}(\bar{\hat X}_t - \bar R,\ \bar X_t - \bar R)")
    m.para("**Perturbation discrimination.** Following Wu et al.[@wu2025] and Adduri et al.[@adduri2026], "
           "with $r_t$ the number of other test perturbations whose observed pseudobulk is closer in L1 "
           "distance to the prediction than the true one:", indent=True)
    m.equation(r"\mathrm{PDisc}_t = 1 - \frac{2\,r_t}{T-1},\qquad r_t = \sum_{t'\neq t}\mathbb{1}\left[\|\bar{\hat X}_t - \bar X_{t'}\|_1 < \|\bar{\hat X}_t - \bar X_t\|_1\right]")
    m.para("A perfect model scores 1 and chance scores 0. **Observed differential expression (ground "
           "truth).** Observed cells of each perturbation are tested against $R$ gene by gene with the "
           "Wilcoxon rank-sum test[@wilcoxon1945,mann1947] (asymptotic, tie-corrected), with "
           "Benjamini-Hochberg correction[@benjamini1995] at FDR < 0.05. Observed log2 fold changes are "
           "log2 ratios of mean expm1-transformed expression with pseudocount 0.01. The limitations of "
           "cell-level DE tests are well known[@squair2021,soneson2018].", indent=True)
    m.para("**Predicted differential expression (AMENDMENT-002).** The pre-specified protocol applied the "
           "same Wilcoxon test to predicted cells. Inspection of the results showed that this is invalid "
           "for continuous predictions. Adding a shift to a control cell turns each of its exact zeros "
           "into a small positive value, and against zero-inflated reference cells the rank test then "
           "calls most genes significant whatever the shift. In the HepG2 fold, every shifted method "
           "produced 1,283-1,494 'DEGs' per perturbation, versus 0.1 for unshifted control cells. We "
           "therefore score predictions with a moment-matched z-test on the predicted pseudobulk shift "
           "$\\hat\\delta_g$, using the reference-control standard deviation $s_g$ and the sample sizes "
           "of the observed experiment:", indent=True)
    m.equation(r"z_g = \frac{\hat\delta_g}{s_g\sqrt{1/n + 1/m}},\qquad p_g = 2\,\Phi(-|z_g|)")
    m.para("Here $n$ is the number of perturbed cells and $m = 500$, with BH adjustment. Predicted "
           "fold-change ranks and signs use $\\hat\\delta_g$. Observed ground truth is unchanged. The "
           "pre-specified Wilcoxon results are reported in full in Table S1 and flagged as invalid. "
           "**DE AUPRC** is the average precision[@davis2006,saito2015] for recovering observed DEGs "
           "from predicted -log10 adjusted p values. **DE overlap** is", indent=True)
    m.equation(r"\mathrm{Overlap}_t = \frac{|G^{(N)}_{t,\mathrm{true}} \cap G^{(N)}_{t,\mathrm{pred}}|}{N},\quad N = |G^{\mathrm{DE}}_{t,\mathrm{true}}|")
    m.para("**MAE and MSE** are the mean absolute and mean squared differences between predicted and observed "
           "pseudobulk shifts over the 2,000 modelled genes, $\\frac{1}{G}\\sum_g|\\hat\\delta_g - \\delta_g|$ and "
           "$\\frac{1}{G}\\sum_g(\\hat\\delta_g - \\delta_g)^2$; lower is better, and paired tests are oriented "
           "accordingly.", indent=True)
    m.para("**Calibrated metrics (review round).** Following Miller et al.[@miller2026] and Mejia et "
           "al.[@mejia2025], each held-out perturbation's cells were split into two random halves (seeded per "
           "perturbation). Half A defines the truth; half B is the technical duplicate (positive control). A "
           "Welch t-test of half A against all other cells of the line gives per-gene statistics $t_g$ and "
           "BH-adjusted p values $q_g$. DEG weights are $w_g = ((|t_g| - \\min|t|)/(\\max|t| - \\min|t|))^2$, "
           "the interpolated duplicate is $\\alpha_g\\,\\delta^{B}_g + (1-\\alpha_g)\\,\\bar\\delta$ with "
           "$\\alpha_g = 1 - q_g$, and the negative control $\\bar\\delta$ is the mean shift of the line's "
           "screened perturbations (the perturbed mean). Metrics: Pearson delta; Pearson delta relative to the "
           "perturbed mean on the 100 genes with largest $|t_g|$; MSE; MAE; WMSE "
           "$\\sum_g w_g(\\hat\\delta_g-\\delta_g)^2/\\sum_g w_g$; weighted $R^2$ relative to control and to the "
           "perturbed mean (clipped at -1); NIR (fraction of other held-out perturbations whose truth is farther, "
           "in Euclidean distance, from the prediction than the correct truth) and PDS (the same with Manhattan "
           "distance). The dynamic range fraction of a metric is (technical duplicate - negative control) / "
           "(perfect - negative control), or (negative control - technical duplicate) / negative control for "
           "lower-is-better metrics, clipped to [-1, 1] and averaged over perturbations; for metrics relative to "
           "the perturbed mean the zero-shift prediction is the negative control. Methods are also reported on a "
           "scale from negative control (0) to technical duplicate (1).", indent=True)
    m.para("**Noise-free re-scoring (review round).** Every method's prediction is rendered as cells derived from "
           "$n$ control cells sampled from the input pool ($n$ = number of observed cells of the perturbation), so "
           "its pseudobulk shift contains the sampling noise $\\bar x^{0}_{\\mathrm{sample}} - \\bar x^{0}_{\\mathrm{pool}}$. "
           "We subtracted this term from every method's predicted shift (for CASTER, this treats its output "
           "relative to its own input as the predicted effect) and re-computed all pseudobulk metrics. These "
           "'noise-free' scores are reported alongside the pre-specified ones.", indent=True)
    m.para("**Noise ceiling.** The Pearson correlation $r_h$ between the shifts of the two halves estimates the "
           "split-half reliability; by Spearman-Brown, the reliability of the full-cell shift is "
           "$\\rho = 2r_h/(1+r_h)$, and the highest Pearson delta any predictor can reach against the full-cell "
           "truth is $\\sqrt{\\rho}$.", indent=True)
    m.para("Here both sets are ranked by |log2 FC| among significant genes. **Fold-change Spearman** is "
           "computed over observed DEGs. **Direction agreement** is the fraction of observed DEGs with "
           "the correct predicted sign. **Effect-size Spearman** is the rank correlation, across "
           "perturbations within a fold, between the predicted and observed numbers of DEGs.", indent=True)
    m.heading("Statistics", 2)
    m.para("All hypotheses, tests and the decision rule were fixed in a protocol (RESEARCH_PROTOCOL.md, git "
           "tag protocol-v1.0) before any model was fitted to real data. H1 used one-sided paired Wilcoxon "
           "signed-rank tests over held-out perturbations pooled across the four folds (CASTER > baseline), "
           "with Holm correction[@holm1979] over nine tests (three baselines x three primary metrics). It "
           "was declared supported if at least six were significant and none was significant in the "
           "opposite direction. H2 used the same test against PerturbMean on T2 (Holm over three). H3 "
           "compared per-perturbation single-source transfer accuracy averaged over cancer-to-cancer pairs "
           "with the average over pairs involving RPE1 (two-sided paired Wilcoxon). H4 and H4b used "
           "Spearman correlations with Holm correction over the three cancer lines. Exploratory analyses "
           "are reported without multiplicity claims. Every number in the text is read from the "
           "results.json file written by the analysis pipeline.", indent=True)
