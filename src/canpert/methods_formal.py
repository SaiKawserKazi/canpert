"""STAR Methods: formal definitions of baselines and downstream analyses."""

from __future__ import annotations

from canpert.docxkit import Manuscript


def baselines_formal(m: Manuscript) -> None:
    m.heading("Formal definition of baseline models", 2)
    m.para("Let $\\bar x^{0}_{c}$ be the mean of the model-input control cells of context $c$, and "
           "$\\bar x^{p}_{c}$ the mean of the observed cells of perturbation $p$ in $c$. The observed shift is "
           "$\\hat\\delta_{c,p} = \\bar x^{p}_{c} - \\bar x^{0}_{c}$. For the target context $t$, let "
           "$\\mathcal{S}_t$ be its support perturbations (training + validation) and $\\mathcal{C}$ the "
           "source contexts.", indent=True)
    m.para("**PerturbMean** averages the shift of $p$ over the source contexts that observed it:", indent=True)
    m.equation(r"\hat\delta^{\mathrm{PM}}_{t,p} = \frac{1}{|\mathcal{C}_p|}\sum_{c\in\mathcal{C}_p}\hat\delta_{c,p}")
    m.para("**ContextMean** predicts the same shift for every perturbation of $t$, namely the mean shift "
           "of its support perturbations:", indent=True)
    m.equation(r"\hat\delta^{\mathrm{CM}}_{t,p} = \frac{1}{|\mathcal{S}_t|}\sum_{q\in\mathcal{S}_t}\hat\delta_{t,q}")
    m.para("**Linear transfer.** Each perturbation is described by the concatenation of its source shifts, "
           "$u_p = [\\hat\\delta_{c_1,p}; \\ldots; \\hat\\delta_{c_K,p}] \\in \\mathbb{R}^{KG}$. A source that did "
           "not observe $p$ contributes $\\hat\\delta^{\\mathrm{PM}}_{t,p}$. The features are reduced by PCA, "
           "fitted on source data only, to $k = \\min(64, |\\mathcal{S}_t| - 1)$ components $v_p = W^{\\top}"
           "(u_p - \\bar u)$. A ridge map $B \\in \\mathbb{R}^{k\\times G}$ is then fitted on the support "
           "perturbations:", indent=True)
    m.equation(r"\hat B = \arg\min_{B,b}\sum_{q\in\mathcal{S}_t}\|\hat\delta_{t,q} - B^{\top}v_q - b\|_2^2 + \lambda\|B\|_F^2"
               r" = (V^{\top}V + \lambda I)^{-1}V^{\top}\tilde\Delta")
    m.para("Here $V$ stacks the centred $v_q$ and $\\tilde\\Delta$ the centred target shifts. $\\lambda$ is "
           "chosen from 13 log-spaced values in $[10^{-2}, 10^{4}]$ by efficient leave-one-out "
           "cross-validation. The prediction is $\\hat\\delta^{\\mathrm{Lin}}_{t,p} = \\hat B^{\\top}v_p + "
           "\\hat b$. **NoChange** predicts $\\hat\\delta = 0$. Every pseudobulk prediction is rendered as "
           "cells by adding it to the same control cells, $\\hat X_{t,p} = X^{0}_{t,p} + "
           "\\mathbf{1}\\hat\\delta_{t,p}^{\\top}$.", indent=True)


def downstream_formal(m: Manuscript) -> None:
    m.heading("Formal definition of downstream analyses", 2)
    m.para("**Inverse design.** For the 100 perturbations with the most observed DEGs in a context, the "
           "rank of the correct perturbation among all predictions is", indent=True)
    m.equation(r"\mathrm{rank}_t = \sum_{t'}\mathbb{1}\left[\|\bar X_t - \bar{\hat X}_{t'}\|_1 < \|\bar X_t - \bar{\hat X}_t\|_1\right],"
               r"\quad \mathrm{acc}@k = \frac{1}{100}\sum_t\mathbb{1}[\mathrm{rank}_t < k]")
    m.para("**STRING-pair recovery.** For perturbations with >= 10 observed DEGs, each unordered pair "
           "$(p, q)$ is labelled positive if STRING v12 reports a combined score >= 700 between their "
           "products. It is scored by the cosine similarity of the predicted shifts, "
           "$s_{pq} = \\cos(\\hat\\delta_p, \\hat\\delta_q)$, and summarised by the AUROC.", indent=True)
    m.para("**Gene programs.** Let $Z$ be the observed shift matrix (perturbations x genes), z-scored per "
           "gene. K-means with $K = 15$ (10 restarts, seed 0) on the rows of $Z^{\\top}$ assigns each gene to "
           "a program $k$. The activity of program $k$ under perturbation $p$ is "
           "$a_{pk} = \\frac{1}{|G_k|}\\sum_{g\\in G_k}\\hat\\delta_{p,g}$. Concordance is the Pearson "
           "correlation across perturbations of predicted and observed $a_{\\cdot k}$. Programs are annotated "
           "by the Reactome pathway with the smallest hypergeometric p value, with the modelled genes as the "
           "background and at least 3 overlapping genes.", indent=True)
    m.para("**Perturbation-similarity structure.** Cosine-distance matrices of predicted and observed shifts "
           "(perturbations with >= 10 DEGs) were clustered by average linkage into $k = 2, \\ldots, 20$ "
           "clusters, and the clusterings were compared by the adjusted Rand index.", indent=True)
    m.para("**Dependency prediction (H5).** Within each cancer line, held-out perturbations with a measured "
           "Chronos score were split into 10 folds. A ridge model $f$ (penalty chosen by leave-one-out "
           "cross-validation over 13 values in $[10^{-1}, 10^{5}]$) was fitted on the *observed* shifts of "
           "9 folds and applied to the observed and to each method's predicted shifts of the remaining fold. "
           "The score is the Pearson correlation between the out-of-fold predictions $f(\\hat\\delta_p)$ and the "
           "measured Chronos scores. Because the regressor never sees a predicted shift during training, any "
           "loss of accuracy reflects information missing from the prediction.", indent=True)
    m.para("**Context-specific dependency (H4b).** For gene $p$ observed in all three cancer lines, with "
           "response magnitude $r_{\\ell,p} = \\log(1 + n^{\\mathrm{DEG}}_{\\ell,p})$ and Chronos score "
           "$c_{\\ell,p}$, the context-specific components are", indent=True)
    m.equation(r"\Delta r_{\ell,p} = r_{\ell,p} - \frac{1}{2}\sum_{\ell'\neq\ell} r_{\ell',p},\qquad"
               r"\Delta c_{\ell,p} = c_{\ell,p} - \frac{1}{2}\sum_{\ell'\neq\ell} c_{\ell',p}")
    m.para("H4b tests the Spearman correlation of $\\Delta r_{\\ell,\\cdot}$ and $\\Delta c_{\\ell,\\cdot}$ "
           "per line.", indent=True)
    m.para("**Response similarity across contexts.** For each pair of contexts and each perturbation with "
           ">= 10 DEGs in both, similarity is the Pearson correlation of the two observed shift vectors over "
           "the modelled genes. DEG overlap is the Jaccard index of the two DEG sets. The stratified analysis "
           "repeats this for minimum response sizes of 10, 25, 50, 100 and 200 DEGs.", indent=True)
