"""STAR Methods: formal properties of CASTER and of the context-transfer problem."""

from __future__ import annotations

from canpert.docxkit import Manuscript


def theory(m: Manuscript) -> None:
    m.heading("Formal properties of CASTER and of context transfer", 2)
    m.para("We state four properties that frame the benchmark results. Notation follows the problem "
           "formulation above. $P^{0}_{t}$ denotes the control distribution of context $t$, "
           "$d = d_{t,p}$ the transfer prior, and $\\theta$ the CASTER parameters.", indent=True)

    m.heading("Proposition 1 (CASTER contains PerturbMean)", 3)
    m.para("At initialisation ($g \\equiv 0$, and the final linear layer of $f_{out}$ equal to zero), for any "
           "control set $X^{0}$ and perturbation $p$, the CASTER prediction is", indent=True)
    m.equation(r"\hat X = X^{0} + \mathbf{1}\, d_{t,p}^{\top}")
    m.para("Its pseudobulk shift therefore equals the PerturbMean prediction exactly. *Proof.* The output "
           "is $X^{0} + \\mathbf{1}[(1+g(c))\\odot d]^{\\top} + f_{out}(\\cdot)$. With $g = 0$ the middle term "
           "is $\\mathbf{1}d^{\\top}$, and $f_{out}(z) = W z + b$ with $W = 0$ and $b = 0$ is identically zero. "
           "Because the checkpoint is chosen by validation loss, which includes the epoch-0 model, the "
           "selected CASTER can deviate from PerturbMean only when the deviation lowers the validation "
           "objective. \u220e", indent=True)

    m.heading("Proposition 2 (permutation equivariance)", 3)
    m.para("For any permutation matrix $\\Pi \\in \\{0,1\\}^{S\\times S}$, "
           "$\\mathrm{CASTER}(\\Pi X^{0}, p) = \\Pi\\, \\mathrm{CASTER}(X^{0}, p)$. *Proof.* $f_{cell}$ acts row-wise "
           "and commutes with $\\Pi$. The context summary $c$ is a mean over rows and is invariant. The FiLM and "
           "prior terms are broadcast identically to every row. Self-attention without positional "
           "encodings is permutation-equivariant[@lee2019,zaheer2017]. $f_{out}$ and the residual act row-wise. "
           "A composition of equivariant and row-broadcast invariant maps is equivariant. \u220e The "
           "property is verified numerically in the test suite. It means that predictions depend on the "
           "*distribution* of control cells, not on their order.", indent=True)

    m.heading("Proposition 3 (the training loss identifies distributions)", 3)
    m.para("For distributions $P, Q$ on $\\mathbb{R}^{G}$ with finite first moments, the energy "
           "distance[@szekely2013]", indent=True)
    m.equation(r"\mathcal{E}(P,Q) = 2\,\mathbb{E}\|Y - Z\| - \mathbb{E}\|Y - Y'\| - \mathbb{E}\|Z - Z'\|,\quad Y,Y'\sim P,\ Z,Z'\sim Q")
    m.para("is non-negative and equals zero if and only if $P = Q$. It is the squared MMD with the "
           "(conditionally negative definite) energy kernel[@gretton2012,feydy2019]. The empirical "
           "V-statistic in the loss converges almost surely to $\\mathcal{E}$ as $S \\to \\infty$. Minimising "
           "the population loss therefore matches the full predicted and observed perturbed distributions, "
           "not only their means. The added pseudobulk term is minimised by the conditional mean, so the "
           "combined objective is minimised by any predictor whose distribution equals the observed one, "
           "and both terms vanish there. \u220e", indent=True)

    m.heading("Proposition 4 (when can context transfer beat PerturbMean?)", 3)
    m.para("Decompose the target shift as $\\delta_{t,p} = d_{t,p} + r_{t,p}$, where $r_{t,p}$ is the "
           "residual of the transfer prior. Let $\\mathcal{F}_{t}$ be the information available to a "
           "predictor at test time: the prior, the control cells of $t$ and, in T1, the support "
           "perturbations of $t$. Under squared error, the optimal prediction of $\\delta_{t,p}$ is", indent=True)
    m.equation(r"\delta^{\star}_{t,p} = d_{t,p} + \mathbb{E}\left[r_{t,p} \mid \mathcal{F}_{t}\right]")
    m.para("Its risk improves on that of PerturbMean by exactly "
           "$\\mathbb{E}\\|\\mathbb{E}[r_{t,p}\\mid\\mathcal{F}_t]\\|^{2}$. *Proof.* By the orthogonality of "
           "conditional expectation, "
           "$\\mathbb{E}\\|r - h(\\mathcal{F})\\|^{2} = \\mathbb{E}\\|r - \\mathbb{E}[r\\mid\\mathcal{F}]\\|^{2} + "
           "\\mathbb{E}\\|\\mathbb{E}[r\\mid\\mathcal{F}] - h(\\mathcal{F})\\|^{2}$ for any measurable $h$. "
           "PerturbMean is $h \\equiv 0$. \u220e Consequently, *no* method, however expressive, can "
           "outperform PerturbMean in expectation unless the residual of the transfer prior is predictable "
           "from the target's control cells and support perturbations. The comparison between CASTER (and "
           "the linear transfer model) and PerturbMean is therefore an empirical estimate of "
           "$\\|\\mathbb{E}[r_{t,p}\\mid\\mathcal{F}_t]\\|$. A null result means that this predictable part is "
           "small relative to estimation noise at the available data scale, not that context does not "
           "matter. Indeed $\\mathbb{E}\\|r_{t,p}\\|^{2}$ (the context-specific variance quantified in Figure 1) can be "
           "large while its predictable part is small. In the positive-control simulation the residual is "
           "constructed to be a function of the control profile, so this bound is attained.", indent=True)

    m.heading("Information available to each method (leakage audit)", 2)
    m.table(["Method", "Target controls", "Target support perts", "Source contexts", "Target test perts"], [
        ["NoChange", "input pool", "-", "-", "never"],
        ["ContextMean", "input pool", "yes (mean)", "-", "never"],
        ["PerturbMean", "input pool", "-", "yes", "never"],
        ["Linear transfer", "input pool", "yes (fit)", "yes", "never"],
        ["CASTER (T1)", "input pool", "yes (train/val)", "yes", "never"],
        ["CASTER (T2/T3)", "input pool", "-", "yes (val: 5% source)", "never"],
        ["Evaluation", "reference pool", "-", "-", "observed cells"],
    ], widths=[1.3, 1.2, 1.3, 1.5, 1.2])
    m.para("Control cells of each context are split once (seed 0) into disjoint input and reference pools. "
           "Gene selection uses control cells only. Every perturbation in the held-out test set is "
           "excluded from all training and model-selection steps for that context. This is enforced in "
           "code and covered by unit tests (split disjointness, control-pool disjointness, deterministic "
           "inputs).", indent=True)
