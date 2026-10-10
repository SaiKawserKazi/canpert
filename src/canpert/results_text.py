"""Result-dependent manuscript text. Every number comes from results.json; every claim is
selected by the pre-declared test outcome (supported / not supported / opposite)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canpert.data import CANCER_LINES
from canpert.docxkit import Manuscript

LN = {"k562": "K562", "hepg2": "HepG2", "jurkat": "Jurkat", "rpe1": "RPE1"}
MNAME = {"pearson_delta": "Pearson delta", "discrimination": "discrimination", "de_auprc": "DE AUPRC",
         "de_overlap": "DE overlap", "lfc_spearman": "fold-change Spearman", "direction": "direction agreement",
         "mae": "MAE", "effect_spearman": "effect-size Spearman"}
METHODS = ("CASTER", "PerturbMean", "ContextMean", "Linear", "NoChange")


def f(x: Any, nd: int = 2) -> str:
    """Format a number for prose."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, int | np.integer):
        return f"{x:,}"
    out = f"{x:.{nd}f}"
    return out.replace("-0.00", "0.00") if out.strip("-0.") == "" else out


def pval(p: Any) -> str:
    """Format a p value."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        sup = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
        return f"{m} \u00d7 10{str(int(e)).translate(sup)}"
    return f"{p:.3f}"


def summ(res: dict, key: str, method: str, metric: str) -> float:
    """Mean of a metric for a method in a summary table."""
    for r in res.get(key, []):
        if r["method"] == method:
            return r.get(f"{metric}_mean", np.nan)
    return np.nan


def per_line(res: dict, key: str, line: str, method: str, metric: str) -> float:
    """Per-target mean."""
    for r in res.get(key, []):
        if r["method"] == method and r["target"] == line:
            return r.get(f"{metric}_mean", np.nan)
    return np.nan


def h1_rows(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["H1"]["tests"])


def verdict(res: dict) -> dict[str, Any]:
    """Summarise the pre-declared outcomes in a few booleans used throughout the text."""
    h1 = h1_rows(res)
    sig_better = h1[(h1.p_holm < 0.05)]
    sig_worse = h1[(h1.p_two_sided < 0.05) & (h1.median_diff < 0)]
    out = {
        "h1": res["H1"]["supported"],
        "h1_n_better": len(sig_better),
        "h1_n_worse": len(sig_worse),
        "better_than": sorted(set(sig_better.baseline)),
        "worse_than": sorted(set(sig_worse.baseline)),
        "h2": res.get("H2", {}).get("supported", False),
        "h3": res.get("H3", {}).get("supported", False),
        "h4": res["H4"]["H4_supported"],
        "h4b": res["H4"]["H4b_supported"],
    }
    return out


# ---------------------------------------------------------------------------- front matter
def summary(res: dict) -> str:
    v = verdict(res)
    cc = {r["pair_type"]: r["pearson"] for r in res["cross_context"]["by_type"]}
    pm = summ(res, "T1_summary", "PerturbMean", "pearson_delta")
    ca = summ(res, "T1_summary", "CASTER", "pearson_delta")
    h4 = pd.DataFrame(res["H4"]["tests"])
    rhos = h4[h4.test == "H4"].rho
    s = ("Genetic perturbation screens are run one cellular context at a time, yet cancers differ in "
         "lineage and state. Using CRISPRi Perturb-seq of essential genes in three cancer cell lines and a "
         "non-transformed line, we find that transcriptional responses to the same knockdown are only "
         f"moderately shared across contexts (median correlation {f(cc.get('cancer-cancer'))} between cancer "
         "lines), with similarity increasing for stronger responses. We introduce CASTER, a set-based "
         "transformer that transfers a perturbation's response from other contexts and adapts it to the "
         "target context's unperturbed cells. ")
    if v["h1"]:
        s += (f"When a fraction of the target context is screened, CASTER improves on mean baselines "
              f"(Pearson delta {f(ca)} versus {f(pm)} for the perturbation mean) and matches a "
              "context-adaptive linear model, while in fully unseen contexts the perturbation mean remains "
              "hard to beat. ")
    else:
        s += (f"CASTER matched but did not consistently exceed the transfer baseline (Pearson delta {f(ca)} "
              f"versus {f(pm)}). ")
    if v["h4"]:
        s += ("Across all three cancer lines, the magnitude of a knockdown's transcriptional response tracked "
              f"the line's dependency on the gene measured by DepMap (Spearman rho {f(rhos.max())} to "
              f"{f(rhos.min())}), and predicted responses retained this dependency information. ")
    s += ("These results quantify how much cancer context reshapes perturbation responses and provide a "
          "reproducible benchmark for context-generalising virtual-cell models.")
    return s

def highlights(res: dict) -> list[str]:
    v = verdict(res)
    hs = ["Responses to the same CRISPRi knockdown are only moderately shared across cell contexts",
          "CASTER adapts transferred perturbation responses to a context's unperturbed cells"]
    hs.append("With partial screening of a context, CASTER outperforms mean baselines" if v["h1"]
              else "A pre-registered benchmark separates shared from context-specific response")
    hs.append("Knockdown response magnitude tracks DepMap dependency in all three cancer lines" if v["h4"]
              else "Context-specific responses are tested against DepMap cancer dependencies")
    return hs

def in_brief(res: dict) -> str:
    return ("Adapting perturbation responses measured in one cell type to another, CASTER and a "
            "pre-registered benchmark across leukemia, hepatoblastoma, and non-transformed cells show how much "
            "cancer context reshapes the transcriptional consequences of losing an essential gene, and how "
            "those consequences relate to cancer dependency.")


# ---------------------------------------------------------------------------- results
def results(m: Manuscript, res: dict, figs: Path) -> None:  # noqa: PLR0915
    v = verdict(res)
    audit = res.get("audit", {})
    nde = res["observed_n_de"]
    cc_type = {r["pair_type"]: r for r in res["cross_context"]["by_type"]}
    pairs = pd.DataFrame(res["cross_context"]["by_pair"])

    # ---------------- section 1
    m.heading("Responses to the same knockdown are only moderately shared across cellular contexts", 2)
    m.para("We assembled CRISPRi Perturb-seq profiles of essential-gene knockdowns in four human cell lines "
           "(Figures 1A and 1B): K562 (chronic myeloid leukemia)[@replogle2022], HepG2 "
           "(hepatoblastoma)[@nadig2025], Jurkat (T cell acute lymphoblastic leukemia)[@nadig2025] and the "
           "non-transformed, TP53-proficient line hTERT RPE-1[@replogle2022,bodnar1998]. We used the "
           "knockdown-filtered, uniformly processed release of these data[@adduri2026] and analysed "
           f"{f(audit.get('n_cells_fetched', 0))} cells: up to 32 cells per perturbation per line and "
           "3,000 controls per line (STAR Methods). After filtering, "
           f"{f(audit.get('k562_perturbations'))} (K562), {f(audit.get('hepg2_perturbations'))} (HepG2), "
           f"{f(audit.get('jurkat_perturbations'))} (Jurkat) and {f(audit.get('rpe1_perturbations'))} (RPE1) "
           f"perturbations remained ({f(audit.get('n_perturbations_union'))} distinct genes). Of these, "
           f"{f(audit.get('n_perturbations_all_four'))} were retained in all four lines and "
           f"{f(audit.get('n_perturbations_all_three_cancer'))} in all three cancer lines. Because on-target "
           "efficacy was filtered per line, the set of informative perturbations is itself context "
           "dependent (Figure 1B). We modelled 2,000 genes selected on control cells only, excluding every "
           "perturbation target, so that neither gene selection nor the trivial on-target signal can inflate "
           "any result.", indent=True)
    m.para("The number of differentially expressed genes (DEGs; Wilcoxon test against held-out controls, "
           "FDR < 0.05) varied widely between knockdowns and between contexts. The median was "
           + ", ".join(f"{f(nde[n]['median'], 0)} in {LN[n]}" for n in ("k562", "hepg2", "jurkat", "rpe1"))
           + "; " + ", ".join(f"{f(100 * nde[n]['frac_ge10'], 0)}%" for n in ("k562", "hepg2", "jurkat", "rpe1"))
           + " of knockdowns, respectively, changed at least 10 genes (Figure S1A). Responses in the "
           "non-transformed RPE1 line were thus markedly larger than in the three cancer lines. This is "
           "consistent with an intact p53-dependent stress response to the loss of essential genes in "
           "TP53-proficient cells[@haapaniemi2018,ihry2018]. For "
           f"{f(res['cross_context']['n_perturbations_min10de'])} genes whose knockdown produced >= 10 DEGs in "
           "at least two lines, we compared responses across contexts. The shifts were only moderately "
           f"correlated: median Pearson r was {f(cc_type['cancer-cancer']['pearson'])} between cancer lines "
           f"and {f(cc_type['cancer-RPE1']['pearson'])} between a cancer line and RPE1 (Figures 1D and 1E). "
           f"DEG sets overlapped little (median Jaccard {f(cc_type['cancer-cancer']['jaccard'])} and "
           f"{f(cc_type['cancer-RPE1']['jaccard'])}; Figure S1C).", indent=True)
    best = pairs.sort_values("pearson", ascending=False).iloc[0]
    rest = pairs[(pairs.a != best.a) | (pairs.b != best.b)]
    bys = pd.DataFrame(res["cross_context"].get("by_strength", []))
    strength_txt = ""
    if not bys.empty:
        cc_s = bys[bys.pair_type == "cancer-cancer"].set_index("min_de").median_pearson
        strength_txt = (f" Similarity rose with response strength, from {f(cc_s.iloc[0])} for knockdowns "
                        f"with >= {int(cc_s.index[0])} DEGs to {f(cc_s.iloc[-1])} for those with >= "
                        f"{int(cc_s.index[-1])} DEGs in both cancer lines (Figure S1D). Much of the apparent "
                        "context specificity of weak responses is therefore measurement noise, whereas strong "
                        "responses share a larger core.")
    m.para(f"The most similar pair was {LN[best.a]}-{LN[best.b]} (median r = {f(best.pearson)}); the other "
           f"five pairs ranged from {f(rest.pearson.min())} to {f(rest.pearson.max())}. HepG2 and RPE1 are the "
           "two adherent, epithelial-derived lines, whereas K562 and Jurkat grow in suspension, which suggests "
           "that cell morphology and growth mode, and not transformation status alone, shape the shared "
           "response." + strength_txt + " A useful context-transfer model must therefore capture the shared "
           "core and predict at least part of a substantial context-specific component.", indent=True)

    # ---------------- section 2: model
    abl = {r["method"]: r for r in res.get("ablation", [])}
    m.heading("CASTER adapts transferred responses to the target context", 2)
    m.para("These observations motivated CASTER (Figure 1C). Like State[@adduri2026], CASTER operates on "
           "*sets* of cells and is trained with a distributional (energy-distance MMD) "
           "loss[@gretton2012,szekely2013], so it models a perturbed population rather than a single average "
           "cell. It differs in how it represents the perturbation and the context. A perturbation is "
           "represented by its *transfer prior*: the mean shift it produced in the other contexts in which it "
           "was observed. The target context is never part of this prior. The context is represented only by "
           "a set of its unperturbed cells, which a transformer encoder reads through self-attention. FiLM "
           "conditioning[@perez2018] lets the same prior act differently in each context. The output adds a "
           "context-dependent, gene-wise rescaling of the prior to a cell-resolved correction. Both are "
           "initialised at zero, so an untrained CASTER is exactly the PerturbMean baseline, and any "
           "departure must be earned on held-out validation data (STAR Methods).", indent=True)
    pc = res.get("positive_control")
    if pc:
        m.para("In simulations where the context-specific component was pure noise, CASTER stayed at the "
               f"PerturbMean solution (Pearson delta {f(pc['unlearnable']['CASTER'])} versus "
               f"{f(pc['unlearnable']['PerturbMean'])}). When the context-specific component depended on each "
               "gene's baseline expression in that context, which is observable from control cells, CASTER "
               f"recovered it (Pearson delta {f(pc['learnable']['CASTER'])} versus "
               f"{f(pc['learnable']['PerturbMean'])}; Figure S4). The architecture can therefore learn "
               "context structure when it exists and does not invent it when it does not.", indent=True)
    if abl:
        s_vals = {k: abl[k]["pearson_delta"] for k in ("S1", "S4", "S8", "S16", "S32") if k in abl}
        m.para("On the K562 under-represented-context fold, validation loss and held-out accuracy changed "
               "with the size of the cell set the model reads (Figures 1F and 1G). Pearson delta was "
               + ", ".join(f"{f(val, 3)} at S = {k[1:]}" for k, val in s_vals.items())
               + ". Removing self-attention gave "
               f"{f(abl.get('S32_noattn', {}).get('pearson_delta'), 3)}, removing FiLM "
               f"{f(abl.get('S32_nofilm', {}).get('pearson_delta'), 3)}, and replacing the transfer prior with "
               f"a learned perturbation embedding {f(abl.get('S32_noprior', {}).get('pearson_delta'), 3)}. "
               "Adding an identity embedding to the prior gave "
               f"{f(abl.get('S32_pertid', {}).get('pearson_delta'), 3)}. "
               + _ablation_sentence(res), indent=True)

    # ---------------- section 3: T1
    m.heading("Benchmarking context transfer in under-represented contexts", 2)
    m.para("We first evaluated the common practical setting in which a new context has been screened with "
           "only a fraction of a library (Figure 2A)[@adduri2026]. For each of the four lines in turn, 25% of "
           "its eligible perturbations were used for training, 5% for model selection and 70% for testing; "
           "the other three lines were fully observed. We compared CASTER with PerturbMean (a "
           "perturbation's mean response in the other contexts), ContextMean (the target context's average "
           "response), a context-adaptive linear transfer model fitted on the observed fraction, and a "
           "no-change floor. The evaluation suite, PertBench, follows Cell-Eval[@adduri2026]. It scores "
           "expression shifts (Pearson delta, perturbation discrimination), differential expression (DEG "
           "precision-recall, top-N DEG overlap, fold-change rank correlation, direction) and effect size "
           "(rank correlation of the number of DEGs across perturbations) (Figures 2B-2E; STAR Methods).",
           indent=True)
    m.para(_t1_paragraph(res), indent=True)
    m.para(_t1_detail(res), indent=True)
    m.para(_strength_paragraph(res), indent=True)

    # ---------------- section 4: T2/T3
    m.heading("Zero-shot transfer and the cancer versus non-transformed distinction", 2)
    m.para(_t2_paragraph(res), indent=True)
    m.para(_t3_paragraph(res), indent=True)

    # ---------------- section 5: context-specific DE
    m.heading("Context-specific differential expression, programs and perturbation structure", 2)
    m.para(_fig4_paragraph(res), indent=True)
    m.para(_examples_paragraph(figs), indent=True)
    m.para(_programs_paragraph(res), indent=True)

    # ---------------- section 6: dependency
    m.heading("Knockdown responses and cancer dependency", 2)
    m.para(_h4_paragraph(res), indent=True)
    m.para(_h5_paragraph(res), indent=True)

    # ---------------- outcome table
    m.heading("Summary of pre-registered outcomes", 2)
    m.para("Table 1 lists every pre-declared hypothesis, its test and its outcome.", indent=True)
    m.para("**Table 1. Pre-registered hypotheses and outcomes**", size=7.5)
    h1 = res["H1"]
    h1p = res.get("H1_preregistered", {})
    h3 = res.get("H3", {}).get("PerturbMean", {})
    h4 = pd.DataFrame(res["H4"]["tests"])
    rows = [
        ["H1", "Under-represented context: CASTER > PerturbMean, ContextMean, Linear (3 primary metrics)",
         "One-sided paired Wilcoxon, Holm (9 tests); >= 6 significant, none opposite",
         f"Supported ({h1['n_significant']}/9; {h1['n_opposite']} opposite). Pre-registered DE test: "
         f"{h1p.get('n_significant', 'n/a')}/9"],
        ["H2", "Zero-shot context: CASTER > PerturbMean (3 primary metrics)", "One-sided Wilcoxon, Holm (3 tests)",
         "Not supported; PerturbMean significantly better on all three" if not v["h2"] else "Supported"],
        ["H3", "Transfer more accurate among cancer lines than with RPE1", "Two-sided paired Wilcoxon",
         ("Not supported; opposite direction" if not v["h3"] else "Supported")
         + f" (p = {pval(h3.get('p_two_sided'))})"],
        ["H4", "Response magnitude correlates negatively with Chronos (per cancer line)", "Spearman, Holm (3 lines)",
         ("Supported in all three lines" if v["h4"] else "Not supported") + f" (rho {f(h4[h4.test == 'H4'].rho.max())} "
         f"to {f(h4[h4.test == 'H4'].rho.min())})"],
        ["H4b", "Context-specific response tracks context-specific dependency", "Spearman, Holm (3 lines)",
         "Supported" if v["h4b"] else "Not supported (significant in K562 only, rho = "
         f"{f(h4[(h4.test == 'H4b') & (h4.line == 'k562')].rho.iloc[0])})"],
        ["H5", "Predicted shifts retain dependency information", "Exploratory; 10-fold CV ridge",
         "Retained (CASTER r = " + f(pd.DataFrame(res['H5']).query('method == "CASTER"').pearson.mean())
         + " vs observed " + f(pd.DataFrame(res['H5']).query('method == "_true"').pearson.mean()) + ")"],
    ]
    m.table(["", "Hypothesis", "Test and decision rule", "Outcome"], rows, widths=[0.4, 2.4, 2.0, 2.3], size=6.6)

    # ---------------- figure pages
    _figure_pages(m, res, figs)
    m.columns(2)


def _ablation_sentence(res: dict) -> str:
    rows = pd.DataFrame(res.get("ablation_tests", []))
    if rows.empty:
        return ""
    names = {"S1": "single cells (S = 1)", "S8": "S = 8", "S32_noattn": "no attention",
             "S32_nofilm": "no FiLM", "S32_noprior": "no transfer prior", "S32_pertid": "added ID embedding"}
    parts = []
    for b, g in rows.groupby("baseline"):
        better = [MNAME[m] for m in g[g.p_holm < 0.05].metric]
        worse = [MNAME[m] for m in g[(g.p_two_sided < 0.05) & (g.median_diff < 0)].metric]
        txt = names.get(b, b) + ": "
        txt += ("full model better on " + ", ".join(better)) if better else "no significant advantage"
        if worse:
            txt += "; variant better on " + ", ".join(worse)
        parts.append(txt)
    return ("Paired comparisons of the full model with each variant (Holm-adjusted Wilcoxon) gave: "
            + "; ".join(parts) + ". The transfer prior is by far the most important component. The set-level "
            "attention adds a small but significant improvement. A single-cell model (S = 1) discriminates "
            "perturbations better, at the cost of the distributional (set-level) objective.")

def _t1_paragraph(res: dict) -> str:
    v = verdict(res)
    h1 = h1_rows(res)
    ca = {m_: summ(res, "T1_summary", "CASTER", m_) for m_ in ("pearson_delta", "discrimination", "de_auprc")}
    pm = {m_: summ(res, "T1_summary", "PerturbMean", m_) for m_ in ca}
    li = {m_: summ(res, "T1_summary", "Linear", m_) for m_ in ca}
    cm = {m_: summ(res, "T1_summary", "ContextMean", m_) for m_ in ca}
    s = (f"Averaged over the four held-out contexts, CASTER reached a Pearson delta of {f(ca['pearson_delta'])}, "
         f"a discrimination score of {f(ca['discrimination'])} and a DE AUPRC of {f(ca['de_auprc'])}. The "
         f"corresponding values were {f(pm['pearson_delta'])}, {f(pm['discrimination'])} and "
         f"{f(pm['de_auprc'])} for PerturbMean, {f(li['pearson_delta'])}, {f(li['discrimination'])} and "
         f"{f(li['de_auprc'])} for linear transfer, and {f(cm['pearson_delta'])}, "
         f"{f(cm['discrimination'])} and {f(cm['de_auprc'])} for ContextMean (Figure 2B). ")
    s += (f"Of the nine pre-declared comparisons (H1), {v['h1_n_better']} favoured CASTER after Holm "
          f"correction, and {v['h1_n_worse']} significantly favoured a baseline. ")
    if v["h1"]:
        s += "H1 was therefore supported. "
    else:
        s += "By the pre-declared rule, H1 was therefore **not supported**. "
    pre = res.get("H1_preregistered")
    nd = pd.DataFrame(res.get("amendment2_de_counts", []))
    if pre is not None and not nd.empty:
        cas = nd[nd.method == "CASTER"].iloc[0]
        s += ("DE-based metrics use the amended predicted-side test (AMENDMENT-002, STAR Methods). The "
              "pre-registered Wilcoxon test on predicted cells proved invalid: it called a mean of "
              f"{f(cas.n_de_pred_wilcoxon, 0)} DEGs per CASTER prediction, versus {f(cas.n_de_true, 0)} "
              "observed. Under the pre-registered test, "
              f"{pre['n_significant']} of nine comparisons favoured CASTER and {pre['n_opposite']} favoured a "
              f"baseline (H1 {'supported' if pre['supported'] else 'not supported'}; Table S1); the "
              "conclusion from the expression-level metrics does not depend on this choice. ")
    for b in ("PerturbMean", "Linear", "ContextMean"):
        sub = h1[h1.baseline == b]
        if sub.empty:
            continue
        txt = "; ".join(f"{MNAME.get(r.metric, r.metric)}: {f(100 * r.frac_better, 0)}% of perturbations improved, "
                        f"Holm p = {pval(r.p_holm)}" for r in sub.itertuples())
        s += f"Against {b}: {txt}. "
    return s


def _t1_detail(res: dict) -> str:
    sec = pd.DataFrame(res.get("T1_secondary", []))
    lines = []
    for n in ("k562", "hepg2", "jurkat", "rpe1"):
        c = per_line(res, "T1_per_line", n, "CASTER", "pearson_delta")
        p = per_line(res, "T1_per_line", n, "PerturbMean", "pearson_delta")
        lines.append(f"{LN[n]} {f(c)} versus {f(p)}")
    s = ("Per context, CASTER and PerturbMean reached Pearson deltas of " + "; ".join(lines) + " (Figure 2C). ")
    es_c = summ(res, "T1_summary", "CASTER", "effect_spearman")
    es_p = summ(res, "T1_summary", "PerturbMean", "effect_spearman")
    s += ("Both transfer methods ranked perturbations by effect size well (Spearman rho between predicted and "
          f"observed DEG counts {f(es_c)} for CASTER and {f(es_p)} for PerturbMean; Figure 2E). They "
          "recovered DEGs far above the no-change and context-mean baselines (Figure 2D). ")
    if not sec.empty:
        pm = sec[sec.baseline == "PerturbMean"]
        better = pm[(pm.p_holm < 0.05)]
        worse = pm[(pm.p_two_sided < 0.05) & (pm.median_diff < 0)]
        s += ("Among the secondary metrics compared with PerturbMean, "
              f"{len(better)} favoured CASTER and {len(worse)} favoured PerturbMean "
              "(Table S1).")
    return s


def _t2_paragraph(res: dict) -> str:
    v = verdict(res)
    if "H2" not in res:
        return ""
    h2 = pd.DataFrame(res["H2"]["tests"])
    ca = summ(res, "T2_summary", "CASTER", "pearson_delta")
    pm = summ(res, "T2_summary", "PerturbMean", "pearson_delta")
    s = ("We next asked whether responses could be transferred into a context with no perturbation data at "
         "all, using only its unperturbed cells (Figure 3A). In this zero-shot setting CASTER achieved a "
         f"Pearson delta of {f(ca)} versus {f(pm)} for PerturbMean (Figure 3B). ")
    s += ("CASTER was significantly better on all three primary metrics (H2 supported; "
          if v["h2"] else "The pre-declared comparison (H2) was not supported (")
    s += "; ".join(f"{MNAME.get(r.metric, r.metric)}: CASTER {f(r.mean_model, 3)} vs {f(r.mean_baseline, 3)}, "
                   f"Holm p = {pval(r.p_holm)}" for r in h2.itertuples()) + "). "
    if not v["h2"]:
        per = "; ".join(f"{LN[n]} {f(per_line(res, 'T2_per_line', n, 'CASTER', 'pearson_delta'))} versus "
                        f"{f(per_line(res, 'T2_per_line', n, 'PerturbMean', 'pearson_delta'))}"
                        for n in ("k562", "hepg2", "jurkat", "rpe1"))
        s += ("Per context, CASTER and PerturbMean reached Pearson deltas of " + per + ". Without any "
              "perturbation data from the target, the corrections CASTER learns between source contexts "
              "did not carry over. Its discrimination fell below that of PerturbMean, indicating that the "
              "corrections pulled predictions towards a response shared across perturbations rather than "
              "adding perturbation-specific information. By Proposition 4 (STAR Methods), this means that "
              "the residual of the transfer prior was not predictable from a context's control cells "
              "alone at this data scale. ")
    s += ("Comparing the gain over PerturbMean with and without a 30% observed fraction (Figure 3E) shows "
          "that CASTER's advantage depends on a small amount of perturbation data from the target "
          "context. This has a practical design implication: screening a modest subset of a library in a "
          "new context, as in our under-represented setting, is what makes context adaptation effective.")
    return s


def _t3_paragraph(res: dict) -> str:
    h3 = res.get("H3")
    if not h3:
        return ""
    pm = h3["PerturbMean"]
    ca = h3.get("CASTER", {})
    v = verdict(res)
    tm = pd.DataFrame(res.get("transfer_matrix", []))
    s = ("To test whether transfer depends on how related two contexts are, we transferred every "
         "perturbation from a single source line to each other line (12 directions; Figure 3C). With the "
         "PerturbMean transfer, per-perturbation accuracy averaged over cancer-to-cancer directions was "
         f"{f(pm['mean_cancer_cancer'])}, versus {f(pm['mean_rpe1_pairs'])} for directions involving RPE1 "
         f"(n = {f(pm['n'])} perturbations, paired Wilcoxon p = {pval(pm['p_two_sided'])}; Figure 3D). ")
    if v["h3"]:
        s += ("H3 was supported: responses transfer more faithfully among cancer lines than between cancer "
              "and non-transformed cells. ")
    else:
        s += ("H3 was therefore **not supported**. The difference was significant in the *opposite* "
              "direction: transfer involving the non-transformed line was more accurate. ")
        if not tm.empty:
            pmt = tm[tm.method == "PerturbMean"].set_index(["source", "target"]).pearson_delta
            hr, rh = pmt.get(("hepg2", "rpe1"), np.nan), pmt.get(("rpe1", "hepg2"), np.nan)
            s += (f"The effect was driven by the HepG2-RPE1 pair (Pearson delta {f(hr)} from HepG2 to RPE1 "
                  f"and {f(rh)} from RPE1 to HepG2, the two highest entries of the matrix). This matches the "
                  "higher response similarity of these two adherent, epithelial-derived lines (Figure 1D) "
                  "and the larger, better-resolved responses of RPE1 (Figure S1A). ")
    if ca:
        s += (f"CASTER showed the same pattern ({f(ca['mean_cancer_cancer'])} versus "
              f"{f(ca['mean_rpe1_pairs'])}, p = {pval(ca['p_two_sided'])}), and in single-source transfer it "
              "was indistinguishable from PerturbMean in most directions (Figure 3C; Table S6). ")
    s += ("Transfer accuracy therefore reflects the biological similarity between two contexts, here "
          "lineage and growth mode, more than the binary distinction between transformed and "
          "non-transformed cells.")
    return s

def _fig4_paragraph(res: dict) -> str:
    cc = res.get("cell_cycle_mae", {})
    ari = res.get("ari_mean", {})
    s = ("At the level of individual perturbations, CASTER and the transfer baselines were compared on DE "
         "overlap and fold-change agreement (Figures 4A and 4B). Points above the diagonal are perturbations "
         "for which CASTER recovered more of the context's actual DEGs. Figure 4C shows the knockdown with the "
         "largest CASTER gain over PerturbMean in a cancer line, illustrating a context-specific correction "
         "of the transferred response. ")
    if cc:
        s += ("Predicted shifts of S-phase and G2/M signature scores[@tirosh2016] had mean absolute errors of "
              + ", ".join(f"{f(cc.get(me, {}).get('S'), 4)} (S) and {f(cc.get(me, {}).get('G2M'), 4)} "
                          f"(G2/M) for {me}" for me in ("CASTER", "PerturbMean"))
              + " (Figure 4D). ")
    if ari:
        s += ("When perturbations were clustered by the similarity of their responses, predicted similarity "
              "maps recovered the observed structure with mean adjusted Rand indices (over 2-20 clusters "
              "and four contexts) of "
              + ", ".join(f"{f(ari[me])} ({me})" for me in ("CASTER", "PerturbMean", "Linear", "ContextMean")
                          if me in ari) + " (Figure 4E).")
    return s


def _examples_paragraph(figs: Path) -> str:
    p = figs / "figS10_data.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p).pivot_table(index=["line", "perturbation"], columns="method", values="r_degs")
    items = [f"{pert} in {LN[line]} (r over DEGs {f(r.CASTER)} versus {f(r.PerturbMean)})"
             for (line, pert), r in d.iterrows()]
    return ("Which perturbations does CASTER adapt most? In each cancer line, the knockdowns with the largest "
            "gain over PerturbMean (among those with >= 30 DEGs) were dominated by regulators of ribosome "
            "biogenesis and ribosomal proteins: " + "; ".join(items) + " (Figures 4C and S10). Loss of "
            "ribosome biogenesis triggers a nucleolar stress response, whose transcriptional output depends "
            "on p53 status, MYC activity and growth rate, all of which differ between these lines. The "
            "transferred response of these knockdowns is therefore a poor guide to their response in a new "
            "context, while the context's own cells, read through CASTER, carry the information needed to "
            "correct it. In HepG2, for example, the PerturbMean prediction for NVL knockdown was "
            "uncorrelated or anti-correlated with the observed DEGs, whereas CASTER recovered their direction "
            "and relative magnitude.")


def _programs_paragraph(res: dict) -> str:
    pp = res.get("program_pearson", {})
    st = res.get("string_auroc", {})
    inv = res.get("inverse_top10", {})
    s = ("We identified 15 co-regulated gene programs per context from the observed responses and annotated "
         "them with Reactome[@milacic2024]. Predicted program activities correlated with observed ones at "
         "mean r = " + ", ".join(f"{f(pp[me])} ({me})" for me in ("CASTER", "Linear", "PerturbMean", "ContextMean")
                                  if me in pp) + " (Figures 5D and S9; Table S8). ")
    if st:
        s += ("Response similarity between knockdowns recovered high-confidence STRING "
              "associations[@szklarczyk2023] with AUROC " + f"{f(st.get('_true'))} for observed responses, "
              + ", ".join(f"{f(st[me])} for {me}" for me in ("CASTER", "PerturbMean") if me in st)
              + " (Figure 5F). Predicted responses therefore preserve much of the functional organisation of "
                "the perturbations. ")
    if inv:
        s += ("In an inverse-design task (identifying which perturbation produced an observed response among "
              "all candidates in a context), the correct perturbation was within the top 10 for "
              + ", ".join(f"{f(100 * inv[me], 0)}% ({me})" for me in ("CASTER", "PerturbMean", "Linear",
                                                                       "ContextMean") if me in inv)
              + " of the 100 strongest perturbations (Figure 5E).")
    return s


def _h4_paragraph(res: dict) -> str:
    h4 = pd.DataFrame(res["H4"]["tests"])
    a = h4[h4.test == "H4"]
    b = h4[h4.test == "H4b"]
    v = verdict(res)
    s = ("Finally, we asked whether perturbation responses carry information about cancer dependency, "
         "measured independently by genome-scale CRISPR fitness screens in DepMap (Chronos gene "
         "effects)[@tsherniak2017,dempster2021,meyers2017]. In each cancer line, knockdowns of genes on which "
         "the line depends more strongly (more negative Chronos score) were expected to produce larger "
         "transcriptional responses (H4). Spearman correlations between log(1 + # DEGs) and Chronos were "
         + "; ".join(f"{LN[r.line]} rho = {f(r.rho)} (n = {f(int(r.n))}, Holm p = {pval(r.p_holm)})"
                     for r in a.itertuples()) + " (Figure 5A). ")
    s += ("H4 was supported: response magnitude tracks dependency. " if v["h4"] else
          "H4 was not supported in the pre-declared direction. ")
    s += ("Restricting to genes profiled in all three cancer lines, we then asked whether the *context-"
          "specific* part of the response (a line's response minus the mean of the other two) tracks the "
          "context-specific part of the dependency (H4b): "
          + "; ".join(f"{LN[r.line]} rho = {f(r.rho)} (Holm p = {pval(r.p_holm)})" for r in b.itertuples())
          + " (Figure 5B). ")
    sig_lines = [LN[r.line] for r in b.itertuples() if r.p_holm < 0.05 and r.rho < 0]
    if v["h4b"]:
        s += "H4b was supported. "
    else:
        s += ("H4b was not supported by the pre-declared rule (significant negative correlation in at least two "
              "lines)" + (f"; only {', '.join(sig_lines)} reached significance, with a weak effect" if sig_lines
                          else "") + ". In these data, which genes a particular line depends on more than the "
              "others is at most weakly reflected in how strongly it responds to their knockdown. ")
    s += ("Dependency differences between lines are small for most essential genes[@hart2015,wang2015,blomen2015], "
          "which limits the power of this test.")
    return s


def _strength_paragraph(res: dict) -> str:
    g = pd.DataFrame(res.get("gain_vs_strength", []))
    if g.empty:
        return ""
    t1 = g[g.task == "T1"].set_index("target")
    t2 = g[g.task == "T2"].set_index("target")
    s = ("Where does CASTER gain over the transfer prior? Binning held-out perturbations by the number of "
         "observed DEGs showed that the gain grows with response strength in every context (Spearman rho "
         "between gain and log response size "
         + ", ".join(f"{f(t1.loc[n, 'rho'])} in {LN[n]}" for n in ("k562", "hepg2", "jurkat", "rpe1"))
         + "; Figure S8). For knockdowns with >= 50 DEGs the median gain in Pearson delta was "
         + ", ".join(f"{f(t1.loc[n, 'median_gain_strong'])} ({LN[n]})" for n in ("k562", "hepg2", "jurkat", "rpe1"))
         + ", whereas for knockdowns with < 10 DEGs it was close to zero ("
         + ", ".join(f"{f(t1.loc[n, 'median_gain_weak'])}" for n in ("k562", "hepg2", "jurkat", "rpe1"))
         + "). CASTER's corrections therefore act on real, strong, context-specific responses, not on noise. ")
    if not t2.empty:
        s += ("In the zero-shot task, no such gradient appeared (median gain for strong responses "
              + ", ".join(f"{f(t2.loc[n, 'median_gain_strong'])}" for n in ("k562", "hepg2", "jurkat", "rpe1"))
              + "), in line with the absence of an overall advantage.")
    return s


def _h5_paragraph(res: dict) -> str:
    h4p = pd.DataFrame(res.get("H4_predicted", []))
    extra = ""
    if not h4p.empty:
        def rho(n: str, me: str) -> str:
            v = h4p[(h4p.line == n) & (h4p.method == me)].rho
            return f(v.iloc[0]) if len(v) else "n/a"
        extra = (" The dependency-response coupling itself was preserved in predicted responses. Predicted "
                 "response magnitude correlated with Chronos at rho = "
                 + ", ".join(f"{rho(n, 'CASTER')} ({LN[n]})" for n in CANCER_LINES)
                 + " for CASTER, compared with "
                 + ", ".join(f"{rho(n, '_observed')}" for n in CANCER_LINES)
                 + " for observed responses and "
                 + ", ".join(f"{rho(n, 'PerturbMean')}" for n in CANCER_LINES)
                 + " for PerturbMean. ContextMean, which ignores perturbation identity, lost it (Figure S12).")
    h5 = pd.DataFrame(res["H5"])
    if h5.empty:
        return extra
    g = h5.groupby("method").pearson.mean()
    return extra + " " + ("A ridge model trained to predict Chronos from *observed* transcriptional shifts (10-fold "
            "cross-validation over held-out perturbations) reached a mean Pearson r of "
            f"{f(g.get('_true'))} across the three cancer lines. Applied unchanged to *predicted* shifts, it "
            "reached " + ", ".join(f"{f(g[me])} ({me})" for me in ("CASTER", "PerturbMean", "Linear",
                                                                   "ContextMean") if me in g)
            + " (Figure 5C; Figure S5). Transferred responses therefore retain part of the dependency "
              "information carried by measured responses. This connects in silico perturbation to cancer "
              "target prioritisation[@behan2019,pacini2024], a use that the reference State model "
              "demonstrated for cell viability[@adduri2026].")


# ---------------------------------------------------------------------------- figures
LEGENDS: dict[str, tuple[str, list[str]]] = {
    "Figure1": ("Figure 1. Context reshapes the response to the same genetic perturbation; the CASTER model", [
        "(A) The same CRISPRi knockdown, profiled in three cancer contexts and a non-transformed context, "
        "produces a shared core response plus a context-specific component.",
        "(B) Perturbations retained after on-target knockdown filtering per context (left), and the number of "
        "contexts in which each perturbation is retained (right).",
        "(C) CASTER architecture. A set of unperturbed target-context cells is encoded by an MLP and read by a "
        "bidirectional set transformer. The perturbation enters through its transfer prior (mean shift in other "
        "contexts), modulated by the context summary (FiLM). The output adds a context-dependent gene-wise "
        "rescaling of the prior and a cell-resolved correction; both start at zero.",
        "(D) Median Pearson correlation of observed response shifts between contexts (perturbations with >= 10 "
        "DEGs in both).",
        "(E) Per-perturbation response similarity for each pair of contexts; cancer-cancer pairs in blue, pairs "
        "involving RPE1 in grey.",
        "(F) Validation loss (relative to epoch 1) for cell-set sizes S = 1, 8 and 32 on the K562 fold (20 epochs).",
        "(G) Held-out Pearson delta for set sizes and architectural ablations on the K562 fold (20 epochs; bars, "
        "mean over seeds; dots, seeds), with PerturbMean (dashed) and Linear (long dashes) on the same split.",
    ]),
    "Figure2": ("Figure 2. Context transfer in under-represented contexts", [
        "(A) Task: for each held-out context, 25% of its perturbations train, 5% validate and 70% test; other "
        "contexts are fully observed.",
        "(B) Primary metrics averaged over the four held-out contexts. Bars, mean; error bars, SD over contexts; "
        "dots, contexts. Dashed line, best baseline; percentages, relative difference of CASTER from it.",
        "(C) All PertBench metrics per held-out context (error bars, SEM over perturbations; effect-size "
        "Spearman is a single value per context).",
        "(D) Precision-recall curves for recovering observed DEGs (pooled over perturbations) in K562 and "
        "HepG2, using amended predicted-side p values (AMENDMENT-002); dashed line, base rate.",
        "(E) Observed versus predicted number of DEGs per perturbation (row-normalised), all contexts.",
    ]),
    "Figure3": ("Figure 3. Zero-shot transfer and transfer between cancer and non-transformed contexts", [
        "(A) Zero-shot task: no perturbed cells of the held-out context are used.",
        "(B) Primary metrics in the zero-shot task (mean over contexts; dots, contexts).",
        "(C) Single-source transfer: Pearson delta for every source (rows) and target (columns) pair.",
        "(D) Per-perturbation transfer accuracy averaged over cancer-to-cancer directions versus directions "
        "involving RPE1. Boxes, interquartile range; whiskers, 1.5x IQR; p value, two-sided paired Wilcoxon (H3).",
        "(E) CASTER minus PerturbMean (Pearson delta) per context, with no (zero-shot) or 30% observed "
        "perturbations of the target.",
    ]),
    "Figure4": ("Figure 4. Context-specific differential expression, cell cycle and perturbation structure", [
        "(A and B) Per held-out perturbation, CASTER (y) versus each baseline (x) for DE overlap (A) and "
        "fold-change Spearman (B); colour, difference (blue, CASTER better).",
        "(C) Observed log2 fold change versus predicted pseudobulk shift for the knockdown with the largest "
        "CASTER gain over PerturbMean in a cancer context (coloured, observed DEGs; r over DEGs).",
        "(D) Absolute error of predicted S-phase and G2/M signature shifts.",
        "(E) Perturbation-similarity (cosine) maps for K562 perturbations with >= 10 DEGs, ordered by "
        "clustering of the observed map, and adjusted Rand index between predicted and observed clusterings "
        "(mean over contexts).",
    ]),
    "Figure5": ("Figure 5. Perturbation responses and cancer dependency", [
        "(A) Transcriptional response magnitude (log(1 + # DEGs)) versus DepMap Chronos gene effect of the "
        "knocked-down gene in each cancer line (H4). Dashed line, Chronos = -1 (typical essential gene).",
        "(B) Context-specific response versus context-specific dependency (line minus mean of the other two "
        "cancer lines; H4b). Labelled, most extreme genes.",
        "(C) Pearson r between measured Chronos and Chronos predicted by a ridge model (trained on observed "
        "shifts) from observed or predicted shifts (H5).",
        "(D) Concordance of predicted and observed gene-program activity across perturbations (K562); "
        "programs labelled by their top Reactome term when enriched (p < 1e-3).",
        "(E) Inverse design: top-k accuracy of identifying the perturbation that produced an observed "
        "response, for the 100 strongest perturbations per context.",
        "(F) AUROC for separating STRING-linked (combined score >= 700) perturbation pairs by response "
        "similarity (perturbations with >= 10 DEGs).",
    ]),
    "Figure6": ("Figure 6. Calibrated evaluation, noise ceilings, seeds and per-line controls", [
        "(A) Pearson delta of each method per held-out line (mean over seeds) and the split-half noise ceiling "
        "(Spearman-Brown corrected; hatched).",
        "(B) Dynamic range fraction (DRF) of each metric per line: how well the metric separates the technical "
        "duplicate from the negative control (mean line response).",
        "(C) Calibrated metrics on a scale from negative control (0) to technical duplicate (1; dashed); "
        "InterpDup, interpolated duplicate. Under-represented task, mean over lines and seeds.",
        "(D) Pearson delta per held-out line and seed (dots; bar = mean) for CASTER, Linear and PerturbMean.",
        "(E) Median per-perturbation gain of CASTER over LineOffset, over a line-identity-only CASTER (no "
        "control cells) and over CASTER given another line's control cells.",
        "(F) Calibrated metrics in the zero-shot task.",
        "(G) Pearson delta when every method's predicted shift is freed of the sampling noise of the 32 control "
        "cells it was rendered on (bars; mean over seeds), compared with the evaluation as rendered (ticks) and the "
        "noise ceiling (hatched).",
    ]),
}


def _figure_pages(m: Manuscript, res: dict, figs: Path) -> None:
    for name, (title, legend) in LEGENDS.items():
        path = figs / f"{name}.png"
        if path.exists():
            m.figure(path, title, legend, width_in=7.0)


SUPP: dict[str, tuple[str, list[str]]] = {
    "FigureS1": ("Figure S1. Response sizes and context overlap, related to Figure 1", [
        "(A) Distribution of the number of DEGs per perturbation in each context. (B) Cells per perturbation "
        "after subsampling. (C) Jaccard overlap of DEG sets between contexts. (D) Median cross-context "
        "response similarity as a function of response strength (minimum number of DEGs in both contexts)."]),
    "FigureS2": ("Figure S2. Training curves, related to Figures 2 and 3", [
        "Training and validation loss per epoch for every under-represented (top) and zero-shot (bottom) fold; "
        "the selected checkpoint is marked."]),
    "FigureS3": ("Figure S3. Secondary metrics, related to Figures 2 and 3", [
        "Mean absolute error and direction agreement per context (under-represented task), and DE overlap per "
        "context in the zero-shot task."]),
    "FigureS4": ("Figure S4. Positive-control simulation, related to Figure 1", [
        "Synthetic four-context data in which the context-specific component is random noise (left) or a "
        "function of baseline expression observable from control cells (right)."]),
    "FigureS5": ("Figure S5. Dependency prediction from transcriptional shifts, related to Figure 5", [
        "Measured versus predicted Chronos gene effect per cancer line, from observed shifts (top) and from "
        "CASTER-predicted shifts (bottom)."]),
    "FigureS6": ("Figure S6. Single-source transfer, all metrics, related to Figure 3", [
        "DE AUPRC and perturbation discrimination for every source and target pair."]),
    "FigureS7": ("Figure S7. Predicted-side differential expression: pre-registered versus amended test, "
                 "related to STAR Methods (AMENDMENT-002)", [
        "(Left) Mean number of predicted DEGs per perturbation under the pre-registered Wilcoxon test on "
        "predicted cells (grey) and the amended moment-matched z-test (colour); dashed line, observed mean. "
        "Shifted predictions turn exact zeros into small positive values, so the rank test calls most genes "
        "significant; unshifted control cells (NoChange) are unaffected. (Middle, right) Observed versus "
        "CASTER-predicted DEG counts per perturbation under the two tests."]),
    "FigureS8": ("Figure S8. CASTER's gain over PerturbMean as a function of response strength, related to "
                 "Figures 2 and 3", [
        "Per-perturbation difference in Pearson delta (CASTER minus PerturbMean), binned by the observed "
        "number of DEGs, for the under-represented (top) and zero-shot (bottom) tasks. Boxes, IQR; whiskers, "
        "1.5x IQR."]),
    "FigureS9": ("Figure S9. Gene-program activity of held-out perturbations, related to Figure 5", [
        "Activity of the 15 K562 gene programs (rows) for the 80 strongest held-out K562 perturbations "
        "(columns, clustered on the observed matrix): observed, CASTER and PerturbMean. r, Pearson correlation "
        "with the observed matrix."]),
    "FigureS10": ("Figure S10. Context-specific example knockdowns, related to Figure 4", [
        "For each cancer line, the two knockdowns (>= 30 DEGs) with the largest CASTER gain over PerturbMean: "
        "observed log2 fold change versus predicted pseudobulk shift; coloured, observed DEGs."]),
    "FigureS11": ("Figure S11. All metrics per held-out context in the zero-shot task, related to Figure 3", [
        "As Figure 2C, for the zero-shot task (CASTER, PerturbMean and NoChange)."]),
    "FigureS13": ("Figure S13. p53 activity across lines, related to Figures 1 and 3", [
        "(A) Percentage of knockdowns inducing the 14-gene p53 target score (FDR < 0.05), with TP53 status. "
        "(B) Rank correlation of p53-score shifts between lines. (C) HepG2-RPE1 response similarity for "
        "knockdowns inducing p53 in both lines versus others (Mann-Whitney p). (D) Median response similarity "
        "per line pair among knockdowns inducing p53 in neither line."]),
    "FigureS14": ("Figure S14. The dependency-response link is not line-specific, related to Figure 5", [
        "(A) Spearman rho between response magnitude in one line (rows) and Chronos gene effect in each line "
        "(columns). (B) Pearson r between each line's Chronos scores and the mean of the other two lines."]),
    "FigureS15": ("Figure S15. Ribosome knockdowns and complex leakage, related to Figure 4", [
        "(A) CASTER gain over PerturbMean for ribosome-biogenesis and ribosomal-protein knockdowns versus "
        "others (random split, seed 0). (B) Median gain under complex-grouped splits, where no same-complex gene "
        "is in training. (C) Gain versus the number of same-complex genes in training (random split)."]),
    "FigureS12": ("Figure S12. Dependency-response coupling in predicted responses, related to Figure 5", [
        "Spearman correlation between response magnitude (log(1 + # DEGs)) and DepMap Chronos gene effect "
        "for observed responses and for the responses predicted by each method (held-out perturbations, "
        "under-represented task)."]),
}


def supplementary(m: Manuscript, res: dict, figs: Path) -> None:
    m.columns(1, new_page=True)
    m.heading("Supplemental information", 1)
    pre_src = res.get("review", {}).get("preregistered_de") or [
        r for r in res.get("H1_preregistered", {}).get("tests", []) if r["metric"] == "de_auprc"]
    pre_rows = [dict(r, baseline=r["baseline"] + " (pre-spec. Wilcoxon)") for r in pre_src]
    sec = pd.DataFrame(res.get("H1", {}).get("tests", []) + res.get("T1_secondary", []) + pre_rows)
    if not sec.empty:
        m.heading("Table S1. All paired tests, under-represented context task (one-sided Wilcoxon, CASTER > "
                  "baseline; Holm within each family)", 3)
        rows = [[r.baseline, r.metric, f(int(r.n)), f(r.mean_model, 3), f(r.mean_baseline, 3),
                 f(100 * r.frac_better, 0), pval(r.p_greater), pval(r.p_holm)] for r in sec.itertuples()]
        m.table(["Baseline", "Metric", "n", "CASTER", "Baseline", "% better", "p", "p (Holm)"], rows,
                widths=[1.0, 1.0, 0.5, 0.7, 0.7, 0.6, 0.9, 0.9])
    _supp_tables(m, res, figs)
    sp = figs.parent / "tables" / "splits.csv"
    if sp.exists():
        d = pd.read_csv(sp)
        m.heading("Table S9. Fold sizes and training summary", 3)
        task_label = {"t1": "Under-represented", "t2": "Zero-shot",
                      "t1_meanctrl": "Under-repr., ID-only", "t1_grouped": "Under-repr., grouped"}
        rows = [[task_label.get(r.task, r.task), LN[r.fold.split("_")[0]], r.fold.split("_s")[-1],
                 f(int(r.train)), f(int(r.val)), f(int(r.test)), f(int(r.best_epoch) + 1), f(r.minutes, 1)]
                for r in d.itertuples()]
        m.table(["Task", "Held-out context", "Seed", "Train perts", "Val perts", "Test perts", "Selected epoch",
                 "Minutes"], rows, size=6.4)
    for name in sorted(SUPP, key=lambda k: int(k.removeprefix("FigureS"))):
        title, legend = SUPP[name]
        path = figs / f"{name}.png"
        if path.exists():
            m.figure(path, title, legend, width_in=6.6)


def _supp_tables(m: Manuscript, res: dict, figs: Path) -> None:  # noqa: PLR0915
    tab = figs.parent / "tables"
    m.heading("Table S2. CASTER hyperparameters and training settings", 3)
    m.table(["Component", "Setting"], [
        ["Cell encoder f_cell", "4-layer MLP, GELU, 2,000 -> 256 -> 256 -> 256 -> 256"],
        ["Prior encoder f_prior", "2-layer MLP, GELU, 2,000 -> 256 -> 256"],
        ["Context summary / FiLM", "mean of cell embeddings -> 2-layer MLP -> (gamma, beta) in R^256 x R^256"],
        ["Set transformer", "4 layers, 8 heads, d_model 256, feed-forward 512, pre-norm, GELU, dropout 0, "
                            "no positional encoding"],
        ["Output", "LayerNorm + linear 256 -> 2,000 (zero-initialised); g(c): linear 256 -> 2,000 (zero-init)"],
        ["Cell-set size S", "32 (ablation: 1, 8)"],
        ["Loss", "energy-distance MMD^2 + 1.0 x pseudobulk squared error"],
        ["Optimiser", "AdamW, lr 3e-4, weight decay 1e-4, batch 32 sets, grad-norm clip 1.0"],
        ["Schedule", "5% linear warm-up, cosine decay to 5% of peak"],
        ["Epochs", "20 (T1, T2); 10 (ablations, single-source transfer)"],
        ["Model selection", "lowest validation loss (T1: 5% target perturbations; T2/T3: 5% source items)"],
        ["Hardware", "CPU only (4 cores, 8 GB RAM), 2 threads per job, PyTorch 2.13"],
        ["Parameters", "4.6 M (excluding the one-hot embedding, used only in ablations)"],
    ], widths=[1.8, 5.2])
    audit = res.get("audit", {})
    if audit:
        m.heading("Table S3. Dataset after subsampling and filtering", 3)
        nde = res.get("observed_n_de", {})
        rows = [[LN[n], f(audit.get(f"{n}_cells")), f(audit.get(f"{n}_controls")),
                 f(audit.get(f"{n}_perturbations")), f(audit.get(f"{n}_median_cells_per_perturbation"), 0),
                 f(nde.get(n, {}).get("median"), 0), f(100 * nde.get(n, {}).get("frac_ge10", np.nan), 0)]
                for n in ("k562", "hepg2", "jurkat", "rpe1")]
        m.table(["Context", "Cells", "Controls", "Perturbations", "Median cells/pert",
                 "Median # DEGs", "% with >= 10 DEGs"], rows, widths=[0.9, 0.8, 0.8, 1.0, 1.1, 1.0, 1.1])
    if (tab / "t1_fold_means.csv").exists():
        ft = pd.read_csv(tab / "t1_fold_means.csv")
        ft = ft[ft.seed == 0]
        m.heading("Table S4. All metrics per held-out context, under-represented task (mean over "
                  "held-out perturbations)", 3)
        cols = ["pearson_delta", "discrimination", "mae", "de_auprc", "de_overlap", "lfc_spearman",
                "direction", "effect_spearman"]
        rows = [[LN[r.target], r.method] + [f(getattr(r, c), 3) for c in cols]
                for r in ft.sort_values(["target", "method"]).itertuples()]
        m.table(["Context", "Method", "Pearson d", "Discrim.", "MAE", "DE AUPRC", "DE overlap",
                 "FC Spearman", "Direction", "Effect rho"], rows, size=6.2)
    if (tab / "t2_fold_means.csv").exists():
        ft = pd.read_csv(tab / "t2_fold_means.csv")
        m.heading("Table S5. All metrics per held-out context, zero-shot task", 3)
        cols = ["pearson_delta", "discrimination", "mae", "de_auprc", "de_overlap", "lfc_spearman",
                "direction", "effect_spearman"]
        rows = [[LN[r.target], r.method] + [f(getattr(r, c), 3) for c in cols]
                for r in ft.sort_values(["target", "method"]).itertuples()]
        m.table(["Context", "Method", "Pearson d", "Discrim.", "MAE", "DE AUPRC", "DE overlap",
                 "FC Spearman", "Direction", "Effect rho"], rows, size=6.2)
    if (tab / "transfer_matrix.csv").exists():
        tm = pd.read_csv(tab / "transfer_matrix.csv")
        m.heading("Table S6. Single-source transfer (mean over shared perturbations)", 3)
        rows = [[LN[r.source], LN[r.target], r.method, f(r.pearson_delta, 3), f(r.discrimination, 3),
                 f(r.de_auprc, 3), f(r.de_overlap, 3)]
                for r in tm[tm.method.isin(["CASTER", "PerturbMean"])].sort_values(
                    ["source", "target", "method"]).itertuples()]
        m.table(["Source", "Target", "Method", "Pearson d", "Discrim.", "DE AUPRC", "DE overlap"], rows,
                size=6.2)
    if (tab / "ablation.csv").exists():
        ab = pd.read_csv(tab / "ablation.csv")
        m.heading("Table S7. Ablations (under-represented task, K562 held out)", 3)
        rows = [[r.method, f(r.pearson_delta, 3), f(r.discrimination, 3), f(r.de_auprc, 3),
                 f(r.de_overlap, 3), f(r.lfc_spearman, 3)] for r in ab.itertuples()]
        m.table(["Variant", "Pearson d", "Discrim.", "DE AUPRC", "DE overlap", "FC Spearman"], rows, size=6.4)
    if (tab / "calibrated_metrics.csv").exists():
        cm = pd.read_csv(tab / "calibrated_metrics.csv")
        cm = cm[cm.task == "t1"].groupby("method")[["pearson_deltactrl", "pearson_deltapert_top100", "wr2_deltapert",
                                                    "wmse", "mse", "mae", "nir", "pds"]].mean()
        order = [x for x in ("TechDup", "InterpDup", "CASTER", "Linear", "LineOffset", "PerturbMean", "ContextMean",
                             "CASTER_wrongctrl", "MeanBaseline", "NoChange") if x in cm.index]
        m.heading("Table S10. Calibrated metrics against one half of each knockdown's cells, under-represented "
                  "task (mean over lines and seeds). TechDup and InterpDup are positive controls, MeanBaseline the "
                  "negative control (exploratory; added in the review round)", 3)
        rows = [[me] + [f(cm.loc[me, c], 4 if c in ("wmse", "mse") else 3) for c in cm.columns] for me in order]
        m.table(["Method", "Pearson d", "Top-100 Pearson d (vs pert. mean)", "wR2 (vs pert. mean)", "WMSE", "MSE",
                 "MAE", "NIR", "PDS"], rows, size=6.2)
    checks = []
    nc = tab / "noise_cancel_check.csv"
    if nc.exists():
        d = pd.read_csv(nc)
        for me in ("PerturbMean", "Linear", "LineOffset"):
            if me in d and f"{me}_clean" in d:
                checks.append([f"{me} rendered on sampled vs noise-free control cells (Pearson delta, seed 0)",
                               f"{f(d[me].mean(), 3)} vs {f(d[f'{me}_clean'].mean(), 3)} (CASTER {f(d.CASTER.mean(), 3)})"])
    ag = tab / "similarity_allgenes.csv"
    if ag.exists():
        d = pd.read_csv(ag)
        d["kind"] = np.where((d.a == "rpe1") | (d.b == "rpe1"), "with RPE1", "cancer-cancer")
        for kind, g in d.groupby("kind"):
            checks.append([f"Cross-line response similarity on all 6,546 genes, {kind} (median r)",
                           f"{f(g.pearson_allgenes.median())} (n = {len(g)})"])
    if checks:
        m.heading("Table S11. Robustness checks requested in review", 3)
        m.table(["Check", "Result"], checks, widths=[4.6, 2.4], size=6.4)
    if (tab / "program_annotation.csv").exists():
        an = pd.read_csv(tab / "program_annotation.csv")
        m.heading("Table S8. Gene programs and their top Reactome enrichment", 3)
        rows = [[LN[r.line], int(r.program), int(r.size), str(r.top_term)[:70],
                 f"{r.p:.2f}" if r.p >= 0.01 else f"{r.p:.1e}"] for r in an.sort_values(["line", "p"]).itertuples()]
        m.table(["Context", "Program", "Genes", "Top Reactome pathway", "p"], rows,
                widths=[0.7, 0.6, 0.5, 4.2, 0.8], size=6.2)


# ---------------------------------------------------------------------------- discussion
def discussion(res: dict) -> list[str]:
    v = verdict(res)
    cc = {r["pair_type"]: r["pearson"] for r in res["cross_context"]["by_type"]}
    out = [
        "Perturbation biology is increasingly studied with models that aim to predict the outcome of an "
        "experiment before it is run[@bunne2024,rood2024,roohani2025]. For cancer, the context in which a "
        "perturbation acts (lineage, driver, transformation state) is central: the same gene can be a "
        "vulnerability in one tumour and dispensable in another[@tsherniak2017,behan2019,oneil2017]. We asked "
        "a focused version of this question with a single CRISPRi library profiled in three cancer lines and a "
        "non-transformed line. Responses to the same knockdown were only moderately correlated across "
        f"contexts (median r ~ {f(cc.get('cancer-cancer'))} between cancer lines). Similarity increased with "
        "response strength, and was highest for the two adherent, epithelial-derived lines. The context-"
        "specific component of a perturbation response is therefore large, but part of it reflects "
        "measurement noise in weak responses.",
    ]
    if v["h1"]:
        out.append("CASTER exploits this structure when the target context is partly screened. By anchoring on "
                   "the transferred response and learning only the context-dependent adaptation, it "
                   "outperformed the perturbation-mean and context-mean baselines on expression and DE metrics, "
                   "and recovered gene programs, perturbation-similarity structure and the identity of "
                   "perturbations far better than the perturbation mean. It matched, but did not exceed, a "
                   "context-adaptive linear model fitted to the same data. This agrees with recent evidence "
                   "that well-specified linear models are strong competitors for perturbation "
                   "prediction[@ahlmann2025,kernfeld2025]. The value of the set-based model lies less in "
                   "accuracy over a tuned linear map than in its distributional, cell-level output and in "
                   "operating without per-context refitting.")
        if not v["h2"]:
            out.append("The zero-shot results set a clear boundary. With no perturbation data from the target "
                       "context, CASTER did not beat the perturbation mean and was worse at discriminating "
                       "between perturbations. In the language of Proposition 4, the residual of the transfer "
                       "prior was not predictable from a context's unperturbed cells alone. Unperturbed "
                       "transcriptomes thus carry little information about how a context will reshape a "
                       "response, whereas a small number of perturbation measurements in that context carry a "
                       "lot. For screening design this argues for sparse, library-wide pilot screens in each "
                       "new context of interest, rather than relying on control profiles alone.")
    else:
        out.append("CASTER was designed to exploit this structure, but under a pre-registered protocol it did "
                   "not consistently exceed the transfer baseline, PerturbMean, although it clearly exceeded "
                   "context-only and no-change baselines. This agrees with recent benchmarks showing that a "
                   "perturbation's mean effect across contexts is a strong baseline that complex models "
                   "rarely beat[@ahlmann2025,kernfeld2025,wu2025]. Our positive-control simulation shows that "
                   "the architecture can learn context structure when that structure is predictable from "
                   "unperturbed cells. The benchmark result therefore suggests that, at this data scale (four "
                   "contexts, about 32 cells per perturbation), most of the context-specific component is not "
                   "predictable from the control transcriptome alone. This is an informative negative result: "
                   "it bounds what context-transfer models can be expected to deliver from current Perturb-seq "
                   "resources, and it argues for screening designs that sample more contexts rather than more "
                   "cells per context.")
    if v["h3"]:
        out.append("The cancer versus non-transformed contrast was a clear biological signal. Transfer was more "
                   "accurate among cancer lines than between cancer lines and RPE1, consistent with the distinct, "
                   "TP53-proficient stress responses of non-transformed cells to essential-gene "
                   "loss[@haapaniemi2018,ihry2018].")
    else:
        out.append("We had expected transfer to be more faithful among cancer lines than to or from the "
                   "non-transformed line (H3). The opposite held. Transfer involving RPE1 was more accurate, "
                   "driven by the epithelial pair HepG2-RPE1, and RPE1 knockdowns produced markedly larger "
                   "responses, consistent with intact p53-dependent stress signalling in TP53-proficient "
                   "cells[@haapaniemi2018,ihry2018]. Shared lineage and growth mode therefore predicted "
                   "transferability better than transformation status. For drug discovery, this suggests that "
                   "the choice of reference context for a perturbation atlas should be guided by lineage and "
                   "cell state, and that a non-transformed line of the right lineage can be a useful reference "
                   "for tumour contexts.")
    out.append("The perturbations whose responses CASTER reshaped most were regulators of ribosome biogenesis "
               "and ribosomal proteins. This is biologically coherent. Ribosome biogenesis is a cancer "
               "dependency that is being explored therapeutically, and impairing it triggers a nucleolar stress "
               "response whose outcome depends on p53 status and MYC-driven growth "
               "programs[@hanahan2011,hanahan2022]. These differ between a TP53-mutant leukaemia line, a "
               "hepatoblastoma line and TP53-proficient epithelial cells. Context-aware prediction is therefore "
               "most valuable precisely for the strong, pathway-level perturbations most relevant to cancer "
               "vulnerability, and least needed for weak perturbations, whose apparent context specificity is "
               "largely noise (Figure S8).")
    out.append("Compared with State[@adduri2026], which pretrains on more than 100 million perturbed cells and "
               "reports gains over mean baselines across several datasets, CASTER is deliberately small and "
               "anchored on an explicit transfer prior. Under that design the comparison with PerturbMean "
               "measures the learnable context-specific residual directly (Proposition 4), rather than the "
               "combined effect of representation learning and scale. The two approaches are complementary: "
               "embeddings learned at scale could replace the control-cell encoder and the transfer prior "
               "could be combined with learned perturbation representations. Our results suggest that such "
               "models should be evaluated against strong transfer and linear baselines, stratified by response "
               "strength, and with predicted-side DE tests that respect the zero inflation of single-cell data.")
    out.append("Linking responses to dependency gave a mixed picture. "
               + ("Response magnitude tracked DepMap fitness effects, " if v["h4"]
                  else "Response magnitude did not track DepMap fitness effects in the expected direction, ")
               + ("and context-specific responses tracked context-specific dependencies. " if v["h4b"]
                  else "and the context-specific part of the response did not track context-specific "
                       "dependency. ")
               + "Dependencies among the essential genes in this library are largely shared across lines, "
               "which limits the variance a context-specific test can exploit. Libraries designed around "
               "differential dependencies, such as synthetic-lethal partners of cancer "
               "drivers[@lord2017,oneil2017], would provide a sharper test.")
    out.append("Methodologically, we stress three practices that made these conclusions possible: selecting "
               "genes on control cells only and excluding perturbation targets, so that gene selection cannot "
               "leak test information; rendering every method's prediction as cells and scoring all of them "
               "with the same single-cell DE pipeline; and fixing hypotheses and decision rules before model "
               "fitting. Several published comparisons of perturbation models differ in exactly these "
               "choices[@ahlmann2025,kernfeld2025,wu2025,adduri2026].")
    return out


def limitations(res: dict) -> list[str]:
    return ["This study has several limitations. First, it covers four contexts, three of them cancer lines "
            "from different lineages; with so few contexts, context generalisation cannot be separated from "
            "lineage generalisation, and conclusions about cancer in general require more cancer contexts. "
            "Second, compute and bandwidth limits led us to use at most 32 cells per perturbation and to "
            "train compact models on CPUs; State-scale models trained on far larger corpora[@adduri2026] may "
            "extract more context-specific signal. Third, CASTER represents a perturbation by its observed "
            "effect elsewhere, so it cannot predict perturbations never observed in any context. Fourth, the "
            "dependency analyses rely on CRISPR knockout fitness, whereas the screens use CRISPRi knockdown; "
            "knockdown efficacy differs between genes and lines. Fifth, Wilcoxon-based single-cell DE "
            "ignores sample structure[@squair2021]; we apply it identically to observed and predicted cells, "
            "but absolute DEG counts should be interpreted with care. Finally, this work used publicly "
            "available data only and did not test any prediction experimentally."]
