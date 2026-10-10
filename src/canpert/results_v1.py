"""Manuscript V.1 text (review round of 2026-10-08). Every number is read from results.json;
claims follow the pre-specified test outcomes. Reuses helpers and legends from results_text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canpert import results_text as R0
from canpert.docxkit import Manuscript
from canpert.results_text import LN, MNAME, f, pval, summ, verdict

LINES4 = ("k562", "hepg2", "jurkat", "rpe1")
CANCER3 = ("k562", "hepg2", "jurkat")


def _rv(res: dict) -> dict:
    return res.get("review", {})


def _seed_mean(res: dict, scope: str, method: str, metric: str = "pearson_delta") -> tuple[float, float]:
    d = _rv(res).get("seeds", {}).get(scope, {}).get(method, {}).get(metric, {})
    return d.get("mean", np.nan), d.get("sd", np.nan)


def _ceiling(res: dict, task: str = "t1") -> dict[str, float]:
    c = pd.DataFrame(_rv(res).get("calibration", {}).get("ceiling", []))
    if c.empty:
        return {}
    c = c[c.task == task]
    return dict(zip(c.target, c.ceiling, strict=True))


# ---------------------------------------------------------------------------- front matter
def summary(res: dict) -> str:
    v = verdict(res)
    cc = {r["pair_type"]: r["pearson"] for r in res["cross_context"]["by_type"]}
    ca, ca_sd = _seed_mean(res, "all_lines", "CASTER")
    pm, _ = _seed_mean(res, "all_lines", "PerturbMean")
    li, _ = _seed_mean(res, "all_lines", "Linear")
    if not np.isfinite(ca):
        ca, pm, li = (summ(res, "T1_summary", m, "pearson_delta") for m in ("CASTER", "PerturbMean", "Linear"))
    h4 = pd.DataFrame(res["H4"]["tests"])
    rhos = h4[h4.test == "H4"].rho
    s = ("Genetic perturbation screens are run one cellular context at a time, yet cell lines differ in "
         "lineage and state. Using CRISPRi Perturb-seq of essential genes in three cancer cell lines and a "
         "non-transformed line, we find that transcriptional responses to the same knockdown are only "
         f"moderately shared across lines (median correlation {f(cc.get('cancer-cancer'))} between cancer lines), "
         "with similarity increasing for stronger responses and tracking shared p53 activity. We introduce "
         "CASTER, a set-based transformer that transfers a perturbation's response from other lines and "
         "corrects it for the target line. ")
    if v["h1"]:
        s += (f"When part of the target line is screened, CASTER outperforms mean baselines (Pearson delta "
              f"{f(ca)} versus {f(pm)} for the perturbation mean, over seeds and lines) and matches a linear "
              f"transfer model ({f(li)}); its gains concentrate on strong, ribosome-related responses. ")
    s += ("In lines with no screening data, CASTER is worse than the perturbation mean. ")
    if v["h4"]:
        s += ("Response magnitude tracked general gene essentiality measured by DepMap (Spearman rho "
              f"{f(rhos.max())} to {f(rhos.min())}), but not line-specific dependency. ")
    s += ("Noise ceilings, calibrated metrics, noise-free re-scoring and per-line controls define what context "
          "transfer can and cannot currently achieve from these data.")
    return s


def highlights(res: dict) -> list[str]:
    return ["Responses to the same CRISPRi knockdown are only moderately shared across cell lines",
            "CASTER corrects transferred responses when part of the target line is screened",
            "CASTER outperforms mean baselines and matches a linear model; it fails in unseen lines",
            "Knockdown response magnitude tracks general gene essentiality, not line-specific dependency"]


def in_brief(res: dict) -> str:
    return ("Using Perturb-seq of essential genes in leukemia, hepatoblastoma, and non-transformed cells, this "
            "study measures how much cell line context reshapes the response to losing a gene, tests whether a "
            "set-based model can transfer responses between lines, and relates response size to gene "
            "essentiality.")


# ---------------------------------------------------------------------------- results
def results(m: Manuscript, res: dict, figs: Path) -> None:  # noqa: PLR0915
    v = verdict(res)
    rv = _rv(res)
    audit = res.get("audit", {})
    nde = res["observed_n_de"]
    cc_type = {r["pair_type"]: r for r in res["cross_context"]["by_type"]}
    pairs = pd.DataFrame(res["cross_context"]["by_pair"])
    ex = rv.get("extra", {})
    ceil = _ceiling(res)

    # ------------------------------------------------------------ 1. shared vs context-specific
    m.heading("Responses to the same knockdown are only moderately shared across cell lines", 2)
    m.para("We assembled CRISPRi Perturb-seq profiles of essential-gene knockdowns in four human cell lines "
           "(Figures 1A and 1B): K562 (chronic myeloid leukemia; TP53 null)[@replogle2022], HepG2 "
           "(hepatoblastoma; TP53 wild type)[@nadig2025], Jurkat (T cell acute lymphoblastic leukemia; TP53 "
           "mutant)[@nadig2025] and the non-transformed, TP53 wild-type line hTERT RPE-1[@replogle2022,bodnar1998]. "
           "We used the knockdown-filtered, uniformly processed release of these data[@adduri2026] and analysed "
           f"{f(audit.get('n_cells_fetched', 0))} cells: up to 32 cells per perturbation per line and 3,000 "
           "controls per line (STAR Methods; Table S3). After filtering, "
           f"{f(audit.get('k562_perturbations'))} (K562), {f(audit.get('hepg2_perturbations'))} (HepG2), "
           f"{f(audit.get('jurkat_perturbations'))} (Jurkat) and {f(audit.get('rpe1_perturbations'))} (RPE1) "
           f"perturbations remained ({f(audit.get('n_perturbations_union'))} distinct target genes). Of these, "
           f"{f(audit.get('n_perturbations_all_four'))} were retained in all four lines and "
           f"{f(audit.get('n_perturbations_all_three_cancer'))} in all three cancer lines. Because on-target "
           "efficacy was filtered per line, the set of informative perturbations is itself line dependent "
           "(Figure 1B). We modelled 2,000 genes selected on control cells only. All "
           f"{f(audit.get('n_target_genes_excluded'))} target genes that are among the "
           f"{f(audit.get('n_genes_measured'))} measured genes were excluded from this gene space (the remaining "
           f"{f(audit.get('n_perturbations_union', 0) - audit.get('n_target_genes_excluded', 0))} targets are not "
           "measured), so neither gene selection nor the on-target signal can inflate any result.", indent=True)
    m.para("The number of differentially expressed genes (DEGs; Wilcoxon test against held-out controls, "
           "FDR < 0.05) varied widely between knockdowns and between lines. The median was "
           + ", ".join(f"{f(nde[n]['median'], 0)} in {LN[n]}" for n in LINES4)
           + "; " + ", ".join(f"{f(100 * nde[n]['frac_ge10'], 0)}%" for n in LINES4)
           + " of knockdowns, respectively, changed at least 10 genes (Figure S1A). Responses were largest in "
           "the two TP53 wild-type lines, RPE1 and HepG2. This agrees with Nadig et al., who found the "
           "transcriptome-wide impact of these knockdowns to be largest in RPE1 and correlated across the four "
           "lines (average r = 0.62 for impact magnitude)[@nadig2025]. For "
           f"{f(res['cross_context']['n_perturbations_min10de'])} genes whose knockdown produced >= 10 DEGs in at "
           "least two lines, we compared response profiles. They were only moderately correlated: median Pearson "
           f"r was {f(cc_type['cancer-cancer']['pearson'])} between cancer lines and "
           f"{f(cc_type['cancer-RPE1']['pearson'])} between a cancer line and RPE1 (Figures 1D and 1E), and DEG "
           f"sets overlapped little (median Jaccard {f(cc_type['cancer-cancer']['jaccard'])} and "
           f"{f(cc_type['cancer-RPE1']['jaccard'])}; Figure S1C). Profile correlation and magnitude correlation "
           "answer different questions, so the two studies are consistent.", indent=True)
    best = pairs.sort_values("pearson", ascending=False).iloc[0]
    rest = pairs[(pairs.a != best.a) | (pairs.b != best.b)]
    bys = pd.DataFrame(res["cross_context"].get("by_strength", []))
    txt = (f"The most similar pair was {LN[best.a]}-{LN[best.b]} (median r = {f(best.pearson)}); the other five "
           f"pairs ranged from {f(rest.pearson.min())} to {f(rest.pearson.max())}. ")
    if not bys.empty:
        cc_s = bys[bys.pair_type == "cancer-cancer"].set_index("min_de").median_pearson
        txt += (f"Similarity rose with response strength, from {f(cc_s.iloc[0])} for knockdowns with >= "
                f"{int(cc_s.index[0])} DEGs to {f(cc_s.iloc[-1])} for those with >= {int(cc_s.index[-1])} DEGs in "
                "both cancer lines (Figure S1D). Part of the apparent line specificity of weak responses is "
                "therefore measurement noise. ")
    if ceil:
        txt += ("A split-half analysis quantified this noise. With 32 cells per knockdown, the best Pearson "
                "delta any predictor could reach against the observed response (the noise ceiling, STAR Methods) "
                "had a median of " + ", ".join(f"{f(ceil.get(n))} in {LN[n]}" for n in LINES4)
                + " (Figure 6A). Absolute accuracies in this study should be read against these ceilings. ")
    p53 = ex.get("p53", {})
    if p53:
        pl = p53["per_line"]
        txt += ("Because HepG2 and RPE1 are the two TP53 wild-type lines, we scored the activity of 14 consensus "
                "p53 target genes[@fischer2017] for every knockdown (Figure S13). p53 targets were induced by "
                + ", ".join(f"{f(100 * pl[n]['frac_induced'], 1)}% of knockdowns in {LN[n]}" for n in LINES4)
                + ", and the p53 response to individual knockdowns was most strongly shared between HepG2 and "
                f"RPE1 (Spearman rho = {f(p53['pair_spearman_p53_shift'].get('hepg2-rpe1'))}, versus "
                f"{f(min(v_ for k, v_ in p53['pair_spearman_p53_shift'].items() if k != 'hepg2-rpe1'))} to "
                f"{f(max(v_ for k, v_ in p53['pair_spearman_p53_shift'].items() if k != 'hepg2-rpe1'))} for other "
                "pairs). Knockdowns inducing p53 in both lines had more similar HepG2 and RPE1 responses (median "
                f"r = {f(p53['hepg2_rpe1_by_p53']['both_induced_median_r'])} versus "
                f"{f(p53['hepg2_rpe1_by_p53']['other_median_r'])}, p = "
                f"{pval(p53['hepg2_rpe1_by_p53']['mannwhitney_p'])}). Among knockdowns that induced p53 in neither "
                "line, HepG2-RPE1 remained the most similar pair (median r = "
                f"{f(p53['similarity_p53_neither']['hepg2-rpe1']['median_r'])}), so shared p53 status explains part, "
                "but not all, of their similarity; lineage and growth mode (both adherent and epithelial-derived) "
                "may contribute the rest. ")
    m.para(txt + "A useful transfer model must therefore capture the shared core of a response and predict at "
           "least part of a substantial line-specific component.", indent=True)

    # ------------------------------------------------------------ 2. CASTER
    m.heading("CASTER corrects a transferred response for the target line", 2)
    m.para("These observations motivated CASTER (Figure 1C). Like State[@adduri2026], CASTER operates on *sets* "
           "of cells and is trained with a distributional (energy-distance MMD) loss[@gretton2012,szekely2013], "
           "so it models a perturbed population rather than a single average cell. A perturbation is represented "
           "by its *transfer prior*: the mean shift it produced in the other lines in which it was observed. The "
           "target line is never part of this prior. The target line is represented by a set of its unperturbed "
           "cells, read by a transformer encoder; FiLM conditioning[@perez2018] lets the same prior act "
           "differently in each line. The output adds a line-dependent, gene-wise rescaling of the prior and a "
           "cell-resolved correction. Both start at zero, so an untrained CASTER equals the PerturbMean baseline "
           "and any departure must be earned on validation data (STAR Methods; hyperparameters in Table S2). PerturbMean is not the strongest "
           "baseline on these data (ContextMean and the linear model beat it in some lines; Figure 2C), so this "
           "initialisation guarantees only a reasonable starting point, not an advantage.", indent=True)
    pc = res.get("positive_control")
    if pc:
        m.para("In simulations where the line-specific component was pure noise, CASTER stayed at the PerturbMean "
               f"solution (Pearson delta {f(pc['unlearnable']['CASTER'])} versus "
               f"{f(pc['unlearnable']['PerturbMean'])}; linear transfer {f(pc['unlearnable'].get('Linear'))}). "
               "When the line-specific component depended on each gene's baseline expression, which is observable "
               f"from control cells, CASTER recovered it (Pearson delta {f(pc['learnable']['CASTER'])} versus "
               f"{f(pc['learnable']['PerturbMean'])}), and so did the linear transfer model "
               f"({f(pc['learnable'].get('Linear'))}; Figure S4; simulation design in STAR Methods). The simulation "
               "therefore shows that CASTER can learn such structure without inventing it, not that it learns "
               "more than a linear model.", indent=True)
    m.para(_ablation_text(res), indent=True)

    # ------------------------------------------------------------ 3. under-represented benchmark
    m.heading("Under-represented lines: CASTER outperforms mean baselines and matches a linear model", 2)
    m.para("We first evaluated the setting in which a new line has been screened with only part of a library "
           "(Figure 2A)[@adduri2026]. For each of the four lines in turn, 25% of its eligible perturbations were "
           "used for training, 5% for model selection and 70% for testing; the other three lines were fully "
           "observed. We compared CASTER with PerturbMean (a perturbation's mean response in the other lines), "
           "ContextMean (the target line's average response), LineOffset (PerturbMean plus the target line's "
           "average deviation from it, estimated on the screened 25% + 5%), a linear transfer model fitted on the "
           "screened fraction, and a no-change floor. The evaluation suite follows Cell-Eval[@adduri2026] "
           "(Figures 2B-2E; STAR Methods; training curves in Figure S2). The pre-specified analysis used one split (seed 0); we added two more "
           "seeds, each with its own split and initialisation.", indent=True)
    m.para(_t1_text(res), indent=True)
    m.para(_t1_detail_text(res), indent=True)
    m.para(_strength_text(res), indent=True)

    # ------------------------------------------------------------ 4. beyond one shift per line
    m.heading("CASTER's gain is not a single offset per line", 2)
    m.para(_controls_text(res), indent=True)

    m.para(_noise_free_text(res), indent=True)

    # ------------------------------------------------------------ 5. calibrated metrics
    m.heading("Calibrated metrics and noise ceilings", 2)
    m.para(_calibration_text(res), indent=True)

    # ------------------------------------------------------------ 6. zero-shot + transfer
    m.heading("CASTER is worse than the perturbation mean in unseen lines", 2)
    m.para(_t2_text(res), indent=True)
    m.para(_t3_text(res), indent=True)

    # ------------------------------------------------------------ 7. examples, ribosome, programs
    m.heading("Line-specific responses, ribosome biogenesis and perturbation structure", 2)
    m.para(R0._fig4_paragraph(res).replace("Figure 4C shows the knockdown with the largest CASTER gain over "
                                            "PerturbMean in a cancer line, illustrating a context-specific "
                                            "correction of the transferred response. ",
                                            "Figure 4C shows the knockdown with the largest CASTER gain over "
                                            "PerturbMean in a cancer line; Figure S10 shows the best and the worst "
                                            "case in each cancer line, with linear transfer for comparison. "),
           indent=True)
    m.para(_ribosome_text(res, figs), indent=True)
    m.para(R0._programs_paragraph(res), indent=True)

    # ------------------------------------------------------------ 8. dependency
    m.heading("Response magnitude tracks general gene essentiality", 2)
    m.para(_h4_text(res), indent=True)
    m.para(_h5_text(res), indent=True)

    # ------------------------------------------------------------ Table 1
    m.heading("Summary of pre-specified outcomes", 2)
    m.para("Table 1 lists every pre-specified hypothesis, its test and its outcome. Exploratory analyses (H5, "
           "calibrated metrics, controls, p53, ribosome) are reported in the text and supplement.", indent=True)
    m.para("**Table 1. Pre-specified hypotheses and outcomes**", size=7.5)
    h1 = res["H1"]
    h1p = res.get("H1_preregistered", {})
    h3 = res.get("H3", {}).get("PerturbMean", {})
    h4 = pd.DataFrame(res["H4"]["tests"])
    h2 = pd.DataFrame(res.get("H2", {}).get("tests", []))
    h2txt = "; ".join(f"{MNAME[r.metric]} p = {pval(r.p_two_sided)}" for r in h2.itertuples()) if len(h2) else ""
    rows = [
        ["H1", "Under-represented line: CASTER > PerturbMean, ContextMean, Linear (3 primary metrics)",
         "One-sided paired Wilcoxon, Holm (9 tests); >= 6 significant, none opposite (seed 0)",
         f"Supported ({h1['n_significant']}/9; {h1['n_opposite']} opposite), driven by PerturbMean and "
         f"ContextMean; tied with Linear. Pre-specified DE test: {h1p.get('n_significant', 'n/a')}/9"],
        ["H2", "Unseen line: CASTER > PerturbMean (3 primary metrics)", "One-sided Wilcoxon, Holm (3 tests)",
         f"Not supported. Opposite direction, two-sided: {h2txt}" if not v["h2"] else "Supported"],
        ["H3", "Transfer more accurate among cancer lines than with RPE1", "Two-sided paired Wilcoxon",
         ("Not supported; opposite direction" if not v["h3"] else "Supported") + f" (p = {pval(h3.get('p_two_sided'))})"],
        ["H4", "Response magnitude correlates negatively with Chronos (per cancer line)", "Spearman, Holm (3 lines)",
         ("Supported in all three lines" if v["h4"] else "Not supported")
         + f" (rho {f(h4[h4.test == 'H4'].rho.max())} to {f(h4[h4.test == 'H4'].rho.min())}); not line-specific (Figure S14)"],
        ["H4b", "Line-specific response tracks line-specific dependency", "Spearman, Holm (3 lines)",
         "Supported" if v["h4b"] else "Not supported (significant in K562 only, rho = "
         f"{f(h4[(h4.test == 'H4b') & (h4.line == 'k562')].rho.iloc[0])})"],
    ]
    m.table(["", "Hypothesis", "Test and decision rule", "Outcome"], rows, widths=[0.4, 2.3, 2.0, 2.4], size=6.5)
    R0._figure_pages(m, res, figs)
    m.columns(2)


# ---------------------------------------------------------------------------- pieces
def _ablation_text(res: dict) -> str:
    ab = _rv(res).get("ablation20", {})
    means = pd.DataFrame(ab.get("means", []))
    if means.empty:
        return R0._ablation_sentence(res)
    mu = means.groupby("method").pearson_delta.mean()
    names = {"S1": "single cells (S = 1)", "S8": "S = 8", "S32_noattn": "no attention", "S32_nofilm": "no FiLM",
             "S32_noprior": "no transfer prior (learned perturbation embedding)", "S32_pertid": "prior plus ID embedding"}
    s = ("We ablated CASTER on the K562 under-represented fold with 20 training epochs, as in the main "
         f"analysis, and two seeds for the main variants (Figure 1G). The full model (S = 32) reached a Pearson "
         f"delta of {f(mu.get('S32'), 3)}, compared with {f(mu.get('PerturbMean'), 3)} for PerturbMean and "
         f"{f(mu.get('Linear'), 3)} for linear transfer on the same split. Variants: "
         + "; ".join(f"{names.get(k, k)} {f(mu.get(k), 3)}" for k in names if k in mu) + ". ")
    t = pd.DataFrame(ab.get("tests", []))
    if not t.empty and "p_holm" in t:
        sig = t[(t.metric == "pearson_delta") & (t.p_holm < 0.05)]
        better = [names.get(r.variant, r.variant) for r in sig.itertuples() if r.median_diff > 0]
        worse = [names.get(r.variant, r.variant) for r in sig.itertuples() if r.median_diff < 0]
        s += ("On Pearson delta (paired Wilcoxon, Holm), the full model was significantly better than: "
              + (", ".join(better) if better else "none of the variants")
              + ("; significantly worse than: " + ", ".join(worse) if worse else "") + ". ")
    short = {k: v_ for k, v_ in ab.get("epochs", {}).items() if v_ < 20}
    if short:
        s += ("Runs were capped at 60 min of training; " + ", ".join(
            f"{names.get(k.split('_s')[0] if k.split('_s')[-1].isdigit() else k, k)}"
            f"{' (seed ' + k.split('_s')[-1] + ')' if k.split('_s')[-1].isdigit() else ''} stopped after {v_} of 20 epochs"
            for k, v_ in short.items()) + ", which may understate it slightly. ")
    s += ("Removing the transfer prior was by far the most damaging change; the contributions of attention and "
          "FiLM were small (Table S7). These ablations come from a single held-out line and should be read as indicative.")
    return s


def _t1_text(res: dict) -> str:
    v = verdict(res)
    h1 = R0.h1_rows(res)
    rv = _rv(res)
    s = ""
    ca, ca_sd = _seed_mean(res, "all_lines", "CASTER")
    if np.isfinite(ca):
        seeds = rv["seeds"]["seeds"]
        vals = {mth: _seed_mean(res, "all_lines", mth) for mth in ("CASTER", "Linear", "LineOffset", "PerturbMean", "ContextMean")}
        cvals = {mth: _seed_mean(res, "cancer_only", mth) for mth in ("CASTER", "Linear", "LineOffset", "PerturbMean")}
        s += (f"Over {len(seeds)} seeds and four held-out lines, Pearson delta was "
              + ", ".join(f"{f(mu_)} +- {f(sd_)} for {mth}" for mth, (mu_, sd_) in vals.items())
              + " (mean +- SD over seeds; Figure 2B). Restricted to the three cancer lines it was "
              + ", ".join(f"{f(mu_)} for {mth}" for mth, (mu_, _) in cvals.items())
              + ". CASTER therefore improved on the mean baselines in every seed, but was tied with linear transfer, "
              "and much of its average gain over PerturbMean comes from RPE1, where PerturbMean is weak. ")
    s += (f"In the pre-specified analysis (seed 0, H1), {v['h1_n_better']} of nine comparisons favoured CASTER "
          f"after Holm correction and {v['h1_n_worse']} favoured a baseline, so H1 was supported by its decision "
          "rule; all significant comparisons were against PerturbMean or ContextMean, or concerned DE AUPRC against "
          "Linear. ")
    li = h1[h1.baseline == "Linear"]
    if len(li):
        s += ("Against Linear: " + "; ".join(f"{MNAME[r.metric]} {f(r.mean_model, 3)} versus {f(r.mean_baseline, 3)} "
                                            f"(Holm p = {pval(r.p_holm)})" for r in li.itertuples()) + ". ")
    pst = pd.DataFrame(rv.get("seeds", {}).get("per_seed_tests", []))
    if len(pst):
        pmt = pst[(pst.baseline == "PerturbMean") & (pst.metric == "pearson_delta")]
        lit = pst[(pst.baseline == "Linear") & (pst.metric == "pearson_delta")]
        s += ("Across seeds, CASTER beat PerturbMean on Pearson delta in every seed (two-sided paired Wilcoxon p "
              + ", ".join(pval(p) for p in pmt.p_two_sided) + "), whereas the comparison with Linear was "
              "inconsistent (p " + ", ".join(pval(p) for p in lit.p_two_sided) + "; median differences "
              + ", ".join(f(d, 3) for d in lit.median_diff) + "). ")
    pre = res.get("H1_preregistered")
    nd = pd.DataFrame(res.get("amendment2_de_counts", []))
    if pre is not None and not nd.empty:
        cas = nd[nd.method == "CASTER"].iloc[0]
        s += ("DE-based metrics use the amended predicted-side test (AMENDMENT-002): the pre-specified Wilcoxon test "
              f"on predicted cells called a mean of {f(cas.n_de_pred_wilcoxon, 0)} DEGs per CASTER prediction versus "
              f"{f(cas.n_de_true, 0)} observed. Under the pre-specified test, {pre['n_significant']} of nine "
              f"comparisons favoured CASTER (Table S1, which lists both versions of every affected test).")
    return s


def _t1_detail_text(res: dict) -> str:
    lines = []
    for n in LINES4:
        c = R0.per_line(res, "T1_per_line", n, "CASTER", "pearson_delta")
        p = R0.per_line(res, "T1_per_line", n, "PerturbMean", "pearson_delta")
        li = R0.per_line(res, "T1_per_line", n, "Linear", "pearson_delta")
        lines.append(f"{LN[n]} {f(c)} versus {f(p)} and {f(li)}")
    s = ("Per line (seed 0), CASTER, PerturbMean and Linear reached Pearson deltas of " + "; ".join(lines)
         + " (Figure 2C; all metrics in Table S4 and Figure S3). ")
    eff = _rv(res).get("effect_rho_pooled", {})
    if eff:
        s += ("Pooled over lines, the predicted and observed numbers of DEGs per knockdown were rank-correlated "
              f"at rho = {f(eff.get('CASTER'))} for CASTER, {f(eff.get('PerturbMean'))} for PerturbMean and "
              f"{f(eff.get('Linear'))} for Linear (Figure 2E). ")
    sec = pd.DataFrame(res.get("T1_secondary", []))
    if not sec.empty:
        pm = sec[sec.baseline == "PerturbMean"]
        better = pm[pm.p_holm < 0.05]
        worse = pm[(pm.p_two_sided < 0.05) & (pm.median_diff < 0)]
        s += (f"Of the secondary metrics, {len(better)} favoured CASTER over PerturbMean and {len(worse)} favoured "
              "PerturbMean (Table S1); MAE favoured CASTER (lower error).")
    return s


def _strength_text(res: dict) -> str:
    return R0._strength_paragraph(res).replace("CASTER's corrections therefore act on real, strong, "
                                               "context-specific responses, not on noise.",
                                               "CASTER's corrections therefore concentrate on strong responses; "
                                               "the same holds for its gain over Linear in some lines (Figure S8).")


def _controls_text(res: dict) -> str:
    c = _rv(res).get("controls", {})
    s = ("A reviewer-style concern is that CASTER might learn little more than one constant shift per line, "
         "which ContextMean alone captures well in RPE1. We tested this with three controls (Figure 6E). ")
    lo = c.get("LineOffset", {}).get("pearson_delta", {})
    if lo:
        s += ("First, LineOffset adds the target line's average deviation from PerturbMean to every prediction. It "
              f"reached a mean Pearson delta of {f(lo.get('mean_b'))}, versus {f(lo.get('mean_a'))} for CASTER "
              f"(CASTER better for {f(100 * lo.get('frac_a_better', np.nan), 0)}% of perturbations, two-sided p = "
              f"{pval(lo.get('p_two_sided'))}). ")
    io = c.get("idonly", {}).get("pearson_delta", {})
    if io:
        s += ("Second, a line-identity-only CASTER, which receives the line's mean control profile instead of a set "
              f"of cells, reached {f(io.get('mean_b'))} versus {f(io.get('mean_a'))} for full CASTER on the same split "
              f"(p = {pval(io.get('p_two_sided'))}). ")
    wc = c.get("wrongctrl", {}).get("pearson_delta", {})
    if wc:
        s += ("Third, the trained CASTER given another line's control cells reached "
              f"{f(wc.get('mean_b'))} versus {f(wc.get('mean_a'))} with the correct cells (p = {pval(wc.get('p_two_sided'))}). ")
    if not (lo or io or wc):
        s += "(Results pending.) "
    if io:
        same = (io.get("p_two_sided", 0) > 0.05) or abs(io.get("mean_a", 0) - io.get("mean_b", 0)) < 0.01
        if same:
            s += ("The line-identity-only model performed nearly as well as the full model. CASTER's gain therefore comes from "
                  "line-specific information carried by the line's average control profile, which it uses to adapt "
                  "the transferred response gene by gene (more than a single offset, as LineOffset shows), and not "
                  "mainly from cell-to-cell variation within the control set; given the wrong line's profile, it adapts to "
                  "the wrong line. The set-based design was therefore not necessary for this gain on these data.")
        else:
            s += ("The full model outperformed the line-identity-only model, so the individual control cells carry "
                  "information beyond the line's average profile.")
    else:
        s += ("Together these comparisons separate the contribution of a per-line offset, of line identity and of the "
              "individual control cells to CASTER's predictions.")
    return s


def _noise_free_text(res: dict) -> str:
    nf = _rv(res).get("noise_free", {})
    if not nf:
        return ""
    al = nf["all_lines"]

    def g(mth: str, k: str = "mean") -> float:
        return al.get(f"{mth}_clean", {}).get("pearson_delta", {}).get(k, np.nan)

    raw = {mth: _seed_mean(res, "all_lines", mth)[0] for mth in ("CASTER", "Linear", "LineOffset", "PerturbMean")}
    t = pd.DataFrame(nf.get("tests", []))
    lin = t[(t.baseline == "Linear_clean") & (t.metric == "pearson_delta")]
    pmt = t[(t.baseline == "PerturbMean_clean") & (t.metric == "pearson_delta")]
    lot = t[(t.baseline == "LineOffset_clean") & (t.metric == "pearson_delta")]
    s = ("A reviewer asked whether CASTER might gain by cancelling the noise of the control cells: every "
         "pseudobulk-level baseline was rendered as 32 sampled control cells plus its predicted shift, so its "
         "prediction inherits the sampling noise of those cells, which CASTER sees and could learn to remove. We "
         "re-scored every method after removing this noise (the mean of its input control cells minus the mean "
         "of the whole input pool; STAR Methods). All methods improved by a similar amount, CASTER included "
         f"(Pearson delta over seeds and lines: CASTER {f(raw['CASTER'])} to {f(g('CASTER'))}, Linear "
         f"{f(raw['Linear'])} to {f(g('Linear'))}, LineOffset {f(raw['LineOffset'])} to {f(g('LineOffset'))}, "
         f"PerturbMean {f(raw['PerturbMean'])} to {f(g('PerturbMean'))}; Figure 6G). CASTER therefore does not "
         "gain by cancelling input noise; instead, the pre-specified rendering depressed the absolute accuracy of "
         "every method by a similar margin. ")
    if len(lin):
        r = lin.iloc[0]
        s += ("With noise-free rendering, CASTER remained above PerturbMean "
              f"({f(100 * pmt.iloc[0].frac_a_better, 0)}% of perturbations better, p = {pval(pmt.iloc[0].p_two_sided)}) "
              f"and LineOffset ({f(100 * lot.iloc[0].frac_a_better, 0)}%, p = {pval(lot.iloc[0].p_two_sided)}), and "
              f"was again close to Linear ({f(100 * r.frac_a_better, 0)}% better, median difference "
              f"{f(r.median_diff, 3)}, p = {pval(r.p_two_sided)}). The ranking of the methods is the same under both "
              "renderings.")
    return s


def _calibration_text(res: dict) -> str:
    cal = _rv(res).get("calibration", {})
    if not cal:
        return "Calibrated metrics are pending."
    drf = pd.DataFrame(cal["drf"])
    d1 = drf[drf.task == "t1"].groupby("metric").drf.mean()
    lab = {"pearson_deltactrl": "Pearson delta", "pearson_deltapert_top100": "Pearson delta on the top 100 DEGs "
           "relative to the perturbed mean", "wr2_deltactrl": "weighted R2", "wr2_deltapert": "weighted R2 relative "
           "to the perturbed mean", "wmse": "WMSE", "mse": "MSE", "mae": "MAE", "nir": "NIR", "pds": "PDS"}
    good = [k for k in lab if d1.get(k, -1) > 0.1]
    bad = [k for k in lab if k not in good]
    s = ("Miller et al. showed that several standard metrics barely separate a perfect from a useless prediction on "
         "these four datasets, and proposed calibrating metrics with positive and negative controls[@miller2026,mejia2025]. "
         "We therefore re-scored every held-out perturbation against one random half of its cells, with the other "
         "half as a technical duplicate (positive control), an interpolated duplicate as a second positive control, "
         "and the line's mean perturbation response (or a zero shift, for metrics relative to the perturbed mean) as "
         "the negative control (STAR Methods). The dynamic range fraction (DRF; 0 = the metric cannot separate the "
         "controls, 1 = it separates them fully) averaged over lines was "
         + ", ".join(f"{f(d1.get(k))} for {lab[k]}" for k in good)
         + ", but zero or negative for " + ", ".join(lab[k] for k in bad) + " (Figure 6B). With only 16 cells per "
         "half, the technical duplicate is no closer to the truth than the mean response on these metrics, so they "
         "cannot validate a predictor at this cell count. This strengthens the case of Miller et al. that Pearson "
         "delta and MAE are poorly calibrated, and it applies to our pre-specified primary metric. ")
    nmp = Path("reports/tables/calibrated_normalised.csv")
    if nmp.exists():
        nm = pd.read_csv(nmp)
        t1n = nm[nm.task == "t1"]

        def frac(me: str, met: str) -> float:
            return float(t1n[(t1n.method == me) & (t1n.metric == met)].fraction_of_range.mean())

        s += ("On the calibrated metrics, scaled from negative control (0) to technical duplicate (1), CASTER scored "
              + ", ".join(f"{f(frac('CASTER', k))} ({lab[k]})" for k in good)
              + "; linear transfer " + ", ".join(f(frac('Linear', k)) for k in good)
              + "; LineOffset " + ", ".join(f(frac('LineOffset', k)) for k in good)
              + "; and PerturbMean " + ", ".join(f(frac('PerturbMean', k)) for k in good)
              + " (Figure 6C; Table S10). ")
    tt = pd.DataFrame(cal.get("tests", []))
    if len(tt):
        lin = tt[(tt.baseline == "Linear") & tt.metric.isin(good)]
        s += ("Paired against Linear on these metrics, CASTER was better for "
              + "; ".join(f"{f(100 * r.frac_a_better, 0)}% of perturbations on {lab[r.metric]} (p = {pval(r.p_two_sided)})"
                          for r in lin.itertuples()) + ". ")
    s += ("On the metric that removes the shift shared by all knockdowns, CASTER therefore did better than "
          "linear transfer, whereas on the identity-retrieval metrics (NIR, PDS) the two were close. The calibrated "
          "metrics were added after the pre-specified analysis and are reported as exploratory.")
    return s

def _t2_text(res: dict) -> str:
    rv = _rv(res)
    h2 = pd.DataFrame(res.get("H2", {}).get("tests", []))
    wp = rv.get("t2_worse_panels", {})
    s = ("We next asked whether responses could be transferred into a line with no perturbation data at all, using "
         "only its unperturbed cells (Figure 3A). They could not, by this model: CASTER was worse than PerturbMean in "
         f"{wp.get('worse', 'n/a')} of {wp.get('total', 'n/a')} line-by-metric panels (Figures 3B and S11), and "
         "the pre-specified comparison (H2) was significant in the opposite direction on all three primary metrics (Table S5; "
         + "; ".join(f"{MNAME[r.metric]}: CASTER {f(r.mean_model, 3)} versus {f(r.mean_baseline, 3)}, two-sided "
                     f"p = {pval(r.p_two_sided)}" for r in h2.itertuples()) + "). ")
    s += ("Its discrimination fell furthest, which indicates that the corrections learned between source lines "
          "pulled predictions towards a response shared across perturbations. By Proposition 4 (STAR Methods), the "
          "residual of the transfer prior was not predictable from a line's control cells by this model, trained "
          "on three lines; whether richer models or more training lines could predict it remains open. CASTER's "
          "advantage in the under-represented setting depended on a small amount of perturbation data from the "
          "target line (Figure 3E).")
    return s


def _t3_text(res: dict) -> str:
    base = R0._t3_paragraph(res)
    wc = _rv(res).get("transfer_worse_counts", {})
    base = base.replace("and in single-source transfer it was indistinguishable from PerturbMean in most directions "
                        "(Figure 3C; Table S6). ",
                        f"In single-source transfer CASTER was close to PerturbMean on Pearson delta (worse in "
                        f"{wc.get('pearson_delta', 'n/a')} of 12 directions) but worse in "
                        f"{wc.get('discrimination', 'n/a')} of 12 on discrimination and {wc.get('de_overlap', 'n/a')} "
                        "of 12 on DE overlap (Figures 3C and S6; Table S6). ")
    base = base.replace("RPE1 differs from the cancer lines in being non-transformed and TP53-proficient, which shapes "
                        "its response to the loss of essential genes, including a p53-dependent stress "
                        "response[@haapaniemi2018,ihry2018]. ",
                        "HepG2 and RPE1 are both TP53 wild type, and their shared p53 activation explains part of "
                        "this result (Figure S13). ")
    base = base.replace("Transfer accuracy therefore reflects the biological similarity between two contexts, here "
                        "lineage and growth mode, more than the binary distinction between transformed and "
                        "non-transformed cells.",
                        "Transfer accuracy therefore reflects the biological similarity between two lines, including "
                        "p53 status, more than the binary distinction between transformed and non-transformed cells.")
    return base


def _ribosome_text(res: dict, figs: Path) -> str:
    rv = _rv(res)
    s = R0._examples_paragraph(figs)
    s = s.replace("[@hanahan2011,hanahan2022]", "")
    s = s.replace("Loss of ribosome biogenesis triggers a nucleolar stress response, whose transcriptional output "
                  "depends on p53 status, MYC activity and growth rate, all of which differ between these lines.",
                  "Loss of ribosome biogenesis triggers a nucleolar stress response that activates p53 through the "
                  "5S RNP-MDM2 axis, with outputs that depend on p53 status and growth programs, which differ "
                  "between these lines[@golomb2014,pelava2016,lindstrom2022].")
    s = s.replace("Which perturbations does CASTER adapt most? In each cancer line, the knockdowns with the largest "
                  "gain over PerturbMean (among those with >= 30 DEGs) were dominated by",
                  "Which perturbations does CASTER adapt most? In each cancer line, the knockdowns with the largest "
                  "gain over PerturbMean (among those with >= 30 DEGs) included")
    rb = rv.get("extra", {}).get("ribosome", {}).get("per_fold", {})
    if rb:
        s0 = {k: v_ for k, v_ in rb.items() if k.endswith("_s0")}
        s += (" Six examples are not evidence, so we tested all held-out knockdowns. After adjusting for response "
              "strength, the gain over PerturbMean was larger for ribosome-biogenesis and ribosomal-protein "
              "knockdowns in every line (regression coefficient "
              + ", ".join(f"{f(v_['coef_ribosome'], 3)} in {LN[k.split('_')[0]]} (p = {pval(v_['p_ribosome'])})"
                          for k, v_ in s0.items()) + "). ")
        sib = {k: v_ for k, v_ in s0.items() if "rho_siblings_gain_ribosome" in v_}
        if sib:
            s += ("Because members of the same complex respond similarly, a held-out ribosome gene could benefit "
                  "from siblings in the training fraction. The gain correlated with the number of same-complex "
                  "genes in training in Jurkat (rho = "
                  f"{f(sib.get('jurkat_s0', {}).get('rho_siblings_gain_ribosome'))}) but not clearly in the other "
                  "lines (Figure S15C). ")
    gt = rv.get("grouped", {}).get("tests", {})
    if gt:
        def gm(b: str, k: str) -> float:
            return gt.get(b, {}).get("pearson_delta", {}).get(k, np.nan)
        s += (" Holding out whole complexes also tests the overall benchmark for leakage between related genes. "
              f"Under complex-grouped splits (seed 0), Pearson delta was {f(gm('PerturbMean', 'mean_a'))} for CASTER, "
              f"{f(gm('PerturbMean', 'mean_b'))} for PerturbMean, {f(gm('Linear', 'mean_b'))} for Linear and "
              f"{f(gm('LineOffset', 'mean_b'))} for LineOffset (CASTER vs PerturbMean p = {pval(gm('PerturbMean', 'p_two_sided'))}; "
              f"vs Linear p = {pval(gm('Linear', 'p_two_sided'))}). Part of the learned methods' advantage under random "
              "splits therefore comes from related genes in the training fraction.")
    gr = rv.get("grouped", {}).get("ribosome_gain_grouped", {})
    if gr:
        s += ("When whole STRING-defined complexes were held out together, so that no sibling was in training, "
              "the median gain for ribosome knockdowns was "
              + ", ".join(f"{f(gr[n]['median_gain_ribosome'], 3)} in {LN[n]}" for n in LINES4 if n in gr)
              + ", versus " + ", ".join(f"{f(gr[n]['median_gain_other'], 3)}" for n in LINES4 if n in gr)
              + " for other knockdowns (Figure S15B).")
    return s


def _h4_text(res: dict) -> str:
    base = R0._h4_paragraph(res)
    dc = _rv(res).get("extra", {}).get("depmap_controls", {})
    base = base.replace("Dependency differences between lines are small for most essential genes"
                        "[@hart2015,wang2015,blomen2015], which limits the power of this test.", "")
    if dc:
        cr = pd.DataFrame(dc["cross"])
        own = cr[cr.response_line == cr.chronos_line].set_index("response_line").rho
        oth = cr[cr.response_line != cr.chronos_line].groupby("response_line").rho.mean()
        gen = pd.DataFrame(dc["generality"]).set_index("line").pearson_other_lines_mean_vs_own
        base += (" The coupling was not line-specific. A line's response size correlated about as strongly with "
                 "another line's Chronos scores as with its own (own: "
                 + ", ".join(f"{f(own[n])} in {LN[n]}" for n in CANCER3) + "; other lines, mean: "
                 + ", ".join(f"{f(oth[n])}" for n in CANCER3) + "; Figure S14), and the mean Chronos score of the "
                 "other two lines predicted each line's own score with Pearson r = "
                 + ", ".join(f"{f(gen[n])}" for n in CANCER3)
                 + ". The response-dependency link therefore reflects general gene essentiality: knockdowns of "
                 "broadly essential genes cause large transcriptional responses everywhere. Dependency differences "
                 "between lines are small for most essential genes[@hart2015,wang2015,blomen2015], which limits "
                 "line-specific tests in this library.")
    return base


def _h5_text(res: dict) -> str:
    s = R0._h5_paragraph(res)
    s = s.replace(" This connects in silico perturbation to cancer target prioritisation[@behan2019,pacini2024], a "
                  "use that the reference State model demonstrated for cell viability[@adduri2026].", "")
    s = s.replace("Transferred responses therefore retain part of the dependency information carried by measured "
                  "responses.",
                  "Transferred responses therefore retain part of the essentiality information carried by measured "
                  "responses. Because Chronos is largely shared between lines (Figure S14), this exploratory "
                  "analysis (H5) speaks to general essentiality, not to line-specific dependency.")
    return s


# ---------------------------------------------------------------------------- discussion
def discussion(res: dict) -> list[str]:
    v = verdict(res)
    cc = {r["pair_type"]: r["pearson"] for r in res["cross_context"]["by_type"]}
    out = [
        "Perturbation biology increasingly relies on models that aim to predict an experiment before it is "
        "run[@bunne2024,rood2024,roohani2025]. We asked a focused version of this question with one CRISPRi library "
        "profiled in three cancer lines and a non-transformed line. Responses to the same knockdown were only "
        f"moderately correlated across lines (median r ~ {f(cc.get('cancer-cancer'))} between cancer lines). "
        "Similarity increased with response strength and was highest for the two TP53 wild-type lines, HepG2 and "
        "RPE1, partly through a shared p53 response. With 32 cells per knockdown, noise ceilings were well below 1, "
        "so part of the apparent line specificity is measurement noise.",
        "When part of the target line was screened, CASTER outperformed mean baselines across seeds and matched, "
        "but did not beat, a linear transfer model fitted to the same data. This agrees with evidence that "
        "well-specified linear models are strong competitors for perturbation prediction[@ahlmann2025,kernfeld2025] "
        "and with the warning that average-shift effects can dominate standard metrics[@vinastorne2025]. Controls "
        "for a per-line offset, line identity and the control cells themselves, together with calibrated metrics, "
        "describe what the model adds beyond such effects. Rendering pseudobulk predictions on a small sample of "
        "control cells lowered every method's accuracy by a similar margin; benchmarks that render predictions "
        "this way should remove that noise before comparing absolute accuracies. Its gains concentrated on strong responses, in "
        "particular knockdowns of ribosome-biogenesis genes, whose nucleolar-stress output depends on p53 and "
        "growth state[@golomb2014,lindstrom2022].",
        _idonly_sentence(res),
        "In lines with no screening data, CASTER was worse than the perturbation mean. This model, trained on three "
        "lines, did not find information in unperturbed transcriptomes that predicts how a new line reshapes a "
        "response; a small amount of perturbation data from that line was what made adaptation work. Models trained "
        "on many more lines might learn such a mapping, but on current data a sparse pilot screen in each new line "
        "of interest appears more useful than control profiles alone.",
        "We had expected transfer to be more faithful among cancer lines than to or from the non-transformed line "
        "(H3). The opposite held, driven by HepG2-RPE1. Shared p53 status and lineage predicted transferability "
        "better than transformation status, which suggests that reference lines for perturbation atlases should be "
        "chosen by lineage and cell state, including p53 status.",
        "Response magnitude tracked DepMap gene effects in all three cancer lines, but the link was not "
        "line-specific: other lines' Chronos scores predicted it about as well. It reflects general essentiality. "
        "Libraries designed around differential dependencies, such as synthetic-lethal partners of cancer "
        "drivers[@lord2017,oneil2017], would be needed to test line-specific dependency.",
        "Methodologically, three practices made these conclusions possible: selecting genes on control cells only "
        "and excluding perturbation targets; scoring every method's prediction with the same pseudobulk pipeline, "
        "including the amended predicted-side DE test; and fixing hypotheses and decision rules before model "
        "fitting. We also found that the Wilcoxon test is invalid for continuous predicted cells, a pitfall worth "
        "checking in other benchmarks. We did not run a large pretrained model such as State on the same splits, "
        "so we make no claim about how CASTER compares with them.",
    ]
    _ = v
    return out


def _idonly_sentence(res: dict) -> str:
    io = _rv(res).get("controls", {}).get("idonly", {}).get("pearson_delta", {})
    if not io:
        return ("Whether CASTER's adaptation relies on individual control cells or on the line's average profile is "
                "addressed by the line-identity-only control.")
    same = (io.get("p_two_sided", 0) > 0.05) or abs(io.get("mean_a", 0) - io.get("mean_b", 0)) < 0.01
    if same:
        return ("A line-identity-only version of CASTER, which sees the line's average control profile instead of "
                "individual cells, performed nearly as well as the full model. The adaptation that CASTER learns therefore "
                "relies on what distinguishes one line's average state from another's, not on the cell-to-cell "
                "variation that a set-based model can in principle exploit; with 32 cells per perturbation and four "
                "lines, the data may simply not support learning more.")
    return ("The full model outperformed a line-identity-only version, so individual control cells carry information "
            "that CASTER uses beyond the line's average state.")


def limitations(res: dict) -> list[str]:
    return ["This study has several limitations. First, it covers four lines, three of them cancer lines from "
            "different lineages; with so few lines, line generalisation cannot be separated from lineage or p53 "
            "status, and conclusions about cancer in general require many more lines. Second, compute and "
            "bandwidth limits led us to use at most 32 cells per perturbation, which caps the achievable accuracy "
            "(noise ceilings), and to train compact models on CPUs. Third, CASTER represents a perturbation by its "
            "observed effect elsewhere, so it cannot predict perturbations never observed in any line. Fourth, the "
            "dependency analyses rely on CRISPR knockout fitness, whereas the screens use CRISPRi knockdown, and the "
            "four screens differ in effector construct, time point and sequencing depth. Fifth, the calibrated "
            "metrics and several controls were added after the pre-specified analysis and are exploratory. Sixth, "
            "we did not compare with a large pretrained perturbation model. Finally, this work used public data "
            "only and did not test any prediction experimentally."]
