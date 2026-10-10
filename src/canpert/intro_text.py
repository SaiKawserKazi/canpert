"""Result-independent Introduction text."""

INTRODUCTION = [
    "Pooled CRISPR screens with single-cell transcriptomic readouts (Perturb-seq) measure the "
    "transcriptome-wide consequence of a genetic perturbation in thousands of cells at "
    "once[@dixit2016,adamson2016,jaitin2016,datlinger2017]. CRISPR interference (CRISPRi)[@qi2013,gilbert2014] "
    "has made such screens scalable to every expressed or essential gene[@replogle2020,replogle2022], and "
    "they are now used to map gene function, genetic interactions and regulatory "
    "programs[@norman2019,replogle2022,jiang2025,przybyla2022]. In cancer research they have identified "
    "regulators of immune evasion, drug response and oncogenic variant "
    "function[@frangieh2021,papalexi2021,ursu2022,mcfarland2020,mcfalinefigueroa2024]. Each screen, "
    "however, is performed in one cellular context. A tumour is not one context: cancers differ by lineage, "
    "driver, ploidy and baseline transcriptional state, and these differences shape how a cell responds to "
    "losing a gene[@hanahan2011,hanahan2022,kinker2020,gavish2023]. Screening every perturbation in every "
    "relevant context is not feasible, so computational models that *transfer* perturbation responses across "
    "contexts are needed[@bunne2024,rood2024].",

    "Many models predict single-cell perturbation responses, including autoencoder and latent-arithmetic "
    "models[@lotfollahi2019,lotfollahi2023,lopez2018,bereket2023,piran2024], optimal-transport "
    "maps[@bunne2023], knowledge-graph models[@roohani2024] and single-cell foundation "
    "models[@theodoris2023,cui2024,hao2024]. Recently, State, a transformer that operates on *sets* of cells "
    "and is trained with a distributional loss, was shown to generalise perturbation effects to new contexts "
    "better than these approaches on several large datasets[@adduri2026]. At the same time, careful "
    "benchmarks have shown that simple baselines, including the mean response of a perturbation across "
    "training contexts and linear models, are hard to beat[@ahlmann2025,kernfeld2025,wu2025]. A "
    "community-wide challenge now asks for virtual-cell models that pass a 'Turing test' on held-out "
    "contexts[@roohani2025]. Two questions are therefore open. How much of a perturbation's response is "
    "really specific to the context, as opposed to shared? And does that context-specific part carry "
    "biological information, for example about which genes a particular cancer depends on?",

    "Here we address both questions in the setting where context matters most for cancer biology: the same "
    "essential-gene CRISPRi library profiled in three cancer cell lines from different lineages (K562 chronic "
    "myeloid leukaemia, HepG2 hepatoblastoma and Jurkat T cell acute lymphoblastic leukaemia) and in "
    "non-transformed hTERT RPE-1 cells[@replogle2022,nadig2025]. We first quantify how similar the responses "
    "to the same knockdown are across these contexts. We then introduce CASTER (Context-Adaptive Set "
    "Transformer for pErturbation Responses). CASTER represents a perturbation by its observed effect in "
    "other contexts (a *transfer prior*) and learns, from the unperturbed cells of the target context and, "
    "when available, a screened fraction of its perturbations, how that context rescales and reshapes the "
    "response. By construction CASTER starts at the PerturbMean transfer baseline and moves away from it only "
    "where held-out data support the change. We compare it with mean, offset and linear baselines, with "
    "calibrated metrics and noise ceilings, and with controls that remove the control cells or swap them "
    "between lines.",

    "We evaluate CASTER with a pre-specified protocol and a biologically interpretable evaluation suite "
    "(PertBench), modelled on the Cell-Eval framework[@adduri2026]. We test it in under-represented contexts, "
    "in completely unseen contexts and in every single-source transfer direction. Finally, we connect "
    "perturbation responses to cancer dependency, asking whether the magnitude and context specificity of a "
    "knockdown's transcriptional response track CRISPR fitness effects measured independently in the "
    "Cancer Dependency Map[@tsherniak2017,dempster2021,behan2019,pacini2024]. We report every pre-declared "
    "test, including those that do not support our hypotheses.",
]
