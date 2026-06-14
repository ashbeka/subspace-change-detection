const LINKS = {
  reset: { label: "Research reset audit", url: "RESEARCH_RESET_AUDIT.md" },
  brief: { label: "Project brief", url: "PROJECT_BRIEF.md" },
  runbook: { label: "Runbook", url: "RUNBOOK.md" },
  experiments: { label: "Experiment notes", url: "../notes/experiments.md" },
  methods: { label: "Method notes", url: "../notes/methods.md" },
  literature: { label: "Literature notes", url: "../notes/literature.md" },
  feedback: { label: "Feedback notes", url: "../notes/feedback.md" },
  bookmarks: { label: "Reference bookmarks", url: "../notes/reference_bookmarks.md" },
  tpami: { label: "Fukui and Maki TPAMI 2015", url: "https://www.cs.tsukuba.ac.jp/~kfukui/papers/TPAMI2408358-revised.pdf" },
  oscd: { label: "OSCD / FC-Siamese", url: "https://ieeexplore.ieee.org/document/8451652" },
  celik: { label: "Celik PCA-kmeans 2009", url: "https://ieeexplore.ieee.org/document/4802550" },
  irmad: { label: "Nielsen IR-MAD 2007", url: "https://pubmed.ncbi.nlm.nih.gov/17269639/" },
  geeImad: { label: "GEE MAD/iMAD tutorial", url: "https://developers.google.com/earth-engine/tutorials/community/imad-tutorial-pt1" },
  harmonized: { label: "Harmonized Sentinel-2 L2A", url: "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED" },
  multisenge: { label: "MultiSenGE dataset", url: "https://arxiv.org/abs/2004.09011" },
  xbd: { label: "xBD dataset", url: "https://arxiv.org/abs/1911.09296" },
  changeos: { label: "ChangeOS", url: "https://github.com/Z-Zheng/ChangeOS" },
  greenhouse: { label: "WUN abandoned greenhouse project", url: "https://wun.ac.uk/wun/research/view/mapping-abandoned-greenhouses-multimodal-fusion/" },
  greenhouseIndex: { label: "Sentinel-2 Plastic Greenhouse Index", url: "https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/plastic_greenhouse/" },
  tensorGds: { label: "n-mode GDS for tensor data", url: "https://arxiv.org/abs/1909.01954" },
  globalPcg: { label: "Global-PCG-10 greenhouse map", url: "https://essd.copernicus.org/articles/17/5065/2025/essd-17-5065-2025.html" },
  metricCd: { label: "Metric-CD WACV 2025", url: "https://openaccess.thecvf.com/content/WACV2025/html/Saha_Unsupervised_Change_Detection_in_Satellite_Images_Using_Deep-Learning_Based_Representations_WACV_2025_paper.html" }
};

const ROUTES = [
  {
    id: "convince",
    title: "0. Convince Yourself",
    subtitle: "Read this when the research idea feels forced.",
    summary: "Start from field problems, advisor feedback, current evidence, and then decide whether subspaces are useful.",
    tags: ["read first", "problem framing", "anti-bias"],
    nodes: ["problem", "field-map", "advisor-question", "current-evidence", "claim-boundary", "decision-reset"]
  },
  {
    id: "sensei",
    title: "1. Sensei-First Temporal Route",
    subtitle: "Highest advisor alignment after Jang/Aono meetings.",
    summary: "Generate date subspaces from Harmonized Sentinel-2, then measure first DS, second DS, and geodesic projection readiness.",
    tags: ["priority", "advisor", "temporal"],
    nodes: ["harmonized-s2", "sequence-feasibility", "date-subspaces", "first-second-ds", "geodesic", "sensei-report"]
  },
  {
    id: "spatial",
    title: "2. Spatial DS Verification Route",
    subtitle: "Answer the spatial information criticism on a labeled benchmark.",
    summary: "Use OSCD labels to compare global pixel DS, patch-vector DS, local-window DS, and close baselines.",
    tags: ["OSCD", "spatial", "evidence"],
    nodes: ["oscd", "global-pixel-ds", "patch-ds", "window-ds", "core5-result", "score-ablation", "failure-modes"]
  },
  {
    id: "subspace",
    title: "3. Subspace Math Route",
    subtitle: "The concepts you need to explain in seminar.",
    summary: "Follow the chain from PCA bases to canonical angles, DS/GDS, KDS/KGDS, Grassmann/geodesic tools, and paper-to-code checks.",
    tags: ["math", "subspace", "seminar"],
    nodes: ["pca-basis", "canonical-angles", "ds-gds", "kpca-kds", "kgds", "grassmann", "paper-code"]
  },
  {
    id: "baselines",
    title: "4. Baseline Pressure Route",
    subtitle: "Know what can beat us and what must be cited.",
    summary: "Classical and deep change-detection baselines that prevent weak novelty claims.",
    tags: ["baselines", "literature", "comparison"],
    nodes: ["raw-cva", "pca-diff", "celik", "irmad", "unet-siamese", "modern-cd", "metrics"]
  },
  {
    id: "future",
    title: "5. Future And Pivot Route",
    subtitle: "Ideas worth keeping without overclaiming them now.",
    summary: "Green Learning, multiscale pyramids, MultiSenGE, greenhouses, xBD, and open-vocabulary change are future routes unless evidence catches up.",
    tags: ["future", "pivots", "applications"],
    nodes: ["green-learning", "multiscale-pyramid", "multisenge", "greenhouse", "xbd", "open-vocab", "reference-code-reminder"]
  }
];

const NODES = {
  problem: {
    type: "foundation",
    title: "Problem before method",
    summary: "The project should not be 'I have DS, so I need a satellite task.' The problem is interpretable changed-area evidence in multispectral satellite imagery.",
    tags: ["read first", "anti-bias"],
    learn: ["Remote-sensing change detection is already a large field.", "Subspace methods are a candidate representation, not the problem itself.", "The thesis survives if it honestly shows where DS helps or fails."],
    read: [LINKS.reset, LINKS.brief, LINKS.literature],
    code: ["No code first. Define the task and claim boundary before implementation."],
    experiment: "Research reset audit and Sensei feedback review."
  },
  "field-map": {
    type: "foundation",
    title: "Remote-sensing change detection field map",
    summary: "Surveys and modern work define the real problem space: labels, pseudo-change, registration, seasonal effects, semantic change, and foundation-model pressure.",
    tags: ["literature", "review papers"],
    learn: ["Do not compare only against weak classical baselines.", "Label-efficient CD is real, but modern methods already address it.", "Pseudo-change can be a stronger problem than raw accuracy."],
    read: [LINKS.literature, LINKS.metricCd],
    code: ["None. This is reading and framing."],
    experiment: "Use the literature map to select baselines and claims."
  },
  "advisor-question": {
    type: "risk",
    title: "Sensei's core question",
    summary: "Sensei agrees the algorithm can make a subspace, but warned that it can break spatial information. He also asked for time-sequential frames and first/second DS/geodesic trials.",
    tags: ["advisor", "priority", "risk"],
    learn: ["Spatial information is not a minor implementation detail.", "Time-sequential data is now the most advisor-aligned track.", "Jang/Aono meetings helped, but validation remains our responsibility."],
    read: [LINKS.feedback, LINKS.experiments],
    code: ["Upcoming: Harmonized Sentinel-2 sequence feasibility command."],
    experiment: "Generate date subspaces, then test first/second DS and geodesic quantities."
  },
  "current-evidence": {
    type: "experiment",
    title: "Current evidence snapshot",
    summary: "OSCD core5 showed patch DS improves over global DS, but PCA-diff/raw L2 remain stronger overall.",
    tags: ["implemented", "mixed result"],
    learn: ["Evidence supports 'construction matters,' not 'DS wins.'", "Patch DS is currently diagnostic, not a proven superior detector."],
    read: [LINKS.experiments, { label: "Core5 report", url: "experiment_reports/oscd_spatial_subspace_sweep_core5_2026-06-14.md" }],
    code: ["phase1/scripts/sweep_oscd_spatial_subspaces.py", "project_cli.py phase1-spatial-subspace-sweep"],
    experiment: "Inspect comparison grids, then run patch-score ablation."
  },
  "claim-boundary": {
    type: "risk",
    title: "Claim boundary",
    summary: "Do not claim completed disaster damage segmentation, xBD evaluation, or DS superiority over deep learning.",
    tags: ["forbidden claims", "thesis writing"],
    learn: ["Safe claims must be tied to code, experiments, advisor feedback, or literature.", "Negative results are acceptable if they explain method limits."],
    read: [LINKS.brief, LINKS.reset],
    code: ["No code. This controls writing and seminar answers."],
    experiment: "Use safe/unsafe claim table before slides or thesis text."
  },
  "decision-reset": {
    type: "foundation",
    title: "Decision reset",
    summary: "After spatial DS and temporal feasibility evidence, decide whether the thesis is spatial DS, temporal GDS/geodesic, priors for learning, pseudo-change diagnosis, or a pivot.",
    tags: ["decision gate"],
    learn: ["The thesis direction should be chosen after evidence, not before.", "The next two artifacts should make this decision easier."],
    read: [LINKS.reset, LINKS.experiments],
    code: ["No new code until current evidence is summarized."],
    experiment: "Decision gate after sequence feasibility and patch-score ablation."
  },
  "harmonized-s2": {
    type: "dataset",
    title: "Harmonized Sentinel-2 L2A",
    summary: "Sensei-supported source for time-sequential satellite images. First task is feasibility, not full GDS.",
    tags: ["priority", "dataset", "time sequence"],
    learn: ["Need area, date range, cloud filtering, bands, and frame count.", "This answers Sensei's question about time step and total frames."],
    read: [LINKS.harmonized, LINKS.experiments],
    code: ["Upcoming CLI: project_cli.py phase1-hs2-sequence-feasibility"],
    experiment: "Create a valid-frame table and per-date subspace readiness verdict."
  },
  "sequence-feasibility": {
    type: "experiment",
    title: "Sequence feasibility report",
    summary: "The next concrete artifact: date count, spacing, cloud/no-data, band stack shape, alignment assumptions, and candidate subspace definition.",
    tags: ["do now", "advisor-first"],
    learn: ["No first/second DS without a reliable sequence object.", "This prevents vague claims about temporal subspaces."],
    read: [LINKS.experiments, LINKS.runbook],
    code: ["To implement: phase1/scripts/audit_harmonized_s2_sequence.py", "To expose: project_cli.py phase1-hs2-sequence-feasibility"],
    experiment: "One small region, one controlled date range, one report."
  },
  "date-subspaces": {
    type: "method",
    title: "One subspace per date",
    summary: "For each aligned date image, define the sample unit and fit a basis. The definition must be explicit: pixels, patches, windows, objects, or deep features.",
    tags: ["seminar-critical", "subspace construction"],
    learn: ["Subspace construction is the first thing to explain.", "Basis shape depends on sample definition, not just dataset."],
    read: [LINKS.methods, LINKS.feedback],
    code: ["Reuse/extend phase1 DS utilities after sample object is defined."],
    experiment: "Report input matrix/tensor shape and basis shape for each date."
  },
  "first-second-ds": {
    type: "method",
    title: "First and second DS magnitude",
    summary: "Sensei asked whether changes in first and second DS magnitudes can be calculated soon.",
    tags: ["advisor", "temporal DS"],
    learn: ["First DS compares neighboring subspaces.", "Second DS should describe change in the change relation, so it needs a sequence."],
    read: [LINKS.literature],
    code: ["Reference code from Jang/Aono should be compared before final implementation."],
    experiment: "Measure magnitudes along a date sequence and plot them over time."
  },
  geodesic: {
    type: "method",
    title: "Geodesic projection/decomposition",
    summary: "Grassmannian/geodesic analysis may be meaningful for subspace dynamics over time.",
    tags: ["advisor", "Grassmann"],
    learn: ["Geodesic quantities are about relationships between subspaces, not pixel labels directly.", "This likely fits time sequences better than two-date OSCD."],
    read: [LINKS.literature, LINKS.tensorGds],
    code: ["phase1 geodesic utilities if present; otherwise implement after reference-code check."],
    experiment: "Plot geodesic/projection quantities against time and scene events."
  },
  "sensei-report": {
    type: "experiment",
    title: "Sensei-facing report",
    summary: "A short report should show frame count, subspace definition, first/second DS magnitude plots, and what failed.",
    tags: ["advisor output"],
    learn: ["Report both the thing Sensei asked for and the labeled OSCD verification track.", "Do not oversell performance."],
    read: [LINKS.feedback, LINKS.experiments],
    code: ["Generated from the sequence feasibility and temporal DS outputs."],
    experiment: "One concise PDF/Markdown summary after first sequence run."
  },
  oscd: {
    type: "dataset",
    title: "OSCD binary change benchmark",
    summary: "Current labeled benchmark: Sentinel-2 pre/post images with binary changed-area masks.",
    tags: ["implemented", "labels"],
    learn: ["OSCD supports binary change detection, not disaster damage severity.", "It is useful for verification because it has labels."],
    read: [LINKS.oscd, LINKS.brief],
    code: ["phase1/data/oscd_dataset.py", "phase2/data/oscd_seg_dataset.py"],
    experiment: "Use for spatial DS and supervised prior checks."
  },
  "global-pixel-ds": {
    type: "method",
    title: "Global pixel DS",
    summary: "Each valid Sentinel-2 pixel is one 13-D vector. PCA fits one global basis per date image.",
    tags: ["implemented", "risk"],
    learn: ["This is not per-band PCA and not whole-image-vector PCA.", "It ignores pixel position while fitting the subspace."],
    read: [LINKS.methods, LINKS.feedback],
    code: ["phase1/ds/pca_utils.py", "phase1/scripts/compare_oscd_spatial_subspaces.py"],
    experiment: "Baseline subspace construction for spatial comparison."
  },
  "patch-ds": {
    type: "method",
    title: "Patch-vector DS",
    summary: "A k x k x 13 patch becomes one vector. This preserves local spatial context inside the sample.",
    tags: ["implemented", "best DS candidate"],
    learn: ["Patch5 improved DS-family results in core5 but remained weaker than PCA-diff/raw L2.", "Next question: is the score formula the bottleneck?"],
    read: [LINKS.experiments, LINKS.methods],
    code: ["phase1/scripts/compare_oscd_spatial_subspaces.py", "phase1/scripts/sweep_oscd_spatial_subspaces.py"],
    experiment: "Run patch-score definition ablations."
  },
  "window-ds": {
    type: "method",
    title: "Local-window DS",
    summary: "Fit a subspace inside a spatial window and score locally. Current 128x128 setting was weak.",
    tags: ["implemented", "needs ablation"],
    learn: ["The idea is spatially natural, but current configuration did not help.", "Window size/stride and score aggregation need testing before dismissal."],
    read: [LINKS.experiments],
    code: ["phase1/scripts/compare_oscd_spatial_subspaces.py"],
    experiment: "Try smaller windows only after patch-score ablation."
  },
  "core5-result": {
    type: "experiment",
    title: "Core5 spatial sweep",
    summary: "Five cities, rank 4/6/8 configs. Patch DS > global DS on average; PCA-diff/raw L2 remain stronger.",
    tags: ["implemented", "mixed result"],
    learn: ["Use this as evidence for construction sensitivity, not DS superiority.", "Norcia patch5 exception is worth inspecting."],
    read: [{ label: "Core5 report", url: "experiment_reports/oscd_spatial_subspace_sweep_core5_2026-06-14.md" }, LINKS.experiments],
    code: ["project_cli.py phase1-spatial-subspace-sweep"],
    experiment: "Open comparison grids and write a failure-mode table."
  },
  "score-ablation": {
    type: "experiment",
    title: "Patch score ablation",
    summary: "Compare squared projection norm, unsquared norm, normalized ratio, residual energy, and robust scaling.",
    tags: ["next", "calibration"],
    learn: ["DS may be weak because the scalar map is poorly calibrated, not because the subspace is useless.", "Compare same samples, different scoring."],
    read: [LINKS.experiments, LINKS.methods],
    code: ["New script should extend spatial comparison scoring options."],
    experiment: "Patch3 and patch5 at rank 8 on core5 cities."
  },
  "failure-modes": {
    type: "experiment",
    title: "Failure-mode map inspection",
    summary: "Inspect water, vegetation, shadows, registration, seasonal texture, and label sparsity.",
    tags: ["do now", "interpretability"],
    learn: ["A good thesis can explain failures, not just report scores.", "False positives may be real spectral change outside OSCD label policy."],
    read: [LINKS.experiments],
    code: ["Open comparison_grid.png under the core5 sweep output."],
    experiment: "Short per-city table before more code."
  },
  "pca-basis": {
    type: "foundation",
    title: "PCA basis",
    summary: "Columns are samples. PCA returns basis vectors that span a low-dimensional subspace of the feature space.",
    tags: ["math", "seminar-critical"],
    learn: ["For OSCD global pixels, X is 13 x N.", "Rank 6 means six basis directions in 13-band space, not six pixels."],
    read: [LINKS.methods, LINKS.literature],
    code: ["phase1/ds/pca_utils.py"],
    experiment: "Always report sample definition and basis shape."
  },
  "canonical-angles": {
    type: "foundation",
    title: "Canonical angles",
    summary: "Canonical angles measure how two subspaces align. They are the bridge to DS/GDS, MSM, Grassmann methods, and some CCA views.",
    tags: ["math"],
    learn: ["Useful for comparing subspaces independent of individual basis-vector signs.", "Needed for explaining subspace distance/magnitude."],
    read: [LINKS.literature],
    code: ["reference_code/Subspace Toolbox", "MagTool reference code"],
    experiment: "Toy test equal subspaces and random subspaces before satellite use."
  },
  "ds-gds": {
    type: "method",
    title: "DS and GDS",
    summary: "DS compares two subspaces. GDS extends the idea to three or more subspaces.",
    tags: ["must understand", "lab core"],
    learn: ["OSCD naturally fits DS because it is pre/post.", "Multi-date sequences fit GDS/KGDS better than OSCD."],
    read: [LINKS.tpami, LINKS.literature],
    code: ["phase1/ds/pca_utils.py", "phase1/scripts/venus_kds_faithful.py if present"],
    experiment: "Do not claim GDS on OSCD unless using more than two subspaces."
  },
  "kpca-kds": {
    type: "method",
    title: "KPCA and KDS",
    summary: "Kernel PCA builds nonlinear subspaces in implicit feature space. KDS is the nonlinear DS version from TPAMI2015.",
    tags: ["future", "Sensei requested"],
    learn: ["Sensei asked that the nonlinear DS be run directly.", "For OSCD, memory and sampling choices matter."],
    read: [LINKS.tpami, LINKS.literature],
    code: ["Venus demo/reference code", "Subspace Toolbox KPCA/KCCA"],
    experiment: "First: Venus proof. Later: sampled OSCD KDS prototype."
  },
  kgds: {
    type: "method",
    title: "KGDS",
    summary: "KGDS compares three or more kernel subspaces. It is more natural for multi-date data than two-date OSCD.",
    tags: ["future", "temporal"],
    learn: ["Do not force KGDS into a two-date dataset.", "MultiSenGE or Harmonized Sentinel-2 sequences are better candidates."],
    read: [LINKS.tpami, LINKS.multisenge],
    code: ["Future after KDS/KPCA proof and sequence feasibility."],
    experiment: "Multi-date prototype only after valid sequence is built."
  },
  grassmann: {
    type: "foundation",
    title: "Grassmann / subspace geometry",
    summary: "Subspaces can be treated as points on a manifold. Geodesic distances/projections describe movement between subspaces.",
    tags: ["math", "geodesic"],
    learn: ["This is the language for temporal subspace dynamics.", "It may produce interpretable trends even without pixel labels."],
    read: [LINKS.literature, LINKS.tensorGds],
    code: ["MagTool, Jang/Aono reference tools, project geodesic modules"],
    experiment: "Sequence plot of geodesic quantity over dates."
  },
  "paper-code": {
    type: "risk",
    title: "Paper-to-code verification",
    summary: "Every niche method needs a source-to-code trail: source -> math object -> satellite adaptation -> code -> toy test -> one-city output -> claim.",
    tags: ["required", "anti-hallucination"],
    learn: ["This prevents AI-generated implementations from becoming unexamined truth.", "It is especially critical for DS/GDS/KDS/KGDS, IR-MAD, CCA/KCCA, Celik, and spatial variants."],
    read: [LINKS.methods, LINKS.literature],
    code: ["Tests and small scripts, not giant sweeps."],
    experiment: "Toy shape/dimension checks before serious claims."
  },
  "raw-cva": {
    type: "baseline",
    title: "Raw L2 / CVA",
    summary: "Simple spectral difference is a required pressure baseline. If DS cannot beat or explain it, claims must be narrower.",
    tags: ["baseline", "implemented"],
    learn: ["Raw difference often detects numeric change but not necessarily meaningful change.", "It is strong enough that weak DS claims will fail."],
    read: [LINKS.literature],
    code: ["phase1 baselines and spatial comparison scripts"],
    experiment: "Always include it in Phase 1 tables."
  },
  "pca-diff": {
    type: "baseline",
    title: "PCA-diff",
    summary: "Current strongest classical baseline in several OSCD runs. It is not novelty.",
    tags: ["baseline", "strong"],
    learn: ["If PCA-diff wins, DS contribution must be interpretability, construction analysis, temporal/geodesic behavior, or a specific failure mode."],
    read: [LINKS.celik, LINKS.experiments],
    code: ["phase1 baselines"],
    experiment: "Compare against every DS-family map."
  },
  celik: {
    type: "baseline",
    title: "Celik PCA-kmeans",
    summary: "A close classical patch-style baseline because it uses local difference-image patches.",
    tags: ["baseline", "must compare"],
    learn: ["This is the closest pressure test for patch-vector DS.", "If Celik wins clearly, patch DS cannot be claimed as a strong detector without qualification."],
    read: [LINKS.celik, LINKS.literature],
    code: ["phase1/baselines/celik_pca_kmeans.py"],
    experiment: "Add to the same core5 spatial table."
  },
  irmad: {
    type: "baseline",
    title: "IR-MAD",
    summary: "Mature CCA/MAD-based multivariate change detector. It is close enough to DS to be a required baseline.",
    tags: ["baseline", "audit required"],
    learn: ["Do not trust the local lightweight implementation until checked.", "IR-MAD should emphasize likely unchanged pixels during iterative background estimation."],
    read: [LINKS.irmad, LINKS.geeImad, LINKS.literature],
    code: ["phase1/baselines/ir_mad.py"],
    experiment: "Formula audit, toy checks, then fair OSCD comparison."
  },
  "unet-siamese": {
    type: "baseline",
    title: "U-Net and Siamese baselines",
    summary: "Deep baselines pressure any claim that subspace priors improve segmentation.",
    tags: ["implemented", "baseline"],
    learn: ["Do not try to beat deep learning as the only success criterion.", "Use geometry as interpretable prior/diagnostic or label-efficient aid."],
    read: [LINKS.oscd, LINKS.experiments],
    code: ["phase2/models/unet2d.py", "phase2/models/siamese_unet.py"],
    experiment: "Run only after Phase 1 maps are meaningful."
  },
  "modern-cd": {
    type: "baseline",
    title: "Modern CD pressure",
    summary: "Metric-CD, semantic CD, foundation models, and open-vocabulary CD show that the field has strong alternatives.",
    tags: ["literature", "risk"],
    learn: ["We may cite/discuss without reimplementing.", "This prevents weak novelty claims around priors or unsupervised maps."],
    read: [LINKS.metricCd, LINKS.changeos, LINKS.literature],
    code: ["Not immediate implementation."],
    experiment: "Protocol-aware discussion or small comparison only if time permits."
  },
  metrics: {
    type: "foundation",
    title: "Metrics and masks",
    summary: "AUROC/AP measure ranking, F1/IoU depend on thresholds, and valid masks decide which pixels exist.",
    tags: ["evaluation"],
    learn: ["Otsu can fail badly even when AUROC is decent.", "No-data/cloud masks can hide changed pixels if not audited."],
    read: [LINKS.experiments, LINKS.runbook],
    code: ["phase1/eval metrics", "spatial comparison scripts"],
    experiment: "Valid-mask audit and thresholding audit."
  },
  "green-learning": {
    type: "future",
    title: "Green Learning / PixelHop",
    summary: "Senpai-inspired path for local/multiscale feature construction before subspace comparison.",
    tags: ["future", "spatial features"],
    learn: ["Do not call our pyramid Green Learning until the exact source is verified.", "The useful idea is preserving local and multiscale structure."],
    read: [LINKS.literature],
    code: ["Future feature-construction screen."],
    experiment: "One-city feature audit before any full framework."
  },
  "multiscale-pyramid": {
    type: "future",
    title: "Multiscale subspace pyramid",
    summary: "Whole image -> 2x2 -> 4x4 -> 8x8 cells, fit subspaces per cell, aggregate DS scores.",
    tags: ["future", "spatial"],
    learn: ["This may preserve coarse and local structure.", "It may also create block artifacts, so compare against local-window DS."],
    read: [LINKS.experiments],
    code: ["Future: phase1/scripts/compare_multiscale_subspace_pyramid.py"],
    experiment: "Compare pyramid aggregation against patch/window DS."
  },
  multisenge: {
    type: "dataset",
    title: "MultiSenGE",
    summary: "Multi-date Sentinel-2 candidate for GDS/KGDS/geodesic ideas, but not current supervised benchmark.",
    tags: ["future", "multi-date"],
    learn: ["Useful because GDS/KGDS need 3+ subspaces.", "Evaluation is harder because labels differ from OSCD."],
    read: [LINKS.multisenge, LINKS.experiments],
    code: ["phase1 scripts for MultiSenGE manifest and temporal geodesic exploration"],
    experiment: "Use after Harmonized S2 feasibility or as comparison dataset."
  },
  greenhouse: {
    type: "future",
    title: "Abandoned greenhouse use case",
    summary: "Real application motivation, but not evidence until object/task labels and evaluation exist.",
    tags: ["application", "future"],
    learn: ["Could become mapping, abandonment classification, or temporal monitoring.", "Do not mix it into OSCD claims."],
    read: [LINKS.greenhouse, LINKS.greenhouseIndex, LINKS.globalPcg],
    code: ["No active implementation."],
    experiment: "Dataset/protocol definition before modeling."
  },
  xbd: {
    type: "future",
    title: "xBD / xBD-S12",
    summary: "Damage assessment is a different task: object/building-level labels and damage severity.",
    tags: ["future", "do not overclaim"],
    learn: ["OSCD binary change does not prove damage performance.", "Use only after damage pipeline and metrics exist."],
    read: [LINKS.xbd, LINKS.changeos],
    code: ["phase2/data/damage_dataset_adapter.py is only a template."],
    experiment: "Future pivot, not current evidence."
  },
  "open-vocab": {
    type: "future",
    title: "Semantic/open-vocabulary CD",
    summary: "Modern pivot direction if the research moves from generic changed pixels to object/class-specific change.",
    tags: ["future", "comparison pressure"],
    learn: ["Powerful but may be too broad for current thesis deadline.", "Useful to know for related work and pivot logic."],
    read: [LINKS.changeos, LINKS.literature],
    code: ["No active implementation."],
    experiment: "Future-only unless a tightly scoped dataset appears."
  },
  "reference-code-reminder": {
    type: "risk",
    title: "Pending reference code reminder",
    summary: "The user has more senpai reference code to add. Inventory it before any major new subspace-method implementation.",
    tags: ["reminder", "paper-to-code"],
    learn: ["Reference code is research material, not clutter.", "It can verify expected dimensions, formulas, and input assumptions."],
    read: [LINKS.literature],
    code: ["references/reference_code/"],
    experiment: "Add and inventory the remaining code before a major DS/GDS/KDS rewrite."
  }
};

const EXPERIMENTS = [
  {
    priority: 1,
    stage: "now",
    title: "Harmonized Sentinel-2 sequence feasibility report",
    why: "Directly answers Sensei's latest questions about frames, time step, and sequential subspaces.",
    output: "Area/date selection, valid-frame table, cloud/no-data summary, per-date subspace definition, readiness verdict.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-hs2-sequence-feasibility --help"
  },
  {
    priority: 2,
    stage: "now",
    title: "Inspect OSCD core5 comparison grids",
    why: "Before more code, understand where patch DS helps and where it fails.",
    output: "Failure-mode table for Beirut, Dubai, Las Vegas, Milano, Norcia.",
    command: "Get-ChildItem phase1\\outputs\\oscd_spatial_subspace_sweep_core5_20260614_004823\\runs -Recurse -Filter comparison_grid.png"
  },
  {
    priority: 3,
    stage: "next",
    title: "Patch DS score ablation",
    why: "Test whether the scalar score definition is the bottleneck.",
    output: "Core5 table for squared norm, norm, normalized ratio, residual energy, robust scaling.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-spatial-score-ablation --cities core5 --methods patch3,patch5 --rank 8"
  },
  {
    priority: 4,
    stage: "next",
    title: "Celik pressure baseline",
    why: "Closest classical local-patch pressure test for patch-vector DS.",
    output: "Same cities, masks, metrics, and visual grids as patch DS.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-celik-compare --cities core5"
  },
  {
    priority: 5,
    stage: "next",
    title: "IR-MAD formula and fair comparison audit",
    why: "IR-MAD is a strong multivariate CCA-related remote-sensing baseline.",
    output: "Toy checks, corrected implementation notes, then OSCD comparison.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-irmad-inspect --city beirut"
  },
  {
    priority: 6,
    stage: "later",
    title: "First/second DS and geodesic sequence plot",
    why: "This is the actual Sensei-aligned temporal result after sequence feasibility.",
    output: "Magnitude/projection curves over date sequence and short interpretation.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-temporal-ds-geodesic --sequence <sequence_manifest.json>"
  },
  {
    priority: 7,
    stage: "later",
    title: "Multiscale subspace pyramid",
    why: "Tests the Green Learning/wavelet-inspired spatial-preservation idea.",
    output: "1x1, 2x2, 4x4, 8x8 cell-level DS comparison against patch/window DS.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase1-multiscale-subspace-pyramid --city beirut"
  },
  {
    priority: 8,
    stage: "blocked",
    title: "Phase 2 prior rerun",
    why: "Only useful after Phase 1 spatial/temporal maps become meaningful.",
    output: "Raw U-Net vs raw + best spatial/temporal prior; no long sweep until decision gate.",
    command: ".\\.venv\\Scripts\\python.exe project_cli.py phase2-sweep --preset core --epochs 150"
  }
];

const READINGS = [
  ["P0", "Research reset and thesis framing", [LINKS.reset], "Decide whether the research is defensible and what not to claim.", "Research / 01 Read First - Thesis Core"],
  ["P0", "Sensei feedback and task order", [LINKS.feedback], "Prioritize time-sequential subspaces, first/second DS, geodesic, and spatial-information concern.", "Project notes"],
  ["P0", "DS/GDS/KDS/KGDS", [LINKS.tpami, LINKS.literature], "Explain the lab method family and avoid claiming invention.", "Research / Subspace DS KDS GDS And CCA"],
  ["P0", "OSCD and FC-Siamese", [LINKS.oscd], "Understand the current labeled benchmark and deep baseline.", "Research / OSCD And Classical Change Detection Baselines"],
  ["P0", "Harmonized Sentinel-2 sequence", [LINKS.harmonized, LINKS.experiments], "Prepare the next Sensei-first dataset artifact.", "Research / Datasets"],
  ["P1", "Celik PCA-kmeans", [LINKS.celik], "Pressure-test patch DS against a local-patch classical method.", "Research / Classical Change Detection Baselines"],
  ["P1", "IR-MAD / CCA-MAD", [LINKS.irmad, LINKS.geeImad], "Required multivariate baseline and CCA connection.", "Research / Classical Change Detection Baselines"],
  ["P1", "Metric-CD and modern unsupervised CD", [LINKS.metricCd], "Comparison pressure for unsupervised CD claims.", "Research / Review Papers And Reality Checks"],
  ["P1", "Spatial-spectral and tensor subspace methods", [LINKS.tensorGds, LINKS.literature], "Novelty boundary for spatial satellite subspace claims.", "Research / Subspace Geometry"],
  ["P2", "Greenhouse mapping", [LINKS.greenhouse, LINKS.greenhouseIndex, LINKS.globalPcg], "Application motivation only until dataset/evaluation exists.", "Research / Applications"],
  ["P2", "xBD/xBD-S12 and object-level damage", [LINKS.xbd, LINKS.changeos], "Future pivot if the task becomes damage or object-level change.", "Research / Datasets / Damage"]
];

const PIPELINE = [
  ["1. Source", "Paper, advisor note, reference code, or dataset documentation. No method should start from vibe."],
  ["2. Math object", "State the sample unit, matrix/tensor shape, subspace count, basis rank, and comparison equation."],
  ["3. Satellite adaptation", "Explain whether samples are pixels, patches, windows, date images, objects, or deep features."],
  ["4. Code path", "Name exact files/functions and include a concise source/provenance docstring for niche methods."],
  ["5. Toy check", "Equal subspaces, random subspaces, synthetic change, invalid masks, rank edge cases."],
  ["6. One-city output", "Generate maps, metrics, runtime, and inspect visual failure modes."],
  ["7. Multi-city evidence", "Only after one-city behavior makes sense."],
  ["8. Thesis claim", "State only what the evidence supports, including negative results."]
];

const CLAIMS = {
  safe: [
    "Current code implements OSCD binary changed-area experiments, not full disaster damage mapping.",
    "Patch-vector DS changed behavior and improved over global pixel DS in core5, but did not beat PCA-diff/raw L2 overall.",
    "Sensei's spatial-information concern is real and directly motivates spatial subspace comparisons.",
    "Time-sequential subspaces are now the most advisor-aligned next track."
  ],
  weak: [
    "DS improves segmentation on OSCD.",
    "Adding priors is novel by itself.",
    "Global pixel DS preserves spatial structure.",
    "Greenhouse mapping is solved by this project."
  ],
  forbidden: [
    "This project has completed xBD/xBD-S12 damage evaluation.",
    "OSCD binary change proves disaster damage performance.",
    "DS/KDS/GDS are invented by this project.",
    "The current method beats deep learning."
  ]
};

let currentRoute = ROUTES[0].id;
let currentFilter = "all";
let currentSearch = "";

function linkHtml(item) {
  if (typeof item === "string") return item;
  return `<a href="${item.url}" target="_blank" rel="noreferrer">${item.label}</a>`;
}

function tagHtml(tags) {
  return tags.map(tag => {
    const cls = tag.includes("priority") || tag.includes("read first") || tag.includes("do now")
      ? " hot"
      : tag.includes("implemented")
        ? " done"
        : tag.includes("risk") || tag.includes("forbidden") || tag.includes("do not")
          ? " warn"
          : "";
    return `<span class="tag${cls}">${tag}</span>`;
  }).join("");
}

function nodeMatches(node) {
  const filterOk = currentFilter === "all" || node.type === currentFilter || node.tags.includes(currentFilter);
  if (!filterOk) return false;
  if (!currentSearch) return true;
  const haystack = [
    node.title,
    node.summary,
    node.type,
    node.tags.join(" "),
    node.learn.join(" "),
    node.read.map(x => typeof x === "string" ? x : `${x.label} ${x.url}`).join(" "),
    node.code.join(" "),
    node.experiment
  ].join(" ").toLowerCase();
  return haystack.includes(currentSearch);
}

function renderRoutes() {
  const host = document.getElementById("routeButtons");
  host.innerHTML = ROUTES.map(route => `
    <button class="${route.id === currentRoute ? "active" : ""}" data-route="${route.id}">
      <strong>${route.title}</strong>
      <span>${route.subtitle}</span>
    </button>
  `).join("");
  host.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      currentRoute = button.dataset.route;
      renderAll();
    });
  });
}

function renderMap() {
  const route = ROUTES.find(item => item.id === currentRoute) || ROUTES[0];
  document.getElementById("routeHeader").innerHTML = `
    <h3>${route.title}</h3>
    <p>${route.summary}</p>
    <div class="route-meta">${tagHtml(route.tags)}</div>
  `;

  const nodes = route.nodes
    .map((id, index) => ({ id, index, node: NODES[id] }))
    .filter(item => item.node && nodeMatches(item.node));

  document.getElementById("nodeCanvas").innerHTML = nodes.length
    ? nodes.map(item => `
      <article class="map-node ${item.node.type}" data-node="${item.id}">
        <div class="node-index">${item.index + 1}</div>
        <div class="node-copy">
          <h4>${item.node.title}</h4>
          <p>${item.node.summary}</p>
          <div class="tag-row">${tagHtml([item.node.type, ...item.node.tags])}</div>
        </div>
        <button class="small-button" data-open="${item.id}">Open</button>
      </article>
    `).join("")
    : `<div class="detail-empty">No nodes match this search/filter in the selected route.</div>`;

  document.querySelectorAll("[data-open]").forEach(button => {
    button.addEventListener("click", () => openNode(button.dataset.open));
  });
}

function openNode(id) {
  const node = NODES[id];
  if (!node) return;
  document.getElementById("detailTitle").textContent = node.title;
  document.getElementById("detailSubtitle").textContent = node.summary;
  document.getElementById("detailBody").innerHTML = `
    <div class="tag-row">${tagHtml([node.type, ...node.tags])}</div>
    <h4>What to understand</h4>
    <ul>${node.learn.map(x => `<li>${x}</li>`).join("")}</ul>
    <h4>Read / inspect</h4>
    <ul>${node.read.map(x => `<li>${linkHtml(x)}</li>`).join("")}</ul>
    <h4>Code path</h4>
    <ul>${node.code.map(x => `<li><code>${x}</code></li>`).join("")}</ul>
    <h4>Experiment connection</h4>
    <p>${node.experiment}</p>
  `;
}

function renderExperiments() {
  const active = document.querySelector(".board-filter.active")?.dataset.stage || "all";
  const items = EXPERIMENTS.filter(item => active === "all" || item.stage === active);
  document.getElementById("experimentBoard").innerHTML = items.map(item => `
    <article class="experiment-card">
      <header>
        <span class="priority">${item.priority}</span>
        <span class="stage">${item.stage}</span>
      </header>
      <h3>${item.title}</h3>
      <p><strong>Why:</strong> ${item.why}</p>
      <p><strong>Output:</strong> ${item.output}</p>
      <pre><code>${item.command}</code></pre>
    </article>
  `).join("");
}

function renderReading() {
  document.getElementById("readingRows").innerHTML = READINGS.map(row => `
    <tr>
      <td><strong>${row[0]}</strong></td>
      <td>${row[1]}</td>
      <td>${row[2].map(linkHtml).join("<br>")}</td>
      <td>${row[3]}</td>
      <td>${row[4]}</td>
    </tr>
  `).join("");
}

function renderPipeline() {
  document.getElementById("pipelineGrid").innerHTML = PIPELINE.map(item => `
    <article class="pipeline-card">
      <h3>${item[0]}</h3>
      <p>${item[1]}</p>
    </article>
  `).join("");
}

function renderClaims() {
  document.getElementById("claimColumns").innerHTML = Object.entries(CLAIMS).map(([kind, items]) => `
    <article class="claim-card">
      <p class="eyebrow">${kind}</p>
      <h3>${kind === "safe" ? "Safe Claims" : kind === "weak" ? "Weak Claims" : "Forbidden Claims"}</h3>
      <ul>${items.map(item => `<li>${item}</li>`).join("")}</ul>
    </article>
  `).join("");
}

function renderAll() {
  renderRoutes();
  renderMap();
}

document.querySelectorAll(".nav-button").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".page-section").forEach(x => x.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.section).classList.add("active");
  });
});

document.querySelectorAll(".filter-chip").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filter-chip").forEach(x => x.classList.remove("active"));
    button.classList.add("active");
    currentFilter = button.dataset.filter;
    renderMap();
  });
});

document.querySelectorAll(".board-filter").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".board-filter").forEach(x => x.classList.remove("active"));
    button.classList.add("active");
    renderExperiments();
  });
});

document.getElementById("globalSearch").addEventListener("input", event => {
  currentSearch = event.target.value.trim().toLowerCase();
  renderMap();
});

renderAll();
renderExperiments();
renderReading();
renderPipeline();
renderClaims();
openNode("problem");
