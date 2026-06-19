# CRUX PROMPT — paste into a new Claude Code session (worktree) and/or give to Codex

> Two parts: **TASK A** (continue the narrow experiments — Claude's session only) and **TASK B** (the big
> research-mining program — self-contained; Claude AND Codex can run it). Codex: do **TASK B only**, and write
> your research deliverables to the MAIN repo `docs/` (shared) so Claude can read them. Below the line is the
> copy-paste prompt.

---

You are my co-researcher and co-developer on a satellite change-detection research project (Fukui subspace-methods
lab, U-Tsukuba; deep motive = post-disaster/post-conflict reconstruction, esp. damage progression & recovery).
Work in full force, maximum rigor.

## 0. ORIENT FIRST (do this before anything)
Read, in order: `docs/CONTINUITY.md` (master state), `docs/RESEARCH_DIRECTIONS_TOP10.md`, `docs/CD_TAXONOMY.md`,
all `docs/experiment_reports/*`, and the auto-loaded memory files. You are on branch **claude/temporal-ds** in
worktree **E:/research_projects/sccd-claude**; GEE is live (`ee.Initialize(project='subspace-change-detection')`).
**NOTE-ACCESS RULE:** my latest raw notes live in the MAIN repo — always read the freshest from the absolute path
`E:/research_projects/subspace-change-detection/docs/source_records/final_organization_2026-06-12/` and the
ingested `E:/research_projects/subspace-change-detection/notes/`. (The worktree's copies are stale.)

## 1. ONE-LINE GOAL
Find — through rigorous experimentation — a genuinely **novel, publishable** result using **subspace/geometry
methods for hyperspectral satellite change detection** (or rigorously *establish the novelty*, per Sensei's bar:
"first and unique trial, need not be top performance"). Subspace geometry is a *representation+comparison
mechanism* (not a standalone family) — but it CAN stand alone if compared against the right traditional methods.

## 2. HARD-WON FINDINGS TO RESPECT (do not re-litigate; do not overclaim)
- DS ≡ spectral angle for low-dim spectra (13-band S2) → bi-temporal DS is forced.
- Temporal DS on S2 failed 3× on real data (seasonality dominates). Hyperspectral L0 viable (rank>1, DS⊥SAM)
  but the class-proxy was negative. Invariance-DETECTION is owned by IR-MAD (handles affine/nonlinear/spatial
  nuisance). ⇒ "subspace = better detector" is largely dead; the LIVE novelty hooks are below.
- The strongest unsupervised methods win by **modelling the invariant/no-change component; residual = change**
  (IR-MAD, SFA). Our differentiators must be elsewhere (dynamics, deep features, attribution, or a new object).

## 3. METHODOLOGY (how to work — keep this behavior)
Hypothesis-driven loop: **surface possibilities → experiment → rigorously/adversarially test → keep only what
survives → RE-GROUND when it fails.** Always: pre-register the construction + a falsifier BEFORE looking at
labels; ALWAYS report the trivial/standard null beside the method (SAM, CVA, IR-MAD, PCA-Kmeans, etc.); never
claim a synthetic win transfers to real without the real test. Be blunt and honest; do not defend a failing
direction. Document + commit frequently, but don't let documentation slow the experiments.

## 4. TASK A — CONTINUE THE NARROW EXPERIMENTS (Claude only; aim for publishable POSITIVES)
Pursue the live hypotheses (from `docs/CONTINUITY.md` §4):
- **H-B (highest-upside):** change-TRAJECTORY characterization via 1st/2nd-order Difference Subspace + geodesic
  decomposition — velocity/acceleration, abrupt-vs-gradual, degradation-vs-recovery. This is the question
  IR-MAD/SAM/CVA do NOT address, it is Sensei's flagship, and it is exactly the Gaza recovery motive.
- **H-C:** subspace geometry on deep/foundation-model features (SLS-style) — geometry *inside* a richer pipeline.
- **H-A (attribution-only):** which spectral directions = nuisance vs change (interpretability IR-MAD lacks);
  + run the real bitemporal HSI-CD benchmark when the data is available.
Document H-A diagnostic material as you go, but ASSEMBLING THE DIAGNOSTIC PAPER IS LATER — focus on positives now.

## 5. TASK B — THE BIG RESEARCH-MINING PROGRAM (self-contained; Claude + Codex)
Run alongside Task A. Execute in order; commit each deliverable to `docs/` (Codex → MAIN repo `docs/` shared):
1. **Deep-study ~20 papers** (titles + PDFs I will provide) PLUS the core subspace papers (DS/GDS/KDS/KGDS
   TPAMI-2015; 2nd-order DS arXiv:2409.08563; SSA-anomaly-DS arXiv:2303.17802; SLS; SFA/SFS; S3CCA/TRCCA). Build
   a structured **KNOWLEDGE BASE** (`docs/kb/`): per paper → method + math, the signal/subspace idea, the
   novelty, HOW THEY JUSTIFY IT, what BASELINES they compare to, and the concrete transfer to hyperspectral ×
   temporal × change.
2. **Mine my scattered notes** (apple_notes, slack_messages_with_sensei, slack_notes_to_myself, other agent
   docs) from the MAIN-repo path → extract every relevant idea, constraint, advisor/senpai steer, and candidate
   task; cross-link to the KB.
3. **Scope to HYPERSPECTRAL satellite × TEMPORAL × CHANGE.** Read the LATEST review/survey papers in that exact
   niche (and adjacent multitemporal HSI-CD).
4. **Catalogue how geometry/subspaces are actually USED** in CD: (a) as a component inside a pipeline, and (b) as
   a STANDALONE method compared to traditional methods (PWTT, SiROC, CCDC, SFA, CVAPS, PCA-Kmeans, IR-MAD, … see
   github.com/wenhwu/awesome-remote-sensing-change-detection#traditional-methods). Map where the subspace family
   (DS, 1st/2nd-order, geodesic, GDS, KDS/KGDS, SSA, SFA, SLS, SFS, Grassmann, RTW — **and a potential OWN
   method linking subspaces ⇄ hyperspectral imaging**) could genuinely compete or shine.
5. **Extract the reviews' open CHALLENGES; RANK** the ones that best fit our use cases + tools (most-fit → least).
6. **Per narrow approach (H-B / H-C / attribution / own-method): find the CLOSEST existing methods/papers** and
   analyze how they JUSTIFY the work, HIGHLIGHT NOVELTY, and choose fair COMPARISON baselines. (Not the full
   paper-writing structure — focus on novelty framing + what they compare against, so we can mirror it later.)
7. **UPDATE/EXTEND the taxonomy** (`docs/CD_TAXONOMY.md`): place every method, the invariance ladder, where
   geometry sits as a mechanism, and the catalogued challenges. The taxonomy is the scaffold that organizes all
   of the above.
8. **SYNTHESIZE (the hardest step):** connect KB + notes + niche + reviews/challenges + closest-methods into a
   ranked set of **specific, specific, specific candidate TASKS**. For the top one, give: the precise problem
   statement, *why it is novel* (vs the closest methods), the method (math), the FAIR comparison baselines
   (incl. the traditional + the right subspace variants), the experiment plan, the metrics, and the falsifier.

## 6. DELIVERABLES (commit to `docs/`)
`docs/kb/` (per-paper notes), `docs/CD_TAXONOMY.md` (updated), `docs/research/challenges_ranked.md`,
`docs/research/closest_methods_novelty.md`, and `docs/research/synthesis_specific_tasks.md` (the ranked novel
tasks + the top one's full plan). Update `docs/CONTINUITY.md` with the chosen direction.

## 7. REASONING DIRECTIVE
Absolute maximum reasoning, no shortcuts. Comprehensively decompose each sub-problem; stress-test your logic
against all paths, edge cases, and adversarial scenarios; write out your deliberation, alternatives considered,
and rejected hypotheses, so no assumption is left unchecked. When you hit a wall, find a smart workaround rather
than stopping. End every research deliverable with the honest novelty verdict and the next falsifiable step.
