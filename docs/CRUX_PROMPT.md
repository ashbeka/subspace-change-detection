# RESEARCH-MINING PROMPT (agent-agnostic) — paste into any agent/session

> One self-contained, extensive task: an exhaustive, rigorous research-mining program that converges on a
> specific, defensible, NOVEL research task. Usable by ANY agent in ANY session/worktree. Nothing in this
> project is restricted to any one agent — every path below is readable via absolute path from any session.
> (Hands-on experiments are being run separately; this prompt is research-mining only.)

---

You are a co-researcher on a satellite change-detection research project (Fukui subspace-methods lab,
U-Tsukuba; deep motive = post-disaster/post-conflict reconstruction, esp. damage progression & recovery).
Your task is the RESEARCH-MINING program below: study the literature + the researcher's notes, map where
subspace/geometry methods genuinely fit hyperspectral satellite change detection, and converge on a specific,
defensible, NOVEL task to pursue. Work in full force, maximum rigor.

## 0. ORIENT — read everything first (nothing in this project is off-limits to you)
All project knowledge is readable via ABSOLUTE PATH from any session/worktree; read all of it before
reasoning. None of it is restricted to any one agent or model.
- **Established research state & findings** (the research worktree) — read ALL of:
  - `E:/research_projects/sccd-claude/docs/CONTINUITY.md`  (master state — read FIRST)
  - `E:/research_projects/sccd-claude/docs/RESEARCH_DIRECTIONS_TOP10.md`
  - `E:/research_projects/sccd-claude/docs/CD_TAXONOMY.md`
  - `E:/research_projects/sccd-claude/docs/METHOD.md`
  - `E:/research_projects/sccd-claude/docs/experiment_reports/*`  (every report)
  - and any other files under `E:/research_projects/sccd-claude/docs/`.
- **The researcher's RAW NOTES + provided papers** (the main repo; always the freshest):
  - `E:/research_projects/subspace-change-detection/docs/source_records/final_organization_2026-06-12/`
    (apple_notes.md, slack_messages_with_sensei.md, slack_notes_to_myself.md, most_essential_knowledge_base.md,
    long_prompt.md, the bookmark HTMLs, and the provided paper PDFs incl.
    "Geometry of subspace set and its application to machine learning 2024.pdf")
  - `E:/research_projects/subspace-change-detection/notes/`
    (my_notes.md, methods.md, experiments.md, feedback.md, literature.md, research_paper_plan.md)
- If you work in your own git worktree, these absolute paths remain readable — use them. Write your OWN
  deliverables into `docs/research/` and `docs/kb/` in your working copy; do not overwrite another agent's
  files (namespace by a subfolder if multiple agents share a tree).

## 1. ONE-LINE GOAL
Through deep literature + notes mining, converge on a genuinely **NOVEL, publishable** task using
subspace/geometry methods for hyperspectral satellite change detection (Sensei's bar: first/unique trial,
need not be top performance). Subspace geometry is a *representation+comparison MECHANISM* (not a standalone
family) — but it can stand alone if compared against the right traditional methods.

## 2. HARD-WON FINDINGS TO RESPECT (do not re-litigate; do not overclaim)
- First-order DS ≡ spectral angle (SAM) for low-dim spectra → bi-temporal DS is forced.
- On real Sentinel-2 + Salinas, plain subspace detectors LOSE to simple scalars (SAM/CVA) and to IR-MAD;
  invariance-detection is owned by IR-MAD (affine/nonlinear/spatial) + SFA.
- Subspace change-TRAJECTORY dynamics (1st/2nd-order DS velocity & curvature) is REDUNDANT with the
  mean-spectrum vector 2nd-difference; 2nd-order DS never fired; not nuisance-invariant. (Trajectory-dynamics
  hypothesis largely closed.)
- The one residual subspace edge: detecting MEAN-PRESERVING distributional change (the local spectral
  distribution shifts while the per-pixel/patch mean holds) — but modest, and ordinary covariance statistics
  also detect it; not scale-invariant.
- THE PATTERN: a simple/standard baseline keeps matching the subspace; the only real levers are INVARIANCE or
  capturing STRUCTURE the mean discards. Your job is to find where such a lever is genuinely OPEN in the
  literature — or to establish, with citations, that it is not.

## 3. METHODOLOGY (how to work)
Hypothesis-driven loop: surface possibilities → interrogate them against the literature + the established
findings → keep only what survives scrutiny → re-ground when it fails. Always name the trivial/standard null
beside any proposed method (SAM/CVA/IR-MAD/PCA-Kmeans/covariance baseline); never assert novelty without
checking the closest prior work; be blunt and honest — do not defend a failing direction. Commit deliverables
frequently. When you hit a wall, find a smart workaround rather than stopping.

## 4. THE RESEARCH-MINING PROGRAM (execute in order; commit each deliverable)
1. **Deep-study the provided ~20 papers** (I'll provide you the papers in the prompt) PLUS the core subspace papers
   (DS/GDS/KDS/KGDS TPAMI-2015; 2nd-order DS arXiv:2409.08563; SSA-anomaly-DS arXiv:2303.17802; SLS; SFA/SFS;
   S3CCA/TRCCA; and "Geometry of subspace set and its application to machine learning" 2024). Build a
   structured **KNOWLEDGE BASE** (`docs/kb/`): per paper → method + math, the signal/subspace idea, the
   novelty, HOW THEY JUSTIFY IT, the BASELINES they compare to, and the concrete transfer to hyperspectral ×
   temporal × change.
2. **Mine the researcher's notes** → extract every relevant idea, hard constraint, advisor/senpai steer
   (esp. Sensei), and candidate task already floated; cross-link to the KB.
3. **Scope to HYPERSPECTRAL satellite × TEMPORAL × CHANGE.** Read the LATEST review/survey papers in that exact
   niche (+ adjacent multitemporal HSI-CD and distributional/covariance change detection).
4. **Catalogue how geometry/subspaces are actually USED in CD:** (a) as a component inside a pipeline, and
   (b) as a STANDALONE method compared to traditional methods (PWTT, SiROC, CCDC, SFA, CVAPS, PCA-Kmeans,
   IR-MAD, … see github.com/wenhwu/awesome-remote-sensing-change-detection#traditional-methods). Map where the
   subspace family (DS, 1st/2nd-order, geodesic, GDS, KDS/KGDS, SSA, SFA, SLS, SFS, Grassmann, RTW — **and a
   potential OWN method linking subspaces ⇄ hyperspectral imaging**) could genuinely compete or shine.
5. **Extract the reviews' open CHALLENGES; RANK** the ones that best fit our use cases + tools (most-fit first).
6. **Per candidate direction: find the CLOSEST existing methods/papers** and analyze how they JUSTIFY the work,
   HIGHLIGHT NOVELTY, and choose fair COMPARISON baselines (focus on novelty framing + what they compare
   against, so we can mirror it later).
7. **UPDATE/EXTEND the taxonomy** (`docs/CD_TAXONOMY.md`): place every method, the invariance ladder, where
   geometry sits as a mechanism, and the catalogued challenges. The taxonomy is the scaffold for all the above.
8. **SYNTHESIZE (the hardest step):** connect KB + notes + niche + reviews/challenges + closest-methods into a
   ranked set of **specific, specific, specific candidate TASKS**. For the top one, give: the precise problem
   statement, *why it is novel* (vs the closest methods), the method (math), the FAIR comparison baselines
   (traditional + the right subspace variant + the mean/covariance nulls we already know fail or don't), the
   experiment plan, the metrics, and the falsifier.

## 5. DELIVERABLES (commit to `docs/`)
`docs/kb/` (per-paper notes), `docs/CD_TAXONOMY.md` (updated), `docs/research/challenges_ranked.md`,
`docs/research/closest_methods_novelty.md`, and `docs/research/synthesis_specific_tasks.md` (the ranked novel
tasks + the top one's full plan). End every deliverable with the honest novelty verdict and the next
falsifiable step.

## 6. REASONING DIRECTIVE
Absolute maximum reasoning, no shortcuts. Comprehensively decompose each sub-problem; stress-test your logic
against all paths, edge cases, and adversarial scenarios; write out your deliberation, alternatives considered,
and rejected hypotheses, so no assumption is left unchecked.
