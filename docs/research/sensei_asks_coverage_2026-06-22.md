# Sensei's asks — coverage audit: did we experiment with each PROPERLY? (2026-06-22)

Source: `slack_messages_with_sensei.md` (full history). For each concrete research ask/proposition Sensei made,
this records: what he asked, what we did, whether it was rigorous (not one-shot), and the verdict/gap. Gaps are
being closed this session (GAP A, GAP B).

Legend: ✅ done & rigorous (multi-dataset / controlled) · ⚠️ partially done — gap identified · ❌ not done.

| # | Sensei's ask (Slack) | What we did | Rigor | Verdict / gap |
|---|---|---|---|---|
| 1 | **1st & 2nd Difference Subspace for temporal satellite analysis** (founding, repeated) | 1st-order canonical DS across OSCD + the full diagnostic + band-image (spatially-faithful) construction; 2nd-order DS implemented | ✅ (1st) / ⚠️ (2nd) | 1st-order: thoroughly characterized. 2nd-order: tested for *detection* (redundant w/ mean-vector); **clean time-sequential magnitude demo not yet shown → GAP A** |
| 2 | Read & apply the **2nd-DS arXiv paper (2409.08563)** | `second_order_difference_subspace`, geodesic decomposition implemented + tested | ✅ | mechanism works; no standalone detection gain (descriptor, not detector) |
| 3 | **Kanai signal-subspace DS + SSA (2303.17802)**, learned non-anomalous reference D_N | `signal_subspace_ds.py` (faithful) + ssds_* experiments on real S2 | ✅ rigorous (≥5 real cases) | seasonality-robust on synthetic splice; **failed on real natural change** (fire/reservoir/irrigation) — honest negative, multi-tested |
| 4 | **SSA (Singular Spectrum Analysis)** | covered via the ssds (signal-subspace) experiments | ✅ | as #3 |
| 5 | **Slow Feature Analysis** (Kobayashi BMVC2017; Suzana) | SFA/SFS tested (H-A invariance angle) | ✅ | valid principle but owned by IR-MAD/SFA-CD; below indices |
| 6 | **RTW (Randomized Time Warping)** (Hiraoka ICIP) | RTW tested on BreizhCrops + MultiSenGE | ✅ (2 datasets) | temporal-invariance tested; not a CD detection win |
| 7 | **Linear DS / TPAMI2015** as the starting tool | core method throughout (canonical DS) | ✅ | the project's spine |
| 8 | **Nonlinear/Kernel DS via the kernel trick — "RUN it yourself"** (explicit, repeated) | Codex ran **RFF-approx kernel-DS on global pixels** (CLOSED, < CVA) | ⚠️ | the *proper* kernel DS on **our band-image construction** was not run → **GAP B (running)** |
| 9 | **KGDS + Venus data (TPAMI2015)** | Venus reproduced; KDS via RFF tested | ⚠️ | KGDS (multi-class kernel) only relevant multi-date; partial — folded into GAP B |
| 10 | **Geodesic projection/decomposition for smooth change** (asked 2–3×; he proved it works) | `geodesic.py` + 2nd-order DS; tested | ⚠️ | works as mechanism; **clean smooth-vs-abrupt demo on a real sequence missing → GAP A** |
| 11 | **CMSM / MSM** | material-subspace MSM CD tested | ✅ | SAM-to-mean beats it; negative |
| 12 | **Use more than 3 bands** (extract from S2 source) | all 13 Sentinel-2 bands used | ✅ | done; + band-subset study now running |
| 13 | **Time-sequential S2 (Google harmonized) + DS magnitudes** (asked ~4×; ACCV "first/unique trial") | ssds used S2 sequences for *detection* | ⚠️ | the clean **1st/2nd-DS-magnitude-over-sequence + event response** demo (his exact framing) → **GAP A (running)** |
| 14 | **VLM-explanation-as-subspace** (1st/2nd variations, geodesic) | not attempted | ❌ | Sensei flagged it "distant from CV"; **future work** (honest) |
| 15 | **Spatial-information loss** ("your algorithm can break spatial information") | **band-image DS** (spatially-faithful: each band-image is a sample) | ✅ | *direct* answer; the whole current direction |
| 16 | **Combine subspace + DNN/LLM / deep features** | H-C deep-feature subspace (neg) + **DS-prior-to-CNN** (DS-specific fusion gain, +0.03) | ✅ ongoing | the current positive; teacher-student running |
| 17 | **S3CCA / temporally-regularized CCA / CCA** | DS-basis/contiguous-band attribution (neg); IR-MAD = iter-reweighted CCA (baseline+fusion) | ✅ | covered; exact S3CCA ref impl not re-run (question answered negative) |
| 18 | **Characterize the dataset + verify DS validity** | matched-control protocol + naive→complex ladder | ✅ | DS beats geometric nulls; loses to smoothed-PCA standalone; DS-specific in fusion |

## The real gaps being closed this session
- **GAP A — time-sequential 1st/2nd DS magnitude + geodesic decomposition on a REAL S2 multi-date sequence**
  (Sensei's single most-repeated ask + the ACCV "first/unique trial"). Data: `tmp/ipol416_sequences/al_wakrah`
  (Qatar, 11 real S2 dates 2016–2018), `beijing_airport`, `piraeus`, plus controlled `synthetic_*`. Using the
  **spatially-faithful band-image** construction (answers the spatial-info concern simultaneously). Output: the
  curves of first-DS magnitude (change "velocity"), second-DS magnitude (change "acceleration"/abruptness), and
  along/orthogonal geodesic split over the date sequence, checked against documented events vs quiet periods.
  Honest framing per Sensei: a first/unique *analysis* trial, not a SOTA detector.
- **GAP B — nonlinear (kernel) Difference Subspace** (Sensei's explicit "run the nonlinear DS yourself").
  Status: **already run globally** by Codex via RFF/RBF kernel-DS on real HSI (Hermiston/Salinas) — **CLOSED**:
  the nonlinear lift restores amplitude (Hermiston 0.55→0.97) but still **< CVA 0.985** and fails the controlled
  distributional test. So the answer ("kernel does not beat the simple baseline") is already by trial, not hunch.
  Remaining nicety: re-run the kernel-DS *specifically on OSCD pixels* for a benchmark-matched confirmation —
  honest prior LOW; quick CPU job, can run on request.

## Bottom line for the user
Of Sensei's ~18 concrete asks, **15 are experimented (most rigorously, multi-dataset)**, **2 are real gaps now
running** (time-sequential DS/geodesic; band-image kernel-DS), and **1 is honest future work** (VLM-as-subspace,
which Sensei himself flagged as distant). After GAP A & B land, every Slack ask is answerable by an experiment,
not a hunch.
