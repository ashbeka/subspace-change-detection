# #1 Spectral-signal DS — L0 viability (PASS) + first real-data proxy (NEGATIVE)

Date 2026-06-19. Branch claude/temporal-ds. Code: temporal/hsi_spectral_ds.py,
temporal/experiments/hsi_dimensionality.py. Data: Salinas 204-band AVIRIS (ehu.eus; gitignored).

## Data access (BLOCKED for the real CD benchmarks)
The labeled hyperspectral CD benchmarks (Bay Area/Santa Barbara AVIRIS, Hermiston/River/Wetland Hyperion)
could NOT be pulled headlessly: CiTIUS GitLab clone fails (invalid index-pack, likely Git-LFS), its archive
zip downloads corrupt/truncated (1.1GB, not a valid zip), the SicongLiuRS GitHub repo name 404s, and GEE
EO-1 Hyperion has zero coverage at the benchmark sites (tasked instrument). USER ACTION NEEDED to run the
decisive labeled test (see below). Salinas (classification, single image) was used as a real-hyperspectral
stand-in for L0 + a class-change proxy.

## L0 viability — PASS [experiment-evidence]
On real 204-band spectra: spectral-SSA intrinsic rank (energy95) median 2-3 (range up to 23) => the
per-pixel spectral subspace is genuinely MULTI-DIMENSIONAL, NOT rank-1. DS vs SAM Spearman = -0.03 (vs 0.99
for 13-band S2) => DS carries information genuinely different from the spectral angle. The construction does
not degenerate. This is the strongest positive signal the subspace direction has produced.

## First real labeled test (class-change proxy) — NEGATIVE [experiment-evidence]
Salinas same-class (no-change) vs different-class (change) pairs; AUC by band count:
| bands | DS | SAM | CVA | DS-SAM |
|---|---|---|---|---|
| 13 | 0.79 | 0.93 | 0.96 | -0.14 |
| 50 | 0.76 | 0.92 | 0.96 | -0.16 |
| 100 | 0.68 | 0.92 | 0.95 | -0.24 |
| 204 | 0.70 | 0.92 | 0.96 | -0.23 |
The dimensionality-threshold hypothesis is FALSIFIED on this proxy: DS-SAM gets MORE negative with bands
(predicted: more positive). DS loses to both SAM and CVA at every band count.

## Honest standing + the one untested regime
The "different but less discriminative" result is consistent with the project-wide pattern (subspace methods
lose to simple scalars on real data). BUT the proxy is class-discrimination on ONE clean image with NO
date-to-date illumination/atmospheric nuisance -- the ONLY regime where DS's amplitude-invariance (a subspace
is scale-invariant) could beat amplitude-sensitive CVA/SAM. That regime = real bitemporal hyperspectral CD,
which is exactly the data that is blocked. DECISIVE next test: run temporal/experiments/hsi_dimensionality.py
on a real labeled CD .mat (Bay Area/Hermiston) once downloaded. If DS still loses there -> #1 is dead, #10
(diagnostic) stands. The prior is now negative; do not invest further before the real CD benchmark runs.
