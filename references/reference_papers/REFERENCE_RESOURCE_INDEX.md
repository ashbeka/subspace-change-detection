# Reference Paper And Resource Index

Purpose: keep the research-reading resources useful without turning this repository into a paper dump. Chrome/Zotero should remain the main paper-management system. This file records what the local/reference resources are for, how they connect to the current research problem, and what should be imported or read first.

Current working problem:

```text
Can spatially aware Difference Subspace construction preserve the spatial structure of multispectral Sentinel-2 images well enough to produce interpretable changed-area evidence, and where does it help or fail compared with raw spectral difference, PCA-diff, Celik/IR-MAD, and neural change-detection baselines?
```

## 1. Reading Priority

| Priority | Theme | Why it matters now | Main action |
|---|---|---|---|
| 1 | DS/GDS/KDS/KGDS and subspace geometry | This is the lab-aligned method family and must be implemented with correct dimensions and edge cases. | Read the DS/GDS/KDS sources before changing formulas. Cross-check with reference code. |
| 2 | Classical remote-sensing change detection | Raw L2/CVA, PCA-diff/Celik, MAD/IR-MAD, CCA/KCCA are required baselines and comparison pressure. | Verify implementations before using them in claims. |
| 3 | Spatial and spectral-spatial change detection | Sensei's strongest criticism is spatial information loss in global pixel subspaces. | Read spatial-spectral subspace, patch/window, and semantic/spatial-context CD resources. |
| 4 | Deep learning and label-efficient CD | Neural methods are the current performance pressure; label scarcity is a real field problem. | Use as comparison/framing, not as an excuse to run broad sweeps before method verification. |
| 5 | Multi-date temporal change | GDS/KGDS are more natural with three or more date subspaces. | Evaluate MultiSenGE/Harmonized Sentinel-2 after OSCD spatial DS is understood. |
| 6 | Applications and pivots | Greenhouse, xBD/xBD-S12, semantic/object-level CD are possible thesis pivots. | Keep as candidate routes until labels/evaluation are concrete. |

## 2. Local Paper Resources

| Local file | Status | Project role | Use carefully because |
|---|---|---|---|
| `references/reference_papers/MVA_2025_human_motion_analysis.pdf` | Preserved tracked PDF. | Shape/human-motion subspace reference. Useful analogy for subspace contribution analysis: which landmarks/parts contribute to a DS magnitude. | It is not remote-sensing evidence and does not prove satellite spatial DS is valid. Use it as a method-analogy lead only. |

## 3. Bookmark Resource Pointers

The organized bookmark export is the main active resource map:

- Importable organized Chrome file: `docs/source_records/final_organization_2026-06-12/chrome_bookmarks_organized_all_2026-06-20.html`
- Bookmark QA summary: `docs/source_records/final_organization_2026-06-12/bookmark_organization_summary_2026-06-20.json`
- Active bookmark notes: `notes/reference_bookmarks.md`
- Concept-to-reading table: `notes/literature.md`

The 2026-06-20 merge added `37` new URLs to the prior `1742` entries. The final
organized output preserves `1779` bookmark entries. All `58` URLs in the
essential knowledge-base source are represented in the final import file.

## 4. Concept-To-Resource Map

| Concept | Use in the project | Resource location |
|---|---|---|
| Paper-faithful DS/GDS/KDS/KGDS | Formula source for subspace comparisons, nonlinear KDS/KGDS, and future multi-date work. | `Research / 01 Read First - Thesis Core / 01 Subspace DS KDS GDS And CCA`; `notes/literature.md`; `references/REFERENCE_CODE_INVENTORY.md` |
| Spatially aware DS | Immediate experiment route: global pixel DS versus patch-vector DS versus local-window DS versus multiscale pyramid. | `notes/experiments.md`; `notes/methods.md`; spatial CD bookmarks under `Research / 01 Read First - Thesis Core / 05 Spatial And Semantic Change Questions` |
| Classical baselines | Required fair comparisons before any DS contribution claim. | `Research / 01 Read First - Thesis Core / 02 OSCD And Classical Change Detection Baselines`; `phase1/baselines/`; `notes/literature.md` |
| IR-MAD / CCA / KCCA | Strong classical multivariate baseline and possible bridge to subspace/CCA family. | Subspace toolbox code, CCA bookmarks, `notes/methods.md`, `notes/experiments.md` |
| KPCA/KDS | Nonlinear variant requested by Sensei; not current OSCD default. | Fukui/Maki TPAMI resources, Venus demos, SubspaceMethodsToolBox, `notes/methods.md` |
| Green Learning / wavelet / multiscale pyramid | Possible spatial-preservation extension inspired by Senpai notes. | `notes/experiments.md` multiscale subspace pyramid, Green Learning bookmarks |
| Semantic/open-vocabulary CD | Pivot/comparison pressure if binary OSCD is too narrow. | `Research / 03 Methods - Change Detection / 02 Semantic And Damage Change Detection` |
| Greenhouse mapping | Possible application route; not current evidence. | `Research / 05 Applications - Use Cases And Problem Framing / 01 Greenhouses Agriculture Environmental Monitoring` |
| xBD/xBD-S12 damage | Future extension only; do not use OSCD evidence as damage proof. | `Research / 04 Datasets - Current Candidate Future / 03 Disaster Damage And Semantic Datasets` |
| Seasonal irrigation-regime change | Candidate real-observation-set experiment: dates within one field/year form a subspace; annual subspaces are compared by first/second DS. | IrrMapper paper/catalog, temporal SITS resources, `notes/methods.md`, and `notes/experiments.md` |
| Hyperspectral change and unmixing | Separate pivot testing spectral/spatial geometry with hundreds of bands and established HSI baselines. | `Research / 02 Methods - Subspace Geometry / 04 Hyperspectral Geometry Unmixing And Spectral Methods` and `Research / 04 Datasets - Current Candidate Future / 05 Hyperspectral Change And Analysis Datasets` |

## 5. Zotero Queue

Import into Zotero first:

1. DS/GDS/KDS/KGDS core papers.
2. Remote-sensing change-detection surveys.
3. OSCD and FC-Siamese/Daudt papers.
4. CVA/PCA-diff/Celik and MAD/IR-MAD/CCA/KPCA papers.
5. Spatial-spectral subspace and semantic/spatial-context CD papers.
6. Label-efficient and prior-guided CD papers.
7. Greenhouse and xBD/xBD-S12 papers only if those routes become active.

Do not import every bookmarked tutorial or tool into Zotero. Keep general learning resources in Chrome bookmarks unless they become cited sources.

## 6. Rules For Future Paper Imports

- If a paper is used to implement code, record the source-to-code trail in `notes/methods.md` and the method file docstring.
- If a paper only motivates a future idea, put it in `notes/literature.md` and leave it in Chrome/Zotero.
- If a paper is physically copied into `references/reference_papers/`, explain why it must live in the repo instead of Zotero.
- If a paper has supplementary code, connect the paper and code in `references/REFERENCE_CODE_INVENTORY.md`.
