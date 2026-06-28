# Reference Bookmarks

## Table Of Contents

- [1. Current Source And Import File](#1-current-source-and-import-file)
- [2. Research Priority Order](#2-research-priority-order)
- [3. Categorization Policy](#3-categorization-policy)
- [4. Bookmark Tree](#4-bookmark-tree)
- [5. Research Reading And Zotero Queue](#5-research-reading-and-zotero-queue)
- [6. Review Buckets](#6-review-buckets)
- [7. How To Use This](#7-how-to-use-this)
- [8. 2026-06-25 Refinement Plan](#8-2026-06-25-refinement-plan)
- [9. Applied Deduplication And Resource-Type Policy](#9-applied-deduplication-and-resource-type-policy)

## 1. Current Source And Import File

Current organized Chrome import file:

`docs/source_records/bookmarks/chrome_bookmarks_organized_all_2026-06-25.html`

Refined trial import file with canonical-paper/resource separation,
research deduplication, and flattened one-item folders:

`docs/source_records/bookmarks/chrome_bookmarks_refined_taxonomy_dedup_2026-06-25.html`

Latest import file for review, with the emoji importance labels, numbered
Research folders removed, additional duplicate cleanup, and stronger resource
type separation:

`docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html`

Use the latest labeled cleanup file for the next Chrome import test. The older
organized and refined files were preserved for comparison and rollback.

Latest source merge:

- Base organized file: `chrome_bookmarks_organized_all_2026-06-20.html`
- New source parsed: `newest_uningested_2026-06-25.html`
- Manual source parsed: `manual_links_ingest_2026-06-25.txt`
- Chrome export cross-check: `bookmarks_6_25_26.html`
- Merge summary: `bookmark_organization_summary_2026-06-25.json`

| check | count |
|---|---:|
| Base organized bookmark entries | 1779 |
| 2026-06-25 source bookmarks | 149 |
| 2026-06-25 source unique URLs | 149 |
| Manual 2026-06-25 raw URL tokens | 76 |
| Manual unique URLs after duplicate normalization | 71 |
| Manual new unique URLs added | 62 |
| Final organized entries after duplicate cleanup and taxonomy rebuild | 1972 |
| Final normalized unique URLs | 1972 |
| Remaining duplicate normalized URLs | 0 |
| Chrome export normalized URLs missing from organized file | 0 |
| Research entries after taxonomy rebuild | 836 |
| Date/source folder entries remaining | 0 |

The 2026-06-25 organized file covers the checked Chrome export plus the manual pasted batch. Extra URLs relative to the Chrome export are expected because the manual batch was ingested after that export.

## 2. Research Priority Order

The latest labeled cleanup keeps a topical/context tree instead of a separate
`Read First` folder. Bookmark titles use emoji-only importance labels:

`🔥 = Core/read first; ⭐ = High priority; 📌 = Useful support; ❓ = Triage/unclear.`

- `🔥`: directly supports the current thesis spine or pressure baselines:
  DS/GDS/KDS/KGDS, spatial subspace construction, Successive Saab/Green
  Learning/PixelHop, OSCD/FC-Siamese, Celik/PCA/CVA/IR-MAD/SAM, Metric-CD,
  xBD-S12 as bounded external evidence, HLS/MultiSenGE, and review/reality-check
  papers needed to avoid overclaiming.
- `⭐`: strong comparison pressure or plausible next research route:
  temporal subspaces, SFA/SSA/RTW, HSI/spatial-spectral/tensor methods,
  semantic/foundation/open-vocabulary change detection, disaster/building
  change, registration/radiometry/fusion, and important candidate datasets or
  code.
- `📌`: supporting background, tools, workflows, application context, or
  papers that may help later but are not on the current critical path.
- `❓`: search queries, bare/weak PDFs, unresolved links, and off-spine
  material kept intentionally without pretending it is important.

Within a folder, read `🔥` first, then `⭐`, then `📌`. Do not interpret `❓` as
bad; it means "kept, but unresolved or low-confidence."

## 3. Categorization Policy

Bookmarks are organized by concept first, then function. `Research` is intentionally broad but not an everything-bucket: it keeps thesis-relevant papers, methods, datasets, remote-sensing resources, reference code, and application/problem-framing leads. Generic ML courses, books, beginner tutorials, generic LLM resources, and generic programming tools are routed to `Learning` or `Programming and Tools` unless they directly support remote sensing, change detection, or subspace-method work.

Shared-category bookmarks get one primary folder to avoid artificial duplicate clutter. Primary placement priority is: thesis research relevance, active project/work relevance, then school/career/language/life/personal relevance. Topic rules use bookmark title and URL; source folder is provenance only.

## 4. Bookmark Tree

| root folder | bookmark entries |
|---|---:|
| Career and Opportunities | 105 |
| Language Study | 61 |
| Learning | 245 |
| Life Admin | 69 |
| Personal Notes and Culture | 19 |
| Programming and Tools | 123 |
| Projects | 280 |
| Research | 740 |
| To Review | 169 |
| University | 58 |

Latest labeled `Research` top-level folders:

| tree path | entries |
|---|---:|
| Research/Papers - Canonical Literature | 371 |
| Research/Data - Datasets, Benchmarks, Evaluation | 82 |
| Research/Applications - Problem Framing | 73 |
| Research/Parking - Search Leads And Unclear | 66 |
| Research/Code - Repositories, Toolboxes, Implementations | 64 |
| Research/Workflow - Search, Venues, Writing, Reference | 62 |
| Research/Background - Learning Tutorials | 22 |

Latest labeled `Research` importance counts:

| label | entries |
|---|---:|
| 🔥 | 54 |
| ⭐ | 198 |
| 📌 | 414 |
| ❓ | 74 |

The detailed table below describes the older numbered organized tree and is
kept as historical provenance, not as the current import target.


Research and learning details:

| tree path | entries |
|---|---:|
| Learning/Computer Science/Foundations and Curricula | 15 |
| Learning/General Learning/Courses Books Tutorials | 28 |
| Learning/General Learning/Reading and Note Taking | 1 |
| Learning/Machine Learning and AI/Books and References | 32 |
| Learning/Machine Learning and AI/Classic Papers | 1 |
| Learning/Machine Learning and AI/Courses | 35 |
| Learning/Machine Learning and AI/Datasets and Benchmarks | 3 |
| Learning/Machine Learning and AI/Deep Learning Visuals | 2 |
| Learning/Machine Learning and AI/General Resources | 22 |
| Learning/Machine Learning and AI/LLMs and Modern AI | 33 |
| Learning/Machine Learning and AI/Papers and Concepts | 7 |
| Learning/Machine Learning and AI/Roadmaps and Study Plans | 1 |
| Learning/Math/Calculus Geometry Manifolds | 16 |
| Learning/Math/Linear Algebra | 13 |
| Learning/Statistics/Causality Probabilistic Inference | 1 |
| Learning/Statistics/Experimental Design | 5 |
| Learning/Statistics/Probability Inference | 30 |
| Research/01 Read First - Thesis Core/01 Subspace DS KDS GDS And CCA | 37 |
| Research/01 Read First - Thesis Core/02 OSCD And Classical Change Detection Baselines | 22 |
| Research/01 Read First - Thesis Core/03 Datasets And Problem Setting | 21 |
| Research/01 Read First - Thesis Core/04 Review Papers And Reality Checks | 27 |
| Research/01 Read First - Thesis Core/05 Spatial And Semantic Change Questions | 15 |
| Research/02 Methods - Subspace Geometry/01 General Subspace Geometry And Toolboxes | 47 |
| Research/02 Methods - Subspace Geometry/02 Kernel PCA And Kernel Subspace | 1 |
| Research/02 Methods - Subspace Geometry/03 Optional Green Learning Wavelets Compression | 23 |
| Research/02 Methods - Subspace Geometry/04 Hyperspectral Geometry Unmixing And Spectral Methods | 16 |
| Research/03 Methods - Change Detection/01 Binary And Unsupervised Change Detection | 77 |
| Research/03 Methods - Change Detection/02 Semantic And Damage Change Detection | 46 |
| Research/03 Methods - Change Detection/03 Remote Sensing Deep Learning | 97 |
| Research/03 Methods - Change Detection/04 Preprocessing Multitemporal And Sensor Issues | 58 |
| Research/04 Datasets - Current Candidate Future/01 Sentinel OSCD MultiSenGE And EO Datasets | 34 |
| Research/04 Datasets - Current Candidate Future/02 Earth Observation Portals And Catalogs | 23 |
| Research/04 Datasets - Current Candidate Future/03 Disaster Damage And Semantic Datasets | 11 |
| Research/04 Datasets - Current Candidate Future/04 Land Cover And Spectral Background | 35 |
| Research/04 Datasets - Current Candidate Future/05 Hyperspectral Change And Analysis Datasets | 5 |
| Research/05 Applications - Use Cases And Problem Framing/01 Greenhouses Agriculture Environmental Monitoring | 3 |
| Research/05 Applications - Use Cases And Problem Framing/02 Urban Infrastructure And Built Environment | 26 |
| Research/05 Applications - Use Cases And Problem Framing/03 Humanitarian And Conflict Mapping | 11 |
| Research/05 Applications - Use Cases And Problem Framing/04 Ecology And Wildlife Monitoring | 9 |
| Research/06 Background - Read As Needed/01 Remote Sensing Books And Guides | 6 |
| Research/06 Background - Read As Needed/02 ML CV Foundation Papers | 16 |
| Research/07 Literature Workflow - Search Venues Writing/01 Conferences Journals And Calls | 50 |
| Research/07 Literature Workflow - Search Venues Writing/02 Search And Reference Management | 17 |
| Research/07 Literature Workflow - Search Venues Writing/03 Writing Posters And Presentation | 10 |
| Research/08 Tools And Code - Implementation Support/01 Reference Implementations | 17 |
| Research/08 Tools And Code - Implementation Support/03 Geospatial Tools And Labeling | 12 |
| Research/09 Parking Lot - Lower Priority/02 Nonproject Papers To Triage Later | 46 |
| Research/09 Parking Lot - Lower Priority/03 Misc Project-Relevant Research Links | 18 |

## 5. Research Reading And Zotero Queue

For the latest labeled cleanup, Zotero/import triage should be topical first
and label-ordered second:

1. Open the topical folder that matches the current task, for example
   `Papers - Canonical Literature/Subspace Geometry...` for DS/GDS/KDS or
   `Data - Datasets.../OSCD, Sentinel-2, HLS, MultiSenGE` for dataset gates.
2. Within that folder, process `🔥` before `⭐`, and use `📌`
   only when it supports the immediate task.
3. Leave `❓` alone unless the task is explicitly bookmark cleanup,
   Zotero cleanup, or identifying weak/bare PDFs.

There is no separate `Read First` folder in the latest file. The user preferred
topical organization with inline importance labels over a duplicated queue.

The new 2026-06-17 source adds many modern change-detection and benchmark links. Read them as comparison pressure and problem-framing material; do not let them replace the immediate spatial-DS experiment unless the research direction is intentionally pivoted.

The 2026-06-20 batch adds focused temporal-irrigation, hyperspectral CD,
anomalous-change, SAM/MAD, tensor GDS, MultiSenGE, and dataset resources. The
highest-priority additions are IrrMapper, Theiler anomalous change, the CiTIUS
hyperspectral benchmark, the original MAD/MAF source, MultiSenGE, and n-mode
GDS. Generic diffusion/LLM links were kept under `Learning`, not promoted into
the thesis reading queue.

The 2026-06-25 batch adds many modern change-detection, foundation/open-vocabulary,
subspace/CCA/GDS, Green Learning, temporal-subspace, hyperspectral, dataset, and
reference-code leads. Treat these as reading pressure and candidate mechanisms,
not as proof that the project should pivot before a new held-out gate supports
the change.

## 6. Review Buckets

`To Review` still contains `169` entries from the full Chrome/Safari organization. These are preserved because the organizer could not classify them confidently without risking wrong placement.

- `To Review/Search Social Video` (84)
- `To Review/Uncategorized` (85)

The latest labeled `Research` parking folder contains `66` lower-priority or ambiguous research-like entries:

- `Research/Parking - Search Leads And Unclear/Search Queries - Other` (40)
- `Research/Parking - Search Leads And Unclear/Search Queries - Change Detection Remote Sensing` (14)
- `Research/Parking - Search Leads And Unclear/Search Queries - Subspace Methods` (7)
- `Research/Parking - Search Leads And Unclear/Search Queries - Datasets, Venues, Tools` (5)

## 7. How To Use This

Import `docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html` into Chrome when ready. In Chrome, open `Research`, choose the topical folder that matches the task, and read by label order: `🔥`, then `⭐`, then `📌`. Use `Learning` for basic ML/statistics/programming study rather than treating every tutorial as thesis literature.

Final QA note:

- A 2026-06-12 QA pass removed obvious non-research false positives from `Research`, including personal searches, app-development examples, voice-AI links, generic LSTM tutorials, and generic ML roadmap material.
- Every source URL remains preserved in the organized import file; the pass changed only folder placement and reading priority.
- Shared-concept links receive one primary folder. If a paper/resource is both programming/tooling and research-relevant, it stays under `Research` only when it directly supports thesis methods, datasets, evaluation, or literature framing.

## 8. 2026-06-25 Refinement Plan

Status: completed in
`docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html`.

Implemented choices:

- Removed the confusing numeric prefixes from the live `Research` taxonomy while
  preserving deliberate order through Chrome folder order.
- Kept topical/context folders instead of a separate `Read First` queue.
- Added inline importance labels: `🔥`, `⭐`, `📌`, `❓`.
- Separated papers, datasets, code/tooling, applications, workflow/reference,
  background, and search/unclear parking more strongly.
- Deduplicated obvious intellectual duplicates across arXiv abstract/PDF,
  publisher pages, author PDFs, signed ScienceDirect PDFs, IEEE/author mirrors,
  and repeated dataset/catalog variants.
- Flattened one-item leaf folders and fixed leading title artifacts such as
  stray `))` prefixes.

Historical planning note before the completed labeled cleanup: the earlier
2026-06-25 bookmark tree mixed three axes: reading priority, subject area, and
resource type. That made some folders useful for "what should I read next?"
but less precise for "where would this information live?"

Refinement principle: keep a primary no-duplicate tree, but make folder paths
answer a user intent. Reading-priority folders should be small queues, while
the durable archive should be organized by method, data/evaluation, application,
implementation, and literature workflow.

Earlier proposed `Research` top-level structure, kept as provenance and not
used as the final import target:

1. `00 Start Here - Active Queues`
   - `01 Zotero Import Next`
   - `02 Thesis Core Must Read`
   - `03 Baselines To Compare`
   - `04 Dataset Gates To Verify`
   - `05 Advisor Questions And Reality Checks`
2. `10 Methods - Subspace Geometry`
   - `01 DS GDS KDS And Difference Subspaces`
   - `02 CCA KCCA MSM KMSM And Subspace Recognition`
   - `03 Grassmann Geometry Distances And Toolboxes`
   - `04 Temporal Subspaces SSA SFA DTW RTW`
   - `05 Green Learning PixelHop Wavelets Compression`
   - `06 Hyperspectral Geometry Unmixing Spectral Subspaces`
3. `20 Methods - Change Detection Baselines`
   - `01 Classical Unsupervised CVA SAM PCA MAD IRMAD SFA`
   - `02 Anomaly And Distribution Shift Change Detection`
   - `03 Object Building And Semantic Change Detection`
   - `04 Damage Disaster And Humanitarian Change Detection`
4. `21 Methods - Neural And Foundation Change Detection`
   - `01 CNN Siamese UNet Transformer CD`
   - `02 Self Supervised Weakly Supervised And Domain Adaptation`
   - `03 Open Vocabulary VLM SAM Foundation Models`
   - `04 Multimodal Fusion And Cross Sensor Models`
5. `30 Data Evaluation And Protocols`
   - `01 OSCD Sentinel2 MSI And Current Benchmarks`
   - `02 Candidate Multispectral HLS MultiSenGE SpaceNet xBD S12`
   - `03 Hyperspectral Change Detection Datasets`
   - `04 Land Cover Buildings Roads And Ancillary Data`
   - `05 Portals Catalogs Download And Earth Engine`
   - `06 Metrics Protocols Label Quality And Reality Checks`
   - `07 Preprocessing Registration Cloud Radiometry Nuisance`
6. `40 Applications And Problem Framing`
   - `01 Disaster Damage Humanitarian Conflict`
   - `02 Urban Built Environment Infrastructure`
   - `03 Agriculture Greenhouses Irrigation`
   - `04 Ecology Environment Water Land Use`
7. `50 Implementation Code And Tools`
   - `01 Reference Implementations`
   - `02 Geospatial Toolchains Labeling And Annotation`
   - `03 Model Zoos Awesome Lists And Repos`
   - `04 Visualization Notebooks And Reproducibility`
8. `60 Literature Workflow Venues And Writing`
   - `01 Search Engines Scholar Zotero Reference Management`
   - `02 Conferences Journals Calls And Workshops`
   - `03 Writing Posters Presentations Review Process`
9. `80 Background - Read As Needed`
   - `01 Remote Sensing Books Guides And Fundamentals`
   - `02 ML CV Foundation Papers`
   - `03 Math Linear Algebra Manifolds Statistics`
10. `90 Search Leads Parking And Unclear`
   - `01 Search Queries To Resolve`
   - `02 Bare PDFs Or Weak Titles`
   - `03 Nonproject Papers`
   - `04 Misc Research-Like Links`

[completed] The final labeled cleanup split the largest mixed folders by
resource type and method family, removed numbering, and fixed high-priority
weak labels where the title could be inferred safely. Two weak/bare PDFs remain
in `Bare PDFs Or Weak Paper Titles To Identify` because the link itself is not
enough evidence to assign a confident title.

## 9. Applied Deduplication And Resource-Type Policy

Status: applied to the latest labeled cleanup file. The policy remains useful
for future bookmark imports.

User concern from 2026-06-25: the current research bookmark tree contains many
duplicate URLs for the same intellectual resource, such as publisher pages,
direct PDFs, arXiv abstract pages, arXiv PDFs, author-hosted PDFs, Google
Scholar/search links, and Chrome extension PDF wrappers. The target should be
one bookmark per paper/resource in the main taxonomy, not one bookmark per way
to access it.

Proposed deduplication unit: group by intellectual item, not exact URL. A paper,
dataset, code repository, survey page, or search lead should appear once in the
main tree unless the URLs are genuinely different resources.

Proposed canonical-link priority:

1. For arXiv papers, keep the `https://arxiv.org/abs/<id>` page rather than the
   direct `https://arxiv.org/pdf/<id>` link. The abstract page carries title,
   authors, version history, citation metadata, and a PDF link, and it is easier
   to understand later.
2. For final published papers, prefer DOI or publisher landing pages when they
   are useful for citation and metadata. If the publisher page is not practically
   accessible and an author or university PDF is the only full-text route, keep
   the accessible PDF instead and use a clear title that includes the paper
   title and source hint.
3. Remove `chrome-extension://` PDF-wrapper links when the underlying PDF or
   landing page exists.
4. Remove Google Search, Google Scholar, and generic search-query bookmarks when
   a direct paper, dataset, or code URL is already known. Keep search queries
   only in a search-leads folder when the exact target is still unknown.
5. Keep GitHub/code, datasets, benchmark pages, and tooling outside paper-only
   literature buckets even if they accompany a paper.
6. Flatten any folder that contains only one bookmark unless the folder is a
   stable semantic category expected to receive more items soon. In Chrome
   bookmark organization, one-item folders are usually retrieval friction.

Applied split in the latest labeled cleanup:

- `Papers - Canonical Literature`
- `Data - Datasets, Benchmarks, Evaluation`
- `Code - Repositories, Toolboxes, Implementations`
- `Applications - Problem Framing`
- `Workflow - Search, Venues, Writing, Reference`
- `Background - Learning Tutorials`
- `Parking - Search Leads And Unclear`

[completed] The duplicate-candidate audit grouped links by normalized arXiv ID,
DOI, IEEE article number, ScienceDirect PII, title tokens, known author/PDF
equivalents, and obvious PDF/abstract pairs. The latest file keeps one
canonical bookmark per detected group while preserving structural count checks.

Latest labeled cleanup status:

- File: `docs/source_records/bookmarks/chrome_bookmarks_research_labeled_cleaned_2026-06-25.html`
- Total bookmarks: `1869`
- Research bookmarks: `740`
- Research duplicate removals from the refined trial: `17`
- Exact duplicate URLs: `0`
- One-item leaf folders: `0`
- Numbered Research folders: `0`
- Remaining leading-title artifacts: `0`
- Research label counts: `🔥` 54, `⭐` 198, `📌` 414, `❓` 74
- Read-first queue: not included. The user preferred topical organization with
  inline importance labels.
