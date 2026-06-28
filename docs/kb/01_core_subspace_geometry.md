# KB 01: Core Subspace Geometry

Each entry records the mathematical object, claimed novelty, evidence, baselines, transfer, and project-specific failure risk.

## Difference Subspace And Its Generalization For Subspace-Based Methods

Fukui and Maki, *IEEE TPAMI* 37(11), 2015. `[P]`

**Object and math.** For subspaces with bases `U` and `V`, principal angles satisfy `svd(U^T V) = diag(cos(theta_i))`. The difference directions are proportional to paired canonical-vector differences. Analytically they are eigenvectors of `P_U + P_V` whose eigenvalues lie below one; common/principal-component directions have eigenvalues above one. The DS magnitude used by this project is

```text
Mag(D(U,V)) = 2 * sum_i (1 - cos(theta_i)).
```

GDS extends the construction to many class subspaces by eigendecomposing the sum of projectors and retaining directions of small aggregate occupancy. KDS/KGDS perform the corresponding operations after kernel PCA in an RKHS.

**Novelty and justification.** The paper contributes a subspace-valued representation of *what differs*, visualization through projection onto that basis, and GDS projection as a quasi-orthogonalizing discriminant transform. Evidence is image-set shape, face, and hand recognition on synthetic 3-D objects, Yale B+, Multi-PIE, and multi-view hand data. Comparisons include MSM/CMSM, orthogonal-subspace variants, DCC/KDCC, and kernel counterparts.

**Transfer boundary.** The input object is a set/class subspace, not a pair of isolated spectra. For a rank-1 spectral vector, DS magnitude is a monotone transform of SAM, so the method has no new detection information. For local centered HSI distributions, the DS basis may expose orientation change, but a covariance eigenspace supplies the same object; covariance is therefore the mandatory null.

**Verdict.** Foundational mechanism, not a satellite-CD novelty claim. Its only open contribution here is basis-level characterization after a nontrivial, justified set construction.

**Next falsifiable step.** Prove numerically and algebraically that the proposed local orientation score and DS leverage attribution are not merely a repackaging of a standard covariance/SPD decomposition.

## Discriminant Feature Extraction By Generalized Difference Subspace

Fukui, Sogi, Kobayashi, Xue, and Maki, TPAMI manuscript/publication lineage, 2022-2023. `[P]`

**Object and math.** The paper introduces geometrical FDA (gFDA) under the practical assumption that an uncentered class mean aligns with the first principal component. It shows GDS projection is equivalent to gFDA up to a small correction. Kernel gFDA/KGDS and CNN-feature variants extend the result.

**Novelty and justification.** The contribution is theoretical: it explains why GDS has Fisher-like discrimination while avoiding FDA's small-sample singularity. Experiments cover Yale B+, Multi-PIE, ALOI, ETH-80, MNIST, CIFAR-10, and CNN features. FDA/KFDA and subspace classifiers are the relevant baselines. Linear GDS does not automatically improve features already optimized by discriminative deep training.

**Transfer boundary.** GDS requires multiple reference classes or regimes. Unsupervised binary CD lacks these unless pseudo-classes are introduced, which risks circularity. It is more plausible as a supervised attribution/compression layer than as the detector.

**Verdict.** Strong explanation of GDS, weak direct fit to label-scarce HSI CD.

**Next falsifiable step.** Do not use GDS unless a real multi-regime training object can be specified without labels derived from the test change map.

## Second-Order Difference Subspace

Fukui, Valois, Souza, and Kobayashi, arXiv:2409.08563, 2024. `[P+V]`

**Object and math.** For sequential subspaces `S1,S2,S3`, form the principal-component/Karcher midpoint `M(S1,S3)` and define

```text
D2(S1,S2,S3) = D(S2, M(S1,S3)).
```

This is motivated by the central second difference. On a Grassmann manifold first- and second-order DS are interpreted as velocity and acceleration. Projection of `S2` to the sum space of `S1,S3` yields along-geodesic and orthogonal components. The rendered equation page confirms that the paper defines magnitude through the first-order DS of `S2` and the midpoint; it does not recover a signed acceleration vector.

**Novelty and justification.** The paper establishes the definition and demonstrates physical naturalness on CMU walking/jumping shape subspaces and one UCR biomedical anomaly series. The walking second-order magnitude correlates 0.948 with the derivative of first-order magnitude. There is no competitive change-detection benchmark.

**Transfer boundary.** The project's controlled tests show the satellite trajectory hypothesis is largely closed: mean-spectrum vector second differences match or beat curvature, second-order DS did not fire in the intended turn test, and the geometry was not nuisance-invariant. This paper must remain in the taxonomy but cannot be the top research task.

**Verdict.** Novel mathematical representation, experimentally unsupported as a distinct satellite-CD contribution in this project.

**Next falsifiable step.** None for the current top task; preserve it as a rejected/diagnostic branch unless a new structural output is identified that a vector second difference cannot produce.

## Generalized Mutual Subspace-Based Methods For Image-Set Classification

Kobayashi, image-set classification paper. `[P]`

**Object and math.** Mutual subspace similarity is softened by weighting basis directions rather than hard-selecting a subspace rank. The generalized discriminative construction uses those weighted similarities to avoid brittle dimension choice.

**Novelty and justification.** The paper addresses rank selection in image-set classification and validates on 3-D object image sets against conventional MSM/CMSM-like methods.

**Transfer boundary.** Soft rank weighting is directly useful as a stability ablation for local HSI eigenspaces. It does not create a new change signal; it only reduces rank sensitivity.

**Verdict.** Useful robustness device, not a candidate task.

**Next falsifiable step.** Compare fixed-rank, energy-rank, shrinkage, and soft-weighted orientation scores under eigengap degeneracy.

## Tensor Analysis With n-Mode Generalized Difference Subspace

Gatto et al., arXiv:1909.01954v2, 2020. `[P+V]`

**Object and math.** A tensor is unfolded mode by mode; each unfolding yields a factor subspace on a Grassmann manifold. Mode-specific GDS projections and a product of factor manifolds preserve separable tensor modes. The inspected figure confirms that tensor unfolding, not a single flattened PCA, is the defining sample construction.

**Novelty and justification.** The contribution is a tensor-classification extension of GDS with mode-wise discriminant projections and n-mode Fisher scores. Evidence is classification, not change detection.

**Transfer boundary.** HSI naturally has two spatial and one spectral mode, so the representation is appealing. However, patch-tensor HSI CD and low-rank tensor methods already exist. A proposal must name a mode-specific CD output and beat a conventional tensor-decomposition null; “first tensor subspace for HSI CD” would be false.

**Verdict.** Plausible secondary candidate with substantial closest-prior pressure and parameter cost.

**Next falsifiable step.** On a controlled tensor, verify that a mode-wise Grassmann score localizes a spectral-only change without duplicating Tucker-factor or patch-tensor reconstruction error.

## Geometry Of Subspace Set And Its Application To Machine Learning

Fukui, C-AIR lecture, 2024. `[P]`

**Content.** The lecture traces correlation/subspace methods, MSM, canonical angles, Grassmann learning, DS/GDS, and integration with learned features. Its value is conceptual provenance, not an independent empirical claim.

**Transfer boundary.** It supports the mechanism framing: “subspace” names a representation and comparison operator, not a complete CD family. It also legitimizes combining geometry with latent features, but not attributing the encoder's performance to geometry.

**Verdict.** Taxonomy scaffold; cite the primary papers for formal claims.

**Next falsifiable step.** Every proposed method must complete a construction card before implementation: sample unit, matrix, rank, preserved/lost information, operator, decision, and scalar/statistical null.

## Honest Novelty Verdict

Core geometry supplies mathematically valid representations but no surviving generic detector advantage. Second-order trajectory dynamics is closed by project evidence; n-mode and GDS are crowded mechanisms. The open use is a local centered eigenspace orientation and its DS basis attribution, provided covariance/SPD controls do not subsume it.

## Next Falsifiable Step

Derive the exact equivalence and non-equivalence relations among DS magnitude, Grassmann chordal distance, projector-Frobenius distance, and the orientation term of an SPD covariance decomposition before naming a new method.
