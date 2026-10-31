# ResearchRadar — Supported Knowledge Gaps in Robot Learning

Automated Knowledge Manifold Optimization (KMO) over a 120-document corpus spanning three conflicting robotics regimes: classical system identification, sim-to-real / domain randomization, and deep reinforcement learning control. The corpus is founded on user-curated seeding papers and expanded by a recency-bounded arXiv harvest.

Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. Corpus `120` documents (15 seeding + 105 harvested; RL 45, Sim2Real 40, SysID 35), represented as within-word character 4-7-gram TF-IDF (paper Sec. 5.2); no language model participates in any objective (120 documents on the chart).

All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.

## 1. Knowledge Manifold

| Quantity | Value |
| --- | --- |
| Chart | Ω = [−1, 1]², fixed |
| Documents | 120 |
| Corpus composition | 15 seeding papers (`doc 1`-`doc 15`) + 105 harvested (RL 45, Sim2Real 40, SysID 35) |
| Harvest window | arXiv `submittedDate` from 2016 (= earliest seed year minus the offset, floored by the lookback limit) |
| Gauge-fixing anchors | 9 (8 perimeter + 1 central) |
| Stress minimization | metric stress 607.16, Spearman(distance) 0.6070 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.25 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0529, q60 0.1644, q85 0.3105, q98 0.4294 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **69 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **64 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | +0.40 | -0.80 | 0.892 | 1.000 | 0.829 | 0.432 | 0.6032 | Archetype A | 15.5 |
| 2 | +0.30 | -0.80 | 0.851 | 1.000 | 0.683 | 0.501 | 0.6100 | Archetype A | 17.5 |
| 3 | +0.40 | -0.70 | 0.930 | 1.000 | 0.595 | 0.530 | 0.6246 | Archetype B | 18.4 |
| 4 | +0.70 | +0.50 | 0.864 | 0.947 | 0.719 | 0.443 | 0.6406 | Archetype C | 15.2 |
| 5 | +0.50 | +0.70 | 0.744 | 0.947 | 0.754 | 0.453 | 0.6541 | Archetype D | 15.7 |
| 6 | +0.50 | -0.70 | 0.763 | 0.947 | 0.751 | 0.437 | 0.6618 | Archetype B | 14.2 |
| 7 | +0.40 | -0.60 | 0.944 | 0.946 | 0.455 | 0.629 | 0.6642 | Archetype B | 21.5 |
| 8 | +0.50 | -0.60 | 0.799 | 0.950 | 0.566 | 0.535 | 0.6685 | Archetype B | 17.7 |
| 9 | +0.60 | +0.50 | 0.849 | 0.950 | 0.512 | 0.567 | 0.6717 | Archetype C | 18.9 |
| 10 | +0.30 | -0.70 | 0.910 | 1.000 | 0.461 | 0.602 | 0.6754 | Archetype E | 20.9 |
| 11 | +0.40 | +0.80 | 0.691 | 1.000 | 0.724 | 0.464 | 0.6771 | Archetype D | 17.0 |
| 12 | +0.30 | -0.90 | 0.761 | 1.000 | 1.000 | 0.366 | 0.6772 | Archetype A | 13.2 |
| 13 | +0.80 | +0.40 | 0.750 | 1.000 | 0.707 | 0.441 | 0.6790 | Archetype C | 16.1 |
| 14 | +0.50 | +0.80 | 0.769 | 1.000 | 1.000 | 0.354 | 0.6862 | Archetype D | 12.4 |
| 15 | +0.50 | +0.60 | 0.744 | 0.950 | 0.530 | 0.574 | 0.6863 | Archetype D | 19.2 |
| 16 | +0.70 | +0.40 | 0.859 | 1.000 | 0.464 | 0.538 | 0.7219 | Archetype C | 18.8 |
| 17 | +0.80 | +0.20 | 0.654 | 1.000 | 0.563 | 0.532 | 0.7280 | Archetype F | 18.3 |
| 18 | +0.80 | +0.10 | 0.587 | 0.927 | 0.607 | 0.544 | 0.7338 | Archetype F | 18.2 |
| 19 | +0.20 | -0.70 | 0.846 | 1.000 | 0.380 | 0.619 | 0.7434 | Archetype E | 22.3 |
| 20 | +0.40 | +0.70 | 0.665 | 1.000 | 0.502 | 0.556 | 0.7464 | Archetype D | 19.9 |
| 21 | +0.40 | +0.90 | 0.733 | 0.638 | 1.000 | 0.383 | 0.7636 | Archetype D | 14.1 |
| 22 | +0.40 | -0.90 | 0.842 | 0.638 | 1.000 | 0.334 | 0.7742 | Archetype A | 12.5 |
| 23 | +0.90 | +0.30 | 0.518 | 1.000 | 0.997 | 0.393 | 0.7754 | Archetype F | 14.1 |
| 24 | +0.80 | -0.10 | 0.460 | 0.927 | 0.638 | 0.545 | 0.7968 | Archetype G | 17.9 |
| 25 | -0.20 | -0.80 | 0.553 | 1.000 | 0.507 | 0.547 | 0.8050 | Archetype H | 19.3 |
| 26 | +0.50 | +0.50 | 0.780 | 0.696 | 0.370 | 0.663 | 0.8074 | Archetype C | 22.8 |
| 27 | +0.60 | +0.40 | 0.873 | 0.740 | 0.333 | 0.644 | 0.8094 | Archetype C | 22.1 |
| 28 | +0.60 | +0.80 | 0.803 | 0.941 | 1.000 | 0.213 | 0.8130 | Archetype D | 7.7 |
| 29 | +0.40 | +0.60 | 0.676 | 0.818 | 0.362 | 0.651 | 0.8169 | Archetype D | 23.1 |
| 30 | +0.90 | +0.20 | 0.457 | 1.000 | 0.999 | 0.384 | 0.8211 | Archetype F | 13.1 |
| 31 | +0.50 | -0.50 | 0.846 | 0.531 | 0.436 | 0.634 | 0.8339 | Archetype B | 21.2 |
| 32 | +0.70 | +0.00 | 0.677 | 0.924 | 0.314 | 0.660 | 0.8347 | Archetype G | 22.8 |
| 33 | +0.70 | -0.10 | 0.628 | 1.000 | 0.336 | 0.653 | 0.8365 | Archetype G | 22.4 |
| 34 | +0.70 | +0.10 | 0.732 | 1.000 | 0.287 | 0.654 | 0.8365 | Archetype F | 22.6 |
| 35 | +0.30 | +0.70 | 0.630 | 1.000 | 0.334 | 0.631 | 0.8465 | Archetype I | 22.5 |
| 36 | +0.70 | +0.20 | 0.785 | 0.869 | 0.276 | 0.630 | 0.8514 | Archetype F | 22.1 |
| 37 | -0.10 | -0.70 | 0.662 | 1.000 | 0.288 | 0.670 | 0.8548 | Archetype H | 24.4 |
| 38 | +0.80 | +0.00 | 0.522 | 0.602 | 0.639 | 0.530 | 0.8591 | Archetype G | 18.0 |
| 39 | -0.30 | -0.90 | 0.402 | 1.000 | 0.865 | 0.397 | 0.8599 | Archetype H | 14.7 |
| 40 | +0.80 | -0.20 | 0.398 | 1.000 | 0.601 | 0.516 | 0.8694 | Archetype G | 17.6 |
| 41 | +0.30 | -0.60 | 0.946 | 0.354 | 0.361 | 0.678 | 0.9650 | Archetype E | 24.5 |
| 42 | +0.10 | +0.70 | 0.604 | 0.818 | 0.187 | 0.679 | 0.9765 | Archetype I | 25.0 |
| 43 | +0.30 | +0.20 | 0.882 | 0.354 | 0.248 | 0.978 | 0.9982 | Archetype J | 45.5 |
| 44 | +0.30 | -0.20 | 0.947 | 0.285 | 0.298 | 0.960 | 1.0044 | Archetype K | 44.6 |
| 45 | +0.60 | +0.60 | 0.833 | 0.167 | 0.758 | 0.455 | 1.0381 | Archetype C | 14.8 |
| 46 | +0.60 | -0.10 | 0.777 | 0.387 | 0.224 | 0.744 | 1.0456 | Archetype G | 26.7 |
| 47 | -0.70 | +0.30 | 0.614 | 0.959 | 0.074 | 0.678 | 1.0542 | Archetype L | 26.4 |
| 48 | +0.30 | -0.30 | 0.957 | 0.200 | 0.312 | 0.906 | 1.0602 | Archetype K | 39.8 |
| 49 | +0.70 | +0.60 | 0.850 | 0.160 | 0.972 | 0.330 | 1.0853 | Archetype C | 10.7 |
| 50 | -0.60 | -0.40 | 0.499 | 0.837 | 0.093 | 0.695 | 1.0920 | Archetype M | 26.8 |
| 51 | +0.20 | +0.00 | 0.888 | 0.246 | 0.218 | 1.000 | 1.0923 | Archetype J | 55.4 |
| 52 | +0.30 | -0.50 | 0.956 | 0.160 | 0.327 | 0.766 | 1.1019 | Archetype E | 28.9 |
| 53 | -0.70 | +0.00 | 0.688 | 0.571 | 0.061 | 0.749 | 1.1078 | Archetype N | 28.9 |
| 54 | -0.50 | -0.50 | 0.513 | 0.563 | 0.130 | 0.715 | 1.1252 | Archetype M | 26.5 |
| 55 | -0.20 | +0.70 | 0.462 | 0.802 | 0.079 | 0.701 | 1.1252 | Archetype O | 26.0 |
| 56 | -0.60 | +0.40 | 0.558 | 0.670 | 0.053 | 0.711 | 1.1334 | Archetype L | 27.7 |
| 57 | +0.40 | -0.30 | 0.981 | 0.085 | 0.336 | 0.851 | 1.1405 | Archetype K | 33.8 |
| 58 | -0.40 | +0.60 | 0.429 | 0.881 | 0.044 | 0.703 | 1.1587 | Archetype O | 27.2 |
| 59 | -0.10 | +0.70 | 0.533 | 0.514 | 0.109 | 0.691 | 1.1590 | Archetype O | 26.1 |
| 60 | +0.40 | -0.50 | 0.967 | 0.053 | 0.386 | 0.728 | 1.1619 | Archetype B | 25.2 |
| 61 | -0.30 | -0.60 | 0.624 | 0.396 | 0.113 | 0.738 | 1.1667 | Archetype H | 27.6 |
| 62 | +0.20 | -0.10 | 0.878 | 0.121 | 0.227 | 0.986 | 1.1766 | Archetype J | 53.9 |
| 63 | +0.30 | -0.10 | 0.939 | 0.044 | 0.278 | 0.987 | 1.1995 | Archetype J | 47.6 |
| 64 | +0.40 | +0.10 | 0.948 | 0.010 | 0.237 | 0.925 | 1.2534 | Archetype J | 39.8 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 15 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 4 | (+0.40, -0.80) | P_boundary 0.89, W 1.00, G 0.83, S 0.43 | 0.6032 |
| Archetype B | 6 | (+0.40, -0.70) | P_boundary 0.93, W 1.00, G 0.59, S 0.53 | 0.6246 |
| Archetype C | 8 | (+0.70, +0.50) | P_boundary 0.86, W 0.95, G 0.72, S 0.44 | 0.6406 |
| Archetype D | 8 | (+0.50, +0.70) | P_boundary 0.74, W 0.95, G 0.75, S 0.45 | 0.6541 |
| Archetype E | 4 | (+0.30, -0.70) | P_boundary 0.91, W 1.00, G 0.46, S 0.60 | 0.6754 |
| Archetype F | 6 | (+0.80, +0.20) | P_boundary 0.65, W 1.00, G 0.56, S 0.53 | 0.7280 |
| Archetype G | 6 | (+0.80, -0.10) | P_boundary 0.46, W 0.93, G 0.64, S 0.54 | 0.7968 |
| Archetype H | 4 | (-0.20, -0.80) | P_boundary 0.55, W 1.00, G 0.51, S 0.55 | 0.8050 |
| Archetype I | 2 | (+0.30, +0.70) | P_boundary 0.63, W 1.00, G 0.33, S 0.63 | 0.8465 |
| Archetype J | 5 | (+0.30, +0.20) | P_boundary 0.88, W 0.35, G 0.25, S 0.98 | 0.9982 |
| Archetype K | 3 | (+0.30, -0.20) | P_boundary 0.95, W 0.28, G 0.30, S 0.96 | 1.0044 |
| Archetype L | 2 | (-0.70, +0.30) | P_boundary 0.61, W 0.96, G 0.07, S 0.68 | 1.0542 |
| Archetype M | 2 | (-0.60, -0.40) | P_boundary 0.50, W 0.84, G 0.09, S 0.69 | 1.0920 |
| Archetype N | 1 | (-0.70, +0.00) | P_boundary 0.69, W 0.57, G 0.06, S 0.75 | 1.1078 |
| Archetype O | 3 | (-0.20, +0.70) | P_boundary 0.46, W 0.80, G 0.08, S 0.70 | 1.1252 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (+0.40, -0.80)

**4** points in this archetype. Objectives: **P_boundary 0.892**, **W 1.000**, **G 0.829**, **S 0.432**; distance to ideal `0.6032`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| SysID | 0.036 | 0.021 | 0.001 |
| Sim2Real | 0.000 | 0.111 | 0.040 |
| RL | 0.448 | 0.306 | 0.361 |

Mixture entropy `H = 0.711`; effective contributing documents `N_eff = 15.5`; top-3 weight concentration `T3 = 0.321`; 5th-neighbour distance `d5 = 0.406`; relative uncertainty `u = 0.0071` (RBF surrogate; Matern `0.1250`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 31` | 0.1230 | SysID | 2016 | harvested | The interplay between system identification and machine learning |
| `doc 4` | 0.1108 | RL | 2025 | seed | Warm Starting of CMA-ES for Contextual Optimization Problems |
| `doc 47` | 0.0871 | SysID | 2022 | harvested | Distributed Online System Identification for LTI Systems Using Reverse Experience Replay |
| `doc 84` | 0.0743 | Sim2Real | 2025 | harvested | Statistical Guarantees for Offline Domain Randomization |
| `doc 9` | 0.0650 | RL | 2025 | seed | TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning |

**L2 theme lens:** `representation` (df=107), `information-theoretic` (df=4), `parameterized` (df=40), `optimization` (df=110), `parameterization` (df=14), `optimized` (df=77), `reparameterization` (df=2), `distribution` (df=90), `generalization` (df=69), `generalize` (df=64), `reparametrized` (df=3), `expectation-maximization` (df=2)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #3 at (+0.40, -0.70)

**6** points in this archetype. Objectives: **P_boundary 0.930**, **W 1.000**, **G 0.595**, **S 0.530**; distance to ideal `0.6246`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| SysID | 0.034 | 0.024 | 0.003 |
| Sim2Real | 0.000 | 0.119 | 0.051 |
| RL | 0.440 | 0.266 | 0.313 |

Mixture entropy `H = 0.707`; effective contributing documents `N_eff = 18.4`; top-3 weight concentration `T3 = 0.278`; 5th-neighbour distance `d5 = 0.325`; relative uncertainty `u = 0.0032` (RBF surrogate; Matern `0.0934`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 4` | 0.0982 | RL | 2025 | seed | Warm Starting of CMA-ES for Contextual Optimization Problems |
| `doc 31` | 0.0902 | SysID | 2016 | harvested | The interplay between system identification and machine learning |
| `doc 47` | 0.0892 | SysID | 2022 | harvested | Distributed Online System Identification for LTI Systems Using Reverse Experience Replay |
| `doc 84` | 0.0760 | Sim2Real | 2025 | harvested | Statistical Guarantees for Offline Domain Randomization |
| `doc 39` | 0.0654 | SysID | 2022 | harvested | Reinforcement Learning Approach to Estimation in Linear Systems |

**L2 theme lens:** `representation` (df=107), `information-theoretic` (df=4), `parameterized` (df=40), `optimization` (df=110), `parameterization` (df=14), `optimized` (df=77), `reparameterization` (df=2), `generalize` (df=64), `generalization` (df=69), `reparametrized` (df=3), `distribution` (df=90), `overparametrization` (df=2)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype C — gap #4 at (+0.70, +0.50)

**8** points in this archetype. Objectives: **P_boundary 0.864**, **W 0.947**, **G 0.719**, **S 0.443**; distance to ideal `0.6406`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| SysID | 0.000 | 0.000 | 0.000 |
| Sim2Real | 0.115 | 0.035 | 0.016 |
| RL | 0.422 | 0.001 | 0.002 |

Mixture entropy `H = 0.603`; effective contributing documents `N_eff = 15.2`; top-3 weight concentration `T3 = 0.339`; 5th-neighbour distance `d5 = 0.366`; relative uncertainty `u = 0.0031` (RBF surrogate; Matern `0.0812`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 17` | 0.1460 | SysID | 2024 | harvested | Cyclic Reformulation Based System Identification for Periodically Time-varying Systems |
| `doc 49` | 0.1018 | SysID | 2016 | harvested | Variable p norm constrained LMS algorithm based on gradient of root relative deviation.pdf |
| `doc 43` | 0.0909 | SysID | 2016 | harvested | Modeling Parallel Wiener-Hammerstein Systems Using Tensor Decomposition of Volterra Kernels |
| `doc 41` | 0.0749 | SysID | 2023 | harvested | A numerically efficient output-only system-identification framework for stochastically forced self-sustained oscillators |
| `doc 36` | 0.0610 | SysID | 2019 | harvested | Identification of abrupt stiffness changes of structures with tuned mass dampers under sudden events |

**L2 theme lens:** `system-identification` (df=3), `representation` (df=107), `identification` (df=59), `specific` (df=101), `parameterized` (df=40), `transformation` (df=56), `parameterization` (df=14), `quantification` (df=6), `complex` (df=104), `reparameterization` (df=2), `information-theoretic` (df=4), `robustification` (df=2)

**L1 concentration lens:** `wiener-hammerstein` (df=5), `linear-nonlinear-linear` (df=1), `single-input-single-output` (df=1), `identified` (df=46), `best-identified` (df=1), `misidentified` (df=1), `block-structure` (df=1), `system-identification` (df=3), `circulant-structured` (df=1), `identifiable` (df=4), `identification` (df=59), `deep-bayesian-system-identification` (df=1)

Terms present in **both** lenses (strongest evidence): `system-identification`, `identification`

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (+0.40, -0.80)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `representation` | 0.46528 | 107 | theme |
| L2 theme | `information-theoretic` | 0.41943 | 4 | theme |
| L2 theme | `parameterized` | 0.40735 | 40 | theme |
| L2 theme | `optimization` | 0.40088 | 110 | theme |
| L2 theme | `parameterization` | 0.36093 | 14 | theme |
| L2 theme | `optimized` | 0.35007 | 77 | theme |
| L2 theme | `reparameterization` | 0.34377 | 2 | theme |
| L2 theme | `distribution` | 0.34012 | 90 | theme |
| L2 theme | `generalization` | 0.33904 | 69 | theme |
| L2 theme | `generalize` | 0.33856 | 64 | theme |
| L2 theme | `reparametrized` | 0.33841 | 3 | theme |
| L2 theme | `expectation-maximization` | 0.33623 | 2 | theme |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

_None at gap #1._

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (+0.40, -0.80) | *(none)* | 0 |
| Archetype B | (+0.40, -0.70) | *(none)* | 0 |
| Archetype C | (+0.70, +0.50) | `wiener-hammerstein` (df=5), `identified` (df=46), `identifiable` (df=4), `identification` (df=59) | 8 |

## 6. Distance-maximising baseline (Table 4 analogue)

The negative control of Obayashi et al. (2026) Sec. 8.8: pick the 16 grid points with the largest nearest-neighbour distance, once over the whole grid and once over the interior only (excluding `x = ±1` and `y = ±1`), then compare their mean scores with the practical front. An unconstrained novelty search maximises emptiness, not discovery value.

| Selection | d_nn | S | R | P_boundary | G |
| --- | --- | --- | --- | --- | --- |
| **Practical Pareto front (this report)** | 0.2826 | 0.6126 | 0.7946 | 0.7363 | 0.4766 |
| Distance-maximising, all grid | 0.4402 | 0.0620 | 0.0298 | 0.1348 | 0.2261 |
| Distance-maximising, interior only | 0.3393 | 0.2930 | 0.3056 | 0.3926 | 0.6196 |

All 16 unconstrained picks sit on the chart edge, and 0 of the 16 interior-only picks do. Restricting the search to the 361 interior points recovers support (`S` 0.062 -> 0.293) but still trails the practical front, which is the paper's point: distance alone is not a discovery criterion. Two caveats when reading this table. The distance column is the *fifth*-neighbour distance used throughout this report, not the nearest-neighbour distance of the paper's Table 4, so the absolute numbers are not comparable across the two studies. And `R` is reported as a diagnostic only: it rewards low surrogate uncertainty (Sec. 3.7), which is why it is excluded from the objectives. The front scoring high on `R` therefore says the selected locations sit in well-sampled regions of the surrogate, not that the front was optimised for it.

## 7. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (+0.40, -0.80) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_+0.40_-0.80.json`, as produced by `code/make_evidence.py`. 
No real paper occupies that location.  Your task is to write the 
**virtual document**: a description of the research that *would* occupy 
it. 

Evidence terms are **whole words**, deterministically reconstructed from 
the corpus text (each entry lists its attested `word_forms`).  No 
fragment decoding is needed. 

The package contains: 
- `theme_lens_L2` — words indicating the local research theme. 
- `concentration_lens_L1` — words whose probability mass is locally 
  concentrated; each carries `df`, the number of corpus documents 
  containing the word, and, where `df <= 3`, its `source_docs`. 
- `contributing_documents_top3` — the highest-weighted neighboring 
  papers (titles and weights). 
- `mixture_entropy_H`, `effective_contributing_documents`, 
  `relative_uncertainty_u` — how broad and how well-supported the 
  location is. 

Rules (binding) 
1. **Theme from the theme lens.**  Build the consensus research theme 
   from the L2 words and the contributing documents.  Words appearing 
   in BOTH lens lists carry the greatest weight.  Titles of 
   contributing documents may help you interpret the words, but the 
   description must not be a paraphrase of those titles. 
2. **Serendipity from the concentration lens, graded by df.**  Words 
   appearing only in the L1 list are candidates for speculative 
   additions.  If `df >= 4`: you may present the word as an emergent 
   shared topic.  If `df <= 3`: you may mention it ONLY as a 
   single-source hypothesis, explicitly citing its `source_docs` 
   (e.g. "as explored in only one neighboring study [doc 18]"). 
   Never present a `df <= 3` word as established regional knowledge. 
3. **No outside facts.**  Do not assert anything the evidence package 
   does not support.  Your general knowledge of robotics may be used 
   to connect the evidence into fluent prose, not to add claims absent 
   from it. 
4. **Use every listed word or account for it.**  All terms in the 
   package are corpus words that survived a noise filter; none is a 
   bibliographic fragment.  If you nevertheless judge a term 
   unusable, say so explicitly in one line rather than silently 
   dropping it. 
5. **Close with the evidence list.**  End your answer with a section 
   `Evidence used:` listing every word you relied on, each with its 
   df, plus any term you set aside under Rule 4. 

Deliverables 
- **Task A**: the virtual document, in English, 200–280 words, 
  structured as a research abstract (topic, approach, expected 
  contribution).  Include a one-line caveat reflecting 
  `relative_uncertainty_u` and `effective_contributing_documents`. 
- **Task B**: a 45-word summary of the same virtual document. 
Output Task A, then Task B, then the `Evidence used:` list.  Do not 
include anything else — no reasoning trace, no extra sections. 

--- attached: evidence_point_+0.40_-0.80.json ---
{
  "location": {
    "x": 0.4,
    "y": -0.8
  },
  "theme_lens_L2": [
    {
      "term": "representation",
      "weight": 0.46528,
      "df": 107,
      "word_forms": [
        "represent",
        "representation",
        "representational",
        "representations",
        "represented",
        "representer",
        "representing",
        "represents"
      ]
    },
    {
      "term": "information-theoretic",
      "weight": 0.41943,
      "df": 4,
      "word_forms": [
        "information-theoretic",
        "information-theoretical",
        "information-theoretically"
      ]
    },
    {
      "term": "parameterized",
      "weight": 0.40735,
      "df": 40,
      "word_forms": [
        "parameterizations",
        "parameterize",
        "parameterized",
        "parameterizes"
      ]
    },
    {
      "term": "optimization",
      "weight": 0.40088,
      "df": 110,
      "word_forms": [
        "optimal",
        "optimally",
        "optimisation",
        "optimising",
        "optimization",
        "optimizing"
      ]
    },
    {
      "term": "parameterization",
      "weight": 0.36093,
      "df": 14,
      "word_forms": [
        "parameterisation",
        "parameterization",
        "parameterizing"
      ]
    },
    {
      "term": "optimized",
      "weight": 0.35007,
      "df": 77,
      "word_forms": [
        "optimizations",
        "optimize",
        "optimized",
        "optimizer",
        "optimizers",
        "optimizes"
      ]
    },
    {
      "term": "reparameterization",
      "weight": 0.34377,
      "df": 2,
      "word_forms": [
        "reparameterization"
      ]
    },
    {
      "term": "distribution",
      "weight": 0.34012,
      "df": 90,
      "word_forms": [
        "distribution",
        "distributional",
        "distributions"
      ]
    },
    {
      "term": "generalization",
      "weight": 0.33904,
      "df": 69,
      "word_forms": [
        "generalisation",
        "generality",
        "generalization",
        "generalizing"
      ]
    },
    {
      "term": "generalize",
      "weight": 0.33856,
      "df": 64,
      "word_forms": [
        "generalizations",
        "generalize",
        "generalized",
        "generalizes"
      ]
    },
    {
      "term": "reparametrized",
      "weight": 0.33841,
      "df": 3,
      "word_forms": [
        "reparametrizations",
        "reparametrized"
      ]
    },
    {
      "term": "expectation-maximization",
      "weight": 0.33623,
      "df": 2,
      "word_forms": [
        "expectation-maximization"
      ]
    }
  ],
  "concentration_lens_L1": [],
  "contributing_documents_top3": [
    {
      "doc": "doc 31",
      "source": "https://arxiv.org/abs/1612.09158v1",
      "is_seed": false,
      "title": "The interplay between system identification and machine learning",
      "authors": [
        "Gianluigi Pillonetto"
      ],
      "year": 2016,
      "regime": "SysID",
      "weight": 0.123
    },
    {
      "doc": "doc 4",
      "source": "https://arxiv.org/abs/2502.12555v1",
      "is_seed": true,
      "title": "Warm Starting of CMA-ES for Contextual Optimization Problems",
      "authors": [
        "Yuta Sekino",
        "Kento Uchida",
        "Shinichi Shirakawa"
      ],
      "year": 2025,
      "regime": "RL",
      "weight": 0.1108
    },
    {
      "doc": "doc 47",
      "source": "https://arxiv.org/abs/2207.01062v2",
      "is_seed": false,
      "title": "Distributed Online System Identification for LTI Systems Using Reverse Experience Replay",
      "authors": [
        "Ting-Jui Chang",
        "Shahin Shahrampour"
      ],
      "year": 2022,
      "regime": "SysID",
      "weight": 0.0871
    }
  ],
  "mixture_entropy_H": 0.7113,
  "effective_contributing_documents": 15.5,
  "relative_uncertainty_u": 0.0071
}
```

Output expectations from the template: Task A is a 200–280-word virtual abstract, Task B an exactly 45-word summary, followed by an `Evidence used:` list. The audit checklist in Appendix A.2 applies unchanged.

## 8. Reproduce and verify

```bash
.venv/bin/python -m src.harvest --verify-only
.venv/bin/python -m src.embed   --verify-only
.venv/bin/python -m src.manifold --verify-only
.venv/bin/python -m src.pareto  --verify-only
.venv/bin/python -m src.report   --verify-only
```

## 9. Caveats

- Novelty is **corpus-relative**. These are gaps in a 120-paper arXiv sample, not proof that the topics are unexplored in the wider literature.
- Abstracts, not full text, feed the vocabulary and the embeddings. The paper uses full-text character n-grams, so its exact term statistics are not reproducible at this scale.
- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately **not** a Pareto objective: maximizing low uncertainty favours already-dense regions.
- The sim-to-real query returns only ~140 candidates corpus-wide, so its 40 papers are the top 40 by relevance with no headroom beyond review rejection.

