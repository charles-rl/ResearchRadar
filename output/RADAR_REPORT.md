# ResearchRadar — Supported Knowledge Gaps in Robot Learning

Automated Knowledge Manifold Optimization (KMO) over a 120-document corpus spanning three conflicting robotics regimes: classical system identification, sim-to-real / domain randomization, and deep reinforcement learning control. The corpus is founded on user-curated seeding papers and expanded by a recency-bounded arXiv harvest.

Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. Corpus `120` documents (15 seeding + 105 harvested; Legged-Control 40, ZeroShot-Rep 41, Sim2Real-Physics 39), represented as within-word character 4-7-gram TF-IDF (paper Sec. 5.2); no language model participates in any objective (120 documents on the chart).

All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.

## 1. Knowledge Manifold

| Quantity | Value |
| --- | --- |
| Chart | Ω = [−1, 1]², fixed |
| Documents | 120 |
| Corpus composition | 15 seeding papers (`doc 1`-`doc 15`) + 105 harvested (Legged-Control 40, ZeroShot-Rep 41, Sim2Real-Physics 39) |
| Harvest window | arXiv `submittedDate` from 2015 (= earliest seed year minus the offset, floored by the lookback limit) |
| Gauge-fixing anchors | 9 (8 perimeter + 1 central) |
| Stress minimization | metric stress 578.95, Spearman(distance) 0.7053 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.25 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0502, q60 0.1924, q85 0.3105, q98 0.4294 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **65 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **60 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | +0.20 | -0.80 | 0.963 | 1.000 | 0.789 | 0.517 | 0.5283 | Archetype A | 16.8 |
| 2 | +0.10 | -0.80 | 0.990 | 0.746 | 0.827 | 0.523 | 0.5681 | Archetype A | 16.5 |
| 3 | +0.40 | -0.80 | 0.887 | 1.000 | 0.833 | 0.425 | 0.6099 | Archetype B | 15.4 |
| 4 | +0.30 | -0.90 | 0.955 | 1.000 | 1.000 | 0.379 | 0.6226 | Archetype A | 12.9 |
| 5 | +0.20 | -0.90 | 0.976 | 1.000 | 1.000 | 0.364 | 0.6366 | Archetype A | 11.8 |
| 6 | +0.20 | -0.70 | 0.931 | 0.757 | 0.477 | 0.612 | 0.6981 | Archetype C | 21.1 |
| 7 | +0.10 | -0.70 | 0.975 | 0.579 | 0.562 | 0.636 | 0.7081 | Archetype C | 21.9 |
| 8 | +0.40 | -0.70 | 0.826 | 1.000 | 0.495 | 0.528 | 0.7130 | Archetype B | 18.6 |
| 9 | +0.30 | -0.70 | 0.886 | 0.781 | 0.417 | 0.576 | 0.7621 | Archetype C | 20.2 |
| 10 | +0.00 | -0.70 | 0.999 | 0.451 | 0.597 | 0.658 | 0.7623 | Archetype D | 22.4 |
| 11 | +0.50 | -0.60 | 0.712 | 0.764 | 0.458 | 0.570 | 0.7857 | Archetype B | 19.4 |
| 12 | +0.60 | -0.50 | 0.606 | 0.764 | 0.453 | 0.590 | 0.8238 | Archetype E | 20.5 |
| 13 | +0.70 | -0.40 | 0.542 | 1.000 | 0.437 | 0.571 | 0.8427 | Archetype F | 21.7 |
| 14 | +0.50 | +0.60 | 0.806 | 0.764 | 0.322 | 0.592 | 0.8481 | Archetype G | 22.2 |
| 15 | +0.90 | -0.30 | 0.395 | 1.000 | 0.936 | 0.407 | 0.8495 | Archetype F | 15.5 |
| 16 | +0.90 | +0.40 | 0.487 | 0.638 | 0.848 | 0.407 | 0.8773 | Archetype H | 15.0 |
| 17 | +0.80 | -0.30 | 0.421 | 1.000 | 0.511 | 0.549 | 0.8817 | Archetype F | 20.8 |
| 18 | +0.40 | +0.70 | 0.629 | 1.000 | 0.291 | 0.613 | 0.8885 | Archetype G | 22.7 |
| 19 | -0.80 | +0.20 | 0.600 | 1.000 | 0.316 | 0.578 | 0.8980 | Archetype I | 20.2 |
| 20 | +0.40 | +0.60 | 0.711 | 0.939 | 0.188 | 0.694 | 0.9165 | Archetype G | 26.5 |
| 21 | +0.40 | -0.60 | 0.732 | 0.632 | 0.267 | 0.637 | 0.9357 | Archetype B | 22.1 |
| 22 | +0.00 | -0.30 | 1.000 | 0.236 | 0.436 | 0.965 | 0.9506 | Archetype J | 51.0 |
| 23 | +0.50 | -0.50 | 0.590 | 0.796 | 0.222 | 0.682 | 0.9569 | Archetype E | 23.6 |
| 24 | +0.60 | -0.40 | 0.516 | 1.000 | 0.210 | 0.661 | 0.9868 | Archetype F | 24.6 |
| 25 | -0.30 | -0.60 | 0.689 | 0.569 | 0.213 | 0.693 | 0.9984 | Archetype D | 25.3 |
| 26 | -0.10 | -0.60 | 0.939 | 0.186 | 0.468 | 0.762 | 1.0029 | Archetype D | 28.4 |
| 27 | +0.10 | -0.30 | 0.950 | 0.192 | 0.400 | 0.985 | 1.0083 | Archetype J | 48.8 |
| 28 | +0.70 | -0.30 | 0.442 | 1.000 | 0.230 | 0.644 | 1.0150 | Archetype F | 24.3 |
| 29 | +0.10 | -0.60 | 0.971 | 0.163 | 0.457 | 0.738 | 1.0318 | Archetype C | 27.2 |
| 30 | -0.70 | +0.30 | 0.412 | 0.766 | 0.223 | 0.654 | 1.0601 | Archetype I | 24.7 |
| 31 | +0.00 | -0.50 | 1.000 | 0.077 | 0.493 | 0.866 | 1.0615 | Archetype D | 35.4 |
| 32 | +0.00 | -0.20 | 1.000 | 0.114 | 0.393 | 0.975 | 1.0749 | Archetype J | 57.5 |
| 33 | +0.50 | +0.50 | 0.741 | 0.416 | 0.181 | 0.711 | 1.0785 | Archetype G | 26.2 |
| 34 | +0.20 | -0.10 | 0.920 | 0.194 | 0.282 | 0.983 | 1.0821 | Archetype K | 56.1 |
| 35 | -0.70 | +0.00 | 0.745 | 0.471 | 0.127 | 0.719 | 1.0891 | Archetype L | 26.4 |
| 36 | +0.30 | +0.60 | 0.557 | 0.613 | 0.111 | 0.735 | 1.0986 | Archetype G | 29.1 |
| 37 | -0.70 | -0.20 | 0.556 | 0.860 | 0.029 | 0.701 | 1.1175 | Archetype M | 26.1 |
| 38 | +0.10 | +0.10 | 0.865 | 0.160 | 0.273 | 0.985 | 1.1193 | Archetype N | 63.4 |
| 39 | +0.40 | +0.50 | 0.781 | 0.349 | 0.146 | 0.768 | 1.1200 | Archetype G | 30.4 |
| 40 | -0.10 | -0.30 | 0.940 | 0.096 | 0.340 | 0.996 | 1.1209 | Archetype J | 49.3 |
| 41 | -0.70 | -0.10 | 0.641 | 0.531 | 0.076 | 0.709 | 1.1341 | Archetype L | 26.5 |
| 42 | -0.70 | -0.30 | 0.433 | 1.000 | 0.033 | 0.673 | 1.1676 | Archetype M | 25.1 |
| 43 | +0.00 | -0.10 | 0.957 | 0.011 | 0.344 | 0.987 | 1.1878 | Archetype J | 62.0 |
| 44 | +0.50 | -0.40 | 0.506 | 0.450 | 0.085 | 0.755 | 1.2017 | Archetype E | 27.4 |
| 45 | -0.30 | -0.20 | 0.788 | 0.247 | 0.077 | 1.000 | 1.2094 | Archetype O | 46.0 |
| 46 | -0.60 | -0.40 | 0.354 | 0.896 | 0.027 | 0.696 | 1.2110 | Archetype M | 25.5 |
| 47 | -0.60 | -0.30 | 0.482 | 0.608 | 0.000 | 0.741 | 1.2201 | Archetype M | 28.3 |
| 48 | -0.30 | +0.00 | 0.925 | 0.136 | 0.136 | 0.997 | 1.2239 | Archetype O | 53.6 |
| 49 | +0.10 | +0.70 | 0.320 | 0.663 | 0.067 | 0.705 | 1.2381 | Archetype P | 26.8 |
| 50 | +0.20 | -0.20 | 0.867 | 0.007 | 0.269 | 0.999 | 1.2405 | Archetype K | 50.6 |
| 51 | -0.10 | -0.10 | 1.000 | 0.000 | 0.258 | 1.000 | 1.2454 | Archetype J | 59.5 |
| 52 | -0.60 | +0.30 | 0.358 | 0.373 | 0.156 | 0.770 | 1.2531 | Archetype I | 29.7 |
| 53 | -0.20 | +0.10 | 0.860 | 0.084 | 0.120 | 1.000 | 1.2783 | Archetype O | 58.2 |
| 54 | +0.30 | +0.10 | 0.714 | 0.055 | 0.176 | 1.000 | 1.2860 | Archetype N | 51.1 |
| 55 | -0.60 | +0.40 | 0.225 | 0.457 | 0.162 | 0.708 | 1.2968 | Archetype I | 26.7 |
| 56 | +0.10 | +0.30 | 0.623 | 0.097 | 0.138 | 0.994 | 1.3039 | Archetype N | 53.0 |
| 57 | -0.40 | +0.10 | 0.809 | 0.012 | 0.135 | 1.000 | 1.3271 | Archetype O | 47.3 |
| 58 | -0.50 | -0.40 | 0.438 | 0.355 | 0.000 | 0.779 | 1.3345 | Archetype M | 28.5 |
| 59 | +0.60 | -0.20 | 0.402 | 0.319 | 0.041 | 0.772 | 1.3392 | Archetype F | 29.3 |
| 60 | -0.30 | +0.70 | 0.000 | 1.000 | 0.037 | 0.682 | 1.4244 | Archetype Q | 25.5 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 17 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 4 | (+0.20, -0.80) | P_boundary 0.96, W 1.00, G 0.79, S 0.52 | 0.5283 |
| Archetype B | 4 | (+0.40, -0.80) | P_boundary 0.89, W 1.00, G 0.83, S 0.42 | 0.6099 |
| Archetype C | 4 | (+0.20, -0.70) | P_boundary 0.93, W 0.76, G 0.48, S 0.61 | 0.6981 |
| Archetype D | 4 | (+0.00, -0.70) | P_boundary 1.00, W 0.45, G 0.60, S 0.66 | 0.7623 |
| Archetype E | 3 | (+0.60, -0.50) | P_boundary 0.61, W 0.76, G 0.45, S 0.59 | 0.8238 |
| Archetype F | 6 | (+0.70, -0.40) | P_boundary 0.54, W 1.00, G 0.44, S 0.57 | 0.8427 |
| Archetype G | 6 | (+0.50, +0.60) | P_boundary 0.81, W 0.76, G 0.32, S 0.59 | 0.8481 |
| Archetype H | 1 | (+0.90, +0.40) | P_boundary 0.49, W 0.64, G 0.85, S 0.41 | 0.8773 |
| Archetype I | 4 | (-0.80, +0.20) | P_boundary 0.60, W 1.00, G 0.32, S 0.58 | 0.8980 |
| Archetype J | 6 | (+0.00, -0.30) | P_boundary 1.00, W 0.24, G 0.44, S 0.97 | 0.9506 |
| Archetype K | 2 | (+0.20, -0.10) | P_boundary 0.92, W 0.19, G 0.28, S 0.98 | 1.0821 |
| Archetype L | 2 | (-0.70, +0.00) | P_boundary 0.75, W 0.47, G 0.13, S 0.72 | 1.0891 |
| Archetype M | 5 | (-0.70, -0.20) | P_boundary 0.56, W 0.86, G 0.03, S 0.70 | 1.1175 |
| Archetype N | 3 | (+0.10, +0.10) | P_boundary 0.87, W 0.16, G 0.27, S 0.99 | 1.1193 |
| Archetype O | 4 | (-0.30, -0.20) | P_boundary 0.79, W 0.25, G 0.08, S 1.00 | 1.2094 |
| Archetype P | 1 | (+0.10, +0.70) | P_boundary 0.32, W 0.66, G 0.07, S 0.71 | 1.2381 |
| Archetype Q | 1 | (-0.30, +0.70) | P_boundary 0.00, W 1.00, G 0.04, S 0.68 | 1.4244 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (+0.20, -0.80)

**4** points in this archetype. Objectives: **P_boundary 0.963**, **W 1.000**, **G 0.789**, **S 0.517**; distance to ideal `0.5283`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| ZeroShot-Rep | 0.029 | 0.217 | 0.264 |
| Sim2Real-Physics | 0.223 | 0.002 | 0.002 |
| Legged-Control | 0.136 | 0.008 | 0.016 |

Mixture entropy `H = 0.827`; effective contributing documents `N_eff = 16.8`; top-3 weight concentration `T3 = 0.296`; 5th-neighbour distance `d5 = 0.316`; relative uncertainty `u = 0.0034` (RBF surrogate; Matern `0.0998`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 89` | 0.1069 | Legged-Control | 2018 | harvested | Bipedal Hopping: Reduced-order Model Embedding via Optimization-based Control |
| `doc 31` | 0.0980 | ZeroShot-Rep | 2024 | harvested | Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity |
| `doc 4` | 0.0912 | Legged-Control | 2025 | seed | Warm Starting of CMA-ES for Contextual Optimization Problems |
| `doc 85` | 0.0902 | Sim2Real-Physics | 2024 | harvested | Domains as Objectives: Domain-Uncertainty-Aware Policy Optimization through Explicit Multi-Domain Convex Coverage Set Learning |
| `doc 1` | 0.0744 | Legged-Control | 2020 | seed | Warm Starting CMA-ES for Hyperparameter Optimization |

**L2 theme lens:** `representation` (df=113), `representation-learning` (df=4), `state-representation` (df=2), `optimization` (df=111), `computationally-efficient` (df=2), `computed` (df=101), `optimized` (df=83), `parameterized` (df=59), `fb-representation` (df=2), `transformer` (df=42), `generalize` (df=78), `generalization` (df=64)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #3 at (+0.40, -0.80)

**4** points in this archetype. Objectives: **P_boundary 0.887**, **W 1.000**, **G 0.833**, **S 0.425**; distance to ideal `0.6099`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| ZeroShot-Rep | 0.011 | 0.094 | 0.228 |
| Sim2Real-Physics | 0.091 | 0.001 | 0.003 |
| Legged-Control | 0.078 | 0.016 | 0.038 |

Mixture entropy `H = 0.715`; effective contributing documents `N_eff = 15.4`; top-3 weight concentration `T3 = 0.334`; 5th-neighbour distance `d5 = 0.418`; relative uncertainty `u = 0.0072` (RBF surrogate; Matern `0.1307`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 48` | 0.1191 | ZeroShot-Rep | 2018 | harvested | Transfer with Model Features in Reinforcement Learning |
| `doc 31` | 0.1170 | ZeroShot-Rep | 2024 | harvested | Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity |
| `doc 25` | 0.0977 | ZeroShot-Rep | 2019 | harvested | Successor Features Combine Elements of Model-Free and Model-based Reinforcement Learning |
| `doc 85` | 0.0774 | Sim2Real-Physics | 2024 | harvested | Domains as Objectives: Domain-Uncertainty-Aware Policy Optimization through Explicit Multi-Domain Convex Coverage Set Learning |
| `doc 89` | 0.0583 | Legged-Control | 2018 | harvested | Bipedal Hopping: Reduced-order Model Embedding via Optimization-based Control |

**L2 theme lens:** `representation` (df=113), `representation-learning` (df=4), `state-representation` (df=2), `fb-representation` (df=2), `optimization` (df=111), `computationally-efficient` (df=2), `computed` (df=101), `transformer` (df=42), `generalize` (df=78), `approximate` (df=93), `generalization` (df=64), `optimized` (df=83)

**L1 concentration lens:** `state-action-value` (df=1), `state-action-feature` (df=1), `model-reduction` (df=1), `successor-feature` (df=6), `successorfeature` (df=1), `feature-predictive` (df=1), `value-predictive` (df=2)

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype C — gap #6 at (+0.20, -0.70)

**4** points in this archetype. Objectives: **P_boundary 0.931**, **W 0.757**, **G 0.477**, **S 0.612**; distance to ideal `0.6981`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| ZeroShot-Rep | 0.030 | 0.166 | 0.199 |
| Sim2Real-Physics | 0.175 | 0.004 | 0.004 |
| Legged-Control | 0.136 | 0.015 | 0.017 |

Mixture entropy `H = 0.801`; effective contributing documents `N_eff = 21.1`; top-3 weight concentration `T3 = 0.247`; 5th-neighbour distance `d5 = 0.265`; relative uncertainty `u = 0.0015` (RBF surrogate; Matern `0.0715`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 31` | 0.0876 | ZeroShot-Rep | 2024 | harvested | Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity |
| `doc 85` | 0.0828 | Sim2Real-Physics | 2024 | harvested | Domains as Objectives: Domain-Uncertainty-Aware Policy Optimization through Explicit Multi-Domain Convex Coverage Set Learning |
| `doc 4` | 0.0763 | Legged-Control | 2025 | seed | Warm Starting of CMA-ES for Contextual Optimization Problems |
| `doc 1` | 0.0698 | Legged-Control | 2020 | seed | Warm Starting CMA-ES for Hyperparameter Optimization |
| `doc 25` | 0.0609 | ZeroShot-Rep | 2019 | harvested | Successor Features Combine Elements of Model-Free and Model-based Reinforcement Learning |

**L2 theme lens:** `representation` (df=113), `representation-learning` (df=4), `state-representation` (df=2), `optimization` (df=111), `computationally-efficient` (df=2), `computed` (df=101), `fb-representation` (df=2), `parameterized` (df=59), `generalize` (df=78), `optimized` (df=83), `generalization` (df=64), `transformer` (df=42)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (+0.20, -0.80)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `representation` | 0.52035 | 113 | theme |
| L2 theme | `representation-learning` | 0.47208 | 4 | theme |
| L2 theme | `state-representation` | 0.41876 | 2 | theme |
| L2 theme | `optimization` | 0.38903 | 111 | theme |
| L2 theme | `computationally-efficient` | 0.37297 | 2 | theme |
| L2 theme | `computed` | 0.34448 | 101 | theme |
| L2 theme | `optimized` | 0.34304 | 83 | theme |
| L2 theme | `parameterized` | 0.33426 | 59 | theme |
| L2 theme | `fb-representation` | 0.33109 | 2 | theme |
| L2 theme | `transformer` | 0.32941 | 42 | theme |
| L2 theme | `generalize` | 0.32401 | 78 | theme |
| L2 theme | `generalization` | 0.32282 | 64 | theme |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

_None at gap #1._

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (+0.20, -0.80) | *(none)* | 0 |
| Archetype B | (+0.40, -0.80) | `successor-feature` (df=6) | 6 |
| Archetype C | (+0.20, -0.70) | *(none)* | 0 |

## 6. Distance-maximising baseline (Table 4 analogue)

The negative control of Obayashi et al. (2026) Sec. 8.8: pick the 16 grid points with the largest nearest-neighbour distance, once over the whole grid and once over the interior only (excluding `x = ±1` and `y = ±1`), then compare their mean scores with the practical front. An unconstrained novelty search maximises emptiness, not discovery value.

| Selection | d_nn | S | R | P_boundary | G |
| --- | --- | --- | --- | --- | --- |
| **Practical Pareto front (this report)** | 0.2367 | 0.7270 | 0.8400 | 0.7058 | 0.3266 |
| Distance-maximising, all grid | 0.4402 | 0.0634 | 0.0248 | 0.1432 | 0.2498 |
| Distance-maximising, interior only | 0.3402 | 0.3116 | 0.3027 | 0.3381 | 0.5799 |

All 16 unconstrained picks sit on the chart edge, and 0 of the 16 interior-only picks do. Restricting the search to the 361 interior points recovers support (`S` 0.063 -> 0.312) but still trails the practical front, which is the paper's point: distance alone is not a discovery criterion. Two caveats when reading this table. The distance column is the *fifth*-neighbour distance used throughout this report, not the nearest-neighbour distance of the paper's Table 4, so the absolute numbers are not comparable across the two studies. And `R` is reported as a diagnostic only: it rewards low surrogate uncertainty (Sec. 3.7), which is why it is excluded from the objectives. The front scoring high on `R` therefore says the selected locations sit in well-sampled regions of the surrogate, not that the front was optimised for it.

## 7. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (+0.20, -0.80) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_+0.20_-0.80.json`, as produced by `code/make_evidence.py`. 
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

--- attached: evidence_point_+0.20_-0.80.json ---
{
  "location": {
    "x": 0.2,
    "y": -0.8
  },
  "theme_lens_L2": [
    {
      "term": "representation",
      "weight": 0.52035,
      "df": 113,
      "word_forms": [
        "represent",
        "representation",
        "representational",
        "representations",
        "represented",
        "representing",
        "represents"
      ]
    },
    {
      "term": "representation-learning",
      "weight": 0.47208,
      "df": 4,
      "word_forms": [
        "representation-learning"
      ]
    },
    {
      "term": "state-representation",
      "weight": 0.41876,
      "df": 2,
      "word_forms": [
        "state-representation",
        "state-representations"
      ]
    },
    {
      "term": "optimization",
      "weight": 0.38903,
      "df": 111,
      "word_forms": [
        "optimal",
        "optimally",
        "optimising",
        "optimization",
        "optimizing"
      ]
    },
    {
      "term": "computationally-efficient",
      "weight": 0.37297,
      "df": 2,
      "word_forms": [
        "computationally-efficient"
      ]
    },
    {
      "term": "computed",
      "weight": 0.34448,
      "df": 101,
      "word_forms": [
        "computation",
        "computational",
        "computations",
        "compute",
        "computed",
        "computer",
        "computers",
        "computes",
        "computing"
      ]
    },
    {
      "term": "optimized",
      "weight": 0.34304,
      "df": 83,
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
      "term": "parameterized",
      "weight": 0.33426,
      "df": 59,
      "word_forms": [
        "parameterizations",
        "parameterize",
        "parameterized",
        "parameterizes"
      ]
    },
    {
      "term": "fb-representation",
      "weight": 0.33109,
      "df": 2,
      "word_forms": [
        "fb-representation"
      ]
    },
    {
      "term": "transformer",
      "weight": 0.32941,
      "df": 42,
      "word_forms": [
        "transform",
        "transformation",
        "transformations",
        "transformed",
        "transformer",
        "transformers",
        "transforming",
        "transforms"
      ]
    },
    {
      "term": "generalize",
      "weight": 0.32401,
      "df": 78,
      "word_forms": [
        "generalizations",
        "generalize",
        "generalized",
        "generalizes"
      ]
    },
    {
      "term": "generalization",
      "weight": 0.32282,
      "df": 64,
      "word_forms": [
        "generalisation",
        "generalising",
        "generality",
        "generalization",
        "generalizing"
      ]
    }
  ],
  "concentration_lens_L1": [],
  "contributing_documents_top3": [
    {
      "doc": "doc 89",
      "source": "https://arxiv.org/abs/1807.08037v2",
      "is_seed": false,
      "title": "Bipedal Hopping: Reduced-order Model Embedding via Optimization-based Control",
      "authors": [
        "Xiaobin Xiong",
        "Aaron Ames"
      ],
      "year": 2018,
      "regime": "Legged-Control",
      "weight": 0.1069
    },
    {
      "doc": "doc 31",
      "source": "https://arxiv.org/abs/2410.17904v1",
      "is_seed": false,
      "title": "Reinforcement Learning under Latent Dynamics: Toward Statistical and Algorithmic Modularity",
      "authors": [
        "Philip Amortila",
        "Dylan J. Foster",
        "Nan Jiang"
      ],
      "year": 2024,
      "regime": "ZeroShot-Rep",
      "weight": 0.098
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
      "regime": "Legged-Control",
      "weight": 0.0912
    }
  ],
  "mixture_entropy_H": 0.8265,
  "effective_contributing_documents": 16.83,
  "relative_uncertainty_u": 0.003379
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
- The vocabulary and the term statistics come from **full text** extracted from each paper's PDF, not from abstracts. Section boundaries are recovered by heading detection, so a paper whose headings the PDF text layer mangles contributes a slightly different term distribution than its author intended.
- The corpus is founded on 15 user-curated seeding papers plus 35 harvested candidates per regime. The sim-to-real query returns only ~140 candidates corpus-wide, so that regime's 35 harvested slots are the top 35 by relevance and have no headroom beyond review rejection.
- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately **not** a Pareto objective: maximizing low uncertainty favours already-dense regions.

