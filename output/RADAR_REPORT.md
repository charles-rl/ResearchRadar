# ResearchRadar — Supported Knowledge Gaps in Robot Learning

Automated Knowledge Manifold Optimization (KMO) over a 100-document corpus spanning conflicting robotics regimes: zero-shot sim-to-real transfer · success rate, successor measures · forward-backward representatio…, language models · reward, llms · capabilities, control · planning. The corpus is founded on user-curated seeding papers and expanded by a recency-bounded arXiv harvest.

Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. Corpus `100` documents (0 seeding + 100 harvested; successor measures · forward-backward representatio… 22, language models · reward 19, control · planning 15, zero-shot sim-to-real transfer · success rate 25, llms · capabilities 19), represented as within-word character 4-7-gram TF-IDF (paper Sec. 5.2); no language model participates in any objective (100 documents on the chart).

All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.

## 1. Knowledge Manifold

| Quantity | Value |
| --- | --- |
| Chart | Ω = [−1, 1]², fixed |
| Documents | 100 |
| Corpus composition | 0 seeding papers (`doc 1`-`doc 0`) + 100 harvested (successor measures · forward-backward representatio… 22, language models · reward 19, control · planning 15, zero-shot sim-to-real transfer · success rate 25, llms · capabilities 19) |
| Harvest window | arXiv `submittedDate` from 2021 (= earliest seed year minus the offset, floored by the lookback limit) |
| Gauge-fixing anchors | 9 (8 perimeter + 1 central) |
| Stress minimization | metric stress 451.38, Spearman(distance) 0.5030 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.25 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0545, q60 0.1747, q85 0.3105, q98 0.4294 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **82 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **75 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | +0.40 | +0.80 | 0.914 | 1.000 | 0.755 | 0.454 | 0.6051 | Archetype A | 13.6 |
| 2 | -0.40 | +0.80 | 0.801 | 1.000 | 0.797 | 0.459 | 0.6113 | Archetype B | 13.8 |
| 3 | -0.80 | +0.40 | 0.706 | 1.000 | 0.887 | 0.471 | 0.6159 | Archetype C | 14.2 |
| 4 | +0.30 | +0.90 | 0.874 | 1.000 | 1.000 | 0.391 | 0.6216 | Archetype A | 11.8 |
| 5 | +0.30 | +0.80 | 0.868 | 1.000 | 0.611 | 0.527 | 0.6268 | Archetype D | 15.3 |
| 6 | -0.80 | +0.30 | 0.754 | 1.000 | 0.649 | 0.531 | 0.6352 | Archetype C | 15.9 |
| 7 | +0.20 | +0.80 | 0.854 | 1.000 | 0.581 | 0.540 | 0.6386 | Archetype D | 15.4 |
| 8 | +0.10 | +0.80 | 0.851 | 0.846 | 0.590 | 0.541 | 0.6518 | Archetype D | 15.1 |
| 9 | +0.50 | +0.70 | 0.969 | 0.865 | 0.703 | 0.436 | 0.6522 | Archetype A | 12.6 |
| 10 | -0.50 | +0.70 | 0.783 | 0.865 | 0.808 | 0.423 | 0.6595 | Archetype E | 12.7 |
| 11 | -0.70 | +0.40 | 0.740 | 1.000 | 0.564 | 0.573 | 0.6633 | Archetype C | 16.9 |
| 12 | +0.80 | +0.40 | 0.877 | 1.000 | 0.605 | 0.473 | 0.6697 | Archetype F | 14.4 |
| 13 | +0.70 | +0.50 | 0.930 | 0.865 | 0.620 | 0.467 | 0.6722 | Archetype F | 13.1 |
| 14 | +0.50 | +0.80 | 0.911 | 1.000 | 0.942 | 0.333 | 0.6750 | Archetype A | 9.8 |
| 15 | -0.60 | +0.50 | 0.757 | 0.867 | 0.571 | 0.550 | 0.6803 | Archetype C | 16.3 |
| 16 | -0.50 | +0.60 | 0.790 | 0.867 | 0.543 | 0.551 | 0.6867 | Archetype E | 16.1 |
| 17 | -0.40 | +0.70 | 0.814 | 1.000 | 0.498 | 0.564 | 0.6904 | Archetype B | 16.5 |
| 18 | +0.50 | +0.60 | 0.996 | 0.867 | 0.470 | 0.573 | 0.6938 | Archetype G | 16.1 |
| 19 | +0.40 | +0.70 | 0.950 | 1.000 | 0.473 | 0.547 | 0.6969 | Archetype A | 16.1 |
| 20 | +0.40 | +0.90 | 0.879 | 0.638 | 1.000 | 0.373 | 0.7342 | Archetype A | 11.1 |
| 21 | +0.70 | +0.40 | 0.935 | 1.000 | 0.384 | 0.561 | 0.7593 | Archetype F | 17.1 |
| 22 | -0.60 | +0.40 | 0.763 | 1.000 | 0.331 | 0.661 | 0.7863 | Archetype C | 19.8 |
| 23 | +0.30 | +0.70 | 0.889 | 1.000 | 0.332 | 0.590 | 0.7918 | Archetype D | 18.1 |
| 24 | -0.70 | +0.30 | 0.758 | 1.000 | 0.355 | 0.610 | 0.7919 | Archetype H | 19.1 |
| 25 | +0.50 | +0.50 | 0.994 | 0.863 | 0.292 | 0.670 | 0.7930 | Archetype I | 19.9 |
| 26 | +0.40 | +0.60 | 0.975 | 0.810 | 0.301 | 0.642 | 0.8081 | Archetype G | 19.2 |
| 27 | -0.30 | +0.70 | 0.795 | 1.000 | 0.316 | 0.618 | 0.8098 | Archetype B | 18.4 |
| 28 | +0.20 | +0.70 | 0.875 | 0.908 | 0.294 | 0.631 | 0.8112 | Archetype D | 19.1 |
| 29 | -0.40 | +0.60 | 0.815 | 0.924 | 0.297 | 0.645 | 0.8128 | Archetype B | 19.3 |
| 30 | +0.10 | +0.70 | 0.873 | 0.756 | 0.296 | 0.675 | 0.8233 | Archetype D | 19.6 |
| 31 | +0.00 | +0.70 | 0.884 | 0.742 | 0.294 | 0.682 | 0.8243 | Archetype J | 19.6 |
| 32 | +0.70 | +0.30 | 0.897 | 1.000 | 0.246 | 0.614 | 0.8531 | Archetype F | 19.5 |
| 33 | -0.50 | +0.50 | 0.797 | 0.521 | 0.325 | 0.669 | 0.9144 | Archetype E | 19.6 |
| 34 | -0.60 | +0.30 | 0.759 | 0.595 | 0.193 | 0.692 | 0.9838 | Archetype H | 22.1 |
| 35 | +0.70 | +0.10 | 0.721 | 0.619 | 0.158 | 0.691 | 1.0139 | Archetype K | 21.4 |
| 36 | +0.60 | +0.30 | 0.928 | 0.448 | 0.163 | 0.728 | 1.0415 | Archetype F | 22.9 |
| 37 | +0.20 | -0.80 | 0.235 | 1.000 | 0.436 | 0.574 | 1.0416 | Archetype L | 17.6 |
| 38 | +0.60 | -0.40 | 0.527 | 0.827 | 0.121 | 0.692 | 1.0583 | Archetype M | 22.0 |
| 39 | +0.60 | +0.60 | 0.946 | 0.141 | 0.676 | 0.460 | 1.0663 | Archetype G | 12.3 |
| 40 | -0.60 | +0.60 | 0.732 | 0.141 | 0.854 | 0.439 | 1.0698 | Archetype E | 12.1 |
| 41 | +0.50 | -0.50 | 0.637 | 0.502 | 0.166 | 0.701 | 1.0789 | Archetype M | 22.0 |
| 42 | -0.60 | -0.20 | 0.706 | 0.487 | 0.123 | 0.765 | 1.0836 | Archetype N | 24.4 |
| 43 | -0.20 | +0.60 | 0.828 | 0.392 | 0.150 | 0.737 | 1.0919 | Archetype B | 22.9 |
| 44 | +0.20 | -0.70 | 0.332 | 0.807 | 0.205 | 0.680 | 1.1037 | Archetype L | 21.6 |
| 45 | -0.30 | +0.60 | 0.808 | 0.345 | 0.179 | 0.716 | 1.1044 | Archetype B | 21.5 |
| 46 | +0.50 | +0.40 | 0.980 | 0.277 | 0.193 | 0.735 | 1.1159 | Archetype I | 23.5 |
| 47 | +0.20 | +0.60 | 0.909 | 0.294 | 0.177 | 0.737 | 1.1201 | Archetype O | 22.9 |
| 48 | -0.10 | +0.40 | 0.957 | 0.322 | 0.097 | 0.953 | 1.1313 | Archetype P | 34.6 |
| 49 | +0.30 | +0.30 | 0.992 | 0.271 | 0.116 | 0.962 | 1.1465 | Archetype O | 35.4 |
| 50 | +0.10 | -0.80 | 0.101 | 0.846 | 0.438 | 0.589 | 1.1476 | Archetype L | 17.5 |
| 51 | -0.60 | +0.10 | 0.802 | 0.325 | 0.114 | 0.781 | 1.1525 | Archetype H | 24.4 |
| 52 | +0.30 | +0.60 | 0.932 | 0.215 | 0.212 | 0.698 | 1.1549 | Archetype O | 21.4 |
| 53 | +0.10 | -0.70 | 0.215 | 0.964 | 0.199 | 0.714 | 1.1586 | Archetype L | 22.4 |
| 54 | -0.50 | -0.50 | 0.249 | 0.962 | 0.131 | 0.708 | 1.1860 | Archetype N | 21.5 |
| 55 | +0.00 | +0.60 | 0.927 | 0.172 | 0.157 | 0.773 | 1.2055 | Archetype J | 24.2 |
| 56 | +0.40 | +0.50 | 0.990 | 0.108 | 0.213 | 0.735 | 1.2182 | Archetype I | 22.9 |
| 57 | -0.40 | +0.00 | 0.851 | 0.228 | 0.042 | 0.975 | 1.2393 | Archetype Q | 36.4 |
| 58 | -0.50 | -0.40 | 0.405 | 0.478 | 0.071 | 0.769 | 1.2419 | Archetype N | 24.4 |
| 59 | +0.40 | -0.50 | 0.691 | 0.204 | 0.118 | 0.775 | 1.2479 | Archetype M | 25.3 |
| 60 | +0.40 | +0.30 | 0.982 | 0.082 | 0.126 | 0.886 | 1.2722 | Archetype I | 31.1 |
| 61 | -0.50 | +0.30 | 0.775 | 0.123 | 0.116 | 0.801 | 1.2811 | Archetype H | 25.5 |
| 62 | -0.10 | +0.20 | 1.000 | 0.121 | 0.056 | 0.994 | 1.2899 | Archetype P | 46.2 |
| 63 | +0.20 | +0.40 | 0.948 | 0.047 | 0.132 | 0.919 | 1.2928 | Archetype O | 33.2 |
| 64 | +0.20 | +0.30 | 0.974 | 0.066 | 0.104 | 0.979 | 1.2949 | Archetype O | 39.0 |
| 65 | +0.00 | -0.70 | 0.099 | 0.673 | 0.172 | 0.724 | 1.2964 | Archetype L | 22.5 |
| 66 | +0.10 | +0.20 | 1.000 | 0.098 | 0.062 | 0.997 | 1.3013 | Archetype P | 46.7 |
| 67 | +0.30 | +0.00 | 0.906 | 0.169 | 0.000 | 0.997 | 1.3036 | Archetype R | 43.2 |
| 68 | +0.00 | -0.80 | 0.002 | 0.545 | 0.410 | 0.589 | 1.3114 | Archetype L | 17.3 |
| 69 | -0.10 | -0.10 | 0.771 | 0.131 | 0.026 | 0.997 | 1.3249 | Archetype S | 51.1 |
| 70 | +0.10 | +0.50 | 0.956 | 0.010 | 0.130 | 0.856 | 1.3270 | Archetype O | 29.0 |
| 71 | +0.00 | +0.20 | 0.994 | 0.028 | 0.061 | 1.000 | 1.3516 | Archetype P | 47.6 |
| 72 | +0.10 | -0.10 | 0.875 | 0.063 | 0.019 | 0.997 | 1.3620 | Archetype S | 51.5 |
| 73 | -0.30 | -0.10 | 0.764 | 0.049 | 0.031 | 1.000 | 1.3778 | Archetype Q | 42.2 |
| 74 | +0.10 | -0.40 | 0.537 | 0.078 | 0.057 | 0.999 | 1.3977 | Archetype T | 38.4 |
| 75 | -0.10 | -0.40 | 0.334 | 0.136 | 0.003 | 0.989 | 1.4773 | Archetype T | 37.4 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 20 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 6 | (+0.40, +0.80) | P_boundary 0.91, W 1.00, G 0.75, S 0.45 | 0.6051 |
| Archetype B | 6 | (-0.40, +0.80) | P_boundary 0.80, W 1.00, G 0.80, S 0.46 | 0.6113 |
| Archetype C | 5 | (-0.80, +0.40) | P_boundary 0.71, W 1.00, G 0.89, S 0.47 | 0.6159 |
| Archetype D | 6 | (+0.30, +0.80) | P_boundary 0.87, W 1.00, G 0.61, S 0.53 | 0.6268 |
| Archetype E | 4 | (-0.50, +0.70) | P_boundary 0.78, W 0.86, G 0.81, S 0.42 | 0.6595 |
| Archetype F | 5 | (+0.80, +0.40) | P_boundary 0.88, W 1.00, G 0.60, S 0.47 | 0.6697 |
| Archetype G | 3 | (+0.50, +0.60) | P_boundary 1.00, W 0.87, G 0.47, S 0.57 | 0.6938 |
| Archetype H | 4 | (-0.70, +0.30) | P_boundary 0.76, W 1.00, G 0.35, S 0.61 | 0.7919 |
| Archetype I | 4 | (+0.50, +0.50) | P_boundary 0.99, W 0.86, G 0.29, S 0.67 | 0.7930 |
| Archetype J | 2 | (+0.00, +0.70) | P_boundary 0.88, W 0.74, G 0.29, S 0.68 | 0.8243 |
| Archetype K | 1 | (+0.70, +0.10) | P_boundary 0.72, W 0.62, G 0.16, S 0.69 | 1.0139 |
| Archetype L | 6 | (+0.20, -0.80) | P_boundary 0.23, W 1.00, G 0.44, S 0.57 | 1.0416 |
| Archetype M | 3 | (+0.60, -0.40) | P_boundary 0.53, W 0.83, G 0.12, S 0.69 | 1.0583 |
| Archetype N | 3 | (-0.60, -0.20) | P_boundary 0.71, W 0.49, G 0.12, S 0.76 | 1.0836 |
| Archetype O | 6 | (+0.20, +0.60) | P_boundary 0.91, W 0.29, G 0.18, S 0.74 | 1.1201 |
| Archetype P | 4 | (-0.10, +0.40) | P_boundary 0.96, W 0.32, G 0.10, S 0.95 | 1.1313 |
| Archetype Q | 2 | (-0.40, +0.00) | P_boundary 0.85, W 0.23, G 0.04, S 0.97 | 1.2393 |
| Archetype R | 1 | (+0.30, +0.00) | P_boundary 0.91, W 0.17, G 0.00, S 1.00 | 1.3036 |
| Archetype S | 2 | (-0.10, -0.10) | P_boundary 0.77, W 0.13, G 0.03, S 1.00 | 1.3249 |
| Archetype T | 2 | (+0.10, -0.40) | P_boundary 0.54, W 0.08, G 0.06, S 1.00 | 1.3977 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (+0.40, +0.80)

**6** points in this archetype. Objectives: **P_boundary 0.914**, **W 1.000**, **G 0.755**, **S 0.454**; distance to ideal `0.6051`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| zero-shot sim-to-real transfer · success rate | 0.374 | 0.031 | 0.004 |
| successor measures · forward-backward representatio… | 0.182 | 0.374 | 0.302 |
| language models · reward | 0.147 | 0.146 | 0.211 |
| llms · capabilities | 0.118 | 0.017 | 0.013 |
| control · planning | 0.180 | 0.259 | 0.298 |

Mixture entropy `H = 0.944`; effective contributing documents `N_eff = 13.6`; top-3 weight concentration `T3 = 0.358`; 5th-neighbour distance `d5 = 0.403`; relative uncertainty `u = 0.0075` (RBF surrogate; Matern `0.1289`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 39` | 0.1328 | language models · reward | 2024 | harvested | Proto Successor Measure: Representing the Behavior Space of an RL Agent |
| `doc 15` | 0.1253 | language models · reward | 2025 | harvested | Direct Preference Optimization for Speech Autoregressive Diffusion Models |
| `doc 42` | 0.1004 | language models · reward | 2026 | harvested | From Evaluation to Optimisation: Hierarchy-Aware Training Signals for CWE Prediction in Python |
| `doc 54` | 0.0877 | language models · reward | 2024 | harvested | Learning to Describe for Predicting Zero-shot Drug-Drug Interactions |
| `doc 4` | 0.0714 | language models · reward | 2022 | harvested | Multimodal Knowledge Alignment with Reinforcement Learning |

**L2 theme lens:** `representation` (df=95), `representation-learning` (df=2), `reinforcement-learning` (df=3), `reinforcement-learningbased` (df=2), `generalization` (df=83), `optimal` (df=87), `representation-based` (df=2), `specific` (df=93), `reward-maximizing` (df=4), `transformer` (df=52), `maximizing` (df=34), `conditions` (df=87)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #2 at (-0.40, +0.80)

**6** points in this archetype. Objectives: **P_boundary 0.801**, **W 1.000**, **G 0.797**, **S 0.459**; distance to ideal `0.6113`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| zero-shot sim-to-real transfer · success rate | 0.166 | 0.004 | 0.000 |
| successor measures · forward-backward representatio… | 0.004 | 0.156 | 0.134 |
| language models · reward | 0.406 | 0.359 | 0.319 |
| llms · capabilities | 0.030 | 0.242 | 0.263 |
| control · planning | 0.394 | 0.002 | 0.038 |

Mixture entropy `H = 0.722`; effective contributing documents `N_eff = 13.8`; top-3 weight concentration `T3 = 0.347`; 5th-neighbour distance `d5 = 0.410`; relative uncertainty `u = 0.0082` (RBF surrogate; Matern `0.1305`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 100` | 0.1356 | llms · capabilities | 2025 | harvested | Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts |
| `doc 57` | 0.1184 | llms · capabilities | 2026 | harvested | Selective Test-Time Debiasing for CLIP via Reward Gating |
| `doc 21` | 0.0926 | language models · reward | 2023 | harvested | Test-Time Adaptation with CLIP Reward for Zero-Shot Generalization in Vision-Language Models |
| `doc 35` | 0.0842 | llms · capabilities | 2026 | harvested | VoiceTTA: Enhancing Zero-Shot Text-to-Speech via Reinforcement Learning-Based Test-Time Adaptation |
| `doc 14` | 0.0700 | llms · capabilities | 2026 | harvested | Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP |

**L2 theme lens:** `reinforcement-learningbased` (df=2), `reinforcement-learning` (df=3), `representation` (df=95), `representation-learning` (df=2), `generalization` (df=83), `specific` (df=93), `instruction-following` (df=2), `representation-based` (df=2), `classification` (df=21), `optimal` (df=87), `personality` (df=3), `transformer` (df=52)

**L1 concentration lens:** `fine-tuning` (df=54), `fine-tuning-based` (df=1)

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype C — gap #3 at (-0.80, +0.40)

**5** points in this archetype. Objectives: **P_boundary 0.706**, **W 1.000**, **G 0.887**, **S 0.471**; distance to ideal `0.6159`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| zero-shot sim-to-real transfer · success rate | 0.124 | 0.000 | 0.000 |
| successor measures · forward-backward representatio… | 0.000 | 0.008 | 0.008 |
| language models · reward | 0.363 | 0.193 | 0.127 |
| llms · capabilities | 0.022 | 0.290 | 0.371 |
| control · planning | 0.490 | 0.055 | 0.085 |

Mixture entropy `H = 0.661`; effective contributing documents `N_eff = 14.2`; top-3 weight concentration `T3 = 0.339`; 5th-neighbour distance `d5 = 0.401`; relative uncertainty `u = 0.0080` (RBF surrogate; Matern `0.1317`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 100` | 0.1389 | llms · capabilities | 2025 | harvested | Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts |
| `doc 88` | 0.1163 | llms · capabilities | 2023 | harvested | Language-Conditioned Semantic Search-Based Policy for Robotic Manipulation Tasks |
| `doc 50` | 0.0841 | llms · capabilities | 2025 | harvested | YingMusic-SVC: Real-World Robust Zero-Shot Singing Voice Conversion with Flow-GRPO and Singing-Specific Inductive Biases |
| `doc 35` | 0.0834 | llms · capabilities | 2026 | harvested | VoiceTTA: Enhancing Zero-Shot Text-to-Speech via Reinforcement Learning-Based Test-Time Adaptation |
| `doc 93` | 0.0758 | llms · capabilities | 2024 | harvested | OpenFMNav: Towards Open-Set Zero-Shot Object Navigation via Vision-Language Foundation Models |

**L2 theme lens:** `representation` (df=95), `representation-learning` (df=2), `reinforcement-learningbased` (df=2), `reinforcement-learning` (df=3), `representation-based` (df=2), `generalization` (df=83), `segmentation` (df=26), `specific` (df=93), `instruction-following` (df=2), `transformer` (df=52), `augmentation` (df=43), `conditions` (df=87)

**L1 concentration lens:** `flow-matching-based` (df=1), `reasoning-based` (df=1)

Terms present in **both** lenses (strongest evidence): *(none)*

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (+0.40, +0.80)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `representation` | 0.49634 | 95 | theme |
| L2 theme | `representation-learning` | 0.46944 | 2 | theme |
| L2 theme | `reinforcement-learning` | 0.44890 | 3 | theme |
| L2 theme | `reinforcement-learningbased` | 0.44604 | 2 | theme |
| L2 theme | `generalization` | 0.40353 | 83 | theme |
| L2 theme | `optimal` | 0.39025 | 87 | theme |
| L2 theme | `representation-based` | 0.37037 | 2 | theme |
| L2 theme | `specific` | 0.35532 | 93 | theme |
| L2 theme | `reward-maximizing` | 0.33784 | 4 | theme |
| L2 theme | `transformer` | 0.33550 | 52 | theme |
| L2 theme | `maximizing` | 0.32613 | 34 | theme |
| L2 theme | `conditions` | 0.32295 | 87 | theme |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

_None at gap #1._

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (+0.40, +0.80) | *(none)* | 0 |
| Archetype B | (-0.40, +0.80) | `fine-tuning` (df=54) | 1 |
| Archetype C | (-0.80, +0.40) | *(none)* | 2 |

## 6. Distance-maximising baseline (Table 4 analogue)

The negative control of Obayashi et al. (2026) Sec. 8.8: pick the 16 grid points with the largest nearest-neighbour distance, once over the whole grid and once over the interior only (excluding `x = ±1` and `y = ±1`), then compare their mean scores with the practical front. An unconstrained novelty search maximises emptiness, not discovery value.

| Selection | d_nn | S | R | P_boundary | G |
| --- | --- | --- | --- | --- | --- |
| **Practical Pareto front (this report)** | 0.2601 | 0.6873 | 0.8365 | 0.7779 | 0.3325 |
| Distance-maximising, all grid | 0.4402 | 0.0581 | 0.0258 | 0.1567 | 0.2500 |
| Distance-maximising, interior only | 0.3401 | 0.2878 | 0.2975 | 0.4020 | 0.6481 |

All 16 unconstrained picks sit on the chart edge, and 0 of the 16 interior-only picks do. Restricting the search to the 361 interior points recovers support (`S` 0.058 -> 0.288) but still trails the practical front, which is the paper's point: distance alone is not a discovery criterion. Two caveats when reading this table. The distance column is the *fifth*-neighbour distance used throughout this report, not the nearest-neighbour distance of the paper's Table 4, so the absolute numbers are not comparable across the two studies. And `R` is reported as a diagnostic only: it rewards low surrogate uncertainty (Sec. 3.7), which is why it is excluded from the objectives. The front scoring high on `R` therefore says the selected locations sit in well-sampled regions of the surrogate, not that the front was optimised for it.

## 7. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (+0.40, +0.80) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_+0.40_+0.80.json`, as produced by `code/make_evidence.py`. 
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

--- attached: evidence_point_+0.40_+0.80.json ---
{
  "location": {
    "x": 0.4,
    "y": 0.8
  },
  "theme_lens_L2": [
    {
      "term": "representation",
      "weight": 0.49634,
      "df": 95,
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
      "weight": 0.46944,
      "df": 2,
      "word_forms": [
        "representation-learning"
      ]
    },
    {
      "term": "reinforcement-learning",
      "weight": 0.4489,
      "df": 3,
      "word_forms": [
        "reinforcement-learned",
        "reinforcement-learning"
      ]
    },
    {
      "term": "reinforcement-learningbased",
      "weight": 0.44604,
      "df": 2,
      "word_forms": [
        "reinforcement-learningbased"
      ]
    },
    {
      "term": "generalization",
      "weight": 0.40353,
      "df": 83,
      "word_forms": [
        "generalisation",
        "generalising",
        "generality",
        "generalization",
        "generalizing"
      ]
    },
    {
      "term": "optimal",
      "weight": 0.39025,
      "df": 87,
      "word_forms": [
        "optim",
        "optimal",
        "optimally",
        "optimisation",
        "optimising",
        "optimization",
        "optimizing"
      ]
    },
    {
      "term": "representation-based",
      "weight": 0.37037,
      "df": 2,
      "word_forms": [
        "representation-based"
      ]
    },
    {
      "term": "specific",
      "weight": 0.35532,
      "df": 93,
      "word_forms": [
        "specific",
        "specifically",
        "specification",
        "specifications",
        "specificity",
        "specifics"
      ]
    },
    {
      "term": "reward-maximizing",
      "weight": 0.33784,
      "df": 4,
      "word_forms": [
        "reward-maximising",
        "reward-maximization",
        "reward-maximizing"
      ]
    },
    {
      "term": "transformer",
      "weight": 0.3355,
      "df": 52,
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
      "term": "maximizing",
      "weight": 0.32613,
      "df": 34,
      "word_forms": [
        "maximal",
        "maximally",
        "maximisation",
        "maximising",
        "maximization",
        "maximizing"
      ]
    },
    {
      "term": "conditions",
      "weight": 0.32295,
      "df": 87,
      "word_forms": [
        "condition",
        "conditional",
        "conditionally",
        "conditioned",
        "conditioner",
        "conditioning",
        "conditions"
      ]
    }
  ],
  "concentration_lens_L1": [],
  "contributing_documents_top3": [
    {
      "doc": "doc 39",
      "source": "https://arxiv.org/abs/2411.19418v2",
      "is_seed": false,
      "title": "Proto Successor Measure: Representing the Behavior Space of an RL Agent",
      "authors": [
        "Siddhant Agarwal",
        "Harshit Sikchi",
        "Peter Stone"
      ],
      "year": 2024,
      "regime": "language models · reward",
      "weight": 0.1328
    },
    {
      "doc": "doc 15",
      "source": "https://arxiv.org/abs/2509.18928v1",
      "is_seed": false,
      "title": "Direct Preference Optimization for Speech Autoregressive Diffusion Models",
      "authors": [
        "Zhijun Liu",
        "Dongya Jia",
        "Xiaoqiang Wang"
      ],
      "year": 2025,
      "regime": "language models · reward",
      "weight": 0.1253
    },
    {
      "doc": "doc 42",
      "source": "https://arxiv.org/abs/2607.21069v1",
      "is_seed": false,
      "title": "From Evaluation to Optimisation: Hierarchy-Aware Training Signals for CWE Prediction in Python",
      "authors": [
        "Muntasir Adnan",
        "Manile Srun",
        "Carlos C. N. Kuhn"
      ],
      "year": 2026,
      "regime": "language models · reward",
      "weight": 0.1004
    }
  ],
  "mixture_entropy_H": 0.9444,
  "effective_contributing_documents": 13.59,
  "relative_uncertainty_u": 0.00752
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

- Novelty is **corpus-relative**. These are gaps in a 100-paper arXiv sample, not proof that the topics are unexplored in the wider literature.
- The vocabulary and the term statistics come from **full text** extracted from each paper's PDF, not from abstracts. Section boundaries are recovered by heading detection, so a paper whose headings the PDF text layer mangles contributes a slightly different term distribution than its author intended.
- The corpus is founded on 0 user-curated seeding papers plus a per-regime harvest quota. A regime whose query returns few candidates fills its quota from the top results by relevance and has no headroom beyond review rejection.
- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately **not** a Pareto objective: maximizing low uncertainty favours already-dense regions.

