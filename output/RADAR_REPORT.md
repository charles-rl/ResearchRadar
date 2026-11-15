# ResearchRadar — Supported Knowledge Gaps in Robot Learning

Automated Knowledge Manifold Optimization (KMO) over a 120-document corpus spanning conflicting robotics regimes: ZeroShot-Rep, Sim2Real-Physics, Arm-Manipulation. The corpus is founded on user-curated seeding papers and expanded by a recency-bounded arXiv harvest.

Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. Corpus `120` documents (15 seeding + 105 harvested; ZeroShot-Rep 41, Sim2Real-Physics 40, Arm-Manipulation 39), represented as within-word character 4-7-gram TF-IDF (paper Sec. 5.2); no language model participates in any objective (120 documents on the chart).

All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.

## 1. Knowledge Manifold

| Quantity | Value |
| --- | --- |
| Chart | Ω = [−1, 1]², fixed |
| Documents | 120 |
| Corpus composition | 15 seeding papers (`doc 1`-`doc 15`) + 105 harvested (ZeroShot-Rep 41, Sim2Real-Physics 40, Arm-Manipulation 39) |
| Harvest window | arXiv `submittedDate` from 2019 (= earliest seed year minus the offset, floored by the lookback limit) |
| Gauge-fixing anchors | 9 (8 perimeter + 1 central) |
| Stress minimization | metric stress 0.14, Spearman(distance) 0.6728 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.95 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0479, q60 0.0876, q85 0.1281, q98 0.3030 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **34 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **28 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | +0.10 | +0.50 | 0.891 | 1.000 | 0.835 | 0.946 | 0.2054 | Archetype A | 64.3 |
| 2 | +0.20 | +0.50 | 0.927 | 0.974 | 0.824 | 0.853 | 0.2421 | Archetype A | 64.6 |
| 3 | +0.30 | +0.50 | 0.905 | 1.000 | 0.768 | 0.894 | 0.2721 | Archetype A | 64.2 |
| 4 | +0.20 | -0.40 | 0.776 | 0.974 | 0.948 | 0.741 | 0.3472 | Archetype B | 60.3 |
| 5 | +0.20 | +0.30 | 0.995 | 1.000 | 1.000 | 0.638 | 0.3625 | Archetype C | 61.8 |
| 6 | +0.20 | +0.60 | 0.898 | 1.000 | 0.657 | 0.921 | 0.3658 | Archetype A | 63.9 |
| 7 | +0.10 | -0.40 | 0.762 | 0.845 | 1.000 | 0.715 | 0.4023 | Archetype B | 59.4 |
| 8 | +0.30 | -0.50 | 0.963 | 0.935 | 0.683 | 0.751 | 0.4098 | Archetype D | 59.7 |
| 9 | +0.00 | +0.50 | 0.769 | 0.713 | 0.812 | 0.955 | 0.4158 | Archetype A | 63.2 |
| 10 | +0.30 | -0.20 | 0.871 | 0.697 | 0.977 | 0.718 | 0.4346 | Archetype B | 60.9 |
| 11 | +0.00 | +0.40 | 0.789 | 0.732 | 1.000 | 0.714 | 0.4451 | Archetype A | 61.8 |
| 12 | +0.20 | +0.40 | 0.946 | 0.587 | 1.000 | 0.822 | 0.4526 | Archetype A | 64.0 |
| 13 | +0.40 | +0.60 | 0.920 | 1.000 | 0.562 | 0.880 | 0.4608 | Archetype E | 62.4 |
| 14 | -0.20 | +0.40 | 0.637 | 1.000 | 0.862 | 0.734 | 0.4704 | Archetype F | 60.3 |
| 15 | +0.30 | +0.30 | 0.642 | 0.901 | 1.000 | 0.694 | 0.4819 | Archetype C | 62.8 |
| 16 | +0.20 | +0.70 | 0.832 | 1.000 | 0.520 | 0.964 | 0.5097 | Archetype E | 62.6 |
| 17 | -0.30 | +0.30 | 0.594 | 1.000 | 0.926 | 0.674 | 0.5260 | Archetype F | 59.2 |
| 18 | +0.30 | -0.60 | 0.947 | 1.000 | 0.615 | 0.642 | 0.5284 | Archetype D | 58.3 |
| 19 | +0.40 | +0.70 | 0.912 | 0.878 | 0.463 | 0.938 | 0.5613 | Archetype E | 61.2 |
| 20 | -0.20 | -0.20 | 0.453 | 0.904 | 0.871 | 0.805 | 0.6024 | Archetype G | 61.2 |
| 21 | +0.20 | +0.80 | 0.788 | 0.552 | 0.406 | 0.972 | 0.7741 | Archetype E | 61.1 |
| 22 | -0.40 | -0.50 | 0.373 | 1.000 | 0.296 | 0.970 | 0.9434 | Archetype H | 62.7 |
| 23 | -0.40 | -0.10 | 0.087 | 0.766 | 0.648 | 0.954 | 1.0072 | Archetype G | 62.5 |
| 24 | -0.20 | -0.40 | 0.042 | 0.908 | 0.633 | 1.000 | 1.0303 | Archetype H | 63.3 |
| 25 | -0.40 | -0.20 | 0.069 | 0.554 | 0.546 | 1.000 | 1.1275 | Archetype G | 63.9 |
| 26 | -0.30 | -0.30 | 0.085 | 0.457 | 0.589 | 0.964 | 1.1411 | Archetype G | 64.2 |
| 27 | -0.20 | -0.50 | 0.169 | 0.277 | 0.525 | 0.969 | 1.1997 | Archetype H | 62.4 |
| 28 | -0.60 | -0.30 | 0.044 | 1.000 | 0.257 | 1.000 | 1.2108 | Archetype I | 62.3 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 9 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 7 | (+0.10, +0.50) | P_boundary 0.89, W 1.00, G 0.83, S 0.95 | 0.2054 |
| Archetype B | 3 | (+0.20, -0.40) | P_boundary 0.78, W 0.97, G 0.95, S 0.74 | 0.3472 |
| Archetype C | 2 | (+0.20, +0.30) | P_boundary 1.00, W 1.00, G 1.00, S 0.64 | 0.3625 |
| Archetype D | 2 | (+0.30, -0.50) | P_boundary 0.96, W 0.94, G 0.68, S 0.75 | 0.4098 |
| Archetype E | 4 | (+0.40, +0.60) | P_boundary 0.92, W 1.00, G 0.56, S 0.88 | 0.4608 |
| Archetype F | 2 | (-0.20, +0.40) | P_boundary 0.64, W 1.00, G 0.86, S 0.73 | 0.4704 |
| Archetype G | 4 | (-0.20, -0.20) | P_boundary 0.45, W 0.90, G 0.87, S 0.80 | 0.6024 |
| Archetype H | 3 | (-0.40, -0.50) | P_boundary 0.37, W 1.00, G 0.30, S 0.97 | 0.9434 |
| Archetype I | 1 | (-0.60, -0.30) | P_boundary 0.04, W 1.00, G 0.26, S 1.00 | 1.2108 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (+0.10, +0.50)

**7** points in this archetype. Objectives: **P_boundary 0.891**, **W 1.000**, **G 0.835**, **S 0.946**; distance to ideal `0.2054`.

Eq. 6 regime mixture over the **named corpus regions** (the map colours), global and adaptive weights:

| Corpus region | global | adaptive |
| --- | --- | --- |
| ZeroShot-Rep | 0.306 | 0.000 |
| Sim2Real-Physics | 0.504 | 0.858 |
| Arm-Manipulation | 0.190 | 0.142 |

Eq. 6 cluster mixture at the two resolutions the boundary objective is computed from (Sec. 3.2). Cluster `c` at K=5 is unrelated to cluster `c` at K=8, so the two blocks are never read across:

##### K=5 Discovery level

| Cluster | Documents | global |
| --- | --- | --- |
| K5·1 reinforcement learning framework · domain randomiza… | 25 | 0.398 |
| K5·2 dexterous manipulation · human demonstrations | 32 | 0.196 |
| K5·3 successor features · reward functions | 22 | 0.136 |
| K5·4 latent dynamics · foundation models | 19 | 0.190 |
| K5·5 real world · vision-language-action vla | 22 | 0.080 |

##### K=8 Reference level

| Cluster | Documents | global | adaptive |
| --- | --- | --- | --- |
| K8·1 Region 8 | 1 | 0.007 | 0.000 |
| K8·2 latent dynamics · models | 11 | 0.090 | 0.039 |
| K8·3 zero-shot sim-to-real transfer · contact-rich manip… | 22 | 0.153 | 0.000 |
| K8·4 human videos · real world | 33 | 0.241 | 0.416 |
| K8·5 behavioral foundation · test time | 13 | 0.062 | 0.000 |
| K8·6 reinforcement learning systems · successor features | 21 | 0.206 | 0.126 |
| K8·7 domain randomization · bipedal | 14 | 0.240 | 0.418 |
| K8·8 hand · robotic | 5 | 0.001 | 0.000 |


Mixture entropy `H = 0.917`; effective contributing documents `N_eff = 64.3`; top-3 weight concentration `T3 = 0.078`; 5th-neighbour distance `d5 = 0.213`; relative uncertainty `u = 0.4700` (RBF surrogate; Matern `0.4510`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 2` | 0.0266 | Sim2Real-Physics | 2021 | seed | Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning |
| `doc 81` | 0.0262 | Sim2Real-Physics | 2026 | harvested | Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks |
| `doc 14` | 0.0256 | Sim2Real-Physics | 2025 | seed | Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning |
| `doc 3` | 0.0249 | Arm-Manipulation | 2024 | seed | ASID: ACTIVE EXPLORATION FOR SYSTEM IDENTIFICATION IN ROBOTIC MANIPULATION |
| `doc 76` | 0.0248 | Sim2Real-Physics | 2024 | harvested | Robust Contact-rich Manipulation through Implicit Motor Adaptation |

**L2 theme lens:** `representation` (df=115), `representation-learning` (df=4), `reinforcement-learning` (df=3), `state-representation` (df=2), `generalization` (df=86), `parameterized` (df=51), `generalize` (df=90), `observations` (df=117), `simulation-to-reality` (df=3), `optimization` (df=104), `parameter-conditioned` (df=3), `demonstration-based` (df=2)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #4 at (+0.20, -0.40)

**3** points in this archetype. Objectives: **P_boundary 0.776**, **W 0.974**, **G 0.948**, **S 0.741**; distance to ideal `0.3472`.

Eq. 6 regime mixture over the **named corpus regions** (the map colours), global and adaptive weights:

| Corpus region | global | adaptive |
| --- | --- | --- |
| ZeroShot-Rep | 0.446 | 0.383 |
| Sim2Real-Physics | 0.071 | 0.000 |
| Arm-Manipulation | 0.483 | 0.617 |

Eq. 6 cluster mixture at the two resolutions the boundary objective is computed from (Sec. 3.2). Cluster `c` at K=5 is unrelated to cluster `c` at K=8, so the two blocks are never read across:

##### K=5 Discovery level

| Cluster | Documents | global |
| --- | --- | --- |
| K5·1 reinforcement learning framework · domain randomiza… | 25 | 0.039 |
| K5·2 dexterous manipulation · human demonstrations | 32 | 0.194 |
| K5·3 successor features · reward functions | 22 | 0.211 |
| K5·4 latent dynamics · foundation models | 19 | 0.244 |
| K5·5 real world · vision-language-action vla | 22 | 0.312 |

##### K=8 Reference level

| Cluster | Documents | global | adaptive |
| --- | --- | --- | --- |
| K8·1 Region 8 | 1 | 0.000 | 0.000 |
| K8·2 latent dynamics · models | 11 | 0.162 | 0.740 |
| K8·3 zero-shot sim-to-real transfer · contact-rich manip… | 22 | 0.105 | 0.001 |
| K8·4 human videos · real world | 33 | 0.322 | 0.081 |
| K8·5 behavioral foundation · test time | 13 | 0.229 | 0.179 |
| K8·6 reinforcement learning systems · successor features | 21 | 0.132 | 0.000 |
| K8·7 domain randomization · bipedal | 14 | 0.017 | 0.000 |
| K8·8 hand · robotic | 5 | 0.034 | 0.000 |


Mixture entropy `H = 0.920`; effective contributing documents `N_eff = 60.3`; top-3 weight concentration `T3 = 0.083`; 5th-neighbour distance `d5 = 0.243`; relative uncertainty `u = 0.4704` (RBF surrogate; Matern `0.4521`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 98` | 0.0286 | Arm-Manipulation | 2025 | harvested | World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation |
| `doc 44` | 0.0275 | ZeroShot-Rep | 2025 | harvested | Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models |
| `doc 15` | 0.0272 | ZeroShot-Rep | 2025 | seed | ZERO-SHOT WHOLE-BODY HUMANOID CONTROL VIA BEHAVIORAL FOUNDATION MODELS |
| `doc 111` | 0.0265 | Arm-Manipulation | 2025 | harvested | Human-in-the-loop Online Rejection Sampling for Robotic Manipulation |
| `doc 88` | 0.0255 | Arm-Manipulation | 2022 | harvested | Dexterous Imitation Made Easy: A Learning-Based Framework for Efficient Dexterous Manipulation |

**L2 theme lens:** `representation-learning` (df=4), `representation` (df=115), `state-representation` (df=2), `reinforcement-learning` (df=3), `demonstration-based` (df=2), `generalization` (df=86), `action-conditional` (df=4), `imitation-learning` (df=5), `generalize` (df=90), `observations` (df=117), `optimization` (df=104), `fb-representation` (df=2)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype C — gap #5 at (+0.20, +0.30)

**2** points in this archetype. Objectives: **P_boundary 0.995**, **W 1.000**, **G 1.000**, **S 0.638**; distance to ideal `0.3625`.

Eq. 6 regime mixture over the **named corpus regions** (the map colours), global and adaptive weights:

| Corpus region | global | adaptive |
| --- | --- | --- |
| ZeroShot-Rep | 0.406 | 0.050 |
| Sim2Real-Physics | 0.356 | 0.181 |
| Arm-Manipulation | 0.238 | 0.769 |

Eq. 6 cluster mixture at the two resolutions the boundary objective is computed from (Sec. 3.2). Cluster `c` at K=5 is unrelated to cluster `c` at K=8, so the two blocks are never read across:

##### K=5 Discovery level

| Cluster | Documents | global |
| --- | --- | --- |
| K5·1 reinforcement learning framework · domain randomiza… | 25 | 0.268 |
| K5·2 dexterous manipulation · human demonstrations | 32 | 0.181 |
| K5·3 successor features · reward functions | 22 | 0.194 |
| K5·4 latent dynamics · foundation models | 19 | 0.251 |
| K5·5 real world · vision-language-action vla | 22 | 0.107 |

##### K=8 Reference level

| Cluster | Documents | global | adaptive |
| --- | --- | --- | --- |
| K8·1 Region 8 | 1 | 0.002 | 0.000 |
| K8·2 latent dynamics · models | 11 | 0.108 | 0.000 |
| K8·3 zero-shot sim-to-real transfer · contact-rich manip… | 22 | 0.113 | 0.000 |
| K8·4 human videos · real world | 33 | 0.252 | 0.379 |
| K8·5 behavioral foundation · test time | 13 | 0.115 | 0.000 |
| K8·6 reinforcement learning systems · successor features | 21 | 0.252 | 0.440 |
| K8·7 domain randomization · bipedal | 14 | 0.156 | 0.181 |
| K8·8 hand · robotic | 5 | 0.001 | 0.000 |


Mixture entropy `H = 0.973`; effective contributing documents `N_eff = 61.8`; top-3 weight concentration `T3 = 0.090`; 5th-neighbour distance `d5 = 0.279`; relative uncertainty `u = 0.4704` (RBF surrogate; Matern `0.4520`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Origin | Title |
| --- | --- | --- | --- | --- | --- |
| `doc 92` | 0.0303 | Arm-Manipulation | 2021 | harvested | SCAPE: Learning Stiffness Control from Augmented Position Control Experiences |
| `doc 3` | 0.0303 | Arm-Manipulation | 2024 | seed | ASID: ACTIVE EXPLORATION FOR SYSTEM IDENTIFICATION IN ROBOTIC MANIPULATION |
| `doc 2` | 0.0293 | Sim2Real-Physics | 2021 | seed | Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning |
| `doc 45` | 0.0278 | ZeroShot-Rep | 2024 | harvested | Inferring Behavior-Specific Context Improves Zero-Shot Generalization in Reinforcement Learning |
| `doc 97` | 0.0255 | Arm-Manipulation | 2024 | harvested | Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning |

**L2 theme lens:** `representation` (df=115), `representation-learning` (df=4), `reinforcement-learning` (df=3), `state-representation` (df=2), `generalization` (df=86), `generalize` (df=90), `optimization` (df=104), `parameterized` (df=51), `observations` (df=117), `action-conditional` (df=4), `imitation-learning` (df=5), `demonstration-based` (df=2)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (+0.10, +0.50)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `representation` | 0.54446 | 115 | theme |
| L2 theme | `representation-learning` | 0.51670 | 4 | theme |
| L2 theme | `reinforcement-learning` | 0.46171 | 3 | theme |
| L2 theme | `state-representation` | 0.42289 | 2 | theme |
| L2 theme | `generalization` | 0.39693 | 86 | theme |
| L2 theme | `parameterized` | 0.39602 | 51 | theme |
| L2 theme | `generalize` | 0.39452 | 90 | theme |
| L2 theme | `observations` | 0.38977 | 117 | theme |
| L2 theme | `simulation-to-reality` | 0.38975 | 3 | theme |
| L2 theme | `optimization` | 0.38890 | 104 | theme |
| L2 theme | `parameter-conditioned` | 0.38195 | 3 | theme |
| L2 theme | `demonstration-based` | 0.37839 | 2 | theme |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

_None at gap #1._

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (+0.10, +0.50) | *(none)* | 0 |
| Archetype B | (+0.20, -0.40) | *(none)* | 0 |
| Archetype C | (+0.20, +0.30) | *(none)* | 0 |

## 6. Distance-maximising baseline (Table 4 analogue)

The negative control of Obayashi et al. (2026) Sec. 8.8: pick the 16 grid points with the largest nearest-neighbour distance, once over the whole grid and once over the interior only (excluding `x = ±1` and `y = ±1`), then compare their mean scores with the practical front. An unconstrained novelty search maximises emptiness, not discovery value.

| Selection | d_nn | S | R | P_boundary | G |
| --- | --- | --- | --- | --- | --- |
| **Practical Pareto front (this report)** | 0.2243 | 0.8510 | 0.9747 | 0.6460 | 0.7222 |
| Distance-maximising, all grid | 0.3421 | 0.0538 | 0.0000 | 0.0868 | 0.0123 |
| Distance-maximising, interior only | 0.2201 | 0.1157 | 0.0371 | 0.2070 | 0.0438 |

All 12 unconstrained picks sit on the chart edge, and 0 of the 16 interior-only picks do. Restricting the search to the 361 interior points recovers support (`S` 0.054 -> 0.116) but still trails the practical front, which is the paper's point: distance alone is not a discovery criterion. Two caveats when reading this table. The distance column is the *fifth*-neighbour distance used throughout this report, not the nearest-neighbour distance of the paper's Table 4, so the absolute numbers are not comparable across the two studies. And `R` is reported as a diagnostic only: it rewards low surrogate uncertainty (Sec. 3.7), which is why it is excluded from the objectives. The front scoring high on `R` therefore says the selected locations sit in well-sampled regions of the surrogate, not that the front was optimised for it.

## 7. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (+0.10, +0.50) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_+0.10_+0.50.json`, as produced by `code/make_evidence.py`. 
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

--- attached: evidence_point_+0.10_+0.50.json ---
{
  "location": {
    "x": 0.1,
    "y": 0.5
  },
  "theme_lens_L2": [
    {
      "term": "representation",
      "weight": 0.54446,
      "df": 115,
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
      "weight": 0.5167,
      "df": 4,
      "word_forms": [
        "representation-learning"
      ]
    },
    {
      "term": "reinforcement-learning",
      "weight": 0.46171,
      "df": 3,
      "word_forms": [
        "reinforcement-learned",
        "reinforcement-learning"
      ]
    },
    {
      "term": "state-representation",
      "weight": 0.42289,
      "df": 2,
      "word_forms": [
        "state-representation",
        "state-representations"
      ]
    },
    {
      "term": "generalization",
      "weight": 0.39693,
      "df": 86,
      "word_forms": [
        "generalisation",
        "generalising",
        "generality",
        "generalization",
        "generalizing"
      ]
    },
    {
      "term": "parameterized",
      "weight": 0.39602,
      "df": 51,
      "word_forms": [
        "parameterizations",
        "parameterize",
        "parameterized",
        "parameterizes"
      ]
    },
    {
      "term": "generalize",
      "weight": 0.39452,
      "df": 90,
      "word_forms": [
        "generalizations",
        "generalize",
        "generalized",
        "generalizes"
      ]
    },
    {
      "term": "observations",
      "weight": 0.38977,
      "df": 117,
      "word_forms": [
        "observation",
        "observational",
        "observations",
        "observe",
        "observed",
        "observer",
        "observers",
        "observes",
        "observing"
      ]
    },
    {
      "term": "simulation-to-reality",
      "weight": 0.38975,
      "df": 3,
      "word_forms": [
        "simulation-to-reality"
      ]
    },
    {
      "term": "optimization",
      "weight": 0.3889,
      "df": 104,
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
      "term": "parameter-conditioned",
      "weight": 0.38195,
      "df": 3,
      "word_forms": [
        "parameter-conditioned"
      ]
    },
    {
      "term": "demonstration-based",
      "weight": 0.37839,
      "df": 2,
      "word_forms": [
        "demonstration-based"
      ]
    }
  ],
  "concentration_lens_L1": [],
  "contributing_documents_top3": [
    {
      "doc": "doc 2",
      "source": "https://arxiv.org/abs/2109.11978v3",
      "is_seed": true,
      "title": "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning",
      "authors": [
        "Nikita Rudin",
        "David Hoeller",
        "Philipp Reist"
      ],
      "year": 2021,
      "regime": "Sim2Real-Physics",
      "weight": 0.0266
    },
    {
      "doc": "doc 81",
      "source": "https://arxiv.org/abs/2604.09487v4",
      "is_seed": false,
      "title": "Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks",
      "authors": [
        "Jan Schneider",
        "Mridul Mahajan",
        "Le Chen"
      ],
      "year": 2026,
      "regime": "Sim2Real-Physics",
      "weight": 0.0262
    },
    {
      "doc": "doc 14",
      "source": "https://arxiv.org/abs/2505.14266v1",
      "is_seed": true,
      "title": "Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning",
      "authors": [
        "Nikhil Sobanbabu",
        "Guanqi He",
        "Tairan He"
      ],
      "year": 2025,
      "regime": "Sim2Real-Physics",
      "weight": 0.0256
    }
  ],
  "mixture_entropy_H": 0.9166,
  "effective_contributing_documents": 64.31,
  "relative_uncertainty_u": 0.469952
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
- The corpus is founded on 0 user-curated seeding papers plus a per-regime harvest quota. A regime whose query returns few candidates fills its quota from the top results by relevance and has no headroom beyond review rejection.
- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately **not** a Pareto objective: maximizing low uncertainty favours already-dense regions.

