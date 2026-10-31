# ResearchRadar — Supported Knowledge Gaps in Robot Learning

Automated Knowledge Manifold Optimization (KMO) over a balanced 120-paper arXiv corpus spanning three conflicting robotics regimes: classical system identification, sim-to-real / domain randomization, and deep reinforcement learning control.

Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. Corpus `120` papers (SysID 40, Sim2Real 40, RL 40), embedded with `codefuse-ai/F2LLM-4B` (120 documents on the chart).

All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.

## 1. Knowledge Manifold

| Quantity | Value |
| --- | --- |
| Chart | Ω = [−1, 1]², fixed |
| Documents | 120 |
| Gauge-fixing anchors | 9 (8 perimeter + 1 central) |
| Stress minimization | metric stress 610.78, Spearman(distance) 0.6039 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.25 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0511, q60 0.1663, q85 0.3105, q98 0.4294 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **61 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **50 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -0.20 | +0.80 | 0.951 | 1.000 | 0.797 | 0.497 | 0.5442 | Archetype A | 17.6 |
| 2 | -0.30 | +0.80 | 0.834 | 1.000 | 0.884 | 0.462 | 0.5748 | Archetype A | 16.6 |
| 3 | -0.10 | +0.80 | 0.948 | 0.912 | 0.709 | 0.502 | 0.5860 | Archetype A | 17.5 |
| 4 | -0.20 | +0.70 | 0.955 | 1.000 | 0.501 | 0.605 | 0.6380 | Archetype A | 22.4 |
| 5 | -0.40 | +0.70 | 0.693 | 0.923 | 0.745 | 0.501 | 0.6436 | Archetype A | 17.1 |
| 6 | -0.30 | +0.70 | 0.872 | 0.785 | 0.600 | 0.559 | 0.6453 | Archetype A | 20.3 |
| 7 | -0.20 | +0.90 | 0.871 | 1.000 | 1.000 | 0.336 | 0.6763 | Archetype A | 11.5 |
| 8 | -0.10 | +0.70 | 0.964 | 1.000 | 0.407 | 0.655 | 0.6872 | Archetype A | 23.4 |
| 9 | -0.40 | +0.80 | 0.644 | 1.000 | 1.000 | 0.385 | 0.7104 | Archetype A | 13.9 |
| 10 | +0.10 | +0.80 | 0.780 | 0.912 | 0.528 | 0.508 | 0.7221 | Archetype A | 17.5 |
| 11 | -0.50 | +0.60 | 0.513 | 0.935 | 0.710 | 0.513 | 0.7505 | Archetype A | 17.1 |
| 12 | -0.50 | +0.70 | 0.468 | 0.932 | 0.940 | 0.404 | 0.8039 | Archetype A | 13.2 |
| 13 | -0.20 | +0.60 | 0.948 | 0.603 | 0.348 | 0.704 | 0.8201 | Archetype A | 26.8 |
| 14 | -0.10 | -0.70 | 0.750 | 0.832 | 0.275 | 0.711 | 0.8365 | Archetype A | 26.2 |
| 15 | -0.60 | +0.50 | 0.341 | 0.862 | 0.630 | 0.538 | 0.8964 | Archetype A | 17.6 |
| 16 | -0.40 | +0.50 | 0.760 | 0.386 | 0.431 | 0.711 | 0.9176 | Archetype A | 25.1 |
| 17 | +0.40 | +0.60 | 0.704 | 1.000 | 0.183 | 0.666 | 0.9310 | Archetype A | 24.9 |
| 18 | -0.20 | -0.70 | 0.817 | 0.407 | 0.352 | 0.670 | 0.9556 | Archetype A | 24.6 |
| 19 | +0.50 | +0.50 | 0.634 | 1.000 | 0.170 | 0.699 | 0.9558 | Archetype A | 26.0 |
| 20 | -0.50 | +0.50 | 0.550 | 0.416 | 0.507 | 0.643 | 0.9561 | Archetype A | 21.2 |
| 21 | -0.40 | +0.60 | 0.732 | 0.281 | 0.557 | 0.612 | 0.9673 | Archetype A | 20.9 |
| 22 | +0.00 | -0.70 | 0.675 | 0.633 | 0.189 | 0.719 | 0.9885 | Archetype A | 27.0 |
| 23 | -0.10 | +0.60 | 0.960 | 0.405 | 0.249 | 0.756 | 0.9895 | Archetype A | 28.3 |
| 24 | -0.30 | -0.30 | 0.832 | 0.337 | 0.266 | 0.943 | 1.0048 | Archetype A | 42.8 |
| 25 | +0.70 | -0.30 | 0.833 | 1.000 | 0.063 | 0.672 | 1.0065 | Archetype B | 26.0 |
| 26 | +0.70 | -0.20 | 0.876 | 0.930 | 0.045 | 0.717 | 1.0065 | Archetype B | 27.4 |
| 27 | -0.60 | -0.40 | 0.466 | 0.685 | 0.284 | 0.657 | 1.0070 | Archetype A | 23.3 |
| 28 | +0.70 | -0.10 | 0.883 | 0.781 | 0.055 | 0.733 | 1.0124 | Archetype B | 28.1 |
| 29 | -0.30 | -0.60 | 0.838 | 0.282 | 0.333 | 0.742 | 1.0267 | Archetype A | 26.8 |
| 30 | -0.30 | +0.10 | 0.924 | 0.257 | 0.277 | 0.977 | 1.0392 | Archetype A | 48.3 |
| 31 | -0.60 | +0.40 | 0.364 | 0.548 | 0.413 | 0.618 | 1.0491 | Archetype A | 21.4 |
| 32 | +0.70 | +0.00 | 0.838 | 0.615 | 0.071 | 0.731 | 1.0537 | Archetype B | 28.1 |
| 33 | +0.30 | +0.60 | 0.782 | 0.613 | 0.088 | 0.701 | 1.0580 | Archetype A | 26.7 |
| 34 | -0.70 | +0.40 | 0.185 | 1.000 | 0.524 | 0.520 | 1.0594 | Archetype A | 18.4 |
| 35 | -0.30 | -0.10 | 0.842 | 0.227 | 0.271 | 0.985 | 1.0747 | Archetype A | 49.7 |
| 36 | +0.50 | -0.50 | 0.762 | 0.620 | 0.061 | 0.731 | 1.0752 | Archetype B | 27.8 |
| 37 | -0.30 | +0.30 | 0.917 | 0.176 | 0.295 | 0.911 | 1.0914 | Archetype A | 39.5 |
| 38 | -0.20 | +0.40 | 0.949 | 0.215 | 0.248 | 0.881 | 1.0950 | Archetype A | 37.9 |
| 39 | -0.20 | +0.10 | 0.977 | 0.212 | 0.232 | 0.994 | 1.1006 | Archetype A | 54.3 |
| 40 | -0.30 | +0.50 | 0.903 | 0.132 | 0.365 | 0.760 | 1.1058 | Archetype A | 28.8 |
| 41 | +0.40 | +0.50 | 0.743 | 0.472 | 0.071 | 0.753 | 1.1264 | Archetype A | 29.3 |
| 42 | -0.20 | +0.30 | 0.964 | 0.131 | 0.233 | 0.953 | 1.1604 | Archetype A | 44.2 |
| 43 | -0.10 | +0.00 | 0.952 | 0.135 | 0.154 | 1.000 | 1.2112 | Archetype A | 59.9 |
| 44 | +0.60 | -0.30 | 0.850 | 0.331 | 0.015 | 0.784 | 1.2199 | Archetype B | 30.4 |
| 45 | -0.10 | +0.20 | 0.967 | 0.131 | 0.143 | 0.998 | 1.2206 | Archetype A | 53.6 |
| 46 | +0.50 | +0.40 | 0.681 | 0.338 | 0.046 | 0.767 | 1.2264 | Archetype A | 30.0 |
| 47 | -0.90 | +0.40 | 0.000 | 0.638 | 0.952 | 0.390 | 1.2272 | Archetype A | 13.5 |
| 48 | -0.60 | +0.60 | 0.300 | 0.177 | 0.892 | 0.427 | 1.2279 | Archetype A | 13.3 |
| 49 | -0.20 | -0.20 | 0.831 | 0.023 | 0.222 | 1.000 | 1.2603 | Archetype A | 53.0 |
| 50 | +0.30 | +0.20 | 0.856 | 0.168 | 0.000 | 1.000 | 1.3085 | Archetype A | 49.2 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 2 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 44 | (-0.20, +0.80) | P_boundary 0.95, W 1.00, G 0.80, S 0.50 | 0.5442 |
| Archetype B | 6 | (+0.70, -0.30) | P_boundary 0.83, W 1.00, G 0.06, S 0.67 | 1.0065 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (-0.20, +0.80)

**44** points in this archetype. Objectives: **P_boundary 0.951**, **W 1.000**, **G 0.797**, **S 0.497**; distance to ideal `0.5442`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| SysID | 0.096 | 0.266 | 0.112 |
| Sim2Real | 0.468 | 0.125 | 0.084 |
| RL | 0.209 | 0.000 | 0.001 |

Mixture entropy `H = 0.860`; effective contributing documents `N_eff = 17.6`; top-3 weight concentration `T3 = 0.293`; 5th-neighbour distance `d5 = 0.351`; relative uncertainty `u = 0.0032` (RBF surrogate; Matern `0.0970`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Title |
| --- | --- | --- | --- | --- |
| `doc 102` | 0.1153 | RL | 2023 | Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models |
| `doc 4` | 0.0993 | SysID | 2026 | BLISS: Global Blind Identification of Linear Systems with Sparse Inputs |
| `doc 65` | 0.0789 | Sim2Real | 2022 | Photo-realistic Neural Domain Randomization |
| `doc 74` | 0.0786 | Sim2Real | 2025 | Statistical Guarantees for Offline Domain Randomization |
| `doc 31` | 0.0665 | SysID | 2022 | Robust Policy Learning over Multiple Uncertainty Sets |

**L2 theme lens:** `representation` (df=105), `experiments` (df=111), `reinforcement-learning` (df=3), `optimization` (df=108), `parameterized` (df=34), `generalize` (df=60), `information-theoretic` (df=4), `generalization` (df=70), `optimized` (df=73), `specific` (df=100), `parameterization` (df=13), `implementation` (df=87)

**L1 concentration lens:** *(none)*

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #25 at (+0.70, -0.30)

**6** points in this archetype. Objectives: **P_boundary 0.833**, **W 1.000**, **G 0.063**, **S 0.672**; distance to ideal `1.0065`.

| Regime | K=5 global | K=8 global | K=8 adaptive |
| --- | --- | --- | --- |
| SysID | 0.006 | 0.172 | 0.104 |
| Sim2Real | 0.000 | 0.099 | 0.116 |
| RL | 0.263 | 0.000 | 0.000 |

Mixture entropy `H = 0.672`; effective contributing documents `N_eff = 26.0`; top-3 weight concentration `T3 = 0.206`; 5th-neighbour distance `d5 = 0.265`; relative uncertainty `u = 0.0020` (RBF surrogate; Matern `0.0792`).

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Title |
| --- | --- | --- | --- | --- |
| `doc 116` | 0.0732 | RL | 2025 | SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending |
| `doc 81` | 0.0695 | RL | 2025 | Accelerating Reinforcement Learning via Error-Related Human Brain Signals |
| `doc 120` | 0.0630 | RL | 2025 | UniBYD: A Unified Framework for Learning Robotic Manipulation Across Embodiments Beyond Imitation of Human Demonstrations |
| `doc 72` | 0.0554 | Sim2Real | 2025 | A General Peg-in-Hole Assembly Policy Based on Domain Randomized Reinforcement Learning |
| `doc 80` | 0.0529 | Sim2Real | 2023 | Towards Transferring Tactile-based Continuous Force Control Policies from Simulation to Robot |

**L2 theme lens:** `reinforcement-learning` (df=3), `representation` (df=105), `experiments` (df=111), `loco-manipulation` (df=5), `demonstration-based` (df=2), `simulation-to-reality` (df=2), `implementation` (df=87), `observation` (df=108), `generalize` (df=60), `simulation-trained` (df=3), `simulation-to-real` (df=4), `locomanipulation` (df=3)

**L1 concentration lens:** `integrated-whole-body-rl` (df=1), `locomotion-control` (df=1), `loco-manipulation` (df=5)

Terms present in **both** lenses (strongest evidence): `loco-manipulation`

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (-0.20, +0.80)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `representation` | 0.46959 | 105 | theme |
| L2 theme | `experiments` | 0.39513 | 111 | theme |
| L2 theme | `reinforcement-learning` | 0.39456 | 3 | theme |
| L2 theme | `optimization` | 0.38922 | 108 | theme |
| L2 theme | `parameterized` | 0.37683 | 34 | theme |
| L2 theme | `generalize` | 0.36412 | 60 | theme |
| L2 theme | `information-theoretic` | 0.36254 | 4 | theme |
| L2 theme | `generalization` | 0.36040 | 70 | theme |
| L2 theme | `optimized` | 0.35918 | 73 | theme |
| L2 theme | `specific` | 0.34426 | 100 | theme |
| L2 theme | `parameterization` | 0.34387 | 13 | theme |
| L2 theme | `implementation` | 0.34309 | 87 | theme |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

_None at gap #1._

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (-0.20, +0.80) | *(none)* | 0 |
| Archetype B | (+0.70, -0.30) | `loco-manipulation` (df=5) | 2 |

## 6. Distance-maximising baseline (Table 4 analogue)

The negative control of Obayashi et al. (2026) Sec. 8.8: pick the 16 grid points with the largest nearest-neighbour distance, once over the whole grid and once over the interior only (excluding `x = ±1` and `y = ±1`), then compare their mean scores with the practical front. An unconstrained novelty search maximises emptiness, not discovery value.

| Selection | d_nn | S | R | P_boundary | G |
| --- | --- | --- | --- | --- | --- |
| **Practical Pareto front (this report)** | 0.2402 | 0.6941 | 0.8624 | 0.7536 | 0.3866 |
| Distance-maximising, all grid | 0.4402 | 0.0640 | 0.0176 | 0.0399 | 0.2363 |
| Distance-maximising, interior only | 0.3364 | 0.2905 | 0.3026 | 0.2714 | 0.5842 |

All 16 unconstrained picks sit on the chart edge, and 0 of the 16 interior-only picks do. Restricting the search to the 361 interior points recovers support (`S` 0.064 -> 0.290) but still trails the practical front, which is the paper's point: distance alone is not a discovery criterion. Two caveats when reading this table. The distance column is the *fifth*-neighbour distance used throughout this report, not the nearest-neighbour distance of the paper's Table 4, so the absolute numbers are not comparable across the two studies. And `R` is reported as a diagnostic only: it rewards low surrogate uncertainty (Sec. 3.7), which is why it is excluded from the objectives. The front scoring high on `R` therefore says the selected locations sit in well-sampled regions of the surrogate, not that the front was optimised for it.

## 7. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (-0.20, +0.80) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_-0.20_+0.80.json`, as produced by `code/make_evidence.py`. 
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

--- attached: evidence_point_-0.20_+0.80.json ---
{
  "location": {
    "x": -0.2,
    "y": 0.8
  },
  "theme_lens_L2": [
    {
      "term": "representation",
      "weight": 0.46959,
      "df": 105,
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
      "term": "experiments",
      "weight": 0.39513,
      "df": 111,
      "word_forms": [
        "experiment",
        "experimental",
        "experimentally",
        "experimentation",
        "experimented",
        "experimenting",
        "experiments"
      ]
    },
    {
      "term": "reinforcement-learning",
      "weight": 0.39456,
      "df": 3,
      "word_forms": [
        "reinforcement-learned",
        "reinforcement-learning"
      ]
    },
    {
      "term": "optimization",
      "weight": 0.38922,
      "df": 108,
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
      "term": "parameterized",
      "weight": 0.37683,
      "df": 34,
      "word_forms": [
        "parameterizations",
        "parameterize",
        "parameterized",
        "parameterizes"
      ]
    },
    {
      "term": "generalize",
      "weight": 0.36412,
      "df": 60,
      "word_forms": [
        "generalizations",
        "generalize",
        "generalized",
        "generalizes"
      ]
    },
    {
      "term": "information-theoretic",
      "weight": 0.36254,
      "df": 4,
      "word_forms": [
        "information-theoretic",
        "information-theoretical",
        "information-theoretically"
      ]
    },
    {
      "term": "generalization",
      "weight": 0.3604,
      "df": 70,
      "word_forms": [
        "generalisation",
        "generality",
        "generalization",
        "generalizing"
      ]
    },
    {
      "term": "optimized",
      "weight": 0.35918,
      "df": 73,
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
      "term": "specific",
      "weight": 0.34426,
      "df": 100,
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
      "term": "parameterization",
      "weight": 0.34387,
      "df": 13,
      "word_forms": [
        "parameterisation",
        "parameterization",
        "parameterizing"
      ]
    },
    {
      "term": "implementation",
      "weight": 0.34309,
      "df": 87,
      "word_forms": [
        "implementation",
        "implementations",
        "implemented",
        "implementing"
      ]
    }
  ],
  "concentration_lens_L1": [],
  "contributing_documents_top3": [
    {
      "doc": "doc 102",
      "arxiv": "2310.18308v1",
      "title": "Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models",
      "authors": [
        "Pushkal Katara",
        "Zhou Xian",
        "Katerina Fragkiadaki"
      ],
      "year": 2023,
      "regime": "RL",
      "weight": 0.1153
    },
    {
      "doc": "doc 4",
      "arxiv": "2604.00520v1",
      "title": "BLISS: Global Blind Identification of Linear Systems with Sparse Inputs",
      "authors": [
        "Kyle Poe",
        "Uday Kiran Reddy Tadipatri",
        "Benjamin D. Haeffele"
      ],
      "year": 2026,
      "regime": "SysID",
      "weight": 0.0993
    },
    {
      "doc": "doc 65",
      "arxiv": "2210.12682v1",
      "title": "Photo-realistic Neural Domain Randomization",
      "authors": [
        "Sergey Zakharov",
        "Rares Ambrus",
        "Vitor Guizilini"
      ],
      "year": 2022,
      "regime": "Sim2Real",
      "weight": 0.0789
    }
  ],
  "mixture_entropy_H": 0.8603,
  "effective_contributing_documents": 17.63,
  "relative_uncertainty_u": 0.003208
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

