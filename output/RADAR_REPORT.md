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
| Stress minimization | metric stress 204.69, Spearman(distance) 0.8658 |
| Evaluation grid | 21 × 21 = 441 candidates |
| SPH bandwidth h (global) | 0.25 |
| Adaptive neighbourhood | 8 nearest documents |
| White-space quantiles (this corpus) | q20 0.0707, q60 0.2154, q85 0.3231, q98 0.4294 |
| White-space quantiles (paper) | q20 0.0517, q60 0.1000, q85 0.1376, q98 0.2000 |

The band-pass window for `W` is recomputed on this corpus's own nearest-document distances (`use_empirical_quantiles: true`), because the paper's quantiles belong to its aerodynamic corpus at a different map scale. The paper's values are retained here as the documented reference.

## 2. Practical Pareto front

Exact pairwise dominance over all 441 candidates gives **45 non-dominated** points; requiring `W(x) > 0.0` (Eq. 25) leaves the **33 practical gaps** below, ranked by Euclidean distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge stay in the front but are interpreted cautiously (Sec. 4.5).

| Rank | x | y | P_boundary | W | G | S | d_ideal | Archetype | N_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -0.10 | -0.50 | 0.982 | 1.000 | 0.730 | 0.638 | 0.4519 | Archetype A | 29.8 |
| 2 | -0.10 | -0.60 | 0.957 | 1.000 | 0.996 | 0.541 | 0.4607 | Archetype A | 21.4 |
| 3 | -0.20 | -0.60 | 0.957 | 0.851 | 0.857 | 0.556 | 0.4911 | Archetype A | 24.1 |
| 4 | +0.00 | -0.50 | 0.857 | 0.797 | 0.781 | 0.588 | 0.5284 | Archetype A | 24.8 |
| 5 | -0.20 | -0.50 | 0.910 | 1.000 | 0.548 | 0.666 | 0.5696 | Archetype A | 31.3 |
| 6 | -0.20 | -0.70 | 0.959 | 1.000 | 1.000 | 0.427 | 0.5748 | Archetype A | 14.2 |
| 7 | -0.10 | -0.40 | 0.966 | 0.612 | 0.592 | 0.757 | 0.6140 | Archetype A | 39.9 |
| 8 | +0.10 | +0.60 | 0.958 | 0.747 | 0.469 | 0.755 | 0.6390 | Archetype B | 41.4 |
| 9 | -0.20 | -0.40 | 0.846 | 0.649 | 0.400 | 0.780 | 0.7449 | Archetype A | 40.3 |
| 10 | +0.00 | +0.60 | 0.847 | 0.820 | 0.295 | 0.834 | 0.7623 | Archetype B | 48.1 |
| 11 | +0.00 | +0.70 | 0.831 | 1.000 | 0.313 | 0.712 | 0.7643 | Archetype B | 36.9 |
| 12 | +0.10 | +0.50 | 0.936 | 0.471 | 0.463 | 0.869 | 0.7677 | Archetype B | 50.4 |
| 13 | +0.00 | -0.20 | 0.821 | 0.362 | 0.590 | 0.970 | 0.7797 | Archetype A | 61.6 |
| 14 | -0.10 | -0.30 | 0.960 | 0.329 | 0.504 | 0.893 | 0.8416 | Archetype A | 53.2 |
| 15 | -0.10 | +0.70 | 0.709 | 1.000 | 0.154 | 0.748 | 0.9297 | Archetype B | 40.5 |
| 16 | +0.00 | +0.50 | 0.820 | 0.385 | 0.299 | 0.941 | 0.9512 | Archetype B | 57.6 |
| 17 | +0.00 | -0.40 | 0.878 | 0.134 | 0.696 | 0.699 | 0.9739 | Archetype A | 34.5 |
| 18 | -0.80 | +0.40 | 0.880 | 1.000 | 0.044 | 0.740 | 0.9980 | Archetype C | 45.3 |
| 19 | -0.70 | +0.30 | 0.863 | 1.000 | 0.000 | 0.904 | 1.0138 | Archetype C | 57.7 |
| 20 | -0.70 | +0.20 | 0.827 | 1.000 | 0.000 | 0.928 | 1.0175 | Archetype C | 58.2 |
| 21 | -0.80 | +0.30 | 0.877 | 1.000 | 0.003 | 0.802 | 1.0236 | Archetype C | 49.8 |
| 22 | -0.10 | +0.60 | 0.674 | 0.549 | 0.134 | 0.859 | 1.0384 | Archetype B | 50.7 |
| 23 | -0.60 | +0.40 | 0.796 | 0.812 | 0.000 | 0.937 | 1.0397 | Archetype C | 59.1 |
| 24 | +0.10 | -0.40 | 0.503 | 0.206 | 0.657 | 0.654 | 1.0556 | Archetype A | 28.0 |
| 25 | -0.20 | +0.70 | 0.609 | 1.000 | 0.044 | 0.773 | 1.0578 | Archetype B | 42.4 |
| 26 | -0.10 | -0.10 | 0.997 | 0.135 | 0.368 | 0.993 | 1.0717 | Archetype A | 79.9 |
| 27 | +0.00 | +0.40 | 0.855 | 0.123 | 0.316 | 0.995 | 1.1216 | Archetype B | 67.0 |
| 28 | +0.10 | -0.20 | 0.264 | 0.187 | 0.634 | 0.896 | 1.1611 | Archetype A | 49.6 |
| 29 | -0.20 | +0.60 | 0.476 | 0.608 | 0.025 | 0.883 | 1.1800 | Archetype B | 51.3 |
| 30 | -0.60 | +0.30 | 0.851 | 0.390 | 0.000 | 0.972 | 1.1810 | Archetype C | 63.7 |
| 31 | -0.60 | +0.20 | 0.783 | 0.365 | 0.000 | 0.987 | 1.2043 | Archetype C | 65.2 |
| 32 | +0.10 | -0.10 | 0.258 | 0.052 | 0.611 | 0.962 | 1.2657 | Archetype A | 59.3 |
| 33 | -0.50 | +0.40 | 0.629 | 0.218 | 0.000 | 0.997 | 1.3224 | Archetype C | 63.5 |

## 3. Discovery archetypes

Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points represent the same opportunity. Single-linkage grouping at radius `0.35` yields 3 archetypes.

| Archetype | Points | Representative | Objectives | d_ideal |
| --- | --- | --- | --- | --- |
| Archetype A | 15 | (-0.10, -0.50) | P_boundary 0.98, W 1.00, G 0.73, S 0.64 | 0.4519 |
| Archetype B | 10 | (+0.10, +0.60) | P_boundary 0.96, W 0.75, G 0.47, S 0.75 | 0.6390 |
| Archetype C | 8 | (-0.80, +0.40) | P_boundary 0.88, W 1.00, G 0.04, S 0.74 | 0.9980 |

## 4. Deep dive: top 3 Pareto archetypes

### Archetype A — gap #1 at (-0.10, -0.50)

**15** points in this archetype. Objectives: **P_boundary 0.982**, **W 1.000**, **G 0.730**, **S 0.638**; distance to ideal `0.4519`.

| Regime | Global mixture | Adaptive mixture |
| --- | --- | --- |
| SysID | 0.497 | 0.426 |
| Sim2Real | 0.046 | 0.110 |
| RL | 0.457 | 0.464 |

Mixture entropy `H = 0.771`; effective contributing documents `N_eff = 29.8`; top-3 weight concentration `T3 = 0.206`; 5th-neighbour distance `d5 = 0.347`; relative uncertainty `u = 0.1139`.

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Title |
| --- | --- | --- | --- | --- |
| `2110.00207v1` | 0.0821 | SysID | 2021 | Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial |
| `2205.03504v1` | 0.0651 | SysID | 2022 | Reinforcement Learning Approach to Estimation in Linear Systems |
| `2011.12360v1` | 0.0588 | RL | 2020 | A reinforcement learning control approach for underwater manipulation under position and torque constraints |
| `2205.09836v1` | 0.0559 | SysID | 2022 | Concurrent Policy Blending and System Identification for Generalized Assistive Control |
| `1912.09093v3` | 0.0473 | SysID | 2019 | Identification of abrupt stiffness changes of structures with tuned mass dampers under sudden events |

**L2 theme lens:** `identification` (df=43), `learning` (df=87), `reinforcement` (df=68), `control` (df=44), `manipulation` (df=54), `systems` (df=28), `constraints` (df=9), `robust` (df=18), `linear` (df=13), `state` (df=22), `tasks` (df=51), `demonstrations` (df=16)

**L1 concentration lens:** `contraction-based` (df=1), `behavioural` (df=1), `contraction` (df=1), `armax` (df=1), `lqg` (df=1), `underwater` (df=1), `learnt` (df=2), `guaranteed` (df=2), `adapted` (df=3), `blending` (df=2), `manipulator` (df=3), `stiffness` (df=2)

Terms present in **both** lenses (strongest evidence): *(none)*

### Archetype B — gap #8 at (+0.10, +0.60)

**10** points in this archetype. Objectives: **P_boundary 0.958**, **W 0.747**, **G 0.469**, **S 0.755**; distance to ideal `0.6390`.

| Regime | Global mixture | Adaptive mixture |
| --- | --- | --- |
| SysID | 0.416 | 0.392 |
| Sim2Real | 0.527 | 0.523 |
| RL | 0.057 | 0.084 |

Mixture entropy `H = 0.788`; effective contributing documents `N_eff = 41.4`; top-3 weight concentration `T3 = 0.154`; 5th-neighbour distance `d5 = 0.309`; relative uncertainty `u = 0.0842`.

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Title |
| --- | --- | --- | --- | --- |
| `2511.06839v1` | 0.0600 | SysID | 2025 | Vision-Based System Identification of a Quadrotor |
| `2404.08362v2` | 0.0491 | SysID | 2024 | Optimization-Based System Identification and Moving Horizon Estimation Using Low-Cost Sensors for a Miniature Car-Like Robot |
| `2504.21586v1` | 0.0445 | Sim2Real | 2025 | One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms |
| `2308.03898v2` | 0.0415 | SysID | 2023 | System Identification and Control of Front-Steered Ackermann Vehicles through Differentiable Physics |
| `1809.08862v1` | 0.0361 | SysID | 2018 | Characterization of Biologically Relevant Network Structures form Time-series Data |

**L2 theme lens:** `identification` (df=43), `sim-to-real` (df=41), `randomization` (df=42), `domain` (df=43), `control` (df=44), `transfer` (df=35), `learning` (df=87), `real-world` (df=37), `simulation` (df=48), `gap` (df=24), `parameters` (df=31), `models` (df=36)

**L1 concentration lens:** `quadrotor` (df=1), `vehicle` (df=3), `racing` (df=2), `vision` (df=3), `vehicles` (df=4), `vision-based` (df=6), `controller` (df=13), `controllers` (df=8), `potential` (df=12), `techniques` (df=17), `time` (df=15), `gap` (df=24)

Terms present in **both** lenses (strongest evidence): `gap`

### Archetype C — gap #18 at (-0.80, +0.40)

**8** points in this archetype. Objectives: **P_boundary 0.880**, **W 1.000**, **G 0.044**, **S 0.740**; distance to ideal `0.9980`.

| Regime | Global mixture | Adaptive mixture |
| --- | --- | --- |
| SysID | 0.004 | 0.039 |
| Sim2Real | 0.571 | 0.513 |
| RL | 0.425 | 0.448 |

Mixture entropy `H = 0.641`; effective contributing documents `N_eff = 45.3`; top-3 weight concentration `T3 = 0.139`; 5th-neighbour distance `d5 = 0.401`; relative uncertainty `u = 0.1604`.

##### Top 5 contributing anchor papers

| Document | Weight | Regime | Year | Title |
| --- | --- | --- | --- | --- |
| `2210.12682v1` | 0.0594 | Sim2Real | 2022 | Photo-realistic Neural Domain Randomization |
| `2504.04148v1` | 0.0439 | Sim2Real | 2025 | A General Peg-in-Hole Assembly Policy Based on Domain Randomized Reinforcement Learning |
| `2409.06613v2` | 0.0354 | Sim2Real | 2024 | DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots |
| `2504.14989v1` | 0.0327 | RL | 2025 | Dynamic Legged Ball Manipulation on Rugged Terrains with Hierarchical Reinforcement Learning |
| `2505.08458v1` | 0.0324 | Sim2Real | 2025 | Zero-Shot Sim-to-Real Reinforcement Learning for Fruit Harvesting |

**L2 theme lens:** `manipulation` (df=54), `sim-to-real` (df=41), `learning` (df=87), `reinforcement` (df=68), `simulation` (df=48), `tasks` (df=51), `domain` (df=43), `transfer` (df=35), `randomization` (df=42), `robot` (df=37), `real` (df=36), `robotic` (df=41)

**L1 concentration lens:** `assembly` (df=3), `primitives` (df=3), `videos` (df=11), `pipeline` (df=9), `generalization` (df=11), `objects` (df=11), `object` (df=21), `zero-shot` (df=16), `success` (df=15), `environment` (df=15), `dexterous` (df=10), `grasping` (df=7)

Terms present in **both** lenses (strongest evidence): *(none)*

## 5. Local vocabulary analysis

The L2 lens marks the local research theme (direction of the l2-normalized TF-IDF field); the L1 lens marks terms whose probability mass is locally concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: `df >= 4` may be read as an emergent shared topic, while `df <= 3` may only be a limited-source hypothesis that names its source documents.

### 5.1 Lens table for gap #1 (-0.10, -0.50)

| Lens | Term | Score | df | Status |
| --- | --- | --- | --- | --- |
| L2 theme | `identification` | 0.06708 | 43 | theme |
| L2 theme | `learning` | 0.06696 | 87 | theme |
| L2 theme | `reinforcement` | 0.05327 | 68 | theme |
| L2 theme | `control` | 0.04610 | 44 | theme |
| L2 theme | `manipulation` | 0.04088 | 54 | theme |
| L2 theme | `systems` | 0.03926 | 28 | theme |
| L2 theme | `constraints` | 0.03408 | 9 | theme |
| L2 theme | `robust` | 0.03279 | 18 | theme |
| L2 theme | `linear` | 0.03271 | 13 | theme |
| L2 theme | `state` | 0.03162 | 22 | theme |
| L2 theme | `tasks` | 0.03027 | 51 | theme |
| L2 theme | `demonstrations` | 0.03009 | 16 | theme |
| L1 concentration | `contraction-based` | x9.86 | 1 | limited-source hypothesis |
| L1 concentration | `behavioural` | x9.86 | 1 | limited-source hypothesis |
| L1 concentration | `contraction` | x9.86 | 1 | limited-source hypothesis |
| L1 concentration | `armax` | x7.82 | 1 | limited-source hypothesis |
| L1 concentration | `lqg` | x7.82 | 1 | limited-source hypothesis |
| L1 concentration | `underwater` | x7.06 | 1 | limited-source hypothesis |
| L1 concentration | `learnt` | x6.98 | 2 | limited-source hypothesis |
| L1 concentration | `guaranteed` | x5.98 | 2 | limited-source hypothesis |
| L1 concentration | `adapted` | x5.54 | 3 | limited-source hypothesis |
| L1 concentration | `blending` | x5.51 | 2 | limited-source hypothesis |
| L1 concentration | `manipulator` | x5.20 | 3 | limited-source hypothesis |
| L1 concentration | `stiffness` | x5.19 | 2 | limited-source hypothesis |

### 5.2 Emergent shared topics (df >= 4)

_No L1 concentration term reaches this df at gap #1._

### 5.3 Limited-source hypotheses (df <= 3)

| Term | df | Enrichment | Source documents |
| --- | --- | --- | --- |
| `contraction-based` | 1 | x9.86 | `2110.00207v1` (w=0.082) |
| `behavioural` | 1 | x9.86 | `2110.00207v1` (w=0.082) |
| `contraction` | 1 | x9.86 | `2110.00207v1` (w=0.082) |
| `armax` | 1 | x7.82 | `2205.03504v1` (w=0.065) |
| `lqg` | 1 | x7.82 | `2205.03504v1` (w=0.065) |
| `underwater` | 1 | x7.06 | `2011.12360v1` (w=0.059) |
| `learnt` | 2 | x6.98 | `2110.00207v1` (w=0.082); `2301.12587v3` (w=0.000) |
| `guaranteed` | 2 | x5.98 | `2205.03504v1` (w=0.065); `2204.08092v1` (w=0.003) |
| `adapted` | 3 | x5.54 | `2110.00207v1` (w=0.082); `2512.11609v3` (w=0.005); `2606.31101v1` (w=0.001) |
| `blending` | 2 | x5.51 | `2205.09836v1` (w=0.056); `2506.09366v1` (w=0.003) |
| `manipulator` | 3 | x5.20 | `2011.12360v1` (w=0.059); `2511.18878v1` (w=0.042); `2607.00534v1` (w=0.015) |
| `stiffness` | 2 | x5.19 | `1912.09093v3` (w=0.047); `2102.08442v2` (w=0.041) |

These terms must not be presented as established regional knowledge; each may only be raised as a hypothesis attributed to the listed documents.

### 5.4 Emergent shared topics across the archetype representatives

| Archetype | Location | Emergent shared topics (df >= 4) | Limited-source terms |
| --- | --- | --- | --- |
| Archetype A | (-0.10, -0.50) | *(none)* | 12 |
| Archetype B | (+0.10, +0.60) | `vehicles` (df=4), `vision-based` (df=6), `controller` (df=13), `controllers` (df=8), `potential` (df=12), `techniques` (df=17), `time` (df=15), `gap` (df=24) | 4 |
| Archetype C | (-0.80, +0.40) | `videos` (df=11), `pipeline` (df=9), `generalization` (df=11), `objects` (df=11), `object` (df=21), `zero-shot` (df=16), `success` (df=15), `environment` (df=15), `dexterous` (df=10), `grasping` (df=7) | 2 |

## 6. Evidence-constrained verbalization prompt

The block below is the **verbatim** Appendix A.1 prompt template from Obayashi et al. (2026), with only the placeholder file name `evidence_point_<x>_<y>.json` replaced. The attached evidence package for gap #1 — chart (-0.10, -0.50) — follows the template in the same block, so the whole block can be pasted into an LLM unchanged.

```text
You are participating in a controlled experiment.  A deterministic 
pipeline has analyzed a corpus of papers and exported a machine-computed 
**evidence package** for one location on its knowledge map — the attached 
`evidence_point_-0.10_-0.50.json`, as produced by `code/make_evidence.py`. 
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
   does not support.  Your general knowledge of composites may be used 
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

--- attached: evidence_point_-0.10_-0.50.json ---
{
  "location": {
    "x": -0.1,
    "y": -0.5
  },
  "theme_lens_L2": [
    {
      "term": "identification",
      "weight": 0.06708,
      "df": 43,
      "word_forms": [
        "identification"
      ]
    },
    {
      "term": "learning",
      "weight": 0.06696,
      "df": 87,
      "word_forms": [
        "learn",
        "learned",
        "learning",
        "learns"
      ]
    },
    {
      "term": "reinforcement",
      "weight": 0.05327,
      "df": 68,
      "word_forms": [
        "reinforcement"
      ]
    },
    {
      "term": "control",
      "weight": 0.0461,
      "df": 44,
      "word_forms": [
        "control",
        "controls"
      ]
    },
    {
      "term": "manipulation",
      "weight": 0.04088,
      "df": 54,
      "word_forms": [
        "manipulated",
        "manipulating",
        "manipulation",
        "manipulations"
      ]
    },
    {
      "term": "systems",
      "weight": 0.03926,
      "df": 28,
      "word_forms": [
        "systems"
      ]
    },
    {
      "term": "constraints",
      "weight": 0.03408,
      "df": 9,
      "word_forms": [
        "constraint",
        "constraints"
      ]
    },
    {
      "term": "robust",
      "weight": 0.03279,
      "df": 18,
      "word_forms": [
        "robust"
      ]
    },
    {
      "term": "linear",
      "weight": 0.03271,
      "df": 13,
      "word_forms": [
        "linear",
        "linearity"
      ]
    },
    {
      "term": "state",
      "weight": 0.03162,
      "df": 22,
      "word_forms": [
        "state",
        "states"
      ]
    },
    {
      "term": "tasks",
      "weight": 0.03027,
      "df": 51,
      "word_forms": [
        "task",
        "tasks"
      ]
    },
    {
      "term": "demonstrations",
      "weight": 0.03009,
      "df": 16,
      "word_forms": [
        "demonstrated",
        "demonstrates",
        "demonstrating",
        "demonstration",
        "demonstrations"
      ]
    }
  ],
  "concentration_lens_L1": [
    {
      "term": "contraction-based",
      "df": 1,
      "word_forms": [
        "contraction-based"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2110.00207v1",
          "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
          "weight": 0.0821
        }
      ]
    },
    {
      "term": "behavioural",
      "df": 1,
      "word_forms": [
        "behavioural"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2110.00207v1",
          "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
          "weight": 0.0821
        }
      ]
    },
    {
      "term": "contraction",
      "df": 1,
      "word_forms": [
        "contraction"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2110.00207v1",
          "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
          "weight": 0.0821
        }
      ]
    },
    {
      "term": "armax",
      "df": 1,
      "word_forms": [
        "armax"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2205.03504v1",
          "title": "Reinforcement Learning Approach to Estimation in Linear Systems",
          "weight": 0.0651
        }
      ]
    },
    {
      "term": "lqg",
      "df": 1,
      "word_forms": [
        "lqg"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2205.03504v1",
          "title": "Reinforcement Learning Approach to Estimation in Linear Systems",
          "weight": 0.0651
        }
      ]
    },
    {
      "term": "underwater",
      "df": 1,
      "word_forms": [
        "underwater"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2011.12360v1",
          "title": "A reinforcement learning control approach for underwater manipulation under position and torque constraints",
          "weight": 0.0588
        }
      ]
    },
    {
      "term": "learnt",
      "df": 2,
      "word_forms": [
        "learnt"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2110.00207v1",
          "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
          "weight": 0.0821
        },
        {
          "doc": "2301.12587v3",
          "title": "Zero-Shot Transfer of Haptics-Based Object Insertion Policies",
          "weight": 0.0004
        }
      ]
    },
    {
      "term": "guaranteed",
      "df": 2,
      "word_forms": [
        "guaranteed",
        "guarantees"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2205.03504v1",
          "title": "Reinforcement Learning Approach to Estimation in Linear Systems",
          "weight": 0.0651
        },
        {
          "doc": "2204.08092v1",
          "title": "The Existence and Uniqueness of Solutions for Kernel-Based System Identification",
          "weight": 0.0028
        }
      ]
    },
    {
      "term": "adapted",
      "df": 3,
      "word_forms": [
        "adapt",
        "adapted",
        "adapting",
        "adaptive",
        "adapts"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2110.00207v1",
          "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
          "weight": 0.0821
        },
        {
          "doc": "2512.11609v3",
          "title": "UniBYD: A Unified Framework for Learning Robotic Manipulation Across Embodiments Beyond Imitation of Human Demonstrations",
          "weight": 0.0052
        },
        {
          "doc": "2606.31101v1",
          "title": "Efficient Sim-to-Real Transfer of World-Action Models from Synthetic Priors",
          "weight": 0.0013
        }
      ]
    },
    {
      "term": "blending",
      "df": 2,
      "word_forms": [
        "blending",
        "blends"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2205.09836v1",
          "title": "Concurrent Policy Blending and System Identification for Generalized Assistive Control",
          "weight": 0.0559
        },
        {
          "doc": "2506.09366v1",
          "title": "SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending",
          "weight": 0.0034
        }
      ]
    },
    {
      "term": "manipulator",
      "df": 3,
      "word_forms": [
        "manipulator",
        "manipulators"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "2011.12360v1",
          "title": "A reinforcement learning control approach for underwater manipulation under position and torque constraints",
          "weight": 0.0588
        },
        {
          "doc": "2511.18878v1",
          "title": "Accelerating Reinforcement Learning via Error-Related Human Brain Signals",
          "weight": 0.042
        },
        {
          "doc": "2607.00534v1",
          "title": "Learning from Demonstration via Spatiotemporal Tubes for Unknown Euler-Lagrange Systems",
          "weight": 0.0155
        }
      ]
    },
    {
      "term": "stiffness",
      "df": 2,
      "word_forms": [
        "stiffness"
      ],
      "in_both_lenses": false,
      "source_docs": [
        {
          "doc": "1912.09093v3",
          "title": "Identification of abrupt stiffness changes of structures with tuned mass dampers under sudden events",
          "weight": 0.0473
        },
        {
          "doc": "2102.08442v2",
          "title": "SCAPE: Learning Stiffness Control from Augmented Position Control Experiences",
          "weight": 0.0412
        }
      ]
    }
  ],
  "contributing_documents_top3": [
    {
      "doc": "2110.00207v1",
      "title": "Contraction-Based Methods for Stable Identification and Robust Machine Learning: a Tutorial",
      "authors": [
        "Ian R. Manchester",
        "Max Revay",
        "Ruigang Wang"
      ],
      "year": 2021,
      "regime": "SysID",
      "weight": 0.0821
    },
    {
      "doc": "2205.03504v1",
      "title": "Reinforcement Learning Approach to Estimation in Linear Systems",
      "authors": [
        "Minyue Fu"
      ],
      "year": 2022,
      "regime": "SysID",
      "weight": 0.0651
    },
    {
      "doc": "2011.12360v1",
      "title": "A reinforcement learning control approach for underwater manipulation under position and torque constraints",
      "authors": [
        "Ignacio Carlucho",
        "Mariano De Paula",
        "Gerardo G. Acosta"
      ],
      "year": 2020,
      "regime": "RL",
      "weight": 0.0588
    }
  ],
  "mixture_entropy_H": 0.7713,
  "effective_contributing_documents": 29.83,
  "relative_uncertainty_u": 0.113934
}
```

Output expectations from the template: Task A is a 200–280-word virtual abstract, Task B an exactly 45-word summary, followed by an `Evidence used:` list. The audit checklist in Appendix A.2 applies unchanged.

## 7. Reproduce and verify

```bash
.venv/bin/python -m src.harvest --verify-only
.venv/bin/python -m src.embed   --verify-only
.venv/bin/python -m src.manifold --verify-only
.venv/bin/python -m src.pareto  --verify-only
.venv/bin/python -m src.report   --verify-only
```

## 8. Caveats

- Novelty is **corpus-relative**. These are gaps in a 120-paper arXiv sample, not proof that the topics are unexplored in the wider literature.
- Abstracts, not full text, feed the vocabulary and the embeddings. The paper uses full-text character n-grams, so its exact term statistics are not reproducible at this scale.
- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately **not** a Pareto objective: maximizing low uncertainty favours already-dense regions.
- The sim-to-real query returns only ~140 candidates corpus-wide, so its 40 papers are the top 40 by relevance with no headroom beyond review rejection.

