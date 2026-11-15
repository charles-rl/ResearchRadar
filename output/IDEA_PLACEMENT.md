# Idea placement: Your proposal

- document: `docs/papers/Current_Research_Proposal.md`
- sha256: `ae3db032ea7ddcc3`
- chart: 120 documents, 21x21 grid, stress 0.14, distance correlation 0.673

This document is **not** part of the corpus and is **not** included in any objective, front, or ranking. It is projected through the stored vocabulary and scored at its own coordinate.

## Document

- 1996 words, 15435 characters after markdown/LaTeX stripping
- vocabulary match: 86.1% of its n-grams are known to the corpus vocabulary
- text distance to the corpus: mean 1.231 (percentile 5.8 of corpus documents), nearest 1.121 (percentile 25.0)

## Position

- chart coordinate: **(-0.805, +0.973)**
- placement residual: 0.438 chart units (Spearman distance fit 0.641)
- leave-one-out position error on this corpus: median 0.356, p90 0.585 chart units (typical document spacing 1.005)

## In plain words

**Thinly supported corner of the map inside a single field, with textual isolation rather than white space as its novelty evidence.**

- **Where it sits.** The local mixture is Sim2Real-Physics at 77%. The objectives put you inside a single literature, so the contribution has to be read as depth in that field rather than a bridge between two.

- **What the novelty evidence is.** Novelty rests on the text distances, not on W. Mean distance to the corpus 1.231 (6% of corpus documents are further), nearest 1.121 (25%). In plain words: the document is more isolated than almost every paper in the corpus, so it is not a paraphrase of anything already on the map. 86% of its wording is corpus vocabulary, so the position is earned inside this map rather than imported from another field.

- **What the front says.** As a front candidate this position would not earn a slot: gap #1 (Archetype A) scores higher on P_boundary, W, G, S -- P_boundary 0.891 vs 0.244, W 1.000 vs 0.445, G 0.835 vs 0.000, S 0.946 vs 0.021. That compares positions on four axes; it does not say the work is weak, and most published papers are not front points either.

- **P_boundary 0.244** (32th) — border strength: 1 = a hard seam between literatures, 0 = inside one literature. For this document: there is a boundary near you but you are not on it, and the clusterings disagree about where it runs.
- **W 0.445** (38th) — white space: band-pass on the distance to the nearest paper, so 0 can mean incremental or unsupported and 1 means the healthy window between them. For this document: middle distance to the nearest paper: the space is partly occupied.
- **G 0.000** (0th) — gradient: how fast the map's topics change around this point. For this document: the surrounding literature is uniform, so this area is saturated with similar work.
- **S 0.021** (6th) — support: how much corpus evidence sits nearby to build on. For this document: thin support: fewer papers to build on, so claims here rest on less prior art.

_Every sentence is selected from the artifact's own values by fixed thresholds; nothing is generated. Percentiles are shares of the evaluation grid, not of published papers._

## Objectives

| objective | value | percentile among 441 grid points |
| :--- | ---: | ---: |
| P_boundary | 0.244 | 32 |
| W | 0.445 | 38 |
| G | 0.000 | 0 |
| S | 0.021 | 6 |

- distance to the ideal corner (1,1,1,1): **1.685** (best map gap 0.205, grid median 1.229). If this point were a candidate it would rank 403 of 441 by that distance; it is not one.
- `W` plateau: 25.2% of the 441 grid points score the same `W` value, because the band-pass saturates between q60 and q85. A plateau score means the nearest document is at a healthy distance; it does not mean the position is unique.

## Region

| regime | share |
| :--- | ---: |
| ZeroShot-Rep | 0.083 |
| Sim2Real-Physics | 0.767 |
| Arm-Manipulation | 0.151 |

- nearest-document distance d_nn = 0.225 (corpus quantiles q20 0.048, q60 0.088, q85 0.128, q98 0.303)
- persistent boundary views: k5_global 0.155, k8_global 0.544, k8_adaptive 0.251
- support: N_eff 53.0, d5 0.358, T3 0.095

## Nearest corpus documents

Two orderings, reported separately: the chart distorts text distance (stress 0.14, distance correlation 0.673), so the document nearest in the chart is not always the document nearest in words.

### Nearest in text (cosine similarity)

| text rank | chart rank | cos sim | chart d | regime | title |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | 28 | 0.371 | 0.867 | Sim2Real-Physics | Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning |
| 2 | 22 | 0.301 | 0.779 | Sim2Real-Physics | Adaptive-MHE : A Sampling-Based Adaptive MPC for Legged Loco-Manipulation via Moving Horizon Estimation |
| 3 | 65 | 0.295 | 1.416 | ZeroShot-Rep | BFM-ZERO: A PROMPTABLE BEHAVIORAL FOUNDATION MODEL FOR HUMANOID CONTROL USING UNSUPERVISED REINFORCEMENT LEARNING |

### Nearest on the map (chart distance)

| chart rank | text rank | chart d | cos sim | regime | title |
| ---: | ---: | ---: | ---: | :--- | :--- |
| 1 | 87 | 0.225 | 0.222 | Sim2Real-Physics | ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control |
| 2 | 34 | 0.257 | 0.259 | Sim2Real-Physics | Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting |
| 3 | 24 | 0.299 | 0.270 | Sim2Real-Physics | Quantifying and Visualizing Sim-to-Real Gaps: Physics-Guided Regularization for Reproducibility |

### Full neighbour union

| chart d | cos sim | regime | title |
| ---: | ---: | :--- | :--- |
| 0.225 | 0.222 | Sim2Real-Physics | ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control |
| 0.257 | 0.259 | Sim2Real-Physics | Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting |
| 0.299 | 0.270 | Sim2Real-Physics | Quantifying and Visualizing Sim-to-Real Gaps: Physics-Guided Regularization for Reproducibility |
| 0.327 | 0.267 | Sim2Real-Physics | Learning End-to-End Control for Omnidirectional Aerial Motion on Overactuated Tilt-rotor Quadrotors |
| 0.357 | 0.293 | Sim2Real-Physics | Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning |
| 0.382 | 0.280 | Sim2Real-Physics | Real-World Deployment of Massively Parallel Sampling-Based MPC for Contact-Rich Manipulation |
| 0.461 | 0.252 | Sim2Real-Physics | Sim-to-real transfer of active suspension control using deep reinforcement learning |
| 0.480 | 0.257 | ZeroShot-Rep | PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis |

## Nearest discovery gaps

The idea has no gap rank: the front holds a finite set of discrete candidates ranked by d_ideal, and this document is not a candidate. The rows below are geometric neighbours, not a ranking. Where the document would fall if inserted:

- nearest on the map: gap #14 (0.833 chart units, its d_ideal 0.470) -- a projection neighbour.
- nearest in objectives: gap #27 (Delta 1.099 in P, W, G, S, its d_ideal 1.200) -- the actually similar proposal, which the 2-D projection cannot express.
- if inserted by d_ideal: #29 of 29 (hypothetical: insertion is arithmetic, not front membership).
- front membership: no. Gap #1 scores at least as high on all four objectives, so it dominates the document regardless of its d_ideal.

| gap | x | y | d_ideal | archetype | distance |
| ---: | ---: | ---: | ---: | :--- | ---: |
| #14 | -0.20 | +0.40 | 0.470 | Archetype F | 0.833 |
| #17 | -0.30 | +0.30 | 0.526 | Archetype F | 0.841 |
| #9 | +0.00 | +0.50 | 0.416 | Archetype A | 0.933 |

## Terms

- vocabulary the corpus barely covers (df <= 3): `anisotropic`, `unidentifiability`, `simulation-to-reality`, `epistemic-uncertainty-guided`, `configuration-invariant`, `augmentation-invariant`, `reinforcement-learning-based`, `electro-mechanical`, `reidentification`, `distribution-matching`, `reparameterization`, `cross-morphology`, `non-identifiable`, `trajectory-dependent`, `simulation-transferable`
- what the map says dominates this location: `representation`, `representation-learning`, `simulation-to-reality`, `simulation-to-real`, `demonstration-based`, `reinforcement-learning`, `simulation-trained`, `observations`, `randomized`, `simulation-specific`, `domain-randomization`, `parameterized`

## Caveats

- Placement residual is larger than the calibration's own error (0.438 chart units).
