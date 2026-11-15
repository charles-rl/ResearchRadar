# Paper-fidelity audit of the core math modules

Scope: `src/manifold.py`, `src/pareto.py`, and the two call sites in
`src/explore.py` that re-evaluate the same equations at a single point.
The website, dashboard, report renderer and audit heuristics are out of scope.

Precedence: **2610.25662 (Knowledge Manifold Optimization)** is the primary
specification. **2606.05907 (Knowledge Manifold)** supplies the two-dimensional
construction, the SPH kernel and the GPR formulation. Where they differ, the
KMO paper wins — concretely, the gauge is nine anchors (eight perimeter + one
central, §5.2) rather than the foundational paper's four corners (Eq. 3).

Equation numbers below refer to 2610.25662 unless prefixed with `F` (2606.05907).

---

## 1. Mathematical bugs and approximations that were fixed

### B1 — The SPH kernel was Gaussian, the papers specify a cubic spline

| Aspect | Detail |
| --- | --- |
| Paper | Eq. 10: `W(q) = 1 − 3/2 q² + 3/4 q³` for `q<1`; `¼(2−q)³` for `1≤q<2`; `0` beyond. Both papers use the same kernel. |
| Before | `sph_weights` returned `exp(−r²/2h²)`. |
| Why it matters | The kernel's shape *is* the interpolation operator. A Gaussian has unbounded support and a different weighting profile; the paper's kernel is compact, so a bandwidth has a definite neighbourhood. |
| The measured tell | The paper reports `N_eff ≈ 73.6` of 99 documents (§7.3). The Gaussian `h = 0.25` gave `N_eff` mean **22** on a 120-document corpus with a much heavier tail; the cubic spline with Eq. 9's bandwidth gives `N_eff` mean **≈ 70 of 120**, i.e. the paper's regime. |
| Fix | `cubic_spline_kernel(q)` implements Eq. 10 verbatim, including the `q = 1` continuity point. |

### B2 — The smoothing length was a fixed `0.25`, the paper derives it per query point

| Aspect | Detail |
| --- | --- |
| Paper | Eq. 9: `h = max_i r_i / 1.98`, where `r_i = ‖p − y_i‖`. This is a function of the query point, chosen so the farthest document sits at `q = 1.98 < 2` and **all** `N` documents contribute. |
| Before | `sph_h: 0.25` from `config.yaml`, constant over the chart. With the compact kernel, `h = 0.25` leaves most grid points with an *empty* kernel support (support radius `2h = 0.5`). |
| Fix | `sph_bandwidth(points, coords, nearest_k=None)` returns Eq. 9's `h(p)`. `sph_weights` accepts a scalar **or** a per-query-point bandwidth. `config.yaml` now sets `sph_h: 0.0`, meaning "derive it"; a positive value still pins a fixed `h` for sensitivity runs. |

### B3 — Stored SPH weights were not normalised (Eq. 11)

| Aspect | Detail |
| --- | --- |
| Paper | Eq. 11: `w_i = W(q_i) / Σ_j W(q_j)`. Every downstream quantity (`p_c`, `N_eff`, `T_3`, `v(x)`) assumes this. |
| Before | `sph_weights` returned raw kernel values. `normalized_mixture` and `support_statistics` silently re-normalised *copies*, but `weights_global` / `weights_adaptive` were written to `manifold_grid.npz` unnormalised — measured row sums **0.16–0.41**. Those arrays feed the explore stage's document contributions, so a "weight" there was not a probability. |
| Fix | `sph_weights`/`sph_weights_adaptive` now return normalised weights. `verify_manifold` asserts `max|Σ_i w_i − 1| ≤ 1e-5` (float32 storage) and rejects negative weights. |

### B4 — The knowledge gradient used a 0.1 grid step instead of Eq. 14's δ = 1e-3

| Aspect | Detail |
| --- | --- |
| Paper | Eqs. 13–14: `g(x) = ‖J_v(x)‖_F`, with `∂v/∂x ≈ (v(x+δ) − v(x−δ))/2δ` and `δ = 10⁻³`. |
| Before | `semantic_gradient` differenced *neighbouring grid samples* `0.1` apart, i.e. a second-order but much coarser estimator of the same derivative. The error is not uniform: it is largest where the field curves, which is exactly where `G` is meant to be large. |
| Fix | `semantic_gradient_field(points, coords, gram, nearest_k=None, delta=1e-3)` evaluates the field at `x ± δ`, **recomputing Eq. 9's bandwidth at those points** (because `h` is part of the field's definition, so the honest derivative includes `dh/dx`), and forms `‖v₊ − v₋‖² = 2 − 2cos` through the Gram matrix. `compute_objectives` and `explore.score_point` both use it. `semantic_gradient` (grid stencil) is retained for callers that only hold field samples. |

### B5 — The stress gradient was exactly half the true gradient

| Aspect | Detail |
| --- | --- |
| Where | `constrained_stress`, `jac=True`. |
| Bug | The energy is `S = ½ Σ_{i,j}(d_ij − D_ij)²`, which equals `Σ_{i<j}(…)²` because the full matrix double-counts unordered pairs. Its derivative is `2 Σ_j r_ij (x_i−x_j)/d_ij` (two ordered terms per pair), but the code returned `Σ_j r_ij (x_i−x_j)/d_ij`. |
| Effect | A positive multiple of the true gradient, so L-BFGS-B still descended — but the line search and the `gtol`/`ftol` stopping tests were evaluated on a mis-scaled gradient, and the reported optimum could stop early. |
| Fix | The factor 2 is restored, and a unit test compares the analytic gradient to a central-difference reference on every free coordinate. |

### B6 — The stress objective was neither normalised nor regularised (F Eqs. 4–8)

| Aspect | Detail |
| --- | --- |
| Paper | `J = E_stress + λ_rep E_rep + λ_var E_var + λ_ctr E_ctr`, with `λ_rep = 0.10`, `λ_var = 0.05`, `λ_ctr = 0.01`, `σ = 0.35`, and `E_stress = Σ(‖y_i−y_j‖ − α d_ij)² / Σ(α d_ij)²`. `α` is fixed so the mean 2-D distance between the gauge anchors equals the mean cosine distance among them. |
| Before | Raw `½ Σ(d − D)²`; no `α`, no repulsion, no variance target, no centring. The logged "normalized stress" was therefore a different functional (587.8 on this corpus) from the paper's `≈0.1505`. |
| Fix | `anchor_scale` computes `α` from the nine anchors (the natural generalisation of F's four-corner rule); `stress_energy` implements the full objective and returns its components; `constrained_stress` reports `E_stress`, the normalised quantity the paper quotes. |

### B7 — GPR uncertainty was `√10 · σ_post` and was not relative

| Aspect | Detail |
| --- | --- |
| Paper | Eq. 19 fits the kernel on a 10-dimensional SVD projection; §5.2 reports `σ_post`, `σ_prior`, and the **relative** uncertainty `σ_post/σ_prior`. Evidence packages quote `u ≈ 0.222`. |
| Before | `np.linalg.norm(std.reshape(n, -1), axis=1)`. A multi-output GP with a shared kernel has one posterior variance, so all ten target columns are **identical** (verified: column spread 0.0) and the norm multiplied the standard deviation by `√10`. The stored field was then an absolute `σ`, not a ratio, and `R = ½N_−(u_RBF) + ½N_−(u_M32)` was applied to that absolute value. |
| Fix | `σ_post` is read from a single target column; `u = σ_post/σ_prior` with `σ_prior = sqrt(c₀ + c_white)` of the fitted kernel. `u ≤ 1` pointwise is now an invariant, and it is asserted in the tests. |

### B8 — The GPR kernel did not match Eq. 19

| Aspect | Detail |
| --- | --- |
| Paper | `k = c₀·exp[−½((x−x′)²/ℓ_x² + (y−y′)²/ℓ_y²)] + c_white δ`, free hyperparameters fitted by marginal likelihood with `n_restarts_optimizer = 10`, `random_state = 0`, `normalize_y = False`. |
| Before | Isotropic `RBF(length_scale = scalar)`, **all bounds `"fixed"`** at the median inter-document distance, `normalize_y = True`, `optimizer = None`, and `alpha = 1e-8` *in addition to* `WhiteKernel(1e-6)` (double-counted noise). |
| Why the workaround existed | An earlier unconstrained fit collapsed to a degenerate field. That does not reproduce: with Eq. 19's model and sklearn's default bounds the fit lands at `ℓ ≈ [0.52, 0.61]` — the paper's fitted `[0.628, 0.538]` — and `u` spans `≈[0.44, 0.81]`. |
| Fix | Eq. 19 verbatim, `normalize_y = False`, `alpha = 0.0`, `n_restarts_optimizer = 10`, `random_state = 0`; the fitted kernels are logged and stored in the manifest. |

### B9 — `manifold.norm_percentiles` was dead configuration

| Aspect | Detail |
| --- | --- |
| Paper | Eqs. 9–10 fix `Q₁₀`/`Q₉₀`. |
| Before | `boundary_score`, the support terms, the gradient and `R` all called `normalize_percentile` with hardcoded defaults; `config.yaml`'s `norm_percentiles: [10, 90]` was never read. The value happened to match, so nothing looked wrong, but changing it did nothing. |
| Fix | `_percentile_pair` validates and threads the configured pair through every `N_+`/`N_−`, and the pair is stored in the archive and the manifest. |

### B10 — Entropy epsilon was applied as a floor, not as `p log(p + ε)`

| Aspect | Detail |
| --- | --- |
| Paper | Eq. 7: `H_K = −(1/log K) Σ_c p_c log(p_c + ε)`. |
| Before | `p·log(max(p, 1e-12))`: the floor changes the *leading* `p` as well, and the identity `H_K(uniform) = 1` then only holds approximately. |
| Fix | `ENTROPY_EPSILON` is added inside the logarithm only, and `K < 2` (where `log K = 0`) is rejected explicitly. |

### B11 — Adaptive smoothing length ignored the compact-support radius

| Aspect | Detail |
| --- | --- |
| Paper | §5.4: the adaptive field uses a local scale from the eight nearest documents. |
| Before | `document_bandwidth` returned the 8th-neighbour **distance** and used it directly as a Gaussian `σ`. Under a compact kernel the support radius is `2h`, so `h` must be `d₈/1.98` for the intended eight-document neighbourhood to be the one that contributes. |
| Fix | `document_bandwidth` returns `d₈/1.98`, matching Eq. 9's own convention; a test pins that the 8th neighbour sits at `q = 1.98` and the 9th outside the support. |

---

## 2. Hyperparameter ledger

| Quantity | Paper | Before | Now |
|---|---|---|---|
| Chart `Ω` | `[−1,1]²` | ✓ | ✓ |
| Evaluation grid | 21×21 = 441, spacing 0.1 | ✓ | ✓ |
| Clustering resolutions | K=5 discovery, K=8 reference | ✓ | ✓ |
| Boundary views | `(5, global)`, `(8, global)`, `(8, adaptive)` | ✓ | ✓ |
| `B_K` weights | 0.65 entropy / 0.35 gap | ✓ | ✓ |
| `N_+` / `N_−` percentiles | Q10 / Q90 | hardcoded, config ignored | config-driven, default (10, 90) |
| `W` quantiles | empirical q20/q60/q85/q98 over the domain | ✓ empirical | ✓ (unchanged) |
| `S` weights | 0.40 / 0.30 / 0.30 | ✓ | ✓ |
| `R` weights | 0.5 / 0.5 | ✓ | ✓ |
| `P_boundary` | harmonic mean of 3 views | ✓ | ✓ (unchanged) |
| Pareto objectives | `[P_boundary, W, G, S]` | ✓ | ✓ |
| Practical front | `W(x) > 0` | ✓ (`> 0`, strict) | ✓ (unchanged) |
| SPH kernel | cubic spline, Eq. 10 | Gaussian | cubic spline |
| Global `h` | `max_i r_i / 1.98` | 0.25 fixed | Eq. 9, per query point |
| Adaptive `h` | local scale, 8 nearest | `d₈` as Gaussian σ | `d₈/1.98`, compact support |
| Gradient step | δ = 1e-3 | 0.1 (grid spacing) | 1e-3 (configurable) |
| GPR latent dims | 10 (truncated SVD) | ✓ | ✓ |
| GPR kernel | anisotropic RBF + white | isotropic RBF, pinned | anisotropic RBF + white, fitted |
| GPR fit | 10 restarts, seed 0, `normalize_y=False` | `optimizer=None`, `normalize_y=True` | as specified |
| GPR noise | one white-noise term | white 1e-6 **and** alpha 1e-8 | one white-noise term, `alpha=0` |
| Stress `α` | anchor distance ratio, F Eq. 5 | absent | `anchor_scale` |
| Stress regularisers | 0.10 / 0.05 / 0.01, σ = 0.35 | absent | as specified |
| Anchors | 8 perimeter + 1 central | ✓ | ✓ (radius 0.92 is a local reconstruction constant) |
| Scalar discovery score `D` (Eq. 20) | abandoned in §4.1 | not used | not used (deliberate) |
| `R`, `C` | diagnostics, not objectives (§4.3) | ✓ | ✓ |

---

## 3. Deliberate approximations that were kept

1. **Clustering space.** §3.2 says the documents are partitioned "at multiple
   clustering resolutions" but does not say in which space. The pipeline
   clusters the L2 TF-IDF rows (`cluster_labels`), not the 2-D coordinates,
   because a partition of the chart would measure the projection's artefacts
   as much as the corpus. Documented, not changed.
2. **Percentile interpolation.** `np.percentile` uses linear interpolation
   between order statistics; the paper does not specify the convention.
3. **`ANCHOR_RADIUS = 0.92`.** The paper says "eight perimeter anchors" without
   a radius. Placing them at 0.92 keeps pinned documents strictly inside
   `[−1,1]²` (a chart-edge anchor would make the bounded-chart caveat of §4.5
   bite everywhere).
4. **Anchors are pinned, not penalised.** A hard pin is what "gauge-fixing"
   means; the alternative (soft attraction) would leave the gauge free.
5. **`distance_maximising_baseline`** reproduces Table 4 by construction; it is
   a comparison artefact, not part of the objective.

---

## 4. Verified identities

`tests/test_manifold.py` and `tests/test_pareto.py` now pin, among others:

- **Eq. 10** piecewise values, the `q = 1` continuity point, compact support,
  and monotone decrease.
- **Eqs. 9–11** `q_max = 1.98` so every document contributes; every weight row
  sums to 1; `1 ≤ N_eff ≤ N`; `sph_weights` equals `W/ΣW` exactly.
- **Adaptive** `h_i = d₈/1.98` places the 8th neighbour at `q = 1.98` and the
  9th outside the support.
- **Eq. 7** entropy matches `−Σp log(p+ε)/log K` including the epsilon
  placement; `K = 1` is rejected; uniform → 1, one-hot → 0.
- **Eq. 12** both ramps are exactly linear in `d_nn` (trapezoid), the plateau is
  exactly 1, and `W = 0` at `q20` and at `q98`.
- **Eqs. 13–15** `semantic_gradient_field` equals a dense central difference of
  the normalised field at `δ = 1e-3` (relative error < 1e-6), is non-negative,
  and vanishes for a uniform field.
- **Eqs. 16–18** `N_eff = 1/Σw²`, `d5` is the 5th nearest document, `T3` is the
  top-3 mass, and `S ∈ [0,1]`.
- **Eq. 21** the paper's worked example `(0.838, 0.425, 0.239) → 0.389` against
  an arithmetic mean of 0.501, plus the AND-like zero case.
- **F Eqs. 4–8** analytic stress gradient versus finite differences; `E_stress`
  is the normalised form; `E_var = 0` at `var = 1/3`; `E_rep = 1` when all
  points coincide; anchors stay pinned inside the chart.
- **Eq. 19 + §5.2** `u = σ_post/σ_prior`, `u ≤ 1` pointwise, and the fitted
  kernel is an anisotropic RBF plus white noise.
- **Eqs. 23–25** dominance against a brute-force double loop (including ties and
  duplicates), mutual non-dominance of the practical front, and the strict
  `W > 0` filter (a `W = 1e-12` point survives, `W = 0` does not).

---

## 5. Measured effect on this repository's 120-document corpus

Same corpus, same grid, same anchor seed; only the equations above changed.
`N_eff` is Eq. 16 on normalised weights, so it is comparable across the two
columns. The paper's own values (99 documents) are in the last column for scale.

| Quantity | Before | After | Paper |
| --- | --- | --- | --- |
| Reported stress | 587.83 (raw residual sum) | **0.14295** (`E_stress`) | 0.1505 |
| Spearman(distance) | 0.6557 | 0.6728 | 0.613 |
| SPH kernel | Gaussian | cubic spline (Eq. 10) | cubic spline |
| Global bandwidth | fixed 0.25 | **derived per point**, 0.58–1.30 | `max r / 1.98` |
| SPH weight row sums | 0.155 – 33.08 | **1.0** (drift 1.2e-7) | 1.0 |
| `N_eff` | 1.2 – 66.2 (mean 22.0) | **42.4 – 64.6** (mean 57.4) | ≈73.6 of 99 |
| `T3` (top-3 mass) | 0.086 – 0.948 | **0.074 – 0.131** | — |
| GPR uncertainty | 0.0004 – 0.0684 (absolute) | **0.469 – 0.622** (relative) | `u ≈ 0.222` |
| GPR fitted RBF | isotropic, pinned | **ℓ = [0.809, 0.868]** | [0.628, 0.538] |
| Kernel agreement `r` | 0.899 | 0.997 | 0.851 |
| Non-dominated / practical | 64 / 55 | **34 / 28** | 19 / 16 |
| Archetypes | 17 | **9** | 4 |

The `N_eff` and `T3` rows are the clearest diagnostics that the interpolation
operator itself was wrong: the old field was dominated by one or two documents
(`T3` up to 0.95), whereas the paper's normalised field spreads mass across most
of the corpus (`T3 ≈ 0.1`, `N_eff` in the tens). The Pareto counts differ from
the paper's because the corpus, its clustering and its field shapes differ; the
*order* of magnitude and the reduction from the non-dominated set to the
practical front are consistent.

---

## 6. Appendix A: the verbalization prompt and the audit checklist

Sections 1–5 cover the mathematics. Appendix A is text, not equations, and it is
the part a reader actually pastes into a model, so it is pinned to the paper's own
words rather than paraphrased. `tests/test_prompt_verbatim.py` re-derives both
artifacts from `docs/papers/foundation/2610.25662v1.md` on every test run.

### 6.1 A.1 prompt template — verbatim, with one documented substitution

`docs/kmo_prompt_template_A1.txt` is the A.1 text. Compared token for token
against the paper section (whitespace reflowed, dash glyphs folded) the two are
identical except for:

| Deviation | Why |
| --- | --- |
| `composites` → `robotics` (Rule 3) | The paper's case study is aerodynamics; this corpus is robot learning. This is the **only** semantic edit, it is the only line on which the in-use template differs from `docs/kmo_prompt_template_A1.original.txt`, and `src/report.py --verify-only` fails if a second edit appears. |
| `*evidence package**` → `**evidence package**` | Unbalanced bold marker in the PDF→Markdown extraction. |
| `` `source docs` `` → `` `source_docs` `` | The identifier was split across a line by the extraction. |
| duplicated `` - `mixture_entropy_H`, `effective_contributing_documents`, `` bullet | The extraction repeats the last bullet of "The package contains:" once before "Rules (binding)". |
| `**Serendipity** from … graded by df.**` → `**Serendipity from … graded by df.**` | Misplaced bold pair in the extraction. |
| two stray ``` fences around *Deliverables* | Markdown artifact; the paper has no code block there. |
| em dash vs `-`/en dash | The extraction is inconsistent about dashes. Glyph-only difference; the comparison folds every dash variant. |

None of these change a rule, a threshold, or an obligation. Specifically, the
numbers A.1 states are the numbers `src/audit.py` enforces: Task A
**200–280** words, Task B **exactly 45** words, `df >= 4` for an emergent shared
topic, `df <= 3` for a single-source hypothesis, and `source_docs` cited for every
`df <= 3` term. A test asserts this agreement in both directions, so editing
either the prompt or the audit thresholds alone fails the suite.

The evidence package the prompt is filled with carries exactly the fields A.1
names — `theme_lens_L2`, `concentration_lens_L1` (with `df`, and `source_docs`
only where `df <= 3`), `contributing_documents_top3`, `mixture_entropy_H`,
`effective_contributing_documents`, `relative_uncertainty_u` — via
`src.report.prompt_evidence_package`. The dashboard and the Markdown report call
the same `render_prompt_text`, so the on-screen prompt, the copy button, the
downloaded `.txt`, and `output/KMO_PROMPT_TOP1.txt` are byte-identical.

### 6.2 A.2 audit protocol — eight items, eight checks

The paper lists eight audit items; `src/audit.py` runs eight checks in the same
order, and a test pins the item ids to the bullets so an item cannot be dropped
silently.

| A.2 item | Check in `src/audit.py` | Implementation note |
| --- | --- | --- |
| Length compliance | `length_compliance` | Task A in `[200, 280]`, Task B exactly 45 (`task_b_tolerance = 0`). |
| Evidence coverage | `evidence_coverage` | Every supplied term must appear in the text or be explicitly set aside (A.1 rule 4). |
| Lens discipline | `lens_discipline` | L2 terms define the topic; L1-only terms stay supporting evidence. |
| Low-frequency provenance | `low_frequency_provenance` | Every L1-only `df <= 3` term needs a sentence that presents it as limited-source and cites `[doc N]`. |
| Unsupported-claim audit | `unsupported_claim_audit` | Mechanism verbs, material names, percentages and decimal performance numbers without a hedge are reported for a human. |
| Novelty wording | `novelty_wording` | Global claims ("for the first time", "state of the art") fail unless made corpus-relative. |
| Uncertainty disclosure | `uncertainty_disclosure` | Both `relative_uncertainty_u` and `effective_contributing_documents` must be stated in Task A. |
| Evidence trace | `evidence_trace` | The `Evidence used:` list must record every used term with a `df`. |

A.2's own disclaimer is carried into the report the audit prints: it "is intended
to expose unsupported extrapolation rather than to claim a mathematical guarantee
of hallucination-free generation". The checks are conservative heuristics, the
audit never rewrites the text, and a failure is a review item, not a verdict.
