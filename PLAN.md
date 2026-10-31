# `PLAN.md` — Project ResearchRadar: Automated Knowledge Manifold Optimization for Robotics

> **Status: implemented.** All six phases below are built and verified. The plan
> has been rewritten to describe what the code actually does, which is now a 1:1
> reproduction of Obayashi et al. (2026), *Knowledge Manifold Optimization*, with
> one deliberate structural difference: the corpus is anchored on user-curated
> robotics seeding papers instead of a literature-search API. Every equation,
> percentile, weight, and threshold below is the paper's. See
> `docs/implementation-notes.md` for the non-obvious traps and
> `docs/kmo_prompt_template_A1.original.txt` for the pristine prompt extract.

---

## 1. Project Objective

Build an automated, featherweight "Research Radar" that:

1. Harvests and full-text-extracts a **balanced robotics corpus** across three
   conflicting paradigms:
   * **Regime A:** Classical System Identification (SysID).
   * **Regime B:** Domain Randomization & Sim-to-Real.
   * **Regime C:** Deep Reinforcement Learning & Stochastic Policies.
2. Builds a **sparse character n-gram TF-IDF** representation in two
   normalisations — no language model is used in any objective (paper §5.5).
3. Reconstructs **whole words** from those n-grams so the evidence packages a
   language model later reads contain dictionary terms, not fragments
   (Appendix A.1).
4. Projects the corpus onto a bounded chart `[-1, 1]^2` and evaluates the four
   KMO discovery objectives on a `21 x 21` grid.
5. Runs an **exact pairwise Pareto search**, keeps the practical front, and ranks
   it by distance to the ideal corner.
6. Emits an interactive dashboard, a Markdown report, and a paste-ready
   evidence-constrained prompt in the paper's own Appendix A.1 wording.

---

## 2. What changed from the first draft of this plan

The original draft described an Ollama/`F2LLM` neural embedding pipeline. That
was a reasonable first guess, but the paper specifies a **character n-gram TF-IDF**
representation and states that no language model participates in the geometry.
The neural path has been removed from the analysis; `src/embed.py` now builds the
sparse representation. Ollama is no longer required to reproduce any result.

| First draft | Implemented |
| --- | --- |
| `F2LLM-4B` via Ollama, `data/embeddings.npy` | char n-gram TF-IDF (`n=4..7`), `data/sparse_tfidf.npz` |
| `title + abstract` text | **full text**, extracted with `mutool`, de-bibliographied |
| Abstract-only corpus | `data/corpus_fulltext.json` + `data/corpus.json` metadata |
| Single `B(x)` | three boundary views, harmonic mean (§4.2) |
| `G(x) = ||∇v(x)||_F` via analytic projection | discrete central difference of `v` |
| 4 objectives only | 4 objectives + `R` reliability diagnostic (§3.7) |
| — | `src/audit.py` post-generation checklist (Appendix A.2) |

---

## 3. Directory Structure

```text
ResearchRadar/
├── PLAN.md                       # This file
├── README.md                     # Quick start and verification
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # + pytest
├── config.yaml                   # Queries, TF-IDF, manifold, Pareto, audit
├── src/
│   ├── common.py                 # Config, paths, logging
│   ├── harvest.py                # arXiv queries + PDF/ar5iv extraction + sanitation
│   ├── embed.py                  # Dual-norm char n-gram TF-IDF + word reconstruction
│   ├── manifold.py               # MDS, SPH, 3-view boundary, 4 objectives, GPR
│   ├── pareto.py                 # Exact dominance, practical front, archetypes
│   ├── report.py                 # Plotly map, Markdown report, prompt payload
│   ├── dashboard.py              # Standalone interactive dashboard
│   └── audit.py                  # Appendix A.2 checklist for generated output
├── templates/
│   └── dashboard.html            # Dashboard shell, payload injected at build time
├── tests/
│   ├── test_sanitize.py          # Bibliography / citation strippers
│   ├── test_representation.py    # N-gram matrix, word index, persistence
│   ├── test_manifold.py          # Eqs. 12, 21, 13-18
│   └── dashboard_smoke.js        # Runs the page's own script under a DOM shim
├── docs/
│   ├── implementation-notes.md   # Environment traps and design decisions
│   ├── kmo_prompt_template_A1.txt          # In use (one documented edit)
│   └── kmo_prompt_template_A1.original.txt # Pristine extract from the PDF
├── data/
│   ├── corpus.json               # Metadata for the 120 papers
│   ├── corpus_fulltext.json      # + sanitised full text
│   ├── sparse_tfidf.npz          # Z_l2 and Z_l1
│   ├── tfidf_vocab.json          # n-gram terms + whole-word families
│   ├── manifold_grid.npz         # Grid, objectives, all fields
│   └── pareto_solutions.json     # Ranked practical front
└── output/
    ├── index.html                # Dashboard (git-ignored, regenerate locally)
    ├── radar_map.html            # Plotly map (git-ignored)
    ├── RADAR_REPORT.md           # Report
    └── KMO_PROMPT_TOP1.txt       # Paste-ready prompt for gap #1
```

---

## 4. Phased Implementation

### Phase 1: Corpus Acquisition and Full-Text Extraction (`src/harvest.py`)

**Objective:** a 120-paper robotics corpus with clean full text, founded on
user-curated seeding papers (**Seed-First Expansion**).

1. **Extract the seeding papers first.** Every PDF in `docs/papers/desired/` is
   sanitised and becomes `doc 1` … `doc 15` with `"is_seed": true`. Seeds are
   kept unconditionally, whatever their year: they define what the map is about.
2. **Derive the recency window from the seeds.** The harvest start year is
   `min(seed_years) - corpus.seed_year_offset` (default 2), floored at
   `now - corpus.lookback_years` (default 10), so the window scales with the
   seed set instead of being a fixed constant.
3. Query arXiv with the three regime queries in `config.yaml` plus a
   `submittedDate` clause, paginate, and deduplicate on both arXiv id and
   normalised title, seeding the seen-id set with the seeds' own arXiv ids so no
   seed is harvested twice (`35` harvested per regime = 105).
4. Reject review/survey papers by title keyword and by opening-sentence pattern
   (paper §5.1 explains why review articles are excluded).
5. **Assign each seed a regime** — Priority 1 is an explicit
   `corpus.seed_overrides` entry, Priority 2 is the closest regime by TF-IDF
   cosine similarity against that regime's centroid (its harvested documents
   plus its keyword bundle). Every assignment is logged with its cosine and
   margin, and an ambiguous margin is flagged.
6. Extract text: `mutool draw -F txt` for seeding PDFs in `docs/papers/desired/`,
   the arXiv PDF for harvested papers, with an **ar5iv HTML fallback**.  A seed
   saved under a free-form filename is resolved to its arXiv record by exact
   title match, which supplies a publication year and a working link.
7. **Sanitise before the vocabulary exists** — this is the step that makes
   Appendix A.1 Rule 4 ("none is a bibliographic fragment") true:
   * remove running headers, front matter, and the References section, with a
     body-length guard so an in-text "references" cannot truncate a paper;
   * strip bracket citations, author-year callouts, URLs, DOIs, arXiv ids;
   * verify zero bibliographic remnants survive.
8. Quality gate: `>= 1000` words, mean word length in `[3.5, 9.0]`, and an
   alphanumeric ratio of at least `0.75` **over non-whitespace characters**
   (a total-character denominator rejects equation-dense control papers).
9. Write **both** artifacts from the same ordered list: `data/corpus.json`
   (metadata, no `full_text`) and `data/corpus_fulltext.json` (with `full_text`),
   so every downstream stage reads identical, identically-ordered papers.

A rebuild starts clean: the corpus contains exactly this run's seeds plus this
run's harvest, never papers inherited from an earlier corpus.

### Phase 2: Sparse Representation (`src/embed.py`)

**Objective:** the paper's §5.2 representation, exactly.

1. **Within-word character n-grams**, `n = 4..7`, `analyzer="char_wb"`, so an
   n-gram never straddles a word boundary.
2. Sublinear term frequency `tf' = 1 + log(tf)`.
3. **Smoothed IDF `log(1 + N/df) + 1`** — not scikit-learn's
   `smooth_idf = log((1+N)/(1+df))`; the two differ on rare terms, which is
   exactly where the L1 lens looks.
4. Cap the vocabulary at 250,000 features.
5. Store both normalisations of **one** matrix: `Z_l2` (map and gradient) and
   `Z_l1` (feature allocation), rows summing to 1.
6. Build the **n-gram -> whole-word index**: map each retained n-gram back to the
   corpus words it was attested in, group surface forms into families, and record
   each family's attested `word_forms` and document frequency. Without this the
   lens would be fragments (`vort`, `ortic`) rather than words.
7. Fingerprint the build (version + settings + family rule) and refuse to reuse a
   cached matrix whose fingerprint changed.

### Phase 3: Knowledge Manifold and the Four Objectives (`src/manifold.py`)

**Objective:** §3.3–§3.6, §4.2 on a `21 x 21` grid.

1. **Projection.** Cosine distances between `Z_l2` rows, classical MDS as the
   start, then constrained stress minimisation with **9 gauge anchors** (8
   perimeter + 1 central). Coordinates are frozen afterwards.
2. **Grid.** 441 points, spacing 0.1, `x`-major ordering.
3. **SPH weights.** `w_i(x) = exp(-||x - x_i||^2 / (2 h^2))`, `h = 0.25` globally,
   plus an adaptive field with `h_i` = the document's 8th-neighbour distance.
4. **`P_boundary` — three views, harmonic mean (§4.2):**

   ```text
   B(x)        = 0.65 * N+(H_K(x)) + 0.35 * N-(Δp_K(x))
   P_boundary  = 3 / (1/B_k5_global + 1/B_k8_global + 1/B_k8_adaptive)
   ```

   `H_K = -(1/log K) Σ p_c log(p_c + ε)`; `Δp_K` is the gap between the two
   largest cluster probabilities; `N+`, `N-` are 10th/90th percentile
   normalisations. K=5 is the Discovery level and K=8 the Reference level.
5. **`W` — band-pass white space (§3.4).** Zero below `q20` and above `q98`, a
   linear ramp to 1.0 at `q60`, flat 1.0 through `q85`, ramping down to `q98`.
6. **`G` — semantic gradient (§3.5).** `v(x) = normalize(Σ w_i(x) z_i)`, then
   `g(x) = ||J_v||_F` by **central differences of `v`**, `G = N+(g)`.
7. **`S` — evidential support (§3.6).**
   `S = 0.40 N+(N_eff) + 0.30 N-(d5) + 0.30 N-(T3)`.
8. **`R` — reliability diagnostic (§3.7).** `0.5 N-(u_RBF) + 0.5 N-(u_M32)` from
   two GPR surrogates. Reported, **never** an objective.
9. **Table 4 baseline (§8.8).** The 16 largest-`d_nn` points over the full grid
   and over the interior only, compared with the front on mean `d_nn, S, R, P, G`.
10. Write `data/manifold_grid.npz` + manifest.

### Phase 4: Exact Pareto Search (`src/pareto.py`)

1. Exact pairwise dominance over all 441 points on `[P_boundary, W, G, S]`;
   no evolutionary search is needed at this size (§4.6).
2. Practical front: keep only `W(x) > 0` (§4.5, Eq. 25).
3. Rank by Euclidean distance to the ideal corner `(1, 1, 1, 1)`.
4. Group nearby solutions into archetypes by single-linkage at a fixed radius.
5. Write `data/pareto_solutions.json`.

### Phase 5: Reporting (`src/report.py`, `src/dashboard.py`)

1. **Evidence package per gap (§7.1).**
   * `theme_lens_L2` — whole words ranked by local `ℓ2` weight.
   * `concentration_lens_L1` — whole words ranked by local/global enrichment in
     the `ℓ1` view, each with `df` and, at `df <= 3`, its `source_docs` cited as
     `[doc N]`.
   * `contributing_documents_top3` — highest SPH weights, in `doc N` form.
   * `mixture_entropy_H`, `effective_contributing_documents` (`N_eff`),
     `relative_uncertainty_u` (**RBF surrogate**).
2. **Markdown report.** Front table, archetype deep dives with top-5 anchors,
   L2-vs-L1 lens analysis split at `df >= 4` / `df <= 3`, the Table 4 baseline,
   and the prompt in §7 below.
3. **Dashboard.** Two-panel standalone page: the manifold with toggleable field
   layers on the left, the evidence drawer with a 4-axis radar, anchors, lenses,
   and a copyable prompt on the right.

### Phase 6: Verification and Audit (`src/audit.py`)

Appendix A.2 applied to whatever a language model returns for the prompt. Eight
checks, each reported with PASS/FAIL and the evidence that triggered it:

| # | Item | Rule |
|---|------|------|
| 1 | `length_compliance` | Task A in 200–280 words; Task B exactly 45 |
| 2 | `evidence_coverage` | every supplied term used **or** explicitly set aside (Rule 4) |
| 3 | `lens_discipline` | L2 defines the theme; L1 is supporting evidence |
| 4 | `low_frequency_provenance` | `df <= 3` L1-only terms carry `[doc N]` in the same sentence |
| 5 | `unsupported_claim_audit` | no unhedged mechanism, material, number, or citation |
| 6 | `novelty_wording` | novelty corpus-relative, never global |
| 7 | `uncertainty_disclosure` | the caveat names both `u` and `N_eff` |
| 8 | `evidence_trace` | an `Evidence used:` list records terms and their `df` |

```bash
python -m src.audit --input response.txt --gap 1     # human-readable
python -m src.audit --input response.txt --gap 1 --json
```

Exit code 0 means every check passed, 1 means at least one failed. Thresholds
live under `audit:` in `config.yaml`.

Item 5 is the only heuristic that can be wrong in both directions, so it is a
**flagging** check by design: a sentence is reported when it asserts a mechanism,
a material, a percentage/decimal, or an external citation and carries no hedge.
A.2 states the audit exists "to expose unsupported extrapolation rather than to
claim a mathematical guarantee of hallucination-free generation", and the report
repeats that limitation.

---

## 5. Reproduce

```bash
python -m src.harvest              # or --verify-only
python -m src.harvest --extract-text
python -m src.embed
python -m src.manifold
python -m src.pareto
python -m src.report
python -m src.dashboard

# After an LLM answers output/KMO_PROMPT_TOP1.txt:
python -m src.audit --input response.txt --gap 1
```

Each stage supports `--verify-only`. `pytest` covers the sanitisers, the
representation, the word index, and the equations that are easy to get wrong
(the Eq. 21 worked example in particular).
