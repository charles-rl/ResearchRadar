# ResearchRadar

A research-landscape radar for robotics control: a 1:1 reproduction of
Obayashi et al. (2026), *Knowledge Manifold Optimization*, applied to robotics.

The pipeline harvests an arXiv corpus across three configurable research
regimes, extracts
full text, builds a sparse character n-gram TF-IDF representation, projects the
corpus onto a bounded 2-D chart, and scores every grid location on boundary
pressure, band-pass whitespace, semantic gradient, and evidential support. The
non-dominated locations with genuine whitespace are the candidate research gaps.

No language model participates in any objective (paper Sec. 5.5). A prompt is
generated so a language model can verbalize a gap *after* the geometry is fixed.

Implemented: **Phase 1** (environment + corpus + full text), **Phase 2**
(harvest + sanitation), **Phase 3** (sparse representation), **Phase 4**
(manifold and discovery fields), **Phase 5** (Pareto search), **Phase 6**
(report, dashboard, prompt payload), and **Phase 7** (the Appendix A.2 audit that
checks whatever an LLM returns for that prompt).

## Two modes, one switch

`config.yaml` starts with `profile:`, and every stage reads it:

| `profile` | Corpus | Regimes |
| --- | --- | --- |
| `seed_first` (default) | your PDFs in `docs/papers/desired/` plus a per-regime harvest | declared in `config.yaml::regimes` |
| `single_field` | one broad query harvested to `single_field.target_total_papers`, no seeds | discovered after projection, named from each cluster's titles and abstracts |

In `single_field` (Zero-Seed) mode the regimes below are not used: `src.manifold`
clusters the finished chart at Discovery K=5 and Reference K=8 (paper Sec. 3.2),
names each region from the phrases that over-represent it inside its own titles
and abstracts (full text would name a region after a cited author), and writes
those names into the corpus, so `src.pareto`, `src.report`, and the dashboard all
pick them up without knowing the mode exists. `P_boundary` is unchanged in both
modes — it is the harmonic mean over `k5_global`, `k8_global`, and `k8_adaptive`
(Eq. 21) and never reads the regime labels.

```yaml
profile: single_field
single_field:
  query: 'abs:"robot manipulation" AND cat:cs.RO'
  target_total_papers: 100
  years_back: 5
  discovery_k: 5
  reference_k: 8
```

Both profiles write the same artifact paths, so running one replaces the other's
corpus. See `docs/AGENT_RUNBOOK.md` §5 for the full checklist.

## Regimes

Applies to `profile: seed_first`.

| ID | Label | Query |
| --- | --- | --- |
| 0 | Zero-Shot RL & Predictive Representations | `(abs:"zero-shot" OR abs:"forward-backward" OR abs:"latent dynamics" OR abs:"successor feature") AND abs:"reinforcement learning" AND (cat:cs.RO OR cat:cs.LG)` |
| 1 | Sim-to-Real & Hardware Physics Adaptation | `(abs:"sim-to-real" OR abs:"domain randomization") AND (abs:"friction" OR abs:"actuator" OR abs:"contact dynamics" OR abs:"contact-rich" OR abs:"system identification") AND cat:cs.RO` |
| 2 | Robot Arm Manipulation & Vision-Language-Action Policies | `(abs:"robot manipulation" OR abs:"dexterous manipulation" OR abs:"robotic grasping" OR abs:"vision-language-action") AND (abs:"manipulation policy" OR abs:"imitation learning" OR abs:"reinforcement learning" OR abs:"policy learning" OR abs:"sim-to-real") AND cat:cs.RO` |

35 harvested papers per regime, plus the 15 seeding papers, 120 documents total.
The regimes are defined entirely by `config.yaml`; nothing downstream assumes the
labels above, so a taxonomy change is a config edit plus a re-harvest. Regime 2
was previously "Dynamic Legged Control & Trajectory Optimization"; its seeding
papers are archived in `docs/papers/desired_locomotion/` (outside the
non-recursive seed glob, so they are inert).

## Seeding papers (Seed-First Expansion)

The corpus does not start from a literature search. It starts from the PDFs you
place in `docs/papers/desired/` — the papers that define what this radar is
about. They become `doc 1` … `doc 15` with `"is_seed": true`, and they are kept
whatever their year.

Three things follow from the seeds:

1. **The recency window scales with them.** Harvest starts at
   `min(seed_year) - 2`, clamped so it never reaches further back than
   `corpus.lookback_years` (12), so adding an older seed widens the window
   instead of leaving a temporal gap between the seeds and the harvested
   literature, while one very old seed cannot drag the corpus back a decade.
2. **The harvest cannot duplicate them.** Seed arXiv ids are pre-registered as
   seen with the version suffix stripped, so a seed named `2511.04131.pdf`
   still blocks `2511.04131v2` from the API, and seed titles are pre-registered
   too. A seed whose PDF has a free-form filename (`TD JEPA.pdf`) is resolved to
   its arXiv record by exact title match, which supplies its year and a working
   link.
3. **A rebuild starts clean.** The corpus is exactly this run's seeds plus this
   run's harvest. Nothing is inherited from an earlier corpus.

Each seed is placed in a regime by this priority:

1. `corpus.seed_overrides` in `config.yaml` — `{"2511.04131.pdf": 0}`.
2. Otherwise, the regime whose centroid (its harvested documents plus its keyword
   bundle) is closest in TF-IDF cosine similarity. The cosine and the margin are
   logged for every seed, and a margin below `corpus.seed_match_margin` is
   flagged as ambiguous so you can pin it with an override.

Regime counts therefore are `35 + seeds assigned to that regime`. On the map,
seeds are the hollow diamonds, drawn separately from the filled regime dots so
their positions read against the gap stars.

## Quick start

```bash
python3 -m venv .venv
PIP_CONFIG_FILE=/dev/null .venv/bin/pip install --isolated \
    --index-url https://pypi.org/simple -r requirements.txt

.venv/bin/python -m src.harvest --extract-text
                                     # -> data/corpus.json  (120 documents:
                                     #    15 seeding papers + 105 harvested)
                                     # -> data/corpus_fulltext.json
.venv/bin/python -m src.embed        # -> data/sparse_tfidf.npz (120 x 250000)
.venv/bin/python -m src.manifold     # -> data/manifold_grid.npz  (441-point grid)
.venv/bin/python -m src.pareto       # -> data/pareto_solutions.json
.venv/bin/python -m src.report       # -> output/radar_map.html
                                     #    output/RADAR_REPORT.md
                                     #    output/KMO_PROMPT_TOP1.txt
.venv/bin/python -m src.dashboard    # -> output/index.html (standalone app)

# optional: place one external document of your own on the finished map
.venv/bin/python -m src.explore      # -> data/explorer_idea.json
                                     #    output/IDEA_PLACEMENT.md
.venv/bin/python -m src.dashboard    # re-render to overlay it on the page

# after an LLM answers output/KMO_PROMPT_TOP1.txt:
.venv/bin/python -m src.audit --input response.txt --gap 1
```

Every module supports offline re-verification:

```bash
.venv/bin/python -m src.harvest  --verify-only
.venv/bin/python -m src.harvest  --verify-fulltext
.venv/bin/python -m src.embed    --verify-only
.venv/bin/python -m src.manifold --verify-only
.venv/bin/python -m src.pareto   --verify-only
.venv/bin/python -m src.report   --verify-only
.venv/bin/python -m src.dashboard --verify-only
.venv/bin/python -m src.explore   --verify-only
node tests/dashboard_smoke.js output/index.html
```

## Why the pip flags

`/etc/pip.conf` on some hosts injects `extra-index-url =
https://pypi.ngc.nvidia.com`. When that host is unreachable, pip enters a long
retry storm for every candidate. `PIP_CONFIG_FILE=/dev/null --isolated
--index-url https://pypi.org/simple` bypasses the system config and pins the
index explicitly.

## Serving F2LLM with Ollama (not used by the analysis)

Ollama cannot import these checkpoints from their safetensors directories. Its
safetensors importer builds **MLX** models only, which fails on Linux with
`unsupported MLX architecture: model "Qwen3Model"` and, if forced, a
`MLX runtime is not available` error at request time. Convert to GGUF first.

```bash
# 1. Convert with a local llama.cpp checkout (no project dependencies needed).
git clone https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
python3 -m venv --system-site-packages .venv-convert
PIP_CONFIG_FILE=/dev/null .venv-convert/bin/pip install --isolated \
    --index-url https://pypi.org/simple sentencepiece

PYTHONPATH=/path/to/llama.cpp/gguf-py .venv-convert/bin/python \
    /path/to/llama.cpp/convert_hf_to_gguf.py models/F2LLM-4B \
    --outfile models/gguf/F2LLM-4B-bf16.gguf --outtype bf16

# 2. Register with Ollama.
printf 'FROM ./models/gguf/F2LLM-4B-bf16.gguf\n' > Modelfile.f2llm-4b
~/charles/bin/ollama serve &
~/charles/bin/ollama create codefuse-ai/F2LLM-4B -f Modelfile.f2llm-4b
```

`sentencepiece` is required even for a byte-level BPE Qwen tokenizer: the
converter's `Qwen2Model.set_vocab()` tries SentencePiece first and only falls
back to the GPT-2 path on `FileNotFoundError`, not on `ImportError`.

Confirm the converted file kept LAST-token pooling, which is what F2LLM was
trained with:

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py .venv-convert/bin/python -c '
import gguf; r = gguf.GGUFReader("models/gguf/F2LLM-4B-bf16.gguf")
print(r.fields["qwen3.pooling_type"].contents())'   # 3 == LAST
```

| Model | Ollama tag | Dimensions | Weights size |
| --- | --- | --- | --- |
| F2LLM-4B | `codefuse-ai/F2LLM-4B` | 2560 | 7.6 GB |
| F2LLM-0.6B | `codefuse-ai/F2LLM-0.6B` | 1024 | 1.2 GB |

The 4B model is the default. Switch to the 0.6B model with
`--model codefuse-ai/F2LLM-0.6B`, or by editing `ollama.fallback_model` if VRAM
is tight.

The two models are different, not copies of each other: they produce 2560- and
1024-dimensional vectors. Matrices built by different models are not comparable,
so the embedding cache keys on the model tag and switching forces a re-embed.

`models/gguf/` is deleted once registration succeeds. `ollama create` copies the
GGUF into its own blob store, so the intermediates are redundant; rebuild them
from `models/F2LLM-*` with the recipe above only if you need to re-register.

## Corpus and text sanitation

* Text comes from the arXiv PDF via `mutool draw -F txt`, with an ar5iv HTML
  fallback. All 120 papers extract cleanly; the seed PDFs in
  `docs/papers/desired/` do too.
* **Sanitation runs before the vocabulary exists.** Running headers, front
  matter, the References section, bracket and author-year citations, URLs,
  DOIs, and arXiv ids are removed, with a body-length guard so an in-text
  "references" cannot truncate a paper. Verified: zero bibliographic remnants
  survive. This is what makes Appendix A.1 Rule 4 hold.
* Quality gate: `>= 1000` words, mean word length in `[3.5, 9.0]`, and an
  alphanumeric ratio of at least `0.75` over non-whitespace characters. The
  denominator matters: dividing by all characters rejects equation-dense
  control theory.

## Representation

* **Character n-grams, `n = 4..7`, `analyzer="char_wb"`**, sublinear
  `tf' = 1 + log(tf)`, smoothed `idf = log(1 + N/df) + 1`, vocabulary capped at
  250,000. The IDF is computed explicitly because scikit-learn's `smooth_idf`
  uses `log((1+N)/(1+df))` and the two differ on rare terms.
* Two normalisations of **one** matrix: `Z_l2` (map, gradient) and `Z_l1`
  (feature allocation), the latter with rows summing to 1.
* **Whole words are reconstructed from the n-grams.** Every retained n-gram is
  mapped back to the corpus words it was attested in, and surface forms are
  grouped into families. The evidence lenses then aggregate n-gram scores onto
  whole words with their attested `word_forms`, so a language model reads
  `vortex`, not `vort`.
* Caching keys on the corpus MD5 **and** a build fingerprint, so changing the
  n-gram range or the family rule forces a rebuild instead of serving a stale
  matrix.

## Manifold, Pareto front, and report

* **Projection.** Classical MDS, then constrained stress minimization with 9
  gauge-fixing anchors (8 perimeter + 1 central) onto `[-1, 1]^2`.
  Final metric stress 578.95, Spearman(distance) 0.7053.
* **Grid.** 21x21 = 441 candidates, scored on `P_boundary`, `W`, `G`, and `S`.
* **Three boundary views.** `B_k5_global`, `B_k8_global`, `B_k8_adaptive`
  combined by harmonic mean (Eq. 21), not a single `B`. K=5 is the Discovery
  level, K=8 the Reference level.
* **White-space window.** The `q20/q60/q85/q98` band-pass thresholds are
  recomputed on this corpus's own nearest-document distances
  (`manifold.use_empirical_quantiles: true`). The paper's values are for its
  aerodynamic corpus at a different map scale and are kept only as reference.
* **Gradient.** `G = N+(‖J_v‖_F)` with the Jacobian of
  `v(x) = normalize(Σ wᵢ zᵢ)` taken by **discrete central differences**, because
  `v` is nonlinear and the analytic first-order projection disagrees with the
  discrete difference by a factor of four.
* **Pareto search.** Exact pairwise dominance, no evolutionary optimizer. 65
  non-dominated points, 60 practical after the `W(x) > 0` filter, grouped into
  17 discovery archetypes at `archetype_radius: 0.35`. The best compromise is
  `#1` at `(+0.20, -0.80)` with `P_boundary=0.963`, `W=1.000`, `G=0.789`,
  `S=0.517`, `d_ideal=0.5283`.
* **Table 4 baseline.** The 16 largest-`d_nn` grid points, over the whole grid
  and over the interior only, are scored alongside the front. Every
  unconstrained pick lands on the chart edge with support near zero, which is
  the paper's argument that distance alone is not a discovery criterion.
* **Reliability `R`** is computed and reported but is *not* a Pareto objective:
  maximizing low uncertainty rewards already-dense regions.
* **Evidence packages.** Terms are cited as `[doc N]` (one-based corpus order),
  with the arXiv id kept as a secondary field. `relative_uncertainty_u` is the
  **RBF** surrogate standard deviation; the averaged dual-kernel `R` stays a
  diagnostic.
* **The prompt.** `docs/kmo_prompt_template_A1.txt` holds the Appendix A.1 text
  from the paper with exactly one documented edit: Rule 3 says "robotics" where
  the paper says "composites". `docs/kmo_prompt_template_A1.original.txt` keeps
  the pristine extract, `src/report.py` only substitutes the placeholder file
  name, and `--verify-only` asserts the two files differ on that one line only
  and compares the rendered prompt to the template line by line.

## The dashboard

`output/index.html` is a single self-contained page: the corpus, the four
objective fields, and one evidence package per gap are baked into an inline JSON
payload, so it opens straight from the filesystem with no server and no build
step. Plotly and Lucide load from CDNs (pinned, with SRI hashes); all styling is
plain CSS inlined in the page, so there is no CSS framework in the load path.

* **Left panel** — the manifold on `[-1, 1]^2`, papers coloured by regime, the 60
  practical gaps as stars, and layer buttons (keys `1`-`5`) that swap the
  background between `P_boundary`, `W`, `G`, and `S`. The axes are pinned 1:1, so
  the plot stays square and letterboxes inside a wide pane. The regime legend is
  generated from the payload, so it cannot drift from `config.yaml`, and the gap
  stars use one neutral intensity ramp rather than the regime hues.
* **Right panel** — the selected gap's evidence package: a 4-axis objective
  radar, support statistics, the regime mixture, the top-5 anchor papers, the
  L2/L1 lexical split, and the Appendix A.1 prompt with a copy button.
* **Navigation** — click a star, use the dropdown, or press the arrow keys.
  `#gap=<rank>` deep-links a gap.

The page is responsive without a framework: two stacked panes below 1024px, a
side-by-side split above it, and a 3x2 stat grid in the header on phones.

Why no CSS framework: `cdn.tailwindcss.com` is the only asset in the page that
sends **no** `Access-Control-Allow-Origin` header, so a browser blocks it outright
once the page is served over `http://` — see `docs/implementation-notes.md`.

The prompt text comes from `src.report.render_prompt_text`, the same function
the Markdown report uses, so the pasted prompt and `KMO_PROMPT_TOP1.txt` are
byte-identical — `src.dashboard --verify-only` asserts exactly that.

`tests/dashboard_smoke.js` runs the page's own inline script against the real
payload under a minimal DOM shim, so it catches renderer bugs without a browser:

```bash
node tests/dashboard_smoke.js output/index.html
```

To check the rendered page rather than its logic, serve `output/` and screenshot
it with Playwright's Chromium. This is a throwaway toolchain, deliberately kept
out of the project:

```bash
mkdir -p /tmp/pw && cd /tmp/pw && npm init -y && npm i playwright@1.56.1
PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers npx playwright install chromium
```

Screenshot at 1600x950, 1440x820, 1024x768 and 430x900. The narrower widths are
what catch layout drift: the map's grid row and the header metric strip both
broke silently at 1600px alone.

## Placing your own document on the map

`src/explore.py` answers a different question from the rest of the pipeline:
*given this map, where does my idea sit, and is it novel?* It takes one document
from outside the corpus, represents it with the same character n-gram TF-IDF, and
projects it onto the finished manifold. Nothing is recomputed: the corpus, the
discovery grid, the four objectives, and the Pareto front are read-only inputs,
so the overlay can never move a paper, a field, or a gap.

```yaml
explorer:
  document: docs/papers/Current_Research_Proposal.pdf  # UTF-8 text or a PDF
  label: Your proposal                    # legend/drawer label
  output_path: data/explorer_idea.json
  report_path: output/IDEA_PLACEMENT.md
```

A PDF is read through the corpus's own `mutool` extraction and sanitation chain
(`src.harvest.extract_pdf_text`), not as raw bytes: running headers, the
bibliography and citation call-outs are dropped and the hard line breaks a PDF
text layer leaves mid-sentence are repaired. That matters because a proposal's
reference list is full of *other papers' titles* — embedding it would score the
bibliography as if it were the author's argument. The digest recorded in the
artifact is taken over the extracted text, so the "is this byte-identical to a
corpus paper" guard keeps working for both input types. A scanned PDF with no
text layer fails loudly instead of scoring an empty document.

What the module does, in order:

1. **Represent** the document with the corpus vocabulary (markdown and LaTeX
   noise stripped first, and a PDF sanitized as described above). The share of its
   n-grams the corpus vocabulary knows is
   reported as `vocabulary_match_rate` — low values mean the document is talking
   about something the corpus does not, which is itself a novelty signal.
2. **Place** it on `[-1, 1]^2`. Raw cosine similarity cannot be trilaterated
   directly (the chart's distances are MDS distances, not text distances), so the
   module first fits an isotonic map *cosine distance → chart distance* on all
   corpus pairs, then solves a bounded least-squares trilateration against the
   calibrated distances. The isotonic fit is validated by leave-one-out: the
   reported median/p90 position error is the honest uncertainty of the placement,
   in chart units, next to the typical nearest-neighbour spacing for scale.
   `at_chart_edge` and a warning fire when the fit runs into the boundary or the
   residual exceeds that calibration error.
3. **Score** the point with the same four objectives used for the grid
   (`P_boundary`, `W`, `G`, `S`), plus the percentile of each value among all
   candidates, `d_ideal`, and the rank the point *would* take — reported as a
   hypothetical, because an external document is not a front candidate.
4. **Read the neighbourhood**: nearest corpus documents *twice* — once ranked by
   text similarity (cosine, what the document is about) and once ranked by chart
   distance (where the projection puts it). The chart is a 2-D projection, so its
   nearest neighbour is often not the text-nearest one: in the reference run the
   closest paper was text rank 1 but chart rank 26, and showing only the chart
   order hid it entirely (text rank 1, chart rank 25 in the reference run). The card and the report therefore list both orderings,
   each with the other rank attached. Also reported: the nearest practical gaps,
   the regime mixture at that point, and two term lists — words the document leans
   on that the corpus barely covers (`df <= 3`, i.e. candidate novelty) and words
   shared with the map's own theme lens.

**The idea has no gap rank.** The front is a finite set of *candidates* ranked by
`d_ideal`, and an external document is not one of them, so interleaving it into
that ranking would be arithmetic pretending to be membership. `gap_reference`
reports four facts instead, and the card shows all four:

- `nearest_chart` — the gap nearest on the map. A projection neighbour, not a
  similarity claim: in the reference run gap #50 is 0.108 chart units away but
  1.261 away in `d_ideal`, i.e. farther than the idea itself (1.052).
- `nearest_objective` — the gap nearest in the four-objective space. This is the
  proposal the idea is actually similar to; the 2-D chart frequently puts it far
  away (gap #24, Δ 0.103, at 0.53 chart units).
- `insertion` — the slot the point would take if inserted by `d_ideal`
  (#25 of 56 in the reference run), explicitly labelled hypothetical.
- `dominated_by` / `on_front` — the decisive one. A gap that scores at least as
  high on all four objectives dominates the point, and a dominated point cannot
  be on the front however good its `d_ideal` looks. In the reference run no gap
  and none of the 441 grid points dominates the proposal on all four, so at that
  coordinate it would join the front — and it strictly dominates gaps #35 and
  #44. **Read that as fragile, not as a result.** The placement's own error
  (residual 0.255 chart units, leave-one-out median 0.189) is larger than the
  distance to the nearest dominating region: sampling 400 points inside the
  residual radius, 86% are dominated by *some* gap, and the dominator changes
  from sample to sample (#26, #25, #39, #36, #43, #2, #1, #13 all appear). A
  four-axis non-domination verdict is not stable at the precision this placement
  has, so the report prints the frontier fact and the residual next to it.

`W` needs one caveat next to its value. The band-pass (Eq. 12) saturates between
`q60` and `q85`, so `W = 1.000` is the score of a wide plateau, not a unique
position: `grid_context.white_space_plateau_share` reports the share of grid
points that score the same (25.6% of 441 in the reference run), and both the card
and the report print it beside the value. A plateau score means the nearest
document is at a healthy distance; it does not mean the idea sits somewhere
special.

Deliberately absent: a verdict. The module reports numbers and percentiles and
stops there. Whether "novel enough" matters is a judgement the map cannot make,
and `output/IDEA_PLACEMENT.md` is written to be read next to the page.

`plain_language` sits beside the numbers rather than replacing them. It turns
four scores into sentences using fixed thresholds (no model, no generation), so
every claim can be checked against the value printed next to it: what the
position is (`P_boundary` 57th = near a seam, not on it), what the novelty
evidence actually is (mean text-distance percentile 6.7, not the saturated `W`),
and what front membership does and does not mean (no gap dominates on all four
at this exact coordinate, which says where unexplored combinations score higher,
not whether the work is publishable).
The card and the report print the same sentences, because the page passes the
artifact's strings through instead of re-deriving them.

On the dashboard the result is a cyan diamond labelled *your idea*, distinct
from the paper dots and the gap stars, with a card at the top of the drawer
(click the marker or press `i`). Focusing the idea replaces the gap panes with
that one card, so no gap number can sit under it and read as the idea's rank; the
card's button (or `Esc`, or any arrow key) returns to the selected gap. The
overlay is optional: without `data/explorer_idea.json` the page is exactly the
corpus dashboard, and `tests/dashboard_smoke.js` passes in both states.

## Auditing an LLM answer (Appendix A.2)

The prompt in `output/KMO_PROMPT_TOP1.txt` is meant to be pasted into a language
model. Appendix A.2 says the answer must then be checked, because nothing in the
prompt can stop a model from inventing a mechanism or a number. Save the answer
to a file and run:

```bash
python -m src.audit --input response.txt --gap 1        # human-readable
python -m src.audit --input response.txt --gap 1 --json # machine-readable
```

Exit code is `0` when all eight checks pass and `1` when any fails. Each failure
names the offending terms or sentences rather than just reporting a count:

```text
[FAIL] length_compliance: Task B has 41 words, expected exactly 45
[FAIL] novelty_wording: Global-novelty phrasing without a corpus-relative
       qualifier: the first time
[FAIL] unsupported_claim_audit: 2 sentence(s) assert claims without a hedge
```

`--package <evidence.json>` audits against a saved evidence package instead of
regenerating it from the grid. Thresholds live under `audit:` in `config.yaml`.

The checks are conservative heuristics, and the report repeats the paper's own
scope limit: A.2 exists "to expose unsupported extrapolation rather than to claim
a mathematical guarantee of hallucination-free generation". Item 5 (unsupported
claims) is the one that can be wrong in both directions, so it flags sentences
for a human instead of returning a verdict.

## Layout

```text
config.yaml                       queries, limits, model/analysis parameters
docs/AGENT_RUNBOOK.md             operating checklist: run order, seeds, regimes, zero-seed
docs/implementation-notes.md      non-obvious environment traps and decisions
docs/kmo_prompt_template_A1.txt   Appendix A.1 template (one documented edit)
docs/kmo_prompt_template_A1.original.txt   pristine extract from the paper
src/harvest.py                    arXiv harvesting, PDF/ar5iv text, sanitation
src/embed.py                      char n-gram TF-IDF + whole-word reconstruction
src/common.py                     config loading, profile normalisation, paths, logging
src/regimes.py                    Zero-Seed discovery: chart clusters -> named regions
src/manifold.py                   MDS, discovery grid, the four KMO objectives
src/pareto.py                     exact dominance, practical front, archetypes
src/report.py                     Plotly map, Markdown report, LLM prompt
src/dashboard.py                  standalone output/index.html generator
src/explore.py                    places one external document on the map
src/audit.py                      Appendix A.2 checklist for LLM output
templates/dashboard.html          the page shell and its inline app script
tests/test_sanitize.py            bibliography and citation strippers
tests/test_representation.py      n-gram matrix, word index, persistence
tests/test_manifold.py            the equations that invite silent drift
tests/test_audit.py               the eight Appendix A.2 checks
tests/test_pareto.py              dominance orientation, archetype grouping
tests/test_seed_first.py          seed years, harvest window, front matter
                                  and version-insensitive seed deduplication
tests/dashboard_smoke.js          offline DOM-shim test of the dashboard script
docs/papers/desired/              the seeding PDFs (git-ignored; you supply these)
data/corpus.json                  the 120-document corpus (committed)
data/corpus_fulltext.json         sanitised full text (committed)
data/sparse_tfidf.npz             Z_l2 and Z_l1 (git-ignored)
data/tfidf_vocab.json             n-gram terms and whole-word families
data/manifold_grid.npz            grid coordinates and objective matrix
data/explorer_idea.json           external document placement (never an input)
output/IDEA_PLACEMENT.md          the same placement as a readable report
output/                           rendered artifacts (HTML git-ignored)
```
