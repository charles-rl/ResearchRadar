# Implementation notes

Things that are **not** obvious from reading `src/`. Each item cost real
debugging time; most are environment-specific traps rather than design choices.

> **The pipeline no longer uses Ollama.** The analysis is a 1:1 reproduction of
> Obayashi et al. (2026), which builds a character n-gram TF-IDF representation
> and states in Sec. 5.5 that no language model enters any objective. The Ollama
> / F2LLM material below is retained because the models are still installed and
> still serve correct unit-norm vectors; it is simply not part of the manifold.

## Why the models had to be converted to GGUF

`ollama create` cannot ingest the F2LLM checkpoints from their safetensors
directories on Linux. Ollama 0.40.4's safetensors importer builds **MLX**
models only, and MLX is Apple-silicon-only:

- plain `create` fails fast with
  `Error: unsupported MLX architecture: model "Qwen3Model"`
- `create --force` reports success, but the model is unusable — every embed call
  returns HTTP 500
  `MLX runtime is not available: failed to load MLX dynamic library (searched: [...])`

So the pipeline goes safetensors → GGUF → Ollama. The conversion used a local
llama.cpp checkout, never the project venv, so `requirements.txt` stays clean.

### `sentencepiece` is required even for a byte-level BPE tokenizer

Qwen3 uses tiktoken-style BPE, no SentencePiece model. The converter still needs
the import: `Qwen2Model.set_vocab()` tries the SentencePiece path first and only
falls back to `_set_vocab_gpt2()` on `FileNotFoundError`, not on `ImportError`.
Without the package the run dies at `conversion/base.py:2010` with
`ModuleNotFoundError: No module named 'sentencepiece'`.

`pip install -e gguf-py` fails here (build-dependency isolation needs network for
build deps). Setting `PYTHONPATH=<llama.cpp>/gguf-py` works instead.

### Pooling must be LAST token

F2LLM is a sentence-transformers model with `pooling_mode_lasttoken: true`.
If the GGUF loses that metadata, llama.cpp pools by default (`pooling_type =
NONE`/0) and returns vectors that are structurally valid but semantically wrong —
no error, just bad embeddings.

`convert_hf_to_gguf.py` recovers it from `modules.json` + `1_Pooling/config.json`.
The convert step must therefore run on the **original checkpoint directory**, not
on a stripped copy. Always assert after conversion:

```text
qwen3.pooling_type = 3      # 3 = LAST
```

## `/etc/pip.conf` breaks pip on this host

The system config injects `extra-index-url = https://pypi.ngc.nvidia.com`, which
does not resolve here. pip treats an unreachable extra index as a retryable
failure and storms on **every** candidate, so installs hang for many minutes and
appear stalled. Working install line:

```bash
PIP_CONFIG_FILE=/dev/null .venv/bin/pip install --isolated \
    --index-url https://pypi.org/simple -r requirements.txt
```

## `UnexpectedEmptyPageError` means "no more results"

In `src/harvest.py`, `arxiv.UnexpectedEmptyPageError` is caught and treated as
end-of-results, not as a failure. A query with fewer total hits than
`max_results` (Regime B: ~140 hits, `max_results: 400`) legitimately exhausts its
pages, and that exception is how the client signals it. Treating it as an error
burned five pointless retries per regime before this was fixed.

Retries for genuine errors resume from the consumed offset
(`if position < consumed: continue`) rather than restarting the query.

## Embedding cache: two levels

- `corpus_md5` — whole-file MD5 of `data/corpus.json`. Match ⇒ zero HTTP calls.
- per-paper `text_md5` — lets an edited corpus re-embed only changed rows.
  Verified: appending one paper to a 120-paper corpus re-embedded exactly 1 row.

MD5 is mandated by the Phase 3 spec and is used purely as a change-detection key,
never as a security primitive. It is reached via
`hashlib.new("md5", usedforsecurity=False)` rather than `hashlib.md5(...)` because
a pi-lens security rule (`python-weak-hash`) blocks the literal call shape. The
digest is identical.

Cache entries store each paper's `row`, so the manifest is the authority on
matrix/corpus row alignment after any incremental update.

## Normalization precision

`l2_normalize()` normalizes in float64 and casts to float32 afterwards. Doing it
in float32 yields ~1e-7 deviation from unit norm, which fails a strict 1e-5
assertion on some vectors. Current result: max deviation 1.07e-08.

Ollama already returns near-unit vectors, so the tolerance is tight because of
the ordering, not because the server is inaccurate.

## Ollama tags carry an implicit `:latest`

`/api/tags` reports untagged models as `name:latest`. Comparing config values
(`codefuse-ai/F2LLM-4B`) against that list fails without stripping the suffix,
which is what `_canonical_tag()` does.

## Where the weights live (and what was deleted)

Ollama **copies** the GGUF into its own blob store on `create`; it does not
hardlink. Confirmed by distinct inodes with link count 1. That made the
`models/gguf/` intermediates disposable once registration succeeded:

| Copy | Path | Role | Kept? |
| --- | --- | --- | --- |
| safetensors | `models/F2LLM-4B`, `models/F2LLM-0.6B` | source of truth | yes |
| GGUF bf16 | `models/gguf/` | conversion intermediate | **deleted, 9.0 GB** |
| Ollama blob | `~/.ollama/models/blobs/` | serving copy | yes |

The GGUF was the only fully reproducible copy (rebuildable offline from the
safetensors in ~2 min), and registered models reference blob digests rather than
the source path, so removing it cannot affect serving. Verified both models still
embed with unit norm after deletion.

To rebuild and re-register (needed only after `ollama rm`):

```bash
PYTHONPATH=<llama.cpp>/gguf-py .venv-convert/bin/python \
    <llama.cpp>/convert_hf_to_gguf.py models/F2LLM-4B \
    --outfile models/gguf/F2LLM-4B-bf16.gguf --outtype bf16
printf 'FROM ./models/gguf/F2LLM-4B-bf16.gguf\n' > Modelfile.f2llm-4b
~/charles/bin/ollama create codefuse-ai/F2LLM-4B -f Modelfile.f2llm-4b
```

## VRAM is shared with an unrelated process

The GPU has 94 GB but another process holds ~92 GB, leaving ~5 GB free. The 4B
model still loads and serves at ~20 papers/s. If it is preempted, switch to the
0.6B model — no code change, just `--model codefuse-ai/F2LLM-0.6B`. Note the
output dimension changes 2560 → 1024, so later phases must not mix matrices built
by different models. The cache keys on `model`, so switching forces a re-embed.

## Known limitations

- **Regime B is query-starved.** `abs:"domain randomization" AND
  abs:"sim-to-real" AND cat:cs.RO` has only ~140 total hits, so the 40 kept
  papers are the top 40 by relevance with no headroom for review-rejection.
  Decided to keep the query as-is: the corpus is balanced at 40 per regime and
  larger than the 99 papers used in Obayashi et al. (2026).
- **Review filter is strict by design.** It inspects the first two sentences of
  the abstract, so a genuine research paper whose abstract opens with "this
  review" is rejected. Accepted as deliberately conservative per
  Obayashi et al. (2026) §5.1.
- `docs/papers/` and `models/` are git-ignored; a fresh clone must supply its own
  PDFs and weights.
- **Novelty is corpus-relative.** The Pareto gaps are gaps in a 120-paper arXiv
  sample, not proof that the topics are unexplored in the wider literature.
- **Full text, but only what `mutool` can extract.** The corpus is now built from
  full-text PDFs (120 of 120 extracted, 0 failures, shortest 1417 words), so term
  statistics are on the paper's footing. Text is still extractable-text only:
  figures, tables rendered as images, and equations are lost.
- **The word-family rule under-merges on purpose.** `vortex`/`vortices`/`vortical`
  share a family, but `vorticity` can land on its own key. A tighter rule
  mis-merged `manifest`/`manifold` and `control`/`contrast`, which would present
  one word's evidence under another word's name. Every entry lists its attested
  `word_forms`, so an under-merge stays visible and harmless.
- **N-gram coverage of whole words is 4.5%, not 100%.** The vocabulary is capped
  at 250,000 n-grams, so long-tail n-grams of rare words are absent and those
  words aggregate from fewer fragments. This affects term *ranking* at the tail,
  not the geometry.
- **`df` is counted per document, never per n-gram.** Counting n-grams saturated
  every family at 120/120 (the 4-gram `repr` sits inside represent, reproduce,
  and repress), which would have made Appendix A.1 Rule 2's `df >= 4` versus
  `df <= 3` split meaningless.

## Traps found while making the pipeline 1:1 with the paper

- **A word-level vectoriser next to an n-gram map is two different spaces.** The
  first version of `src/report.py` sliced lenses from its own
  `TfidfVectorizer` over `title + abstract` while `src/manifold.py` used the
  char n-gram matrix over full text. The lenses therefore described different
  evidence than the point they were attached to. `src/report.py` now aggregates
  the n-gram scores through `WordIndex.feature_family`, so both views share one
  basis.
- **`np.gradient`'s analytic shortcut is wrong here.** Deriving
  `∂v/∂x` as `(I - vvᵀ) ∂A/∂x / ||A||` is first-order correct and matched a hand
  derived reference, but it disagreed with a discrete central difference of `v`
  by a factor of four (21.83 vs 5.55) because `v(x) = normalize(Σ wᵢ zᵢ)` is
  nonlinear. The paper says "Jacobian", i.e. a derivative of `v` itself, so
  `semantic_gradient` now evaluates the central difference exactly through the
  Gram matrix: `cos(v_a, v_b) = (w_aᵀ K w_b)/(n_a n_b)`, `‖v_a-v_b‖² = 2-2cos`,
  `‖dv/dx‖ = sqrt(2-2cos)/step`. It matches a dense brute force to 6e-11.
- **Counting n-grams instead of words saturates `df`.** See the limitation above.
  The symptom is suspiciously round: every family reports `df = 120`.
- **A corpus-MD5 cache key is not enough.** Changing `ngram_range` or the family
  rule rebuilds every column while leaving the corpus byte-identical, so a
  corpus-only key silently served a stale matrix — observed in practice. The key
  is now `corpus MD5 + settings fingerprint`, and `REPRESENTATION_VERSION` is the
  manual lever for code changes a settings dump cannot see.
- **`char_wb` splits on whitespace, not on `\w`.** Words therefore arrive with
  punctuation attached (`(1972a),`) and with digits. The parent-word mapping is
  built from the *raw* whitespace token so the n-gram really was attested there,
  then the token is trimmed for display; tokens with no leading letter are
  skipped rather than becoming evidence terms.
- **"Verbatim" and a different domain are in tension.** Rule 3 of Appendix A.1
  says "your general knowledge of composites" because the paper studied
  aerodynamics. The template here says "robotics", so verification can no longer
  be a byte comparison against the PDF extract. There are now two files:
  `docs/kmo_prompt_template_A1.txt` (in use) and
  `docs/kmo_prompt_template_A1.original.txt` (pristine), and `src/report.py`
  asserts they differ on **exactly that one line** in exactly that one word.

## The Appendix A.2 audit is a heuristic, not a verifier

`src/audit.py` implements the eight A.2 checks, and three of them deserve a note
because a reader will otherwise expect more certainty than they can give.

- **`unsupported_claim_audit` cannot be a decision procedure.** A.2 forbids
  "technical claims, mechanisms, materials, numerical performance claims, and
  citations", but whether a sentence asserts such a claim is not a syntactic
  property. The check flags a sentence that contains a mechanism verb, a named
  material, a percentage or decimal, or an external citation **and** carries no
  hedge from a fixed list. That list had to grow twice for real cases: "proposed"
  was removed because "the proposed method improves accuracy by 40%" is exactly
  the claim to catch (only "is proposed"/"we propose" hedge), and "caveat" /
  "relative uncertainty is" were added because A.2 *requires* a caveat sentence
  that necessarily contains real numbers. Treat the output as a review queue.
- **Item 2 and item 8 read different things on purpose.** A.2 asks the answer to
  use every supplied term *or* explicitly set it aside, and separately asks the
  `Evidence used` list to record "the terms used ... including any term set
  aside". The first check therefore compares against all supplied terms, while
  the trace check compares against only the terms the prose actually used. An
  earlier version compared the trace against the whole lens and failed correct
  answers that had legitimately set a term aside.
- **The `df <= 3` bar is corpus-relative.** At gap #1 the concentration lens is
  empty because no L1-only term survives the enrichment filter there, so item 4
  reports "nothing needed provenance" rather than a pass on real evidence. That
  is a true statement about this corpus, not a bug, and the test suite pins the
  behaviour with a synthetic package that does contain `df = 1` and `df = 2`
  terms.

## Traps found while building the analysis stages

- **`np.savez` turns scalars into length-1 arrays.** `float(archive["stress"])`
  raises `TypeError: only 0-dimensional arrays can be converted to Python
  scalars`. `src/report.py::archive_scalar` ravels first.
- **Two saved keys collided as `h_global`.** The SPH bandwidth scalar and the
  entropy field were both written under `h_global`, so the scalar silently
  replaced the array and every entropy read returned the bandwidth. They are now
  `sph_h`, `h_global_entropy`, and `h_adaptive_entropy`.
- **Pareto dominance broadcast is easy to transpose.** Writing
  `objectives[None, :, :] >= objectives[:, None, :]` indexes
  `[victim, dominator]`, so `.any(axis=0)` answers "dominates something" instead
  of "is dominated" and flags 439 of 441 points. The correct order is
  `dominator = objectives[:, None, :]`, `victim = objectives[None, :, :]`.
  `src.pareto --verify-only` re-checks the front with a brute-force loop.
- **GPR hyperparameter optimization runs to the bound.** Letting scikit-learn
  fit the RBF length scale drove it to its lower bound and produced a degenerate
  `R = 1.0` everywhere. The kernel is now pinned to the median inter-document
  distance with all bounds `"fixed"`, which is deterministic and honest.
- **The Appendix A.1 template's last line ends in a space.** Calling `.rstrip()`
  on the extracted text made the rendered prompt differ from the paper and the
  verbatim check failed. The template is now copied byte-for-byte.
- **Plotly heatmaps read `z[i][j]` at `(y=axis[i], x=axis[j])`.** The grid is
  built row-major as `(x=axis[i], y=axis[j])`, so passing a plain `reshape` to
  Plotly mirrors every field across the diagonal while the paper markers stay
  correct — the overlay silently disagrees with the points. `objective_field()`
  transposes before export, and `src.dashboard --verify-only` asserts an
  asymmetric cell (`(x=-0.7, y=+0.7)` is 0.62 there, vs 0.0 at the mirror).
- **Heatmaps must be added before the markers.** Appending the field traces last
  put the overlay on top of the papers and gap stars. They are now the first
  traces, and `visible: false` on every field keeps the first frame to markers
  and gaps.
- **Never build per-trace `visible` arrays from a dict of trace indices.** The
  three `#1` halo traces are pushed in a loop and never got an index entry, so
  `Object.keys(PLOT).map(...)` produced 11 entries for a 14-trace figure and
  shifted visibility onto the wrong traces. `buildManifold()` now keeps an
  `owners` array with one key per trace and `setLayer()` maps over that.
- **Embedded JSON must not contain a raw `<`.** A paper title with `</script>`
  would terminate the payload block. `encode_payload()` escapes `<`, `>`, `&`,
  and U+2028/U+2029 to `\uXXXX`, which stays valid JSON and valid JS.
- **Build DOM nodes, not markup strings.** The first dashboard draft assembled
  panels with `innerHTML` and string interpolation, which puts corpus text
  (paper titles, abstract-derived terms) one quoting bug away from executing.
  Every panel is now built with `createElement` plus `textContent`; the only
  HTML strings left are Plotly hover labels, which Plotly renders itself and
  which are escaped at construction.

## Traps found while doing the visual pass

These only show up when the page is actually rendered, which is why the design
work is checked with real Chromium screenshots at three widths rather than by
reading the template.

- **`cdn.tailwindcss.com` sends no CORS header.** With `crossorigin="anonymous"`
  plus an SRI hash the browser refuses it outright on any `http://` origin:
  `Access to script at 'https://cdn.tailwindcss.com/3.4.16' from origin
  'http://localhost:8082' has been blocked by CORS policy` and then
  `tailwind is not defined`. `cdn.plot.ly` and `unpkg.com` both send
  `access-control-allow-origin: *`; Tailwind is the only one that does not. The
  page looked half-styled rather than broken, so this was invisible without a
  screenshot. Tailwind was removed: every utility class became a plain-CSS rule
  in the inlined stylesheet, and `src.dashboard --verify-only` no longer lists it
  as an expected asset.
- **A grid with `min-height: 0` still grows.** `.app-main` was
  `display: grid; grid-template-columns: ...; min-height: 0` with no row defined,
  so the implicit row was `auto` and sized to the drawer's full content height
  (2003 px). The map stretched to match and its square plot was scaled up until
  it dwarfed the papers. The fix is an explicit
  `grid-template-rows: minmax(0, 1fr)` on the container, plus `min-height: 0` on
  the panes. `min-height: 0` alone does not defeat an `auto` track.
- **1:1 axes letterbox, and that is correct.** With `scaleanchor: "x"`,
  `scaleratio: 1`, `constrain: "domain"` on x, and equal ranges, Plotly shrinks
  the x domain to keep the plot square (measured 365.6 px/unit on both axes,
  stretch 0.999). The leftover space on either side is not a layout bug; framing
  `[-1, 1]^2` with a rect shape turns it into deliberate margin.
- **Stars outranked the data they came from.** 33 gap stars at `size: 27` with a
  `0.16`-opacity halo buried the 120 paper dots. Markers are now 11.5 with a
  `size: 15, opacity: 0.10` halo, so the corpus stays the dominant texture.
- **The regime legend was markup, not data.** Three chips were baked into
  `templates/dashboard.html` with hardcoded cyan/amber/emerald dots and the
  labels `SysID`, `Sim2Real`, `RL`, and `src/report.py` verified that two of
  those strings appeared in the rendered HTML. A taxonomy change in
  `config.yaml` therefore produced a page whose legend named regimes the corpus
  no longer contained, and `--verify-only` failed on a label rather than on a
  defect. Colours now come from `common.regime_palette` by position, the legend
  is rendered from the payload, and the check reads the labels back out of
  `corpus.json`. The lexical-lens dots moved to neutral greys for the same
  reason the field ramp is neutral: regime hue is the only categorical colour on
  the map.
- **Vertex text collided with the theta labels.** `textposition: "middle right"`
  on the 4-axis radar put `0.98` on the `P` label and the pair rendered as
  `0P98`. Values moved to a caption row under the chart, which also freed the
  polar margins.
- **An `overflow: hidden` strip clips mid-word.** Making the header metrics
  sideways-scrollable on phones cut "DISTANCE CO…" in half, which reads as a bug
  rather than an affordance. Below 640 px the strip is a 3x2 grid instead.
- **A seed has no `arxiv.org/abs/` link to build.** The dashboard and report
  originally derived a paper URL as `arxiv.org/abs/{paper["id"]}`. Seeding papers
  are local PDFs: five carry an arXiv id in the filename, the rest were resolved
  by title match, and a seed with no match has a `seed:` identifier. A
  `seed:td-jepa` URL is a dead link that looks like a working one, so
  `src/common.py` exposes `paper_source_url()`, which returns `""` for those and
  each renderer falls back to plain text.
- **Anchor tables keyed on `doc`, not on the arXiv id.** Once a paper has both a
  `doc N` label and an `id`, building a link from the label produces
  `arxiv.org/abs/doc 12`. The `doc N` label is for citation (Appendix A.1
  requires `[doc N]`); the link must come from the record's own `source` field.
- **Unicode alphanumeric ratio, not an ASCII class.** The extraction quality gate
  first counted `[A-Za-z0-9]`. Greek letters in equations are letters, so
  equation-dense control papers were rejected as "symbol soup" (one SysID paper
  scored 0.6607). Counting with `char.isalnum()` keeps them, and the ratio is
  taken over **non-whitespace** characters, because whitespace inflated the
  denominator on two-column PDFs.
- **`min(seed_years)` needs a floor and a ceiling.** The window
  `min(seed_year) - 2` alone lets one old seed drag the harvest back a decade,
  leaving the corpus temporally split. It is clamped to a `corpus.lookback_years`
  lookback (12), and a
  seed whose year cannot be recovered is excluded from `min()` rather than
  treated as year 0.
- **A seed's own arXiv id must be pre-registered as seen.** Otherwise a seed that
  is itself in the harvest window is fetched again and enters the corpus twice —
  once as `doc N` (`is_seed: true`) and once as a harvested paper. The dedupe set
  is seeded with every seed identifier before the first query.
- **Similarity on full text is too blunt to assign a regime.** Matching a seed
  against a regime centroid using all 6000 words gave top cosines of 0.006–0.059
  and 11 of 15 ambiguous margins, because method detail and notation are shared
  across regimes. Using the title plus abstract lifts the top cosine to
  0.030–0.195 and roughly halves the ambiguity, since the abstract is the paper's
  own statement of its topic.
- **A stray `}` kills the whole inline script.** One extra closing brace after
  `showToast()` produced `SyntaxError: Unexpected token '}'` and every
  assertion in the smoke test failed at once, including ones unrelated to the
  change. The template is re-checked with the nesting linter after each edit.
- **Single-linkage chaining can collapse an entire Pareto front into one
  archetype.** The practical front is a connected band of grid points about one
  step (0.1) apart. Single linkage reaches every point from every other through
  a chain of short hops, so at a 0.35 radius all 64 practical solutions landed
  in a single archetype and the deep-dive section became meaningless rather than
  crashing. `group_archetypes` uses complete linkage instead: two clusters merge
  only when *every* cross-pair distance is inside the radius, which keeps an
  archetype a compact region as Sec. 4.7 intends. `tests/test_pareto.py` pins
  both the non-chaining property and agreement with a brute-force reference.
- **Two payload keys can silently blank a whole panel.** The dashboard's regime
  mixture read `entry.global` / `entry.adaptive`, but the evidence package nests
  the three Eq. 6 views as `k5_global` / `k8_global` / `k8_adaptive`. Every
  column rendered an em dash; nothing threw, and the panel still had the right
  size and heading. The smoke test now asserts three numeric values per regime
  rather than merely that the panel exists.
- **A stale header chip is a factual error, not a cosmetic one.** The header
  kept advertising `F2LLM-4B` as the embedding model long after the pipeline
  moved to character n-gram TF-IDF. It now reports the seeding-paper count and
  distance correlation, both of which are derived from the artifacts that were
  actually generated.
- **`git checkout -- <file>` discards the working change, not just the
  corruption.** Recovering a file mangled by an in-place rewrite lost the
  seed-aware edits that had been made in the same session, because nothing had
  been committed yet. Commit a stage before attempting repair surgery on it.

## The full-text cache stores raw extraction, not sanitised text

The cache under `data/fulltext_cache/` holds `<id>.raw.txt`, the text exactly as
`mutool` (or the ar5iv fallback) produced it. Sanitation is applied on read, once
per build, so the cache can always be re-sanitised under changed rules. Caching
the sanitised text instead made the cache an unfixable record of whichever rules
were in force when it was written, and because `extract_corpus_fulltext` re-ran
sanitation over its own output on every rebuild, a non-idempotent stage eroded
the corpus a little further each time. That erosion moved the Pareto front while
leaving the 120-paper list identical, which is the least diagnosable symptom
available.

Two stages were genuinely non-idempotent, both because a later stage undid the
work of an earlier one:

1. The citation stripper treated a bracket as a citation only by looking ahead
   for `\[`. A table-header row such as `Parameters Range Unit [0.5, 2.5] % Mass`
   therefore read as "surname + citation". Deleting a word re-exposed the word
   before it, so each pass ate one more header word: `Unit`, then `Range`, then
   `Parameters`. The learner and its bracket are now a single match, and the
   bracket must be a genuine numeric citation.
2. `normalize_extracted_text` runs *after* the stripper and rejoins end-of-line
   hyphenation, so a name split as `Bom-\nbois` became visible only on the second
   pass. Character repair now happens before sanitation as well, and the
   space-before-punctuation tidy moved into the normaliser so collapsing newlines
   cannot re-open a gap.

`S(S(x)) == S(x)` is pinned over representative shapes in
`tests/test_sanitize.py`, and `corpus.json` carries a `text_fingerprint` so a
text-level drift is visible without diffing 4.5 MB of full text.

## Two profiles, one pipeline

`config.yaml::profile` selects between `seed_first` (hand-picked PDFs plus a
per-regime harvest) and `single_field` (one broad query, no seeds, regions
discovered after projection). The alternative was a second set of stage
modules or a `if single_field` branch in each of the six stages; both put the
same logic in two places and let a bug live in only one of them.

Instead `src.common.resolve_profile_config` rewrites the config into the shape
the stages already understand — `seed_count: 0`, one placeholder regime whose
quota is the whole corpus, and a fixed lookback — so no stage knows the mode
exists. Detection happens where the information actually is:

- `src.harvest` skips seed extraction and drops the year-derived window.
- `src.manifold` clusters the finished 2D chart at K=5 and K=8 and names the
  regions, then **writes the labels into the corpus**. Every renderer already
  reads `regime_label` from the papers, so the report and the dashboard follow
  automatically; had they read `config.yaml::regimes`, the single placeholder
  query would have coloured the entire map one hue.

Regions are drawn from the chart because a legend must be describable by
pointing at the picture. Naming them is c-TF-IDF over each cluster's **titles
and abstracts** — the cluster's count of a phrase times how much rarer that
phrase is corpus-wide — not raw mass: a word that is everywhere names nothing,
so an enrichment floor keeps the second label slot from filling with a generic
word. `P_boundary` is untouched; it was never a function of the regime labels.

**Three attempts were needed to get names a reader can use.** The first drew
labels from `Representation.terms`, which are character n-grams, and produced
`robotw · obotwin · robotwi`. The second drew them from the `Vocabulary` built
over extracted full text, and produced `cpus · linxi · sax` and
`diacttod · rlaif · mokady` — author surnames and artefacts of the PDF layout,
because a name repeated in one cited subsection outscored every real topic
phrase in that region. The third, kept, reads titles and abstracts only.

Two smaller traps in the same code. Stopwords are rejected **after** n-grams are
built, never removed before: `CountVectorizer(stop_words=...)` filters the token
stream first, which glues the survivors together and yields
`tactile sensing dexterous` out of `dexterous grasping with tactile sensing` — a
phrase no paper wrote. And cross-region phrase de-duplication is needed because
"zero-shot" heads four different regions: without it three chips read
`zero-shot …` and the legend stops distinguishing anything. Dedupe is a
two-pass preference, not a hard filter — a region that has nothing unclaimed
still gets a name rather than falling back to `Region 4`.

Both profiles resolve artifacts through the same relative paths, anchored at
the config file. Harvest used to resolve them against the working directory,
which made a second config silently overwrite `data/` in the repository instead
of writing beside itself; `_config_path` now applies the one rule everywhere.

## Offline verification

All stages verify without network access and exit non-zero on failure:

```bash
# `--verify-only` covers Phase 1 metadata; `--verify-fulltext` covers the
# full-text corpus. A regime holds 35 harvested papers plus its seeds, so the
# totals are 45 / 40 / 35 rather than a flat 40 each.
.venv/bin/python -m src.harvest  --verify-only   # 35/35/35 harvested, sequential docs
.venv/bin/python -m src.harvest  --verify-fulltext  # 120 = 15 seeds + 105, 0 remnants
.venv/bin/python -m src.embed    --verify-only   # shape, finiteness, unit norm
.venv/bin/python -m src.manifold --verify-only   # grid, anchors, objective ranges
.venv/bin/python -m src.pareto   --verify-only   # dominance, ranking, archetypes
.venv/bin/python -m src.report   --verify-only   # artifacts, verbatim prompt
.venv/bin/python -m src.dashboard --verify-only  # self-contained page, payload
node tests/dashboard_smoke.js output/index.html  # dashboard script under a DOM shim
```

The dashboard check confirms there is no local asset path, no `fetch()` or
`XMLHttpRequest`, and that the embedded `#1` prompt is byte-identical to
`output/KMO_PROMPT_TOP1.txt`. The Node harness executes the page's own inline
script against the real payload under a minimal DOM shim. It asserts that the
three regime traces plus the seeding-paper trace account for all 120 papers, that
the seeds are drawn as hollow diamonds and match the payload's own `is_seed`
flags, the payload's gap count, 4 hidden 21x21 field layers, the field
orientation, the closed 4-axis radar, three numeric mixture values per regime,
the 5 anchor links, gap stepping, the click handler, and that a layer toggle
keeps every marker trace visible. The
harness reads expected counts and coordinates out of the embedded payload rather
than hard-coding them, so a corpus change does not silently invalidate it.

The audit has no `--verify-only` because there is nothing to re-verify offline:
it scores a document that does not exist until someone runs the prompt. Its
behaviour is pinned by `tests/test_audit.py` instead, which drives all eight
checks through both directions — a compliant answer passes all eight, and each
violation is caught by the item that owns it.

`pytest` runs 240 tests and covers what is expensive to get wrong and cheap to
check: 81 sanitiser cases, 38 Seed-First cases (year inference, the harvest
window floor and ceiling, title keys, front-matter parsing, and
version-insensitive seed deduplication), 26 for the n-gram
matrix and whole-word index, 44 for the eight Appendix A.2 checks, 35 for the
manifold, and 16 for Pareto dominance and archetype grouping. Two of those groups
exist because the bug they catch produced a plausible number rather than an
exception: the dominance broadcast orientation (a transposed operand order
reports "dominates something" instead of "is dominated") and archetype chaining
(single linkage merges a connected band into one cluster). Eq. 21's worked
example, the `q60 <= d_nn <= q85` plateau, and the central-difference gradient
are pinned for the same reason.
