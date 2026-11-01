# Agent runbook

Operational checklist for an AI agent running or changing this repository.
Read this before touching `config.yaml` or `docs/papers/desired/`.

`README.md` explains *what* each stage means and why. This file is about
*doing*: which command to run, in what order, what to check, and what silently
invalidates what.

---

## 1. The one thing to understand first

**A rebuild does not inherit papers. The corpus is always `seeds + this run's
harvest`, never `previous corpus + delta`.**

`src/harvest.py::build_fulltext_corpus` re-queries arXiv from scratch on every
`--extract-text` run. There is no merge step, no incremental update, and no
notion of "keep the old ones that still fit". Consequences:

- Change a **query** and every previously harvested paper from that regime is
  gone, unless the new query happens to return it again.
- Change a **regime label, id, or set of regimes** and the entire harvest is
  discarded, because a paper's `regime_id` is assigned by which query found it.
- Add or remove a **seed** and the harvest window can move (see §3), which
  changes which papers are eligible even if the queries are untouched.

The only thing carried across runs is the **full-text cache**, keyed by arXiv ID:

| Path | Holds | Reused how |
| --- | --- | --- |
| `data/fulltext_cache/pdf/<id>.pdf` | downloaded PDF | not re-downloaded if > 10 KB |
| `data/fulltext_cache/<id>.raw.txt` | **raw** extractor output | not re-extracted |

Cache entries for papers that are no longer selected are simply never read —
they are not garbage, not a hidden corpus, and cannot leak into the output. The
cache changes **only speed, never membership**.

Corollary: if a paper disappears after a rebuild, the cause is always the
selection logic (query, window, exclusion rules), never the cache.

---

## 2. Normal run, nothing changed

```bash
.venv/bin/python -m src.harvest --extract-text   # -> data/corpus.json
                                                 #    data/corpus_fulltext.json
.venv/bin/python -m src.embed                    # -> data/sparse_tfidf.npz
.venv/bin/python -m src.manifold                 # -> data/manifold_grid.npz
.venv/bin/python -m src.pareto                   # -> data/pareto_solutions.json
.venv/bin/python -m src.report                   # -> output/radar_map.html
                                                 #    output/RADAR_REPORT.md
                                                 #    output/KMO_PROMPT_TOP1.txt
.venv/bin/python -m src.dashboard                # -> output/index.html
```

Stages are strictly sequential. Each reads the previous stage's artifact, so a
partial run leaves downstream files describing an older corpus. **After any
corpus change, re-run all six.** Do not hand-run `report` on a stale manifold.

Verify (offline, no network):

```bash
.venv/bin/python -m src.harvest   --verify-only
.venv/bin/python -m src.harvest   --verify-fulltext
.venv/bin/python -m src.embed     --verify-only
.venv/bin/python -m src.manifold  --verify-only
.venv/bin/python -m src.pareto    --verify-only
.venv/bin/python -m src.report    --verify-only
.venv/bin/python -m src.dashboard --verify-only
node tests/dashboard_smoke.js output/index.html
/usr/bin/ruff check src/ tests/ && .venv/bin/python -m pytest -q
```

Harvest takes roughly 10-25 minutes (arXiv rate limiting at
`corpus.delay_seconds`). Run it as a background task rather than blocking.

---

## 3. Checklist: adding or replacing a seeding paper

Seeds are the only curated input. They anchor the topical neighbourhoods and
they set the harvest window.

1. **Drop the PDF** into `docs/papers/desired/`.
2. **Name it well.** A filename beginning with an arXiv ID
   (`2511.04131v1.pdf`) is read directly. A free-form name
   (`cassie biped.pdf`) forces an exact-title lookup against arXiv, which is
   slower and can fail. Prefer the arXiv ID when you have one.

   The version suffix is optional: `2511.04131.pdf` and `2511.04131v3.pdf` both
   work, and both block the paper from being harvested under its API-reported
   id. Deduplication compares ids with the version stripped and compares
   normalised titles, so a seed cannot re-enter the corpus as a harvested paper
   even when the PDF filename, the PDF's hyphenated title block, and the arXiv
   metadata all disagree on spelling.
3. **Pin its regime** in `config.yaml::corpus.seed_overrides`. Without an
   entry the assignment falls back to TF-IDF cosine similarity, which logs
   `[AMBIGUOUS]` whenever the margin is under `seed_match_margin`. Pinning is
   the difference between a documented choice and a guess.
   ```yaml
   seed_overrides:
     "2511.04131.pdf": 0   # the key is the FILENAME, not the title
   ```
4. **Fix the arithmetic.** The build harvests `target_per_regime` papers per
   regime and then requires
   `seed_count + n_regimes × target_per_regime == target_total_papers`.
   Adding a 16th seed to the current config yields `16 + 3×35 = 121 ≠ 120`, and
   the run hard-stops (§6) rather than writing 121 documents. Either raise
   `target_total_papers` to the new sum, or lower `target_per_regime` to
   compensate. Update `seed_count` too — `--verify-fulltext` compares it
   against the number of PDFs actually present.
5. **Re-run everything** (§2).

Watch the window: `corpus.harvest_start_year` is
`min(seed_years) - corpus.seed_year_offset`, clamped so it never reaches
further back than `corpus.lookback_years`. Adding a very old seed widens the
window and can pull in a completely different set of harvested papers; the
clamp is what stops one old paper from dragging the corpus back a decade.
Removing the oldest seed can *narrow* the window, which silently changes the
harvest even though no query changed.

The build logs one line per seed with `cosine` and `margin`. Treat any
`[AMBIGUOUS]` warning about a seed as a to-do: either pin it, or confirm the
cosine guess is what you want.

---

## 4. Checklist: changing regimes or queries

1. **Edit `config.yaml::regimes`.** `id` must remain `0..N-1` and unique.
   `label` is what appears in every artifact, so changing a label changes
   chart legends, report sections, and dashboard filters at once.
2. **Keep `regimes[].keywords` in sync.** They are *not* part of the arXiv
   query. They are used only for seed-to-regime similarity, so stale keywords
   silently mis-assign seeds.
3. **Re-check every `seed_overrides` value** against the new `id`s. Regime ids
   are positional, so reordering the regime list silently repoints existing
   overrides at the wrong regime. Re-derive them from scratch rather than
   eyeballing a diff.
4. **Sanity-check each query against arXiv** before committing to a full
   harvest. A query with zero results is not an error the pipeline catches
   early — it produces a short harvest, and `harvest` will refuse to write
   (§6). Test cheaply:
   ```bash
   .venv/bin/python - <<'PY'
   import arxiv
   r = list(arxiv.Client().results(arxiv.Search(query='YOUR QUERY', max_results=3)))
   print(len(r), [x.get_short_id() for x in r])
   PY
   ```
5. **Check the yield.** Each regime must supply `target_per_regime` papers
   *after* the recency window, review exclusion, and deduplication. A query
   that returns exactly 35 has no headroom: one extra review paper or one year
   of slippage and the run fails. Aim for several times the target.
6. **Re-run everything** (§2). Report artifacts will describe the new regimes
   automatically, but prose in `README.md` and `docs/implementation-notes.md`
   that names the old regimes must be updated by hand.

---

## 5. Checklist: changing text sanitation

Sanitation is a pure function of the raw extraction, so this never needs the
network.

1. **Never cache sanitised text.** `data/fulltext_cache/` stores raw extractor
   output precisely so the rules stay changeable. A cache of post-processed
   text becomes an unfixable record of whichever rules were in force when it
   was written.
2. **Write the test first.** `tests/test_sanitize.py` pins idempotency:
   `S(S(x)) == S(x)` must hold. This is not pedantry — the sanitiser runs on
   cache hits, so a non-idempotent rule erodes text every time the pipeline is
   re-run, and the artefacts stop reproducing.
3. **Apply it offline:**
   ```bash
   .venv/bin/python -m src.harvest --refresh-fulltext
   ```
   This re-sanitises every document from the raw cache, preserving doc order,
   regime assignment, and seed flags, and updates `metadata.text_fingerprint`.
   It contacts no network. Then re-run §2 from `src.embed` onward.

`sanitize_document_text` order is load-bearing: character repair (ligatures,
end-of-line hyphenation) → running headers → front matter → bibliography
truncation → citation callouts → whitespace and punctuation. Repairing earlier
than the stripper matters because a citation split across a line break
(`Bom-⏎bois et al., 2006`) is invisible to the stripper until the hyphen wrap
is undone.

---

## 6. What the guard rails are telling you

| Message | Meaning | Action |
| --- | --- | --- |
| `Refusing to overwrite the existing corpus` | harvest returned fewer than `target_total_papers` | arXiv is usually rate limiting or 503-ing. Wait, then re-run. The existing corpus files are untouched. |
| `seed ... is ambiguous (margin ...)` | automatic regime assignment was a near-tie | pin it in `seed_overrides` |
| `fails the extraction audit` | PDF has no usable text layer | supply a text-layer PDF, or let the ar5iv fallback run |
| `still carries N bibliographic locator(s)` | citation/URL stripping regressed | fix the sanitiser, add a regression test |
| `Cannot refresh text for N paper(s) with no raw source` | cache gap after a rule change | re-run the full harvest instead of `--refresh-fulltext` |

A short harvest is a **hard stop on purpose**. Writing an 85-paper corpus over
a good 120-paper one is silent data loss, and every downstream artefact would
happily be built from it.

---

## 7. Ground rules

- **Use the venv.** `.venv/bin/python` for everything. Never install into the
  base conda environment.
- **`PIP_CONFIG_FILE=/dev/null` when installing.** The host's `/etc/pip.conf`
  injects an unreachable `extra-index-url`, which turns any install into a
  retry storm.
- **`data/` and `output/` are generated.** Never edit them by hand; regenerate.
  `docs/papers/` is git-ignored.
- **Never fabricate numbers.** Every figure quoted in prose must come from a
  verify command or an artefact. If you cannot reproduce it, delete the claim
  rather than rounding it.
- **Objectives are label-blind.** Regime labels group the legend; they never
  enter `P_boundary`, `W`, `G`, or `S`. `P_boundary` is the harmonic mean of
  three clustering views (`k5_global`, `k8_global`, `k8_adaptive`), not a
  measure over the regime partition.
- **One writer.** Do not run two pipeline stages concurrently; they share
  output paths.
