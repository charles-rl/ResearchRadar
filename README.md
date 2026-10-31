# ResearchRadar

A research-landscape radar for robotics control. The pipeline harvests an arXiv
corpus across three research regimes, embeds it with a local F2LLM model, and
projects the embedding manifold into a 2-D chart that is scored on boundary
pressure, whitespace, knowledge-gradient alignment, and structure.

Implemented so far: **Phase 1** (environment), **Phase 2** (harvest),
**Phase 3** (embeddings), **Phase 4** (manifold and discovery fields),
**Phase 5** (Pareto search), **Phase 6** (report and LLM prompt).

## Regimes

| ID | Label | Query |
| --- | --- | --- |
| 0 | Classical System Identification | `abs:"system identification" AND (cat:cs.RO OR cat:eess.SY)` |
| 1 | Sim2Real / Domain Randomization | `abs:"domain randomization" AND abs:"sim-to-real" AND cat:cs.RO` |
| 2 | Deep RL Control | `abs:"reinforcement learning" AND abs:"manipulation" AND cat:cs.RO` |

40 papers per regime, 120 total, balanced by construction.

## Quick start

```bash
python3 -m venv .venv
PIP_CONFIG_FILE=/dev/null .venv/bin/pip install --isolated \
    --index-url https://pypi.org/simple -r requirements.txt

.venv/bin/python -m src.harvest      # -> data/corpus.json   (120 papers)
.venv/bin/python -m src.embed        # -> data/embeddings.npy (120, 2560)
.venv/bin/python -m src.manifold     # -> data/manifold_grid.npz  (441-point grid)
.venv/bin/python -m src.pareto       # -> data/pareto_solutions.json
.venv/bin/python -m src.report       # -> output/radar_map.html
                                     #    output/RADAR_REPORT.md
                                     #    output/KMO_PROMPT_TOP1.txt
.venv/bin/python -m src.dashboard    # -> output/index.html (standalone app)
```

Every module supports offline re-verification:

```bash
.venv/bin/python -m src.harvest  --verify-only
.venv/bin/python -m src.embed    --verify-only
.venv/bin/python -m src.manifold --verify-only
.venv/bin/python -m src.pareto   --verify-only
.venv/bin/python -m src.report   --verify-only
.venv/bin/python -m src.dashboard --verify-only
node tests/dashboard_smoke.js output/index.html
```

## Why the pip flags

`/etc/pip.conf` on some hosts injects `extra-index-url =
https://pypi.ngc.nvidia.com`. When that host is unreachable, pip enters a long
retry storm for every candidate. `PIP_CONFIG_FILE=/dev/null --isolated
--index-url https://pypi.org/simple` bypasses the system config and pins the
index explicitly.

## Serving F2LLM with Ollama

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

## Embedding pipeline

* Input text per paper: `"Title: <title> | Abstract: <abstract>"`.
* Vectors are L2-normalized in float64 and stored as float32, giving
  `||v||_2 = 1` to within ~1e-8.
* `data/embeddings_cache.json` holds the corpus MD5 plus a per-paper text MD5.
  Re-running on an unchanged corpus makes zero HTTP calls; editing one paper
  re-embeds exactly one row.

## Manifold, Pareto front, and report

* **Projection.** Classical MDS, then constrained stress minimization with 9
  gauge-fixing anchors (8 perimeter + 1 central) onto `[-1, 1]^2`.
  Final metric stress 204.69, Spearman(distance) 0.8658.
* **Grid.** 21x21 = 441 candidates, scored on `P_boundary`, `W`, `G`, and `S`.
* **White-space window.** The `q20/q60/q85/q98` band-pass thresholds are
  recomputed on this corpus's own nearest-document distances
  (`manifold.use_empirical_quantiles: true`). The paper's values are for its
  aerodynamic corpus at a different map scale and are kept only as reference.
* **Pareto search.** Exact pairwise dominance, no evolutionary optimizer. 45
  non-dominated points, 33 practical after the `W(x) > 0` filter, grouped into
  3 discovery archetypes at `archetype_radius: 0.35`.
* **Reliability `R`** is computed and reported but is *not* a Pareto objective:
  maximizing low uncertainty rewards already-dense regions.
* **The prompt.** `docs/kmo_prompt_template_A1.txt` holds the Appendix A.1 text
  extracted verbatim from the paper. `src/report.py` only substitutes the
  placeholder file name, and `--verify-only` compares the rendered prompt to the
  template line by line, so the wording cannot drift.

## The dashboard

`output/index.html` is a single self-contained page: the corpus, the four
objective fields, and one evidence package per gap are baked into an inline JSON
payload, so it opens straight from the filesystem with no server and no build
step. Tailwind, Plotly, and Lucide load from CDNs (pinned, with SRI hashes).

* **Left panel** — the manifold on `[-1, 1]^2`, papers coloured by regime, the 33
  practical gaps as stars, and layer buttons (keys `1`-`5`) that swap the
  background between `P_boundary`, `W`, `G`, and `S`.
* **Right panel** — the selected gap's evidence package: a 4-axis objective
  radar, support statistics, the regime mixture, the top-5 anchor papers, the
  L2/L1 lexical split, and the Appendix A.1 prompt with a copy button.
* **Navigation** — click a star, use the dropdown, or press the arrow keys.
  `#gap=<rank>` deep-links a gap.

The prompt text comes from `src.report.render_prompt_text`, the same function
the Markdown report uses, so the pasted prompt and `KMO_PROMPT_TOP1.txt` are
byte-identical — `src.dashboard --verify-only` asserts exactly that.

`tests/dashboard_smoke.js` runs the page's own inline script against the real
payload under a minimal DOM shim, so it catches renderer bugs without a browser:

```bash
node tests/dashboard_smoke.js output/index.html
```

## Layout

```text
config.yaml                       queries, limits, model/analysis parameters
docs/implementation-notes.md      non-obvious environment traps and decisions
docs/kmo_prompt_template_A1.txt   verbatim Appendix A.1 prompt template
src/harvest.py                    arXiv harvesting, review filtering, cleanup
src/embed.py                      Ollama F2LLM embedding with MD5 caching
src/common.py                     config loading, path resolution, logging
src/manifold.py                   MDS, discovery grid, the four KMO objectives
src/pareto.py                     exact dominance, practical front, archetypes
src/report.py                     Plotly map, Markdown report, LLM prompt
src/dashboard.py                  standalone output/index.html generator
templates/dashboard.html          the page shell and its inline app script
tests/dashboard_smoke.js          offline DOM-shim test of the dashboard script
data/corpus.json                  the 120-paper corpus (committed)
data/embeddings.npy               derived embedding matrix (git-ignored)
data/manifold_grid.npz            grid coordinates and objective matrix
output/                           rendered artifacts (HTML git-ignored)
```
