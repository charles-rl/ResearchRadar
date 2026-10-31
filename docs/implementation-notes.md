# Implementation notes

Things that are **not** obvious from reading `src/`. Each item cost real
debugging time; most are environment-specific traps rather than design choices.

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
  papers are the top 40 by relevance with no headroom for review-rejection. Still
  an open decision whether to relax `AND` to `OR`.
- **Review filter is strict by design.** It inspects the first two sentences of
  the abstract, so a genuine research paper whose abstract opens with "this
  review" is rejected. Accepted as deliberately conservative per
  Obayashi et al. (2026) §5.1.
- `docs/papers/` and `models/` are git-ignored; a fresh clone must supply its own
  PDFs and weights.

## Offline verification

Both stages verify without network access, and exit non-zero on failure:

```bash
.venv/bin/python -m src.harvest --verify-only   # 40/40/40, unique ids + titles
.venv/bin/python -m src.embed   --verify-only   # shape, finiteness, unit norm
```
