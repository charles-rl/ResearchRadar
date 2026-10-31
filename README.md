# ResearchRadar

A research-landscape radar for robotics control. The pipeline harvests an arXiv
corpus across three research regimes, embeds it with a local F2LLM model, and
projects the embedding manifold into a 2-D chart that is scored on boundary
pressure, whitespace, knowledge-gradient alignment, and structure.

Implemented so far: **Phase 1** (environment), **Phase 2** (harvest),
**Phase 3** (embeddings).

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
```

Both modules support offline re-verification:

```bash
.venv/bin/python -m src.harvest --verify-only
.venv/bin/python -m src.embed   --verify-only
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

| Model | Ollama tag | Dimensions | GGUF size |
| --- | --- | --- | --- |
| F2LLM-4B | `codefuse-ai/F2LLM-4B` | 2560 | 7.6 GB |
| F2LLM-0.6B | `codefuse-ai/F2LLM-0.6B` | 1024 | 1.2 GB |

The 4B model is the default. Switch to the 0.6B model with
`--model codefuse-ai/F2LLM-0.6B`, or by editing `ollama.fallback_model` if VRAM
is tight.

## Embedding pipeline

* Input text per paper: `"Title: <title> | Abstract: <abstract>"`.
* Vectors are L2-normalized in float64 and stored as float32, giving
  `||v||_2 = 1` to within ~1e-8.
* `data/embeddings_cache.json` holds the corpus MD5 plus a per-paper text MD5.
  Re-running on an unchanged corpus makes zero HTTP calls; editing one paper
  re-embeds exactly one row.

## Layout

```text
config.yaml              queries, limits, model and analysis parameters
src/harvest.py           arXiv harvesting, review filtering, abstract cleanup
src/embed.py             Ollama F2LLM embedding with MD5 caching
data/corpus.json         the 120-paper corpus (committed)
data/embeddings.npy      derived embedding matrix (git-ignored)
```
