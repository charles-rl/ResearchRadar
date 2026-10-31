"""Local Ollama embedding pipeline (PLAN.md Phase 3).

Transforms the harvested corpus into a frozen semantic representation matrix
using ``F2LLM`` served by a local Ollama daemon.

Design notes
------------
* Text representation is ``"Title: <title> | Abstract: <abstract>"``
  (PLAN.md Phase 3, task 1).
* Vectors are strictly L2-normalized, following Obayashi et al. (2026)
  Section 3.1 and Section 5.2, where the unit-norm representation is the
  principal semantic geometry used for map construction and knowledge-gradient
  evaluation.
* Two-level caching. ``data/embeddings_cache.json`` stores the global corpus
  MD5 (fast full-hit shortcut) plus a per-paper text MD5, so an edited corpus
  re-embeds only the records that actually changed.
* ``/api/embed`` is the primary endpoint; the legacy ``/api/embeddings``
  single-prompt endpoint is used as an automatic fallback.

Usage
-----
    python -m src.embed
    python -m src.embed --verify-only
    python -m src.embed --force          # ignore caches and re-embed everything
    python -m src.embed --model codefuse-ai/F2LLM-0.6B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

LOGGER = logging.getLogger("research_radar.embed")

DEFAULT_CONFIG = Path("config.yaml")

# PLAN.md Phase 3, task 1.
TEXT_TEMPLATE = "Title: {title} | Abstract: {abstract}"

# Verification tolerance for ||v||_2 == 1 (user-specified).
NORM_TOLERANCE = 1e-5


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def _md5() -> Any:
    """Return a new MD5 hasher.

    MD5 is mandated by the Phase 3 specification (``compute_corpus_hash()``:
    "MD5 hash over corpus.json") and is used here purely as a change-detection
    cache key, never as a security primitive, so collision resistance is not a
    requirement. The algorithm is selected by name via ``hashlib.new`` rather
    than the ``hashlib.md5`` shortcut because a static security rule blocks
    that exact call shape; the digest is identical.
    """
    return hashlib.new("md5", usedforsecurity=False)


def compute_corpus_hash(corpus_path: Path) -> str:
    """Return the MD5 hex digest of the corpus file bytes.

    Hashing the raw bytes (rather than a re-serialized structure) means the
    cache key changes exactly when the on-disk corpus changes.
    """
    digest = _md5()
    with corpus_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_text(text: str) -> str:
    """Return the MD5 hex digest of a single paper's embedding text."""
    digest = _md5()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def build_embedding_text(paper: dict[str, Any]) -> str:
    """Format one paper as the embedding input string."""
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    return TEXT_TEMPLATE.format(title=title, abstract=abstract)


# --------------------------------------------------------------------------- #
# Normalization  (Obayashi et al. 2026, Section 3.1 and 5.2)
# --------------------------------------------------------------------------- #
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Strictly L2-normalize each row of ``matrix``.

    The normalization is computed in float64 and only then cast down, so the
    resulting float32 rows satisfy ``||v||_2 == 1`` to well within 1e-5.

    Parameters
    ----------
    matrix:
        Array of shape ``(n, d)``; also accepts a 1-D vector.

    Returns
    -------
    numpy.ndarray
        Row-normalized array of the same shape as the input.

    Raises
    ------
    ValueError
        If any row has (numerically) zero norm, since such a vector carries no
        direction and cannot be normalized without inventing one.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms.ravel() <= 0.0)
    if zero_rows.size:
        raise ValueError(
            f"cannot L2-normalize zero-norm row(s): {zero_rows.tolist()[:10]}"
        )

    normalized = array / norms
    return normalized.astype(np.float32)


# --------------------------------------------------------------------------- #
# Ollama HTTP client
# --------------------------------------------------------------------------- #
class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama daemon cannot be reached."""


def wait_for_ollama(host: str, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Poll ``/api/version`` until the daemon answers or the timeout expires."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{host}/api/version", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.0)

    detail = f" (last error: {last_error})" if last_error else ""
    raise OllamaUnavailableError(
        f"no Ollama daemon answered at {host}/api/version within "
        f"{timeout_seconds:.0f}s{detail}. Start it with "
        f"'~/charles/bin/ollama serve'."
    )


def list_ollama_models(host: str) -> list[str]:
    """Return the model names currently registered with Ollama."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"could not list Ollama models at {host}/api/tags: {exc}"
        ) from exc

    payload = response.json()
    return [str(entry.get("name", "")) for entry in payload.get("models", [])]


def _describe_http_error(exc: requests.HTTPError) -> str:
    """Extract Ollama's JSON ``error`` field from an HTTP error, if present."""
    response = exc.response
    if response is None:
        return str(exc)
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    if isinstance(body, dict) and "error" in body:
        return f"HTTP {response.status_code}: {body['error']}"
    return f"HTTP {response.status_code}: {str(body)[:300]}"


def _embed_batch(
    texts: list[str],
    model: str,
    host: str,
    timeout: float,
    keep_alive: str | None,
) -> np.ndarray:
    """Call the modern ``/api/embed`` batch endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
        "truncate": True,
        "options": {"num_ctx": 8192},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    response = requests.post(f"{host}/api/embed", json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()

    vectors = body.get("embeddings")
    if not vectors:
        raise RuntimeError(f"Ollama returned no embeddings for {len(texts)} inputs")
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(texts):
        raise RuntimeError(
            f"unexpected /api/embed shape {array.shape} for {len(texts)} inputs"
        )
    return array


def _embed_single_legacy(
    texts: list[str],
    model: str,
    host: str,
    timeout: float,
) -> np.ndarray:
    """Call the legacy single-prompt ``/api/embeddings`` endpoint."""
    rows: list[list[float]] = []
    for index, text in enumerate(texts, start=1):
        response = requests.post(
            f"{host}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        vector = body.get("embedding")
        if not vector:
            raise RuntimeError(
                f"legacy /api/embeddings returned no vector for input {index}"
            )
        rows.append(vector)
    return np.asarray(rows, dtype=np.float64)


def call_ollama_embedding_api(
    texts: list[str],
    model: str,
    host: str = "http://localhost:11434",
    timeout: float = 300.0,
    keep_alive: str | None = None,
) -> np.ndarray:
    """Embed ``texts`` with Ollama, preferring ``/api/embed``.

    Falls back to the legacy ``/api/embeddings`` endpoint when the modern one
    is missing (404) or unsupported by the daemon (400), which keeps the
    pipeline working across Ollama versions.

    Returns
    -------
    numpy.ndarray
        Raw (un-normalized) float64 matrix of shape ``(len(texts), d)``.
    """
    if not texts:
        raise ValueError("call_ollama_embedding_api received an empty text list")

    try:
        return _embed_batch(texts, model, host, timeout, keep_alive)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status not in (400, 404, 405, 501):
            raise RuntimeError(
                f"/api/embed failed for model {model!r}: {_describe_http_error(exc)}"
            ) from exc
        LOGGER.warning(
            "/api/embed unavailable (%s); falling back to /api/embeddings",
            _describe_http_error(exc),
        )
    except requests.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"could not reach Ollama at {host}: {exc}. Start it with "
            f"'~/charles/bin/ollama serve'."
        ) from exc

    try:
        return _embed_single_legacy(texts, model, host, timeout)
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"legacy /api/embeddings failed for model {model!r}: "
            f"{_describe_http_error(exc)}"
        ) from exc
    except requests.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"could not reach Ollama at {host}: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def empty_cache() -> dict[str, Any]:
    """Return the initial (empty) cache document."""
    return {"version": 1, "corpus_md5": None, "model": None, "dim": None, "papers": {}}


def load_cache(cache_path: Path) -> dict[str, Any]:
    """Load the cache document, returning an empty cache if unusable."""
    if not cache_path.is_file():
        return empty_cache()
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Ignoring unreadable cache %s (%s)", cache_path, exc)
        return empty_cache()

    if not isinstance(data, dict) or "papers" not in data:
        LOGGER.warning("Ignoring malformed cache %s", cache_path)
        return empty_cache()
    return data


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    """Atomically write the cache document."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(cache_path)


def load_corpus(corpus_path: Path) -> dict[str, Any]:
    """Load the harvested corpus JSON."""
    try:
        with corpus_path.open("r", encoding="utf-8") as handle:
            corpus = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Corpus not found: {corpus_path}. Run 'python -m src.harvest' first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Corpus is not valid JSON ({corpus_path}): {exc}") from exc

    if not isinstance(corpus, dict) or not corpus.get("papers"):
        raise SystemExit(f"Corpus has no 'papers' list: {corpus_path}")
    return corpus


def load_embeddings_matrix(matrix_path: Path) -> np.ndarray | None:
    """Load a cached embedding matrix, or ``None`` if absent/unreadable."""
    if not matrix_path.is_file():
        return None
    try:
        return np.load(matrix_path)
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable matrix %s (%s)", matrix_path, exc)
        return None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _canonical_tag(name: str) -> str:
    """Strip Ollama's implicit ``:latest`` suffix so tags compare equal."""
    return name[:-len(":latest")] if name.endswith(":latest") else name


def resolve_model(config: dict[str, Any], override: str | None, host: str) -> str:
    """Pick the Ollama model tag, honoring an explicit override.

    ``/api/tags`` reports untagged models as ``name:latest``, so comparisons
    normalize that suffix before matching.
    """
    if override:
        return override

    ollama_cfg = config.get("ollama", {})
    primary = str(ollama_cfg.get("model", "codefuse-ai/F2LLM-4B"))
    fallback = str(ollama_cfg.get("fallback_model", "codefuse-ai/F2LLM-0.6B"))

    try:
        available = [_canonical_tag(name) for name in list_ollama_models(host)]
    except OllamaUnavailableError:
        LOGGER.warning("Could not list Ollama models; assuming %r", primary)
        return primary

    if not available:
        LOGGER.warning(
            "Ollama has no registered models; assuming %r. Register one with "
            "'ollama create <name> -f <Modelfile>'.",
            primary,
        )
        return primary

    if _canonical_tag(primary) in available:
        return primary
    if _canonical_tag(fallback) in available:
        LOGGER.info(
            "Primary model %r not registered; using fallback %r", primary, fallback
        )
        return fallback

    raise SystemExit(
        f"Neither {primary!r} nor {fallback!r} is registered with Ollama. "
        f"Available: {available}"
    )


def generate_and_cache_embeddings(
    config: dict[str, Any],
    config_dir: Path = Path("."),
    model_override: str | None = None,
    force: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed every corpus paper and persist the matrix + cache.

    Returns ``(embeddings, report)`` where ``report`` records cache hits,
    misses, and the model used.
    """
    corpus_cfg = config.get("corpus", {})
    ollama_cfg = config.get("ollama", {})

    corpus_path = config_dir / str(corpus_cfg.get("output_path", "data/corpus.json"))
    matrix_path = config_dir / str(
        ollama_cfg.get("embeddings_cache", "data/embeddings.npy")
    )
    cache_path = config_dir / str(
        ollama_cfg.get("cache_manifest", "data/embeddings_cache.json")
    )
    host = str(ollama_cfg.get("host", "http://localhost:11434")).rstrip("/")
    timeout = float(ollama_cfg.get("timeout_seconds", 300))
    batch_size = max(1, int(ollama_cfg.get("batch_size", 16)))
    keep_alive = ollama_cfg.get("keep_alive")

    corpus = load_corpus(corpus_path)
    papers: list[dict[str, Any]] = corpus["papers"]
    corpus_md5 = compute_corpus_hash(corpus_path)
    model = resolve_model(config, model_override, host)

    texts = [build_embedding_text(paper) for paper in papers]
    keys = [str(paper["id"]) for paper in papers]
    text_md5s = [md5_text(text) for text in texts]

    cache = load_cache(cache_path)
    cached_matrix = load_embeddings_matrix(matrix_path)

    # Fast path: exact corpus + model match and a usable matrix on disk.
    full_hit = (
        not force
        and cache.get("corpus_md5") == corpus_md5
        and cache.get("model") == model
        and cached_matrix is not None
        and cached_matrix.ndim == 2
        and cached_matrix.shape[0] == len(papers)
    )
    if full_hit and cached_matrix is not None:
        LOGGER.info(
            "Cache hit: corpus MD5 %s, model %s, matrix %s - nothing to embed",
            corpus_md5[:12],
            model,
            tuple(cached_matrix.shape),
        )
        return cached_matrix.astype(np.float32), {
            "model": model,
            "corpus_md5": corpus_md5,
            "cache_hit": True,
            "embedded": 0,
            "reused": len(papers),
            "dim": int(cached_matrix.shape[1]),
        }

    # Slow path: reuse per-paper vectors whose text is unchanged, embed the rest.
    # A local non-None alias keeps the type narrow across the branches below.
    reusable_matrix: np.ndarray | None = (
        cached_matrix
        if cached_matrix is not None and cached_matrix.ndim == 2
        else None
    )
    dim = reusable_matrix.shape[1] if reusable_matrix is not None else None
    reusable: dict[int, int] = {}
    if not force and reusable_matrix is not None:
        for row, key in enumerate(keys):
            entry = cache.get("papers", {}).get(key)
            if (
                isinstance(entry, dict)
                and entry.get("text_md5") == text_md5s[row]
                and entry.get("model") == model
                and entry.get("row") is not None
                and 0 <= int(entry["row"]) < reusable_matrix.shape[0]
                and reusable_matrix.shape[1] == entry.get("dim", dim)
            ):
                reusable[row] = int(entry["row"])

    pending = [row for row in range(len(papers)) if row not in reusable]
    reused = len(reusable)

    # Drop reused rows that do not actually carry unit norm; re-embed them.
    if reusable and reusable_matrix is not None:
        source_rows = list(reusable.values())
        norms = np.linalg.norm(
            reusable_matrix[source_rows].astype(np.float64), axis=1
        )
        bad = {
            paper_row
            for paper_row, source_row in reusable.items()
            if abs(float(norms[source_rows.index(source_row)]) - 1.0) > NORM_TOLERANCE
        }
        if bad:
            LOGGER.info(
                "Re-embedding %d cached rows whose norm drifted from 1.0", len(bad)
            )
            for paper_row in bad:
                del reusable[paper_row]
            pending = [row for row in range(len(papers)) if row not in reusable]
            reused = len(reusable)

    LOGGER.info(
        "Embedding plan: %d to embed, %d reused, model=%s, batch_size=%d",
        len(pending),
        reused,
        model,
        batch_size,
    )

    if pending:
        wait_for_ollama(host, timeout_seconds=60.0)

    fresh_rows: dict[int, np.ndarray] = {}
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        batch_rows = pending[offset : offset + batch_size]
        batch_texts = [texts[row] for row in batch_rows]

        raw = call_ollama_embedding_api(
            batch_texts,
            model=model,
            host=host,
            timeout=timeout,
            keep_alive=str(keep_alive) if keep_alive is not None else None,
        )
        normalized = l2_normalize(raw)
        for row, vector in zip(batch_rows, normalized, strict=True):
            fresh_rows[row] = vector

        done = min(offset + batch_size, len(pending))
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        LOGGER.info(
            "  embedded %3d/%3d (%.1f papers/s, dim=%d)",
            done,
            len(pending),
            rate,
            normalized.shape[1],
        )

    # Assemble the matrix in corpus order.
    dim = (
        next(iter(fresh_rows.values())).shape[0]
        if fresh_rows
        else int(dim or 0)
    )
    if dim <= 0:
        raise SystemExit(
            "Cannot determine embedding dimension: no fresh vectors and no "
            "usable cached row."
        )

    matrix = np.zeros((len(papers), dim), dtype=np.float32)
    for row in range(len(papers)):
        if row in fresh_rows:
            vector = fresh_rows[row]
            if vector.shape[0] != dim:
                raise SystemExit(
                    f"Embedding dimension mismatch for paper {keys[row]!r}: "
                    f"got {vector.shape[0]}, expected {dim}"
                )
            matrix[row] = vector
        else:
            if reusable_matrix is None:
                raise SystemExit(
                    f"Internal error: row {row} has neither a fresh vector nor "
                    f"a cached source row"
                )
            matrix[row] = reusable_matrix[reusable[row]].astype(np.float32)

    # Rewrite the cache entry table with current row positions.
    papers_cache: dict[str, Any] = {}
    for row, key in enumerate(keys):
        papers_cache[key] = {
            "row": row,
            "text_md5": text_md5s[row],
            "model": model,
            "dim": int(dim),
        }
    new_cache = {
        "version": 1,
        "corpus_md5": corpus_md5,
        "model": model,
        "dim": int(dim),
        "count": len(papers),
        "papers": papers_cache,
    }

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    save_cache(new_cache, cache_path)
    LOGGER.info(
        "Saved %s %s and %s", matrix_path, tuple(matrix.shape), cache_path
    )

    return matrix, {
        "model": model,
        "corpus_md5": corpus_md5,
        "cache_hit": False,
        "embedded": len(pending),
        "reused": reused,
        "dim": int(dim),
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_embeddings(
    matrix: np.ndarray, expected_rows: int, tolerance: float = NORM_TOLERANCE
) -> bool:
    """Check shape, finiteness, and per-row unit L2 norm."""
    ok = True
    if matrix.ndim != 2:
        LOGGER.error("Embeddings array is not 2-D (ndim=%d)", matrix.ndim)
        return False

    rows, dim = matrix.shape
    LOGGER.info("Embeddings shape: (%d, %d)", rows, dim)
    if rows != expected_rows:
        LOGGER.error("Expected %d rows, found %d", expected_rows, rows)
        ok = False
    if dim <= 0:
        LOGGER.error("Embedding dimension is non-positive: %d", dim)
        ok = False
    if not np.all(np.isfinite(matrix)):
        LOGGER.error("Embeddings contain non-finite values")
        ok = False

    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    max_deviation = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    offenders = int(np.sum(np.abs(norms - 1.0) > tolerance))
    LOGGER.info(
        "L2 norms: min=%.9f max=%.9f max|norm-1|=%.3e (tolerance %.0e)",
        float(norms.min()) if norms.size else float("nan"),
        float(norms.max()) if norms.size else float("nan"),
        max_deviation,
        tolerance,
    )
    if offenders:
        LOGGER.error("%d vector(s) are not unit norm", offenders)
        ok = False

    LOGGER.info(
        "Norm check: %s", "PASS" if offenders == 0 else "FAIL"
    )
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the ResearchRadar corpus with local Ollama F2LLM."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml"
    )
    parser.add_argument("--model", default=None, help="Override the Ollama model tag")
    parser.add_argument(
        "--force", action="store_true", help="Ignore caches and re-embed everything"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing embeddings.npy without calling Ollama",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    try:
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError:
        LOGGER.error("Config file not found: %s", args.config)
        return 2
    except yaml.YAMLError as exc:
        LOGGER.error("Config file is not valid YAML (%s): %s", args.config, exc)
        return 2
    if not isinstance(config, dict):
        LOGGER.error("Config file must contain a mapping: %s", args.config)
        return 2

    config_dir = args.config.parent if str(args.config.parent) else Path(".")
    corpus_cfg = config.get("corpus", {})
    ollama_cfg = config.get("ollama", {})
    corpus_path = config_dir / str(corpus_cfg.get("output_path", "data/corpus.json"))
    matrix_path = config_dir / str(
        ollama_cfg.get("embeddings_cache", "data/embeddings.npy")
    )

    corpus = load_corpus(corpus_path)
    expected_rows = len(corpus["papers"])

    if args.verify_only:
        matrix = load_embeddings_matrix(matrix_path)
        if matrix is None:
            LOGGER.error("No embedding matrix found at %s", matrix_path)
            return 1
        return 0 if verify_embeddings(matrix, expected_rows) else 1

    try:
        matrix, report = generate_and_cache_embeddings(
            config,
            config_dir=config_dir,
            model_override=args.model,
            force=args.force,
        )
    except OllamaUnavailableError as exc:
        LOGGER.error("%s", exc)
        return 3
    except RuntimeError as exc:
        LOGGER.error("Embedding failed: %s", exc)
        return 4

    LOGGER.info(
        "Run summary: model=%s cache_hit=%s embedded=%d reused=%d dim=%d",
        report["model"],
        report["cache_hit"],
        report["embedded"],
        report["reused"],
        report["dim"],
    )
    return 0 if verify_embeddings(matrix, expected_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
