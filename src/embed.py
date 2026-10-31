"""Phase 2: dual-norm within-word character n-gram TF-IDF.

Obayashi et al. (2026) Sec. 5.2 build the whole representation from sparse
character n-gram statistics and state in Sec. 5.5 that no language model is used
in any objective or in the frontier extraction. This module reproduces that:
no neural encoder is involved anywhere in the pipeline.

Three details are load-bearing and easy to get wrong:

``analyzer="char_wb"``
    n-grams are built from *within* a word only, and each word is padded with a
    space at both edges. So ``"robot"`` yields ``" rob"`` and ``"obot "`` but
    never ``"ts ar"`` across a word boundary. A plain ``"char"`` analyzer would
    straddle boundaries and leak phrase fragments into the vocabulary.

Two normalisations of one matrix
    The same weighted matrix is stored twice: L2-normalised rows drive the map
    and the semantic gradient ``v(x)``, L1-normalised rows drive the
    feature-allocation view ``u(x)`` used when reading evidence packages. They
    must come from the same raw representation, so the weighting happens once
    and the normalisation happens twice.

Sublinear TF and smoothed IDF
    ``tf' = 1 + log(tf)`` and ``idf = log(1 + N/df) + 1``. Both are computed
    explicitly rather than delegated to ``TfidfVectorizer`` because the paper's
    smoothed IDF is ``log(1 + N/df)`` while scikit-learn's ``smooth_idf`` is
    ``log((1 + N)/(1 + df))``. The two differ on rare terms, which is exactly
    where the L1 lens looks.

Whole words reconstructed from the n-grams
    Appendix A.1 requires evidence terms to be "whole words, deterministically
    reconstructed from the corpus text (each entry lists its attested
    ``word_forms``). No fragment decoding is needed." A char n-gram vocabulary
    is fragments by construction (``vort``, ``ortic``, ``rtic``), so this module
    also builds an n-gram -> parent-word index and a word-family map. Slicing the
    lenses then aggregates n-gram scores back onto whole words, which is why the
    lenses share the map's basis instead of a separate word-level vectoriser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.common import (
    LOGGER,
    Paths,
    configure_logging,
    load_config,
    resolve_paths,
)

DEFAULT_CONFIG = Path("config.yaml")

#: Sec. 5.2 defaults, used when config.yaml omits or mangles the block.
DEFAULT_NGRAM_RANGE = (4, 7)
DEFAULT_MAX_FEATURES = 250_000

#: A whitespace token becomes a parent word only if it looks like a word: this
#: keeps equation fragments and bare numbers out of the evidence packages while
#: leaving their n-grams in the geometry, which is what the paper wants.
_WORD_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
_LEADING_TRAILING = "\"'()[]{}<>,.;:!?*_"

MIN_WORD_LENGTH = 3

#: Bump when the way features or word families are derived changes. It feeds the
#: build fingerprint, so a bump invalidates every stored representation.
REPRESENTATION_VERSION = 3

#: Row-normalisation tolerances. Sparse arithmetic in float64 lands far inside
#: these; a failure means a genuinely zero-norm row, not rounding drift.
L1_TOLERANCE = 1e-9
L2_TOLERANCE = 1e-9


class RepresentationError(RuntimeError):
    """Raised when the corpus cannot produce a usable representation."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TfidfSettings:
    """The Sec. 5.2 representation parameters, read from ``config.yaml``."""

    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
    max_features: int = DEFAULT_MAX_FEATURES
    sublinear_tf: bool = True
    lowercase: bool = True

    @classmethod
    def from_config(cls, tfidf_cfg: dict[str, Any]) -> TfidfSettings:
        raw_range = tfidf_cfg.get("ngram_range", list(DEFAULT_NGRAM_RANGE))
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise SystemExit(
                "tfidf.ngram_range must be a two-element list, "
                f"got {raw_range!r}"
            )
        try:
            low, high = int(raw_range[0]), int(raw_range[1])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"tfidf.ngram_range entries must be integers, got {raw_range!r}"
            ) from exc
        if low < 1 or high < low:
            raise SystemExit(
                f"tfidf.ngram_range must satisfy 1 <= low <= high, got {raw_range!r}"
            )
        try:
            max_features = int(
                tfidf_cfg.get("max_features", DEFAULT_MAX_FEATURES)
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "tfidf.max_features must be an integer, "
                f"got {tfidf_cfg.get('max_features')!r}"
            ) from exc
        if max_features < 1:
            raise SystemExit(
                f"tfidf.max_features must be positive, got {max_features}"
            )
        return cls(
            ngram_range=(low, high),
            max_features=max_features,
            sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
            lowercase=bool(tfidf_cfg.get("lowercase", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view, written into the vocabulary artifact."""
        return {
            "analyzer": "char_wb",
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "tf": "1 + log(tf)" if self.sublinear_tf else "tf",
            "idf": "log(1 + N/df) + 1",
            "lowercase": self.lowercase,
            "normalisations": ["l2 (map, semantic gradient)", "l1 (feature allocation)"],
        }

    def fingerprint(self) -> str:
        """Stable hash of everything that changes the stored matrices.

        The corpus MD5 alone is not enough: changing ``ngram_range`` or the
        whole-word family rule re-derives every column while leaving the corpus
        byte-identical, and a corpus-only cache key would then silently serve a
        stale matrix. Bumping ``REPRESENTATION_VERSION`` is the manual lever for
        code changes that this dictionary cannot see.
        """
        payload = json.dumps(
            {
                "version": REPRESENTATION_VERSION,
                "settings": self.as_dict(),
                "family_suffixes": list(_FAMILY_SUFFIXES),
                "min_word_length": MIN_WORD_LENGTH,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.new("md5", payload, usedforsecurity=False).hexdigest()


# --------------------------------------------------------------------------- #
# Corpus I/O
# --------------------------------------------------------------------------- #
def compute_corpus_hash(path: Path) -> str:
    """MD5 of the full-text corpus, used to decide whether a rebuild is needed.

    ``usedforsecurity=False`` is deliberate: this is a cache key, not a
    security control, and the flag keeps stricter FIPS builds happy.
    """
    digest = hashlib.new("md5", usedforsecurity=False)
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Full-text corpus not found: {path}\n"
            "Run `python -m src.harvest --extract-text` first."
        ) from exc
    return digest.hexdigest()


def load_fulltext_corpus(path: Path) -> dict[str, Any]:
    """Load ``data/corpus_fulltext.json`` and fail with an actionable message."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Full-text corpus not found: {path}\n"
            "Run `python -m src.harvest --extract-text` first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Full-text corpus is not valid JSON ({path}): {exc}") from exc

    papers = payload.get("papers")
    if not isinstance(papers, list) or not papers:
        raise SystemExit(f"Full-text corpus has no papers: {path}")
    for paper in papers:
        if not str(paper.get("full_text", "")).strip():
            raise SystemExit(
                f"Paper {paper.get('id')!r} has no full text; "
                "re-run `python -m src.harvest --extract-text`"
            )
    return payload


# --------------------------------------------------------------------------- #
# Representation
# --------------------------------------------------------------------------- #
@dataclass
class WordIndex:
    """n-gram -> whole-word mapping (Appendix A.1 reconstruction).

    ``feature_family`` is an indicator matrix of shape ``(n_families,
    n_features)``: entry ``(g, f)`` is 1 when n-gram ``f`` was attested inside a
    word of family ``g`` in the corpus. Multiplying a per-n-gram score vector by
    its transpose therefore aggregates n-gram scores back onto whole words,
    which is the operation the paper describes as reconstructing terms.
    """

    feature_family: sp.csr_matrix
    family_terms: list[str]
    family_forms: list[list[str]]
    family_document_frequency: np.ndarray
    corpus_word_counts: dict[str, int]
    coverage: float

    @property
    def n_families(self) -> int:
        """Number of whole-word families."""
        return int(self.feature_family.get_shape()[0])


@dataclass
class Representation:
    """The dual-norm representation plus everything downstream stages need."""

    z_l2: sp.csr_matrix
    z_l1: sp.csr_matrix
    terms: list[str]
    document_frequency: np.ndarray
    idf: np.ndarray
    doc_ids: list[str]
    settings: TfidfSettings
    corpus_hash: str
    word_index: WordIndex
    fingerprint: str = ""

    @property
    def n_documents(self) -> int:
        """Number of rows (documents)."""
        return int(self.z_l2.get_shape()[0])

    @property
    def n_features(self) -> int:
        """Number of columns (retained n-gram features)."""
        return int(self.z_l2.get_shape()[1])


def build_weighted_matrix(
    documents: Iterable[str], settings: TfidfSettings
) -> tuple[sp.csr_matrix, list[str], np.ndarray, np.ndarray]:
    """Weighted sparse matrix plus vocabulary, document frequency, and idf.

    One vectoriser pass produces the sublinear term frequencies; the smoothed
    IDF is then applied explicitly so the formula matches the paper exactly.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=settings.ngram_range,
        max_features=settings.max_features,
        lowercase=settings.lowercase,
        sublinear_tf=settings.sublinear_tf,
        use_idf=False,
        # ``norm=None`` keeps raw sublinear term frequencies; both norms are
        # applied afterwards. scikit-learn's stub types this as ``str``.
        norm=None,  # type: ignore[arg-type]
        dtype=np.float64,
    )
    try:
        term_frequency = cast(sp.csr_matrix, vectorizer.fit_transform(documents))
    except ValueError as exc:
        # scikit-learn raises "empty vocabulary" for text with no n-grams at all.
        raise RepresentationError(
            f"The vectoriser could not build a vocabulary: {exc}. Check that "
            "tfidf.ngram_range is not longer than the words in the corpus"
        ) from exc
    if term_frequency.get_shape()[1] == 0:
        raise RepresentationError(
            "The vectoriser produced no features; check that the corpus text is "
            "non-empty and that tfidf.ngram_range is not larger than the words"
        )

    # Document frequency: how many documents contain each feature at least once.
    present = cast(sp.csr_matrix, (term_frequency > 0))
    document_frequency = np.asarray(present.sum(axis=0), dtype=np.float64).ravel()
    n_documents = term_frequency.get_shape()[0]
    # log(1 + N/df) + 1, with df >= 1 for every retained feature.
    idf = np.log1p(n_documents / document_frequency) + 1.0

    weighted = term_frequency.multiply(idf[np.newaxis, :]).tocsr()
    weighted.sort_indices()
    terms = list(vectorizer.get_feature_names_out())
    return weighted, terms, document_frequency, idf


# --------------------------------------------------------------------------- #
# Whole-word reconstruction (Appendix A.1)
# --------------------------------------------------------------------------- #
def _strip_token(token: str) -> str:
    """Lowercase a token and trim punctuation, without touching inner hyphens."""
    return token.strip(_LEADING_TRAILING).lower()


#: Suffixes stripped (longest first) when reducing a surface form to its stem.
#: Deliberately short: this is a family-naming aid, not a linguistic lemmatiser,
#: and every aggregated form is still listed verbatim in ``word_forms`` so a
#: reader can see exactly what was merged.
_FAMILY_SUFFIXES = (
    "ational", "ization", "isation", "ations", "ation", "izing", "ising",
    "ities", "ity", "ives", "ements", "ement", "ings", "ing", "ies", "ied",
    "ers", "er", "ed", "es", "ally", "al", "ly", "s",
)
_VOWELS = frozenset("aeiou")


def family_stem(word: str) -> str:
    """Reduce a surface form to a deterministic family key.

    Four documented transformations, applied in order: strip one suffix from
    ``_FAMILY_SUFFIXES``, collapse a doubled final consonant, fold the Latin
    ``-ex``/``-ix`` ending onto ``-i``, drop a trailing silent ``e``. The result
    is a grouping key only - it is never shown to a reader.

    The rule is deliberately conservative: families may **under-merge**
    (``vorticity`` can land on its own key next to ``vortex``/``vortices``), but
    they never mis-merge unrelated words the way prefix grouping does
    (``manifest``/``manifold``, ``control``/``contrast``). Under-merging is the
    safe failure: every aggregated entry still lists its attested ``word_forms``
    verbatim, so a reader can see exactly what was combined, whereas a bad merge
    would silently present one word's evidence under another word's name.
    """
    stem = word
    for suffix in _FAMILY_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
            stem = stem[: -len(suffix)]
            break
    if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        stem = stem[:-1]
    if len(stem) >= 5 and stem.endswith(("ex", "ix")):
        stem = stem[:-2] + "i"
    if len(stem) >= 4 and stem.endswith("ic"):
        stem = stem[:-1]
    if len(stem) >= 4 and stem.endswith("e"):
        stem = stem[:-1]
    return stem


def word_families(words: Sequence[str]) -> dict[str, str]:
    """Map every corpus word to the family key of its stem."""
    return {word: family_stem(word) for word in words}


def document_word_counts(texts: Sequence[str]) -> list[Counter[str]]:
    """Per-document word counts, using the same token rule as the corpus count."""
    per_document: list[Counter[str]] = []
    for text in texts:
        counts: Counter[str] = Counter()
        for token in text.split():
            word = _strip_token(token)
            if len(word) < MIN_WORD_LENGTH or not _WORD_PATTERN.fullmatch(word):
                continue
            counts[word] += 1
        per_document.append(counts)
    return per_document


def build_word_index(
    texts: Sequence[str],
    terms: Sequence[str],
    ngram_range: tuple[int, int],
) -> WordIndex:
    """Build the n-gram -> whole-word indicator matrix.

    For every corpus word, its padding-free character n-grams are looked up in
    the retained feature vocabulary and the corresponding entries of the
    indicator matrix are set. A n-gram may belong to several families (``vort``
    is in the vortex family and in *vortex-free*), which is expected: the
    aggregation is a projection, not a partition.
    """
    feature_index = {term: position for position, term in enumerate(terms)}
    per_document = document_word_counts(texts)
    counts: Counter[str] = Counter()
    for document_counts in per_document:
        counts.update(document_counts)
    families = word_families(list(counts))

    # Name each family after its most frequent attested form.
    representative: dict[str, str] = {}
    for word, root in families.items():
        current = representative.get(root)
        if current is None or counts[word] > counts[current]:
            representative[root] = word

    forms_by_root: dict[str, list[str]] = {}
    for word, root in families.items():
        forms_by_root.setdefault(root, []).append(word)

    roots = sorted(forms_by_root, key=lambda root: representative[root])
    root_position = {root: position for position, root in enumerate(roots)}

    rows: list[int] = []
    columns: list[int] = []
    low, high = ngram_range
    for root, forms in forms_by_root.items():
        row = root_position[root]
        for form in forms:
            for size in range(low, high + 1):
                if size > len(form):
                    continue
                for start in range(len(form) - size + 1):
                    column = feature_index.get(form[start : start + size])
                    if column is not None:
                        rows.append(row)
                        columns.append(column)

    # Deduplicate the (family, feature) pairs before building the matrix.
    pairs = set(zip(rows, columns, strict=True))
    row_indices = [pair[0] for pair in pairs]
    column_indices = [pair[1] for pair in pairs]
    indicator = sp.csr_matrix(
        (
            np.ones(len(pairs), dtype=np.float64),
            (row_indices, column_indices),
        ),
        shape=(len(roots), len(terms)),
    )
    indicator.sort_indices()

    family_terms = [representative[root] for root in roots]
    family_forms = [sorted(forms_by_root[root]) for root in roots]
    # A family's document frequency counts documents that contain one of its
    # *words*, not one of its n-grams. Counting n-grams would saturate: the
    # 4-gram "repr" sits inside "represent", "reproduce", and "repress", so
    # n-gram presence reports 120/120 documents and Rule 2's df >= 4 / df <= 3
    # grading collapses to "everything is emergent".
    family_document_frequency = np.zeros(len(roots), dtype=np.int64)
    root_position_by_word = {
        word: root_position[families[word]] for word in families
    }
    for document_counts in per_document:
        touched = {
            root_position_by_word[word] for word in document_counts if word in root_position_by_word
        }
        for position in touched:
            family_document_frequency[position] += 1

    covered = set(row_indices)
    coverage = len(covered) / len(feature_index) if feature_index else 0.0
    LOGGER.info(
        "Whole-word index: %d corpus words -> %d families, %d of %d n-grams "
        "attributed to a word (%.1f%%)",
        len(counts),
        len(family_terms),
        len(covered),
        len(feature_index),
        100.0 * coverage,
    )
    return WordIndex(
        feature_family=indicator,
        family_terms=family_terms,
        family_forms=family_forms,
        family_document_frequency=family_document_frequency,
        corpus_word_counts=dict(counts),
        coverage=coverage,
    )


def build_representation(
    corpus: dict[str, Any],
    settings: TfidfSettings,
    corpus_hash: str = "",
) -> Representation:
    """Build both normalisations from one weighting pass."""
    papers = corpus["papers"]
    texts = [str(paper["full_text"]) for paper in papers]
    doc_ids = [str(paper["id"]) for paper in papers]

    started = time.monotonic()
    weighted, terms, document_frequency, idf = build_weighted_matrix(
        texts, settings
    )
    LOGGER.info(
        "Vectorised %d documents into %d within-word %d-%d-grams in %.1fs",
        weighted.get_shape()[0],
        weighted.get_shape()[1],
        settings.ngram_range[0],
        settings.ngram_range[1],
        time.monotonic() - started,
    )

    z_l2 = cast(sp.csr_matrix, normalize(weighted, norm="l2", axis=1, copy=True)).tocsr()
    z_l1 = cast(sp.csr_matrix, normalize(weighted, norm="l1", axis=1, copy=True)).tocsr()
    z_l2.sort_indices()
    z_l1.sort_indices()

    LOGGER.info(
        "Non-zeros: %d (density %.4f%%)",
        z_l2.nnz,
        100.0 * z_l2.nnz / max(1, z_l2.get_shape()[0] * z_l2.get_shape()[1]),
    )
    word_index = build_word_index(texts, terms, settings.ngram_range)

    return Representation(
        z_l2=z_l2,
        z_l1=z_l1,
        terms=terms,
        document_frequency=document_frequency,
        idf=idf,
        doc_ids=doc_ids,
        settings=settings,
        corpus_hash=corpus_hash,
        word_index=word_index,
        fingerprint=settings.fingerprint(),
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_representation(
    representation: Representation, matrix_path: Path, vocab_path: Path
) -> None:
    """Write the sparse matrices and the vocabulary, both atomically."""
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_tmp = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    with matrix_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            z_l2_data=representation.z_l2.data,
            z_l2_indices=representation.z_l2.indices,
            z_l2_indptr=representation.z_l2.indptr,
            z_l1_data=representation.z_l1.data,
            z_l1_indices=representation.z_l1.indices,
            z_l1_indptr=representation.z_l1.indptr,
            shape=np.asarray(representation.z_l2.shape, dtype=np.int64),
            document_frequency=representation.document_frequency,
            idf=representation.idf,
            word_family_data=representation.word_index.feature_family.data,
            word_family_indices=representation.word_index.feature_family.indices,
            word_family_indptr=representation.word_index.feature_family.indptr,
            word_family_shape=np.asarray(
                representation.word_index.feature_family.shape, dtype=np.int64
            ),
            word_family_document_frequency=(
                representation.word_index.family_document_frequency
            ),
        )
    matrix_tmp.replace(matrix_path)

    vocab_tmp = vocab_path.with_suffix(vocab_path.suffix + ".tmp")
    with vocab_tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": representation.settings.as_dict(),
                "corpus_hash": representation.corpus_hash,
                "fingerprint": representation.fingerprint,
                "n_documents": representation.n_documents,
                "n_features": representation.n_features,
                "doc_ids": representation.doc_ids,
                "terms": representation.terms,
                "document_frequency": [
                    int(value) for value in representation.document_frequency
                ],
                "word_family_terms": representation.word_index.family_terms,
                "word_family_forms": representation.word_index.family_forms,
                "word_family_coverage": representation.word_index.coverage,
                "n_word_families": representation.word_index.n_families,
            },
            handle,
            ensure_ascii=False,
        )
        handle.write("\n")
    vocab_tmp.replace(vocab_path)

    LOGGER.info(
        "Wrote %s (%.1f MB) and %s (%.1f MB)",
        matrix_path,
        matrix_path.stat().st_size / 1e6,
        vocab_path,
        vocab_path.stat().st_size / 1e6,
    )


def load_representation(matrix_path: Path, vocab_path: Path) -> Representation:
    """Reload both normalisations with the vocabulary attached."""
    try:
        with np.load(matrix_path, allow_pickle=False) as archive:
            shape = tuple(int(value) for value in archive["shape"])
            z_l2 = sp.csr_matrix(
                (
                    archive["z_l2_data"],
                    archive["z_l2_indices"],
                    archive["z_l2_indptr"],
                ),
                shape=shape,
            )
            z_l1 = sp.csr_matrix(
                (
                    archive["z_l1_data"],
                    archive["z_l1_indices"],
                    archive["z_l1_indptr"],
                ),
                shape=shape,
            )
            document_frequency = np.asarray(archive["document_frequency"])
            idf = np.asarray(archive["idf"])
            family_shape = tuple(int(value) for value in archive["word_family_shape"])
            feature_family = sp.csr_matrix(
                (
                    archive["word_family_data"],
                    archive["word_family_indices"],
                    archive["word_family_indptr"],
                ),
                shape=family_shape,
            )
            family_document_frequency = np.asarray(
                archive["word_family_document_frequency"]
            )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"TF-IDF matrix not found: {matrix_path}\n"
            "Run `python -m src.embed` first."
        ) from exc

    try:
        with vocab_path.open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"TF-IDF vocabulary not found: {vocab_path}\n"
            "Run `python -m src.embed` first."
        ) from exc

    metadata = vocab.get("metadata", {})
    raw_range = metadata.get("ngram_range", list(DEFAULT_NGRAM_RANGE))
    settings = TfidfSettings(
        ngram_range=(int(raw_range[0]), int(raw_range[1])),
        max_features=int(metadata.get("max_features", DEFAULT_MAX_FEATURES)),
        sublinear_tf=bool(metadata.get("sublinear_tf", True)),
        lowercase=bool(metadata.get("lowercase", True)),
    )
    word_index = WordIndex(
        feature_family=feature_family,
        family_terms=list(vocab["word_family_terms"]),
        family_forms=[list(forms) for forms in vocab["word_family_forms"]],
        family_document_frequency=family_document_frequency,
        corpus_word_counts={},
        coverage=float(vocab.get("word_family_coverage", 0.0)),
    )
    return Representation(
        z_l2=z_l2,
        z_l1=z_l1,
        terms=list(vocab["terms"]),
        document_frequency=document_frequency,
        idf=idf,
        doc_ids=list(vocab["doc_ids"]),
        settings=settings,
        corpus_hash=str(vocab.get("corpus_hash", "")),
        word_index=word_index,
        fingerprint=str(vocab.get("fingerprint", "")),
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_representation(
    representation: Representation,
    matrix_path: Path,
    vocab_path: Path,
    expected_documents: int | None = None,
) -> bool:
    """Four independent checks on the stored representation.

    1. Shapes agree with the vocabulary and the expected document count.
    2. Every L2 row has unit Euclidean norm.
    3. Every L1 row sums to one, i.e. it is a point on the probability simplex.
    4. Reloading from disk reproduces the same matrices.
    """
    ok = True
    n_documents, n_features = representation.z_l2.get_shape()
    LOGGER.info(
        "Representation: %d documents x %d features (cap %d)",
        n_documents,
        n_features,
        representation.settings.max_features,
    )

    if representation.z_l1.get_shape() != representation.z_l2.get_shape():
        LOGGER.error(
            "L1 shape %s does not match L2 shape %s",
            representation.z_l1.shape,
            representation.z_l2.shape,
        )
        ok = False
    if len(representation.terms) != n_features:
        LOGGER.error(
            "Vocabulary has %d terms but the matrix has %d columns",
            len(representation.terms),
            n_features,
        )
        ok = False
    if len(representation.doc_ids) != n_documents:
        LOGGER.error(
            "Vocabulary lists %d document ids but the matrix has %d rows",
            len(representation.doc_ids),
            n_documents,
        )
        ok = False
    if expected_documents is not None and n_documents != expected_documents:
        LOGGER.error(
            "Expected %d documents, found %d", expected_documents, n_documents
        )
        ok = False
    if n_features > representation.settings.max_features:
        LOGGER.error(
            "%d features exceeds the configured maximum of %d",
            n_features,
            representation.settings.max_features,
        )
        ok = False

    l2_norms = np.sqrt(np.asarray(representation.z_l2.multiply(representation.z_l2).sum(axis=1)).ravel())
    l1_sums = np.asarray(representation.z_l1.sum(axis=1)).ravel()
    l2_deviation = float(np.max(np.abs(l2_norms - 1.0)))
    l1_deviation = float(np.max(np.abs(l1_sums - 1.0)))
    LOGGER.info(
        "L2 norms: min=%.9f max=%.9f max|norm-1|=%.3e (tolerance %.0e)",
        float(l2_norms.min()),
        float(l2_norms.max()),
        l2_deviation,
        L2_TOLERANCE,
    )
    LOGGER.info(
        "L1 sums:  min=%.9f max=%.9f max|sum-1|=%.3e (tolerance %.0e)",
        float(l1_sums.min()),
        float(l1_sums.max()),
        l1_deviation,
        L1_TOLERANCE,
    )
    if l2_deviation > L2_TOLERANCE:
        zero_rows = int(np.sum(l2_norms == 0.0))
        LOGGER.error(
            "L2 normalisation off by %.3e; %d rows are all-zero",
            l2_deviation,
            zero_rows,
        )
        ok = False
    if l1_deviation > L1_TOLERANCE:
        LOGGER.error("L1 normalisation off by %.3e", l1_deviation)
        ok = False

    if not np.all(np.isfinite(representation.idf)):
        LOGGER.error("IDF vector contains non-finite values")
        ok = False
    if np.any(representation.document_frequency < 1):
        LOGGER.error("A retained feature has document frequency < 1")
        ok = False

    reloaded = load_representation(matrix_path, vocab_path)
    if reloaded.z_l2.get_shape() != representation.z_l2.get_shape():
        LOGGER.error("Reloaded matrix has shape %s", reloaded.z_l2.shape)
        ok = False
    elif (
        abs(reloaded.z_l2 - representation.z_l2).max() > 0.0
        or abs(reloaded.z_l1 - representation.z_l1).max() > 0.0
    ):
        LOGGER.error("Reloaded matrices differ from the in-memory matrices")
        ok = False
    elif reloaded.terms != representation.terms:
        LOGGER.error("Reloaded vocabulary differs from the in-memory vocabulary")
        ok = False
    else:
        LOGGER.info("Reload check: matrices and vocabulary round-trip exactly")

    LOGGER.info("Norm check: %s", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_and_save(
    config: dict[str, Any],
    paths: Paths,
    force: bool = False,
) -> Representation:
    """Build the representation, reusing the cache when the corpus is unchanged."""
    settings = TfidfSettings.from_config(config.get("tfidf", {}) or {})
    corpus_hash = compute_corpus_hash(paths.corpus_fulltext)

    if not force and paths.tfidf_matrix.exists() and paths.tfidf_vocab.exists():
        try:
            cached = load_representation(paths.tfidf_matrix, paths.tfidf_vocab)
        except SystemExit:
            cached = None
        fingerprint = settings.fingerprint()
        if (
            cached is not None
            and cached.corpus_hash == corpus_hash
            and cached.fingerprint == fingerprint
        ):
            LOGGER.info(
                "Cache hit: corpus MD5 %s, build %s, representation %d x %d - "
                "nothing to build",
                corpus_hash[:12],
                fingerprint[:12],
                cached.n_documents,
                cached.n_features,
            )
            return cached
        if cached is not None:
            if cached.corpus_hash != corpus_hash:
                LOGGER.info(
                    "Corpus MD5 changed (%s -> %s); rebuilding",
                    (cached.corpus_hash or "none")[:12],
                    corpus_hash[:12],
                )
            else:
                LOGGER.info(
                    "Build fingerprint changed (%s -> %s); rebuilding",
                    (cached.fingerprint or "none")[:12],
                    fingerprint[:12],
                )

    corpus = load_fulltext_corpus(paths.corpus_fulltext)
    representation = build_representation(corpus, settings, corpus_hash)
    save_representation(representation, paths.tfidf_matrix, paths.tfidf_vocab)
    return representation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the dual-norm within-word character n-gram TF-IDF "
            "representation (Obayashi et al. 2026, Sec. 5.2)."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the cache matches"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing artifacts, do not rebuild",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    paths = resolve_paths(config, args.config)
    expected = int(config.get("corpus", {}).get("target_per_regime", 40)) * 3

    try:
        if args.verify_only:
            representation = load_representation(
                paths.tfidf_matrix, paths.tfidf_vocab
            )
        else:
            representation = build_and_save(config, paths, force=args.force)
    except RepresentationError as exc:
        LOGGER.error("%s", exc)
        return 4
    except SystemExit as exc:
        LOGGER.error("%s", exc)
        return 2

    ok = verify_representation(
        representation,
        paths.tfidf_matrix,
        paths.tfidf_vocab,
        expected_documents=expected,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
