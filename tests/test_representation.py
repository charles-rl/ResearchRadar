"""Tests for the Phase 2 dual-norm character n-gram representation.

The behaviours that matter here are the ones that silently change the science if
they drift: which n-grams are allowed to exist, which weighting formula is
applied, and whether the two normalisations really come from one matrix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.embed import (
    L1_TOLERANCE,
    L2_TOLERANCE,
    Representation,
    RepresentationError,
    TfidfSettings,
    build_representation,
    build_weighted_matrix,
    build_word_index,
    family_stem,
    load_representation,
    save_representation,
    verify_representation,
)

DOCUMENTS = [
    "robot arm control with a learned policy",
    "the policy learns from domain randomization",
    "system identification of a robot arm",
    "a survey of control methods",  # deliberately contains no 7-gram repeats
]


def make_corpus(texts: list[str]) -> dict:
    """Minimal full-text corpus shaped like ``data/corpus_fulltext.json``."""
    return {
        "papers": [
            {
                "id": f"2000.0000{index}",
                "title": f"paper {index}",
                "abstract": "an abstract",
                "full_text": text,
                "authors": ["A. Author"],
                "year": 2020,
                "regime_id": index % 3,
                "regime_label": "SysID",
            }
            for index, text in enumerate(texts)
        ]
    }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_settings_defaults_match_the_paper() -> None:
    settings = TfidfSettings.from_config({})
    assert settings.ngram_range == (4, 7)
    assert settings.max_features == 250_000
    assert settings.sublinear_tf


def test_settings_reject_malformed_ranges() -> None:
    with pytest.raises(SystemExit):
        TfidfSettings.from_config({"ngram_range": [4]})
    with pytest.raises(SystemExit):
        TfidfSettings.from_config({"ngram_range": ["four", "seven"]})
    with pytest.raises(SystemExit):
        TfidfSettings.from_config({"ngram_range": [7, 4]})


def test_settings_reject_malformed_max_features() -> None:
    with pytest.raises(SystemExit):
        TfidfSettings.from_config({"max_features": "lots"})


def test_settings_as_dict_records_the_formulas() -> None:
    payload = TfidfSettings.from_config({}).as_dict()
    assert payload["analyzer"] == "char_wb"
    assert payload["idf"] == "log(1 + N/df) + 1"
    assert payload["tf"] == "1 + log(tf)"


# --------------------------------------------------------------------------- #
# Vocabulary shape
# --------------------------------------------------------------------------- #
def test_ngrams_stay_inside_a_word() -> None:
    """``char_wb`` must never join two words into one n-gram."""
    _, terms, _, _ = build_weighted_matrix(
        ["robot arm control"], TfidfSettings.from_config({})
    )
    for term in terms:
        assert " " not in term.strip(), f"{term!r} contains an internal space"


def test_ngram_lengths_respect_the_configured_range() -> None:
    """No feature is longer than the configured maximum.

    The lower bound needs one caveat: ``char_wb`` emits the whole padded word
    once when a word is shorter than ``n`` (so " a " survives as a 3-character
    feature). Those are the only sub-minimum features allowed.
    """
    _, terms, _, _ = build_weighted_matrix(DOCUMENTS, TfidfSettings.from_config({}))
    lengths = {len(term) for term in terms}
    assert lengths, "expected at least one feature"
    assert max(lengths) <= 7
    short = [term for term in terms if len(term) < 4]
    assert all(len(term.strip()) <= 1 for term in short), short[:5]


def test_max_features_is_enforced() -> None:
    _, terms, _, _ = build_weighted_matrix(
        DOCUMENTS, TfidfSettings.from_config({"max_features": 12})
    )
    assert len(terms) == 12


def test_empty_documents_are_rejected() -> None:
    with pytest.raises(RepresentationError):
        build_weighted_matrix(["", "   ", "\n"], TfidfSettings.from_config({}))


# --------------------------------------------------------------------------- #
# Weighting formulas
# --------------------------------------------------------------------------- #
def test_idf_matches_the_documented_formula() -> None:
    """``idf = log(1 + N/df) + 1`` using the document frequency of each term."""
    texts = ["aaaa bbbb", "aaaa cccc", "aaaa dddd"]
    _, terms, document_frequency, idf = build_weighted_matrix(
        texts, TfidfSettings.from_config({"max_features": 5000})
    )
    n_documents = len(texts)
    for term, df, value in zip(terms, document_frequency, idf, strict=True):
        expected = math.log1p(n_documents / df) + 1.0
        assert value == pytest.approx(expected), term


def test_sublinear_term_frequency_is_applied() -> None:
    """A term occurring twice must weigh ``1 + log(2)``, not 2."""
    # "aaaaaa" repeated produces the 6-gram "aaaaaa" exactly twice in one doc.
    _, terms, _, _ = build_weighted_matrix(
        ["aaaaaa aaaaaa"], TfidfSettings.from_config({"max_features": 5000})
    )
    document_raw = "aaaaaa aaaaaa"
    settings = TfidfSettings.from_config({"max_features": 5000})
    plain = TfidfSettings.from_config(
        {"max_features": 5000, "sublinear_tf": False}
    )
    weighted_sublinear, terms_sublinear, _, _ = build_weighted_matrix(
        ["aaaaaa aaaaaa", "zzzzzz"], settings
    )
    weighted_plain, _, _, _ = build_weighted_matrix(
        ["aaaaaa aaaaaa", "zzzzzz"], plain
    )
    index = terms_sublinear.index("aaaaaa")
    sublinear_value = weighted_sublinear[0, index]
    plain_value = weighted_plain[0, index]
    # Both share the same idf, so the ratio isolates the tf transform.
    assert plain_value / sublinear_value == pytest.approx(
        2.0 / (1.0 + math.log(2.0))
    )
    assert len(terms) > 0
    assert document_raw


# --------------------------------------------------------------------------- #
# Dual normalisation
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def representation() -> Representation:
    return build_representation(
        make_corpus(DOCUMENTS), TfidfSettings.from_config({}), "deadbeef"
    )


def test_l2_rows_have_unit_norm(representation: Representation) -> None:
    norms = np.sqrt(
        np.asarray(representation.z_l2.multiply(representation.z_l2).sum(axis=1)).ravel()
    )
    assert np.allclose(norms, 1.0, atol=L2_TOLERANCE)


def test_l1_rows_sum_to_one(representation: Representation) -> None:
    sums = np.asarray(representation.z_l1.sum(axis=1)).ravel()
    assert np.allclose(sums, 1.0, atol=L1_TOLERANCE)


def test_both_norms_derive_from_one_weighted_matrix(
    representation: Representation,
) -> None:
    """Per row, L1 and L2 differ only by a scalar: the L1 sum of the raw row."""
    z_l2 = representation.z_l2.toarray()
    z_l1 = representation.z_l1.toarray()
    for row in range(z_l2.shape[0]):
        nonzero = z_l2[row] > 0
        ratios = z_l1[row][nonzero] / z_l2[row][nonzero]
        assert np.allclose(ratios, ratios[0]), "normalisations disagree"


def test_document_ids_follow_corpus_order(representation: Representation) -> None:
    assert representation.doc_ids == ["2000.00000", "2000.00001", "2000.00002", "2000.00003"]


# --------------------------------------------------------------------------- #
# Persistence and verification
# --------------------------------------------------------------------------- #
def test_round_trip_through_disk(
    tmp_path: Path, representation: Representation
) -> None:
    matrix_path = tmp_path / "sparse.npz"
    vocab_path = tmp_path / "vocab.json"
    save_representation(representation, matrix_path, vocab_path)
    reloaded = load_representation(matrix_path, vocab_path)
    assert reloaded.n_documents == representation.n_documents
    assert reloaded.n_features == representation.n_features
    assert reloaded.terms == representation.terms
    assert reloaded.doc_ids == representation.doc_ids
    assert reloaded.corpus_hash == "deadbeef"
    assert abs(reloaded.z_l2 - representation.z_l2).max() == 0.0
    assert abs(reloaded.z_l1 - representation.z_l1).max() == 0.0


def test_vocab_artifact_is_json_with_the_formulas(
    tmp_path: Path, representation: Representation
) -> None:
    matrix_path = tmp_path / "sparse.npz"
    vocab_path = tmp_path / "vocab.json"
    save_representation(representation, matrix_path, vocab_path)
    payload = json.loads(vocab_path.read_text())
    assert payload["n_features"] == representation.n_features
    assert payload["metadata"]["idf"] == "log(1 + N/df) + 1"
    assert len(payload["terms"]) == payload["n_features"]
    assert len(payload["document_frequency"]) == payload["n_features"]


def test_verify_passes_on_a_healthy_representation(
    tmp_path: Path, representation: Representation
) -> None:
    matrix_path = tmp_path / "sparse.npz"
    vocab_path = tmp_path / "vocab.json"
    save_representation(representation, matrix_path, vocab_path)
    assert verify_representation(
        representation,
        matrix_path,
        vocab_path,
        expected_documents=len(DOCUMENTS),
    )


def test_verify_rejects_a_wrong_document_count(
    tmp_path: Path, representation: Representation
) -> None:
    matrix_path = tmp_path / "sparse.npz"
    vocab_path = tmp_path / "vocab.json"
    save_representation(representation, matrix_path, vocab_path)
    assert not verify_representation(
        representation, matrix_path, vocab_path, expected_documents=99
    )


def test_verify_rejects_an_unnormalised_row(
    tmp_path: Path, representation: Representation
) -> None:
    matrix_path = tmp_path / "sparse.npz"
    vocab_path = tmp_path / "vocab.json"
    save_representation(representation, matrix_path, vocab_path)
    broken = Representation(
        z_l2=representation.z_l2 * 2.0,
        z_l1=representation.z_l1,
        terms=representation.terms,
        document_frequency=representation.document_frequency,
        idf=representation.idf,
        doc_ids=representation.doc_ids,
        settings=representation.settings,
        corpus_hash=representation.corpus_hash,
        word_index=representation.word_index,
    )
    assert not verify_representation(broken, matrix_path, vocab_path)


def test_load_mentions_the_missing_step(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="src.embed"):
        load_representation(tmp_path / "nope.npz", tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# Whole-word reconstruction (Appendix A.1)
# --------------------------------------------------------------------------- #
def test_family_stem_groups_the_paper_example() -> None:
    """vortex / vortices / vortical must share a key."""
    assert family_stem("vortex") == family_stem("vortices") == family_stem("vortical")


def test_family_stem_keeps_unrelated_words_apart() -> None:
    """Prefix grouping merged these; the suffix rule must not."""
    for left, right in (
        ("manifest", "manifold"),
        ("control", "contrast"),
        ("reinforcement", "reinhardt"),
    ):
        assert family_stem(left) != family_stem(right), (left, right)


def test_family_stem_is_deterministic() -> None:
    assert family_stem("identification") == family_stem("identification")


def test_word_index_attaches_ngrams_to_whole_words() -> None:
    """Grouping must be traceable to attested corpus forms, not guessed."""
    texts = ["vortex vortices vortical", "a vortex ring"]
    _, terms, _, _ = build_weighted_matrix(texts, TfidfSettings(ngram_range=(4, 5)))
    index = build_word_index(texts, terms, (4, 5))

    assert "vortex" in index.family_terms or "vortices" in index.family_terms
    forms = {form for family in index.family_forms for form in family}
    assert {"vortex", "vortices", "vortical"} <= forms
    # Every listed form is a real corpus word, never an n-gram fragment.
    for form in forms:
        assert form.isalpha()
        assert len(form) >= 4


def test_word_index_document_frequency_counts_documents_not_ngrams() -> None:
    """``control`` and ``contrast`` share the 4-gram "contr" but are two words.

    Counting n-grams instead of words reported df=2 for both, which saturated
    every family at the document count on the real corpus and destroyed the
    df >= 4 / df <= 3 grading of Appendix A.1 Rule 2.
    """
    texts = ["control loop", "contrast ratio"]
    _, terms, _, _ = build_weighted_matrix(texts, TfidfSettings(ngram_range=(4, 5)))
    index = build_word_index(texts, terms, (4, 5))

    by_term = dict(zip(index.family_terms, index.family_document_frequency, strict=True))
    assert family_stem("control") != family_stem("contrast")
    assert by_term["control"] == 1
    assert by_term["contrast"] == 1
    assert max(index.family_document_frequency) <= len(texts)


def test_word_index_reports_its_coverage() -> None:
    texts = ["system identification with a controller"]
    _, terms, _, _ = build_weighted_matrix(texts, TfidfSettings(ngram_range=(4, 7)))
    index = build_word_index(texts, terms, (4, 7))
    assert 0.0 < index.coverage <= 1.0
    assert index.n_families > 0
