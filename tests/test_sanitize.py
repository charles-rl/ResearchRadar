"""Tests for the Phase 1 bibliography and citation sanitizers.

Two things are tested for every rule:

* the positive case, where the noise must disappear, and
* the negative case, where text that merely *looks* like noise must survive.

The negative cases matter more.  A sanitizer that deletes ``[-1, 1]`` from a
control paper destroys the mathematical content the manifold is built on, and
the damage is invisible in the output because the text is still fluent.
"""

from __future__ import annotations

import pytest

from src.harvest import (
    QualityGate,
    SanitationResult,
    assess_text_quality,
    count_bibliographic_remnants,
    looks_like_bibliography,
    normalize_extracted_text,
    sanitize_document_text,
    strip_bibliography,
    strip_citation_callouts,
    strip_front_matter,
    strip_running_headers,
)

# A body long enough to clear the 1000-word gate, built from filler that
# contains no citation-shaped tokens of its own.
FILLER = " ".join(f"sentence{index} of body prose" for index in range(300))

# A reference list that satisfies the year-density check.
REFERENCE_LIST = "\n".join(
    f"[{index}] A. Author and B. Author. A study of things. Journal of Stuff, "
    f"{1990 + index}, pp. 1-10."
    for index in range(1, 21)
)


def body_with(tail: str) -> str:
    """Full paper text: filler body followed by ``tail``."""
    return f"{FILLER}\n{tail}"


# --------------------------------------------------------------------------- #
# Bibliography truncation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "REFERENCES",
        "References:",
        "4. References",
        "IV. References",
        "Bibliography",
        "Literature Cited",
        "Works Cited",
        "7 References",
    ],
)
def test_truncates_at_reference_heading(heading: str) -> None:
    body, removed_references, _ = strip_bibliography(
        body_with(f"{heading}\n{REFERENCE_LIST}")
    )
    assert "A. Author" not in body, "reference entries must be cut"
    assert removed_references > 0
    assert "sentence0 of body prose" in body, "body must survive"


def test_truncation_is_case_insensitive_but_line_anchored() -> None:
    # Line-anchored and case-insensitive: a lowercase standalone heading matches.
    body, removed, _ = strip_bibliography(body_with(f"references\n{REFERENCE_LIST}"))
    assert removed > 0 and "A. Author" not in body


def test_in_text_mention_of_references_is_preserved() -> None:
    """A sentence mentioning references must not truncate the paper."""
    text = (
        f"{FILLER} A full list of the works discussed here is given in the "
        "references. We also note that references to prior art are incomplete."
    )
    body, removed, _ = strip_bibliography(text)
    assert removed == 0
    assert body == text


def test_heading_that_is_not_followed_by_a_bibliography_is_ignored() -> None:
    """A stray 'References' line before running prose must not cut the paper."""
    text = body_with(
        "References\nThe next section continues the analysis of the\n"
        "controller and its stability properties in detail.\n"
    )
    body, removed, _ = strip_bibliography(text)
    assert removed == 0
    assert "continues the analysis" in body


def test_truncation_refuses_to_eat_the_body() -> None:
    """When the cut would leave too few words, the text is returned untouched."""
    short = "References\n" + REFERENCE_LIST
    body, removed, _ = strip_bibliography(short, min_body_words=1000)
    assert removed == 0
    assert body == short


def test_acknowledgments_before_references_are_dropped() -> None:
    body, _, removed_ack = strip_bibliography(
        body_with(
            "Acknowledgments\nWe thank the reviewers and the funding agency.\n"
            f"References\n{REFERENCE_LIST}"
        )
    )
    assert removed_ack > 0
    assert "funding agency" not in body
    assert "sentence0 of body prose" in body


def test_acknowledgments_far_from_the_end_are_kept() -> None:
    """Only a trailing acknowledgments section is page furniture."""
    text = body_with(
        "Acknowledgments\nWe thank the reviewers.\n"
        "5. Discussion\nStill real body prose about the controller.\n"
        f"References\n{REFERENCE_LIST}"
    )
    body, _, removed_ack = strip_bibliography(text)
    assert "We thank the reviewers" in body
    assert removed_ack == 0


def test_looks_like_bibliography_detects_a_reference_list() -> None:
    assert looks_like_bibliography(REFERENCE_LIST)
    assert not looks_like_bibliography(FILLER)


# --------------------------------------------------------------------------- #
# Citation callouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "gone"),
    [
        ("built on prior work [12] and more.", "[12]"),
        ("prior art [3, 4] shows this.", "[3, 4]"),
        ("as shown previously [5; 6] and [7].", "[5; 6]"),
        ("widely used [5–9] before.", "[5–9]"),
        ("widely used [5—9] before.", "[5—9]"),
        ("robust [12a] variants.", "[12a]"),
        ("see [Smith 2020] and [Jones et al. 2019].", "[Smith 2020]"),
    ],
)
def test_numeric_and_named_citations_removed(text: str, gone: str) -> None:
    cleaned, removed = strip_citation_callouts(text)
    assert gone not in cleaned
    assert removed >= 1


@pytest.mark.parametrize(
    "text",
    [
        "prior work (Smith, 2020) showed this.",
        "prior work (Smith et al., 2020) showed this.",
        "prior work (Smith and Jones, 2019, 2020) showed this.",
        "prior work (Smith & Jones, 2019) showed this.",
        "Smith et al. (2020) showed this.",
        "Jones and Smith (2021) showed this.",
    ],
)
def test_author_year_citations_removed(text: str) -> None:
    cleaned, removed = strip_citation_callouts(text)
    assert removed >= 1
    assert "Smith" not in cleaned
    assert "Jones" not in cleaned


def test_citation_surname_before_a_bracket_is_removed() -> None:
    cleaned, _ = strip_citation_callouts(
        "As shown by Smith et al. [12] the method works."
    )
    assert "Smith" not in cleaned
    assert "the method works" in cleaned


def test_lowercase_in_trades_a_citation_for_an_interval() -> None:
    """Documented trade-off: lowercase ``in`` before a bracket means interval.

    ``x in [0, 1]`` is far more common than a citation callout written as
    ``in [5; 6]``, and deleting a genuine interval silently corrupts the
    mathematical content, so the ambiguous case is resolved in favour of the
    interval.  A sentence-initial ``In [12]`` is still treated as a citation.
    """
    guarded, _ = strip_citation_callouts("as in [5; 6] the value is clipped.")
    assert "[5; 6]" in guarded

    sentence_initial, removed = strip_citation_callouts("In [12] the authors show it.")
    assert "[12]" not in sentence_initial
    assert removed == 1


# --------------------------------------------------------------------------- #
# Locators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "noise",
    [
        "https://example.org/paper.pdf",
        "http://example.org/x?a=1",
        "www.example.org/preprint",
        "doi:10.1000/xyz123",
        "10.1109/ICRA.2019.8793846",
        "arXiv:2101.12345v2",
        "ISBN 978-3-16-148410-0",
    ],
)
def test_bibliographic_locators_removed(noise: str) -> None:
    cleaned, removed = strip_citation_callouts(f"Available at {noise} for reading.")
    assert noise not in cleaned
    assert removed >= 1
    assert "for reading" in cleaned


# --------------------------------------------------------------------------- #
# Preserved content: the guards that must not fire
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "The chart is the region [-1, 1] and the bounds are fixed.",
        "The grid spans x in [0, 1] and y in [0, 1].",
        "The support spans [0-1] by construction.",
        "Values between [2, 3] are clipped.",
        "The interval [0, 4] is discretised.",
        "Over [1, 2] the field is smooth.",
        "The domain [10, 20] is covered.",
    ],
)
def test_mathematical_intervals_survive(text: str) -> None:
    cleaned, removed = strip_citation_callouts(text)
    assert cleaned == text, "intervals must never be treated as citations"
    assert removed == 0


@pytest.mark.parametrize(
    "text",
    [
        "See Algorithm (2020) for the procedure.",
        "See Section (2020) of the appendix.",
        "Table (2020) reports the numbers.",
        "Figure (2020) shows the layout.",
        "The Lemma (2020) is immediate.",
    ],
)
def test_structural_references_with_years_survive(text: str) -> None:
    cleaned, removed = strip_citation_callouts(text)
    assert cleaned == text
    assert removed == 0


def test_year_without_an_author_is_preserved() -> None:
    text = "The dataset was collected in 2020 and released openly."
    cleaned, _ = strip_citation_callouts(text)
    assert "2020" in cleaned


# --------------------------------------------------------------------------- #
# Running headers and footers
# --------------------------------------------------------------------------- #
def test_running_headers_are_removed() -> None:
    raw = (
        "Preprint submitted to Elsevier\n"
        "Under review as a conference paper\n"
        "arXiv:2101.12345v2 [cs.RO] 12 Jan 2021\n"
        "The body text about control.\n"
        "42\n"
    )
    cleaned, removed = strip_running_headers(raw)
    assert "Preprint submitted" not in cleaned
    assert "Under review" not in cleaned
    assert "arXiv:2101" not in cleaned
    assert "42" not in cleaned
    assert "The body text about control." in cleaned
    assert removed >= 4


def test_repeated_short_lines_are_removed() -> None:
    raw = "\n".join(["Journal of Robotics Research"] * 5 + ["Real body prose here."])
    cleaned, removed = strip_running_headers(raw, repeat_threshold=3)
    assert "Journal of Robotics Research" not in cleaned
    assert "Real body prose here." in cleaned
    assert removed == 5


def test_repeated_numeric_table_lines_survive() -> None:
    """A repeated row of numbers is table data, not a running head."""
    raw = "\n".join(["0.51 0.22 0.87"] * 5 + ["Real body prose here."])
    cleaned, _ = strip_running_headers(raw, repeat_threshold=3)
    assert "0.51 0.22 0.87" in cleaned


# --------------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------------- #
def test_front_matter_before_abstract_is_removed() -> None:
    raw = (
        "Niklas Wahlstrom, Thomas B. Schon, Marc Peter Deisenroth\n"
        "Department of Electrical Engineering, Linkoping University, Sweden\n"
        "Abstract\n"
        "Modeling dynamical systems is important in control and robotics.\n"
    )
    body, removed = strip_front_matter(raw)
    assert "Linkoping University" not in body
    assert "Wahlstrom" not in body
    assert body.startswith("Modeling dynamical systems")
    assert removed > 0


def test_prose_beginning_with_abstract_is_not_a_heading() -> None:
    raw = "Abstract reasoning is a distinct cognitive ability in humans."
    body, removed = strip_front_matter(raw)
    assert removed == 0
    assert body == raw


def test_abstract_heading_outside_the_prefix_window_is_ignored() -> None:
    raw = ("filler prose. " * 400) + "\nAbstract\nThe real abstract.\n"
    body, removed = strip_front_matter(raw)
    assert removed == 0
    assert body == raw


# --------------------------------------------------------------------------- #
# Whole chain
# --------------------------------------------------------------------------- #
def test_sanitize_chain_end_to_end() -> None:
    raw = (
        "Jane Doe1 John Smith2\n"
        "1Some University, Somewhere\n"
        "Abstract\n"
        f"{FILLER}\n"
        "As shown in prior work [12] and by Smith et al. [3, 4], the method "
        "(Jones, 2019) is sound. See https://example.org/x for details.\n"
        "Acknowledgments\nWe thank the reviewers.\n"
        "References\n"
        f"{REFERENCE_LIST}\n"
    )
    result = sanitize_document_text(raw, min_body_words=1000)
    assert isinstance(result, SanitationResult)
    text = result.text
    assert "Some University" not in text, "front matter removed"
    assert "A. Author" not in text, "references removed"
    assert "We thank the reviewers" not in text, "acknowledgments removed"
    assert "[12]" not in text
    assert "https://example.org/x" not in text
    assert "the method" in text, "body prose preserved"
    assert result.removed_references > 0
    assert result.removed_acknowledgments > 0
    assert result.removed_citations >= 3
    assert result.removed_front_matter > 0


def test_sanitize_returns_text_that_passes_the_quality_gate() -> None:
    raw = f"Abstract\n{FILLER}\nReferences\n{REFERENCE_LIST}\n"
    text = sanitize_document_text(raw, min_body_words=1000).text
    assert assess_text_quality(text).ok


def test_sanitize_of_empty_text_is_empty() -> None:
    result = sanitize_document_text("")
    assert result.text == ""
    assert result.removed_total == 0


def test_sanitize_does_not_leave_surviving_locators() -> None:
    raw = f"{FILLER}\nSee https://a.org/1 and doi:10.1/abc and arXiv:2101.00001v1.\n"
    text = sanitize_document_text(raw, min_body_words=1000).text
    assert count_bibliographic_remnants(text) == 0


def test_count_bibliographic_remnants_detects_leftovers() -> None:
    assert count_bibliographic_remnants("see https://a.org/1") == 1
    assert count_bibliographic_remnants("clean prose only") == 0


def test_normalize_joins_hyphenated_line_breaks() -> None:
    assert normalize_extracted_text("estima-\ntion") == "estimation"


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
def test_quality_gate_rejects_short_text() -> None:
    quality = assess_text_quality("too short")
    assert not quality.ok
    assert any("words" in reason for reason in quality.reasons)


def test_quality_gate_accepts_greek_math() -> None:
    """Greek letters are letters: math-dense text is not symbol soup."""
    text = "the state estimate α and β parameters satisfy the bound γ " * 250
    assert assess_text_quality(text).ok


def test_quality_gate_rejects_symbol_soup() -> None:
    text = "!@#$%^&*(){}[]|\\:;\"'<>,.?/~` " * 100
    quality = assess_text_quality(text)
    assert not quality.ok
    assert any("alphanumeric" in reason for reason in quality.reasons)


def test_quality_gate_config_round_trip() -> None:
    gate = QualityGate.from_config(
        {
            "min_words": 500,
            "mean_word_length_range": [3.0, 8.0],
            "min_alnum_ratio": 0.7,
        }
    )
    assert gate.min_words == 500
    assert gate.word_length_range == (3.0, 8.0)
    assert gate.min_alnum_ratio == 0.7
    assert gate.as_dict()["mean_word_length_range"] == [3.0, 8.0]


def test_quality_gate_rejects_non_numeric_config() -> None:
    with pytest.raises(SystemExit):
        QualityGate.from_config({"min_words": "one thousand"})
