"""Tests for Seed-First Expansion: seed metadata, windows, and unification.

These cover the rules that are easy to break silently:

* a seed's year decides the harvest window, so a wrong year silently moves it;
* a seed's identifier becomes its corpus id, so a wrong one breaks links;
* the harvest must not re-fetch a seed that is itself on arXiv.
"""

from __future__ import annotations

import pytest

from src.harvest import (
    QualityGate,
    _seed_filename_slug,
    arxiv_base_id,
    date_clause,
    harvest_start_year,
    infer_seed_year,
    normalize_title,
    normalize_title_key,
    parse_seed_metadata,
    seed_abstract,
)


# --------------------------------------------------------------------------- #
# Year inference
# --------------------------------------------------------------------------- #
def test_arxiv_id_yymm_decides_the_year() -> None:
    """``2012.06932`` is December 2020, not the year 2012."""
    year, source = infer_seed_year("2012.06932v1", "no years here")
    assert (year, source) == (2020, "arxiv-id")


def test_arxiv_stamp_in_front_matter() -> None:
    year, source = infer_seed_year(
        "cassie biped",
        "body text",
        "cassie.pdf",
        front_matter="arXiv:1904.11104v1  [cs.RO]  25 Apr 2019",
    )
    assert (year, source) == (2019, "arxiv-stamp")


def test_front_matter_year_beats_body_year() -> None:
    """A body year is usually a citation; the front matter is the paper's own."""
    year, source = infer_seed_year(
        "paper",
        "Published as a conference paper at ICLR 2025. We compare 2017 work.",
        front_matter="Published as a conference paper at ICLR 2025",
    )
    assert (year, source) == (2025, "front-matter")


def test_body_year_is_the_last_resort() -> None:
    year, source = infer_seed_year(
        "unknown-name",
        "Our method improves on 2019 and 2020 baselines repeatedly.",
        "mystery.pdf",
    )
    assert source == "body"
    assert year is not None


def test_no_year_returns_none_not_a_guess() -> None:
    """Returning None keeps the seed out of min(), instead of faking a year."""
    year, source = infer_seed_year("no-id", "text without any year token")
    assert (year, source) == (None, "unknown")


def test_implausible_years_are_ignored() -> None:
    year, _ = infer_seed_year("x", "in 1899 and 2077 nothing happened")
    assert year is None


# --------------------------------------------------------------------------- #
# Harvest window
# --------------------------------------------------------------------------- #
def _config(offset: float = 2, lookback: int = 10) -> dict[str, object]:
    return {"corpus": {"seed_year_offset": offset, "lookback_years": lookback}}


def test_window_offsets_from_the_oldest_seed() -> None:
    start = harvest_start_year([2020, 2024], _config(), current_year=2026)
    assert start == 2018


def test_window_is_floored_by_the_lookback() -> None:
    """One old seed must not drag the corpus a decade backwards."""
    start = harvest_start_year([2001, 2024], _config(), current_year=2026)
    assert start == 2016


def test_window_uses_the_floor_when_no_seed_year_is_known() -> None:
    start = harvest_start_year([None], _config(), current_year=2026)
    assert start == 2016


def test_date_clause_spans_to_the_end_of_the_year() -> None:
    clause = date_clause(2016, end_year=2026)
    assert clause == "submittedDate:[201601010000 TO 202612312359]"


# --------------------------------------------------------------------------- #
# Title matching and identifiers
# --------------------------------------------------------------------------- #
def test_title_key_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_title_key("TD-JEPA: Latent Predictive!") == (
        normalize_title_key("td jepa latent predictive")
    )


def test_title_key_keeps_accented_letters_distinct_from_missing_ones() -> None:
    assert normalize_title_key("Café Robot") == normalize_title_key("Cafe Robot")


def test_filename_slug_is_stable_and_path_safe() -> None:
    assert _seed_filename_slug("TD JEPA.pdf") == "seed:td-jepa"
    assert _seed_filename_slug("motor DR is more important.pdf") == (
        "seed:motor-dr-is-more-important"
    )


# --------------------------------------------------------------------------- #
# Front-matter parsing
# --------------------------------------------------------------------------- #
def test_single_name_per_line_template() -> None:
    front = (
        "Learning to Walk in Minutes\n"
        "\n"
        "Nikita Rudin\n"
        "ETH Zurich and NVIDIA\n"
        "rudinn@ethz.ch\n"
        "\n"
        "David Hoeller\n"
        "ETH Zurich and NVIDIA\n"
    )
    title, authors = parse_seed_metadata(front)
    assert title == "Learning to Walk in Minutes"
    # A lab name must never become an author.
    assert authors == ["Nikita Rudin", "David Hoeller"]


def test_name_affiliation_country_template() -> None:
    front = (
        "HIL: Hybrid Imitation Learning\n"
        "\n"
        "JIASHUN WANG, Carnegie Mellon University, USA\n"
        "YIFENG JIANG, Carnegie Mellon University, USA\n"
        "\n"
        "Carnegie Mellon University\n"
    )
    title, authors = parse_seed_metadata(front)
    assert title == "HIL: Hybrid Imitation Learning"
    assert authors == ["JIASHUN WANG", "YIFENG JIANG"]


def test_venue_stamp_is_not_a_title() -> None:
    front = (
        "Published as a conference paper at ICLR 2025\n"
        "\n"
        "ZERO-SHOT WHOLE-BODY CONTROL\n"
        "\n"
        "Andrea Tirinzoni\n"
    )
    title, authors = parse_seed_metadata(front)
    assert title == "ZERO-SHOT WHOLE-BODY CONTROL"
    assert authors == ["Andrea Tirinzoni"]


def test_body_prose_is_never_read_as_authors() -> None:
    front = (
        "A Short Paper Title Here\n"
        "\n"
        "Jane Doe\n"
        "\n"
        "Abstract\n"
        "We present a method that does things with robots in a very long "
        "sentence that continues for a while so it cannot be a name at all.\n"
    )
    _, authors = parse_seed_metadata(front)
    assert authors == ["Jane Doe"]


def test_empty_front_matter_is_survivable() -> None:
    assert parse_seed_metadata("") == ("", [])


# --------------------------------------------------------------------------- #
# Seed abstract
# --------------------------------------------------------------------------- #
def test_abstract_drops_a_leading_abstract_word() -> None:
    text = "Abstract This paper studies legged locomotion."
    assert seed_abstract(text).startswith("This paper studies")


def test_abstract_stops_at_the_introduction() -> None:
    filler = "We study gaits thoroughly on rough terrain with a learned policy. "
    # The cut is line-anchored, which is what keeps an in-sentence mention from
    # truncating a real abstract.
    text = filler * 3 + "\nIntroduction\nLegged robots are hard."
    abstract = seed_abstract(text)
    assert "Introduction" not in abstract
    assert "gaits" in abstract


def test_abstract_keeps_a_short_mention_of_introduction() -> None:
    """The cut is guarded past char 120 so an early mention cannot truncate."""
    abstract = seed_abstract("Introduction is discussed. Then we build a policy.")
    assert "Then we build a policy" in abstract


def test_abstract_is_word_bounded() -> None:
    abstract = seed_abstract(" ".join(f"w{index}" for index in range(400)))
    assert len(abstract.split()) == 200


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
def test_gate_rejects_a_short_document() -> None:
    assert not QualityGate().assess("too short to be a paper").ok


def test_gate_accepts_equation_dense_text() -> None:
    """Greek letters are letters; they must not read as symbol soup."""
    body = " ".join(["the model α β γ δ estimates θ φ"] * 400)
    assert QualityGate().assess(body).ok


def test_gate_rejects_symbol_soup() -> None:
    assert not QualityGate().assess("!@#$%^&*()" * 400).ok


@pytest.mark.parametrize("value", ["not-a-number", [1], {"a": 1}])
def test_non_numeric_gate_config_raises(value: object) -> None:
    with pytest.raises(SystemExit):
        QualityGate.from_config({"min_words": value})


def test_unset_gate_config_falls_back_to_the_default() -> None:
    """A blank YAML key is "not configured", not a malformed number."""
    assert QualityGate.from_config({"min_words": None}).min_words == 1000


def test_lookback_floor_wins_over_a_centuries_old_seed() -> None:
    """The floor is a hard cap, so it must beat the seed offset."""
    start = harvest_start_year([2020], _config(), current_year=2030)
    assert start == 2020


def test_current_year_hook_keeps_the_suite_time_independent() -> None:
    assert harvest_start_year([2020], _config(), current_year=2026) == 2018


# --------------------------------------------------------------------------- #
# Seed/version id matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("2511.04131", "2511.04131"),
        ("2511.04131v2", "2511.04131"),
        ("2103.07945v3", "2103.07945"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        ("", ""),
    ],
)
def test_arxiv_base_id_strips_only_the_version(given: str, expected: str) -> None:
    assert arxiv_base_id(given) == expected


def test_unversioned_seed_matches_its_versioned_harvest() -> None:
    """A seed named ``2511.04131.pdf`` must block ``2511.04131v2`` from the API.

    The PDF filename usually carries no version while the API always reports
    one, so comparing raw strings let the seed back in as a harvested paper and
    the corpus verify reported a duplicate title inside a single regime.
    """
    seen = {arxiv_base_id("2511.04131")}
    assert arxiv_base_id("2511.04131v2") in seen


def test_hyphen_wrapped_seed_title_matches_the_api_title() -> None:
    """PDF front matter hyphenates at line breaks; the dedupe key must not care."""
    from_pdf = (
        "BFM-ZERO: A PROMPTABLE BEHAVIORAL FOUNDA- TION MODEL FOR HUMANOID "
        "CONTROL USING UN- SUPERVISED REINFORCEMENT LEARNING"
    )
    from_api = (
        "BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid "
        "Control Using Unsupervised Reinforcement Learning"
    )
    assert normalize_title(from_pdf) == normalize_title(from_api)
