"""Tests for the Appendix A.2 post-generation audit.

Each test pins one of the eight checks in ``src/audit.py``. The audit is a
heuristic layer, so the tests assert the *direction* it must move: a compliant
answer passes, and a specific violation is reported with the right item name.

Run with ``pytest tests/ -q``.
"""

from __future__ import annotations

import json

import pytest

from src.audit import (
    TASK_A_RANGE,
    TASK_B_WORDS,
    Finding,
    audit_response,
    check_coverage,
    check_evidence_trace,
    check_length,
    check_lens_discipline,
    check_low_frequency_provenance,
    check_novelty_wording,
    check_uncertainty_disclosure,
    check_unsupported_claims,
    count_words,
    split_response,
    supplied_terms,
)

# --------------------------------------------------------------------------- #
# A small evidence package shaped exactly like the ones build_evidence_package
# emits, so the checks are exercised without loading the TF-IDF matrix.
# --------------------------------------------------------------------------- #


def make_package() -> dict[str, object]:
    """A minimal but realistic evidence package."""
    return {
        "theme_lens_L2": [
            {
                "term": "identification",
                "df": 88,
                "word_forms": ["identification", "identified"],
            },
            {"term": "learning", "df": 104, "word_forms": ["learning", "learned"]},
            {
                "term": "control",
                "df": 112,
                "word_forms": ["control", "controls", "controller"],
            },
        ],
        "concentration_lens_L1": [
            {
                "term": "scattering",
                "df": 2,
                "source_docs": [{"doc": "doc 12", "weight": 0.3}],
            },
            {
                "term": "armax",
                "df": 1,
                "source_docs": [{"doc": "doc 89", "weight": 0.2}],
            },
            {"term": "identifiability", "df": 9, "source_docs": []},
        ],
        "effective_contributing_documents": 17.63,
        "relative_uncertainty_u": 0.0032,
    }


def compliant_response() -> str:
    """A response that satisfies all eight A.2 items."""
    opening = (
        "This virtual direction links identification and learning inside a control "
        "setting. The corpus associates identification with this region, learning "
        "with this region, and control with this region, all in a hedged and "
        "non-causal way. A candidate direction may involve scattering [doc 12] "
        "and armax [doc 89] as limited-source hypotheses, and may involve "
        "identifiability as a broader but still unproven theme. The proposal is a "
        "candidate hypothesis that may require further validation across the "
        "analyzed corpus before any technical claim is made."
    )
    # Pad Task A into the permitted window with content-free but honest prose.
    cushion = (
        "No mechanism is asserted here because the supplied evidence supports none. "
        "Caveat: relative uncertainty is 0.0032 with 17.63 effective contributing "
        "documents, indicating broad support but not direct evidence for a real paper "
        "at this map location. "
    )
    task_a = opening
    while count_words(task_a) < 205:
        task_a = task_a + " " + cushion

    task_b = (
        "Identification and learning meet under control and the corpus supports "
        "broad themes more strongly than any single mechanism so the proposed "
        "direction may require validation outside the analyzed set of documents "
        "and it remains to be established under a hedge here today and so on"
    )

    used = ["identification 88", "learning 104", "control 112"]
    used += ["scattering 2", "armax 1", "identifiability 9"]
    return (
        f"Task A\n{task_a}\n\nTask B\n{task_b}\n\nEvidence used: {', '.join(used)}.\n"
    )


# --------------------------------------------------------------------------- #
# Splitting and counting
# --------------------------------------------------------------------------- #


def test_split_response_separates_the_three_sections() -> None:
    parts = split_response(compliant_response())
    assert parts["task_a"].startswith("This virtual direction")
    assert parts["task_b"]
    assert "identification" in parts["evidence_used"]


def test_split_response_tolerates_missing_sections() -> None:
    parts = split_response("nothing useful here")
    assert parts["task_a"] == ""
    assert parts["task_b"] == ""
    assert parts["evidence_used"] == ""


def test_count_words_ignores_pure_punctuation_tokens() -> None:
    # "--" and "—" carry no alphanumeric content, so they are not words.
    assert count_words("one two -- three — four") == 4


def test_supplied_terms_merges_both_lenses_and_keeps_df() -> None:
    terms = supplied_terms(make_package())
    assert set(terms) == {
        "identification",
        "learning",
        "control",
        "scattering",
        "armax",
        "identifiability",
    }
    assert terms["scattering"]["df"] == 2
    assert terms["identification"]["lens"] == "L2"


# --------------------------------------------------------------------------- #
# Item 1: length compliance
# --------------------------------------------------------------------------- #


def test_length_compliance_accepts_a_compliant_response() -> None:
    finding = check_length(split_response(compliant_response()))
    assert finding.passed, finding.detail
    assert finding.item == "length_compliance"


def test_length_compliance_rejects_task_a_below_the_window() -> None:
    parts = split_response(
        f"Task A\n{'word ' * 50}\n\nTask B\n{'word ' * TASK_B_WORDS}\n"
    )
    finding = check_length(parts)
    assert not finding.passed
    low, high = TASK_A_RANGE
    assert f"outside {low}-{high}" in finding.detail


def test_length_compliance_rejects_task_a_above_the_window() -> None:
    parts = split_response(
        f"Task A\n{'word ' * 300}\n\nTask B\n{'word ' * TASK_B_WORDS}\n"
    )
    assert not check_length(parts).passed


def test_length_compliance_rejects_a_task_b_that_is_not_exactly_45() -> None:
    parts = split_response(f"Task A\n{'word ' * 220}\n\nTask B\n{'word ' * 44}\n")
    finding = check_length(parts)
    assert not finding.passed
    assert "expected exactly 45" in finding.detail


def test_length_compliance_reports_a_missing_task_section() -> None:
    assert not check_length(split_response("Task A\nonly the first one")).passed
    assert not check_length(split_response("Task B\nonly the second one")).passed


# --------------------------------------------------------------------------- #
# Item 2: evidence coverage
# --------------------------------------------------------------------------- #


def test_coverage_passes_when_every_term_appears() -> None:
    finding = check_coverage(split_response(compliant_response()), make_package())
    assert finding.passed, finding.detail


def test_coverage_reports_terms_that_never_appear() -> None:
    response = (
        f"Task A\n{' '.join(['identification'] + ['filler'] * 240)}\n\n"
        f"Task B\n{'filler ' * TASK_B_WORDS}\n"
    )
    finding = check_coverage(split_response(response), make_package())
    assert not finding.passed
    assert "learning" in finding.evidence
    assert "scattering" in finding.evidence


def test_coverage_accepts_an_explicitly_set_aside_term() -> None:
    package = make_package()
    paragraph = (
        "Identification and learning and control reappear throughout. "
        "Scattering appears in too few documents so it is set aside here. "
        "Armax is also set aside. Identifiability is used directly."
    )
    response = f"Task A\n{paragraph}\n\nTask B\n{paragraph}\n"
    finding = check_coverage(split_response(response), package)
    assert finding.passed, finding.detail


def test_coverage_fails_on_an_empty_lens() -> None:
    finding = check_coverage({"task_a": "x", "task_b": "y", "evidence_used": ""}, {})
    assert not finding.passed


def test_coverage_matches_attested_word_forms_not_just_the_lemma() -> None:
    package = {
        "theme_lens_L2": [
            {
                "term": "identification",
                "df": 88,
                "word_forms": ["identification", "identified"],
            }
        ],
        "concentration_lens_L1": [],
    }
    # "identified" is an attested form, so it counts as coverage of the family.
    finding = check_coverage(
        split_response("Task A\nwe identified the model\n\nTask B\nidentified"),
        package,
    )
    assert finding.passed, finding.detail


# --------------------------------------------------------------------------- #
# Item 3: lens discipline
# --------------------------------------------------------------------------- #


def test_lens_discipline_passes_when_an_l2_term_opens_the_text() -> None:
    finding = check_lens_discipline(
        split_response(compliant_response()), make_package()
    )
    assert finding.passed, finding.detail


def test_lens_discipline_fails_when_no_l2_term_appears_early() -> None:
    response = (
        "Task A\n"
        + " ".join(["neutral"] * 240)
        + "\n\nTask B\n"
        + " ".join(["neutral"] * TASK_B_WORDS)
    )
    finding = check_lens_discipline(split_response(response), make_package())
    assert not finding.passed
    assert "no L2 term" in finding.detail


def test_lens_discipline_flags_a_concentration_term_carrying_the_topic() -> None:
    # "scattering" is L1-only, so leading with it inverts the lens roles.
    response = (
        "Task A\nScattering is the topic here, and identification follows later. "
        + " ".join(["neutral"] * 240)
        + "\n\nTask B\n"
        + " ".join(["neutral"] * TASK_B_WORDS)
    )
    finding = check_lens_discipline(split_response(response), make_package())
    assert not finding.passed
    assert "L1-only" in finding.detail


# --------------------------------------------------------------------------- #
# Item 4: low-frequency provenance
# --------------------------------------------------------------------------- #


def test_low_frequency_provenance_passes_when_each_term_is_cited() -> None:
    finding = check_low_frequency_provenance(
        split_response(compliant_response()), make_package()
    )
    assert finding.passed, finding.detail


def test_low_frequency_provenance_fails_without_a_doc_citation() -> None:
    response = (
        "Task A\nIdentification and learning lead, and scattering is mentioned with "
        "no citation at all in this sentence. Armax [doc 89] is cited properly. "
        + " ".join(["neutral"] * 240)
        + "\n\nTask B\n"
        + " ".join(["neutral"] * TASK_B_WORDS)
    )
    finding = check_low_frequency_provenance(split_response(response), make_package())
    assert not finding.passed
    assert "scattering" in finding.evidence


def test_low_frequency_provenance_ignores_high_df_terms() -> None:
    # "identifiability" has df 9, above the df <= 3 bar, so it needs no [doc N].
    package = make_package()
    response = (
        "Task A\nIdentification and identifiability are discussed together. "
        + " ".join(["neutral"] * 240)
        + "\n\nTask B\n"
        + " ".join(["neutral"] * TASK_B_WORDS)
    )
    finding = check_low_frequency_provenance(split_response(response), package)
    assert "identifiability" not in finding.evidence


# --------------------------------------------------------------------------- #
# Item 5: unsupported-claim audit
# --------------------------------------------------------------------------- #


def test_unsupported_claims_pass_on_a_fully_hedged_answer() -> None:
    assert check_unsupported_claims(split_response(compliant_response())).passed


@pytest.mark.parametrize(
    "sentence",
    [
        "The proposed method improves tracking accuracy by 40%.",
        "The controller reduces settling time to 0.12 s.",
        "The approach outperforms every baseline on all tasks.",
        "Carbon fibre laminates guarantee the stiffness bound.",
        "Prior work [12] established this exact mechanism.",
        "Smith et al. showed the same result.",
    ],
)
def test_unsupported_claims_flag_unhedged_mechanisms_numbers_and_citations(
    sentence: str,
) -> None:
    finding = check_unsupported_claims(
        {"task_a": sentence, "task_b": "", "evidence_used": ""}
    )
    assert not finding.passed, sentence
    assert finding.evidence


def test_unsupported_claims_accepts_a_hedged_version_of_the_same_sentence() -> None:
    hedged = "The proposed method may improve tracking accuracy, but this is not established."
    finding = check_unsupported_claims(
        {"task_a": hedged, "task_b": "", "evidence_used": ""}
    )
    assert finding.passed, finding.detail


def test_unsupported_claims_treats_the_required_caveat_as_a_hedge() -> None:
    # A.2 requires this sentence, and it necessarily contains numbers.
    caveat = (
        "Caveat: relative uncertainty is 0.0032 with 17.63 effective contributing "
        "documents, indicating broad support but not direct evidence for a real paper "
        "at this map location."
    )
    finding = check_unsupported_claims(
        {"task_a": caveat, "task_b": "", "evidence_used": ""}
    )
    assert finding.passed, finding.detail


# --------------------------------------------------------------------------- #
# Item 6: novelty wording
# --------------------------------------------------------------------------- #


def test_novelty_wording_passes_on_a_corpus_relative_claim() -> None:
    finding = check_novelty_wording(
        {"task_a": "No prior work in this corpus covers the combination.", "task_b": ""}
    )
    assert finding.passed, finding.detail


def test_novelty_wording_fails_on_an_unqualified_global_claim() -> None:
    finding = check_novelty_wording(
        {"task_a": "This is the first time anyone has combined the two.", "task_b": ""}
    )
    assert not finding.passed
    assert "first time" in finding.detail


def test_novelty_wording_passes_when_a_global_phrase_is_scoped_to_the_corpus() -> None:
    finding = check_novelty_wording(
        {
            "task_a": "The combination is unprecedented within this corpus.",
            "task_b": "",
        }
    )
    assert finding.passed, finding.detail


# --------------------------------------------------------------------------- #
# Item 7: uncertainty disclosure
# --------------------------------------------------------------------------- #


def test_uncertainty_disclosure_passes_when_both_numbers_are_stated() -> None:
    assert check_uncertainty_disclosure(split_response(compliant_response())).passed


def test_uncertainty_disclosure_fails_when_the_caveat_is_missing() -> None:
    finding = check_uncertainty_disclosure(
        {"task_a": "Identification and learning lead the discussion.", "task_b": ""}
    )
    assert not finding.passed
    assert "relative uncertainty" in finding.detail


def test_uncertainty_disclosure_fails_when_effective_support_is_missing() -> None:
    finding = check_uncertainty_disclosure(
        {
            "task_a": "Caveat: relative uncertainty is 0.0032 for this location.",
            "task_b": "",
        }
    )
    assert not finding.passed
    assert "effective contributing documents" in finding.detail


# --------------------------------------------------------------------------- #
# Item 8: evidence trace
# --------------------------------------------------------------------------- #


def test_evidence_trace_passes_when_all_used_terms_are_recorded() -> None:
    finding = check_evidence_trace(split_response(compliant_response()), make_package())
    assert finding.passed, finding.detail


def test_evidence_trace_fails_without_a_list() -> None:
    finding = check_evidence_trace(
        {"task_a": "identification", "task_b": "learning", "evidence_used": ""},
        make_package(),
    )
    assert not finding.passed
    assert "Evidence used" in finding.detail


def test_evidence_trace_fails_when_a_used_term_is_omitted() -> None:
    response = (
        "Task A\nIdentification and learning and control. Scattering [doc 12]. "
        "Armax [doc 89]. Identifiability.\n\n"
        "Task B\nidentification learning\n\n"
        "Evidence used: identification 88, learning 104, control 112.\n"
    )
    finding = check_evidence_trace(split_response(response), make_package())
    assert not finding.passed
    assert "scattering" in finding.evidence


def test_evidence_trace_fails_when_document_frequencies_are_absent() -> None:
    response = (
        "Task A\nIdentification and learning and control.\n\n"
        "Task B\nidentification learning\n\n"
        "Evidence used: identification, learning, control.\n"
    )
    finding = check_evidence_trace(split_response(response), make_package())
    assert not finding.passed
    assert "document frequencies" in finding.detail


# --------------------------------------------------------------------------- #
# Whole-report behaviour
# --------------------------------------------------------------------------- #


def test_audit_report_passes_end_to_end_on_a_compliant_response() -> None:
    report = audit_response(compliant_response(), make_package())
    failed = [f["item"] for f in report["findings"] if not f["passed"]]
    assert report["passed"], f"unexpected failures: {failed}"
    assert report["items_checked"] == 8
    assert report["items_failed"] == 0
    assert report["word_counts"]["task_b"] == TASK_B_WORDS


def test_audit_report_is_json_serialisable() -> None:
    report = audit_response(compliant_response(), make_package())
    assert json.loads(json.dumps(report))["passed"] is True


def test_audit_report_reports_every_failing_item() -> None:
    report = audit_response("Task A\ntoo short\n\nTask B\ntoo short\n", make_package())
    assert report["passed"] is False
    assert report["items_failed"] >= 4
    items = {finding["item"] for finding in report["findings"]}
    assert {"length_compliance", "evidence_coverage", "evidence_trace"} <= items


def test_audit_report_carries_the_scope_disclaimer() -> None:
    report = audit_response(compliant_response(), make_package())
    assert "not to guarantee" in report["disclaimer"] or "not" in report["disclaimer"]


def test_finding_serialises_evidence_deterministically() -> None:
    finding = Finding("demo", False, "detail", [f"line {n}" for n in range(40)])
    payload = finding.as_dict()
    assert payload["item"] == "demo"
    assert len(payload["evidence"]) == 20
    assert payload["evidence"][0] == "line 0"
