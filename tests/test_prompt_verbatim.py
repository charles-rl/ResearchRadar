"""Pin the verbalization prompt and the audit checklist to Appendix A of the paper.

The prompt is the artifact most likely to drift silently: it is plain text, it is
shown verbatim on the dashboard, and "close enough" is not the standard the paper
sets. These tests re-derive it from the paper's own Markdown
(``docs/papers/foundation/2610.25662v1.md``) and fail on any difference that is
not one of the enumerated transcription artifacts below.

A.2 gets the same treatment in the other direction: the eight audit items in
``src/audit.py`` are pinned to the eight bullets the paper lists, in order.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.audit import AuditSettings, audit_response

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/papers/foundation/2610.25662v1.md"
TEMPLATE = ROOT / "docs/kmo_prompt_template_A1.txt"
PRISTINE = ROOT / "docs/kmo_prompt_template_A1.original.txt"

#: The paper's own domain. The repository studies robot learning, so the in-use
#: template replaces this one word; ``src.report`` pins the same substitution.
PAPER_DOMAIN = "composites"
REPO_DOMAIN = "robotics"

#: Transcription artifacts of the PDF -> Markdown conversion. The paper file is a
#: derived artifact, not the typeset PDF, and these five are visibly broken there
#: (an unbalanced bold marker, an identifier split across a line, a duplicated
#: bullet, a misplaced bold pair, and an unmatched code fence around
#: "Deliverables"). Each is asserted to be present before it is repaired, so the
#: test fails loudly if the paper file is ever re-extracted.
_EXTRACT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("*evidence package**", "**evidence package**"),
    ("`source docs`", "`source_docs`"),
    ("**Serendipity** from", "**Serendipity from"),
    (
        "- `mixture_entropy_H`, `effective_contributing_documents`, Rules",
        "Rules",
    ),
)

#: The eight A.2 bullets, in the paper's order.
A2_ITEMS: tuple[str, ...] = (
    "Length compliance",
    "Evidence coverage",
    "Lens discipline",
    "Low-frequency provenance",
    "Unsupported-claim audit",
    "Novelty wording",
    "Uncertainty disclosure",
    "Evidence trace",
)

#: The item ids ``src.audit`` reports, in the same order.
AUDIT_ITEM_IDS: tuple[str, ...] = (
    "length_compliance",
    "evidence_coverage",
    "lens_discipline",
    "low_frequency_provenance",
    "unsupported_claim_audit",
    "novelty_wording",
    "uncertainty_disclosure",
    "evidence_trace",
)


def _paper() -> str:
    if not PAPER.is_file():
        pytest.skip(f"{PAPER.name} is not in this checkout")
    return PAPER.read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    """Body of one appendix section, without its heading line."""
    begin = text.index(start)
    body = text[begin : text.index(end, begin)]
    return body.split("\n", 1)[1]


#: The PDF -> Markdown conversion is inconsistent about dashes: the typeset em
#: and en dashes come out as "-" in the list separators and as "\u2013" in
#: "200-280". Folding every dash glyph to "-" removes that noise; no other
#: character is touched, so a dropped clause or a softened threshold still fails.
_DASH_FOLD = str.maketrans(
    {"\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"}
)


def _reflow(text: str) -> str:
    """Collapse the line wrapping the PDF extraction introduced.

    Line breaks inside a paragraph are an artifact of the page width, not of the
    prompt: the template is hard-wrapped at 72 columns to sit in a text file. Only
    whitespace and the dash glyph are normalised; every token is compared.
    """
    return re.sub(r"\s+", " ", text.translate(_DASH_FOLD)).strip()


def _repair(text: str) -> str:
    """Undo the five transcription artifacts, asserting each is really there."""
    repaired = text
    for broken, fixed in _EXTRACT_ARTIFACTS:
        assert broken in repaired, f"paper no longer shows the artifact {broken!r}"
        repaired = repaired.replace(broken, fixed, 1)
    fences = repaired.count("```")
    assert fences == 2, f"expected the two stray code fences, found {fences}"
    return repaired.replace("```", "").replace("  ", " ").strip()


def _a1_body() -> str:
    return _reflow(_section(_paper(), "# A.1 Prompt template", "# A.2 Post-generation audit"))


def _a2_body() -> str:
    return _section(_paper(), "# A.2 Post-generation audit protocol", "# Appendix B.")


# --------------------------------------------------------------------------- #
# A.1 - the prompt template
# --------------------------------------------------------------------------- #
def test_the_paper_section_is_the_one_this_repository_quotes() -> None:
    body = _a1_body()
    for anchor in (
        "You are participating in a controlled experiment.",
        "Rules (binding)",
        "**Close with the evidence list.**",
        "Output Task A, then Task B, then the `Evidence used:` list.",
    ):
        assert anchor in body


def test_the_in_use_template_is_the_paper_text_verbatim() -> None:
    """Same words, same order -- after the documented domain substitution.

    The comparison is token-for-token on reflowed text, so a dropped clause, a
    reordered rule, or a softened threshold (200-280, exactly 45, df >= 4,
    df <= 3) all fail here.
    """
    body = _repair(_a1_body())
    assert PAPER_DOMAIN in body, "the domain substitution has nothing to replace"
    expected = _reflow(body.replace(PAPER_DOMAIN, REPO_DOMAIN))
    assert _reflow(TEMPLATE.read_text(encoding="utf-8")) == expected


def test_the_only_punctuation_the_extraction_normalises_is_the_dash() -> None:
    """Proves the em/en dash fold above hides no other glyph difference."""
    body = _a1_body()
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "\u2014" not in body, "the extraction normalised the em dash to an en dash"
    assert "\u2014" in template, "the template lost the typeset em dash"
    for text in (body, template):
        other = {char for char in text if ord(char) > 126 and char not in "\u2013\u2014"}
        assert not other, f"unexplained non-ASCII characters: {other}"


def test_the_domain_substitution_is_the_only_edit_to_the_pristine_extract() -> None:
    """``.original.txt`` is the untouched extract; the diff must stay one line."""
    if not PRISTINE.is_file():
        pytest.skip("pristine extract is not in this checkout")
    original = PRISTINE.read_text(encoding="utf-8").splitlines()
    edited = TEMPLATE.read_text(encoding="utf-8").splitlines()
    assert len(original) == len(edited)
    changed = [
        (left, right)
        for left, right in zip(original, edited, strict=True)
        if left != right
    ]
    assert len(changed) == 1, changed
    left, right = changed[0]
    assert PAPER_DOMAIN in left and REPO_DOMAIN in right


def test_the_thresholds_the_prompt_states_are_the_audits_thresholds() -> None:
    """A.1's numbers and A.2's checks must not disagree about the same bar."""
    body = _a1_body()
    settings = AuditSettings()
    low, high = settings.task_a_words
    assert str(low) in body and str(high) in body and "200-280" in body.replace("\u2013", "-")
    assert f"{settings.task_b_words}-word" in body
    assert f"df >= {settings.emergent_df}" in body
    assert f"df <= {settings.max_supported_df}" in body


# --------------------------------------------------------------------------- #
# A.2 - the post-generation audit
# --------------------------------------------------------------------------- #
def test_the_paper_lists_the_eight_audit_items_in_order() -> None:
    bullets = [
        line.strip().lstrip("\u2022- ").split(":")[0].strip()
        for line in _a2_body().splitlines()
        if line.strip().startswith(("\u2022", "-"))
    ]
    assert bullets == list(A2_ITEMS)


def test_the_audit_implements_one_check_per_paper_item_in_the_same_order() -> None:
    report = audit_response("Task A\nnothing\nTask B\nnothing\nEvidence used:\n", {})
    assert [finding["item"] for finding in report["findings"]] == list(AUDIT_ITEM_IDS)
    assert report["items_checked"] == len(A2_ITEMS)
    assert report["reference"] == "Obayashi et al. (2026) Appendix A.2"


def test_the_audit_records_the_length_and_df_bars_from_the_paper() -> None:
    text = _a2_body()
    assert "200\u2013280 words" in text
    assert "exactly 45 words" in text
    settings = AuditSettings()
    assert settings.task_a_words == (200, 280)
    assert settings.task_b_words == 45
    assert settings.task_b_tolerance == 0
    # A.2 item 4 and A.1 rule 2 use the same df split.
    assert settings.max_supported_df == 3
    assert settings.emergent_df == 4


def test_the_audit_disclaims_what_the_paper_disclaims() -> None:
    """A.2 says the audit exposes extrapolation, it does not prove innocence."""
    report = audit_response("Task A\nx\nTask B\ny\nEvidence used:\n", {})
    assert "not to guarantee" in report["disclaimer"]
    assert "expose unsupported extrapolation" in _reflow(_a2_body())
