"""Appendix A.2 post-generation audit.

The evidence-constrained prompt makes a language model produce a *virtual*
research abstract from a fixed evidence package. Nothing in the prompt can stop
the model from inventing a mechanism, a material, or a performance number, so
Obayashi et al. (2026) Appendix A.2 defines eight checks to run afterwards. This
module implements them.

Two deliberate limits, both from the paper:

* The audit "is intended to expose unsupported extrapolation rather than to
  claim a mathematical guarantee of hallucination-free generation" (A.2). Every
  check here is a conservative, explainable heuristic, not a proof.
* The audit never rewrites the text. It reports findings so a human can decide.

Usage:

    python -m src.audit --input response.txt --gap 1
    python -m src.audit --input response.txt --gap 1 --json
    python -m src.audit --template-check          # verify the stored template only

``--gap N`` loads the evidence package for the N-th practical Pareto solution
from ``data/pareto_solutions.json`` and the report's evidence packages, so the
coverage and provenance checks compare the response against the terms that were
actually supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common import (
    configure_logging,
    load_config,
    resolve_paths,
)

DEFAULT_CONFIG = Path("config.yaml")

#: A.2 length compliance, verbatim: "Task A contains 200-280 words and Task B
#: contains exactly 45 words."
TASK_A_RANGE = (200, 280)
TASK_B_WORDS = 45
#: Word-count tolerance for Task B. The paper says "exactly", and its own
#: examples hit 45; the tolerance exists only so a trailing punctuation mark
#: cannot fail an otherwise compliant answer.
TASK_B_TOLERANCE = 0

_TASK_A_RE = re.compile(
    r"Task\s*A\b[^\n]*\n(?P<body>.*?)(?=Task\s*B\b|\Z)", re.IGNORECASE | re.DOTALL
)
_TASK_B_RE = re.compile(
    r"Task\s*B\b[^\n]*\n(?P<body>.*?)(?=Evidence\s*used|\Z)", re.IGNORECASE | re.DOTALL
)
_EVIDENCE_USED_RE = re.compile(r"Evidence\s*used\s*:?", re.IGNORECASE)
#: ``[doc 12]`` as Appendix A.1 Rule 2 writes it, but tolerate ``[doc 12, 44]``.
_DOC_CITATION_RE = re.compile(r"\[doc\s+(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
_DOC_NUMBER_RE = re.compile(r"\d+")

#: Phrases that claim novelty beyond the analysed corpus. A.2 requires novelty to
#: stay corpus-relative, so a global claim is a finding even when it is true.
_GLOBAL_NOVELTY = (
    "for the first time",
    "the first time",
    "first ever",
    "world first",
    "never been",
    "no prior work",
    "no existing work",
    "unprecedented",
    "state of the art",
    "state-of-the-art",
    "novel contribution to the field",
    "unlike any",
    "no one has",
)
#: Hedges that make a novelty statement corpus-relative instead of global.
_CORPUS_RELATIVE = (
    "in this corpus",
    "within this corpus",
    "in the analyzed corpus",
    "in the analysed corpus",
    "in the sampled corpus",
    "relative to the analyzed",
    "relative to the analysed",
    "across the analyzed",
    "across the analysed",
)
#: A.2 uncertainty disclosure: both numbers must appear in Task A.
_UNCERTAINTY_WORDS = ("relative uncertainty", "uncertainty", "caveat")
_EFFECTIVE_SUPPORT_WORDS = (
    "effective contributing",
    "effective document",
    "effective support",
    "effective contributing documents",
)
#: Hedges that turn a claim into a hypothesis. Used by the unsupported-claim
#: heuristic: a sentence with a hard number or a causal verb and no hedge is
#: reported for a human to read.
_HEDGES = (
    "may",
    "might",
    "could",
    "would",
    "suggest",
    "suggests",
    "indicate",
    "indicates",
    "appears",
    "appear",
    "likely",
    "possibly",
    "potential",
    "potentially",
    "hypothesis",
    "we hypothesize",
    "is expected",
    "are expected",
    # NOTE: "proposed" is deliberately absent. "The proposed method improves
    # accuracy by 40%" is exactly the claim A.2 asks the audit to catch, and
    # "proposed" describes the work, not the evidential status of the claim.
    "is proposed",
    "are proposed",
    "we propose",
    "candidate",
    "remains to be",
    "requires validation",
    "not established",
    "unvalidated",
    # A.2 item 7 requires an explicit caveat; that sentence carries real numbers
    # by design, so the disclosure wording must count as a hedge or the audit
    # would flag the very sentence the prompt demands.
    "caveat",
    "relative uncertainty is",
)
#: Causal / mechanism verbs that A.2 explicitly forbids asserting without
#: evidence: "technical claims, mechanisms, materials, numerical performance
#: claims, and citations".
_MECHANISM_VERBS = (
    "causes",
    "caused",
    "leads to",
    "results in",
    "improves",
    "improved",
    "increases",
    "increased",
    "decreases",
    "decreased",
    "reduces",
    "reduced",
    "outperforms",
    "outperformed",
    "achieves",
    "achieved",
    "yields",
    "guarantees",
    "ensures",
    "proves",
    "demonstrates that",
)
_MATERIALS = (
    "carbon fibre",
    "carbon fiber",
    "aluminium",
    "aluminum",
    "titanium",
    "steel",
    "composite laminate",
    "epoxy",
    "nylon",
    "pla",
    "abs plastic",
    "silicone",
    "polyurethane",
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%|\b\d+\.\d+\b")


def _config_int(key: str, value: object) -> int:
    """Coerce a config value to int, naming the offending key.

    A bare ``int()`` here raises ``ValueError: invalid literal for int()`` with
    no hint about which key is wrong, so the message carries the key name.
    """
    if isinstance(value, bool):
        raise SystemExit(f"{key} must be an integer, got the boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    raise SystemExit(f"{key} must be an integer, got {value!r}")


@dataclass(frozen=True)
class AuditSettings:
    """A.2 thresholds, resolved from ``config.yaml``.

    ``task_a_words`` and ``task_b_words`` are verbatim from A.2 item 1; the
    ``df`` bars mirror the ``df >= 4`` / ``df <= 3`` split the lenses use.
    """

    task_a_words: tuple[int, int] = TASK_A_RANGE
    task_b_words: int = TASK_B_WORDS
    task_b_tolerance: int = TASK_B_TOLERANCE
    max_supported_df: int = 3
    emergent_df: int = 4

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AuditSettings:
        section = config.get("audit", {}) or {}
        window = section.get("task_a_words", list(TASK_A_RANGE))
        if not isinstance(window, list) or len(window) != 2:
            raise SystemExit(
                f"audit.task_a_words must be a two-element list, got {window!r}"
            )
        return cls(
            task_a_words=(
                _config_int("audit.task_a_words[0]", window[0]),
                _config_int("audit.task_a_words[1]", window[1]),
            ),
            task_b_words=_config_int(
                "audit.task_b_words", section.get("task_b_words", TASK_B_WORDS)
            ),
            task_b_tolerance=_config_int(
                "audit.task_b_tolerance",
                section.get("task_b_tolerance", TASK_B_TOLERANCE),
            ),
            max_supported_df=_config_int(
                "audit.max_supported_df", section.get("max_supported_df", 3)
            ),
            emergent_df=_config_int("audit.emergent_df", section.get("emergent_df", 4)),
        )


@dataclass
class Finding:
    """One audit item's outcome."""

    item: str
    passed: bool
    detail: str
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence[:20],
        }


def count_words(text: str) -> int:
    """Words as a reader counts them: whitespace-separated tokens with content."""
    return len([token for token in text.split() if any(c.isalnum() for c in token)])


def split_response(text: str) -> dict[str, str]:
    """Separate Task A, Task B, and the Evidence used list.

    Missing sections come back as empty strings, which the length checks then
    report rather than crashing.
    """
    task_a = _TASK_A_RE.search(text)
    task_b = _TASK_B_RE.search(text)
    evidence = _EVIDENCE_USED_RE.search(text)
    return {
        "task_a": task_a.group("body").strip() if task_a else "",
        "task_b": task_b.group("body").strip() if task_b else "",
        "evidence_used": text[evidence.end() :].strip() if evidence else "",
    }


def check_length(
    parts: dict[str, str], settings: AuditSettings | None = None
) -> Finding:
    """A.2 item 1: Task A 200-280 words, Task B exactly 45."""
    settings = settings or AuditSettings()
    if not parts["task_a"]:
        return Finding("length_compliance", False, "Task A section not found.")
    if not parts["task_b"]:
        return Finding("length_compliance", False, "Task B section not found.")

    words_a = count_words(parts["task_a"])
    words_b = count_words(parts["task_b"])
    low, high = settings.task_a_words
    problems: list[str] = []
    if not low <= words_a <= high:
        problems.append(f"Task A has {words_a} words, outside {low}-{high}")
    if abs(words_b - settings.task_b_words) > settings.task_b_tolerance:
        problems.append(
            f"Task B has {words_b} words, expected exactly {settings.task_b_words}"
        )

    detail = (
        "; ".join(problems)
        if problems
        else f"Task A {words_a} words (allowed {low}-{high}), Task B {words_b} words"
    )
    return Finding("length_compliance", not problems, detail)


def supplied_terms(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every semantic term the evidence package handed the model."""
    terms: dict[str, dict[str, Any]] = {}
    for entry in package.get("theme_lens_L2", []) or []:
        terms[str(entry.get("term", ""))] = {
            "lens": "L2",
            "df": int(entry.get("df", 0)),
            "forms": list(entry.get("word_forms", []) or []),
        }
    for entry in package.get("concentration_lens_L1", []) or []:
        name = str(entry.get("term", ""))
        record = terms.setdefault(
            name,
            {
                "lens": "L1",
                "df": int(entry.get("df", 0)),
                "forms": list(entry.get("word_forms", []) or []),
            },
        )
        if record["lens"] == "L1":
            record["lens"] = "L1"
    return {name: record for name, record in terms.items() if name}


def _contains_term(text: str, term: str, forms: list[str]) -> bool:
    """Word-boundary match against the term or any of its attested forms."""
    haystack = text.lower()
    candidates = {term.lower(), *(form.lower() for form in forms)}
    return any(
        re.search(rf"(?<![a-z]){re.escape(candidate)}(?![a-z])", haystack)
        for candidate in candidates
        if candidate
    )


def check_coverage(parts: dict[str, str], package: dict[str, Any]) -> Finding:
    """A.2 item 2: every supplied term is used or explicitly set aside."""
    terms = supplied_terms(package)
    if not terms:
        return Finding("evidence_coverage", False, "Evidence package has no terms.")

    body = f"{parts['task_a']}\n{parts['task_b']}"
    unused = [
        name
        for name, record in terms.items()
        if not _contains_term(body, name, record["forms"])
    ]
    if not unused:
        return Finding(
            "evidence_coverage",
            True,
            f"All {len(terms)} supplied terms appear in Task A or Task B.",
        )

    # "Explicitly identified as set aside" is the permitted alternative.
    set_aside = [
        name
        for name in unused
        if re.search(
            rf"set aside[^.]*\b{re.escape(name)}\b|\b{re.escape(name)}\b[^.]*set aside",
            body,
            re.IGNORECASE,
        )
    ]
    silent = [name for name in unused if name not in set_aside]
    if not silent:
        return Finding(
            "evidence_coverage",
            True,
            f"{len(set_aside)} term(s) were explicitly set aside.",
            set_aside,
        )
    return Finding(
        "evidence_coverage",
        False,
        f"{len(silent)} of {len(terms)} supplied terms appear neither in the text "
        "nor in a 'set aside' statement.",
        silent,
    )


def check_lens_discipline(parts: dict[str, str], package: dict[str, Any]) -> Finding:
    """A.2 item 3: L2 defines the topic, L1 is supporting evidence.

    Operationalised the only way a checker can: the topic sentence (the opening
    of Task A) must lean on L2 terms, and an L1-only term must not be the first
    supplied term the abstract names.
    """
    theme = {str(e.get("term", "")) for e in package.get("theme_lens_L2", []) or []}
    concentration = {
        str(e.get("term", "")) for e in package.get("concentration_lens_L1", []) or []
    }
    l1_only = concentration - theme
    if not theme:
        return Finding("lens_discipline", False, "Evidence package has no L2 terms.")

    opening = " ".join(parts["task_a"].split()[:60])
    theme_hits = [name for name in theme if _contains_term(opening, name, [])]
    if not theme_hits:
        return Finding(
            "lens_discipline",
            False,
            "The opening of Task A names no L2 term, so the principal topic is not "
            "anchored in the thematic lens.",
        )

    # The first supplied term mentioned must not be an L1-only term.
    body = parts["task_a"].lower()
    positions = [
        (body.find(name.lower()), "L1", name) for name in l1_only if name and name.lower() in body
    ]
    positions += [
        (body.find(name.lower()), "L2", name) for name in theme if name and name.lower() in body
    ]
    positions = [entry for entry in positions if entry[0] >= 0]
    positions.sort()
    if positions and positions[0][1] == "L1":
        return Finding(
            "lens_discipline",
            False,
            f"The first evidence term named is L1-only ({positions[0][2]!r}), which "
            "makes a concentration term carry the principal topic.",
            [positions[0][2]],
        )
    return Finding(
        "lens_discipline",
        True,
        f"Opening names {len(theme_hits)} L2 term(s); first evidence term is "
        f"{positions[0][1] if positions else 'none'}.",
        theme_hits,
    )


def check_low_frequency_provenance(
    parts: dict[str, str],
    package: dict[str, Any],
    settings: AuditSettings | None = None,
) -> Finding:
    """A.2 item 4: every L1-only term with df <= 3 cites its source document."""
    settings = settings or AuditSettings()
    theme = {str(e.get("term", "")) for e in package.get("theme_lens_L2", []) or []}
    body = f"{parts['task_a']}\n{parts['task_b']}"
    cited: set[int] = set()
    for match in _DOC_CITATION_RE.finditer(body):
        cited.update(int(value) for value in _DOC_NUMBER_RE.findall(match.group(1)))

    offenders: list[str] = []
    checked = 0
    for entry in package.get("concentration_lens_L1", []) or []:
        name = str(entry.get("term", ""))
        if not name or name in theme:
            continue
        if int(entry.get("df", 0)) > settings.max_supported_df:
            continue
        if not _contains_term(body, name, list(entry.get("word_forms", []) or [])):
            continue
        checked += 1
        # The citation must sit in the same sentence as the term.
        sentence = _sentence_containing(body, name)
        if not _DOC_CITATION_RE.search(sentence):
            offenders.append(name)

    if offenders:
        return Finding(
            "low_frequency_provenance",
            False,
            f"{len(offenders)} low-frequency L1-only term(s) appear without a "
            "[doc N] citation in the same sentence.",
            offenders,
        )
    if not checked:
        return Finding(
            "low_frequency_provenance",
            True,
            "No df <= 3 L1-only term was used, so nothing needed provenance.",
        )
    return Finding(
        "low_frequency_provenance",
        True,
        f"All {checked} low-frequency L1-only term(s) carry a [doc N] citation.",
    )


def _sentence_containing(text: str, term: str) -> str:
    """The sentence around the first word-boundary occurrence of ``term``."""
    pattern = re.compile(rf"(?<![a-zA-Z]){re.escape(term)}(?![a-zA-Z])", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start()))
    end = text.find(".", match.end())
    if end < 0:
        end = len(text)
    return text[start + 1 : end + 1]


def check_unsupported_claims(parts: dict[str, str]) -> Finding:
    """A.2 item 5: no unsupported mechanisms, materials, numbers, or citations.

    The honest version of this check is a flagging heuristic, not a verdict. A
    sentence is reported when it asserts a mechanism or a number and carries no
    hedge. The paper's own examples are full of hedged mechanism language, so a
    hedge anywhere in the sentence is accepted.
    """
    flags: list[str] = []
    body = parts["task_a"] or parts["task_b"]
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        lowered = sentence.lower()
        if not lowered.strip():
            continue
        if any(hedge in lowered for hedge in _HEDGES):
            continue
        reasons: list[str] = []
        if any(verb in lowered for verb in _MECHANISM_VERBS):
            reasons.append("unhedged mechanism claim")
        if any(material in lowered for material in _MATERIALS):
            reasons.append("specific material")
        if _NUMBER_RE.search(sentence):
            reasons.append("unhedged quantitative claim")
        if re.search(r"\[\d+\]|\(\d{4}\)", sentence):
            reasons.append("external citation")
        if "et al" in lowered:
            reasons.append("external citation")
        if reasons:
            flags.append(f"{', '.join(sorted(set(reasons)))}: {sentence.strip()[:160]}")

    if flags:
        return Finding(
            "unsupported_claim_audit",
            False,
            f"{len(flags)} sentence(s) assert claims without a hedge; review each.",
            flags,
        )
    return Finding(
        "unsupported_claim_audit",
        True,
        "No unhedged mechanism, material, numerical, or citation claim found.",
    )


def check_novelty_wording(parts: dict[str, str]) -> Finding:
    """A.2 item 6: novelty stays relative to the analysed corpus."""
    body = f"{parts['task_a']}\n{parts['task_b']}".lower()
    offending = [phrase for phrase in _GLOBAL_NOVELTY if phrase in body]
    if not offending:
        return Finding(
            "novelty_wording",
            True,
            "No global-novelty phrasing detected.",
        )
    relative = any(phrase in body for phrase in _CORPUS_RELATIVE)
    if relative:
        return Finding(
            "novelty_wording",
            True,
            "Global-novelty phrases appear but the text also scopes novelty to the "
            "corpus; verify each instance reads as corpus-relative.",
            offending,
        )
    return Finding(
        "novelty_wording",
        False,
        "Global-novelty phrasing without a corpus-relative qualifier: "
        + ", ".join(offending),
        offending,
    )


def check_uncertainty_disclosure(parts: dict[str, str]) -> Finding:
    """A.2 item 7: uncertainty and effective support are stated in Task A."""
    task_a = parts["task_a"].lower()
    has_uncertainty = any(word in task_a for word in _UNCERTAINTY_WORDS)
    has_support = any(word in task_a for word in _EFFECTIVE_SUPPORT_WORDS)
    missing = []
    if not has_uncertainty:
        missing.append("relative uncertainty")
    if not has_support:
        missing.append("effective contributing documents")
    if missing:
        return Finding(
            "uncertainty_disclosure",
            False,
            "Task A does not state: " + ", ".join(missing),
            missing,
        )
    return Finding(
        "uncertainty_disclosure",
        True,
        "Task A states both relative uncertainty and effective support.",
    )


def check_evidence_trace(parts: dict[str, str], package: dict[str, Any]) -> Finding:
    """A.2 item 8: an Evidence used list records terms and their document frequencies."""
    trace = parts["evidence_used"]
    if not trace:
        return Finding(
            "evidence_trace",
            False,
            "The output does not end with an 'Evidence used' list.",
        )
    terms = supplied_terms(package)
    body = f"{parts['task_a']}\n{parts['task_b']}"
    # A.2 asks the list to record "the terms used ... including any term set
    # aside", so compare against the terms the answer actually used rather than
    # against the whole lens.
    used = [
        name
        for name, record in terms.items()
        if _contains_term(body, name, record["forms"])
    ]
    listed = [
        name
        for name in used
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", trace, re.IGNORECASE)
    ]
    with_df = [
        name
        for name in listed
        if re.search(
            rf"{re.escape(name)}[^\n]*?\b{terms[name]['df']}\b", trace, re.IGNORECASE
        )
    ]
    missing = [name for name in used if name not in listed]
    if missing:
        return Finding(
            "evidence_trace",
            False,
            f"{len(missing)} of {len(used)} terms used in the text are absent from "
            "the Evidence used list.",
            missing,
        )
    if not with_df:
        return Finding(
            "evidence_trace",
            False,
            "The Evidence used list names terms but records no document frequencies.",
        )
    return Finding(
        "evidence_trace",
        True,
        f"Evidence used list records all {len(listed)} used terms, "
        f"{len(with_df)} with a document frequency.",
    )


def audit_response(
    text: str, package: dict[str, Any], settings: AuditSettings | None = None
) -> dict[str, Any]:
    """Run all eight A.2 items and return a serialisable report."""
    settings = settings or AuditSettings()
    parts = split_response(text)
    findings = [
        check_length(parts, settings),
        check_coverage(parts, package),
        check_lens_discipline(parts, package),
        check_low_frequency_provenance(parts, package, settings),
        check_unsupported_claims(parts),
        check_novelty_wording(parts),
        check_uncertainty_disclosure(parts),
        check_evidence_trace(parts, package),
    ]
    failed = [finding for finding in findings if not finding.passed]
    return {
        "reference": "Obayashi et al. (2026) Appendix A.2",
        "settings": {
            "task_a_words": list(settings.task_a_words),
            "task_b_words": settings.task_b_words,
            "max_supported_df": settings.max_supported_df,
        },
        "passed": not failed,
        "items_checked": len(findings),
        "items_failed": len(failed),
        "word_counts": {
            "task_a": count_words(parts["task_a"]),
            "task_b": count_words(parts["task_b"]),
        },
        "findings": [finding.as_dict() for finding in findings],
        "disclaimer": (
            "A.2 is intended to expose unsupported extrapolation, not to guarantee "
            "hallucination-free text. Failures are review items for a human."
        ),
    }


def load_evidence_package(rank: int, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the evidence package for one ranked practical gap.

    Packages are regenerated from the same inputs the report used, so this works
    without parsing the Markdown.
    """
    from src.report import build_gap_packages, load_inputs

    config = load_config(config_path)
    paths = resolve_paths(config, config_path)
    corpus, grid_data, front = load_inputs(paths)
    _vocabulary, packages = build_gap_packages(corpus, grid_data, front, config, paths)
    # ``packages`` is keyed by grid_index, not by rank, so resolve the rank first.
    for solution in front["solutions"]:
        if int(solution["rank"]) == rank:
            return packages[int(solution["grid_index"])]
    raise SystemExit(f"No practical gap with rank {rank}; the front has {len(packages)}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Appendix A.2 post-generation audit on a model response."
    )
    parser.add_argument("--input", type=Path, help="Response text to audit.")
    parser.add_argument(
        "--gap", type=int, default=1, help="Ranked gap whose evidence package to use."
    )
    parser.add_argument(
        "--package", type=Path, help="Audit against this evidence JSON instead."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Emit JSON, not prose.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not args.input:
        raise SystemExit(
            "Nothing to audit. Pass --input <response.txt>.\n"
            "Paste a model's answer to output/KMO_PROMPT_TOP1.txt into a file first."
        )
    text = args.input.read_text(encoding="utf-8")

    if args.package:
        try:
            package = json.loads(args.package.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Evidence package is not valid JSON: {exc}") from exc
    else:
        package = load_evidence_package(args.gap, args.config)

    settings = (
        AuditSettings.from_config(load_config(args.config))
        if args.config.is_file()
        else AuditSettings()
    )
    report = audit_response(text, package, settings)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Appendix A.2 audit of {args.input}")
        print(f"Task A {report['word_counts']['task_a']} words, "
              f"Task B {report['word_counts']['task_b']} words")
        print()
        for finding in report["findings"]:
            mark = "PASS" if finding["passed"] else "FAIL"
            print(f"[{mark}] {finding['item']}: {finding['detail']}")
            for line in finding["evidence"][:10]:
                print(f"        - {line}")
        print()
        print("RESULT:", "PASS" if report["passed"] else "FAIL")
        print(report["disclaimer"])

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
