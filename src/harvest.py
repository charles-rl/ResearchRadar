"""Automated arXiv literature harvester for Project ResearchRadar.

Implements Phase 2 of ``PLAN.md``:

* Three conflicting robotics regimes are queried (Classical SysID,
  Sim2Real / Domain Randomization, Deep RL Control).
* Review / survey papers are strictly excluded, following the corpus
  construction protocol of Obayashi et al. (2026), Section 5.1: review
  articles intentionally aggregate broad areas of prior work and therefore
  distort local semantic density and kernel interpolation.
* Abstracts are sanitized (LaTeX math stripped, whitespace normalized) so the
  downstream TF-IDF / embedding pipeline sees clean prose.
* Results are deduplicated across arXiv IDs *and* normalized titles, then
  written as a balanced corpus (40 papers per regime, 120 total) to
  ``data/corpus.json``.

Usage
-----
    python -m src.harvest
    python -m src.harvest --config config.yaml --out data/corpus.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arxiv
import requests
import yaml
from lxml import html as lxml_html

LOGGER = logging.getLogger("research_radar.harvest")

DEFAULT_CONFIG = Path("config.yaml")

# LaTeX environments that carry display math.
#: LaTeX display-math environments stripped by ``clean_academic_text``.
#: ``multline`` is spelled correctly: it is amsmath's multi-line environment
#: (there is no ``multiline`` environment), so it must not be "corrected".
MATH_ENVIRONMENTS = (
    "equation",
    "align",
    "eqnarray",
    "gather",
    "split",
    "multline",
    "displaymath",
)

DEFAULT_REVIEW_KEYWORDS = (
    "review",
    "survey",
    "overview",
    "progress in",
    "recent advances",
    "roadmap",
    "perspectives on",
)

DEFAULT_REVIEW_PATTERNS = (
    "a comprehensive survey",
    "this survey",
    "this review",
    "we survey",
    "we review the",
    "an overview of recent",
    "review of the literature",
    "literature review",
)


# --------------------------------------------------------------------------- #
# Review / survey exclusion  (Obayashi et al. 2026, Section 5.1)
# --------------------------------------------------------------------------- #
def is_review_paper(
    title: str,
    abstract: str,
    exclude_keywords: Iterable[str] | None = None,
    abstract_patterns: Iterable[str] | None = None,
) -> bool:
    """Return ``True`` if the record is a review / survey / overview paper.

    The exclusion is deliberately strict: a single aggregated review injects a
    spurious high-density hub into the knowledge manifold, distorting both
    local semantic density and kernel interpolation (Obayashi et al. 2026,
    Section 5.1).

    Parameters
    ----------
    title:
        Raw paper title.
    abstract:
        Raw (uncleaned) abstract text.
    exclude_keywords:
        Whole-word keywords that disqualify a paper when present in the title.
    abstract_patterns:
        Substrings marking a review/survey when they appear in the opening two
        sentences of the abstract (titles are sometimes neutral).
    """
    keywords = tuple(exclude_keywords or DEFAULT_REVIEW_KEYWORDS)
    patterns = tuple(abstract_patterns or DEFAULT_REVIEW_PATTERNS)

    title_lower = (title or "").lower()
    abstract_lower = (abstract or "").lower()

    # Editorial markers on withdrawn entries.
    if title_lower.startswith(("withdrawn", "[withdrawn]")):
        return True

    # 1) Title-level exclusion (whole-word match, so "overviewed" is safe).
    for keyword in keywords:
        pattern = r"(?<![a-z])" + re.escape(keyword.lower()) + r"(?![a-z])"
        if re.search(pattern, title_lower):
            LOGGER.debug("Excluded (title keyword %r): %s", keyword, title)
            return True

    # 2) Abstract-level exclusion: only the opening two sentences are decisive.
    sentences = re.split(r"(?<=[.!?])\s+", abstract_lower.strip())
    opening = " ".join(sentences[:2])
    for phrase in patterns:
        if phrase in opening:
            LOGGER.debug("Excluded (abstract pattern %r): %s", phrase, title)
            return True

    return False


# --------------------------------------------------------------------------- #
# Text sanitization
# --------------------------------------------------------------------------- #
_MATH_BLOCK_RE = re.compile(
    r"\\begin\{(" + "|".join(MATH_ENVIRONMENTS) + r")\*?\}.*?"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?:\\.|[^$\\])*?\$")
_PAREN_MATH_RE = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_LATEX_ENV_RE = re.compile(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", re.DOTALL)
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*")
_LATEX_ESCAPE_RE = re.compile(r"\\([%&_#${}])")
# LaTeX spacing escapes: \, \; \: \! \  \quad \qquad etc.
_LATEX_SPACING_RE = re.compile(r"\\(?:[,;:!\ ]|quad|qquad|thinspace|enspace|medspace)")
# A dollar sign with no partner is a currency marker (e.g. "less than $700"),
# not math: every paired span has already been replaced above.
_STRAY_DOLLAR_RE = re.compile(r"(?<!\\)\$")
_TRAILING_BACKSLASH_RE = re.compile(r"\\+")
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_MULTISPACE_RE = re.compile(r"\s+")
_MATH_PLACEHOLDER_RE = re.compile(r"(?:\s*\[MATH\])+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?)])")


def clean_academic_text(text: str) -> str:
    """Sanitize an academic abstract.

    The sanitization pipeline:

    1. Drops display-math environments (``equation``/``align``/...),
    2. Drops ``$$...$$``, ``$...$``, ``\\(...\\)`` and ``\\[...\\]`` spans,
       replacing each with a single ``[MATH]`` placeholder,
    3. Drops residual LaTeX commands and their arguments,
    4. Unescapes LaTeX special-character escapes (``\\%`` -> ``%``),
    5. Drops LaTeX spacing escapes (``\\,``, ``\\quad``) and unpaired ``$``
       currency markers,
    6. Repairs line-break hyphenation (``estima-\\ntion`` -> ``estimation``),
    7. Collapses all whitespace runs to single spaces.
    """
    if not text:
        return ""

    # arXiv abstracts occasionally carry a LaTeX preamble comment block.
    text = re.sub(r"^\s*%.*$", "", text, flags=re.MULTILINE)

    text = _MATH_BLOCK_RE.sub(" [MATH] ", text)
    text = _DISPLAY_MATH_RE.sub(" [MATH] ", text)
    text = _PAREN_MATH_RE.sub(" [MATH] ", text)
    text = _INLINE_MATH_RE.sub(" [MATH] ", text)
    text = _LATEX_ENV_RE.sub(" ", text)
    text = _LATEX_SPACING_RE.sub(" ", text)
    text = _LATEX_CMD_RE.sub(" ", text)
    text = _LATEX_ESCAPE_RE.sub(r"\1", text)
    text = _STRAY_DOLLAR_RE.sub("", text)
    text = _TRAILING_BACKSLASH_RE.sub(" ", text)

    # Collapse "word-<newline>word" hyphenation before whitespace flattening.
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    text = _MULTISPACE_RE.sub(" ", text)
    text = _MATH_PLACEHOLDER_RE.sub(" [MATH] ", text)
    # Tidy punctuation spacing introduced by placeholder substitution.
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def normalize_title(title: str) -> str:
    """Build a deduplication key from a title (case/punctuation/space agnostic)."""
    if not title:
        return ""
    folded = unicodedata.normalize("NFKD", title.lower())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped)


# --------------------------------------------------------------------------- #
# Harvesting
# --------------------------------------------------------------------------- #
def build_client(corpus_cfg: dict[str, Any]) -> arxiv.Client:
    """Instantiate the arXiv client with polite paging parameters."""
    return arxiv.Client(
        page_size=int(corpus_cfg.get("page_size", 100)),
        delay_seconds=float(corpus_cfg.get("delay_seconds", 3.0)),
        num_retries=int(corpus_cfg.get("num_retries", 5)),
    )


def paged_results(
    client: arxiv.Client,
    search: arxiv.Search,
    delay_seconds: float = 3.0,
    num_retries: int = 5,
) -> Iterator[arxiv.Result]:
    """Yield results with explicit pagination and retry recovery.

    ``arxiv.Client.results`` already pages lazily, but a transient API failure
    mid-corpus raises out of the generator and would abort the whole regime.
    We rebuild the generator with exponential backoff and track how many results
    were already consumed, so a retry resumes rather than restarts.

    ``UnexpectedEmptyPageError`` is *not* treated as a failure: a query whose
    total hit count is smaller than ``Search.max_results`` (e.g. Regime B with
    ~140 hits and ``max_results=400``) legitimately runs out of pages, and that
    exception is how the arxiv client signals end-of-results.
    """
    consumed = 0
    attempts = 0
    limit = int(search.max_results or 0)
    while consumed < limit:
        try:
            for position, result in enumerate(client.results(search)):
                if position < consumed:
                    continue  # replay of an earlier page after a retry
                consumed = position + 1
                yield result
            return
        except arxiv.UnexpectedEmptyPageError:
            # Query exhausted its result set; nothing further to page through.
            LOGGER.debug(
                "arXiv result set exhausted at #%d (max_results=%d)",
                consumed,
                limit,
            )
            return
        except Exception as exc:  # noqa: BLE001 - the arXiv client raises
            # arbitrary HTTP/parse/timeout errors and every one is retryable.
            attempts += 1
            if attempts > num_retries:
                LOGGER.error(
                    "arXiv paging failed after %d retries (%s): %s",
                    num_retries,
                    type(exc).__name__,
                    exc,
                )
                return
            backoff = delay_seconds * (2**attempts)
            LOGGER.warning(
                "arXiv paging error (%s: %s); retry %d/%d in %.1fs "
                "(resuming at #%d)",
                type(exc).__name__,
                exc,
                attempts,
                num_retries,
                backoff,
                consumed,
            )
            time.sleep(backoff)


def harvest_regime_papers(
    client: arxiv.Client,
    regime_cfg: dict[str, Any],
    exclude_keywords: list[str],
    target_count: int,
    max_query_results: int,
    seen_ids: set[str],
    seen_titles: set[str],
    abstract_patterns: list[str] | None = None,
    min_abstract_words: int = 40,
    delay_seconds: float = 3.0,
    num_retries: int = 5,
) -> list[dict[str, Any]]:
    """Harvest up to ``target_count`` papers for a single regime.

    Papers already present in ``seen_ids`` / ``seen_titles`` are skipped so the
    regimes stay disjoint; new IDs and titles are registered as records are
    accepted.
    """
    collected: list[dict[str, Any]] = []
    scanned = 0
    rejected_review = 0
    rejected_short = 0
    rejected_dupe = 0

    search = arxiv.Search(
        query=regime_cfg["query"],
        max_results=max_query_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    for result in paged_results(client, search, delay_seconds, num_retries):
        if len(collected) >= target_count:
            break
        scanned += 1

        short_id = result.get_short_id()
        raw_title = (result.title or "").strip().replace("\n", " ")
        raw_abstract = result.summary or ""
        title_key = normalize_title(raw_title)

        if short_id in seen_ids or title_key in seen_titles:
            rejected_dupe += 1
            continue

        if is_review_paper(
            raw_title, raw_abstract, exclude_keywords, abstract_patterns
        ):
            rejected_review += 1
            continue

        cleaned_abstract = clean_academic_text(raw_abstract)
        if len(cleaned_abstract.split()) < min_abstract_words:
            rejected_short += 1
            continue

        published = result.published
        updated = getattr(result, "updated", None)
        record: dict[str, Any] = {
            "id": short_id,
            "title": re.sub(r"\s+", " ", raw_title),
            "abstract": cleaned_abstract,
            "authors": [author.name for author in result.authors],
            "year": published.year,
            "regime_id": int(regime_cfg["id"]),
            "regime_label": regime_cfg["label"],
            # --- provenance (not required by the schema, kept for auditability)
            "regime_name": regime_cfg["name"],
            "entry_id": result.entry_id,
            "published": published.date().isoformat(),
            "updated": updated.date().isoformat() if updated else None,
            "primary_category": result.primary_category,
            "categories": list(result.categories),
            "doi": result.doi,
            "journal_ref": result.journal_ref,
            "query": regime_cfg["query"],
        }

        collected.append(record)
        seen_ids.add(short_id)
        seen_titles.add(title_key)

        LOGGER.info(
            "[%s] %2d/%d  %s  (%s)",
            regime_cfg["label"],
            len(collected),
            target_count,
            record["title"][:78],
            short_id,
        )

    LOGGER.info(
        "[%s] done: kept=%d scanned=%d dupes=%d reviews=%d short_abstracts=%d",
        regime_cfg["label"],
        len(collected),
        scanned,
        rejected_dupe,
        rejected_review,
        rejected_short,
    )
    if len(collected) < target_count:
        LOGGER.warning(
            "[%s] SHORTFALL: only %d/%d papers after scanning %d candidates "
            "for query: %s",
            regime_cfg["label"],
            len(collected),
            target_count,
            scanned,
            regime_cfg["query"],
        )
    return collected


def harvest_all(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full three-regime harvest and return the corpus document."""
    corpus_cfg = config.get("corpus", {})
    regimes: list[dict[str, Any]] = config.get("regimes", [])
    if not regimes:
        raise ValueError("config.yaml defines no regimes")

    target_per_regime = int(corpus_cfg.get("target_per_regime", 40))
    max_query_results = int(corpus_cfg.get("max_query_results", 400))
    exclude_keywords = [str(k) for k in corpus_cfg.get("exclude_review_keywords", [])]
    abstract_patterns = [
        str(p) for p in corpus_cfg.get("exclude_review_abstract_patterns", [])
    ]
    min_abstract_words = int(corpus_cfg.get("min_abstract_words", 40))
    delay_seconds = float(corpus_cfg.get("delay_seconds", 3.0))
    num_retries = int(corpus_cfg.get("num_retries", 5))

    client = build_client(corpus_cfg)
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    papers: list[dict[str, Any]] = []

    for regime_cfg in regimes:
        LOGGER.info(
            "Harvesting regime %s (%s): %s",
            regime_cfg["id"],
            regime_cfg["label"],
            regime_cfg["query"],
        )
        papers.extend(
            harvest_regime_papers(
                client=client,
                regime_cfg=regime_cfg,
                exclude_keywords=exclude_keywords,
                target_count=target_per_regime,
                max_query_results=max_query_results,
                seen_ids=seen_ids,
                seen_titles=seen_titles,
                abstract_patterns=abstract_patterns,
                min_abstract_words=min_abstract_words,
                delay_seconds=delay_seconds,
                num_retries=num_retries,
            )
        )

    counts = {str(r["label"]): 0 for r in regimes}
    for paper in papers:
        counts[str(paper["regime_label"])] += 1

    return {
        "metadata": {
            "generated_by": "src.harvest",
            "source": "arXiv API via arxiv python client",
            "target_per_regime": target_per_regime,
            "total_papers": len(papers),
            "regime_counts": counts,
            "review_exclusion_keywords": exclude_keywords,
            "review_exclusion_abstract_patterns": abstract_patterns,
            "queries": {str(r["label"]): r["query"] for r in regimes},
            "protocol_reference": (
                "Obayashi et al. (2026) Knowledge Manifold Optimization, Sec. 5.1"
            ),
        },
        "papers": papers,
    }


# --------------------------------------------------------------------------- #
# Persistence & verification
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = (
    "id",
    "title",
    "abstract",
    "authors",
    "year",
    "regime_id",
    "regime_label",
)


def write_corpus(corpus: dict[str, Any], out_path: Path) -> None:
    """Write the corpus JSON atomically (never leaves a partial file)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(corpus, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(out_path)
    LOGGER.info("Wrote %d papers to %s", len(corpus["papers"]), out_path)


def load_corpus(path: Path) -> dict[str, Any]:
    """Load a corpus JSON file, raising ``SystemExit`` with a clear message."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Corpus file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Corpus file is not valid JSON ({path}): {exc}") from exc


def verify_corpus(corpus: dict[str, Any], target_per_regime: int) -> bool:
    """Validate regime balance and the required schema fields."""
    ok = True
    papers = corpus.get("papers", [])
    regimes: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        if not isinstance(paper, dict):
            LOGGER.error("Corpus entry is not an object: %r", paper)
            ok = False
            continue
        missing = [name for name in REQUIRED_FIELDS if name not in paper]
        if missing:
            LOGGER.error(
                "Paper %r missing fields %s", paper.get("id", "<no id>"), missing
            )
            ok = False
            continue
        if not isinstance(paper["authors"], list) or not paper["authors"]:
            LOGGER.error("Paper %r has invalid 'authors'", paper["id"])
            ok = False
        if not isinstance(paper["year"], int):
            LOGGER.error("Paper %r has non-integer 'year'", paper["id"])
            ok = False
        if not isinstance(paper["abstract"], str) or not paper["abstract"].strip():
            LOGGER.error("Paper %r has empty 'abstract'", paper["id"])
            ok = False
        if not isinstance(paper["regime_id"], int):
            LOGGER.error("Paper %r has non-integer 'regime_id'", paper["id"])
            ok = False
        regimes.setdefault(str(paper["regime_label"]), []).append(paper)

    LOGGER.info("Verifying corpus: %d papers total", len(papers))
    for label, items in sorted(regimes.items()):
        ids = {item["id"] for item in items}
        titles = {normalize_title(item["title"]) for item in items}
        status = "OK" if len(items) == target_per_regime else "MISMATCH"
        LOGGER.info(
            "  %-10s %3d papers (unique ids=%d, unique titles=%d) -> %s",
            label,
            len(items),
            len(ids),
            len(titles),
            status,
        )
        if len(items) != target_per_regime:
            ok = False
        if len(ids) != len(items) or len(titles) != len(items):
            LOGGER.error("  %s contains duplicate IDs or titles", label)
            ok = False

    if len(papers) != target_per_regime * len(regimes):
        LOGGER.error(
            "Total corpus size %d != %d (regimes x target)",
            len(papers),
            target_per_regime * len(regimes),
        )
        ok = False

    return ok


# --------------------------------------------------------------------------- #
# Phase 1: full-text extraction (seeding papers + corpus)
# --------------------------------------------------------------------------- #
#: Minimum word count for an extracted document to be considered usable.
FULLTEXT_MIN_WORDS = 1000
#: Acceptable average word length, in characters. Below ~3.5 the text is
#: usually a table of numbers or a mangled font encoding; above ~9.0 it is
#: usually a stream of concatenated glyphs with the spaces dropped.
FULLTEXT_WORD_LENGTH_RANGE = (3.5, 9.0)
#: Minimum share of characters that are letters or digits, counted with
#: Unicode semantics so that math-dense papers are not mistaken for a broken
#: extraction: a Greek alpha in an equation is a letter, not symbol soup.
#: A scanned page whose OCR failed, or a PDF with a broken CMap table, still
#: falls far below the threshold because it emits private-use glyphs and
#: punctuation rather than letters.
FULLTEXT_MIN_ALNUM_RATIO = 0.75
#: arXiv asks clients to identify themselves and to leave >= 3 s between calls.
FULLTEXT_USER_AGENT = "ResearchRadar/0.1 (local research corpus; contact: repo owner)"

_WHITESPACE_RE = re.compile(r"\s+")
#: Text extracted from a PDF keeps the typesetter's hard line breaks. Joining on
#: a bare newline preserves paragraph structure; a hyphen at end of line is a
#: word that was split, so the hyphen and the break both have to go.
_HYPHEN_WRAP_RE = re.compile(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])")
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u00ad": "",
}


@dataclass(frozen=True)
class TextQuality:
    """Result of auditing one extracted document.

    The three checks are deliberately orthogonal: a word count catches an empty
    or truncated extraction, the mean word length catches a bad font encoding,
    and the alphanumeric ratio catches a mostly-symbols page. A single check
    alone lets two different failure modes through.
    """

    words: int
    chars: int
    mean_word_length: float
    alnum_ratio: float
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every extraction check passed."""
        return not self.reasons

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view, used in the provenance block."""
        return {
            "words": self.words,
            "chars": self.chars,
            "mean_word_length": round(self.mean_word_length, 3),
            "alnum_ratio": round(self.alnum_ratio, 4),
            "ok": self.ok,
            "reasons": list(self.reasons),
        }


def normalize_extracted_text(text: str) -> str:
    """Repair the damage a PDF text layer inflicts on running prose."""
    if not text:
        return ""
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = _HYPHEN_WRAP_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Phase 1b: bibliography and citation sanitation
# --------------------------------------------------------------------------- #
# Obayashi et al. Appendix A.1 Rule 4 forbids handing the verbalizer "a
# bibliographic fragment".  The same rule has to hold one stage earlier: author
# surnames, journal titles, DOIs, arXiv IDs and bracketed citation tokens must
# never enter the 250k character n-gram vocabulary, because a term that exists
# only inside a reference list still gets a df count, still enters the L1
# evidence package, and still looks like supporting evidence for a hypothesis
# that no paper in the corpus actually makes.
#
# Sanitation runs on the RAW text, before ``normalize_extracted_text`` collapses
# newlines, because every heading and running-header test below is line-anchored.

#: Optional leading numbering on a section heading ("4.", "IV.", "7").
_HEADING_LEAD = r"^[ \t]*(?:(?:\d{1,2}|[IVXLC]{1,5})\.?[ \t]+)?"

#: Headings that terminate the intellectual body of a paper.  The whole line
#: must be the heading, optionally trailed by a page number, so that a
#: sentence-final mention ("... as shown in the references.") never matches.
_REFERENCES_HEADING_RE = re.compile(
    _HEADING_LEAD
    + r"(?:references|bibliography|literature cited|works cited|cited works)"
    + r"[ \t]*:?[ \t]*(?:\d{1,4})?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: Acknowledgments sit immediately before the reference list, so they are
#: dropped with it rather than being treated as evidence.
_ACKNOWLEDGMENTS_HEADING_RE = re.compile(
    _HEADING_LEAD
    + r"(?:acknowledg(?:e)?ments?|funding|author contributions)"
    + r"[ \t]*:?[ \t]*(?:\d{1,4})?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_BIB_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_BIB_HINT_RE = re.compile(
    r"\b(?:et\s+al|doi|arxiv|vol\.|pp\.|no\.|proceedings|journal|press|preprint"
    r"|springer|elsevier|wiley|ieee|acm)\b",
    re.IGNORECASE,
)


#: The title / author / affiliation block carries surnames, institutions, and
#: e-mail addresses, which are bibliographic fragments by any reading of Rule 4.
#: The block is cut at the abstract heading, which is where argument starts.
#: The separator is required (or end of line), so prose beginning "Abstract
#: reasoning is ..." is never mistaken for the heading.
_ABSTRACT_HEADING_RE = re.compile(
    r"^[ \t]*a[ \t]*b[ \t]*s[ \t]*t[ \t]*r[ \t]*a[ \t]*c[ \t]*t"
    r"[ \t]*(?:[:.\u2014\u2013-][ \t]*|$)",
    re.IGNORECASE | re.MULTILINE,
)
#: Front matter is only stripped when it is genuinely at the front of the file.
FRONT_MATTER_LIMIT = 3000


def strip_front_matter(
    raw: str, max_prefix_chars: int = FRONT_MATTER_LIMIT
) -> tuple[str, int]:
    """Drop the title/author/affiliation block preceding the abstract."""
    if not raw:
        return raw, 0
    match = _ABSTRACT_HEADING_RE.search(raw, 0, max_prefix_chars)
    if match is None:
        return raw, 0
    body = raw[match.end() :].lstrip("\n\t ")
    return body, len(raw) - len(body)


#: An acknowledgments section is page furniture only when it is genuinely the
#: last thing before the references.  A section number after it means it is a
#: real body section that happens to share the heading text, so it stays.
_SECTION_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:\d{1,2}|[IVXLC]{1,5})\.?[ \t]+[A-Z][^\n]{0,80}$"
)
MAX_ACKNOWLEDGMENT_WORDS = 400


def _is_trailing_section(body: str, after_heading: int) -> bool:
    """True when nothing but an acknowledgment-sized stub follows the heading."""
    tail = body[after_heading:]
    if len(tail.split()) > MAX_ACKNOWLEDGMENT_WORDS:
        return False
    return _SECTION_HEADING_RE.search(tail) is None


def looks_like_bibliography(tail: str, year_ratio: float = 0.35) -> bool:
    """True when ``tail`` reads like a reference list rather than prose.

    Reference entries are dense in four-digit years and in publisher-ish tokens;
    running prose is not.  This is what stops a stray "references" line left by
    a bad line break (or a real in-text mention) from truncating a paper body.
    """
    lines = [line for line in tail.splitlines() if line.strip()][:80]
    if not lines:
        return False
    year_lines = sum(1 for line in lines if _BIB_YEAR_RE.search(line))
    ratio = year_lines / len(lines)
    if ratio >= year_ratio:
        return True
    # A short reference list can sit under the ratio and still be unmistakable.
    return ratio >= 0.2 and bool(_BIB_HINT_RE.search(tail[:4000]))


def strip_bibliography(
    raw: str, min_body_words: int = FULLTEXT_MIN_WORDS
) -> tuple[str, int, int]:
    """Cut the reference list, and any trailing acknowledgments, from a paper.

    Returns ``(body, removed_reference_chars, removed_acknowledgment_chars)``.
    A candidate heading is accepted only when

    * it is line-anchored, so an in-text mention cannot match, and
    * the text before it holds at least ``min_body_words`` words, so a heading
      cannot swallow the body, and
    * the text after it looks like a bibliography.

    Among accepted candidates the earliest wins: everything from the first real
    reference heading onwards is bibliography no matter how many headings
    follow it.  If no candidate qualifies the text is returned untouched, so
    nothing is ever truncated on a guess.
    """
    if not raw:
        return raw, 0, 0

    cut_at = -1
    for match in _REFERENCES_HEADING_RE.finditer(raw):
        if len(raw[: match.start()].split()) < min_body_words:
            continue
        if not looks_like_bibliography(raw[match.end() :]):
            continue
        cut_at = match.start()
        break
    if cut_at < 0:
        return raw, 0, 0

    body, references = raw[:cut_at], raw[cut_at:]

    acknowledgments = ""
    ack_matches = [
        match
        for match in _ACKNOWLEDGMENTS_HEADING_RE.finditer(body)
        if len(body[: match.start()].split()) >= min_body_words
        and _is_trailing_section(body, match.end())
    ]
    if ack_matches:
        acknowledgments = body[ack_matches[-1].start() :]
        body = body[: ack_matches[-1].start()]

    if len(body.split()) < min_body_words:
        LOGGER.warning(
            "Bibliography trim would leave %d words (< %d); keeping the "
            "untrimmed text instead",
            len(body.split()),
            min_body_words,
        )
        return raw, 0, 0

    return body, len(references), len(acknowledgments)


#: Bracketed numeric citation callouts: ``[12]``, ``[1, 2]``, ``[5-9]``.
#: The guard keeps mathematical intervals such as ``[-1, 1]`` and ``x in [0, 1]``
#: from being deleted: an interval has a sign, an operator, or a word such as
#: "in" / "between" in front of it.  Ranges are only stripped on an en or em
#: dash, because ``[0-1]`` is far more often an interval than a citation.
_CITATION_BRACKET_RE = re.compile(
    r"\[\s*\d{1,4}[a-z]?(?:\s*[,;]\s*\d{1,4}[a-z]?)*\s*\]"
)
_CITATION_RANGE_RE = re.compile(r"\[\s*\d{1,4}\s*[\u2013\u2014]\s*\d{1,4}\s*\]")
#: ASCII-hyphen ranges are only stripped when they cannot be an interval: a
#: citation range starts at 1 or above and increases, whereas ``[0-1]`` does not.
_CITATION_HYPHEN_RANGE_RE = re.compile(r"\[\s*(\d{1,4})\s*-\s*(\d{1,4})\s*\]")
_CITATION_BRACKET_NAMED_RE = re.compile(r"\[[A-Z][^\]\n]{0,48}(?:19|20)\d{2}[a-z]?\]")
#: Tokens that mark the preceding bracket as an interval, not a citation.
#: A symbol or operator in front of the bracket is decisive.  The word list is
#: deliberately CASE-SENSITIVE: lowercase "in"/"over" introduces an interval
#: ("x in [0, 1]"), while sentence-initial "In" introduces a citation
#: ("In [12] we show").
_BRACKET_SYMBOL_GUARD_RE = re.compile(r"[\u2208\u2282\u2286=< >\u2264\u2265(+\-/]\s*$".replace("< >", "<>"))
#: Guard words are case-insensitive EXCEPT ``in``.  "In [12] we show" is a
#: sentence-initial citation; "x in [0, 1]" is an interval.  No other guard word
#: is commonly used to introduce a citation, so they can be caught in any case
#: ("Over [1, 2] the field is smooth").
_BRACKET_WORD_GUARD_RE = re.compile(
    r"(?:\bin\s*$)"
    r"|(?i:\b(?:within|between|range|interval|from|to|spans?|spanning|domain"
    r"|bounds?|grid|over|covering|region)\s*$)"
)
_CITATION_BRACKET_GUARD_RE = re.compile(
    "(?:" + _BRACKET_SYMBOL_GUARD_RE.pattern + "|" + _BRACKET_WORD_GUARD_RE.pattern + ")"
)

#: A surname written just before a numeric citation ("see Smith et al. [12]").
#: The lookbehind requires lowercase word end or punctuation first, so a
#: sentence-initial word ("In [12] we ...") is never treated as a surname.
_STRUCTURAL_WORDS = (
    "Section|Table|Figure|Fig|Eq|Equation|Chapter|Algorithm|Appendix|Lemma|Theorem"
    "|Remark|Definition|Proposition|Corollary|Listing"
)
_CITATION_LEAD_SURNAME_RE = re.compile(
    r"(?<=[a-z,;]\s)"
    r"(?!" + _STRUCTURAL_WORDS + r"\b)"
    r"[A-Z][A-Za-z'\u2019\-]{2,}"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\u2019\-]{2,}))?"
    r"[ \t]*(?=\[)"
)

#: Parenthetical author-year callouts: ``(Smith, 2020)``,
#: ``(Smith et al., 2020)``, ``(Smith and Jones, 2019, 2020)``.
_CITATION_PAREN_RE = re.compile(
    r"\(\s*[A-Z][A-Za-z'\u2019\-]+"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\u2019\-]+|&\s*[A-Z][A-Za-z'\u2019\-]+))?"
    r"\s*,\s*(?:19|20)\d{2}[a-z]?"
    r"(?:\s*[,;]\s*(?:19|20)\d{2}[a-z]?)*\s*\)"
)
#: Narrative author-year callouts: ``Smith et al. (2020)``.  Structural words
#: are excluded so that ``Algorithm (2020)`` and friends survive untouched.
_CITATION_NARRATIVE_RE = re.compile(
    r"(?!(?:" + _STRUCTURAL_WORDS + r")\b)"
    r"[A-Z][A-Za-z'\u2019\-]+"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\u2019\-]+|&\s*[A-Z][A-Za-z'\u2019\-]+))?"
    r"\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
)

#: Locators that are pure bibliographic noise wherever they appear.
_BIBLIOGRAPHIC_LOCATOR_RES = (
    re.compile(r"https?://\S+"),
    re.compile(r"\bwww\.\S+"),
    re.compile(r"\bdoi\s*:\s*\S+", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+"),
    re.compile(r"\barxiv\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE),
    re.compile(r"\b(?:isbn|issn)\s*[:#]?\s*[\dXx\-]{8,}", re.IGNORECASE),
)

#: Running heads, footers, and publisher boilerplate stamped onto every page.
_RUNNING_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:"
    r"arxiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[[^\]]*\][^\n]*"
    r"|preprint\s+submitted\s+to[^\n]*"
    r"|submitted\s+to[^\n]*"
    r"|under\s+review\s+(?:as|for)[^\n]*"
    r"|published\s+as[^\n]*"
    r"|to\s+appear\s+in[^\n]*"
    r"|authorized\s+licensed\s+use[^\n]*"
    r"|downloaded\s+(?:from|on)[^\n]*"
    r"|(?:ieee|acm)\s+[A-Z][A-Za-z \-]*,?\s*(?:vol\.|no\.)[^\n]*"
    r"|(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?"
    r")\s*$"
)


def strip_running_headers(raw: str, repeat_threshold: int = 3) -> tuple[str, int]:
    """Drop page furniture: publisher stamps, page numbers, repeated heads.

    Two passes.  The first removes the usual boilerplate by pattern; the second
    removes whatever a particular publisher repeats verbatim (journal names,
    licence lines).  The repetition pass only considers short lines, so it
    cannot eat paragraph text.
    """
    if not raw:
        return raw, 0

    kept: list[str] = []
    removed = 0
    for line in raw.split("\n"):
        stripped = line.strip()
        if _RUNNING_HEADER_RE.match(stripped):
            if stripped:
                removed += 1
            continue
        kept.append(line)

    counts: dict[str, int] = {}
    for line in kept:
        stripped = line.strip()
        # Running heads carry words; a repeated line of bare numbers is table
        # data and must survive.
        if 6 <= len(stripped) <= 80 and sum(
            1 for char in stripped if char.isalpha()
        ) >= 8:
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {line for line, count in counts.items() if count >= repeat_threshold}
    if repeated:
        survivors = [line for line in kept if line.strip() not in repeated]
        removed += len(kept) - len(survivors)
        kept = survivors

    return "\n".join(kept), removed


def strip_citation_callouts(text: str) -> tuple[str, int]:
    """Remove citation callouts and bibliographic locators from body prose."""
    if not text:
        return text, 0
    removed = 0

    def window_before(match: re.Match[str]) -> str:
        return text[max(0, match.start() - 14) : match.start()]

    def drop_brackets(match: re.Match[str]) -> str:
        nonlocal removed
        if _CITATION_BRACKET_GUARD_RE.search(window_before(match)):
            return match.group(0)
        removed += 1
        return ""

    def drop_hyphen_range(match: re.Match[str]) -> str:
        nonlocal removed
        low, high = int(match.group(1)), int(match.group(2))
        # A citation range starts at 1 and increases; an interval need not.
        if low < 1 or high <= low:
            return match.group(0)
        if _CITATION_BRACKET_GUARD_RE.search(window_before(match)):
            return match.group(0)
        removed += 1
        return ""

    # Surnames written immediately before a numeric citation go first, so that
    # "see Smith et al. [12]" does not leave a bare surname in the prose.
    text, count = _CITATION_LEAD_SURNAME_RE.subn("", text)
    removed += count

    for pattern in (
        _CITATION_RANGE_RE,
        _CITATION_HYPHEN_RANGE_RE,
        _CITATION_BRACKET_RE,
    ):
        handler = (
            drop_hyphen_range if pattern is _CITATION_HYPHEN_RANGE_RE else drop_brackets
        )
        text = pattern.sub(handler, text)
    # A bracketed author-year (``[Smith 2020]``) can never be a numeric
    # interval, so it is removed without consulting the interval guard.
    text, count = _CITATION_BRACKET_NAMED_RE.subn("", text)
    removed += count
    for pattern in (_CITATION_PAREN_RE, _CITATION_NARRATIVE_RE):
        text, count = pattern.subn("", text)
        removed += count
    for pattern in _BIBLIOGRAPHIC_LOCATOR_RES:
        text, count = pattern.subn("", text)
        removed += count

    # Deleting a callout leaves a space before the punctuation that followed it.
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text, removed


def count_bibliographic_remnants(text: str) -> int:
    """Count bibliographic noise still visible in sanitized text.

    Used by ``verify_fulltext`` as a regression test: every one of these should
    already have been removed, so a non-zero count means sanitation regressed.
    """
    if not text:
        return 0
    return sum(len(pattern.findall(text)) for pattern in _BIBLIOGRAPHIC_LOCATOR_RES)


@dataclass(frozen=True)
class SanitationResult:
    """Sanitized document text plus a record of everything taken out."""

    text: str
    removed_front_matter: int = 0
    removed_references: int = 0
    removed_acknowledgments: int = 0
    removed_citations: int = 0
    removed_headers: int = 0

    @property
    def removed_total(self) -> int:
        """Sum of every removed unit, used for reporting."""
        return (
            self.removed_front_matter
            + self.removed_references
            + self.removed_acknowledgments
            + self.removed_citations
            + self.removed_headers
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly provenance record."""
        return {
            "front_matter_chars": self.removed_front_matter,
            "reference_chars": self.removed_references,
            "acknowledgment_chars": self.removed_acknowledgments,
            "citation_callouts": self.removed_citations,
            "running_headers": self.removed_headers,
        }


def sanitize_document_text(
    raw: str, min_body_words: int = FULLTEXT_MIN_WORDS
) -> SanitationResult:
    """Run the full Phase 1b sanitation chain, then normalize.

    Order matters: front matter and running headers go first so heading
    detection is not fooled by page furniture, then the bibliography is cut,
    then callouts are removed from what remains of the body.
    """
    without_headers, headers = strip_running_headers(raw)
    body, front_matter = strip_front_matter(without_headers)
    body, references, acknowledgments = strip_bibliography(body, min_body_words)
    body, citations = strip_citation_callouts(body)
    return SanitationResult(
        text=normalize_extracted_text(body),
        removed_front_matter=front_matter,
        removed_references=references,
        removed_acknowledgments=acknowledgments,
        removed_citations=citations,
        removed_headers=headers,
    )


def _config_number(key: str, value: Any, default: float) -> float:
    """Coerce one configured threshold, failing loudly on garbage.

    A silently coerced ``None`` or ``"1000"`` would change the extraction gate
    without anyone noticing, so a bad value aborts the run and names the key.
    """
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"fulltext.{key} must be a number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class QualityGate:
    """The three extraction sanity thresholds, read from ``config.yaml``.

    Held in one object so the extractors and the verifier cannot drift apart:
    a document accepted at extraction time must also pass verification.
    """

    min_words: int = FULLTEXT_MIN_WORDS
    word_length_range: tuple[float, float] = FULLTEXT_WORD_LENGTH_RANGE
    min_alnum_ratio: float = FULLTEXT_MIN_ALNUM_RATIO

    @classmethod
    def from_config(cls, fulltext_cfg: dict[str, Any]) -> QualityGate:
        low_high = fulltext_cfg.get(
            "mean_word_length_range", list(FULLTEXT_WORD_LENGTH_RANGE)
        )
        if not isinstance(low_high, (list, tuple)) or len(low_high) != 2:
            raise SystemExit(
                "fulltext.mean_word_length_range must be a two-element list, "
                f"got {low_high!r}"
            )
        return cls(
            min_words=int(
                _config_number(
                    "min_words",
                    fulltext_cfg.get("min_words"),
                    FULLTEXT_MIN_WORDS,
                )
            ),
            word_length_range=(
                _config_number(
                    "mean_word_length_range[0]",
                    low_high[0],
                    FULLTEXT_WORD_LENGTH_RANGE[0],
                ),
                _config_number(
                    "mean_word_length_range[1]",
                    low_high[1],
                    FULLTEXT_WORD_LENGTH_RANGE[1],
                ),
            ),
            min_alnum_ratio=_config_number(
                "min_alnum_ratio",
                fulltext_cfg.get("min_alnum_ratio"),
                FULLTEXT_MIN_ALNUM_RATIO,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view, written into the artifact's metadata."""
        return {
            "min_words": self.min_words,
            "mean_word_length_range": list(self.word_length_range),
            "min_alnum_ratio": self.min_alnum_ratio,
        }

    def assess(self, text: str) -> TextQuality:
        """Audit extracted text against the three checks."""
        return assess_text_quality(
            text,
            min_words=self.min_words,
            word_length_range=self.word_length_range,
            min_alnum_ratio=self.min_alnum_ratio,
        )


def assess_text_quality(
    text: str,
    min_words: int = FULLTEXT_MIN_WORDS,
    word_length_range: tuple[float, float] = FULLTEXT_WORD_LENGTH_RANGE,
    min_alnum_ratio: float = FULLTEXT_MIN_ALNUM_RATIO,
) -> TextQuality:
    """Audit extracted text against the three extraction sanity checks."""
    stripped = text.strip()
    words = stripped.split()
    word_count = len(words)
    char_count = len(stripped)
    if word_count:
        mean_word_length = char_count / word_count
    else:
        mean_word_length = 0.0
    # Measured over non-whitespace characters. The check exists to catch garbled
    # extraction - symbol soup from a broken font encoding - and whitespace
    # carries no signal about that. Dividing by the total character count instead
    # quietly punishes equation-dense papers: two control-theory papers in this
    # corpus score 0.66 that way but 0.83 this way, which is the difference
    # between rejecting good text and catching real garbage.
    visible = [char for char in stripped if not char.isspace()]
    if visible:
        alnum_ratio = sum(1 for char in visible if char.isalnum()) / len(visible)
    else:
        alnum_ratio = 0.0

    reasons: list[str] = []
    if word_count < min_words:
        reasons.append(f"only {word_count} words (need >= {min_words})")
    low, high = word_length_range
    if not low <= mean_word_length <= high:
        reasons.append(
            f"mean word length {mean_word_length:.2f} outside [{low}, {high}]"
        )
    if alnum_ratio <= min_alnum_ratio:
        reasons.append(
            f"alphanumeric ratio {alnum_ratio:.3f} <= {min_alnum_ratio}"
        )
    return TextQuality(
        words=word_count,
        chars=char_count,
        mean_word_length=mean_word_length,
        alnum_ratio=alnum_ratio,
        reasons=tuple(reasons),
    )


def extract_pdf_text(
    pdf_path: Path,
    timeout: float = 300.0,
    min_body_words: int = FULLTEXT_MIN_WORDS,
) -> SanitationResult:
    """Extract with ``mutool`` (MuPDF), then sanitize. Empty result on failure."""
    try:
        result = subprocess.run(
            ["mutool", "draw", "-F", "txt", "-o", "-", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "mutool is required for PDF text extraction but was not found on PATH"
        ) from exc
    except subprocess.TimeoutExpired:
        LOGGER.warning("mutool timed out after %.0fs on %s", timeout, pdf_path.name)
        return SanitationResult(text="")
    if result.returncode != 0:
        LOGGER.debug(
            "mutool exited %d on %s: %s",
            result.returncode,
            pdf_path.name,
            result.stderr.strip()[:300],
        )
    return sanitize_document_text(result.stdout, min_body_words)


def extract_ar5iv_text(
    arxiv_id: str,
    session: requests.Session,
    timeout: float = 45.0,
    min_body_words: int = FULLTEXT_MIN_WORDS,
) -> SanitationResult:
    """Fallback: pull the arXiv HTML5 rendering (ar5iv), reduce it to text."""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.debug("ar5iv failed for %s: %s", arxiv_id, exc)
        return SanitationResult(text="")
    try:
        document = lxml_html.fromstring(response.text)
    except (ValueError, TypeError) as exc:
        LOGGER.debug("ar5iv returned unparseable HTML for %s: %s", arxiv_id, exc)
        return SanitationResult(text="")
    # The bibliography and the acknowledgments are navigation, not argument, so
    # they are dropped by element rather than by heading text.
    for node in document.xpath(
        "//script|//style|//nav"
        "|//*[contains(@class,'ltx_bibliography')]"
        "|//*[contains(@class,'ltx_acknowledgements')]"
        "|//*[contains(@id,'bibliography')]"
    ):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)
    nodes = document.xpath("//article") or document.xpath("//body")
    text = "\n".join(node.text_content() for node in nodes)
    return sanitize_document_text(text, min_body_words)


def download_arxiv_pdf(
    arxiv_id: str,
    cache_dir: Path,
    session: requests.Session,
    delay_seconds: float = 3.0,
    num_retries: int = 4,
) -> Path | None:
    """Download ``arxiv.org/pdf/<id>`` once, caching the bytes on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{arxiv_id}.pdf"
    if target.exists() and target.stat().st_size > 10_000:
        return target
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(num_retries):
        try:
            response = session.get(url, timeout=90.0)
            response.raise_for_status()
            if len(response.content) < 10_000:
                raise requests.RequestException("suspiciously small PDF payload")
            target.write_bytes(response.content)
            time.sleep(delay_seconds)
            return target
        except requests.RequestException as exc:
            wait = delay_seconds * (2**attempt)
            LOGGER.debug(
                "PDF fetch %s attempt %d/%d failed (%s); waiting %.1fs",
                arxiv_id,
                attempt + 1,
                num_retries,
                exc,
                wait,
            )
            time.sleep(wait)
    LOGGER.warning("Could not download the PDF for %s", arxiv_id)
    return None


@dataclass
class ExtractedDocument:
    """One document's full text plus provenance for how it was obtained."""

    identifier: str
    title: str
    text: str
    source: str
    quality: TextQuality
    attempts: list[str] = field(default_factory=list)
    sanitation: dict[str, Any] = field(default_factory=dict)


def _seed_identifier(path: Path) -> str:
    """Recover an arXiv ID from a seed filename, or '' when it is a free-form name."""
    match = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)", path.stem)
    return match.group(1) if match else ""


def extract_seed_documents(
    desired_dir: Path,
    session: requests.Session,
    delay_seconds: float = 3.0,
    gate: QualityGate | None = None,
) -> tuple[list[ExtractedDocument], list[Path]]:
    """Extract every seeding paper, falling back to ar5iv before giving up.

    Returns the usable documents and the list of PDFs that could not be
    recovered automatically, so the caller can name them for the user.
    """
    quality_gate = gate or QualityGate()
    documents: list[ExtractedDocument] = []
    failures: list[Path] = []
    pdfs = sorted(desired_dir.glob("*.pdf"))
    if not pdfs:
        LOGGER.warning("No seeding PDFs found under %s", desired_dir)
        return documents, failures

    for pdf_path in pdfs:
        identifier = _seed_identifier(pdf_path) or pdf_path.stem
        attempts: list[str] = []

        result = extract_pdf_text(pdf_path, min_body_words=quality_gate.min_words)
        quality = quality_gate.assess(result.text)
        attempts.append(f"mutool:{'ok' if quality.ok else 'rejected'}")

        source = "mutool"
        if not quality.ok and identifier != pdf_path.stem:
            LOGGER.info(
                "mutool text for %s failed the audit (%s); trying ar5iv",
                pdf_path.name,
                "; ".join(quality.reasons),
            )
            fallback = extract_ar5iv_text(
                identifier, session, min_body_words=quality_gate.min_words
            )
            fallback_quality = quality_gate.assess(fallback.text)
            attempts.append(f"ar5iv:{'ok' if fallback_quality.ok else 'rejected'}")
            if fallback_quality.ok:
                result, quality, source = fallback, fallback_quality, "ar5iv"

        document = ExtractedDocument(
            identifier=identifier,
            title=pdf_path.stem,
            text=result.text if quality.ok else "",
            source=source if quality.ok else "failed",
            quality=quality,
            attempts=attempts,
            sanitation=result.as_dict(),
        )
        documents.append(document)
        if quality.ok:
            LOGGER.info(
                "  seed %-46s %6d words via %-9s refs=-%d cites=%d",
                pdf_path.name[:46],
                quality.words,
                source,
                result.removed_references + result.removed_acknowledgments,
                result.removed_citations,
            )
        else:
            LOGGER.error(
                "  seed %-58s FAILED: %s",
                pdf_path.name[:58],
                "; ".join(quality.reasons),
            )
            failures.append(pdf_path)
    return documents, failures


def extract_corpus_fulltext(
    corpus: dict[str, Any],
    cache_dir: Path,
    session: requests.Session,
    delay_seconds: float = 3.0,
    gate: QualityGate | None = None,
) -> tuple[list[ExtractedDocument], list[str]]:
    """Extract full text for every corpus paper, reusing the on-disk cache."""
    quality_gate = gate or QualityGate()
    documents: list[ExtractedDocument] = []
    failures: list[str] = []
    papers = corpus.get("papers", [])
    for position, paper in enumerate(papers, start=1):
        arxiv_id = str(paper["id"])
        attempts: list[str] = []
        result = SanitationResult(text="")
        quality = quality_gate.assess("")
        source = ""

        cached = cache_dir / f"{arxiv_id}.txt"
        if cached.exists():
            # The cache holds text that already went through sanitation, so this
            # second pass finds nothing to strip; it is here to keep provenance
            # honest rather than to re-do work.
            result = sanitize_document_text(
                cached.read_text(encoding="utf-8"), quality_gate.min_words
            )
            quality = quality_gate.assess(result.text)
            attempts.append("cache")
            source = "cache"
            if not quality.ok:
                LOGGER.warning(
                    "Cached text for %s fails the audit; re-extracting", arxiv_id
                )
                cached.unlink()
                result = SanitationResult(text="")
                quality = quality_gate.assess("")
                source = ""

        if not quality.ok:
            pdf_path = download_arxiv_pdf(
                arxiv_id, cache_dir / "pdf", session, delay_seconds=delay_seconds
            )
            if pdf_path is not None:
                result = extract_pdf_text(
                    pdf_path, min_body_words=quality_gate.min_words
                )
                quality = quality_gate.assess(result.text)
                attempts.append(f"pdf:{'ok' if quality.ok else 'rejected'}")
                source = "arxiv-pdf"

        if not quality.ok:
            fallback = extract_ar5iv_text(
                arxiv_id, session, min_body_words=quality_gate.min_words
            )
            fallback_quality = quality_gate.assess(fallback.text)
            attempts.append(f"ar5iv:{'ok' if fallback_quality.ok else 'rejected'}")
            if fallback_quality.ok:
                result, quality, source = fallback, fallback_quality, "ar5iv"

        if quality.ok:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(result.text, encoding="utf-8")
            LOGGER.info(
                "  [%3d/%3d] %-10s %6d words via %-9s refs=-%d cites=%d",
                position,
                len(papers),
                arxiv_id,
                quality.words,
                source,
                result.removed_references + result.removed_acknowledgments,
                result.removed_citations,
            )
        else:
            LOGGER.error(
                "  [%3d/%3d] %-10s FAILED: %s",
                position,
                len(papers),
                arxiv_id,
                "; ".join(quality.reasons) or "no text",
            )
            failures.append(arxiv_id)

        documents.append(
            ExtractedDocument(
                identifier=arxiv_id,
                title=str(paper.get("title", "")),
                text=result.text if quality.ok else "",
                source=source if quality.ok else "failed",
                quality=quality,
                attempts=attempts,
                sanitation=result.as_dict(),
            )
        )
    return documents, failures


def _sanitation_totals(documents: list[ExtractedDocument]) -> dict[str, int]:
    """Sum what sanitation removed across a set of documents."""
    totals: dict[str, int] = {}
    for document in documents:
        for key, value in document.sanitation.items():
            totals[key] = totals.get(key, 0) + int(value)
    return {f"removed_{key}": value for key, value in sorted(totals.items())}


def build_fulltext_corpus(config: dict[str, Any]) -> dict[str, Any]:
    """Run Phase 1: seeding papers first, then the whole analysis corpus."""
    fulltext_cfg = config.get("fulltext", {}) or {}
    corpus_cfg = config.get("corpus", {}) or {}
    corpus_path = Path(corpus_cfg.get("output_path", "data/corpus.json"))
    out_path = Path(fulltext_cfg.get("output_path", "data/corpus_fulltext.json"))
    desired_dir = Path(fulltext_cfg.get("seeds_dir", "docs/papers/desired"))
    cache_dir = Path(fulltext_cfg.get("cache_dir", "data/fulltext_cache"))
    delay = float(fulltext_cfg.get("delay_seconds", corpus_cfg.get("delay_seconds", 3.0)))
    gate = QualityGate.from_config(fulltext_cfg)

    corpus = load_corpus(corpus_path)
    session = requests.Session()
    session.headers.update({"User-Agent": FULLTEXT_USER_AGENT})

    LOGGER.info("Extracting seeding papers from %s", desired_dir)
    seeds, seed_failures = extract_seed_documents(desired_dir, session, delay, gate)

    LOGGER.info("Extracting full text for %d corpus papers", len(corpus["papers"]))
    corpus_docs, corpus_failures = extract_corpus_fulltext(
        corpus, cache_dir, session, delay, gate
    )

    by_id = {doc.identifier: doc for doc in corpus_docs}
    papers: list[dict[str, Any]] = []
    for paper in corpus["papers"]:
        document = by_id[str(paper["id"])]
        papers.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "abstract": paper["abstract"],
                "full_text": document.text,
                "authors": paper["authors"],
                "year": paper["year"],
                "regime_id": paper["regime_id"],
                "regime_label": paper["regime_label"],
                "extraction": {
                    "source": document.source,
                    "attempts": document.attempts,
                    **document.quality.as_dict(),
                },
                "sanitation": document.sanitation,
            }
        )

    payload = {
        "metadata": {
            "generated_by": "src.harvest (Phase 1: full-text extraction)",
            "reference": "Obayashi et al. (2026) Knowledge Manifold Optimization, Sec. 5.1-5.2",
            "n_papers": len(papers),
            "n_seeds": len(seeds),
            "corpus_path": str(corpus_path),
            "seeds_dir": str(desired_dir),
            "extractor": "mutool (MuPDF) with ar5iv HTML fallback",
            "quality_gate": gate.as_dict(),
            "sanitation": {
                "rule": (
                    "Appendix A.1 Rule 4: no bibliographic fragments in the "
                    "n-gram vocabulary or the evidence packages"
                ),
                "truncate_at_references": True,
                "drop_trailing_acknowledgments": True,
                "strip_citation_callouts": True,
                "strip_running_headers": True,
                **_sanitation_totals([*seeds, *corpus_docs]),
            },
            "failed_papers": corpus_failures,
            "failed_seeds": [path.name for path in seed_failures],
        },
        "papers": papers,
        "seeds": [
            {
                "identifier": doc.identifier,
                "title": doc.title,
                "full_text": doc.text,
                "extraction": {
                    "source": doc.source,
                    "attempts": doc.attempts,
                    **doc.quality.as_dict(),
                },
                "sanitation": doc.sanitation,
            }
            for doc in seeds
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(out_path)

    LOGGER.info(
        "Wrote full text for %d/%d corpus papers and %d seeds to %s",
        len(papers) - len(corpus_failures),
        len(papers),
        len(seeds) - len(seed_failures),
        out_path,
    )
    if seed_failures:
        LOGGER.error(
            "These seeding PDFs could not be extracted automatically and need "
            "manual text: %s",
            ", ".join(path.name for path in seed_failures),
        )
    return payload


def verify_fulltext(
    path: Path, expected_papers: int, gate: QualityGate | None = None
) -> bool:
    """Validate the Phase 1 artifact: schema, quality gate, and coverage."""
    quality_gate = gate or QualityGate()
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        LOGGER.error("Full-text file not found: %s", path)
        return False
    except json.JSONDecodeError as exc:
        LOGGER.error("Full-text file is not valid JSON (%s): %s", path, exc)
        return False

    required = (
        "id",
        "title",
        "abstract",
        "full_text",
        "authors",
        "year",
        "regime_id",
    )
    papers = payload.get("papers", [])
    ok = True
    if len(papers) != expected_papers:
        LOGGER.error(
            "Expected %d papers with full text, found %d",
            expected_papers,
            len(papers),
        )
        ok = False

    ids: set[str] = set()
    regimes: dict[str, int] = {}
    shortest = 10**9
    remnants = 0
    for paper in papers:
        missing = [name for name in required if name not in paper]
        if missing:
            LOGGER.error("Paper %r missing fields %s", paper.get("id"), missing)
            ok = False
            continue
        if paper["id"] in ids:
            LOGGER.error("Duplicate paper id %r", paper["id"])
            ok = False
        ids.add(paper["id"])
        quality = quality_gate.assess(paper["full_text"])
        if not quality.ok:
            LOGGER.error(
                "Paper %s fails the extraction audit: %s",
                paper["id"],
                "; ".join(quality.reasons),
            )
            ok = False
        leftover = count_bibliographic_remnants(paper["full_text"])
        if leftover:
            LOGGER.error(
                "Paper %s still carries %d bibliographic locator(s); "
                "sanitation regressed",
                paper["id"],
                leftover,
            )
            remnants += leftover
            ok = False
        shortest = min(shortest, quality.words)
        regimes[str(paper["regime_label"])] = (
            regimes.get(str(paper["regime_label"]), 0) + 1
        )

    seeds = payload.get("seeds", [])
    bad_seeds = [
        seed["identifier"]
        for seed in seeds
        if not quality_gate.assess(seed["full_text"]).ok
    ]
    if bad_seeds:
        LOGGER.error("Seeding papers failing the audit: %s", ", ".join(bad_seeds))
        ok = False

    LOGGER.info(
        "Full text: %d papers (+%d seeds), %d unique ids, shortest=%d words",
        len(papers),
        len(seeds),
        len(ids),
        shortest if papers else 0,
    )
    LOGGER.info(
        "Bibliographic remnants surviving sanitation: %d", remnants
    )
    for label, count in sorted(regimes.items()):
        LOGGER.info("  %-10s %3d papers", label, count)
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest the three-regime ResearchRadar arXiv corpus."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yaml"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Override corpus output path"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing corpus file, do not query arXiv",
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help=(
            "Phase 1: extract full text for the seeding papers and every corpus "
            "paper into the full-text corpus file"
        ),
    )
    parser.add_argument(
        "--verify-fulltext",
        action="store_true",
        help="Only verify an existing full-text corpus file",
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

    corpus_cfg = config.get("corpus", {})
    out_path = Path(
        args.out or corpus_cfg.get("output_path", "data/corpus.json")
    )
    target_per_regime = int(corpus_cfg.get("target_per_regime", 40))

    fulltext_cfg = config.get("fulltext", {}) or {}
    fulltext_path = Path(
        fulltext_cfg.get("output_path", "data/corpus_fulltext.json")
    )

    if args.verify_fulltext:
        return (
            0
            if verify_fulltext(
                fulltext_path,
                target_per_regime * 3,
                QualityGate.from_config(fulltext_cfg),
            )
            else 1
        )

    if args.verify_only:
        corpus = load_corpus(out_path)
        return 0 if verify_corpus(corpus, target_per_regime) else 1

    if args.extract_text:
        payload = build_fulltext_corpus(config)
        return (
            0
            if verify_fulltext(
                fulltext_path,
                len(payload["papers"]),
                QualityGate.from_config(fulltext_cfg),
            )
            else 1
        )

    corpus = harvest_all(config)
    write_corpus(corpus, out_path)
    return 0 if verify_corpus(corpus, target_per_regime) else 1


if __name__ == "__main__":
    raise SystemExit(main())
