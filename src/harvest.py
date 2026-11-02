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
* Results are deduplicated across arXiv IDs *and* normalized titles.
* Seed-First Expansion: the hand-picked PDFs in ``fulltext.seeds_dir`` occupy
  ``doc 1``..``doc 15`` of the corpus and the harvested papers fill the rest, so
  seeding papers and harvest share one representation (see
  ``build_fulltext_corpus``).

The corpus is written to ``data/corpus.json`` (metadata) and
``data/corpus_fulltext.json`` (sanitised full text for every document).

Usage
-----
    python -m src.harvest                      # harvest metadata only
    python -m src.harvest --extract-text       # full Seed-First rebuild
    python -m src.harvest --verify-fulltext
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv
import numpy as np
import requests
from lxml import html as lxml_html
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import is_single_field, load_config

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


def arxiv_base_id(identifier: str) -> str:
    """Strip an arXiv version suffix so ``2511.04131`` matches ``2511.04131v2``.

    Seed identifiers come from PDF filenames, which may carry a version or omit
    it, while the API always reports one.  Comparing raw strings therefore let a
    seed back into the harvest under its versioned id, which the corpus verify
    then reported as a duplicate title within a regime.
    """
    return re.sub(r"v\d+$", "", identifier.strip())


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
                "arXiv paging error (%s: %s); retry %d/%d in %.1fs (resuming at #%d)",
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
    date_range: str = "",
) -> list[dict[str, Any]]:
    """Harvest up to ``target_count`` papers for a single regime.

    Papers already present in ``seen_ids`` / ``seen_titles`` are skipped so the
    regimes stay disjoint and no seeding paper is fetched twice; new IDs and
    titles are registered as records are accepted.  ``date_range`` is an optional
    arXiv ``submittedDate`` clause ANDed onto the regime query.
    """
    collected: list[dict[str, Any]] = []
    scanned = 0
    rejected_review = 0
    rejected_short = 0
    rejected_dupe = 0

    base_query = str(regime_cfg["query"])
    query = f"({base_query}) AND {date_range}" if date_range else base_query
    if date_range:
        LOGGER.info("[%s] query with recency window: %s", regime_cfg["label"], query)

    search = arxiv.Search(
        query=query,
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

        if arxiv_base_id(short_id) in seen_ids or title_key in seen_titles:
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
            "is_seed": False,
            # --- provenance (not required by the schema, kept for auditability)
            "regime_name": regime_cfg["name"],
            "entry_id": result.entry_id,
            "published": published.date().isoformat(),
            "updated": updated.date().isoformat() if updated else None,
            "primary_category": result.primary_category,
            "categories": list(result.categories),
            "doi": result.doi,
            "journal_ref": result.journal_ref,
            "query": query,
        }

        collected.append(record)
        seen_ids.add(arxiv_base_id(short_id))
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


def _config_int(block: dict[str, Any], key: str, default: int) -> int:
    """Read an integer from a config block, failing loudly and naming the key.

    A silently coerced value would change the corpus size or the recency window
    without anyone noticing, so a bad value aborts the run instead.
    """
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(str(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key} must be an integer, got {value!r}") from exc
    return value


def _config_float(block: dict[str, Any], key: str, default: float) -> float:
    """Read a float from a config block, failing loudly and naming the key."""
    value = block.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            return float(str(value))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{key} must be a number, got {value!r}") from exc
    return float(value)


def harvest_all(
    config: dict[str, Any],
    start_year: int | None = None,
    seed_identifiers: Iterable[str] = (),
    seed_titles: Iterable[str] = (),
) -> dict[str, Any]:
    """Run the full three-regime harvest and return the corpus document.

    ``start_year`` applies the dynamic recency window as an arXiv
    ``submittedDate`` clause.  ``seed_identifiers`` and ``seed_titles``
    pre-register seeding papers so a seed that is itself on arXiv is never
    fetched a second time, whether its filename yielded an id or not.
    """
    corpus_cfg = config.get("corpus", {})
    regimes: list[dict[str, Any]] = config.get("regimes", [])
    if not regimes:
        raise ValueError("config.yaml defines no regimes")

    target_per_regime = _config_int(corpus_cfg, "target_per_regime", 35)
    max_query_results = _config_int(corpus_cfg, "max_query_results", 400)
    exclude_keywords = [str(k) for k in corpus_cfg.get("exclude_review_keywords", [])]
    abstract_patterns = [
        str(p) for p in corpus_cfg.get("exclude_review_abstract_patterns", [])
    ]
    min_abstract_words = _config_int(corpus_cfg, "min_abstract_words", 40)
    delay_seconds = _config_float(corpus_cfg, "delay_seconds", 3.0)
    num_retries = _config_int(corpus_cfg, "num_retries", 5)

    window = date_clause(start_year) if start_year is not None else ""
    if window:
        LOGGER.info("Recency window applied to every regime query: %s", window)

    client = build_client(corpus_cfg)
    seen_ids: set[str] = {arxiv_base_id(str(item)) for item in seed_identifiers if item}
    seen_titles: set[str] = {
        key for key in (normalize_title(str(t)) for t in seed_titles if t) if key
    }
    if seen_ids or seen_titles:
        LOGGER.info(
            "Pre-registering %d seeding arXiv IDs and %d titles so none is "
            "harvested twice",
            len(seen_ids),
            len(seen_titles),
        )
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
                date_range=window,
            )
        )

    counts = {str(r["label"]): 0 for r in regimes}
    for paper in papers:
        counts[str(paper["regime_label"])] += 1

    return {
        "metadata": {
            "generated_by": "src.harvest",
            "source": "arXiv API via arxiv python client",
            "n_papers": len(papers),
            "regime_counts": counts,
            "harvest_start_year": start_year,
            "date_clause": window or None,
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


def verify_corpus(
    corpus: dict[str, Any],
    target_per_regime: int,
    single_field: bool = False,
    target_total: int | None = None,
) -> bool:
    """Validate regime balance and the required schema fields.

    Seed-First mode checks a per-regime harvest quota.  Zero-Seed Mode cannot:
    its regions are discovered from the chart, so their sizes are whatever the
    partition produced, and the only quota that means anything is the corpus
    total.  Uniqueness, schema, and document labelling are checked in both.
    """
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
    if single_field and target_total is not None and len(papers) != target_total:
        LOGGER.error(
            "Zero-Seed Mode expects %d documents, corpus has %d",
            target_total,
            len(papers),
        )
        ok = False

    for label, items in sorted(regimes.items()):
        ids = {item["id"] for item in items}
        titles = {normalize_title(item["title"]) for item in items}
        harvested = sum(1 for item in items if not item.get("is_seed"))
        seeds = len(items) - harvested
        # Seeds join whichever regime they were assigned to, so a regime holds
        # its harvest quota plus its seeds.  Only a shortfall is an error.  In
        # Zero-Seed Mode there is no quota to compare against: the region sizes
        # come from the discovered partition, not from the config.
        if single_field:
            status = "OK"
        else:
            status = "OK" if harvested == target_per_regime else "SHORT"
        LOGGER.info(
            "  %-10s %3d papers = %d harvested + %d seeds "
            "(unique ids=%d, unique titles=%d) -> %s",
            label,
            len(items),
            harvested,
            seeds,
            len(ids),
            len(titles),
            status,
        )
        if not single_field and harvested != target_per_regime:
            ok = False
        if len(ids) != len(items) or len(titles) != len(items):
            LOGGER.error("  %s contains duplicate IDs or titles", label)
            ok = False

    doc_labels = [str(paper.get("doc")) for paper in papers]
    if any(label for label in doc_labels):
        expected = [f"doc {position}" for position in range(1, len(papers) + 1)]
        if doc_labels != expected:
            LOGGER.error("Document labels are not sequential from 'doc 1'")
            ok = False

    seeded = sum(1 for paper in papers if paper.get("is_seed"))
    LOGGER.info("  %d seeding papers, %d harvested", seeded, len(papers) - seeded)
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

#: A removal upstream leaves a space in front of punctuation.  Tidying it is part
#: of reaching a fixed point: the callout stripper does this too, but
#: ``normalize_extracted_text`` collapses newlines *after* that, which turns
#: ``(L1-LMS)\n,`` into ``(L1-LMS) ,`` and re-creates the artifact.  Both stages
#: share the pattern so they cannot drift apart.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")
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


def repair_extracted_characters(text: str) -> str:
    """Undo ligatures and end-of-line hyphenation without touching line breaks.

    This runs twice: once before sanitation, because a word split as ``Bom-\nbois``
    hides a citation from the callout stripper and only becomes visible once
    rejoined, and again inside ``normalize_extracted_text`` for callers that start
    from already-sanitised text.
    """
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return _HYPHEN_WRAP_RE.sub("", text)


def normalize_extracted_text(text: str) -> str:
    """Repair the damage a PDF text layer inflicts on running prose."""
    if not text:
        return ""
    text = repair_extracted_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _WHITESPACE_RE.sub(" ", text)
    # Last, so that whitespace collapsed above cannot re-open the gap.
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
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

#: When a PDF has no "Abstract" heading at all (some conference templates jump
#: straight into the intro), only the first few hundred characters are the title
#: block.  Capping the fallback there keeps prose out of the metadata parser.
FRONT_MATTER_FALLBACK_CHARS = 800


def split_front_matter(
    raw: str, max_prefix_chars: int = FRONT_MATTER_LIMIT
) -> tuple[str, str]:
    """Split raw text into ``(body, title_block)`` at the abstract heading.

    The second element is the text that ``strip_front_matter`` removes, and is
    the only place a document states its own title and authors.  When no
    abstract heading exists, the title block falls back to the first
    ``FRONT_MATTER_FALLBACK_CHARS`` characters and the body is returned whole.
    """
    if not raw:
        return raw, ""
    match = _ABSTRACT_HEADING_RE.search(raw, 0, max_prefix_chars)
    if match is None:
        return raw, raw[:FRONT_MATTER_FALLBACK_CHARS].strip()
    body = raw[match.end() :].lstrip("\n\t ")
    return body, raw[: match.start()].strip()


def strip_front_matter(
    raw: str, max_prefix_chars: int = FRONT_MATTER_LIMIT
) -> tuple[str, int]:
    """Drop the title/author/affiliation block preceding the abstract."""
    body, front_matter = split_front_matter(raw, max_prefix_chars)
    if not front_matter:
        return body, 0
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
_BRACKET_NUMERIC_INNER = r"\[\s*\d{1,4}[a-z]?(?:\s*[,;]\s*\d{1,4}[a-z]?)*\s*\]"
_BRACKET_DASH_RANGE_INNER = r"\[\s*\d{1,4}\s*[\u2013\u2014]\s*\d{1,4}\s*\]"
#: ASCII-hyphen ranges are only stripped when they cannot be an interval: a
#: citation range starts at 1 or above and increases, whereas ``[0-1]`` does not.
_BRACKET_HYPHEN_RANGE_INNER = r"\[\s*(?P<low>\d{1,4})\s*-\s*(?P<high>\d{1,4})\s*\]"
_CITATION_BRACKET_NAMED_RE = re.compile(r"\[[A-Z][^\]\n]{0,48}(?:19|20)\d{2}[a-z]?\]")
#: Tokens that mark the preceding bracket as an interval, not a citation.
#: A symbol or operator in front of the bracket is decisive.  The word list is
#: deliberately CASE-SENSITIVE: lowercase "in"/"over" introduces an interval
#: ("x in [0, 1]"), while sentence-initial "In" introduces a citation
#: ("In [12] we show").
_BRACKET_SYMBOL_GUARD_RE = re.compile(
    r"[\u2208\u2282\u2286=< >\u2264\u2265(+\-/]\s*$".replace("< >", "<>")
)
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
    "(?:"
    + _BRACKET_SYMBOL_GUARD_RE.pattern
    + "|"
    + _BRACKET_WORD_GUARD_RE.pattern
    + ")"
)

#: A surname written just before a numeric citation ("see Smith et al. [12]").
#: The lookbehind requires lowercase word end or punctuation first, so a
#: sentence-initial word ("In [12] we ...") is never treated as a surname.
_STRUCTURAL_WORDS = (
    "Section|Table|Figure|Fig|Eq|Equation|Chapter|Algorithm|Appendix|Lemma|Theorem"
    "|Remark|Definition|Proposition|Corollary|Listing"
)
#: The surname travels in the SAME match as the bracket it introduces, and only
#: a numeric citation bracket can follow it.  Both properties are load-bearing
#: for idempotency.  An earlier version matched a surname against ``(?=\[)``
#: alone, i.e. against any bracket at all, so the table row
#: ``Parameters Range Unit [0.5, 2.5]`` lost one capitalised word per pass -
#: "Unit", then "Range", then "Parameters" - because deleting a word re-exposes
#: the word before it.  Re-sanitising cached text therefore eroded the corpus a
#: little further on every rebuild and moved the Pareto front.
_CITATION_LEAD_PATTERN = (
    r"(?P<lead>(?<=[a-z,;]\s)"
    r"(?!" + _STRUCTURAL_WORDS + r"\b)"
    r"[A-Z][A-Za-z'\u2019\-]{2,}"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'\u2019\-]{2,}))?"
    r"[ \t]*)?"
)
_CITATION_LEAD_HYPHEN_RE = re.compile(
    _CITATION_LEAD_PATTERN + r"(?P<bracket>" + _BRACKET_HYPHEN_RANGE_INNER + r")"
)
_CITATION_LEAD_DASH_RE = re.compile(
    _CITATION_LEAD_PATTERN + r"(?P<bracket>" + _BRACKET_DASH_RANGE_INNER + r")"
)
_CITATION_LEAD_NUMERIC_RE = re.compile(
    _CITATION_LEAD_PATTERN + r"(?P<bracket>" + _BRACKET_NUMERIC_INNER + r")"
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
        if (
            6 <= len(stripped) <= 80
            and sum(1 for char in stripped if char.isalpha()) >= 8
        ):
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {line for line, count in counts.items() if count >= repeat_threshold}
    if repeated:
        survivors = [line for line in kept if line.strip() not in repeated]
        removed += len(kept) - len(survivors)
        kept = survivors

    return "\n".join(kept), removed


def strip_citation_callouts(text: str) -> tuple[str, int]:
    """Remove citation callouts and bibliographic locators from body prose.

    Idempotent: a second application is a no-op.  The learner and the bracket
    are removed as one match, so deleting a citation cannot re-expose the word
    before it as a fresh match.  ``tests/test_sanitize.py`` pins this property
    over every cached document, because the sanitizer runs again on cache hits.
    """
    if not text:
        return text, 0
    removed = 0

    def is_interval(match: re.Match[str]) -> bool:
        """True when the bracket is a mathematical interval, not a citation."""
        start = match.start("bracket")
        window = text[max(0, start - 14) : start]
        return bool(_CITATION_BRACKET_GUARD_RE.search(window))

    def drop_brackets(match: re.Match[str]) -> str:
        nonlocal removed
        if is_interval(match):
            return match.group(0)
        removed += 1
        return ""

    def drop_hyphen_range(match: re.Match[str]) -> str:
        nonlocal removed
        low = int(match.group("low"))
        high = int(match.group("high"))
        # A citation range starts at 1 and increases; an interval need not.
        if low < 1 or high <= low or is_interval(match):
            return match.group(0)
        removed += 1
        return ""

    # Each pattern owns a disjoint bracket form, so the order only decides which
    # label a removal is counted under.
    text = _CITATION_LEAD_HYPHEN_RE.sub(drop_hyphen_range, text)
    for pattern in (_CITATION_LEAD_DASH_RE, _CITATION_LEAD_NUMERIC_RE):
        text = pattern.sub(drop_brackets, text)
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
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
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
    #: The title/author block that ``strip_front_matter`` removed.  Kept because
    #: it is the only place a seeding PDF states its own title and authors.
    front_matter_text: str = ""

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

    Order matters: character repair first, because a word hyphenated across a line
    break hides whatever it completes; then front matter and running headers so
    heading detection is not fooled by page furniture, then the bibliography, then
    callouts removed from what remains of the body.
    """
    repaired = repair_extracted_characters(raw)
    without_headers, headers = strip_running_headers(repaired)
    # The title block is read from the repaired text: the running-header pass removes
    # any line that repeats on several pages, and a paper whose title is stamped
    # on every page would otherwise lose its own title.
    _, front_matter_text = split_front_matter(
        repaired, min(FRONT_MATTER_LIMIT, len(repaired))
    )
    body, _ = split_front_matter(without_headers)
    body, references, acknowledgments = strip_bibliography(body, min_body_words)
    body, citations = strip_citation_callouts(body)
    return SanitationResult(
        text=normalize_extracted_text(body),
        removed_front_matter=(
            len(without_headers) - len(body) if front_matter_text else 0
        ),
        removed_references=references,
        removed_acknowledgments=acknowledgments,
        removed_citations=citations,
        removed_headers=headers,
        front_matter_text=front_matter_text,
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
        reasons.append(f"alphanumeric ratio {alnum_ratio:.3f} <= {min_alnum_ratio}")
    return TextQuality(
        words=word_count,
        chars=char_count,
        mean_word_length=mean_word_length,
        alnum_ratio=alnum_ratio,
        reasons=tuple(reasons),
    )


def extract_pdf_raw(pdf_path: Path, timeout: float = 300.0) -> str:
    """Extract with ``mutool`` (MuPDF) and return the text exactly as it came out.

    This is the layer that gets cached: sanitation is a pure function of it, so a
    cache that stores raw text can always be re-sanitized under new rules.  Caching
    the sanitized text instead makes the cache an unfixable record of whichever
    rules were in force when it was written.

    ``errors="replace"`` matters: a PDF text layer is not required to be valid
    UTF-8, and MuPDF passes the font's bytes through unchanged.  Decoding strictly
    made a single stray byte abort the whole extraction, which then reported the
    paper as unreadable rather than as having one bad glyph.
    """
    try:
        result = subprocess.run(
            ["mutool", "draw", "-F", "txt", "-o", "-", str(pdf_path)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "mutool is required for PDF text extraction but was not found on PATH"
        ) from exc
    except subprocess.TimeoutExpired:
        LOGGER.warning("mutool timed out after %.0fs on %s", timeout, pdf_path.name)
        return ""
    if result.returncode != 0:
        LOGGER.debug(
            "mutool exited %d on %s: %s",
            result.returncode,
            pdf_path.name,
            result.stderr.strip()[:300],
        )
    return result.stdout


def extract_pdf_text(
    pdf_path: Path,
    timeout: float = 300.0,
    min_body_words: int = FULLTEXT_MIN_WORDS,
) -> SanitationResult:
    """Extract with ``mutool`` (MuPDF), then sanitize. Empty result on failure."""
    return sanitize_document_text(
        extract_pdf_raw(pdf_path, timeout=timeout), min_body_words
    )


def extract_ar5iv_raw(
    arxiv_id: str, session: requests.Session, timeout: float = 45.0
) -> str:
    """Fallback: pull the arXiv HTML5 rendering (ar5iv) and reduce it to raw text."""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.debug("ar5iv failed for %s: %s", arxiv_id, exc)
        return ""
    try:
        document = lxml_html.fromstring(response.text)
    except (ValueError, TypeError) as exc:
        LOGGER.debug("ar5iv returned unparseable HTML for %s: %s", arxiv_id, exc)
        return ""
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
    return "\n".join(node.text_content() for node in nodes)


def extract_ar5iv_text(
    arxiv_id: str,
    session: requests.Session,
    timeout: float = 45.0,
    min_body_words: int = FULLTEXT_MIN_WORDS,
) -> SanitationResult:
    """Fallback: pull the arXiv HTML5 rendering (ar5iv), then sanitize."""
    return sanitize_document_text(
        extract_ar5iv_raw(arxiv_id, session, timeout=timeout), min_body_words
    )


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
    #: Original PDF filename for seeding papers ('' for harvested papers).
    source_file: str = ""
    #: Publication year inferred from the seed PDF, when one could be found.
    year: int | None = None
    #: Author list recovered from the seed PDF's title block.
    authors: list[str] = field(default_factory=list)


def _seed_identifier(path: Path) -> str:
    """Recover an arXiv ID from a seed filename, or '' when it is a free-form name."""
    match = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)", path.stem)
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- #
# Seed-First Expansion: seed year, dynamic cutoff, regime assignment
# --------------------------------------------------------------------------- #

#: Years considered plausible publication years when scanning seed text.
EARLIEST_PLAUSIBLE_YEAR = 1980

#: How much of a seed's text to scan for its publication year and abstract.
SEED_HEAD_CHARS = 5000
SEED_ABSTRACT_CHARS = 3000
SEED_ABSTRACT_WORDS = 200

_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-5]\d)\b")
_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
#: The stamp arXiv prints in the margin of every preprint: ``arXiv:2103.07945v3``.
_ARXIV_STAMP_RE = re.compile(r"arXiv:(\d{2})(\d{2})\.\d{4,5}")
#: Filename of the title-lookup cache kept beside the extracted full text.
SEED_ARXIV_CACHE = "seed_arxiv_ids.json"

#: A seed's abstract ends at one of these headings (Sec. 5.2 consumes full text,
#: but the corpus schema wants an abstract too, so it is recovered from the body).
_ABSTRACT_END_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:\d{1,2}|[IVX]{1,4})\.?[ \t]+)?"
    r"(?:introduction|index terms|keywords|ccs concepts|i\.[ \t]+introduction)\b"
)


def infer_seed_year(
    identifier: str,
    text: str,
    filename: str = "",
    front_matter: str = "",
    current_year: int | None = None,
) -> tuple[int | None, str]:
    """Infer a seeding paper's publication year, most reliable source first.

    Order: the arXiv identifier when the filename carries one (``YYMM.NNNNN``),
    then an arXiv stamp printed in the front matter, then the modal year in the
    front matter, then a year embedded in the filename, and only last the modal
    year of the body text.  The body is the weakest source because a paper
    mentions the years of the work it builds on.

    Returns ``(year, source)`` so the caller can log which rule fired; ``None``
    is honest, and the caller then excludes the seed from the cutoff.
    """
    now = current_year or datetime.now().year
    ceiling = now + 1

    def _plausible(values: list[str]) -> list[int]:
        return [
            int(value)
            for value in values
            if EARLIEST_PLAUSIBLE_YEAR <= int(value) <= ceiling
        ]

    def _modal(years: list[int]) -> int:
        # Most frequent wins; ties break to the earliest year, because a later
        # year on the first page is usually a revision or a reprint notice.
        counts = Counter(years)
        return min(counts, key=lambda year: (-counts[year], year))

    match = _ARXIV_ID_RE.match(identifier)
    if match:
        year = 2000 + int(match.group(1))
        month = int(match.group(2))
        if 1 <= month <= 12 and EARLIEST_PLAUSIBLE_YEAR <= year <= ceiling:
            return year, "arxiv-id"

    stamp = _ARXIV_STAMP_RE.search(front_matter)
    if stamp:
        year = 2000 + int(stamp.group(1))
        month = int(stamp.group(2))
        if 1 <= month <= 12 and EARLIEST_PLAUSIBLE_YEAR <= year <= ceiling:
            return year, "arxiv-stamp"

    front_years = _plausible(_YEAR_RE.findall(front_matter))
    if front_years:
        return _modal(front_years), "front-matter"

    name_years = _plausible(_YEAR_RE.findall(filename))
    if name_years:
        return min(name_years), "filename"

    body_years = _plausible(_YEAR_RE.findall(text))
    if body_years:
        return _modal(body_years), "body"
    return None, "unknown"


def seed_abstract(text: str, max_words: int = SEED_ABSTRACT_WORDS) -> str:
    """Recover an abstract-like summary from a seed's sanitised body text."""
    body = text.strip()
    if body[:8].lower() == "abstract":
        body = body[8:].lstrip(" .:-\u2014\u2013")
    window = body[:SEED_ABSTRACT_CHARS]
    match = _ABSTRACT_END_RE.search(window)
    if match and match.start() > 120:
        window = window[: match.start()]
    return " ".join(window.split()[:max_words]).strip()


def seed_matching_text(
    document: ExtractedDocument, limit: int = SEED_HEAD_CHARS
) -> str:
    """The text used to match a seed against regime centroids.

    Title plus abstract is the paper's own statement of what it is about.  The
    full body is dominated by method detail and notation, which is shared across
    regimes: on this corpus the head of the document lifts the top cosine score
    from 0.006-0.059 to 0.030-0.195 and roughly halves the number of draws the
    margin rule flags as ambiguous.
    """
    head = document.text[:limit]
    return head if head else document.text


def harvest_start_year(
    seed_years: Iterable[int | None],
    config: dict[str, Any],
    current_year: int | None = None,
) -> int:
    """Earliest year a harvested paper may carry: ``min(seed_years) - offset``.

    Clamped so the lookback never exceeds ``corpus.lookback_years`` (Obayashi et
    al. 2026, Sec. 5.1), which keeps the window anchored to the present instead
    of following a very old seed arbitrarily far back.
    """
    corpus_cfg = config.get("corpus", {}) or {}
    offset = _config_int(corpus_cfg, "seed_year_offset", 2)
    lookback = _config_int(corpus_cfg, "lookback_years", 10)
    now = current_year or datetime.now().year
    floor = now - lookback

    known = [int(year) for year in seed_years if year is not None]
    if not known:
        LOGGER.warning(
            "No seed year could be inferred; falling back to the %d-year lookback "
            "floor (%d)",
            lookback,
            floor,
        )
        return floor

    earliest = min(known)
    start = max(earliest - offset, floor)
    LOGGER.info(
        "Dynamic cutoff: earliest seed year %d - offset %d = %d, clamped by the "
        "%d-year lookback floor %d -> harvesting from %d",
        earliest,
        offset,
        earliest - offset,
        lookback,
        floor,
        start,
    )
    return start


def date_clause(start_year: int, end_year: int | None = None) -> str:
    """Build the arXiv ``submittedDate`` range appended to every regime query."""
    last = end_year or datetime.now().year
    return f"submittedDate:[{start_year}01010000 TO {last}12312359]"


def regime_reference_texts(config: dict[str, Any]) -> dict[int, str]:
    """Keyword bundle per regime, used only for seed matching, never as a query."""
    domains = config.get("search_domains") or []
    by_name = {
        str(entry.get("name", "")): [str(word) for word in entry.get("keywords", [])]
        for entry in domains
        if isinstance(entry, dict)
    }
    references: dict[int, str] = {}
    for regime in config.get("regimes", []):
        keywords = regime.get("keywords")
        if not keywords:
            keywords = by_name.get(str(regime.get("name", "")), [])
        parts = [str(regime.get("name", ""))]
        if regime.get("description"):
            parts.append(str(regime["description"]))
        parts.extend(str(word) for word in keywords)
        references[int(regime["id"])] = " ".join(parts)
    return references


def _lookup_seed_override(
    overrides: dict[str, Any], filename: str, identifier: str
) -> int | None:
    """Resolve an explicit seed override by filename, stem, or arXiv ID."""
    stem = Path(filename).stem
    for key in (filename, stem, identifier):
        if key and key in overrides:
            try:
                return int(overrides[key])
            except (TypeError, ValueError):
                LOGGER.error(
                    "seed_overrides[%r] = %r is not an integer regime id",
                    key,
                    overrides[key],
                )
                return None
    return None


def assign_seed_regimes(
    seeds: list[ExtractedDocument],
    config: dict[str, Any],
    harvested_documents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assign every seed to a regime: explicit override first, else TF-IDF cosine.

    The fallback centroid for a regime is the mean of its harvested documents'
    TF-IDF vectors plus its keyword bundle, so a seed is matched against what the
    regime actually looks like in this corpus rather than against the wording of
    the query alone.

    Returns one record per seed with ``regime_id``, ``similarity``, ``margin`` and
    ``assigned_by`` (``"override"`` or ``"similarity"``) for auditability.
    """
    corpus_cfg = config.get("corpus", {}) or {}
    overrides = corpus_cfg.get("seed_overrides") or {}
    margin_threshold = float(corpus_cfg.get("seed_match_margin", 0.02))
    regimes = list(config.get("regimes", []))
    if not regimes:
        raise SystemExit("config.yaml defines no regimes")

    references = regime_reference_texts(config)
    regime_ids = [int(regime["id"]) for regime in regimes]
    labels = {int(regime["id"]): str(regime["label"]) for regime in regimes}

    seed_texts = [seed_matching_text(doc) for doc in seeds]
    reference_texts = [references.get(rid, "") for rid in regime_ids]
    harvested = harvested_documents or []
    harvested_texts = [str(item.get("text", "")) for item in harvested]
    harvested_regimes = [int(item.get("regime_id", -1)) for item in harvested]

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1
    )
    matrix = vectorizer.fit_transform([*seed_texts, *reference_texts, *harvested_texts])
    n_seeds = len(seed_texts)
    n_refs = len(reference_texts)

    seed_rows = matrix[0:n_seeds]
    reference_rows = matrix[n_seeds : n_seeds + n_refs]
    harvested_rows = matrix[n_seeds + n_refs :]

    centroids = np.zeros((len(regime_ids), matrix.shape[1]), dtype=float)
    for position, regime_id in enumerate(regime_ids):
        members = [
            index for index, rid in enumerate(harvested_regimes) if rid == regime_id
        ]
        centroid = np.asarray(reference_rows[position].todense()).ravel().astype(float)
        if members:
            member_rows = harvested_rows[members]
            centroid = (
                centroid * 0.25 + np.asarray(member_rows.mean(axis=0)).ravel() * 0.75
            )
        norm = float(np.linalg.norm(centroid))
        centroids[position] = centroid / norm if norm > 0 else centroid

    seed_dense = np.asarray(seed_rows.todense(), dtype=float)
    norms = np.linalg.norm(seed_dense, axis=1, keepdims=True)
    seed_dense = np.divide(
        seed_dense, norms, out=np.zeros_like(seed_dense), where=norms > 0
    )
    scores = seed_dense @ centroids.T

    assignments: list[dict[str, Any]] = []
    for position, document in enumerate(seeds):
        filename = document.source_file or document.identifier
        override = _lookup_seed_override(overrides, filename, document.identifier)
        rank_scores = sorted(scores[position], reverse=True)
        margin = float(rank_scores[0] - rank_scores[1]) if len(rank_scores) > 1 else 0.0
        best = regime_ids[int(np.argmax(scores[position]))]

        if override is not None and override in labels:
            chosen, assigned_by = override, "override"
            LOGGER.info(
                "  seed %-46s -> %-9s (explicit seed_overrides)",
                filename[:46],
                labels[chosen],
            )
        elif override is not None:
            assigned_by = "similarity"
            chosen = best
            LOGGER.error(
                "  seed %s: seed_overrides names regime %d, which config.yaml does "
                "not define; falling back to similarity (%s)",
                filename,
                override,
                labels[chosen],
            )
        else:
            chosen, assigned_by = best, "similarity"
            LOGGER.info(
                "  seed %-46s -> %-9s (cosine %.3f, margin %.3f)%s",
                filename[:46],
                labels[chosen],
                float(scores[position, int(np.argmax(scores[position]))]),
                margin,
                "  [AMBIGUOUS]" if margin < margin_threshold else "",
            )
            if margin < margin_threshold:
                LOGGER.warning(
                    "  seed %s is ambiguous (margin %.4f < %.4f); add it to "
                    "corpus.seed_overrides in config.yaml to pin the regime",
                    filename,
                    margin,
                    margin_threshold,
                )

        assignments.append(
            {
                "regime_id": int(chosen),
                "regime_label": labels[int(chosen)],
                "assigned_by": assigned_by,
                "similarity": round(
                    float(scores[position, regime_ids.index(int(chosen))]), 4
                ),
                "margin": round(margin, 4),
            }
        )
    return assignments


def parse_seed_metadata(front_matter: str) -> tuple[str, list[str]]:
    """Recover a seeding PDF's title and author list from its title block.

    The front matter is the only place a PDF states its own title and authors,
    and it is exactly what ``strip_front_matter`` throws away for the body text.
    Parsing it beats using the filename (``cassie biped``) as a title, which
    would put nonsense labels on the map.

    Titles and author lists are separated by blank lines in every template in
    use, so the block is split into paragraphs rather than lines: the first
    paragraph is the title, and author paragraphs follow until a paragraph
    yields no names (which is where affiliations and prose begin).

    Returns ``("", [])`` when nothing trustworthy is found, so callers can fall
    back to the filename instead of inventing metadata.
    """
    if not front_matter.strip():
        return "", []

    paragraphs: list[str] = []
    for raw_paragraph in re.split(r"\n\s*\n", front_matter):
        if any(line.strip() for line in raw_paragraph.splitlines()):
            paragraphs.append(raw_paragraph.strip("\n"))
    if not paragraphs:
        return "", []

    # Venue notices and arXiv stamps form whole paragraphs of their own
    # ("Published as a conference paper at ICLR 2025"), so they are skipped
    # paragraph-wise before the title is read.
    while paragraphs and all(
        _is_boilerplate_line(line.strip())
        for line in paragraphs[0].splitlines()
        if line.strip()
    ):
        paragraphs.pop(0)
    if not paragraphs:
        return "", []

    title = " ".join(
        line.strip()
        for line in paragraphs[0].splitlines()
        if line.strip() and not _is_boilerplate_line(line.strip())
    ).strip(" .,;:")
    words = title.split()
    # A real title is short, free of e-mail addresses, and carries words.  A
    # long first paragraph means the template has no title block at all.
    if len(words) < 3 or len(words) > 30 or "@" in title:
        return "", []

    authors: list[str] = []
    for paragraph in paragraphs[1:]:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        # Two author templates are in use: one name per line, and
        # "NAME, Affiliation, Country" on every line.  When any line splits
        # cleanly into nothing but names, trust those lines only -- that is the
        # strict reading and it keeps institution names out of the list.
        strict = [
            line
            for line in lines
            if line.split()
            and all(
                _clean_author_name(candidate)
                for candidate in re.split(r",|\band\b|&|\s{2,}", line)
                if candidate.strip()
            )
        ]
        found: list[str] = []
        for line in strict or lines:
            for candidate in re.split(r",|\band\b|&|\s{2,}", line):
                name = _clean_author_name(candidate)
                if name and name not in authors:
                    found.append(name)
        if not found:
            # The first paragraph with no names is the affiliation block; stop
            # there so body prose can never be mistaken for an author list.
            break
        authors.extend(found)
        if len(authors) > 40:
            break
    return title, authors[:40]


#: Publisher / venue furniture that precedes a title, never part of it.
_BOILERPLATE_RE = re.compile(
    r"^(?:arxiv:|published\s+as|preprint\b|under\s+review|accepted\b|proceedings\b"
    r"|copyright\b|\u00a9|doi\b|https?://|www\.|this\s+is\s+a\s+preprint)",
    re.IGNORECASE,
)

#: Tokens that mark a line as an affiliation, an e-mail, or a venue notice.
_AFFILIATION_RE = re.compile(
    r"(?:@|universit|institut|laborator|college|school\s+of|department|dept\.?\b"
    r"|inc\b|ltd\b|corp\b|gmbh\b|research\b|\bai\b|\bnvidia\b|\bmeta\b|\bdeepseek"
    r"|\bopenai\b|\bgoogle\b|\bmicrosoft\b|\bamazon\b|\bapple\b|\bbosch\b"
    r"|academy|centre|center\b|faculty|fellowship|scholarship)",
    re.IGNORECASE,
)

_AUTHOR_NAME_RE = re.compile(
    r"^[A-Z][A-Za-z'\u2019-]*(?:\s+[A-Z][A-Za-z'\u2019.-]*){1,3}$"
)


def _is_boilerplate_line(line: str) -> bool:
    """True for venue/arXiv/page furniture that precedes a title."""
    if _BOILERPLATE_RE.match(line):
        return True
    # A bare page number or a bare email address line.
    return bool(re.fullmatch(r"\d{1,4}", line)) or "@" in line


def _looks_like_author_line(line: str) -> bool:
    """True when a line reads as a comma-separated list of personal names."""
    candidates = [part for part in re.split(r",|\band\b", line) if part.strip()]
    named = [part for part in candidates if _clean_author_name(part)]
    return len(named) >= 2 and len(named) >= len(candidates) - 1


def _looks_like_affiliation_line(line: str) -> bool:
    """True when a line is an affiliation, e-mail, or footnote marker."""
    return bool(_AFFILIATION_RE.search(line)) or line.startswith(("*∗", "*†", "*‡"))


#: Institution and lab names that appear inside author blocks.  A candidate
#: containing one of these is an affiliation, never an author.  This catches the
#: "ETH Zurich" style leak where an affiliation line still looks like two
#: capitalised words.
_INSTITUTION_TOKENS = frozenset(
    {
        "eth",
        "nvidia",
        "mila",
        "meta",
        "fair",
        "google",
        "deepmind",
        "openai",
        "deepseek",
        "microsoft",
        "amazon",
        "apple",
        "bosch",
        "aist",
        "cmu",
        "ucl",
        "mit",
        "epfl",
        "kaist",
        "tsinghua",
        "princeton",
        "carnegie",
        "stanford",
        "berkeley",
        "oxford",
        "cambridge",
    }
)


def _clean_author_name(raw: str) -> str:
    """Strip footnote markers from a candidate author token, or return ''."""
    token = re.sub(r"[\d\*∗†‡§¶,;]", " ", raw)
    token = " ".join(token.split()).strip(" .")
    if not token or _AFFILIATION_RE.search(token):
        return ""
    if {word.lower() for word in token.split()} & _INSTITUTION_TOKENS:
        return ""
    if not _AUTHOR_NAME_RE.match(token):
        return ""
    return token


def normalize_title_key(title: str) -> str:
    """Case-, punctuation-, and space-insensitive key for exact title matching."""
    folded = unicodedata.normalize("NFKD", title)
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def _load_seed_arxiv_cache(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / SEED_ARXIV_CACHE
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_seed_arxiv_cache(cache_dir: Path, cache: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / SEED_ARXIV_CACHE
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def lookup_seed_arxiv(
    title: str,
    session: requests.Session,
    cache: dict[str, Any],
    cache_dir: Path,
    delay_seconds: float = 3.0,
) -> tuple[str | None, int | None]:
    """Resolve a local seed PDF to its arXiv record by exact title match.

    A seed saved under a free-form filename (``TD JEPA.pdf``) carries no
    identifier, so its publication year cannot be derived from the filename and
    its record has no public link.  An arXiv title search recovers both, but only
    an exactly matching title is accepted - a near-miss title would attach the
    wrong paper's year and URL to the seed, which is worse than no link at all.

    Results are cached under ``data/fulltext_cache`` so a rebuild does not repeat
    the query.
    """
    key = normalize_title_key(title)
    if not key:
        return None, None
    cached = cache.get(key)
    if isinstance(cached, dict):
        identifier = cached.get("id")
        year = cached.get("year")
        return (
            str(identifier) if identifier else None,
            int(year) if isinstance(year, int) else None,
        )

    identifier: str | None = None
    year: int | None = None
    try:
        client = arxiv.Client(page_size=5, delay_seconds=delay_seconds, num_retries=3)
        search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
        for result in client.results(search):
            if normalize_title_key(result.title) != key:
                continue
            identifier = result.get_short_id()
            year = int(result.published.year)
            break
    except Exception as exc:  # noqa: BLE001 - the arXiv client raises broadly
        LOGGER.warning("arXiv title lookup failed for %r: %s", title, exc)
        return None, None

    cache[key] = {"title": title, "id": identifier, "year": year}
    _save_seed_arxiv_cache(cache_dir, cache)
    if identifier:
        LOGGER.info("  seed %r resolved to arXiv %s (%s)", title[:52], identifier, year)
    else:
        LOGGER.info("  seed %r has no exact arXiv title match", title[:52])
    return identifier, year


def extract_seed_documents(
    desired_dir: Path,
    session: requests.Session,
    delay_seconds: float = 3.0,
    gate: QualityGate | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[ExtractedDocument], list[Path]]:
    """Extract every seeding paper, falling back to ar5iv before giving up.

    A seed whose filename carries no arXiv identifier is resolved against arXiv
    by exact title, which supplies the publication year and a working link when
    the PDF itself prints neither.

    Returns the usable documents and the list of PDFs that could not be
    recovered automatically, so the caller can name them for the user.
    """
    quality_gate = gate or QualityGate()
    arxiv_cache = _load_seed_arxiv_cache(cache_dir) if cache_dir else {}
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

        parsed_title, parsed_authors = parse_seed_metadata(result.front_matter_text)
        seed_year, year_source = infer_seed_year(
            identifier,
            result.text,
            pdf_path.name,
            front_matter=result.front_matter_text,
        )

        # A free-form filename leaves the seed with no identifier and often no
        # year.  Look it up on arXiv by exact title; the cache keeps rebuilds
        # offline.
        if cache_dir and not _seed_identifier(pdf_path):
            resolved_id, resolved_year = lookup_seed_arxiv(
                parsed_title or pdf_path.stem,
                session,
                arxiv_cache,
                cache_dir,
                delay_seconds,
            )
            if resolved_id:
                identifier = resolved_id
                attempts.append("arxiv-title-match")
                if seed_year is None and resolved_year is not None:
                    seed_year, year_source = resolved_year, "arxiv-title-match"

        document = ExtractedDocument(
            identifier=identifier,
            title=parsed_title or pdf_path.stem,
            text=result.text if quality.ok else "",
            source=source if quality.ok else "failed",
            quality=quality,
            attempts=attempts,
            sanitation=result.as_dict(),
            source_file=pdf_path.name,
            year=seed_year,
            authors=parsed_authors,
        )
        documents.append(document)
        if quality.ok:
            LOGGER.info(
                "  seed %-46s %6d words via %-9s refs=-%d cites=%d year=%s (%s)",
                pdf_path.name[:46],
                quality.words,
                source,
                result.removed_references + result.removed_acknowledgments,
                result.removed_citations,
                seed_year if seed_year is not None else "unknown",
                year_source,
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
        raw_text: str | None = None

        # The cache holds the *raw* extraction, so sanitation runs exactly once per
        # build and the text can never drift across rebuilds.  Caching sanitized
        # text instead made the cache an unfixable record of the rules that happened
        # to be in force when it was written.
        cached = cache_dir / f"{arxiv_id}.raw.txt"
        if cached.exists():
            raw_text = cached.read_text(encoding="utf-8")
            result = sanitize_document_text(raw_text, quality_gate.min_words)
            quality = quality_gate.assess(result.text)
            attempts.append("cache")
            source = "cache"
            if not quality.ok:
                LOGGER.warning(
                    "Cached text for %s fails the audit; re-extracting", arxiv_id
                )
                cached.unlink()
                raw_text = None
                result = SanitationResult(text="")
                quality = quality_gate.assess("")
                source = ""

        if not quality.ok:
            pdf_path = download_arxiv_pdf(
                arxiv_id, cache_dir / "pdf", session, delay_seconds=delay_seconds
            )
            if pdf_path is not None:
                raw_text = extract_pdf_raw(pdf_path)
                result = sanitize_document_text(raw_text, quality_gate.min_words)
                quality = quality_gate.assess(result.text)
                attempts.append(f"pdf:{'ok' if quality.ok else 'rejected'}")
                source = "arxiv-pdf"

        if not quality.ok:
            raw_text = extract_ar5iv_raw(arxiv_id, session)
            fallback = sanitize_document_text(raw_text, quality_gate.min_words)
            fallback_quality = quality_gate.assess(fallback.text)
            attempts.append(f"ar5iv:{'ok' if fallback_quality.ok else 'rejected'}")
            if fallback_quality.ok:
                result, quality, source = fallback, fallback_quality, "ar5iv"

        if quality.ok:
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Store what the extractor produced, not what sanitation made of it.
            cached.write_text(raw_text or "", encoding="utf-8")
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


def _text_fingerprint(papers: list[dict[str, Any]]) -> dict[str, Any]:
    """Digest the sanitized text so a text-level drift shows up in ``corpus.json``.

    Without this, a rebuild that changes the text but leaves the paper list intact
    is invisible at the corpus level: the first symptom is a moved Pareto front,
    which is exactly the symptom that is hardest to attribute.
    """
    digest = hashlib.new("md5", usedforsecurity=False)
    characters = 0
    for paper in papers:
        text = str(paper.get("full_text", ""))
        characters += len(text)
        digest.update(str(paper.get("id", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return {
        "algorithm": "md5",
        "digest": digest.hexdigest(),
        "characters": characters,
    }


def _sanitation_totals(documents: list[ExtractedDocument]) -> dict[str, int]:
    """Sum what sanitation removed across a set of documents."""
    totals: dict[str, int] = {}
    for document in documents:
        for key, value in document.sanitation.items():
            totals[key] = totals.get(key, 0) + int(value)
    return {f"removed_{key}": value for key, value in sorted(totals.items())}


def _seed_filename_slug(name: str) -> str:
    """Stable, path-safe identifier for a seed that has no arXiv ID."""
    stem = Path(name).stem
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"seed:{slug or 'unnamed'}"


def _seed_record(
    document: ExtractedDocument,
    position: int,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Build one corpus entry for a seeding paper (``doc 1``..``doc N``)."""
    # The identifier may be an arXiv id even when the filename was free-form,
    # because extract_seed_documents resolves those by exact title.  Test the
    # identifier itself rather than the filename.
    has_arxiv_id = bool(_ARXIV_ID_RE.match(document.identifier))
    identifier = (
        document.identifier
        if has_arxiv_id
        else _seed_filename_slug(document.source_file or document.identifier)
    )
    return {
        "id": identifier,
        "doc": f"doc {position}",
        "is_seed": True,
        "title": document.title,
        "abstract": seed_abstract(document.text),
        "authors": list(document.authors),
        "year": document.year,
        "regime_id": int(assignment["regime_id"]),
        "regime_label": str(assignment["regime_label"]),
        "full_text": document.text,
        # --- provenance
        "source_file": document.source_file,
        "seed_assignment": {
            "assigned_by": assignment["assigned_by"],
            "similarity": assignment["similarity"],
            "margin": assignment["margin"],
        },
        "extraction": {
            "source": document.source,
            "attempts": document.attempts,
            **document.quality.as_dict(),
        },
        "sanitation": document.sanitation,
    }


def _harvested_record(
    paper: dict[str, Any],
    document: ExtractedDocument,
    position: int,
) -> dict[str, Any]:
    """Build one corpus entry for a harvested paper (``doc 16``..``doc 120``)."""
    record = {
        "id": str(paper["id"]),
        "doc": f"doc {position}",
        "is_seed": False,
        "title": str(paper["title"]),
        "abstract": str(paper["abstract"]),
        "authors": list(paper["authors"]),
        "year": int(paper["year"]),
        "regime_id": int(paper["regime_id"]),
        "regime_label": str(paper["regime_label"]),
        "full_text": document.text,
        "extraction": {
            "source": document.source,
            "attempts": document.attempts,
            **document.quality.as_dict(),
        },
        "sanitation": document.sanitation,
    }
    for key in (
        "regime_name",
        "entry_id",
        "published",
        "updated",
        "primary_category",
        "categories",
        "doi",
        "journal_ref",
        "query",
    ):
        if key in paper:
            record[key] = paper[key]
    return record


def _config_path(config_dir: Path, value: Any, default: str) -> Path:
    """Resolve a config path against the config file's own directory.

    Every other stage resolves artifacts through ``src.common.resolve_paths``,
    which anchors relative paths at the config file.  Harvest used to read them
    relative to the current working directory instead, so the same config
    produced different files depending on where it was invoked from — and a
    second config kept beside the first wrote into the repository's ``data/``
    rather than beside itself.  Anchoring here restores the one rule.
    """
    path = Path(str(value if value is not None else default)).expanduser()
    return path if path.is_absolute() else Path(config_dir) / path


def build_fulltext_corpus(
    config: dict[str, Any], config_dir: Path = Path(".")
) -> dict[str, Any]:
    """Seed-First Expansion: build the whole corpus in one pass.

    Order is load-bearing.  Seeds are extracted first because their publication
    years decide the harvest window, and they are registered as already-seen IDs
    so a seed that is itself on arXiv cannot be harvested a second time:

    1. Extract and sanitise the seeding PDFs       -> ``doc 1``..``doc N``
    2. Derive the recency cutoff from their years  -> ``min(seed_years) - 2``
    3. Harvest ``target_per_regime`` papers per regime under that cutoff
    4. Extract full text for the harvested papers  -> ``doc N+1``..``doc 120``
    5. Assign each seed a regime (override, else TF-IDF cosine fallback)
    6. Write ``data/corpus.json`` and ``data/corpus_fulltext.json``

    The result contains only seeds plus this run's harvest: a rebuild never
    inherits papers from an earlier corpus.
    """
    fulltext_cfg = config.get("fulltext", {}) or {}
    corpus_cfg = config.get("corpus", {}) or {}
    corpus_path = _config_path(
        config_dir, corpus_cfg.get("output_path"), "data/corpus.json"
    )
    out_path = _config_path(
        config_dir, fulltext_cfg.get("output_path"), "data/corpus_fulltext.json"
    )
    desired_dir = _config_path(
        config_dir, fulltext_cfg.get("seeds_dir"), "docs/papers/desired"
    )
    cache_dir = _config_path(
        config_dir, fulltext_cfg.get("cache_dir"), "data/fulltext_cache"
    )
    delay = _config_float(fulltext_cfg, "delay_seconds", 3.0)
    gate = QualityGate.from_config(fulltext_cfg)
    target_total = _config_int(corpus_cfg, "target_total_papers", 120)

    session = requests.Session()
    session.headers.update({"User-Agent": FULLTEXT_USER_AGENT})

    # --- 1. seeds -----------------------------------------------------------
    single_field = is_single_field(config)
    if single_field:
        # Zero-Seed Mode: there is no seed directory to read, so the harvest
        # window is the fixed lookback in the profile rather than a year derived
        # from seed publication dates.
        seeds = []
        seed_failures = []
        LOGGER.info(
            "Step 1/5: Zero-Seed Mode, no seeding papers (single query, %d documents)",
            target_total,
        )
    else:
        LOGGER.info("Step 1/5: extracting seeding papers from %s", desired_dir)
        seeds, seed_failures = extract_seed_documents(
            desired_dir, session, delay, gate, cache_dir=cache_dir
        )
        if seed_failures:
            names = ", ".join(path.name for path in seed_failures)
            raise SystemExit(
                "These seeding PDFs could not be extracted and are required by "
                f"Seed-First Expansion: {names}"
            )
        if not seeds:
            raise SystemExit(f"No seeding PDFs found under {desired_dir}")
        LOGGER.info(
            "Seeds extracted: %d documents, years %s",
            len(seeds),
            sorted({doc.year for doc in seeds if doc.year is not None}),
        )

    # --- 2. dynamic cutoff --------------------------------------------------
    start_year = harvest_start_year((doc.year for doc in seeds), config)

    # --- 3. harvest ---------------------------------------------------------
    per_regime = _config_int(corpus_cfg, "target_per_regime", 35)
    LOGGER.info(
        "Step 2/5: harvesting %d papers per regime (window from %d)",
        per_regime,
        start_year,
    )
    harvested = harvest_all(
        config,
        start_year=start_year,
        seed_identifiers=(doc.identifier for doc in seeds),
        seed_titles=(doc.title for doc in seeds),
    )
    harvested_papers = harvested["papers"]
    if not harvested_papers:
        raise SystemExit("Harvest returned no papers; check network and queries")

    # --- 4. full text for the harvest ---------------------------------------
    LOGGER.info(
        "Step 3/5: extracting full text for %d harvested papers",
        len(harvested_papers),
    )
    corpus_docs, corpus_failures = extract_corpus_fulltext(
        {"papers": harvested_papers}, cache_dir, session, delay, gate
    )
    if corpus_failures:
        raise SystemExit(
            "Full text extraction failed for harvested papers: "
            + ", ".join(corpus_failures)
        )

    # --- 5. seed regime assignment -----------------------------------------
    if seeds:
        LOGGER.info("Step 4/5: assigning each seed to a regime")
        assignments = assign_seed_regimes(
            seeds,
            config,
            [
                {"text": doc.text, "regime_id": paper["regime_id"]}
                for doc, paper in zip(corpus_docs, harvested_papers, strict=True)
            ],
        )
    else:
        LOGGER.info("Step 4/5: no seeds to assign (Zero-Seed Mode)")
        assignments = []

    # --- 6. unify -----------------------------------------------------------
    papers: list[dict[str, Any]] = []
    for position, (document, assignment) in enumerate(
        zip(seeds, assignments, strict=True), start=1
    ):
        papers.append(_seed_record(document, position, assignment))
    offset = len(papers)
    for index, (paper, document) in enumerate(
        zip(harvested_papers, corpus_docs, strict=True), start=1
    ):
        papers.append(_harvested_record(paper, document, offset + index))

    if len(papers) != target_total:
        # A short corpus must never be written: overwriting a complete corpus with a
        # partial one is silent data loss, and the only way it happens is a harvest
        # that came back short (arXiv answers 503 under load and returns nothing for
        # the regime being asked).  The existing corpus files stay untouched.
        raise SystemExit(
            f"Harvest produced {len(harvested_papers)} papers for "
            f"{len(seeds)} seeds, giving {len(papers)} documents where the config "
            f"wants {target_total}. Refusing to overwrite the existing corpus. "
            f"arXiv was probably rate limiting; re-run when it recovers."
        )

    counts: dict[str, int] = {}
    seed_counts: dict[str, int] = {}
    for paper in papers:
        label = str(paper["regime_label"])
        counts[label] = counts.get(label, 0) + 1
        if paper["is_seed"]:
            seed_counts[label] = seed_counts.get(label, 0) + 1

    sanitation_totals = _sanitation_totals([*seeds, *corpus_docs])

    metadata = {
        "generated_by": (
            "src.harvest (Single-Field Exploration, Zero-Seed Mode)"
            if single_field
            else "src.harvest (Seed-First Expansion, Phase 1)"
        ),
        "profile": "single_field" if single_field else "seed_first",
        "reference": (
            "Obayashi et al. (2026) Knowledge Manifold Optimization, "
            "Sec. 5.1-5.2 and Appendix A.1"
        ),
        "n_papers": len(papers),
        "n_seeds": len(seeds),
        "n_harvested": len(harvested_papers),
        "regime_counts": counts,
        "seed_regime_counts": seed_counts,
        "harvest_start_year": start_year,
        "date_clause": harvested["metadata"].get("date_clause"),
        "seed_years": {doc.source_file: doc.year for doc in seeds},
        "seeds_dir": str(desired_dir),
        "corpus_path": str(corpus_path),
        "extractor": "mutool (MuPDF) with ar5iv HTML fallback",
        "fulltext_cache": {
            "stores": "raw extraction (sanitation runs once per build)",
            "dir": str(cache_dir),
        },
        "text_fingerprint": _text_fingerprint(papers),
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
            **sanitation_totals,
        },
        "queries": harvested["metadata"].get("queries"),
        "failed_papers": [],
        "failed_seeds": [],
    }
    if single_field:
        metadata["lookback_years"] = _config_int(corpus_cfg, "lookback_years", 5)
        metadata["regimes_source"] = "auto-discovered by src.manifold after projection"

    payload = {"metadata": metadata, "papers": papers}

    # corpus.json carries metadata only; corpus_fulltext.json adds the text.
    corpus_payload = {
        "metadata": {
            **metadata,
            "generated_by": (
                "src.harvest (Single-Field)"
                if single_field
                else "src.harvest (Seed-First)"
            ),
        },
        "papers": [
            {key: value for key, value in paper.items() if key != "full_text"}
            for paper in papers
        ],
    }
    write_corpus(corpus_payload, corpus_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(out_path)

    LOGGER.info(
        "Step 5/5: wrote %d documents (%d seeds + %d harvested) to %s",
        len(papers),
        len(seeds),
        len(harvested_papers),
        out_path,
    )
    return payload


def refresh_fulltext_from_cache(
    config: dict[str, Any], config_dir: Path = Path(".")
) -> dict[str, Any]:
    """Re-sanitise an existing corpus from the raw cache, without touching arXiv.

    Sanitation is a pure function of the raw extraction, and the cache holds that
    extraction, so a change to the sanitiser can be applied offline.  Requiring a
    fresh harvest to re-run a text rule would put a network outage between the rule
    and the corpus it governs, and would silently change the paper list as well.

    Only ``full_text``, ``extraction`` and ``sanitation`` are recomputed; every
    other field (doc order, regime assignment, seed flags, provenance) is preserved
    exactly as it stands, so a text-rule change cannot perturb the corpus itself.
    """
    fulltext_cfg = config.get("fulltext", {}) or {}
    corpus_cfg = config.get("corpus", {}) or {}
    fulltext_path = _config_path(
        config_dir, fulltext_cfg.get("output_path"), "data/corpus_fulltext.json"
    )
    desired_dir = _config_path(
        config_dir, fulltext_cfg.get("seeds_dir"), "docs/papers/desired"
    )
    cache_dir = _config_path(
        config_dir, fulltext_cfg.get("cache_dir"), "data/fulltext_cache"
    )
    gate = QualityGate.from_config(fulltext_cfg)

    payload = load_corpus(fulltext_path)
    papers = payload.get("papers", [])
    if not papers:
        raise SystemExit(f"No papers to refresh in {fulltext_path}")

    refreshed = 0
    missing: list[str] = []
    for paper in papers:
        arxiv_id = str(paper["id"])
        raw_text = ""
        raw_source = ""

        if paper.get("is_seed"):
            source_file = str(paper.get("source_file") or "")
            pdf_path = desired_dir / source_file
            if source_file and pdf_path.is_file():
                raw_text = extract_pdf_raw(pdf_path)
                raw_source = "seed-pdf"
        else:
            cached = cache_dir / f"{arxiv_id}.raw.txt"
            if cached.is_file():
                raw_text = cached.read_text(encoding="utf-8")
                raw_source = "cache"

        if not raw_text.strip():
            missing.append(arxiv_id)
            continue

        result = sanitize_document_text(raw_text, gate.min_words)
        quality = gate.assess(result.text)
        if not quality.ok:
            LOGGER.error(
                "Refreshed text for %s fails the audit: %s",
                arxiv_id,
                "; ".join(quality.reasons),
            )
            missing.append(arxiv_id)
            continue

        paper["full_text"] = result.text
        paper["extraction"] = {
            "source": raw_source,
            "attempts": [
                *(paper.get("extraction", {}).get("attempts") or []),
                "refresh",
            ],
            **quality.as_dict(),
        }
        paper["sanitation"] = result.as_dict()
        refreshed += 1

    if missing:
        raise SystemExit(
            f"Cannot refresh text for {len(missing)} paper(s) with no raw "
            f"source: {', '.join(missing)}"
        )

    metadata = payload.setdefault("metadata", {})
    metadata["text_fingerprint"] = _text_fingerprint(papers)
    metadata["fulltext_cache"] = {
        "stores": "raw extraction (sanitation runs once per build)",
        "dir": str(cache_dir),
    }
    metadata["sanitation"] = {
        **(metadata.get("sanitation") or {}),
        "refreshed_from_cache": refreshed,
        "refreshed_at_rule": "raw cache + current sanitizer, no network",
    }

    write_corpus(payload, fulltext_path)

    # corpus.json carries metadata only; keep its fingerprint in step.
    corpus_path = _config_path(
        config_dir, corpus_cfg.get("output_path"), "data/corpus.json"
    )
    if corpus_path.is_file():
        corpus_payload = load_corpus(corpus_path)
        corpus_payload.setdefault("metadata", {})["text_fingerprint"] = metadata[
            "text_fingerprint"
        ]
        corpus_payload["metadata"]["fulltext_cache"] = metadata["fulltext_cache"]
        write_corpus(corpus_payload, corpus_path)

    LOGGER.info("Refreshed %d document(s) from the raw cache", refreshed)
    return payload


def verify_fulltext(
    path: Path,
    expected_papers: int,
    gate: QualityGate | None = None,
    expected_seeds: int | None = None,
) -> bool:
    """Validate the Seed-First corpus: schema, seeding, quality, and coverage."""
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
        "doc",
        "is_seed",
        "title",
        "abstract",
        "full_text",
        "authors",
        "year",
        "regime_id",
        "regime_label",
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
    seeded = 0
    shortest = 10**9
    remnants = 0
    for position, paper in enumerate(papers, start=1):
        missing = [name for name in required if name not in paper]
        if missing:
            LOGGER.error("Paper %r missing fields %s", paper.get("id"), missing)
            ok = False
            continue
        if paper["id"] in ids:
            LOGGER.error("Duplicate paper id %r", paper["id"])
            ok = False
        ids.add(paper["id"])
        if paper["doc"] != f"doc {position}":
            LOGGER.error(
                "Paper %r carries %r at position %d; doc ids must be sequential",
                paper["id"],
                paper["doc"],
                position,
            )
            ok = False
        if not isinstance(paper["is_seed"], bool):
            LOGGER.error("Paper %r has non-boolean 'is_seed'", paper["id"])
            ok = False
        if paper["is_seed"]:
            seeded += 1
            if not paper["authors"]:
                LOGGER.error("Seed %r has no recoverable authors", paper["id"])
                ok = False
            if not isinstance(paper["year"], int):
                LOGGER.error("Seed %r has no inferred publication year", paper["id"])
                ok = False
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

    if expected_seeds is not None and seeded != expected_seeds:
        LOGGER.error("Expected %d seeding papers, found %d", expected_seeds, seeded)
        ok = False

    # Seeds must come first so doc ids match the reading order of the corpus.
    first_harvest = next(
        (index for index, paper in enumerate(papers) if not paper.get("is_seed")),
        len(papers),
    )
    if any(paper.get("is_seed") for paper in papers[first_harvest:]):
        LOGGER.error("Seeding papers are not contiguous at the start of the corpus")
        ok = False

    LOGGER.info(
        "Full text: %d papers (%d seeds + %d harvested), %d unique ids, "
        "shortest=%d words",
        len(papers),
        seeded,
        len(papers) - seeded,
        len(ids),
        shortest if papers else 0,
    )
    LOGGER.info("Bibliographic remnants surviving sanitation: %d", remnants)
    for label, count in sorted(regimes.items()):
        LOGGER.info("  %-10s %3d papers", label, count)
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harvest the ResearchRadar arXiv corpus (seed-first, or a single "
            "broad field in Zero-Seed Mode)."
        )
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
    parser.add_argument(
        "--refresh-fulltext",
        action="store_true",
        help=(
            "Re-sanitise every document from the raw full-text cache without "
            "contacting arXiv, keeping the paper list unchanged"
        ),
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
        config = load_config(args.config)
    except SystemExit as exc:
        LOGGER.error("%s", exc)
        return 2

    corpus_cfg = config.get("corpus", {})
    config_dir = args.config.parent
    out_path = (
        Path(args.out)
        if args.out
        else _config_path(config_dir, corpus_cfg.get("output_path"), "data/corpus.json")
    )
    target_per_regime = int(corpus_cfg.get("target_per_regime", 35))
    target_total = int(corpus_cfg.get("target_total_papers", 120))
    seed_count = int(corpus_cfg.get("seed_count", 15))

    fulltext_cfg = config.get("fulltext", {}) or {}
    fulltext_path = _config_path(
        config_dir, fulltext_cfg.get("output_path"), "data/corpus_fulltext.json"
    )

    if args.verify_fulltext:
        return (
            0
            if verify_fulltext(
                fulltext_path,
                target_total,
                QualityGate.from_config(fulltext_cfg),
                expected_seeds=seed_count,
            )
            else 1
        )

    if args.verify_only:
        corpus = load_corpus(out_path)
        return (
            0
            if verify_corpus(
                corpus,
                target_per_regime,
                single_field=is_single_field(config),
                target_total=target_total,
            )
            else 1
        )

    if args.refresh_fulltext:
        payload = refresh_fulltext_from_cache(config, config_dir=config_dir)
        return (
            0
            if verify_fulltext(
                fulltext_path,
                len(payload["papers"]),
                QualityGate.from_config(fulltext_cfg),
                expected_seeds=int(payload["metadata"]["n_seeds"]),
            )
            else 1
        )

    if args.extract_text:
        payload = build_fulltext_corpus(config, config_dir=config_dir)
        return (
            0
            if verify_fulltext(
                fulltext_path,
                len(payload["papers"]),
                QualityGate.from_config(fulltext_cfg),
                expected_seeds=int(payload["metadata"]["n_seeds"]),
            )
            else 1
        )

    # Metadata-only harvest (no full text).  Kept for query debugging; the
    # Seed-First rebuild is --extract-text, which produces both artifacts.
    corpus = harvest_all(config)
    write_corpus(corpus, out_path)
    return (
        0
        if verify_corpus(
            corpus,
            target_per_regime,
            single_field=is_single_field(config),
            target_total=target_total,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
