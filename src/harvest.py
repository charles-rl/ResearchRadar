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
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import arxiv
import yaml

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
        missing = [field for field in REQUIRED_FIELDS if field not in paper]
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

    if args.verify_only:
        corpus = load_corpus(out_path)
        return 0 if verify_corpus(corpus, target_per_regime) else 1

    corpus = harvest_all(config)
    write_corpus(corpus, out_path)
    return 0 if verify_corpus(corpus, target_per_regime) else 1


if __name__ == "__main__":
    raise SystemExit(main())
