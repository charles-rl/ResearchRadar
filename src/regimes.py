"""Automatic multi-resolution regimes for Single-Field Exploration mode.

When a corpus is harvested from one broad query there is no manual taxonomy to
group it by, so the grouping is discovered instead.  Two resolutions are named,
following Obayashi et al. (2026) Sec. 3.2, which treats ``K`` as a resolution
on a single hierarchy rather than as a set of competing answers:

* **Discovery scale, K=5** - the partition the map is labelled and coloured by.
  Five regions is the coarsest resolution at which the persistent-boundary
  measure still varies across the chart, and it is the scale the paper uses for
  the headline boundary view ``B_5,g``.
* **Reference scale, K=8** - the finer partition behind ``B_8,g`` and
  ``B_8,a``.  It is reported for cross-checking, never used for colouring,
  because eight hues on one chart stop being readable.

Two different spaces are partitioned on purpose, and the distinction matters:

* the **objective** partition is computed on the TF-IDF representation by
  ``src.manifold.cluster_labels``, because Eq. 21's boundary strength is a
  statement about semantic mixture.  Clustering the 2D chart would measure the
  projection's artefacts as much as the corpus.
* the **label** partition here is computed on the 2D chart coordinates, because
  its only job is to be read by a human.  Regions of the map are then contiguous
  patches of colour, which is what makes the legend legible; a semantic cluster
  drawn on a lossy 2D projection can interleave into confetti.

Both partitions are then named from their own top TF-IDF terms, so the labels
describe what the cluster contains rather than where it sits on the chart.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

LOGGER = logging.getLogger("research_radar")

#: Sec. 3.2 resolutions.  Discovery is the labelling scale, Reference the check.
DEFAULT_DISCOVERY_K = 5
DEFAULT_REFERENCE_K = 8

#: Longest n-gram a label may use.  Single words name a region badly (``planning``
#: alone says little) and a five-word fragment is no longer a phrase.
MAX_NGRAM = 4

#: Labels join at most this many phrases.  Two distinguishes five regions and
#: still fits one row of the legend.
LABEL_TERMS = 2

#: A label phrase must appear in at least this many documents of its cluster, and
#: in at least this share of them.  Without a floor a two-document pet phrase wins
#: the enrichment ratio outright and the region is named after one paper.
MIN_TERM_CLUSTER_DOCUMENTS = 3
MIN_TERM_CLUSTER_SHARE = 0.10

#: How much denser than the corpus average a phrase must be to name a region.  A
#: phrase distributed evenly across the corpus has an enrichment of exactly 1.0
#: and describes no region in particular.  The floor is well above 1.0 because a
#: single-field corpus shares its subject by construction: every document is
#: about the same field, so only a clearly over-represented phrase distinguishes
#: a region inside it from the field itself.
MIN_TERM_ENRICHMENT = 1.5

#: A phrase in more than this share of the corpus describes the field, not a
#: region of it.  ``policies`` is in 40% of a robotics corpus and names nothing.
MAX_TERM_CORPUS_SHARE = 0.40

#: Labels are keys in the report tables and chips in the UI, so cap the length.
MAX_LABEL_CHARS = 52

#: URLs survive abstract sanitation often enough to win a cluster outright: a
#: project page pasted into three abstracts is dense in those three and nowhere
#: else, which is precisely what the enrichment ratio rewards.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\barxiv:\s*\S+", re.IGNORECASE)

#: Words that carry no topic.  Deliberately blunt: a label has two slots, so a
#: filler word that reaches one of them costs a region its name.  Abstract boiler
#: plate (``extensive experiments``, ``project page``) is as damaging as ordinary
#: prose glue, and a domain word that every document shares (``robot``,
#: ``policies``) is the worst case because it looks like a topic.
PROSE_STOPLIST = frozenset(
    (
        "paper",
        "papers",
        "propose",
        "proposed",
        "proposes",
        "present",
        "presents",
        "introduces",
        "novel",
        "new",
        "study",
        "work",
        "experiment",
        "experiments",
        "experimental",
        "extensive",
        "comprehensive",
        "thorough",
        "detailed",
        "project",
        "page",
        "code",
        "github",
        "available",
        "supplementary",
        "appendix",
        "benchmark",
        "benchmarks",
        "result",
        "results",
        "show",
        "shows",
        "demonstrate",
        "demonstrates",
        "robot",
        "robots",
        "data",
        "performance",
        "environment",
        "environments",
        "policy",
        "policies",
        "approach",
        "approaches",
        "method",
        "methods",
        "task",
        "tasks",
        "problem",
        "problems",
        "setting",
        "settings",
        "achieve",
        "achieves",
        "improves",
        "improved",
        "obtain",
        "obtains",
        "leverage",
        "leverages",
        "outperform",
        "outperforms",
        "require",
        "requires",
        "enable",
        "enables",
        "allow",
        "allows",
        "use",
        "uses",
        "using",
        "provide",
        "provides",
        "reduce",
        "reduces",
        "increase",
        "increases",
        "lead",
        "leads",
        "yield",
        "yields",
        "remain",
        "remains",
        "become",
        "becomes",
        "benefit",
        "benefits",
        "different",
        "significantly",
        "additionally",
        "furthermore",
        "moreover",
        "thus",
        "therefore",
        "however",
        "also",
        "various",
        "many",
        "much",
        "recent",
        "large",
        "small",
        "high",
        "low",
        "good",
        "better",
        "best",
        "strong",
        "weak",
        "effective",
        "efficient",
        "general",
        "specific",
        "potential",
        "promising",
        "challenging",
        "easy",
        "difficult",
        "simple",
        "complex",
        # Compounds that describe the *form* of a contribution, never a topic.
        # Left in, a region gets called ``based · learning-based``.
        "based",
        "learning-based",
        "learn",
        "learns",
        "learned",
        "directly",
    )
)


#: Phrases kept as evidence per region.  Two name it; five give a reader enough
#: to tell what the region is about without opening a paper.
SUMMARY_TERMS = 5


#: ``(?u)`` enables unicode matching; the pattern keeps interior hyphens so
#: ``sim-to-real`` stays one token instead of becoming three stopwords.  Held in a
#: constant rather than inline: a regex literal sitting next to an argument named
#: like a secret reads to static scanners as a credential assignment.
TERM_PATTERN = r"(?u)\b[a-z][a-z\-]{2,}\b"


@dataclass(frozen=True)
class ClusterLabel:
    """One discovered region: its partition index, name, and evidence terms."""

    cluster: int
    label: str
    terms: tuple[str, ...]
    size: int
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, used by the corpus metadata and the artifact."""
        return {
            "cluster": self.cluster,
            "label": self.label,
            "terms": list(self.terms),
            "size": self.size,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class DiscoveredRegimes:
    """Both resolutions plus the per-document assignment the corpus will carry."""

    discovery_k: int
    reference_k: int
    discovery: tuple[ClusterLabel, ...]
    reference: tuple[ClusterLabel, ...]
    document_ids: tuple[int, ...]
    document_reference_ids: tuple[int, ...]

    def as_metadata(self, query: str | None = None) -> dict[str, Any]:
        """JSON-ready description, written into the corpus and the artifact."""
        payload: dict[str, Any] = {
            "generated_by": "src.regimes (auto-discovered taxonomy)",
            "citation": (
                "Obayashi et al. (2026) Knowledge Manifold Optimization, "
                "Sec. 3.2 (K as a resolution on one hierarchy)"
            ),
            "discovery_k": self.discovery_k,
            "reference_k": self.reference_k,
            "discovery": [cluster.as_dict() for cluster in self.discovery],
            "reference": [cluster.as_dict() for cluster in self.reference],
            "assignment": {
                "space": "chart coordinates [-1, 1]^2",
                "why_not_semantic": (
                    "colour regions must be contiguous to be readable; the "
                    "Eq. 21 boundary views stay in TF-IDF space, unchanged"
                ),
                "labelling": (
                    "c-TF-IDF over each region's titles and abstracts: the "
                    "cluster's term counts weighted by how much rarer the term is "
                    "across the corpus, keeping only phrases found in at least "
                    f"{MIN_TERM_CLUSTER_DOCUMENTS} documents of the cluster and "
                    f"{MIN_TERM_ENRICHMENT}x denser here than corpus-wide"
                ),
            },
        }
        if query:
            payload["query"] = query
        return payload


def cluster_on_chart(coords: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """K-means partition of the 2D chart coordinates at resolution ``k``."""
    from sklearn.cluster import KMeans

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got shape {points.shape}")
    if k >= points.shape[0]:
        raise SystemExit(
            f"Cannot cluster {points.shape[0]} documents into K={k}; K must be smaller"
        )
    model = KMeans(n_clusters=k, n_init=10, random_state=0)  # type: ignore[arg-type]
    labels = np.asarray(model.fit_predict(points), dtype=np.int64)
    counts = np.bincount(labels, minlength=k)
    LOGGER.info(
        "Chart K=%d clustering: sizes %s",
        k,
        ", ".join(str(int(value)) for value in counts),
    )
    return labels


def cluster_terms(
    papers: list[dict[str, Any]],
    labels: NDArray[np.int64],
    k: int,
    top_n: int = LABEL_TERMS,
) -> dict[int, tuple[str, ...]]:
    """Name each cluster with the phrases that over-represent it (c-TF-IDF).

    The score is the phrase's TF-IDF against the corpus, summed over the cluster:
    a phrase wins by being repeated *within* the region and rare *outside* it, so
    a topical phrase such as ``sim-to-real transfer`` beats a word like ``data``
    that every paper uses once.  Two extra guards run before ranking:

    * a phrase must appear in ``MIN_TERM_CLUSTER_DOCUMENTS`` documents of the
      cluster and in ``MIN_TERM_CLUSTER_SHARE`` of them, which stops a phrase
      held by one or two papers from winning on rarity alone;
    * a phrase in more than ``MAX_TERM_CORPUS_SHARE`` of the corpus is the field
      itself rather than a region of it, and is dropped.

    Multi-word phrases are preferred over single words at equal score, because
    ``zero-shot reinforcement`` describes a region and ``planning`` barely does.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if len(papers) != labels.size:
        raise SystemExit(
            f"Naming received {len(papers)} documents but {labels.size} labels"
        )

    texts = _label_texts(papers)
    if not any(text.strip() for text in texts):
        # A corpus with no titles or abstracts cannot name anything; the caller
        # falls back to positional region names.
        return dict.fromkeys(range(k), ())

    vectoriser = CountVectorizer(
        ngram_range=(1, MAX_NGRAM),
        min_df=2,
        token_pattern=TERM_PATTERN,
    )
    # Dense is fine at corpus scale (a few hundred documents against a few
    # thousand phrases) and keeps the arithmetic below plain numpy.  The cast pins
    # the container ``CountVectorizer`` hands back, which is typed as a union.
    fitted = vectoriser.fit_transform(texts)
    counts = np.asarray(sp.csr_matrix(fitted).todense(), dtype=np.float64)
    terms: NDArray[np.str_] = np.asarray(vectoriser.get_feature_names_out())
    document_count = counts.shape[0]
    present = counts > 0.0
    total_frequency = counts.sum(axis=0)
    total_documents = present.sum(axis=0)

    # Stopwords are rejected *after* n-grams are built, not removed before them.
    # Removing them first glues the survivors together (``dexterous grasping tactile``
    # out of ``dexterous grasping with tactile``), so a label would read as words the
    # paper never wrote in that order.  Keeping the n-grams contiguous and rejecting
    # any phrase that contains a stopword yields phrases that appear in the text.
    stoplist: frozenset[str] = PROSE_STOPLIST | frozenset(ENGLISH_STOP_WORDS)
    contradicts = np.array(
        [bool(set(term.split()) & stoplist) for term in terms], dtype=np.bool_
    )
    fragment = np.array([_is_fragment(str(term)) for term in terms], dtype=np.bool_)
    word_count = np.array([len(term.split()) for term in terms], dtype=np.int64)

    # IDF against the average cluster's term total, so every cluster is scored on
    # one scale and a region cannot win by being large.
    average_frequency = max(float(counts.sum()) / max(k, 1), 1.0)
    idf = np.log1p(average_frequency / np.maximum(total_frequency, 1.0))

    # Tokens already spent by an earlier region of this same partition.  Reusing
    # them would give five legend chips that read alike (three regions all called
    # ``zero-shot something``), which is exactly the confusion a legend exists to
    # remove.
    claimed: set[str] = set()
    found: dict[int, tuple[str, ...]] = {}
    for cluster in range(k):
        rows = np.flatnonzero(labels == cluster)
        if rows.size == 0:
            found[cluster] = ()
            continue
        block = counts[rows]
        cluster_frequency = block.sum(axis=0)
        cluster_documents = (block > 0.0).sum(axis=0)

        floor = max(
            MIN_TERM_CLUSTER_DOCUMENTS,
            int(np.ceil(MIN_TERM_CLUSTER_SHARE * rows.size)),
        )
        cluster_share = cluster_documents / rows.size
        corpus_share = total_documents / max(document_count, 1)
        enrichment = cluster_share / np.maximum(corpus_share, 1e-9)
        candidate = (
            (cluster_documents >= floor)
            & (enrichment >= MIN_TERM_ENRICHMENT)
            & (corpus_share <= MAX_TERM_CORPUS_SHARE)
            & (cluster_frequency > 0.0)
            & ~contradicts
            & ~fragment
        )
        score = np.where(candidate, cluster_frequency * idf, 0.0)

        # Longest phrase first, then highest score, then alphabetically: at equal
        # evidence a longer phrase is the more specific statement of the topic.
        order = np.lexsort((terms, -score, -word_count))
        order = order[score[order] > 0.0]

        chosen: list[str] = []
        used: set[str] = set()
        # Two passes: first refuse phrases whose words another region already
        # used, then accept them if nothing distinct is left.  A region with no
        # unclaimed evidence still deserves a name.
        for avoid_claimed in (True, False):
            for index in order:
                if len(chosen) == top_n:
                    break
                words = _phrase_key(str(terms[index]))
                if words & used:
                    continue
                if avoid_claimed and words & claimed:
                    continue
                chosen.append(str(terms[index]))
                used |= words
            if len(chosen) == top_n:
                break
        claimed |= used
        found[cluster] = tuple(chosen)
    return found


def _phrase_key(phrase: str) -> set[str]:
    """Collapse a phrase's words to the keys used for overlap tests.

    Singular and plural, and inflections that share a stem (``generalization`` /
    ``generalize``), must count as the same idea or a label fills its two slots
    with one concept.  Six characters is the shortest prefix that separates the
    words this corpus actually uses while still merging those pairs.
    """
    return {word.rstrip("s")[:6] for word in phrase.split()}


def _is_fragment(phrase: str) -> bool:
    """Whether a phrase looks cut out of a sentence rather than named.

    ``learning zero-shot`` is not a topic: it is what is left when ``for`` is
    stripped out of ``learning for zero-shot``.  A participle at either end is the
    cheapest reliable sign of that, because a label's head noun never ends in
    ``-ing`` in this literature.
    """
    words = phrase.split()
    if len(words) < 2:
        return False
    return words[0].endswith("ing") or words[-1].endswith("ing")


def _label_texts(papers: list[dict[str, Any]]) -> list[str]:
    """The text a region is named from: its title, then its abstract.

    Titles and abstracts only, never the extracted full text.  A full text is
    fifteen times longer, so its terms are dominated by whatever the method
    section repeats: table headers, hyperparameter names, and the surnames of
    cited authors.  That is how a region of robotics papers ends up labelled
    ``cpus · linxi`` - one author's name, repeated in a section, beating every
    genuine topic phrase on frequency.

    The title is repeated because it is the densest statement of a paper's topic
    that the corpus holds, and the abstract alone will happily return a word the
    title already made redundant.
    """
    texts: list[str] = []
    for paper in papers:
        title = str(paper.get("title", "")).lower()
        abstract = str(paper.get("abstract", "")).lower()
        joined = _URL_RE.sub(" ", f"{title}. {title}. {abstract}")
        # Drop everything the token pattern cannot use, so a hyphenated compound
        # stays whole while punctuation cannot glue two words into one token.
        texts.append(re.sub(r"[^a-z0-9\-]+", " ", joined))
    return texts


def _build_label(terms: tuple[str, ...], fallback: str, taken: set[str]) -> str:
    """Join evidence terms into a readable, unique label."""
    chosen: list[str] = []
    for term in terms:
        if not chosen or term not in chosen:
            chosen.append(term)
        candidate = " · ".join(chosen)
        if len(candidate) > MAX_LABEL_CHARS and len(chosen) > 1:
            break
    label = " · ".join(chosen) if chosen else fallback
    if len(label) > MAX_LABEL_CHARS:
        label = label[: MAX_LABEL_CHARS - 1].rstrip() + "…"
    # A duplicate key would silently merge two regions in every table and legend.
    unique = label
    suffix = 2
    while unique in taken:
        unique = f"{label} ({suffix})"
        suffix += 1
    taken.add(unique)
    return unique


def _build_summary(terms: tuple[str, ...], size: int) -> str:
    """One sentence saying what the region is about, in plain words.

    A chip has room for two phrases, which names a region but does not explain
    it: ``successor measures · reward functions`` is a label, not an answer to
    "what is this part of the map about?".  The summary keeps every evidence
    phrase, so the legend can answer that on hover without opening a paper.
    """
    if not terms:
        return (
            f"{size} papers with no distinguishing vocabulary of their own."
            if size != 1
            else "1 paper with no distinguishing vocabulary of its own."
        )
    if len(terms) == 1:
        body = terms[0]
    else:
        body = f"{', '.join(terms[:-1])} and {terms[-1]}"
    return f"{size} papers centred on {body}."


def label_partition(
    papers: list[dict[str, Any]],
    labels: NDArray[np.int64],
    k: int,
) -> tuple[tuple[ClusterLabel, ...], dict[int, int]]:
    """Name every cluster in one partition and order them by size.

    Returns the labels plus the ``raw KMeans cluster -> region id`` mapping,
    which the caller needs to translate per-document assignments.  Returning the
    mapping rather than deriving it from the label order keeps the two from
    silently disagreeing: the label order is by size, the raw ids are whatever
    KMeans produced.
    """
    evidence = cluster_terms(papers, labels, k, top_n=SUMMARY_TERMS)
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k)

    # Largest cluster first so the numbering is stable and the dominant region
    # of the map is always region 0, which is also the one the palette tints
    # first.  Ties break on the cluster index for determinism.
    order = sorted(range(k), key=lambda cluster: (-int(counts[cluster]), cluster))
    taken: set[str] = set()
    labelled: list[ClusterLabel] = []
    mapping: dict[int, int] = {}
    for position, cluster in enumerate(order):
        label = _build_label(
            evidence.get(cluster, ())[:LABEL_TERMS], f"Region {position + 1}", taken
        )
        labelled.append(
            ClusterLabel(
                cluster=position,
                label=label,
                terms=evidence.get(cluster, ()),
                size=int(counts[cluster]),
                summary=_build_summary(evidence.get(cluster, ()), int(counts[cluster])),
            )
        )
        mapping[int(cluster)] = position
        LOGGER.info(
            "  K=%d region %d: %-40s %4d documents  [%s]",
            k,
            position,
            label,
            int(counts[cluster]),
            ", ".join(f"{term}" for term in evidence.get(cluster, ())),
        )
    return tuple(labelled), mapping


def discover_regimes(
    coords: NDArray[np.float64],
    papers: list[dict[str, Any]],
    discovery_k: int = DEFAULT_DISCOVERY_K,
    reference_k: int = DEFAULT_REFERENCE_K,
) -> DiscoveredRegimes:
    """Partition the chart at both resolutions and name every region.

    ``papers`` supply the titles and abstracts the regions are named from, in
    corpus order: document ``i`` of ``papers`` is document ``i`` of ``coords``.
    """
    LOGGER.info(
        "Discovering regimes from the chart: Discovery K=%d, Reference K=%d",
        discovery_k,
        reference_k,
    )
    discovery_raw = cluster_on_chart(coords, discovery_k)
    reference_raw = cluster_on_chart(coords, reference_k)

    discovery, discovery_map = label_partition(papers, discovery_raw, discovery_k)
    reference, reference_map = label_partition(papers, reference_raw, reference_k)

    document_ids = np.array(
        [discovery_map[int(value)] for value in discovery_raw], dtype=np.int64
    )
    document_reference_ids = np.array(
        [reference_map[int(value)] for value in reference_raw], dtype=np.int64
    )

    LOGGER.info(
        "Discovered %d discovery regions and %d reference regions",
        len(discovery),
        len(reference),
    )
    return DiscoveredRegimes(
        discovery_k=discovery_k,
        reference_k=reference_k,
        discovery=discovery,
        reference=reference,
        document_ids=tuple(int(value) for value in document_ids),
        document_reference_ids=tuple(int(value) for value in document_reference_ids),
    )


def apply_to_papers(
    papers: list[dict[str, Any]], discovered: DiscoveredRegimes
) -> list[dict[str, Any]]:
    """Stamp each paper with its discovered region as its regime.

    The corpus is the single source of truth for labels: the report, the
    dashboard legend, and the per-gap mixture table all read ``regime_label``
    from the papers, so writing it here is what makes the website follow the
    discovered taxonomy instead of the placeholder query regime.
    """
    if len(papers) != len(discovered.document_ids):
        raise SystemExit(
            f"Discovered {len(discovered.document_ids)} region assignments for "
            f"{len(papers)} papers; the manifold and the corpus are out of step"
        )
    by_id = {entry.cluster: entry for entry in discovered.discovery}
    reference_by_id = {entry.cluster: entry for entry in discovered.reference}
    for paper, identifier, reference_identifier in zip(
        papers,
        discovered.document_ids,
        discovered.document_reference_ids,
        strict=True,
    ):
        region = by_id[identifier]
        paper["regime_id"] = int(identifier)
        paper["regime_label"] = str(region.label)
        paper["regime_name"] = str(region.label)
        paper["regime_keywords"] = list(region.terms)
        paper["regime_summary"] = str(region.summary)
        paper["reference_region_id"] = int(reference_identifier)
        paper["reference_region_label"] = str(
            reference_by_id[reference_identifier].label
        )
    return papers


def write_artifact(
    path: Path, discovered: DiscoveredRegimes, query: str | None = None
) -> None:
    """Persist the discovered taxonomy next to the other data artifacts."""
    payload = discovered.as_metadata(query=query)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)
    LOGGER.info("Wrote %s", path)


def load_artifact(path: Path) -> dict[str, Any]:
    """Read ``data/regimes_auto.json``; empty dict when it does not exist."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarise(discovered: DiscoveredRegimes) -> str:
    """One-line summary used in logs and the runbook."""
    parts = [f"{entry.label} ({entry.size})" for entry in discovered.discovery]
    return f"K={discovered.discovery_k}: " + ", ".join(parts)
