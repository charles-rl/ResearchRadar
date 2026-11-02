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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

LOGGER = logging.getLogger("research_radar")

#: Sec. 3.2 resolutions.  Discovery is the labelling scale, Reference the check.
DEFAULT_DISCOVERY_K = 5
DEFAULT_REFERENCE_K = 8

#: Terms per cluster label.  Three is enough to distinguish two clusters and
#: short enough to fit a legend chip.
LABEL_TERMS = 3

#: A label term must appear in at least this share of its cluster's documents.
#: Without a floor, one hapax n-gram family wins the enrichment ratio outright.
MIN_TERM_CLUSTER_SHARE = 0.10
MIN_TERM_CLUSTER_DOCUMENTS = 2

#: How much denser than the corpus average a term must be to name a region.  A
#: term distributed evenly across the corpus has an enrichment of exactly 1.0 and
#: describes no region in particular, so without a floor the third label slot
#: fills with whatever generic word happens to be everywhere.
MIN_TERM_ENRICHMENT = 1.05

#: Labels are keys in the report tables and chips in the type, so cap the length.
MAX_LABEL_CHARS = 48


@dataclass(frozen=True)
class ClusterLabel:
    """One discovered region: its partition index, name, and evidence terms."""

    cluster: int
    label: str
    terms: tuple[str, ...]
    size: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, used by the corpus metadata and the artifact."""
        return {
            "cluster": self.cluster,
            "label": self.label,
            "terms": list(self.terms),
            "size": self.size,
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
                    "top TF-IDF terms by cluster enrichment (cluster mean mass "
                    "over corpus mean mass), floored by cluster document count"
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


def _usable_term_mask(vocabulary: Any) -> NDArray[np.bool_]:
    """Vocabulary entries fit to name a cluster with.

    ``vocabulary`` is a ``src.report.Vocabulary``: whole-word families with the
    report's own prose filter already applied in ``.usable``.  Naming a region
    from raw n-gram features would yield fragments such as ``obotwin``, so the
    families (not ``Representation.terms``) are the only thing worth labelling
    with.  Typed ``Any`` to keep this module free of a circular import on
    ``src.report``.
    """
    usable = np.asarray(vocabulary.usable, dtype=np.bool_)
    mask = np.zeros(len(vocabulary.terms), dtype=np.bool_)
    for position, raw in enumerate(vocabulary.terms):
        if not usable[position]:
            continue
        term = str(raw).strip()
        # A term with no vowel is a parse artefact more often than a word.
        if not term.isalpha() or not any(char in "aeiouy" for char in term):
            continue
        mask[position] = True
    return mask


def cluster_terms(
    vocabulary: Any,
    labels: NDArray[np.int64],
    k: int,
    top_n: int = LABEL_TERMS,
) -> dict[int, tuple[str, ...]]:
    """Top vocabulary terms per cluster, ranked by enrichment over the corpus mean.

    Enrichment (cluster mean mass / corpus mean mass) is what makes a term
    *characteristic* of a region rather than merely frequent in it: a word that
    is everywhere scores 1.0 and loses to a word that is three times denser here
    than average, however rare it is overall.  The document-share floor stops a
    single document's pet phrase from winning on ratio alone.
    """
    matrix = sp.csr_matrix(vocabulary.l2)
    terms = np.asarray(vocabulary.terms)
    usable = _usable_term_mask(vocabulary)
    corpus_mean = np.asarray(matrix.mean(axis=0)).ravel()
    labels = np.asarray(labels, dtype=np.int64)

    found: dict[int, tuple[str, ...]] = {}
    for cluster in range(k):
        rows = np.flatnonzero(labels == cluster)
        if rows.size == 0:
            found[cluster] = ()
            continue
        block = matrix[rows]
        cluster_mean = np.asarray(block.mean(axis=0)).ravel()
        presence = np.asarray((block > 0).sum(axis=0)).ravel()
        floor = max(
            MIN_TERM_CLUSTER_DOCUMENTS,
            int(np.ceil(MIN_TERM_CLUSTER_SHARE * rows.size)),
        )
        candidate = (
            usable
            & (presence >= floor)
            & (cluster_mean > 0.0)
            & (cluster_mean > MIN_TERM_ENRICHMENT * corpus_mean)
        )
        # Corpus-mean denominator floored so a term absent elsewhere does not
        # produce an infinite ratio (it is still ranked, just not infinitely).
        enrichment = np.where(candidate, cluster_mean / (corpus_mean + 1e-9), 0.0)
        order = np.argsort(-enrichment)[:top_n]
        found[cluster] = tuple(
            str(terms[index]) for index in order if enrichment[index] > 0.0
        )
    return found


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


def label_partition(
    vocabulary: Any,
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
    evidence = cluster_terms(vocabulary, labels, k)
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
            evidence.get(cluster, ()), f"Region {position + 1}", taken
        )
        labelled.append(
            ClusterLabel(
                cluster=position,
                label=label,
                terms=evidence.get(cluster, ()),
                size=int(counts[cluster]),
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
    vocabulary: Any,
    discovery_k: int = DEFAULT_DISCOVERY_K,
    reference_k: int = DEFAULT_REFERENCE_K,
) -> DiscoveredRegimes:
    """Partition the chart at both resolutions and name every region.

    ``vocabulary`` is a ``src.report.Vocabulary``: whole-word families and their
    per-document L2 shares.  Names must come from words, not from the n-gram
    features that build the representation, or a region ends up called
    ``obotwin``.
    """
    LOGGER.info(
        "Discovering regimes from the chart: Discovery K=%d, Reference K=%d",
        discovery_k,
        reference_k,
    )
    discovery_raw = cluster_on_chart(coords, discovery_k)
    reference_raw = cluster_on_chart(coords, reference_k)

    discovery, discovery_map = label_partition(vocabulary, discovery_raw, discovery_k)
    reference, reference_map = label_partition(vocabulary, reference_raw, reference_k)

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
    parts = [
        f"{entry.label} ({entry.size})" for entry in discovered.discovery
    ]
    return f"K={discovered.discovery_k}: " + ", ".join(parts)
