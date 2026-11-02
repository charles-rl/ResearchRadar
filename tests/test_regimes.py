"""Tests for Zero-Seed (single-field) regime discovery and profile handling."""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from src.common import (
    MIN_SINGLE_FIELD_PAPERS,
    PROFILE_SEED_FIRST,
    PROFILE_SINGLE_FIELD,
    active_profile,
    is_single_field,
    resolve_profile_config,
)
from src.harvest import _config_path, main
from src.regimes import (
    MAX_LABEL_CHARS,
    DiscoveredRegimes,
    apply_to_papers,
    cluster_on_chart,
    cluster_terms,
    discover_regimes,
    label_partition,
    load_artifact,
    summarise,
    write_artifact,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _two_blob_coords(n: int = 60, seed: int = 0) -> np.ndarray:
    """Two separated blobs, so a K=2 partition is unambiguous."""
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-0.5, -0.5), scale=0.05, size=(n // 2, 2))
    right = rng.normal(loc=(0.5, 0.5), scale=0.05, size=(n // 2, 2))
    return np.vstack([left, right])


class _Vocabulary(NamedTuple):
    """Stand-in for ``src.report.Vocabulary`` in these tests.

    ``src.regimes`` only reads ``.terms``, ``.l2`` and ``.usable``, so a real
    ``Vocabulary`` (which needs a full ``Representation``) is more machinery than
    naming needs.
    """

    terms: NDArray[np.str_]
    l2: NDArray[np.float64]
    usable: NDArray[np.bool_]


def _vocabulary(
    rows_per_group: int = 4,
) -> tuple[_Vocabulary, NDArray[np.int64]]:
    """A vocabulary where term 0 is global, terms 1 and 2 are group-specific."""
    terms = np.array(["dataset", "grasping", "legged", "filler", "x1y"])
    rows: list[list[float]] = []
    labels: list[int] = []
    for _ in range(rows_per_group):
        rows.append([1.0, 0.9, 0.0, 1.0, 0.5])
        labels.append(0)
    for _ in range(rows_per_group):
        rows.append([1.0, 0.0, 0.9, 1.0, 0.5])
        labels.append(1)
    usable = np.array([True, True, True, False, True])
    return _Vocabulary(terms=terms, l2=np.array(rows), usable=usable), np.array(labels)


# --------------------------------------------------------------------------- #
# Chart clustering
# --------------------------------------------------------------------------- #


def test_cluster_on_chart_separates_two_blobs() -> None:
    coords = _two_blob_coords()
    labels = cluster_on_chart(coords, 2)
    # The first half and the second half must not share a cluster.
    assert len(set(labels[:30])) == 1
    assert len(set(labels[30:])) == 1
    assert labels[0] != labels[-1]


def test_cluster_on_chart_is_deterministic() -> None:
    coords = _two_blob_coords(seed=7)
    assert np.array_equal(cluster_on_chart(coords, 3), cluster_on_chart(coords, 3))


def test_cluster_on_chart_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"coords must be \(n, 2\)"):
        cluster_on_chart(np.zeros((10, 3)), 2)


def test_cluster_on_chart_rejects_k_at_least_n() -> None:
    with pytest.raises(SystemExit, match="K must be smaller"):
        cluster_on_chart(_two_blob_coords(10), 10)


# --------------------------------------------------------------------------- #
# Cluster naming
# --------------------------------------------------------------------------- #


def test_cluster_terms_prefers_enrichment_over_raw_frequency() -> None:
    vocabulary, labels = _vocabulary()
    found = cluster_terms(vocabulary, labels, 2)
    # "dataset" is in every document, so it is frequent but not characteristic.
    assert "dataset" not in found[0]
    assert found[0][0] == "grasping"
    assert found[1][0] == "legged"


def test_cluster_terms_drops_prose_filler_and_non_words() -> None:
    vocabulary, labels = _vocabulary()
    flat = cluster_terms(vocabulary, labels, 2)
    used = {term for group in flat.values() for term in group}
    # "x1y" is not alphabetic and "filler" is not marked usable by the report's
    # prose filter, so neither may name a region.
    assert "x1y" not in used
    assert "filler" not in used


def test_cluster_terms_returns_empty_tuple_for_empty_cluster() -> None:
    vocabulary, _ = _vocabulary()
    labels = np.array([0] * 8)
    found = cluster_terms(vocabulary, labels, 2)
    assert found[1] == ()


def test_label_partition_orders_by_size_and_maps_consistently() -> None:
    # Cluster 1 is deliberately the larger one, so a mapping that just echoed the
    # KMeans ids would order the regions wrongly.
    terms = np.array(["dataset", "grasping", "legged", "filler", "x1y"])
    usable = np.array([True, True, True, True, True])
    smaller = [[1.0, 0.9, 0.0, 1.0, 0.5]] * 6
    larger = [[1.0, 0.0, 0.9, 1.0, 0.5]] * 8
    vocabulary = _Vocabulary(
        terms=terms, l2=np.array(smaller + larger), usable=usable
    )
    labels = np.array([0] * 6 + [1] * 8)

    regions, mapping = label_partition(vocabulary, labels, 2)

    assert [region.size for region in regions] == [8, 6]
    assert [region.cluster for region in regions] == [0, 1]
    assert mapping[1] == 0
    assert mapping[0] == 1


def test_label_partition_makes_duplicate_labels_unique() -> None:
    # Two identical groups produce identical evidence terms, so the second label
    # must be suffixed rather than collide.
    terms = np.array(["dataset", "grasping", "legged", "filler", "x1y"])
    usable = np.array([True, True, True, True, True])
    rows = np.array([[1.0, 0.9, 0.0, 1.0, 0.5]] * 6)
    vocabulary = _Vocabulary(terms=terms, l2=rows, usable=usable)
    labels = np.array([0, 0, 0, 1, 1, 1])
    regions, _ = label_partition(vocabulary, labels, 2)
    names = [region.label for region in regions]
    assert len(names) == len(set(names))


def test_label_length_is_capped() -> None:
    long_terms = np.array([f"word{chr(97 + index)}" for index in range(40)])
    usable = np.ones(long_terms.size, dtype=np.bool_)
    rows = np.ones((4, long_terms.size))
    rows[2:, :] = 0.0
    vocabulary = _Vocabulary(terms=long_terms, l2=rows, usable=usable)
    regions, _ = label_partition(vocabulary, np.array([0, 0, 0, 1]), 2)
    for region in regions:
        assert len(region.label) <= MAX_LABEL_CHARS


# --------------------------------------------------------------------------- #
# Full discovery
# --------------------------------------------------------------------------- #


def test_discover_regimes_labels_every_document() -> None:
    coords = _two_blob_coords(80, seed=3)
    vocabulary, _ = _vocabulary(rows_per_group=40)
    discovered = discover_regimes(coords, vocabulary, discovery_k=2, reference_k=4)

    assert discovered.discovery_k == 2
    assert discovered.reference_k == 4
    assert len(discovered.discovery) == 2
    assert len(discovered.reference) == 4
    assert len(discovered.document_ids) == 80
    assert set(discovered.document_ids) == {0, 1}
    assert set(discovered.document_reference_ids) == {0, 1, 2, 3}
    # Sizes must add up to the corpus, or a region is being silently dropped.
    assert sum(region.size for region in discovered.discovery) == 80


def test_discovered_regimes_metadata_round_trip(tmp_path) -> None:
    coords = _two_blob_coords(40, seed=1)
    vocabulary, _ = _vocabulary(rows_per_group=20)
    discovered = discover_regimes(coords, vocabulary, discovery_k=2, reference_k=4)

    path = tmp_path / "regimes_auto.json"
    write_artifact(path, discovered, query='abs:"robot manipulation"')
    payload = load_artifact(path)

    assert payload["discovery_k"] == 2
    assert payload["reference_k"] == 4
    assert payload["query"] == 'abs:"robot manipulation"'
    assert len(payload["discovery"]) == 2
    assert len(payload["reference"]) == 4
    assert json.loads(path.read_text(encoding="utf-8"))["discovery"][0]["label"]


def test_load_artifact_returns_empty_dict_when_missing(tmp_path) -> None:
    assert load_artifact(tmp_path / "absent.json") == {}


def test_load_artifact_returns_empty_dict_on_corrupt_json(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_artifact(path) == {}


def test_summarise_lists_every_discovery_region() -> None:
    discovered = DiscoveredRegimes(
        discovery_k=2,
        reference_k=4,
        discovery=(),
        reference=(),
        document_ids=(),
        document_reference_ids=(),
    )
    assert summarise(discovered) == "K=2: "


def test_apply_to_papers_stamps_labels_from_regions() -> None:
    coords = _two_blob_coords(40, seed=5)
    vocabulary, _ = _vocabulary(rows_per_group=20)
    discovered = discover_regimes(coords, vocabulary, discovery_k=2, reference_k=4)
    papers = [
        {"id": f"doc{i}", "regime_id": 0, "regime_label": "Field"}
        for i in range(40)
    ]

    apply_to_papers(papers, discovered)

    by_region = {region.cluster: region for region in discovered.discovery}
    for paper, region_id in zip(papers, discovered.document_ids, strict=True):
        assert paper["regime_id"] == region_id
        assert paper["regime_label"] == by_region[region_id].label
        assert paper["regime_keywords"] == list(by_region[region_id].terms)
    assert len({paper["regime_label"] for paper in papers}) == 2


def test_apply_to_papers_rejects_a_length_mismatch() -> None:
    discovered = DiscoveredRegimes(
        discovery_k=5,
        reference_k=8,
        discovery=(),
        reference=(),
        document_ids=(0, 1),
        document_reference_ids=(0, 1),
    )
    with pytest.raises(SystemExit, match="out of step"):
        apply_to_papers([{"id": "only one"}], discovered)


# --------------------------------------------------------------------------- #
# Profile handling
# --------------------------------------------------------------------------- #


def _config(profile: str | None, **single_field: object) -> dict:
    block = {
        "query": 'abs:"robot manipulation" AND cat:cs.RO',
        "target_total_papers": 100,
        "years_back": 5,
        "discovery_k": 5,
        "reference_k": 8,
        "name": "Robot Manipulation",
        "label": "Field",
    }
    block.update(single_field)
    config: dict = {
        "corpus": {
            "seed_count": 15,
            "target_total_papers": 120,
            "target_per_regime": 35,
            "lookback_years": 12,
            "seed_year_offset": 2,
            "seed_overrides": {"a.pdf": 0},
        },
        "regimes": [
            {"id": 0, "label": "A", "query": "q0"},
            {"id": 1, "label": "B", "query": "q1"},
        ],
        "single_field": block,
    }
    if profile is not None:
        config["profile"] = profile
    return config


def test_seed_first_is_the_default_profile() -> None:
    config = _config(None)
    assert active_profile(config) == PROFILE_SEED_FIRST
    assert not is_single_field(config)
    resolved = resolve_profile_config(config)
    assert len(resolved["regimes"]) == 2
    assert resolved["corpus"]["seed_count"] == 15
    assert resolved["corpus"]["seed_overrides"] == {"a.pdf": 0}


def test_single_field_normalises_into_a_quoteless_seedless_shape() -> None:
    resolved = resolve_profile_config(_config(PROFILE_SINGLE_FIELD))
    corpus = resolved["corpus"]

    assert is_single_field(resolved)
    assert corpus["seed_count"] == 0
    assert corpus["seed_overrides"] == {}
    assert corpus["target_total_papers"] == 100
    # The single regime carries the whole quota, so the per-regime check and the
    # total check agree and no stage needs a second branch.
    assert corpus["target_per_regime"] == 100
    assert corpus["lookback_years"] == 5
    assert len(resolved["regimes"]) == 1
    assert resolved["regimes"][0]["label"] == "Field"
    assert resolved["regimes"][0]["query"] == 'abs:"robot manipulation" AND cat:cs.RO'


def test_single_field_does_not_mutate_the_source_config() -> None:
    config = _config(PROFILE_SINGLE_FIELD)
    resolve_profile_config(config)
    assert len(config["regimes"]) == 2
    assert config["corpus"]["seed_count"] == 15


def test_single_field_requires_a_query() -> None:
    with pytest.raises(SystemExit, match="single_field.query"):
        resolve_profile_config(_config(PROFILE_SINGLE_FIELD, query="  "))


def test_single_field_requires_enough_papers_for_the_reference_partition() -> None:
    assert MIN_SINGLE_FIELD_PAPERS == 24
    with pytest.raises(SystemExit, match="at least"):
        resolve_profile_config(_config(PROFILE_SINGLE_FIELD, target_total_papers=10))


def test_single_field_requires_a_positive_window() -> None:
    with pytest.raises(SystemExit, match="years_back"):
        resolve_profile_config(_config(PROFILE_SINGLE_FIELD, years_back=0))


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(SystemExit, match="profile"):
        resolve_profile_config({"profile": "nonsense", "corpus": {}, "regimes": []})


# --------------------------------------------------------------------------- #
# Path anchoring
# --------------------------------------------------------------------------- #


def test_harvest_paths_anchor_at_the_config_not_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    """A config kept outside the repository writes beside itself.

    Every other stage anchors through ``resolve_paths``.  Harvest once read its
    output paths from the working directory, so a second config written next to
    the first still overwrote the repository's ``data/`` — quietly destroying a
    good corpus in the other profile.
    """
    monkeypatch.chdir(tmp_path)  # pretend to be run from somewhere else
    config_dir = tmp_path / "elsewhere"
    assert _config_path(config_dir, "data/corpus.json", "x") == (
        config_dir / "data/corpus.json"
    )
    assert _config_path(config_dir, None, "data/fulltext_cache") == (
        config_dir / "data/fulltext_cache"
    )


def test_harvest_paths_keep_absolute_paths_absolute(tmp_path) -> None:
    absolute = tmp_path / "corpus.json"
    assert _config_path(tmp_path / "cfg", str(absolute), "x") == absolute


def test_harvest_cli_writes_under_the_config_directory(tmp_path) -> None:
    """End-to-end check of the same rule through ``main``'s verify path."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "profile: seed_first\n"
        "corpus:\n"
        "  output_path: data/corpus.json\n"
        "  target_total_papers: 1\n"
        "  target_per_regime: 1\n"
        "regimes:\n"
        "  - {id: 0, label: A, query: q}\n",
        encoding="utf-8",
    )
    # No corpus exists under config_dir, so verification must look there rather
    # than finding (and reporting on) a corpus in the working directory.
    with pytest.raises(SystemExit, match=r"cfg/data/corpus\.json"):
        main(["--config", str(config_dir / "config.yaml"), "--verify-only"])
