"""Tests for the Phase 3 manifold, the three boundary views, and the fields.

The most valuable test here is the *paper's own worked example* for Eq. 21: at
``x = (-0.2, 0.3)`` the three boundary views are 0.838, 0.425 and 0.239, giving a
harmonic mean of 0.389 against an arithmetic mean of 0.501. If the aggregation
changes, that number moves.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.manifold import (
    ANCHOR_RADIUS,
    OBJECTIVE_NAMES,
    band_pass_whitespace,
    boundary_score,
    boundary_strength,
    build_grid,
    classical_mds,
    cluster_labels,
    cosine_distance_matrix,
    document_bandwidth,
    gram_matrix,
    normalize_percentile,
    normalized_mixture,
    pairwise_distances,
    persistent_boundary,
    select_anchors,
    semantic_gradient,
    sph_weights,
    sph_weights_adaptive,
    support_statistics,
)

#: The published quantiles for the aerodynamic corpus (Section 5.5).
PAPER_QUANTILES = {
    "q20": 0.0517,
    "q60": 0.1000,
    "q85": 0.1376,
    "q98": 0.2000,
}


# --------------------------------------------------------------------------- #
# Eq. 21: persistent boundary
# --------------------------------------------------------------------------- #
def test_persistent_boundary_reproduces_the_paper_worked_example() -> None:
    """Section 4.2: B5g=0.838, B8g=0.425, B8a=0.239 -> 0.389 (not 0.501)."""
    views = {
        "k5_global": np.array([0.838]),
        "k8_global": np.array([0.425]),
        "k8_adaptive": np.array([0.239]),
    }
    score = float(persistent_boundary(views)[0])
    assert score == pytest.approx(0.389, abs=1e-3)
    arithmetic = (0.838 + 0.425 + 0.239) / 3.0
    assert score < arithmetic, "harmonic mean must punish the weak view"
    assert score == pytest.approx(3.0 / (1 / 0.838 + 1 / 0.425 + 1 / 0.239))


def test_persistent_boundary_is_zero_when_any_view_is_zero() -> None:
    """AND-like persistence: one dead view kills the candidate."""
    views = {
        "k5_global": np.array([0.9, 0.0, 1.0]),
        "k8_global": np.array([0.0, 0.9, 1.0]),
        "k8_adaptive": np.array([0.9, 0.9, 1.0]),
    }
    scores = persistent_boundary(views)
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == pytest.approx(1.0)


def test_persistent_boundary_of_equal_views_is_that_value() -> None:
    views = {
        "k5_global": np.array([0.4, 0.7]),
        "k8_global": np.array([0.4, 0.7]),
        "k8_adaptive": np.array([0.4, 0.7]),
    }
    scores = persistent_boundary(views)
    assert scores[0] == pytest.approx(0.4)
    assert scores[1] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Normalisation and boundary strength
# --------------------------------------------------------------------------- #
def test_normalize_percentile_uses_the_10th_and_90th() -> None:
    values = np.arange(101, dtype=float)
    normalized, (low, high) = normalize_percentile(values)
    assert low == pytest.approx(10.0)
    assert high == pytest.approx(90.0)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[50] == pytest.approx(0.5)


def test_boundary_strength_of_a_pure_cluster_has_zero_entropy_and_gap_one() -> None:
    mixture = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    entropy, gap = boundary_strength(mixture)
    assert entropy[0] == pytest.approx(0.0, abs=1e-9)
    assert gap[0] == pytest.approx(1.0)


def test_boundary_strength_of_a_uniform_mixture_is_maximal_entropy() -> None:
    mixture = np.full((1, 5), 0.2)
    entropy, gap = boundary_strength(mixture)
    assert entropy[0] == pytest.approx(1.0)
    assert gap[0] == pytest.approx(0.0, abs=1e-9)


def test_boundary_score_combines_entropy_and_gap() -> None:
    entropy = np.linspace(0.0, 1.0, 20)
    gap = np.linspace(1.0, 0.0, 20)
    score = boundary_score(entropy, gap)
    assert score.shape == entropy.shape
    assert score.min() == pytest.approx(0.0, abs=1e-9)
    assert score.max() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Eq. 12: band-pass white space
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("d_nn", "expected"),
    [
        (0.05, 0.0),  # below q20: incremental, excluded
        (0.0517, 0.0),  # exactly q20
        (0.07585, 0.5),  # halfway between q20 and q60
        (0.1000, 1.0),  # q60: start of the plateau
        (0.1200, 1.0),  # inside the plateau
        (0.1376, 1.0),  # q85: end of the plateau
        (0.1688, 0.5),  # halfway down to q98
        (0.2000, 0.0),  # q98: extrapolation, excluded
        (0.30, 0.0),  # beyond q98
    ],
)
def test_band_pass_whitespace_piecewise(d_nn: float, expected: float) -> None:
    value = band_pass_whitespace(np.array([d_nn]), PAPER_QUANTILES)
    assert float(value[0]) == pytest.approx(expected, abs=1e-6)


def test_band_pass_white_space_never_exceeds_one() -> None:
    values = np.linspace(0.0, 0.5, 441)
    result = band_pass_whitespace(values, PAPER_QUANTILES)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_gram_of_unit_rows_is_cosine_similarity() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    gram = gram_matrix(sp.csr_matrix(vectors))
    assert gram[0, 1] == pytest.approx(1.0 / np.sqrt(2.0))
    assert np.allclose(np.diag(gram), 1.0)


def test_cosine_distance_from_gram_has_a_zero_diagonal() -> None:
    gram = np.array([[1.0, 0.2], [0.2, 1.0]])
    distances = cosine_distance_matrix(gram)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] == pytest.approx(0.8)


def test_classical_mds_recovers_a_square() -> None:
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    distances = pairwise_distances(corners, corners)
    embedded = classical_mds(distances, n_components=2)
    recovered = pairwise_distances(embedded, embedded)
    assert np.allclose(recovered, distances, atol=1e-8)


def test_build_grid_is_x_major_with_spacing_point_one() -> None:
    grid = build_grid({"manifold": {"grid_size": 21, "bounds": [-1.0, 1.0]}})
    assert grid.shape == (441, 2)
    # Row 0 is the corner, row 1 advances in y while x stays fixed.
    assert grid[0] == pytest.approx([-1.0, -1.0])
    assert grid[1] == pytest.approx([-1.0, -0.9])
    assert grid[21] == pytest.approx([-0.9, -1.0])


def test_select_anchors_returns_nine_distinct_documents() -> None:
    rng = np.random.default_rng(0)
    coords = rng.uniform(-0.9, 0.9, size=(40, 2))
    index, targets = select_anchors(coords, n_perimeter=8)
    assert index.shape == (9,)
    assert len({int(value) for value in index}) == 9
    assert targets.shape == (9, 2)
    # Eight perimeter anchors sit on the anchor radius; the ninth is central.
    radii = np.linalg.norm(targets, axis=1)
    assert np.allclose(radii[:8], ANCHOR_RADIUS)
    assert radii[8] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# SPH kernels
# --------------------------------------------------------------------------- #
def test_global_and_adaptive_kernels_agree_for_a_constant_bandwidth() -> None:
    grid = build_grid({"manifold": {"grid_size": 5, "bounds": [-1.0, 1.0]}})
    coords = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.2]])
    global_weights = sph_weights(grid, coords, 0.25)
    adaptive = sph_weights_adaptive(grid, coords, np.full(len(coords), 0.25))
    assert np.allclose(global_weights, adaptive)


def test_adaptive_kernel_weights_a_sparse_document_more_widely() -> None:
    """A larger h_i makes document i's influence decay more slowly."""
    grid = np.array([[0.9, 0.0]])
    coords = np.array([[0.0, 0.0], [0.1, 0.0]])
    narrow = sph_weights_adaptive(grid, coords, np.array([0.1, 0.1]))
    wide = sph_weights_adaptive(grid, coords, np.array([0.5, 0.1]))
    assert wide[0, 0] > narrow[0, 0]


def test_document_bandwidth_is_the_eighth_nearest_neighbour() -> None:
    # Documents on a line at x = 0..9. The self-distance is excluded, so
    # ``nearest_k=8`` selects the 8th nearest *other* document.
    coords = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    bandwidth = document_bandwidth(coords, nearest_k=8)
    # Edge documents see only one direction: distances 1..9 -> 8th is 8.
    assert bandwidth[0] == pytest.approx(8.0)
    assert bandwidth[9] == pytest.approx(8.0)
    # A central document has neighbours on both sides: 1,1,2,2,3,3,4,4 -> 4.
    assert bandwidth[4] == pytest.approx(4.0)
    assert bandwidth[5] == pytest.approx(4.0)


def test_document_bandwidth_is_positive_even_for_coincident_points() -> None:
    coords = np.zeros((12, 2))
    bandwidth = document_bandwidth(coords, nearest_k=8)
    assert np.all(bandwidth > 0.0)


# --------------------------------------------------------------------------- #
# Cluster mixture
# --------------------------------------------------------------------------- #
def test_cluster_labels_partition_the_documents() -> None:
    rng = np.random.default_rng(3)
    dense = np.abs(rng.normal(size=(60, 40)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    labels = cluster_labels(sp.csr_matrix(dense), 8)
    assert labels.shape == (60,)
    assert labels.min() == 0
    assert labels.max() == 7
    assert len(np.unique(labels)) == 8


def test_cluster_labels_reject_an_impossible_k() -> None:
    with pytest.raises(SystemExit):
        cluster_labels(sp.csr_matrix(np.ones((4, 10))), 8)


def test_normalized_mixture_sums_to_one_per_point() -> None:
    weights = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    mixture = normalized_mixture(weights, labels, 2)
    assert np.allclose(mixture.sum(axis=1), 1.0)
    assert mixture[0, 1] == pytest.approx(7.0 / 10.0)


# --------------------------------------------------------------------------- #
# Eq. 16-18: support
# --------------------------------------------------------------------------- #
def test_support_statistics_match_their_definitions() -> None:
    grid = np.array([[0.0, 0.0]])
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]])
    weights = np.array([[0.5, 0.3, 0.1, 0.06, 0.04]])
    n_eff, d5, top3 = support_statistics(weights, grid, coords)
    # N_eff = 1 / sum(w^2) on the normalized weights, which already sum to 1.
    expected_n_eff = 1.0 / float(np.sum(np.array([0.5, 0.3, 0.1, 0.06, 0.04]) ** 2))
    assert float(n_eff[0]) == pytest.approx(expected_n_eff)
    assert float(d5[0]) == pytest.approx(0.4)  # the 5th nearest document
    assert float(top3[0]) == pytest.approx(0.5 + 0.3 + 0.1)


def test_support_statistics_top3_is_never_above_one() -> None:
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.0, 1.0, size=(30, 12))
    grid = rng.uniform(-1.0, 1.0, size=(30, 2))
    coords = rng.uniform(-1.0, 1.0, size=(12, 2))
    _, _, top3 = support_statistics(weights, grid, coords)
    assert top3.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# Eq. 13-15: semantic gradient, checked against the dense definition
# --------------------------------------------------------------------------- #
def test_semantic_gradient_matches_the_dense_bruteforce() -> None:
    """The Gram-matrix shortcut must equal ``||J_v||_F`` computed densely.

    The dense path materialises ``v(x)`` for every grid point and takes
    ``np.gradient`` of the *normalised* field; the shortcut derives the same
    central differences from pairwise inner products. Nothing is approximated,
    so the two must agree to floating-point precision.
    """
    rng = np.random.default_rng(7)
    n_documents, n_features = 6, 40
    dense = np.abs(rng.normal(size=(n_documents, n_features)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)

    grid_size = 7
    grid = build_grid({"manifold": {"grid_size": grid_size, "bounds": [-1.0, 1.0]}})
    coords = rng.uniform(-0.8, 0.8, size=(n_documents, 2))
    spacing = 2.0 / (grid_size - 1)  # the grid spans [-1, 1]

    weights = sph_weights(grid, coords, 0.35)
    shortcut = semantic_gradient(
        weights, gram_matrix(sp.csr_matrix(dense)), grid_size, spacing
    )

    # Dense path: build v(x) explicitly, then take central differences of v.
    field = weights @ dense
    field /= np.linalg.norm(field, axis=1, keepdims=True)
    cube = field.reshape(grid_size, grid_size, n_features)
    dv_dx = np.asarray(np.gradient(cube, spacing, axis=0)).reshape(-1, n_features)
    dv_dy = np.asarray(np.gradient(cube, spacing, axis=1)).reshape(-1, n_features)
    expected = np.sqrt(np.sum(dv_dx**2, axis=1) + np.sum(dv_dy**2, axis=1))
    assert np.allclose(shortcut, expected, rtol=1e-9, atol=1e-12)


def test_semantic_gradient_is_nonzero_for_a_nonuniform_field() -> None:
    rng = np.random.default_rng(11)
    dense = np.abs(rng.normal(size=(8, 30)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    grid_size = 5
    grid = build_grid({"manifold": {"grid_size": grid_size, "bounds": [-1.0, 1.0]}})
    coords = rng.uniform(-0.7, 0.7, size=(8, 2))
    weights = sph_weights(grid, coords, 0.3)
    gradient = semantic_gradient(
        weights, gram_matrix(sp.csr_matrix(dense)), grid_size, 0.5
    )
    assert gradient.shape == (grid_size * grid_size,)
    assert np.all(gradient >= 0.0)
    assert gradient.max() > 0.0
    assert sp.issparse(sp.csr_matrix(dense))


def test_semantic_gradient_rejects_a_mismatched_grid() -> None:
    weights = np.zeros((10, 3))
    with pytest.raises(ValueError):
        semantic_gradient(weights, np.eye(3), grid_size=5, spacing=0.1)


# --------------------------------------------------------------------------- #
# Objective names
# --------------------------------------------------------------------------- #
def test_objective_names_are_the_four_pareto_axes() -> None:
    assert OBJECTIVE_NAMES == ("P_boundary", "W", "G", "S")
