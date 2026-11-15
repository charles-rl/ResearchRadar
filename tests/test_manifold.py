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
    ENTROPY_EPSILON,
    GRADIENT_DELTA,
    OBJECTIVE_NAMES,
    SMOOTHING_SUPPORT_RATIO,
    TARGET_VARIANCE,
    anchor_scale,
    band_pass_whitespace,
    boundary_score,
    boundary_strength,
    build_grid,
    classical_mds,
    cluster_labels,
    constrained_stress,
    cosine_distance_matrix,
    cubic_spline_kernel,
    document_bandwidth,
    gpr_uncertainty,
    gram_matrix,
    normalize_percentile,
    normalized_mixture,
    pairwise_distances,
    persistent_boundary,
    select_anchors,
    semantic_gradient,
    semantic_gradient_field,
    sph_bandwidth,
    sph_weights,
    sph_weights_adaptive,
    stress_energy,
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
    # ``nearest_k=8`` selects the 8th nearest *other* document, divided by 1.98
    # so that neighbour sits just inside the cubic-spline support (Eq. 9's
    # convention, applied to the adaptive field).
    coords = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    bandwidth = document_bandwidth(coords, nearest_k=8)
    # Edge documents see only one direction: distances 1..9 -> 8th is 8.
    assert bandwidth[0] == pytest.approx(8.0 / SMOOTHING_SUPPORT_RATIO)
    assert bandwidth[9] == pytest.approx(8.0 / SMOOTHING_SUPPORT_RATIO)
    # A central document has neighbours on both sides: 1,1,2,2,3,3,4,4 -> 4.
    assert bandwidth[4] == pytest.approx(4.0 / SMOOTHING_SUPPORT_RATIO)
    assert bandwidth[5] == pytest.approx(4.0 / SMOOTHING_SUPPORT_RATIO)


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


# --------------------------------------------------------------------------- #
# Eq. 10: the cubic-spline kernel
# --------------------------------------------------------------------------- #
def test_cubic_spline_kernel_matches_eq_10() -> None:
    """Piecewise values, including the continuity point q = 1."""
    q = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    weight = cubic_spline_kernel(q)
    assert weight[0] == pytest.approx(1.0)
    # 1 - 3/2 q^2 + 3/4 q^3 at q = 1/2.
    assert weight[1] == pytest.approx(1.0 - 1.5 * 0.25 + 0.75 * 0.125)
    # Both branches meet at q = 1: inner gives 1/4, outer gives (2-1)^3/4.
    assert weight[2] == pytest.approx(0.25)
    assert weight[3] == pytest.approx(0.25 * (2.0 - 1.5) ** 3)
    # Compact support: W(q) = 0 for q >= 2.
    assert weight[4] == 0.0
    assert weight[5] == 0.0


def test_cubic_spline_kernel_is_continuous_and_monotone() -> None:
    q = np.linspace(0.0, 2.0, 2001)
    weight = cubic_spline_kernel(q)
    assert weight.min() >= 0.0
    # W is non-increasing: a farther document never receives more weight.
    assert np.all(np.diff(weight) <= 1e-12)
    # Continuity at the branch point (no jump larger than the sampling step).
    left = cubic_spline_kernel(np.array([1.0 - 1e-9]))[0]
    right = cubic_spline_kernel(np.array([1.0 + 1e-9]))[0]
    assert left == pytest.approx(right, abs=1e-8)


# --------------------------------------------------------------------------- #
# Eqs. 9-11: smoothing length and weight normalisation
# --------------------------------------------------------------------------- #
def test_sph_bandwidth_puts_the_farthest_document_at_q_one_point_nine_eight() -> None:
    """Eq. 9: h = max_i r_i / 1.98, so q_max = 1.98 < 2 and all N contribute."""
    rng = np.random.default_rng(2)
    coords = rng.uniform(-0.9, 0.9, size=(15, 2))
    grid = build_grid({"manifold": {"grid_size": 7, "bounds": [-1.0, 1.0]}})
    bandwidth = sph_bandwidth(grid, coords)
    ratio = pairwise_distances(grid, coords).max(axis=1) / bandwidth
    assert np.allclose(ratio, SMOOTHING_SUPPORT_RATIO)
    # Every document therefore receives non-zero weight at every grid point.
    weights = sph_weights(grid, coords, bandwidth)
    assert np.all(weights > 0.0)


def test_sph_weights_are_a_probability_vector() -> None:
    """Eq. 11: w_i = W(q_i) / sum_j W(q_j), so each row sums to one."""
    rng = np.random.default_rng(4)
    coords = rng.uniform(-0.9, 0.9, size=(20, 2))
    grid = build_grid({"manifold": {"grid_size": 9, "bounds": [-1.0, 1.0]}})
    weights = sph_weights(grid, coords, sph_bandwidth(grid, coords))
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights.min() >= 0.0
    # Eq. 16 then satisfies 1 <= N_eff <= N.
    n_eff = 1.0 / np.sum(weights**2, axis=1)
    assert n_eff.min() >= 1.0 - 1e-9
    assert n_eff.max() <= coords.shape[0] + 1e-9


def test_sph_weights_are_the_normalised_kernel_exactly() -> None:
    rng = np.random.default_rng(12)
    coords = rng.uniform(-0.8, 0.8, size=(9, 2))
    grid = np.array([[0.1, 0.2], [-0.7, 0.6]])
    bandwidth = np.array([0.4, 0.9])
    raw = cubic_spline_kernel(
        pairwise_distances(grid, coords) / bandwidth.reshape(-1, 1)
    )
    expected = raw / raw.sum(axis=1, keepdims=True)
    assert np.allclose(sph_weights(grid, coords, bandwidth), expected)


def test_adaptive_bandwidth_limits_support_to_the_eight_nearest() -> None:
    """h_i = d_(8,i)/1.98 puts document i's 8th neighbour just inside support.

    Checked on document i's own kernel, which is the scale ``h_i`` defines: the
    8th neighbour sits at ``q = 1.98 < 2`` and the 9th outside ``q >= 2``.
    Neighbouring kernels may still reach the same point, which is what makes the
    adaptive field smooth rather than a hard 8-nearest-neighbour rule.
    """
    rng = np.random.default_rng(6)
    coords = rng.uniform(-0.9, 0.9, size=(25, 2))
    bandwidth = document_bandwidth(coords, nearest_k=8)
    distances = pairwise_distances(coords[:1], coords)[0]
    distances[0] = np.inf  # exclude the document itself
    ordered = np.sort(distances)
    q = ordered / bandwidth[0]
    assert q[7] == pytest.approx(SMOOTHING_SUPPORT_RATIO)
    assert q[8] > 2.0
    # At its own location the document's kernel is W(0) = 1, the maximum, so it
    # carries the largest normalised weight (it need not be 1: neighbouring
    # kernels with wide h_j also reach the point, which is what Eq. 11 divides by).
    weights = sph_weights_adaptive(coords[:1], coords, bandwidth)
    assert float(weights[0].sum()) == pytest.approx(1.0)
    assert int(np.argmax(weights[0])) == 0
    assert weights[0, 0] > 1.0 / coords.shape[0]


def test_sph_weights_fall_back_to_the_nearest_document() -> None:
    """An empty kernel support is the h -> 0 limit: the nearest document alone."""
    grid = np.array([[5.0, 5.0]])
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    weights = sph_weights(grid, coords, 1e-4)
    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[0, 0] == 0.0


# --------------------------------------------------------------------------- #
# Eq. 7: entropy with the epsilon inside the logarithm
# --------------------------------------------------------------------------- #
def test_boundary_strength_entropy_matches_eq_7() -> None:
    mixture = np.array([[0.5, 0.3, 0.2]])
    entropy, gap = boundary_strength(mixture)
    expected = -float(np.sum(mixture * np.log(mixture + ENTROPY_EPSILON))) / np.log(3)
    assert float(entropy[0]) == pytest.approx(expected)
    assert float(gap[0]) == pytest.approx(0.2)


def test_boundary_strength_rejects_a_single_cluster() -> None:
    """Eq. 7 divides by log K, which is zero at K = 1."""
    with pytest.raises(ValueError):
        boundary_strength(np.ones((1, 1)))


# --------------------------------------------------------------------------- #
# Eqs. 9-10: configurable percentile normalisation
# --------------------------------------------------------------------------- #
def test_normalize_percentile_honours_the_configured_pair() -> None:
    values = np.linspace(0.0, 100.0, 101)
    normalized, bounds = normalize_percentile(values, percentiles=(0.0, 100.0))
    assert bounds == (0.0, 100.0)
    assert float(normalized[0]) == pytest.approx(0.0)
    assert float(normalized[100]) == pytest.approx(1.0)
    assert float(normalized[50]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Eq. 12: the trapezoid is exactly piecewise linear
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_band_pass_whitespace_rising_edge_is_linear(fraction: float) -> None:
    q20, q60 = 0.05, 0.10
    quantiles = {"q20": q20, "q60": q60, "q85": 0.14, "q98": 0.20}
    value = band_pass_whitespace(
        np.array([q20 + fraction * (q60 - q20)]), quantiles
    )
    assert float(value[0]) == pytest.approx(fraction)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_band_pass_whitespace_falling_edge_is_linear(fraction: float) -> None:
    q85, q98 = 0.14, 0.20
    quantiles = {"q20": 0.05, "q60": 0.10, "q85": q85, "q98": q98}
    value = band_pass_whitespace(
        np.array([q98 - fraction * (q98 - q85)]), quantiles
    )
    assert float(value[0]) == pytest.approx(fraction)


# --------------------------------------------------------------------------- #
# Eqs. 13-15: the Jacobian at Eq. 14's step, against a dense brute force
# --------------------------------------------------------------------------- #
def test_semantic_gradient_field_matches_dense_central_differences() -> None:
    """Eq. 14 taken literally: central differences of the normalised field.

    The dense reference materialises ``v`` at ``x +/- delta`` using the *same*
    Eq. 9 bandwidth rule and normalises explicitly, so this checks the Gram-based
    inner-product shortcut against the definition rather than against itself.
    """
    rng = np.random.default_rng(21)
    n_documents, n_features = 7, 25
    dense = np.abs(rng.normal(size=(n_documents, n_features)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    coords = rng.uniform(-0.7, 0.7, size=(n_documents, 2))
    gram = gram_matrix(sp.csr_matrix(dense))

    grid = build_grid({"manifold": {"grid_size": 5, "bounds": [-1.0, 1.0]}})
    delta = GRADIENT_DELTA

    def field_at(points: np.ndarray) -> np.ndarray:
        weights = sph_weights(points, coords, sph_bandwidth(points, coords))
        vectors = weights @ dense
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    expected = np.zeros(len(grid))
    for axis in range(2):
        plus = grid.copy()
        plus[:, axis] += delta
        minus = grid.copy()
        minus[:, axis] -= delta
        difference = field_at(plus) - field_at(minus)
        expected += np.sum(difference**2, axis=1) / (2.0 * delta) ** 2
    expected = np.sqrt(expected)

    computed = semantic_gradient_field(grid, coords, gram, delta=delta)
    assert np.allclose(computed, expected, rtol=1e-6, atol=1e-9)


def test_semantic_gradient_field_is_zero_for_a_uniform_field() -> None:
    coords = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.4]])
    document = np.array([[0.6, 0.8]])
    gram = gram_matrix(sp.csr_matrix(np.repeat(document, 3, axis=0)))
    grid = np.array([[0.0, 0.0], [0.3, -0.2], [-0.6, 0.7]])
    gradient = semantic_gradient_field(grid, coords, gram)
    assert np.allclose(gradient, 0.0, atol=1e-9)


def test_semantic_gradient_field_is_non_negative() -> None:
    rng = np.random.default_rng(31)
    dense = np.abs(rng.normal(size=(6, 18)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    coords = rng.uniform(-0.6, 0.6, size=(6, 2))
    grid = rng.uniform(-1.0, 1.0, size=(11, 2))
    gradient = semantic_gradient_field(grid, coords, gram_matrix(sp.csr_matrix(dense)))
    assert gradient.shape == (11,)
    assert np.all(gradient >= 0.0)
    assert gradient.max() > 0.0


# --------------------------------------------------------------------------- #
# Eqs. 16-18: support identities
# --------------------------------------------------------------------------- #
def test_support_score_weights_sum_to_one_and_stay_in_the_unit_interval() -> None:
    rng = np.random.default_rng(8)
    weights = rng.uniform(0.0, 1.0, size=(40, 15))
    grid = rng.uniform(-1.0, 1.0, size=(40, 2))
    coords = rng.uniform(-1.0, 1.0, size=(15, 2))
    n_eff, d5, top3 = support_statistics(weights, grid, coords)
    n_neff, _ = normalize_percentile(n_eff)
    n_d5, _ = normalize_percentile(d5)
    n_t3, _ = normalize_percentile(top3)
    support = 0.40 * n_neff + 0.30 * (1.0 - n_d5) + 0.30 * (1.0 - n_t3)
    assert np.all(support >= 0.0)
    assert np.all(support <= 1.0)
    assert np.allclose(0.40 + 0.30 + 0.30, 1.0)


# --------------------------------------------------------------------------- #
# Foundational Eqs. 4-8: the stress objective
# --------------------------------------------------------------------------- #
def _random_distances(rng: np.random.Generator, n: int) -> np.ndarray:
    points = rng.uniform(-1.0, 1.0, size=(n, 2))
    distances = pairwise_distances(points, points)
    return distances / distances.max()


def test_stress_energy_gradient_matches_finite_differences() -> None:
    """The analytic gradient must match a numeric one on every free coordinate."""
    rng = np.random.default_rng(17)
    n = 8
    coords = rng.uniform(-0.9, 0.9, size=(n, 2))
    distances = _random_distances(rng, n)
    anchors = np.array([0, 3], dtype=np.int64)

    _, gradient, _ = stress_energy(coords, distances, anchors, alpha=0.8)
    free = np.setdiff1d(np.arange(n), anchors)
    step = 1e-6
    numeric = np.zeros((n, 2))
    for index in free:
        for axis in range(2):
            plus = coords.copy()
            plus[index, axis] += step
            minus = coords.copy()
            minus[index, axis] -= step
            j_plus, _, _ = stress_energy(plus, distances, anchors, alpha=0.8)
            j_minus, _, _ = stress_energy(minus, distances, anchors, alpha=0.8)
            numeric[index, axis] = (j_plus - j_minus) / (2.0 * step)

    assert np.allclose(gradient[free], numeric[free], rtol=1e-5, atol=1e-6)
    # Anchors are held fixed, so their gradient entries are exactly zero.
    assert np.all(gradient[anchors] == 0.0)


def test_stress_energy_is_the_normalised_stress_plus_regularisers() -> None:
    rng = np.random.default_rng(19)
    n = 9
    coords = rng.uniform(-0.9, 0.9, size=(n, 2))
    distances = _random_distances(rng, n)
    anchors = np.array([0], dtype=np.int64)
    total, _, components = stress_energy(coords, distances, anchors, alpha=0.5)

    embedded = pairwise_distances(coords, coords)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scaled = 0.5 * distances
    expected_stress = np.sum((embedded[mask] - scaled[mask]) ** 2) / np.sum(
        scaled[mask] ** 2
    )
    assert components["stress"] == pytest.approx(expected_stress)
    assert total == pytest.approx(
        components["stress"]
        + 0.10 * components["repulsion"]
        + 0.05 * components["variance"]
        + 0.01 * components["centring"]
    )


def test_variance_energy_is_zero_at_the_target_variance() -> None:
    """E_var targets var = 1/3 per axis, the variance of U[-1, 1]."""
    offset = np.sqrt(TARGET_VARIANCE)
    coords = np.array(
        [
            [-offset, -offset],
            [offset, -offset],
            [-offset, offset],
            [offset, offset],
        ]
    )
    assert np.allclose(coords.var(axis=0), TARGET_VARIANCE)
    distances = pairwise_distances(coords, coords)
    _, _, components = stress_energy(coords, distances, np.array([0]), alpha=1.0)
    assert components["variance"] == pytest.approx(0.0, abs=1e-12)
    assert components["centring"] == pytest.approx(0.0, abs=1e-12)


def test_repulsion_energy_is_one_when_every_document_coincides() -> None:
    coords = np.zeros((5, 2))
    distances = np.zeros((5, 5))
    _, _, components = stress_energy(coords, distances, np.array([0]), alpha=1.0)
    assert components["repulsion"] == pytest.approx(1.0)


def test_anchor_scale_is_the_mean_anchor_distance_ratio() -> None:
    distances = np.array(
        [
            [0.0, 0.5, 0.25],
            [0.5, 0.0, 0.75],
            [0.25, 0.75, 0.0],
        ]
    )
    anchors = np.array([0, 1, 2])
    targets = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pair = np.triu(np.ones((3, 3), dtype=bool), k=1)
    chart = pairwise_distances(targets, targets)[pair].mean()
    semantic = distances[pair].mean()
    assert anchor_scale(distances, anchors, targets) == pytest.approx(chart / semantic)


def test_constrained_stress_pins_the_anchors_and_respects_the_chart() -> None:
    rng = np.random.default_rng(23)
    n = 12
    points = rng.uniform(-0.8, 0.8, size=(n, 2))
    distances = pairwise_distances(points, points) / 2.0
    anchor_index = np.array([0, 4, 9], dtype=np.int64)
    anchor_target = np.array([[0.92, 0.0], [-0.92, 0.0], [0.0, 0.0]])
    coords, stress, correlation = constrained_stress(
        distances,
        points,
        anchor_index,
        anchor_target,
        max_iterations=40,
    )
    assert np.allclose(coords[anchor_index], anchor_target)
    assert coords.min() >= -1.0 - 1e-9
    assert coords.max() <= 1.0 + 1e-9
    assert stress >= 0.0
    assert -1.0 <= correlation <= 1.0


# --------------------------------------------------------------------------- #
# Eqs. 19 + Section 5.2: GPR relative uncertainty
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_gpr_relative_uncertainty_is_sigma_post_over_sigma_prior() -> None:
    """Eq. 19 + Section 5.2, and no sqrt(n_targets) inflation.

    A multi-output GP on the SVD scores shares one kernel, so every target has
    the same posterior standard deviation. Summing the identical target columns
    (the earlier behaviour) multiplied sigma_post by sqrt(10) and pushed the
    "relative" uncertainty above one.
    """
    rng = np.random.default_rng(29)
    n_documents, n_features = 14, 26
    dense = np.abs(rng.normal(size=(n_documents, n_features)))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    coords = rng.uniform(-0.8, 0.8, size=(n_documents, 2))
    grid = build_grid({"manifold": {"grid_size": 5, "bounds": [-1.0, 1.0]}})

    fields = gpr_uncertainty(sp.csr_matrix(dense), grid, coords, n_components=3)

    for name in ("u_rbf", "u_matern"):
        assert fields[name].shape == (len(grid),)
        assert np.all(fields[name] > 0.0)
        # sigma_post <= sigma_prior, so a true relative uncertainty is <= 1.
        assert float(fields[name].max()) <= 1.0 + 1e-9
    assert np.allclose(
        fields["u_rbf"], fields["sigma_post_rbf"] / fields["sigma_prior_rbf"]
    )
    assert np.allclose(
        fields["u_matern"], fields["sigma_post_matern"] / fields["sigma_prior_matern"]
    )
    # Section 5.2: sigma_post <= sigma_prior pointwise, so the ratio is a
    # genuine relative uncertainty and can never exceed one.
    assert np.all(fields["sigma_post_rbf"] <= fields["sigma_prior_rbf"] + 1e-12)
    assert np.all(fields["sigma_post_matern"] <= fields["sigma_prior_matern"] + 1e-12)
    # Eq. 19: the fitted kernel is the anisotropic RBF plus a white-noise term.
    assert "RBF" in fields["kernel_rbf"]
    assert "WhiteKernel" in fields["kernel_rbf"]
    assert "Matern" in fields["kernel_matern"]
