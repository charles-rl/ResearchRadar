"""Phase 3: fixed Knowledge Manifold and the four discovery fields.

Reproduces Obayashi et al. (2026) Sections 3.1-3.7, 4.2, 4.3 and 5.2-5.5.

The structure is:

1. Documents have fixed coordinates ``x_i`` on a bounded, gauge-fixed chart
   ``Omega = [-1, 1]^2`` (nine anchors: eight perimeter, one central; the
   remaining documents placed by constrained stress minimisation initialised by
   classical MDS, then frozen).
2. A 21 x 21 regular grid gives 441 candidate locations at spacing 0.1.
3. Four competing objectives are evaluated on the grid. ``P_boundary`` is the
   harmonic mean of *three* boundary-strength views (K=5 global, K=8 global,
   K=8 adaptive), because boundary evidence must persist across clustering
   resolutions and smoothing scales; ``W`` is a band-pass white-space novelty
   term; ``G`` is the semantic gradient of the interpolated field; ``S`` is
   evidential support. ``R`` is computed but is a diagnostic, not an objective.

Two implementation notes that matter for fidelity:

``P_boundary`` is a harmonic mean of three views, not a weighted sum
    A weighted sum lets one strong view compensate for a weak one. The harmonic
    mean is AND-like: any view near zero drags the whole score down, which is
    the intended semantics (Section 4.2).

``G`` is central differences of `v` itself, taken through the Gram matrix
    ``v(x) = normalize(sum_i w_i(x) z_i)`` lives in a 250,000-dimensional space,
    so materialising it for 441 grid points would need ~880 MB. The Jacobian
    norm only needs *inner products between normalised field values*, and those
    collapse onto the ``N x N`` Gram matrix ``K = Z Z^T``. With unnormalised
    ``A_i = Z^T w_i`` and ``n_i = ||A_i||``:

        cos(v_a, v_b) = (w_a^T K w_b) / (n_a n_b)

    so the central-difference estimate

        ||dv/dx|| = ||v(x + dx) - v(x - dx)|| / (2 dx)

    becomes ``sqrt(2 - 2 cos(v_+, v_-)) / (2 dx)``, which is the *same estimator*
    - literal central differences between neighbouring field values, not a
    first-order projection - evaluated exactly at Gram cost. Deriving it as
    ``(I - v v^T) dA/dx / ||A||`` instead would be only first-order accurate and
    drifts by several tens of percent wherever the field turns sharply.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from src.common import (
    LOGGER,
    Paths,
    config_int,
    configure_logging,
    corpus_regime_labels,
    is_single_field,
    load_config,
    resolve_paths,
    single_field_settings,
)
from src.embed import Representation, load_representation
from src.regimes import apply_to_papers, discover_regimes, summarise, write_artifact

DEFAULT_CONFIG = Path("config.yaml")
OBJECTIVE_NAMES = ("P_boundary", "W", "G", "S")

#: Section 3.3: boundary strength puts 0.65 on normalized entropy (the primary
#: mixture measure) and 0.35 on the dominant-second gap.
ENTROPY_WEIGHT = 0.65
GAP_WEIGHT = 0.35

#: Section 3.6 support weights.
SUPPORT_W_NEFF = 0.40
SUPPORT_W_D5 = 0.30
SUPPORT_W_T3 = 0.30

#: Section 3.2: the three views whose boundary strength must persist.
#: (name, cluster count K, smoothing scheme)
VIEW_SPECS = (
    ("k5_global", 5, "global"),
    ("k8_global", 8, "global"),
    ("k8_adaptive", 8, "adaptive"),
)

#: Section 5.2: nine anchors, eight on the perimeter plus one central.
ANCHOR_RADIUS = 0.92

#: Guards a division-by-zero in the SPH kernel when two documents coincide.
MIN_BANDWIDTH = 1e-6


@dataclass
class Manifold:
    """Frozen document coordinates plus chart and clustering metadata."""

    coords: NDArray[np.float64]
    anchor_index: NDArray[np.int64]
    anchor_target: NDArray[np.float64]
    stress: float
    distance_correlation: float
    labels: dict[int, NDArray[np.int64]]


# --------------------------------------------------------------------------- #
# Document coordinates: MDS + gauge-fixing anchors + constrained stress
# --------------------------------------------------------------------------- #
def gram_matrix(z_l2: sp.spmatrix) -> NDArray[np.float64]:
    """``K = Z Z^T``, i.e. cosine similarity for L2-normalised rows.

    Also the whole of the semantic geometry: inner products of document vectors
    are all the gradient objective needs.
    """
    matrix = sp.csr_matrix(z_l2)
    product = sp.csr_matrix(matrix @ matrix.transpose())
    similarity = np.asarray(product.toarray(), dtype=np.float64)
    return np.clip(similarity, -1.0, 1.0)


def cosine_distance_matrix(gram: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cosine distance ``D_ij = 1 - K_ij``, clipped to ``[0, 2]``."""
    distances = np.clip(1.0 - gram, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def classical_mds(
    distances: NDArray[np.float64], n_components: int = 2
) -> NDArray[np.float64]:
    """Embed a distance matrix by classical (Torgerson) metric MDS."""
    n = distances.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1][:n_components]
    positive = np.maximum(eigenvalues[order], 0.0)
    return eigenvectors[:, order] * np.sqrt(positive)


def _scale_to_chart(
    coords: NDArray[np.float64], margin: float = 0.86
) -> NDArray[np.float64]:
    """Scale coordinates so the largest magnitude lands at ``margin``."""
    peak = float(np.abs(coords).max())
    if peak <= 0.0:
        return coords
    return coords * (margin / peak)


def select_anchors(
    coords: NDArray[np.float64], n_perimeter: int = 8
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Pick eight perimeter anchors and one central anchor automatically.

    Perimeter anchors are the documents furthest along eight evenly spaced
    directions, which spreads them around the chart edge. The central anchor is
    the document closest to the centroid. Targets are placed on the perimeter at
    ``ANCHOR_RADIUS`` and at the origin, which fixes translation and scale.
    """
    center = coords.mean(axis=0)
    centered = coords - center

    chosen: list[int] = []
    angles = np.linspace(0.0, 2.0 * np.pi, n_perimeter, endpoint=False)
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        projection = centered @ direction
        for candidate in np.argsort(projection)[::-1]:
            if int(candidate) not in chosen:
                chosen.append(int(candidate))
                break

    central = int(np.argmin(np.linalg.norm(centered, axis=1)))
    if central in chosen:
        # Move it to the runner-up so the central anchor is distinct.
        distances = np.linalg.norm(centered, axis=1)
        for candidate in np.argsort(distances):
            if int(candidate) not in chosen:
                central = int(candidate)
                break

    anchor_index = np.array(chosen + [central], dtype=np.int64)
    targets = [
        [ANCHOR_RADIUS * np.cos(angle), ANCHOR_RADIUS * np.sin(angle)]
        for angle in angles
    ]
    targets.append([0.0, 0.0])
    return anchor_index, np.array(targets, dtype=np.float64)


def constrained_stress(
    distances: NDArray[np.float64],
    init: NDArray[np.float64],
    anchor_index: NDArray[np.int64],
    anchor_target: NDArray[np.float64],
    bounds: tuple[float, float] = (-1.0, 1.0),
    max_iterations: int = 800,
) -> tuple[NDArray[np.float64], float, float]:
    """Minimize metric stress with the anchor coordinates pinned.

    Stress is ``0.5 * sum_{i<j} (||x_i - x_j|| - D_ij)^2``, with the nine anchors
    held at their gauge-fixing targets and every free coordinate bounded to the
    chart so no document leaves ``[-1, 1]^2``.
    """
    from scipy.optimize import minimize

    n = distances.shape[0]
    coords = init.copy()
    coords[anchor_index] = anchor_target
    free = np.setdiff1d(np.arange(n), anchor_index)

    def unpack(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        full = coords.copy()
        full[free] = flat.reshape(-1, 2)
        return full

    def stress_and_gradient(
        flat: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        full = unpack(flat)
        delta = full[:, None, :] - full[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        safe = np.where(dist <= 1e-12, 1.0, dist)
        residual = dist - distances
        stress = 0.5 * float(np.sum(residual**2))

        # d/dx_i of 0.5 * sum_{i<j} r_ij^2 = sum_{j!=i} r_ij (x_i - x_j)/d_ij
        coefficient = np.where(dist <= 1e-12, 0.0, residual / safe)
        gradient_full = np.einsum("ij,ijk->ik", coefficient, delta)
        gradient_full[anchor_index] = 0.0
        return stress, gradient_full[free].ravel()

    start = coords[free].ravel()
    lower, upper = bounds
    result = minimize(
        stress_and_gradient,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(lower, upper)] * start.size,
        options={"maxiter": max_iterations, "maxfun": max_iterations * 4},
    )
    final = unpack(result.x)

    pair_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    final_dist = np.linalg.norm(final[:, None, :] - final[None, :, :], axis=-1)
    original = distances[pair_mask]
    embedded = final_dist[pair_mask]
    # Shepard-style agreement between embedding and true distances.
    corr = float(
        np.corrcoef(np.argsort(np.argsort(original)), np.argsort(np.argsort(embedded)))[
            0, 1
        ]
    )
    LOGGER.info(
        "Stress minimization: final stress=%.5f, Spearman(distance)=%.4f, "
        "converged=%s (%d iterations)",
        float(result.fun),
        corr,
        bool(result.success),
        int(result.nit),
    )
    return final, float(result.fun), corr


def cluster_labels(z_l2: sp.spmatrix, k: int) -> NDArray[np.int64]:
    """K-means partition of the documents at resolution ``K`` (Section 3.2).

    Labels carry no meaning across resolutions; only the scalar boundary
    measures computed from them are comparable (Section 3.2).
    """
    from sklearn.cluster import KMeans

    n_documents = z_l2.get_shape()[0]
    if k >= n_documents:
        raise SystemExit(
            f"Cannot cluster {n_documents} documents into K={k}; K must be smaller"
        )
    model = KMeans(n_clusters=k, n_init=10, random_state=0)  # type: ignore[arg-type]
    labels = np.asarray(model.fit_predict(sp.csr_matrix(z_l2)), dtype=np.int64)
    counts = np.bincount(labels, minlength=k)
    inertia = model.inertia_
    LOGGER.info(
        "K=%d clustering: sizes %s (inertia %s)",
        k,
        ", ".join(str(int(value)) for value in counts),
        "n/a" if inertia is None else f"{float(inertia):.4f}",
    )
    return labels


def build_manifold(gram: NDArray[np.float64], config: dict[str, Any]) -> Manifold:
    """Construct the frozen Knowledge Manifold, including the K=5 and K=8 views."""
    manifold_cfg = config.get("manifold", {})
    n_perimeter = int(manifold_cfg.get("perimeter_anchors", 8))
    bounds_cfg = manifold_cfg.get("bounds", [-1.0, 1.0])
    bounds = (float(bounds_cfg[0]), float(bounds_cfg[1]))

    distances = cosine_distance_matrix(gram)
    init = _scale_to_chart(classical_mds(distances, n_components=2))
    anchor_index, anchor_target = select_anchors(init, n_perimeter=n_perimeter)
    coords, stress, correlation = constrained_stress(
        distances, init, anchor_index, anchor_target, bounds=bounds
    )

    LOGGER.info(
        "Manifold: %d documents on [%.2f, %.2f]^2, bounds x=[%.3f, %.3f] "
        "y=[%.3f, %.3f]",
        coords.shape[0],
        bounds[0],
        bounds[1],
        coords[:, 0].min(),
        coords[:, 0].max(),
        coords[:, 1].min(),
        coords[:, 1].max(),
    )
    return Manifold(
        coords=coords,
        anchor_index=anchor_index,
        anchor_target=anchor_target,
        stress=stress,
        distance_correlation=correlation,
        labels={},
    )


# --------------------------------------------------------------------------- #
# Grid and SPH fields
# --------------------------------------------------------------------------- #
def build_grid(config: dict[str, Any]) -> NDArray[np.float64]:
    """Return the ``grid_size^2`` candidate coordinates in ``[-1, 1]^2``."""
    manifold_cfg = config.get("manifold", {})
    size = int(manifold_cfg.get("grid_size", 21))
    bounds_cfg = manifold_cfg.get("bounds", [-1.0, 1.0])
    axis = np.linspace(float(bounds_cfg[0]), float(bounds_cfg[1]), size)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def pairwise_distances(
    points: NDArray[np.float64], targets: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Euclidean distance matrix between ``points`` rows and ``targets`` rows."""
    delta = points[:, None, :] - targets[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def sph_weights(
    grid: NDArray[np.float64],
    coords: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Global SPH kernel weights ``exp(-||x - x_i||^2 / (2 h^2))``.

    ``bandwidth`` is one smoothing scale over the full map (Section 5.4).
    """
    squared = pairwise_distances(grid, coords) ** 2
    return np.exp(-squared / (2.0 * float(bandwidth) ** 2))


def sph_weights_adaptive(
    grid: NDArray[np.float64],
    coords: NDArray[np.float64],
    document_bandwidth: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Adaptive SPH weights with one smoothing length ``h_i`` per document.

    ``w_i(x) = exp(-||x - x_i||^2 / (2 h_i^2))``, the standard adaptive SPH
    kernel: a document in a sparse neighbourhood smooths over a longer scale.
    """
    squared = pairwise_distances(grid, coords) ** 2
    h = np.maximum(np.asarray(document_bandwidth, dtype=np.float64), MIN_BANDWIDTH)
    return np.exp(-squared / (2.0 * h[None, :] ** 2))


def document_bandwidth(
    coords: NDArray[np.float64], nearest_k: int = 8
) -> NDArray[np.float64]:
    """Per-document smoothing length: distance to the ``nearest_k``-th neighbour.

    Section 5.4: "the adaptive field used a local scale based on the eight
    nearest documents". ``nearest_k`` counts the document itself as neighbour
    zero, so ``k=8`` is the 8th nearest other document.
    """
    distances = pairwise_distances(coords, coords)
    np.fill_diagonal(distances, np.inf)
    k = min(max(1, nearest_k), distances.shape[1] - 1)
    ordered = np.sort(distances, axis=1)
    return np.maximum(ordered[:, k - 1], MIN_BANDWIDTH)


def normalize_percentile(
    values: NDArray[np.float64], low: float = 10.0, high: float = 90.0
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    """``N_+(z) = clip((z - P_low) / (P_high - P_low), 0, 1)`` (Eq. 9)."""
    p_low = float(np.percentile(values, low))
    p_high = float(np.percentile(values, high))
    span = p_high - p_low
    if span <= 1e-12:
        return np.zeros_like(values), (p_low, p_high)
    return np.clip((values - p_low) / span, 0.0, 1.0), (p_low, p_high)


def normalized_mixture(
    weights: NDArray[np.float64], labels: NDArray[np.int64], n_clusters: int
) -> NDArray[np.float64]:
    """Normalized cluster probability vector for each grid point (Eq. 6)."""
    totals = weights.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 1e-300, 1.0, totals)
    normalized = weights / totals
    mixture = np.zeros((weights.shape[0], n_clusters), dtype=np.float64)
    for cluster in range(n_clusters):
        mixture[:, cluster] = normalized[:, labels == cluster].sum(axis=1)
    return mixture


def boundary_strength(
    mixture: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return raw normalized entropy ``H_K`` and dominant-second gap ``Dp_K``."""
    n_clusters = mixture.shape[1]
    safe = np.clip(mixture, 1e-12, None)
    entropy = -(mixture * np.log(safe)).sum(axis=1) / np.log(n_clusters)
    entropy = np.clip(entropy, 0.0, 1.0)
    ordered = np.sort(mixture, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    return entropy, gap


def boundary_score(
    entropy: NDArray[np.float64], gap: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``B_K = 0.65 N_+(H_K) + 0.35 N_-(Dp_K)`` (Eq. 11)."""
    normalized_entropy, _ = normalize_percentile(entropy)
    normalized_gap, _ = normalize_percentile(gap)
    return ENTROPY_WEIGHT * normalized_entropy + GAP_WEIGHT * (1.0 - normalized_gap)


def persistent_boundary(views: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Harmonic mean of the three boundary views (Eq. 21).

    ``P = 3 / (1/B_5g + 1/B_8g + 1/B_8a)``. A zero in any view makes the
    reciprocal infinite and the score zero, which is the AND-like behaviour the
    paper argues for in Section 4.2.
    """
    ordered = [views[name] for name, _, _ in VIEW_SPECS]
    stack = np.column_stack(ordered)
    positive = stack > 0.0
    reciprocals = np.where(positive, 1.0 / np.where(positive, stack, 1.0), np.inf)
    total = reciprocals.sum(axis=1)
    return np.where(
        np.isfinite(total), len(ordered) / np.where(np.isfinite(total), total, 1.0), 0.0
    )


def band_pass_whitespace(
    d_nn: NDArray[np.float64], quantiles: dict[str, float]
) -> NDArray[np.float64]:
    """Band-pass white-space novelty ``W`` (Eq. 12)."""
    q20 = quantiles["q20"]
    q60 = quantiles["q60"]
    q85 = quantiles["q85"]
    q98 = quantiles["q98"]

    width = np.zeros_like(d_nn)
    rising = (d_nn > q20) & (d_nn < q60) & (q60 > q20)
    width[rising] = (d_nn[rising] - q20) / (q60 - q20)
    plateau = (d_nn >= q60) & (d_nn <= q85)
    width[plateau] = 1.0
    falling = (d_nn > q85) & (d_nn < q98) & (q98 > q85)
    width[falling] = (q98 - d_nn[falling]) / (q98 - q85)
    return np.clip(width, 0.0, 1.0)


def empirical_dnn_quantiles(
    d_nn: NDArray[np.float64], config: dict[str, Any]
) -> dict[str, float]:
    """Take ``d_nn`` quantiles over the candidate domain (Section 5.5).

    The paper defines these over its own search grid. ``config.yaml`` records the
    published values as a reference; with ``use_empirical_quantiles: true`` (the
    default) they are recomputed for this corpus so the band-pass window matches
    its own sampling density instead of the aerodynamic corpus's scale.
    """
    manifold_cfg = config.get("manifold", {})
    configured = manifold_cfg.get("q_whitespace", {}) or {}
    empirical = {
        "q20": float(np.percentile(d_nn, 20)),
        "q60": float(np.percentile(d_nn, 60)),
        "q85": float(np.percentile(d_nn, 85)),
        "q98": float(np.percentile(d_nn, 98)),
    }
    LOGGER.info(
        "d_nn quantiles (empirical): q20=%.4f q60=%.4f q85=%.4f q98=%.4f",
        empirical["q20"],
        empirical["q60"],
        empirical["q85"],
        empirical["q98"],
    )
    if configured:
        LOGGER.info(
            "d_nn quantiles (published, different corpus): q20=%.4f q60=%.4f "
            "q85=%.4f q98=%.4f",
            float(configured.get("q20", 0.0)),
            float(configured.get("q60", 0.0)),
            float(configured.get("q85", 0.0)),
            float(configured.get("q98", 0.0)),
        )
    if bool(manifold_cfg.get("use_empirical_quantiles", True)):
        return empirical
    return {
        key: float(configured.get(key, empirical[key]))
        for key in ("q20", "q60", "q85", "q98")
    }


def _difference_pairs(
    grid_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Neighbour pairs and divisors matching ``np.gradient``'s conventions.

    Interior points use the central three-point stencil; the two edge points use
    a one-sided difference, exactly as ``np.gradient`` does, so the estimator is
    directly comparable with a dense finite-difference implementation.
    """
    low: list[int] = []
    high: list[int] = []
    divisor: list[float] = []
    for index in range(grid_size):
        if index == 0:
            low.append(0)
            high.append(1)
            divisor.append(1.0)
        elif index == grid_size - 1:
            low.append(grid_size - 2)
            high.append(grid_size - 1)
            divisor.append(1.0)
        else:
            low.append(index - 1)
            high.append(index + 1)
            divisor.append(2.0)
    return (
        np.array(low, dtype=np.int64),
        np.array(high, dtype=np.int64),
        np.array(divisor, dtype=np.float64),
    )


def semantic_gradient(
    weights: NDArray[np.float64],
    gram: NDArray[np.float64],
    grid_size: int,
    spacing: float,
) -> NDArray[np.float64]:
    """Raw Jacobian Frobenius norm ``g(x) = ||J_v(x)||_F`` (Eqs. 13-14).

    Central differences of the *normalised* interpolated field, evaluated from
    pairwise inner products so the 250,000-dimensional field is never built.
    Rows of ``weights`` follow the ``x``-major grid layout, so the index grid
    below has axis 0 = x and axis 1 = y.
    """
    n_points = grid_size * grid_size
    if weights.shape[0] != n_points:
        raise ValueError(
            f"weights has {weights.shape[0]} rows but the grid has {n_points} points"
        )

    projected = weights @ gram
    norm = np.sqrt(np.maximum(np.einsum("mi,mi->m", weights, projected), 0.0))
    index_grid = np.arange(n_points).reshape(grid_size, grid_size)

    low, high, divisor = _difference_pairs(grid_size)
    total = np.zeros(n_points, dtype=np.float64)
    for axis in range(2):
        for position in range(grid_size):
            if axis == 0:
                low_index = index_grid[low[position], :]
                high_index = index_grid[high[position], :]
                target = index_grid[position, :]
            else:
                low_index = index_grid[:, low[position]]
                high_index = index_grid[:, high[position]]
                target = index_grid[:, position]

            # cos(v_a, v_b) = w_a^T K w_b / (n_a n_b), and ||v_a - v_b||^2 =
            # 2 - 2 cos(v_a, v_b), so this is the plain central difference of the
            # normalised field, taken between real neighbouring grid values.
            inner = np.einsum("mi,mi->m", projected[low_index], weights[high_index])
            scale = norm[low_index] * norm[high_index]
            scale = np.where(scale <= 0.0, 1.0, scale)
            cosine = np.clip(inner / scale, -1.0, 1.0)
            step = (float(divisor[position]) * spacing) ** 2
            total[target] += np.maximum(2.0 - 2.0 * cosine, 0.0) / step
    return np.sqrt(total)


def support_statistics(
    weights: NDArray[np.float64],
    grid: NDArray[np.float64],
    coords: NDArray[np.float64],
    fifth_index: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``N_eff``, ``d5``, and ``T3`` for each grid point (Eqs. 16-17)."""
    totals = weights.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 1e-300, 1.0, totals)
    normalized = weights / totals

    n_eff = 1.0 / np.sum(normalized**2, axis=1)

    distances = pairwise_distances(grid, coords)
    k = min(fifth_index, distances.shape[1] - 1)
    d5 = np.sort(distances, axis=1)[:, k]

    ordered = np.sort(normalized, axis=1)
    top_three = ordered[:, -3:].sum(axis=1)
    return n_eff, d5, top_three


def gpr_uncertainty(
    z_l2: sp.spmatrix,
    grid: NDArray[np.float64],
    coords: NDArray[np.float64],
    n_components: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Relative uncertainty from two GPR surrogates (Section 3.7, Eq. 19).

    A Gaussian process is fitted on a truncated-SVD reduction of the L2
    representation and its predictive standard deviation is evaluated on the
    grid. Two covariance assumptions are used - an anisotropic-baseline RBF and a
    Matern-3/2 - and their low-uncertainty scores are averaged into ``R``.

    Hyperparameters are held fixed at a data-derived length scale instead of
    being optimized. Letting the marginal likelihood choose the length scale
    drove it to the lower bound: the GP then memorizes the 120 training points,
    giving near-zero std inside the sample and a spike outside it, which is a
    property of the optimizer rather than of the manifold. A fixed scale of one
    median inter-document distance keeps the field monotone in "distance from
    supporting evidence", which is the diagnostic the paper actually uses.

    ``R`` is a reliability diagnostic only and is deliberately excluded from the
    Pareto objectives (Section 4.3): maximizing low uncertainty would favour
    densely sampled regions and suppress frontier exploration.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    n_documents, n_features = z_l2.get_shape()
    components = max(1, min(n_components, n_documents - 1, n_features - 1))
    reduced = TruncatedSVD(n_components=components, random_state=0).fit_transform(z_l2)

    pairwise = pairwise_distances(coords, coords)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    length_scale = float(np.median(upper)) if upper.size else 1.0
    length_scale = max(length_scale, 1e-3)
    LOGGER.info(
        "GPR diagnostic: fixed length_scale=%.4f (median inter-document distance), "
        "%d SVD components",
        length_scale,
        components,
    )

    def predictive_std(kernel: object) -> NDArray[np.float64]:
        model = GaussianProcessRegressor(
            kernel=kernel,  # type: ignore[arg-type]
            normalize_y=True,
            alpha=1e-8,
            # ``optimizer=None`` is the documented way to skip hyperparameter
            # optimization (sklearn _gpr.py:64,81); the stub under-declares it as
            # str-only. All bounds are "fixed" anyway, so fitting is a no-op.
            optimizer=None,  # type: ignore[arg-type]
        )
        model.fit(coords, reduced)
        prediction = model.predict(grid, return_std=True)
        std = np.asarray(prediction[1], dtype=np.float64)
        return np.linalg.norm(std.reshape(len(grid), -1), axis=1)

    base = ConstantKernel(1.0, constant_value_bounds="fixed")
    noise = WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    u_rbf = predictive_std(
        base * RBF(length_scale=length_scale, length_scale_bounds="fixed") + noise
    )
    u_matern = predictive_std(
        base * Matern(length_scale=length_scale, nu=1.5, length_scale_bounds="fixed")
        + noise
    )
    return {"u_rbf": u_rbf, "u_matern": u_matern, "u_mean": 0.5 * (u_rbf + u_matern)}


# --------------------------------------------------------------------------- #
# Objective assembly
# --------------------------------------------------------------------------- #
def distance_maximising_baseline(
    d_nn: np.ndarray,
    grid: np.ndarray,
    objectives: np.ndarray,
    support: np.ndarray,
    reliability: np.ndarray,
    boundary: np.ndarray,
    gradient: np.ndarray,
    count: int = 16,
) -> dict[str, Any]:
    """Reproduce Table 4: unconstrained distance maximisation versus the front.

    The paper's negative control picks the ``count`` points with the largest
    nearest-neighbour distance twice: once over the whole grid, and once over the
    interior only (excluding the chart edge, ``x = +-1`` or ``y = +-1``). Every
    unconstrained pick lands on the boundary, which is exactly why an
    unconstrained novelty search is not a discovery criterion: it maximises
    emptiness and pays for it in support, boundary strength, and gradient.
    """
    edge = np.isclose(np.abs(grid), 1.0).any(axis=1)
    interior = np.flatnonzero(~edge)

    def pick(indices: np.ndarray) -> np.ndarray:
        order = indices[np.argsort(d_nn[indices])[::-1]]
        return order[:count]

    def summary(indices: np.ndarray) -> dict[str, float]:
        return {
            "d_nn": round(float(d_nn[indices].mean()), 4),
            "S": round(float(support[indices].mean()), 4),
            "R": round(float(reliability[indices].mean()), 4),
            "P_boundary": round(float(boundary[indices].mean()), 4),
            "G": round(float(gradient[indices].mean()), 4),
            "points_on_chart_edge": int(edge[indices].sum()),
        }

    all_grid = pick(np.arange(len(d_nn)))
    interior_only = pick(interior)
    return {
        "reference": "Obayashi et al. (2026) Table 4",
        "count": count,
        "all_grid": summary(all_grid),
        "interior_only": summary(interior_only),
        "all_grid_indices": [int(value) for value in all_grid],
        "interior_only_indices": [int(value) for value in interior_only],
        "interior_candidates": len(interior),
        "notes": (
            "Columns are mean values over the selected points. R is the averaged "
            "dual-kernel reliability diagnostic and is not a Pareto objective."
        ),
    }


def compute_objectives(
    config: dict[str, Any],
    representation: Representation,
    regimes: NDArray[np.int64],
    n_regimes: int,
    manifold: Manifold,
) -> dict[str, Any]:
    """Compute the four discovery fields over the evaluation grid."""
    manifold_cfg = config.get("manifold", {})
    grid = build_grid(config)
    coords = manifold.coords
    bounds_cfg = manifold_cfg.get("bounds", [-1.0, 1.0])
    grid_size = int(manifold_cfg.get("grid_size", 21))
    spacing = float((float(bounds_cfg[1]) - float(bounds_cfg[0])) / (grid_size - 1))

    h_global = float(manifold_cfg.get("sph_h", 0.25))
    nearest_k = int(manifold_cfg.get("nearest_k", 8))

    # Two smoothing schemes: one scale over the whole map, and one scale per
    # document from its own 8th-nearest neighbour.
    w_global = sph_weights(grid, coords, h_global)
    h_documents = document_bandwidth(coords, nearest_k)
    w_adaptive = sph_weights_adaptive(grid, coords, h_documents)
    LOGGER.info(
        "SPH bandwidth: global h=%.4f; adaptive h_i in [%.4f, %.4f] (median %.4f)",
        h_global,
        float(h_documents.min()),
        float(h_documents.max()),
        float(np.median(h_documents)),
    )

    gram = gram_matrix(representation.z_l2)

    views: dict[str, NDArray[np.float64]] = {}
    entropies: dict[str, NDArray[np.float64]] = {}
    gaps: dict[str, NDArray[np.float64]] = {}
    mixtures: dict[str, NDArray[np.float64]] = {}
    labels: dict[int, NDArray[np.int64]] = {}

    for name, k, scheme in VIEW_SPECS:
        if k not in labels:
            labels[k] = cluster_labels(representation.z_l2, k)
        weights = w_global if scheme == "global" else w_adaptive
        mixture = normalized_mixture(weights, labels[k], k)
        entropy, gap = boundary_strength(mixture)
        views[name] = boundary_score(entropy, gap)
        entropies[name] = entropy
        gaps[name] = gap
        mixtures[name] = mixture
        LOGGER.info(
            "Boundary view %-12s K=%d %-8s B in [%.3f, %.3f], mean %.3f",
            name,
            k,
            scheme,
            float(views[name].min()),
            float(views[name].max()),
            float(views[name].mean()),
        )

    p_boundary = persistent_boundary(views)
    LOGGER.info(
        "P_boundary (harmonic mean of three views): [%.3f, %.3f], mean %.3f, "
        "zero at %d points",
        float(p_boundary.min()),
        float(p_boundary.max()),
        float(p_boundary.mean()),
        int((p_boundary == 0.0).sum()),
    )

    d_nn = pairwise_distances(grid, coords).min(axis=1)
    quantiles = empirical_dnn_quantiles(d_nn, config)
    whitespace = band_pass_whitespace(d_nn, quantiles)

    gradient_raw = semantic_gradient(w_global, gram, grid_size, spacing)
    gradient, p_gradient = normalize_percentile(gradient_raw)

    n_eff, d5, top_three = support_statistics(w_global, grid, coords)
    n_neff, p_neff = normalize_percentile(n_eff)
    n_d5, p_d5 = normalize_percentile(d5)
    n_t3, p_t3 = normalize_percentile(top_three)
    support = (
        SUPPORT_W_NEFF * n_neff
        + SUPPORT_W_D5 * (1.0 - n_d5)
        + SUPPORT_W_T3 * (1.0 - n_t3)
    )

    uncertainty_fields = gpr_uncertainty(representation.z_l2, grid, coords)
    n_u_rbf, _ = normalize_percentile(uncertainty_fields["u_rbf"])
    n_u_matern, _ = normalize_percentile(uncertainty_fields["u_matern"])
    reliability = 0.5 * (1.0 - n_u_rbf) + 0.5 * (1.0 - n_u_matern)
    uncertainty = uncertainty_fields["u_mean"]
    kernel_agreement = float(
        np.corrcoef(uncertainty_fields["u_rbf"], uncertainty_fields["u_matern"])[0, 1]
    )

    regime_mixture = normalized_mixture(w_global, regimes, n_regimes)

    objectives = np.column_stack([p_boundary, whitespace, gradient, support])

    LOGGER.info(
        "Objective ranges: P_boundary=[%.3f, %.3f] W=[%.3f, %.3f] "
        "G=[%.3f, %.3f] S=[%.3f, %.3f]",
        p_boundary.min(),
        p_boundary.max(),
        whitespace.min(),
        whitespace.max(),
        gradient.min(),
        gradient.max(),
        support.min(),
        support.max(),
    )
    LOGGER.info(
        "Field stats: d_nn=[%.4f, %.4f] N_eff=[%.2f, %.2f] d5=[%.4f, %.4f] "
        "T3=[%.3f, %.3f]",
        d_nn.min(),
        d_nn.max(),
        n_eff.min(),
        n_eff.max(),
        d5.min(),
        d5.max(),
        top_three.min(),
        top_three.max(),
    )
    LOGGER.info(
        "GPR diagnostic: u=[%.5f, %.5f], R=[%.3f, %.3f], kernel agreement r=%.3f "
        "(R is not a Pareto objective)",
        uncertainty.min(),
        uncertainty.max(),
        reliability.min(),
        reliability.max(),
        kernel_agreement,
    )

    return {
        "grid": grid,
        "objectives": objectives,
        "d_nn": d_nn,
        "whitespace_quantiles": quantiles,
        "views": views,
        "entropies": entropies,
        "gaps": gaps,
        "mixtures": mixtures,
        "labels": labels,
        "weights_global": w_global,
        "weights_adaptive": w_adaptive,
        "h_documents": h_documents,
        "n_eff": n_eff,
        "d5": d5,
        "top3": top_three,
        "gradient_raw": gradient_raw,
        "uncertainty": uncertainty,
        "uncertainty_rbf": uncertainty_fields["u_rbf"],
        "uncertainty_matern": uncertainty_fields["u_matern"],
        "reliability": reliability,
        "kernel_agreement": kernel_agreement,
        "mixture_regime": regime_mixture,
        "percentiles": {
            "gradient": p_gradient,
            "n_eff": p_neff,
            "d5": p_d5,
            "top3": p_t3,
        },
        "spacing": spacing,
        "sph_h": h_global,
        "nearest_k": nearest_k,
        "grid_size": grid_size,
        "baseline": distance_maximising_baseline(
            d_nn=d_nn,
            grid=grid,
            objectives=objectives,
            support=support,
            reliability=reliability,
            boundary=p_boundary,
            gradient=gradient,
        ),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_grid(
    paths: Paths,
    manifold: Manifold,
    fields: dict[str, Any],
    corpus: dict[str, Any],
) -> None:
    """Write the projection and objective matrix to ``data/manifold_grid.npz``."""
    paths.grid_npz.parent.mkdir(parents=True, exist_ok=True)
    papers = corpus["papers"]
    views = fields["views"]
    entropies = fields["entropies"]
    gaps = fields["gaps"]
    mixtures = fields["mixtures"]

    archives: dict[str, Any] = {
        "grid": fields["grid"],
        "objectives": fields["objectives"],
        "objective_names": np.array(OBJECTIVE_NAMES),
        "doc_coords": manifold.coords,
        "doc_regime": np.array(
            [int(paper["regime_id"]) for paper in papers], dtype=np.int64
        ),
        "doc_ids": np.array([str(paper["id"]) for paper in papers]),
        "anchor_index": manifold.anchor_index,
        "anchor_target": manifold.anchor_target,
        "weights_global": fields["weights_global"].astype(np.float32),
        "weights_adaptive": fields["weights_adaptive"].astype(np.float32),
        "h_documents": fields["h_documents"],
        "d_nn": fields["d_nn"],
        "d5": fields["d5"],
        "n_eff": fields["n_eff"],
        "top3": fields["top3"],
        "gradient_raw": fields["gradient_raw"],
        "uncertainty": fields["uncertainty"],
        "uncertainty_rbf": fields["uncertainty_rbf"],
        "uncertainty_matern": fields["uncertainty_matern"],
        "reliability": fields["reliability"],
        "kernel_agreement": np.array([fields["kernel_agreement"]]),
        "mixture_regime": fields["mixture_regime"],
        "whitespace_quantiles": np.array(
            [
                fields["whitespace_quantiles"][key]
                for key in ("q20", "q60", "q85", "q98")
            ]
        ),
        "stress": np.array([manifold.stress]),
        "distance_correlation": np.array([manifold.distance_correlation]),
        "grid_size": np.array([fields["grid_size"]]),
        "spacing": np.array([fields["spacing"]]),
        "sph_h": np.array([fields["sph_h"]]),
        "nearest_k": np.array([fields["nearest_k"]]),
    }
    for name, _, _ in VIEW_SPECS:
        archives[f"b_{name}"] = views[name]
        archives[f"entropy_{name}"] = entropies[name]
        archives[f"gap_{name}"] = gaps[name]
        archives[f"mixture_{name}"] = mixtures[name]
    for k, labels in fields["labels"].items():
        archives[f"labels_k{k}"] = labels

    np.savez_compressed(paths.grid_npz, **archives)
    LOGGER.info(
        "Saved %s: grid=%s objectives=%s",
        paths.grid_npz,
        fields["grid"].shape,
        fields["objectives"].shape,
    )

    manifest = {
        "grid_size": fields["grid_size"],
        "grid_points": len(fields["grid"]),
        "documents": int(manifold.coords.shape[0]),
        "spacing": fields["spacing"],
        "sph_h_global": fields["sph_h"],
        "nearest_k_adaptive": fields["nearest_k"],
        "stress": manifold.stress,
        "distance_correlation": manifold.distance_correlation,
        "anchors": {
            "indices": [int(i) for i in manifold.anchor_index],
            "targets": manifold.anchor_target.tolist(),
        },
        "boundary_views": [
            {"name": name, "k": k, "scheme": scheme} for name, k, scheme in VIEW_SPECS
        ],
        "persistent_boundary": "harmonic mean of b_k5_global, b_k8_global, b_k8_adaptive",
        "whitespace_quantiles": fields["whitespace_quantiles"],
        "objective_names": list(OBJECTIVE_NAMES),
        "kernel_agreement": fields["kernel_agreement"],
        "baseline": fields["baseline"],
    }
    manifest_path = paths.grid_npz.with_suffix(".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Saved manifest %s", manifest_path)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_manifold(grid_npz: Path, expected_docs: int, expected_points: int) -> bool:
    """Check shapes, bounds, objective ranges, and the Eq. 21 aggregation."""
    if not grid_npz.is_file():
        LOGGER.error("Missing %s", grid_npz)
        return False

    with np.load(grid_npz) as data:
        grid = data["grid"]
        objectives = data["objectives"]
        doc_coords = data["doc_coords"]
        weights = data["weights_global"]
        view_arrays = {name: data[f"b_{name}"] for name, _, _ in VIEW_SPECS}

    ok = True
    if grid.shape != (expected_points, 2):
        LOGGER.error("Grid shape %s != (%d, 2)", grid.shape, expected_points)
        ok = False
    if objectives.shape != (expected_points, 4):
        LOGGER.error(
            "Objective matrix shape %s != (%d, 4)", objectives.shape, expected_points
        )
        ok = False
    if doc_coords.shape != (expected_docs, 2):
        LOGGER.error("Doc coords shape %s != (%d, 2)", doc_coords.shape, expected_docs)
        ok = False
    if weights.shape != (expected_points, expected_docs):
        LOGGER.error(
            "Weights shape %s != (%d, %d)",
            weights.shape,
            expected_points,
            expected_docs,
        )
        ok = False
    for name, values in view_arrays.items():
        if values.shape != (expected_points,):
            LOGGER.error("Boundary view %s has shape %s", name, values.shape)
            ok = False

    if grid.min() < -1.0 - 1e-9 or grid.max() > 1.0 + 1e-9:
        LOGGER.error("Grid leaves the chart: [%.3f, %.3f]", grid.min(), grid.max())
        ok = False
    if doc_coords.min() < -1.0 - 1e-9 or doc_coords.max() > 1.0 + 1e-9:
        LOGGER.error(
            "Document coordinates leave the chart: [%.3f, %.3f]",
            doc_coords.min(),
            doc_coords.max(),
        )
        ok = False
    if not np.all(np.isfinite(objectives)):
        LOGGER.error("Objective matrix contains non-finite values")
        ok = False
    if objectives.min() < -1e-9 or objectives.max() > 1.0 + 1e-9:
        LOGGER.error(
            "Objectives outside [0, 1]: [%.4f, %.4f]",
            objectives.min(),
            objectives.max(),
        )
        ok = False
    if not np.all(np.isfinite(weights)):
        LOGGER.error("SPH weight matrix contains non-finite values")
        ok = False

    # Eq. 21 must be reconstructible from the stored views alone.
    recomputed = persistent_boundary(view_arrays)
    drift = float(np.max(np.abs(recomputed - objectives[:, 0])))
    LOGGER.info("Eq. 21 reconstruction drift: %.3e", drift)
    if drift > 1e-12:
        LOGGER.error(
            "P_boundary is not the harmonic mean of the three stored views "
            "(max drift %.3e)",
            drift,
        )
        ok = False

    per_objective = ", ".join(
        f"{name}=[{objectives[:, index].min():.3f}, {objectives[:, index].max():.3f}]"
        for index, name in enumerate(OBJECTIVE_NAMES)
    )
    LOGGER.info("Objectives: %s", per_objective)
    LOGGER.info(
        "W > 0 at %d/%d grid points", int((objectives[:, 1] > 0).sum()), expected_points
    )
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Knowledge Manifold and the four discovery fields."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify data/manifold_grid.npz only"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _persist_discovered_regimes(
    paths: Paths,
    config: dict[str, Any],
    corpus: dict[str, Any],
    papers: list[dict[str, Any]],
    discovered: Any,
) -> None:
    """Write the discovered taxonomy back to the corpus and to its artifact.

    Two files are written, and both are needed.  ``data/regimes_auto.json`` is the
    standalone record of what was discovered (both resolutions, their evidence
    terms, and the query); the corpus rewrite is what the report and the
    dashboard actually read, because they take a paper's label from the paper.
    The full-text corpus is rewritten too, from the same labels, so the two
    corpus files cannot disagree about who belongs to which region.
    """
    settings = single_field_settings(config)
    query = str(settings.get("query", "")) or None
    artifact_path = paths.grid_npz.parent / "regimes_auto.json"
    write_artifact(artifact_path, discovered, query=query)

    counts: dict[str, int] = {}
    for paper in papers:
        label = str(paper["regime_label"])
        counts[label] = counts.get(label, 0) + 1

    corpus.setdefault("metadata", {})["regimes_auto"] = discovered.as_metadata(query=query)
    corpus["metadata"]["regime_counts"] = counts
    corpus["metadata"]["n_seeds"] = sum(1 for paper in papers if paper.get("is_seed"))

    # corpus.json carries metadata only; corpus_fulltext.json adds the text.  The
    # text is taken from the existing file so this rewrite never touches it.
    fulltext_by_id: dict[str, str] = {}
    if paths.corpus_fulltext.is_file():
        try:
            with paths.corpus_fulltext.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Cannot read %s to refresh regimes: %s", paths.corpus_fulltext, exc)
            previous = {}
        for entry in previous.get("papers", []):
            text = entry.get("full_text")
            if isinstance(text, str) and text:
                fulltext_by_id[str(entry.get("id", ""))] = text

    _write_json(
        paths.corpus,
        {
            # ``generated_by`` already names the build that produced this corpus
            # (seed-first or single-field); keep it rather than re-stamping it.
            "metadata": dict(corpus["metadata"]),
            "papers": [
                {key: value for key, value in paper.items() if key != "full_text"}
                for paper in papers
            ],
        },
    )

    fulltext_papers: list[dict[str, Any]] = []
    for paper in papers:
        entry = dict(paper)
        text = fulltext_by_id.get(str(paper.get("id", "")))
        if text:
            entry["full_text"] = text
        fulltext_papers.append(entry)
    _write_json(
        paths.corpus_fulltext,
        {"metadata": dict(corpus["metadata"]), "papers": fulltext_papers},
    )
    LOGGER.info(
        "Rewrote %s and %s with %d discovered regions",
        paths.corpus,
        paths.corpus_fulltext,
        discovered.discovery_k,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write, so a crash cannot leave a half-written corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths = resolve_paths(config, args.config)

    if not paths.corpus.is_file():
        LOGGER.error("Corpus not found: %s", paths.corpus)
        return 1

    try:
        with paths.corpus.open("r", encoding="utf-8") as handle:
            corpus = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Could not read corpus %s: %s", paths.corpus, exc)
        return 1
    papers = corpus["papers"]

    manifold_cfg = config.get("manifold", {})
    grid_size = int(manifold_cfg.get("grid_size", 21))
    expected_points = grid_size * grid_size

    if args.verify_only:
        return 0 if verify_manifold(paths.grid_npz, len(papers), expected_points) else 1

    try:
        representation = load_representation(paths.tfidf_matrix, paths.tfidf_vocab)
    except SystemExit as exc:
        LOGGER.error("%s", exc)
        return 2
    if representation.doc_ids != [str(paper["id"]) for paper in papers]:
        LOGGER.error(
            "TF-IDF row order does not match corpus order; re-run 'python -m src.embed'"
        )
        return 2

    regimes = np.array([int(paper["regime_id"]) for paper in papers], dtype=np.int64)
    n_regimes = len(corpus_regime_labels(papers)) or (int(regimes.max()) + 1)

    LOGGER.info(
        "Building manifold for %d documents across %d domain regimes "
        "(K=5 and K=8 clusterings are computed separately)",
        len(papers),
        n_regimes,
    )
    manifold = build_manifold(gram_matrix(representation.z_l2), config)

    discovered = None
    if is_single_field(config):
        # Zero-Seed Mode: there is no manual taxonomy, so the regions are read off
        # the chart after projection and named from their own words.  The corpus
        # is rewritten here because every renderer reads its labels from the
        # papers; doing it anywhere later would let the chart and the legend
        # disagree for one stage.
        #
        # Naming uses the whole-word vocabulary, not the n-gram features that
        # build the representation: an n-gram label reads as ``obotwin · priva``,
        # which names nothing.  Imported here rather than at module scope because
        # ``src.report`` pulls in the rendering stack this stage does not need.
        from .report import build_vocabulary

        vocabulary = build_vocabulary(representation)
        discovered = discover_regimes(
            coords=manifold.coords,
            vocabulary=vocabulary,
            discovery_k=config_int(single_field_settings(config), "discovery_k", 5),
            reference_k=config_int(single_field_settings(config), "reference_k", 8),
        )
        apply_to_papers(papers, discovered)
        regimes = np.array(
            [int(paper["regime_id"]) for paper in papers], dtype=np.int64
        )
        n_regimes = discovered.discovery_k
        LOGGER.info("Discovered regimes -> %s", summarise(discovered))

    fields = compute_objectives(config, representation, regimes, n_regimes, manifold)
    save_grid(paths, manifold, fields, corpus)

    if discovered is not None:
        _persist_discovered_regimes(paths, config, corpus, papers, discovered)

    return 0 if verify_manifold(paths.grid_npz, len(papers), expected_points) else 1


if __name__ == "__main__":
    sys.exit(main())
