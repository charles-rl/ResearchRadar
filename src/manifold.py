"""Knowledge Manifold construction and the four KMO discovery fields.

Implements PLAN.md Phase 4 for Obayashi et al. (2026), "Knowledge Manifold
Optimization":

* Section 2.1 / 5.2 - a fixed bounded chart ``Omega = [-1, 1]^2``. Nine
  automatically selected anchor documents (eight perimeter, one central) fix the
  coordinate gauge; the remaining documents are placed by constrained stress
  minimization seeded with classical MDS. Coordinates are then frozen.
* Section 3.3 - knowledge-regime boundary field ``B_K`` from normalized cluster
  entropy ``H_K`` and the dominant-second probability gap ``Dp_K``.
* Section 4.2 - persistent boundary objective ``P_boundary`` as the harmonic mean
  of the global and adaptive boundary strengths.
* Section 3.4 - band-pass white-space novelty ``W`` over nearest-document
  distance quantiles ``q20 < q60 <= q85 < q98``.
* Section 3.5 - knowledge-gradient field ``G`` from the Frobenius norm of the
  semantic field Jacobian, computed by central differences.
* Section 3.6 - evidential support ``S`` from inverse Simpson ``N_eff``, the
  fifth-nearest document distance ``d5``, and top-three weight mass ``T3``.
* Section 3.7 - Gaussian-process relative uncertainty ``u``, retained as a
  reliability diagnostic (not a Pareto objective, per Section 4.3).
* Section 5.5 - a 21 x 21 evaluation grid, i.e. 441 candidates at spacing 0.1.
* Equations 9-10 - ``N_+``/``N_-`` percentile normalization at the 10th/90th
  percentiles.

The whitened vector seeds in ``_RNG`` are not used for any field; every value is
deterministic given the corpus and embeddings.

Usage
-----
    python -m src.manifold
    python -m src.manifold --verify-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import LOGGER, Paths, configure_logging, load_config, resolve_paths

DEFAULT_CONFIG = Path("config.yaml")
OBJECTIVE_NAMES = ("P_boundary", "W", "G", "S")

# Obayashi et al. Eq. 11 weights.
ENTROPY_WEIGHT = 0.65
GAP_WEIGHT = 0.35
# Obayashi et al. Eq. 18 weights.
SUPPORT_W_NEFF = 0.40
SUPPORT_W_D5 = 0.30
SUPPORT_W_T3 = 0.30


@dataclass
class Manifold:
    """Frozen document coordinates plus chart metadata."""

    coords: NDArray[np.float64]
    anchor_index: NDArray[np.int64]
    anchor_target: NDArray[np.float64]
    stress: float
    distance_correlation: float


# --------------------------------------------------------------------------- #
# Document coordinates: MDS + gauge-fixing anchors + constrained stress
# --------------------------------------------------------------------------- #
def cosine_distance_matrix(embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return the pairwise cosine distance matrix of unit-norm rows."""
    unit = embeddings.astype(np.float64)
    norms = np.linalg.norm(unit, axis=1, keepdims=True)
    unit = unit / np.where(norms <= 0.0, 1.0, norms)
    similarity = unit @ unit.T
    return np.clip(1.0 - similarity, 0.0, 2.0)


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
    radius = 0.92
    targets = [
        [radius * np.cos(angle), radius * np.sin(angle)] for angle in angles
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
) -> Manifold:
    """Minimize metric stress with the anchor coordinates pinned.

    Stress is ``sum_{i<j} (||x_i - x_j|| - D_ij)^2`` over all document pairs, with
    the nine anchors held at their gauge-fixing targets. The free coordinates are
    bounded to the chart so no document leaves ``[-1, 1]^2``.
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

        # d/dx_i of sum_{i<j} r_ij^2 = 2 * sum_{j!=i} r_ij (x_i - x_j)/d_ij
        coefficient = np.where(dist <= 1e-12, 0.0, residual / safe)
        gradient_full = 2.0 * np.einsum("ij,ijk->ik", coefficient, delta)
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
    final_dist = np.linalg.norm(
        final[:, None, :] - final[None, :, :], axis=-1
    )
    original = distances[pair_mask]
    embedded = final_dist[pair_mask]
    # Shepard-style agreement between embedding and true distances.
    corr = float(
        np.corrcoef(
            np.argsort(np.argsort(original)), np.argsort(np.argsort(embedded))
        )[0, 1]
    )
    LOGGER.info(
        "Stress minimization: final stress=%.5f, Spearman(distance)=%.4f, "
        "converged=%s (%d iterations)",
        float(result.fun),
        corr,
        bool(result.success),
        int(result.nit),
    )
    return Manifold(
        coords=final,
        anchor_index=anchor_index,
        anchor_target=anchor_target,
        stress=float(result.fun),
        distance_correlation=corr,
    )


def build_manifold(
    embeddings: NDArray[np.float32], config: dict[str, Any]
) -> Manifold:
    """Construct the frozen Knowledge Manifold for the corpus."""
    manifold_cfg = config.get("manifold", {})
    n_perimeter = int(manifold_cfg.get("perimeter_anchors", 8))
    bounds_cfg = manifold_cfg.get("bounds", [-1.0, 1.0])
    bounds = (float(bounds_cfg[0]), float(bounds_cfg[1]))

    distances = cosine_distance_matrix(embeddings)
    init = _scale_to_chart(classical_mds(distances, n_components=2))
    anchor_index, anchor_target = select_anchors(init, n_perimeter=n_perimeter)
    manifold = constrained_stress(
        distances, init, anchor_index, anchor_target, bounds=bounds
    )

    LOGGER.info(
        "Manifold: %d documents on [%.2f, %.2f]^2, bounds x=[%.3f, %.3f] "
        "y=[%.3f, %.3f]",
        manifold.coords.shape[0],
        bounds[0],
        bounds[1],
        manifold.coords[:, 0].min(),
        manifold.coords[:, 0].max(),
        manifold.coords[:, 1].min(),
        manifold.coords[:, 1].max(),
    )
    return manifold


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
    bandwidth: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """SPH kernel weights ``exp(-||x - x_i||^2 / (2 h^2))``.

    ``bandwidth`` may be a scalar (global field) or one value per grid point
    (adaptive field).
    """
    squared = pairwise_distances(grid, coords) ** 2
    h = np.asarray(bandwidth, dtype=np.float64)
    if h.ndim == 0:
        return np.exp(-squared / (2.0 * float(h) ** 2))
    return np.exp(-squared / (2.0 * h[:, None] ** 2))


def adaptive_bandwidth(
    grid: NDArray[np.float64], coords: NDArray[np.float64], nearest_k: int
) -> NDArray[np.float64]:
    """Local scale: mean distance to the ``nearest_k`` nearest documents."""
    distances = pairwise_distances(grid, coords)
    k = min(max(1, nearest_k), distances.shape[1])
    nearest = np.partition(distances, k - 1, axis=1)[:, :k]
    return np.maximum(nearest.mean(axis=1), 1e-6)


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


def regime_mixture(
    weights: NDArray[np.float64], regimes: NDArray[np.int64], n_regimes: int
) -> NDArray[np.float64]:
    """Normalized regime probability vector for each grid point (Eq. 6)."""
    totals = weights.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 1e-300, 1.0, totals)
    normalized = weights / totals
    mixture = np.zeros((weights.shape[0], n_regimes), dtype=np.float64)
    for regime in range(n_regimes):
        mixture[:, regime] = normalized[:, regimes == regime].sum(axis=1)
    return mixture


def boundary_strength(
    mixture: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return raw normalized entropy ``H_K`` and dominant-second gap ``Dp_K``."""
    n_regimes = mixture.shape[1]
    safe = np.clip(mixture, 1e-12, None)
    entropy = -(mixture * np.log(safe)).sum(axis=1) / np.log(n_regimes)
    entropy = np.clip(entropy, 0.0, 1.0)
    ordered = np.sort(mixture, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    return entropy, gap


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

    The paper defines these empirically over the search grid. The values recorded
    in ``config.yaml`` come from the paper's aerodynamic corpus and are kept as a
    documented reference; with ``use_empirical_quantiles: true`` (the default) the
    quantiles are recomputed for this corpus so the band-pass window matches its
    own sampling density.
    """
    manifold_cfg = config.get("manifold", {})
    configured = manifold_cfg.get("q_whitespace", {}) or {}
    empirical = {
        "q20": float(np.percentile(d_nn, 20)),
        "q60": float(np.percentile(d_nn, 60)),
        "q85": float(np.percentile(d_nn, 85)),
        "q98": float(np.percentile(d_nn, 98)),
    }
    if bool(manifold_cfg.get("use_empirical_quantiles", True)):
        LOGGER.info(
            "d_nn quantiles (empirical): q20=%.4f q60=%.4f q85=%.4f q98=%.4f",
            empirical["q20"],
            empirical["q60"],
            empirical["q85"],
            empirical["q98"],
        )
        if configured:
            LOGGER.info(
                "d_nn quantiles (config reference, paper corpus): "
                "q20=%.4f q60=%.4f q85=%.4f q98=%.4f",
                float(configured.get("q20", 0.0)),
                float(configured.get("q60", 0.0)),
                float(configured.get("q85", 0.0)),
                float(configured.get("q98", 0.0)),
            )
        return empirical

    merged = dict(empirical)
    merged.update({key: float(value) for key, value in configured.items()})
    LOGGER.info(
        "d_nn quantiles (config): q20=%.4f q60=%.4f q85=%.4f q98=%.4f",
        merged["q20"],
        merged["q60"],
        merged["q85"],
        merged["q98"],
    )
    return merged


def semantic_field(
    weights: NDArray[np.float64], embeddings: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Interpolated semantic field ``v(x) = sum_i w_i v_i / sum_i w_i``."""
    totals = weights.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 1e-300, 1.0, totals)
    return (weights / totals) @ embeddings.astype(np.float64)


def gradient_magnitude(
    field: NDArray[np.float64], size: int, spacing: float
) -> NDArray[np.float64]:
    """Frobenius norm of the field Jacobian by central differences (Eq. 14)."""
    cube = field.reshape(size, size, -1)
    grad_x, grad_y = np.gradient(cube, spacing, spacing, axis=(0, 1))  # type: ignore[misc]
    d_dx = np.asarray(grad_x, dtype=np.float64)
    d_dy = np.asarray(grad_y, dtype=np.float64)
    squared = (d_dx**2).sum(axis=-1) + (d_dy**2).sum(axis=-1)
    return np.sqrt(squared).ravel()


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
    embeddings: NDArray[np.float32],
    grid: NDArray[np.float64],
    coords: NDArray[np.float64],
    n_components: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Relative uncertainty from two GPR surrogates (Section 3.7, Eq. 19).

    A Gaussian process is fitted on a PCA reduction of the document embeddings
    and its predictive standard deviation is evaluated on the grid. As in the
    paper, two covariance assumptions are used - an anisotropic-baseline RBF and
    a Matern-3/2 - and their low-uncertainty scores are averaged into ``R``.

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
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    unit = embeddings.astype(np.float64)
    components = min(n_components, unit.shape[0] - 1, unit.shape[1])
    reduced = PCA(n_components=components, random_state=0).fit_transform(unit)

    pairwise = pairwise_distances(coords, coords)
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    length_scale = float(np.median(upper)) if upper.size else 1.0
    length_scale = max(length_scale, 1e-3)
    LOGGER.info(
        "GPR diagnostic: fixed length_scale=%.4f (median inter-document distance), "
        "%d PCA components",
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
        base
        * Matern(length_scale=length_scale, nu=1.5, length_scale_bounds="fixed")
        + noise
    )
    return {"u_rbf": u_rbf, "u_matern": u_matern, "u_mean": 0.5 * (u_rbf + u_matern)}


# --------------------------------------------------------------------------- #
# Objective assembly
# --------------------------------------------------------------------------- #
def compute_objectives(
    config: dict[str, Any],
    embeddings: NDArray[np.float32],
    regimes: NDArray[np.int64],
    n_regimes: int,
    manifold: Manifold,
) -> dict[str, Any]:
    """Compute the four discovery fields over the evaluation grid."""
    manifold_cfg = config.get("manifold", {})
    grid = build_grid(config)
    coords = manifold.coords
    bounds_cfg = manifold_cfg.get("bounds", [-1.0, 1.0])
    spacing = float(
        (float(bounds_cfg[1]) - float(bounds_cfg[0]))
        / (int(manifold_cfg.get("grid_size", 21)) - 1)
    )

    h_global = float(manifold_cfg.get("sph_h", 0.25))
    nearest_k = int(manifold_cfg.get("nearest_k", 8))

    w_global = sph_weights(grid, coords, h_global)
    h_adaptive = adaptive_bandwidth(grid, coords, nearest_k)
    w_adaptive = sph_weights(grid, coords, h_adaptive)

    mix_global = regime_mixture(w_global, regimes, n_regimes)
    mix_adaptive = regime_mixture(w_adaptive, regimes, n_regimes)

    h_g, gap_g = boundary_strength(mix_global)
    h_a, gap_a = boundary_strength(mix_adaptive)

    n_h_g, p_h_g = normalize_percentile(h_g)
    n_gap_g, p_gap_g = normalize_percentile(gap_g)
    n_h_a, p_h_a = normalize_percentile(h_a)
    n_gap_a, p_gap_a = normalize_percentile(gap_a)

    b_global = ENTROPY_WEIGHT * n_h_g + GAP_WEIGHT * (1.0 - n_gap_g)
    b_adaptive = ENTROPY_WEIGHT * n_h_a + GAP_WEIGHT * (1.0 - n_gap_a)

    # Harmonic mean is the AND-like persistence operator (Eq. 21). A zero in
    # either view must give zero, so guard the reciprocal explicitly.
    denominator = np.where(
        (b_global > 0) & (b_adaptive > 0),
        1.0 / np.where(b_global > 0, b_global, 1.0)
        + 1.0 / np.where(b_adaptive > 0, b_adaptive, 1.0),
        np.inf,
    )
    p_boundary = np.where(np.isinf(denominator), 0.0, 2.0 / denominator)

    d_nn = pairwise_distances(grid, coords).min(axis=1)
    quantiles = empirical_dnn_quantiles(d_nn, config)
    whitespace = band_pass_whitespace(d_nn, quantiles)

    field = semantic_field(w_global, embeddings)
    gradient_raw = gradient_magnitude(field, int(manifold_cfg.get("grid_size", 21)), spacing)
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

    uncertainty_fields = gpr_uncertainty(embeddings, grid, coords)
    n_u_rbf, _ = normalize_percentile(uncertainty_fields["u_rbf"])
    n_u_matern, _ = normalize_percentile(uncertainty_fields["u_matern"])
    reliability = 0.5 * (1.0 - n_u_rbf) + 0.5 * (1.0 - n_u_matern)
    uncertainty = uncertainty_fields["u_mean"]

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
        "Field stats: H_global=[%.3f, %.3f] H_adaptive=[%.3f, %.3f] "
        "d_nn=[%.4f, %.4f] N_eff=[%.2f, %.2f] d5=[%.4f, %.4f] T3=[%.3f, %.3f]",
        h_g.min(),
        h_g.max(),
        h_a.min(),
        h_a.max(),
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
        "GPR diagnostic: u=[%.5f, %.5f], R=[%.3f, %.3f] (not a Pareto objective)",
        uncertainty.min(),
        uncertainty.max(),
        reliability.min(),
        reliability.max(),
    )

    return {
        "grid": grid,
        "objectives": objectives,
        "d_nn": d_nn,
        "whitespace_quantiles": quantiles,
        "h_global": h_g,
        "h_adaptive": h_a,
        "gap_global": gap_g,
        "gap_adaptive": gap_a,
        "b_global": b_global,
        "b_adaptive": b_adaptive,
        "mixture_global": mix_global,
        "mixture_adaptive": mix_adaptive,
        "weights_global": w_global,
        "h_adaptive_bandwidth": h_adaptive,
        "n_eff": n_eff,
        "d5": d5,
        "top3": top_three,
        "gradient_raw": gradient_raw,
        "uncertainty": uncertainty,
        "uncertainty_rbf": uncertainty_fields["u_rbf"],
        "uncertainty_matern": uncertainty_fields["u_matern"],
        "reliability": reliability,
        "percentiles": {
            "H_global": p_h_g,
            "gap_global": p_gap_g,
            "H_adaptive": p_h_a,
            "gap_adaptive": p_gap_a,
            "gradient": p_gradient,
            "n_eff": p_neff,
            "d5": p_d5,
            "top3": p_t3,
        },
        "spacing": spacing,
        "sph_h": h_global,
        "nearest_k": nearest_k,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_grid(paths: Paths, manifold: Manifold, fields: dict[str, Any], corpus: dict[str, Any]) -> None:
    """Write the projection and objective matrix to ``data/manifold_grid.npz``."""
    paths.grid_npz.parent.mkdir(parents=True, exist_ok=True)
    papers = corpus["papers"]

    np.savez_compressed(
        paths.grid_npz,
        grid=fields["grid"],
        objectives=fields["objectives"],
        objective_names=np.array(OBJECTIVE_NAMES),
        doc_coords=manifold.coords,
        doc_regime=np.array([int(paper["regime_id"]) for paper in papers], dtype=np.int64),
        doc_ids=np.array([str(paper["id"]) for paper in papers]),
        anchor_index=manifold.anchor_index,
        anchor_target=manifold.anchor_target,
        weights_global=fields["weights_global"].astype(np.float32),
        d_nn=fields["d_nn"],
        d5=fields["d5"],
        n_eff=fields["n_eff"],
        top3=fields["top3"],
        gradient_raw=fields["gradient_raw"],
        uncertainty=fields["uncertainty"],
        uncertainty_rbf=fields["uncertainty_rbf"],
        uncertainty_matern=fields["uncertainty_matern"],
        reliability=fields["reliability"],
        mixture_global=fields["mixture_global"],
        mixture_adaptive=fields["mixture_adaptive"],
        h_global_entropy=fields["h_global"],
        h_adaptive_entropy=fields["h_adaptive"],
        gap_global=fields["gap_global"],
        gap_adaptive=fields["gap_adaptive"],
        b_global=fields["b_global"],
        b_adaptive=fields["b_adaptive"],
        h_adaptive_bandwidth=fields["h_adaptive_bandwidth"],
        whitespace_quantiles=np.array(
            [fields["whitespace_quantiles"][key] for key in ("q20", "q60", "q85", "q98")]
        ),
        stress=np.array([manifold.stress]),
        distance_correlation=np.array([manifold.distance_correlation]),
        grid_size=np.array([int(np.sqrt(len(fields["grid"])))]),
        spacing=np.array([fields["spacing"]]),
        sph_h=np.array([fields["sph_h"]]),
        nearest_k=np.array([fields["nearest_k"]]),
    )
    LOGGER.info(
        "Saved %s: grid=%s objectives=%s",
        paths.grid_npz,
        fields["grid"].shape,
        fields["objectives"].shape,
    )

    manifest = {
        "grid_size": int(np.sqrt(len(fields["grid"]))),
        "grid_points": len(fields["grid"]),
        "documents": int(manifold.coords.shape[0]),
        "spacing": fields["spacing"],
        "sph_h_global": fields["sph_h"],
        "nearest_k": fields["nearest_k"],
        "stress": manifold.stress,
        "distance_correlation": manifold.distance_correlation,
        "anchors": {
            "indices": [int(i) for i in manifold.anchor_index],
            "targets": manifold.anchor_target.tolist(),
        },
        "whitespace_quantiles": fields["whitespace_quantiles"],
        "objective_names": list(OBJECTIVE_NAMES),
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
    """Check shapes, bounds, and objective ranges of the saved grid."""
    if not grid_npz.is_file():
        LOGGER.error("Missing %s", grid_npz)
        return False

    with np.load(grid_npz) as data:
        grid = data["grid"]
        objectives = data["objectives"]
        doc_coords = data["doc_coords"]
        weights = data["weights_global"]

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
        LOGGER.error(
            "Doc coords shape %s != (%d, 2)", doc_coords.shape, expected_docs
        )
        ok = False
    if weights.shape != (expected_points, expected_docs):
        LOGGER.error(
            "Weights shape %s != (%d, %d)",
            weights.shape,
            expected_points,
            expected_docs,
        )
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths = resolve_paths(config, args.config)

    if not paths.corpus.is_file():
        LOGGER.error("Corpus not found: %s", paths.corpus)
        return 1
    if not paths.embeddings.is_file():
        LOGGER.error(
            "Embeddings not found: %s. Run 'python -m src.embed' first.", paths.embeddings
        )
        return 1

    try:
        with paths.corpus.open("r", encoding="utf-8") as handle:
            corpus = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("Could not read corpus %s: %s", paths.corpus, exc)
        return 1
    papers = corpus["papers"]
    embeddings = np.load(paths.embeddings)

    if embeddings.shape[0] != len(papers):
        LOGGER.error(
            "Embedding rows (%d) != corpus papers (%d)", embeddings.shape[0], len(papers)
        )
        return 1

    manifold_cfg = config.get("manifold", {})
    grid_size = int(manifold_cfg.get("grid_size", 21))
    expected_points = grid_size * grid_size

    if args.verify_only:
        return (
            0
            if verify_manifold(paths.grid_npz, len(papers), expected_points)
            else 1
        )

    regimes = np.array([int(paper["regime_id"]) for paper in papers], dtype=np.int64)
    n_regimes = len({int(regime["id"]) for regime in config.get("regimes", [])}) or (
        int(regimes.max()) + 1
    )

    LOGGER.info(
        "Building manifold for %d documents across %d regimes", len(papers), n_regimes
    )
    manifold = build_manifold(embeddings, config)
    fields = compute_objectives(config, embeddings, regimes, n_regimes, manifold)
    save_grid(paths, manifold, fields, corpus)

    return 0 if verify_manifold(paths.grid_npz, len(papers), expected_points) else 1


if __name__ == "__main__":
    raise SystemExit(main())
