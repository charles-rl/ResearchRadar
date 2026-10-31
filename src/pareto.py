"""Exact Pareto dominance search over the four KMO discovery objectives.

Implements PLAN.md Phase 5 for Obayashi et al. (2026), Sections 4.4-4.6:

* Eq. 23 - dominance: ``x_a`` dominates ``x_b`` if it is no worse in every
  objective and strictly better in at least one.
* Eq. 24 - the Pareto set ``P`` of non-dominated grid points.
* Eq. 25 - the practical Pareto front ``P_practical = {x in P : W(x) > 0}``.
  Because white-space novelty is band-pass filtered, ``W(x) = 0`` marks either
  regions too close to existing documents or regions too remote to count as
  supported gaps, so those points are mathematically non-dominated but
  scientifically uninteresting.
* Section 4.6 - with only 441 candidates the front is obtained by exact pairwise
  comparison; no evolutionary optimizer is needed and no stochasticity is added.

Ranking is by Euclidean distance to the ideal corner ``(1, 1, 1, 1)``. Per
Section 4.7 the front is also a *structured set*, not a ranked list, so points are
grouped into discovery archetypes by chart proximity and the best-ranked member of
each group is reported as its representative.

Usage
-----
    python -m src.pareto
    python -m src.pareto --verify-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import LOGGER, Paths, configure_logging, load_config, resolve_paths

DEFAULT_CONFIG = Path("config.yaml")
DEFAULT_OBJECTIVES = ("P_boundary", "W", "G", "S")


# --------------------------------------------------------------------------- #
# Dominance
# --------------------------------------------------------------------------- #
def dominance_mask(objectives: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return ``dominated[i] = True`` when some point strictly dominates ``i``.

    Exact pairwise test, vectorized over the grid: ``j`` dominates ``i`` when
    ``F_j >= F_i`` on all objectives and ``F_j > F_i`` on at least one (Eq. 23).
    The diagonal is excluded so a point cannot dominate itself.
    """
    # Shape (n, n, m). ``objectives[:, None, :]`` puts the potential dominator on
    # axis 0 and ``objectives[None, :, :]`` the victim on axis 1, so the result is
    # indexed ``dominates[dominator, victim]``. Swapping these two operands
    # silently transposes the matrix and reports "dominates something" instead of
    # "is dominated" - which flagged 439 of 441 points as dominated.
    dominator = objectives[:, None, :]
    victim = objectives[None, :, :]
    no_worse = np.all(dominator >= victim, axis=2)
    strictly_better = np.any(dominator > victim, axis=2)
    dominates = no_worse & strictly_better
    np.fill_diagonal(dominates, False)
    return dominates.any(axis=0)


def extract_pareto_front(
    objectives: NDArray[np.float64], white_space: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(pareto_indices, practical_indices)`` for the objective matrix.

    ``pareto_indices`` is the full non-dominated set ``P`` (Eq. 24);
    ``practical_indices`` keeps only members with ``W(x) > 0`` (Eq. 25).
    """
    dominated = dominance_mask(objectives)
    pareto_indices = np.flatnonzero(~dominated)
    on_chart = white_space[pareto_indices] > 0.0
    practical_indices = pareto_indices[on_chart]
    return pareto_indices, practical_indices


# --------------------------------------------------------------------------- #
# Ranking and archetypes
# --------------------------------------------------------------------------- #
def ideal_distance(
    objectives: NDArray[np.float64], ideal: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Euclidean distance from each row to the ideal corner."""
    return np.linalg.norm(objectives - ideal[None, :], axis=1)


def group_archetypes(
    points: NDArray[np.float64], radius: float
) -> list[NDArray[np.int64]]:
    """Group front points that are spatially close into discovery archetypes.

    Section 4.7 notes that nearby Pareto points usually represent the same broad
    opportunity, while separated regions correspond to qualitatively different
    discovery mechanisms. Single-linkage grouping at ``radius`` on the chart
    recovers that structure without imposing an arbitrary cluster count.
    """
    n = points.shape[0]
    if n == 0:
        return []
    parent = np.arange(n)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= radius:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    return [np.array(members, dtype=np.int64) for members in groups.values()]


def _describe_point(
    index: int,
    objectives: NDArray[np.float64],
    grid: NDArray[np.float64],
    distance: NDArray[np.float64],
    names: tuple[str, ...],
) -> dict[str, Any]:
    scores = {name: float(objectives[index, position]) for position, name in enumerate(names)}
    return {
        "grid_index": int(index),
        "x": float(grid[index, 0]),
        "y": float(grid[index, 1]),
        "objectives": scores,
        "ideal_distance": float(distance[index]),
        "on_outer_boundary": bool(
            abs(grid[index, 0]) > 0.95 or abs(grid[index, 1]) > 0.95
        ),
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def extract_front(
    config: dict[str, Any], grid_npz: Path, output_json: Path
) -> dict[str, Any]:
    """Load the objective matrix, extract the front, and write the JSON record."""
    pareto_cfg = config.get("pareto", {})
    names = tuple(pareto_cfg.get("objectives", list(DEFAULT_OBJECTIVES)))
    ideal = np.array(
        pareto_cfg.get("ideal_point", [1.0, 1.0, 1.0, 1.0]), dtype=np.float64
    )
    min_whitespace = float(pareto_cfg.get("min_whitespace", 0.0))
    archetype_radius = float(pareto_cfg.get("archetype_radius", 0.35))

    with np.load(grid_npz) as data:
        objectives = np.asarray(data["objectives"], dtype=np.float64)
        grid = np.asarray(data["grid"], dtype=np.float64)
        stored_names = tuple(str(name) for name in data["objective_names"])

    if stored_names != names:
        LOGGER.warning(
            "Config objective order %s differs from saved %s; using saved order",
            names,
            stored_names,
        )
    names = stored_names

    white_space = objectives[:, names.index("W")]
    dominated = dominance_mask(objectives)
    pareto_indices = np.flatnonzero(~dominated)
    practical_indices = pareto_indices[white_space[pareto_indices] > min_whitespace]

    LOGGER.info(
        "Exact dominance over %d candidates: %d non-dominated, %d practical (W > %.2f)",
        objectives.shape[0],
        pareto_indices.size,
        practical_indices.size,
        min_whitespace,
    )
    if practical_indices.size == 0:
        raise SystemExit(
            "Practical Pareto front is empty; check the white-space field and "
            "min_whitespace."
        )

    distance = ideal_distance(objectives, ideal)
    order = np.argsort(distance[practical_indices])

    ranked: list[dict[str, Any]] = []
    for rank, position in enumerate(order, start=1):
        entry = _describe_point(
            int(practical_indices[position]), objectives, grid, distance, names
        )
        entry["rank"] = rank
        ranked.append(entry)

    ranked_indices = np.array(
        [entry["grid_index"] for entry in ranked], dtype=np.int64
    )
    groups = group_archetypes(grid[ranked_indices], archetype_radius)

    archetypes: list[dict[str, Any]] = []
    for members in groups:
        # Members are positions into ``ranked``; the first is already best-ranked.
        member_positions = np.sort(members)
        representative = ranked[int(member_positions[0])]
        archetypes.append(
            {
                "archetype_id": None,
                "size": int(member_positions.size),
                "representative": representative,
                "members": [ranked[int(position)] for position in member_positions],
            }
        )
    archetypes.sort(key=lambda item: item["representative"]["rank"])
    for index, archetype in enumerate(archetypes):
        archetype["archetype_id"] = index
        archetype["label"] = f"Archetype {chr(ord('A') + index)}"

    LOGGER.info(
        "Grouped %d practical points into %d archetypes at radius %.2f",
        ranked_indices.size,
        len(archetypes),
        archetype_radius,
    )
    LOGGER.info("Best compromise: %s", _format_point(ranked[0]))

    payload = {
        "metadata": {
            "method": "exact pairwise Pareto dominance over a regular 21x21 grid",
            "reference": "Obayashi et al. (2026) Knowledge Manifold Optimization, "
            "Eqs. 23-25 and Sec. 4.6",
            "objective_names": list(names),
            "ideal_point": ideal.tolist(),
            "min_whitespace": min_whitespace,
            "archetype_radius": archetype_radius,
            "candidates": int(objectives.shape[0]),
            "non_dominated": int(pareto_indices.size),
            "practical_front": int(practical_indices.size),
            "archetype_count": len(archetypes),
            "ranking": "Euclidean distance to the ideal corner (1, 1, 1, 1)",
            "grid_output": str(grid_npz),
        },
        "solutions": ranked,
        "archetypes": archetypes,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(output_json)
    LOGGER.info("Saved %s (%d solutions)", output_json, len(ranked))
    return payload


def _format_point(entry: dict[str, Any]) -> str:
    scores = " ".join(
        f"{name}={value:.3f}" for name, value in entry["objectives"].items()
    )
    return (
        f"#{entry['rank']} x=({entry['x']:+.2f}, {entry['y']:+.2f}) "
        f"{scores} d_ideal={entry['ideal_distance']:.4f}"
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_front(pareto_json: Path) -> bool:
    """Independently re-check the saved front for correctness."""
    if not pareto_json.is_file():
        LOGGER.error("Missing %s", pareto_json)
        return False

    with pareto_json.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid JSON in %s: %s", pareto_json, exc)
            return False

    solutions = payload.get("solutions", [])
    metadata = payload.get("metadata", {})
    if not solutions:
        LOGGER.error("No practical Pareto solutions recorded")
        return False

    names = tuple(metadata.get("objective_names", DEFAULT_OBJECTIVES))
    scores = np.array(
        [[solution["objectives"][name] for name in names] for solution in solutions],
        dtype=np.float64,
    )

    ok = True
    # 1. Every reported solution must be non-dominated by every other reported
    #    solution, and must satisfy W > min_whitespace.
    white = scores[:, names.index("W")]
    if np.any(white <= float(metadata.get("min_whitespace", 0.0))):
        LOGGER.error("Practical front contains a point with W <= min_whitespace")
        ok = False

    internal = dominance_mask(scores)
    if internal.any():
        worse = [solutions[i]["grid_index"] for i in np.flatnonzero(internal)]
        LOGGER.error("Front contains internally dominated solutions: %s", worse)
        ok = False

    # 2. Ranking must be monotone in ideal distance.
    distances = np.array([solution["ideal_distance"] for solution in solutions])
    ranks = np.array([solution["rank"] for solution in solutions])
    if not np.array_equal(ranks, np.arange(1, len(solutions) + 1)):
        LOGGER.error("Ranks are not 1..n in order")
        ok = False
    if np.any(np.diff(distances) < -1e-12):
        LOGGER.error("Ideal distance is not non-decreasing along the ranking")
        ok = False

    # 3. Distances must match their stored objective vectors from the ideal corner.
    ideal = np.array(metadata.get("ideal_point", [1.0, 1.0, 1.0, 1.0]))
    recomputed = np.linalg.norm(scores - ideal[None, :], axis=1)
    drift = float(np.abs(recomputed - distances).max())
    if drift > 1e-9:
        LOGGER.error("Stored ideal distance drift %.3e exceeds 1e-9", drift)
        ok = False

    # 4. Archetype membership must partition the solutions exactly.
    grouped = [
        member["grid_index"]
        for archetype in payload.get("archetypes", [])
        for member in archetype["members"]
    ]
    if sorted(grouped) != sorted(solution["grid_index"] for solution in solutions):
        LOGGER.error("Archetype grouping does not partition the practical front")
        ok = False

    LOGGER.info(
        "Practical front: %d solutions, %d archetypes, distance drift %.3e, "
        "objective range P=[%.3f, %.3f] W=[%.3f, %.3f] G=[%.3f, %.3f] "
        "S=[%.3f, %.3f]",
        len(solutions),
        len(payload.get("archetypes", [])),
        drift,
        scores[:, 0].min(),
        scores[:, 0].max(),
        white.min(),
        white.max(),
        scores[:, 2].min(),
        scores[:, 2].max(),
        scores[:, 3].min(),
        scores[:, 3].max(),
    )
    LOGGER.info("Best compromise: %s", _format_point(solutions[0]))
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the practical Pareto front of KMO discovery gaps."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify data/pareto_solutions.json"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths: Paths = resolve_paths(config, args.config)

    if args.verify_only:
        return 0 if verify_front(paths.pareto_json) else 1

    if not paths.grid_npz.is_file():
        LOGGER.error(
            "Missing %s. Run 'python -m src.manifold' first.", paths.grid_npz
        )
        return 1

    extract_front(config, paths.grid_npz, paths.pareto_json)
    return 0 if verify_front(paths.pareto_json) else 1


if __name__ == "__main__":
    raise SystemExit(main())
