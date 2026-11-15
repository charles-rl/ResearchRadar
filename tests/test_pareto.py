"""Tests for exact Pareto dominance and discovery-archetype grouping.

Both are places where an array-orientation slip produces a plausible-looking but
wrong answer instead of a crash, so each is pinned against a brute-force
reference.
"""

from __future__ import annotations

import numpy as np

from src.pareto import (
    dominance_mask,
    extract_pareto_front,
    group_archetypes,
    ideal_distance,
)


# --------------------------------------------------------------------------- #
# Dominance
# --------------------------------------------------------------------------- #
def test_a_point_dominated_in_one_objective_is_dominated() -> None:
    objectives = np.array([[1.0, 1.0], [1.0, 0.5]])
    # Row 1 is worse on the second objective and no better on the first.
    assert list(dominance_mask(objectives)) == [False, True]


def test_a_point_better_everywhere_dominates() -> None:
    objectives = np.array([[1.0, 1.0], [0.5, 0.5]])
    assert list(dominance_mask(objectives)) == [False, True]


def test_equal_points_do_not_dominate_each_other() -> None:
    """Dominance is strict in at least one objective (Eq. 23)."""
    objectives = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert list(dominance_mask(objectives)) == [False, False]


def test_trade_off_points_both_survive() -> None:
    objectives = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert list(dominance_mask(objectives)) == [False, False]


def test_dominance_matches_a_bruteforce_reference() -> None:
    """The vectorised mask must agree with an explicit double loop."""
    rng = np.random.default_rng(3)
    objectives = rng.random((60, 4))

    expected = np.zeros(len(objectives), dtype=bool)
    for victim in range(len(objectives)):
        for other in range(len(objectives)):
            if victim == other:
                continue
            no_worse = bool(np.all(objectives[other] >= objectives[victim]))
            strictly_better = bool(np.any(objectives[other] > objectives[victim]))
            if no_worse and strictly_better:
                expected[victim] = True
                break

    assert np.array_equal(dominance_mask(objectives), expected)


def test_transposing_the_broadcast_would_flip_the_answer() -> None:
    """Guards the exact bug that once flagged 439 of 441 points as dominated.

    With the operands transposed the mask answers "dominates something" instead
    of "is dominated", so the single best point is wrongly flagged.
    """
    objectives = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]])
    mask = dominance_mask(objectives)
    assert not mask[0], "the global optimum must never be reported as dominated"
    assert mask[1] and mask[2]


# --------------------------------------------------------------------------- #
# Practical front
# --------------------------------------------------------------------------- #
def test_practical_front_excludes_zero_whitespace_points() -> None:
    """Eq. 25: a non-dominated point with W = 0 is not practically usable."""
    objectives = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]])
    white_space = np.array([0.5, 0.0, 0.4])
    _, practical = extract_pareto_front(objectives, white_space)
    # Only row 0 is non-dominated; rows 1 and 2 are dominated, and row 0 has W>0.
    assert list(practical) == [0]


def test_practical_front_is_a_subset_of_the_full_front() -> None:
    rng = np.random.default_rng(5)
    objectives = rng.random((80, 4))
    white_space = (rng.random(80) > 0.5).astype(float)
    full, practical = extract_pareto_front(objectives, white_space)
    assert set(practical.tolist()) <= set(full.tolist())


def test_zero_whitespace_removes_a_non_dominated_point() -> None:
    objectives = np.array([[1.0, 1.0], [0.0, 0.0]])
    white_space = np.array([0.0, 0.2])
    full, practical = extract_pareto_front(objectives, white_space)
    assert list(full) == [0]
    assert practical.size == 0


# --------------------------------------------------------------------------- #
# Ideal-corner ranking
# --------------------------------------------------------------------------- #
def test_ideal_distance_is_euclidean_to_the_corner() -> None:
    objectives = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    distances = ideal_distance(objectives, np.ones(4))
    assert distances[0] == 0.0
    assert np.isclose(distances[1], 2.0)


def test_ideal_distance_ranks_the_better_point_first() -> None:
    objectives = np.array([[0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5]])
    distances = ideal_distance(objectives, np.ones(4))
    assert int(np.argmin(distances)) == 0


# --------------------------------------------------------------------------- #
# Archetype grouping
# --------------------------------------------------------------------------- #
def test_empty_input_gives_no_archetypes() -> None:
    assert group_archetypes(np.zeros((0, 2)), 0.35) == []


def test_one_point_is_one_archetype() -> None:
    groups = group_archetypes(np.array([[0.1, 0.2]]), 0.35)
    assert len(groups) == 1
    assert list(groups[0]) == [0]


def test_distant_points_form_separate_archetypes() -> None:
    points = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9], [0.85, 0.9]])
    groups = group_archetypes(points, 0.35)
    assert len(groups) == 2
    assert sorted(len(group) for group in groups) == [2, 2]


def test_complete_linkage_does_not_chain_across_a_band() -> None:
    """A chain of adjacent points is one region only if it fits in the radius.

    Single linkage would merge this whole line into one archetype because each
    adjacent pair is within the radius. Complete linkage must not, since the ends
    are far apart - that chaining is what once collapsed 64 practical solutions
    into a single archetype.
    """
    points = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11)], axis=1)
    groups = group_archetypes(points, 0.3)
    assert len(groups) > 1
    for group in groups:
        span = np.ptp(points[group], axis=0)
        assert span.max() <= 0.3 + 1e-9


def test_grouping_covers_every_point_exactly_once() -> None:
    rng = np.random.default_rng(13)
    points = rng.uniform(-1.0, 1.0, size=(40, 2))
    groups = group_archetypes(points, 0.25)
    seen = sorted(index for group in groups for index in group.tolist())
    assert seen == list(range(40))


# --------------------------------------------------------------------------- #
# Eqs. 23-25: the front against an independent brute-force reference
# --------------------------------------------------------------------------- #
def _brute_force_front(objectives: np.ndarray, white_space: np.ndarray) -> tuple[list[int], list[int]]:
    """Explicit double loop over dominance, then Eq. 25's ``W > 0`` filter."""
    dominated = [False] * len(objectives)
    for victim in range(len(objectives)):
        for other in range(len(objectives)):
            if other == victim:
                continue
            no_worse = bool(np.all(objectives[other] >= objectives[victim]))
            strictly_better = bool(np.any(objectives[other] > objectives[victim]))
            if no_worse and strictly_better:
                dominated[victim] = True
                break
    front = [index for index, flag in enumerate(dominated) if not flag]
    practical = [index for index in front if white_space[index] > 0.0]
    return front, practical


def test_pareto_front_matches_a_bruteforce_reference() -> None:
    rng = np.random.default_rng(41)
    objectives = rng.random((90, 4))
    # Force ties and duplicates so the "strictly better" clause is exercised.
    objectives[10] = objectives[11]
    objectives[20] = objectives[30]
    white_space = np.where(rng.random(90) > 0.35, rng.random(90), 0.0)

    front, practical = extract_pareto_front(objectives, white_space)
    expected_front, expected_practical = _brute_force_front(objectives, white_space)

    assert sorted(front.tolist()) == expected_front
    assert sorted(practical.tolist()) == expected_practical


def test_practical_front_members_are_mutually_non_dominated() -> None:
    """No reported solution may dominate another one (Eq. 24 restricted to P)."""
    rng = np.random.default_rng(43)
    objectives = rng.random((120, 4))
    white_space = np.where(rng.random(120) > 0.5, rng.random(120), 0.0)
    _, practical = extract_pareto_front(objectives, white_space)
    assert practical.size > 0
    assert not dominance_mask(objectives[practical]).any()


def test_white_space_filter_is_strictly_positive() -> None:
    """Eq. 25 keeps ``W(x) > 0``; ``W(x) = 0`` is excluded, however small.

    Each row here is best on exactly one objective, so all four are mutually
    non-dominated and only the white-space filter can remove anything.
    """
    objectives = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    white_space = objectives[:, 1]
    front, practical = extract_pareto_front(objectives, white_space)
    assert sorted(front.tolist()) == [0, 1, 2, 3]
    assert list(practical) == [1]
    assert not dominance_mask(objectives).any()

    # A tiny but positive W survives; exactly zero does not.
    tiny = np.array([[1.0, 1e-12], [0.0, 0.0]])
    assert list(extract_pareto_front(tiny, tiny[:, 1])[1]) == [0]
    zero = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert extract_pareto_front(zero, zero[:, 1])[1].size == 0


def test_every_non_dominated_point_with_white_space_is_practical() -> None:
    rng = np.random.default_rng(47)
    objectives = rng.random((60, 4))
    white_space = rng.random(60)
    front, practical = extract_pareto_front(objectives, white_space)
    assert practical.size == front.size
    assert np.array_equal(practical, front)
