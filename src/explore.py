"""Phase 6: place an external idea on a finished chart (the "explorer").

Every other stage in this pipeline *builds* the map. This one only *reads* it.
An external document - a proposal, an abstract, a draft - is scored against the
finished chart so the author can see where their own idea sits relative to the
harvested literature, and whether it lands in a region the map already calls a
discovery gap.

Three invariants follow from that and are enforced, not just intended:

1. The corpus is not touched. The document never enters ``corpus.json``, never
   contributes a row to the TF-IDF matrix, never moves a document coordinate,
   and never changes a grid objective. The idea is projected *through* the
   stored vocabulary; nothing is refitted.
2. The objectives are the grid's own, not a new normalisation. ``P_boundary``,
   ``W``, ``G``, and ``S`` are computed at the placed coordinate with the same
   kernels, the same SPH bandwidths, and the same percentile bounds that
   produced the 441 grid values (recovered from the saved raw fields), so the
   four numbers are directly comparable with any point on the map. Concretely,
   the global weights use Eq. 9's ``h(x) = max_i ||x - x_i|| / 1.98`` at the
   placed point, and ``G`` is Eq. 14's central difference at
   ``delta = gradient_delta`` (1e-3), not a grid-spaced stencil.
3. Placement is honest about its own error. The chart is a distortion of
   cosine geometry (the build records a distance correlation near 0.66), so a
   raw-distance trilateration would extrapolate off-chart. A monotone
   (isotonic) map from cosine distance to chart distance is fitted on the
   corpus's own pairs first, and the resulting placement error is measured by
   leave-one-out on the corpus and reported alongside the position.

Why isotonic, and why trilateration: the chart is the paper's bounded chart
``Omega = [-1, 1]^2``, not an unconstrained MDS solution, so the textbook
Nystrom extension of a classical MDS is not available (measured on this corpus
it was 2x less accurate than the approach below anyway). What *is* available is
the map's own answer to "where does a document with this distance profile go":
the point whose distances to all 120 documents best match the calibrated
distances. Least-squares trilateration answers exactly that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize

from src.common import LOGGER, Paths, configure_logging, load_config, resolve_paths
from src.embed import Representation, load_representation
from src.manifold import (
    band_pass_whitespace,
    boundary_strength,
    normalized_mixture,
    pairwise_distances,
    semantic_gradient_field,
    sph_bandwidth,
    sph_weights,
    sph_weights_adaptive,
    support_statistics,
)
from src.report import (
    LIMITED_SOURCE_DF,
    MIN_LOCAL_L1_SHARE,
    Vocabulary,
    build_vocabulary,
    local_lenses,
)

DEFAULT_CONFIG = Path("config.yaml")

OBJECTIVE_NAMES = ("P_boundary", "W", "G", "S")

#: Corpus sanitation floor (README Sec. 3.1): below this a document has too few
#: word statistics for a char n-gram profile to mean anything.
MIN_WORDS = 1_000

#: Verification tolerance for a re-derived position, in chart units. The grid
#: spacing is 0.1, so this is "the same cell, plus slack".
PLACEMENT_TOLERANCE = 0.02

#: Verification tolerance for a re-derived objective score.
SCORE_TOLERANCE = 1e-6

#: A document whose distinct n-grams are below this share of the vocabulary is
#: mostly new vocabulary: its position is driven by the few terms it shares and
#: should not be read as a confident placement.
MIN_VOCABULARY_MATCH = 0.70

#: Number of corpus documents listed as nearest neighbours.
NEIGHBOUR_COUNT = 8

#: How many documents each neighbour list shows on the page: one list ranks by
#: text similarity, the other by chart distance, because the two disagree.
NEIGHBOUR_VIEW_SIZE = 3

#: ``W`` saturates on a plateau between ``q60`` and ``q85`` (Eq. 12), so many
#: grid points score exactly 1.0. A value within this distance of the maximum
#: counts as plateau, which is what makes "W = 1.0" readable as a share.
WHITE_SPACE_PLATEAU_EPSILON = 1e-4

#: Truncate the evidence lists so the artifact stays readable.
LENS_SIZE = 15

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]{1,120}\$", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>\n]{1,80}>")
_MD_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_MD_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
_MD_LIST = re.compile(r"^\s{0,4}(?:[-*+]|\d+[.)])\s+")
_MD_EMPHASIS = re.compile(r"[*_`~]+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")


# --------------------------------------------------------------------------- #
# Chart context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChartContext:
    """Everything the explorer reads off a finished chart.

    Loaded once from ``manifold_grid.npz`` and the corpus; no field here is ever
    written back. The raw per-grid-point fields are kept because they are the
    only record of the percentile bounds the build used - recovering them is
    what makes the idea's scores comparable with the grid's.
    """

    coords: np.ndarray
    doc_regime: np.ndarray
    doc_ids: list[str]
    h_documents: np.ndarray
    grid: np.ndarray
    objectives: np.ndarray
    d_nn: np.ndarray
    quantiles: dict[str, float]
    spacing: float
    sph_h: float
    gradient_delta: float
    nearest_k: int
    grid_size: int
    labels: dict[int, np.ndarray]
    raw: dict[str, np.ndarray]
    regime_mixture: np.ndarray
    stress: float
    distance_correlation: float
    gram: np.ndarray


def load_chart(paths: Paths, representation: Representation) -> ChartContext:
    """Read the saved chart and recompute the corpus Gram matrix."""
    try:
        archive = np.load(paths.grid_npz, allow_pickle=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Manifold grid not found: {paths.grid_npz}\nRun `python -m src.manifold` first."
        ) from exc

    with archive as data:
        coords = np.asarray(data["doc_coords"], dtype=np.float64)
        doc_regime = np.asarray(data["doc_regime"], dtype=np.int64)
        doc_ids = [str(value) for value in data["doc_ids"]]
        quantile_values = np.asarray(data["whitespace_quantiles"], dtype=np.float64)
        raw = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in (
                "d_nn",
                "d5",
                "n_eff",
                "top3",
                "gradient_raw",
                "uncertainty",
                "reliability",
            )
        }
        for name in (
            "k5_global",
            "k8_global",
            "k8_adaptive",
        ):
            raw[f"entropy_{name}"] = np.asarray(data[f"entropy_{name}"], dtype=np.float64)
            raw[f"gap_{name}"] = np.asarray(data[f"gap_{name}"], dtype=np.float64)
        gram = np.asarray(
            sp.csr_matrix(representation.z_l2 @ representation.z_l2.T).toarray(),
            dtype=np.float64,
        )
        context = ChartContext(
            gram=gram,
            coords=coords,
            doc_regime=doc_regime,
            doc_ids=doc_ids,
            h_documents=np.asarray(data["h_documents"], dtype=np.float64),
            grid=np.asarray(data["grid"], dtype=np.float64),
            objectives=np.asarray(data["objectives"], dtype=np.float64),
            d_nn=raw["d_nn"],
            quantiles={
                key: float(value)
                for key, value in zip(
                    ("q20", "q60", "q85", "q98"), quantile_values, strict=True
                )
            },
            spacing=float(np.ravel(data["spacing"])[0]),
            sph_h=float(np.ravel(data["sph_h"])[0]),
            gradient_delta=float(
                np.ravel(data["gradient_delta"])[0]
                if "gradient_delta" in data.files
                else 1e-3
            ),
            nearest_k=int(np.ravel(data["nearest_k"])[0]),
            grid_size=int(np.ravel(data["grid_size"])[0]),
            labels={
                5: np.asarray(data["labels_k5"], dtype=np.int64),
                8: np.asarray(data["labels_k8"], dtype=np.int64),
            },
            raw=raw,
            regime_mixture=np.asarray(data["mixture_regime"], dtype=np.float64),
            stress=float(np.ravel(data["stress"])[0]),
            distance_correlation=float(np.ravel(data["distance_correlation"])[0]),
        )

    if context.coords.shape[0] != representation.n_documents:
        raise SystemExit(
            "Chart and representation disagree: "
            f"{context.coords.shape[0]} coordinates vs "
            f"{representation.n_documents} documents. Re-run the pipeline."
        )
    if context.doc_ids != list(representation.doc_ids):
        raise SystemExit(
            "Chart document order does not match the TF-IDF matrix. "
            "Rebuild from `src.embed` through `src.manifold`."
        )

    return context


# --------------------------------------------------------------------------- #
# Document handling
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Strip markup that carries no scientific content.

    The representation is character n-grams over words, so a Markdown table
    rule, a LaTeX equation, or a URL contributes n-grams that no paper in the
    corpus contains: pure noise in the profile, and noise that inflates the
    distance to every real document. What survives is prose plus inline code.
    """
    text = _FENCE.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _DISPLAY_MATH.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    kept: list[str] = []
    for line in text.splitlines():
        if _MD_TABLE.match(line) or _MD_RULE.match(line):
            continue
        line = _MD_HEADING.sub("", line)
        line = _MD_LIST.sub("", line)
        line = _MD_EMPHASIS.sub("", line)
        line = re.sub(r"^\s*>\s?", "", line)
        if line.strip():
            kept.append(line)
    return "\n".join(kept)


def load_json(path: Path, what: str) -> dict[str, Any]:
    """Read a JSON artifact, failing with the path and the reason."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{what.capitalize()} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{what.capitalize()} is not valid JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{what.capitalize()} must be a JSON object: {path}")
    return payload


def _read_pdf_document(path: Path) -> str:
    """Extract a PDF through the corpus's own ``mutool`` + sanitation chain.

    A proposal is sanitized the way an in-map paper is sanitized, not the way a
    Markdown draft is: running headers, the bibliography and citation call-outs
    are dropped and the hard line breaks a PDF text layer leaves mid-sentence are
    repaired. Without that pass the reference list's own titles would be embedded
    as if they were the author's argument, and ``cross-\nbody`` would be one term
    the chart has never seen. ``src.harvest`` is imported here rather than at
    module scope because it pulls in the harvesting network stack.
    """
    from src.harvest import extract_pdf_text  # local import: heavy module

    if not path.is_file():
        raise SystemExit(f"Idea document not found: {path}")
    extracted = extract_pdf_text(path, min_body_words=MIN_WORDS)
    if not extracted.text.strip():
        raise SystemExit(
            f"No text layer could be extracted from {path}; a scanned PDF needs OCR"
        )
    return extracted.text


def read_document(path: Path) -> tuple[str, str]:
    """Return ``(raw_text, sha256)`` for the external document.

    ``.pdf`` is extracted and sanitized exactly as the corpus papers were; any
    other suffix must be UTF-8 text. The digest is taken over the text the chart
    actually scores rather than over the file bytes, so the "is this byte-identical
    to a corpus paper" guard in :func:`build_artifact` still means the same thing
    for both readings.
    """
    if path.suffix.lower() == ".pdf":
        raw = _read_pdf_document(path)
    else:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SystemExit(f"Idea document not found: {path}") from exc
        except UnicodeDecodeError as exc:
            raise SystemExit(f"Idea document is not UTF-8 text ({path}): {exc}") from exc
        if not raw.strip():
            raise SystemExit(f"Idea document is empty: {path}")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw, digest


def embed_document(
    text: str, representation: Representation
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Project the document through the *stored* vocabulary and IDF.

    Nothing is refitted: ``vocabulary=`` pins the columns to the corpus's
    retained n-grams and the stored ``idf`` is applied by hand, exactly as
    ``src.embed.build_weighted_matrix`` did when the corpus was built. A term
    the corpus never contained simply has no column, which is the honest
    behaviour - the chart has no coordinate for it.

    ``lowercase`` is disabled and the text is lowercased here because
    scikit-learn's own lowercase pass is precisely ``str.lower()``; doing it
    explicitly avoids its "uppercase in vocabulary" warning without changing a
    single n-gram.
    """
    vocabulary = {term: index for index, term in enumerate(representation.terms)}
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=representation.settings.ngram_range,
        lowercase=False,
        sublinear_tf=representation.settings.sublinear_tf,
        use_idf=False,
        norm=None,  # type: ignore[arg-type]
        vocabulary=vocabulary,
        dtype=np.float64,
    )
    prepared = text.lower()
    try:
        term_frequency = vectorizer.fit_transform([prepared])
    except ValueError as exc:  # empty vocabulary, e.g. one-character text
        raise SystemExit(f"The idea document produced no n-grams: {exc}") from exc
    weighted = term_frequency.multiply(representation.idf[np.newaxis, :]).tocsr()
    weighted.sort_indices()
    z_l2 = normalize(weighted, norm="l2", axis=1, copy=True)
    z_l1 = normalize(weighted, norm="l1", axis=1, copy=True)
    return sp.csr_matrix(z_l2), sp.csr_matrix(z_l1)


def vocabulary_match_rate(text: str, representation: Representation) -> tuple[int, int]:
    """``(matched, total)`` distinct n-grams against the retained vocabulary."""
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=representation.settings.ngram_range,
        lowercase=False,
        dtype=np.int64,
    )
    try:
        vectorizer.fit_transform([text.lower()])
    except ValueError:
        return 0, 0
    present = set(vectorizer.get_feature_names_out())
    matched = present.intersection(representation.terms)
    return len(matched), len(present)


# --------------------------------------------------------------------------- #
# Distance calibration and placement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """Monotone map from cosine distance to the chart's own distance scale."""

    model: IsotonicRegression
    observed_min: float
    observed_max: float
    residual_rmse: float

    def chart_distance(self, distances: np.ndarray) -> np.ndarray:
        """Map cosine distances to chart distances."""
        return np.asarray(
            self.model.predict(np.asarray(distances, dtype=np.float64)), dtype=np.float64
        )


def dense_similarity(z_l2: sp.spmatrix) -> np.ndarray:
    """Dense L2 inner-product matrix across the rows of ``z_l2``."""
    matrix = sp.csr_matrix(z_l2)
    return np.asarray(sp.csr_matrix(matrix @ matrix.T).toarray(), dtype=np.float64)


def cosine_distance_matrix(z_l2: sp.spmatrix) -> np.ndarray:
    """``d = sqrt(2 - 2 cos)`` for every document pair, zero on the diagonal."""
    similarity = dense_similarity(z_l2)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def _fit_calibration(
    cosine: np.ndarray, coords: np.ndarray, keep: np.ndarray | None = None
) -> Calibration:
    """Fit isotonic distance calibration on the corpus's own document pairs.

    Only the upper triangle is used (pairs are symmetric) and only pairs between
    kept documents, which is what makes the leave-one-out estimate honest: the
    held-out document contributes to neither side of the fit.
    """
    subset = np.arange(cosine.shape[0]) if keep is None else np.flatnonzero(keep)
    chart = np.asarray(
        pairwise_distances(coords[subset], coords[subset]), dtype=np.float64
    )
    rows, columns = np.triu_indices(subset.size, k=1)
    source = cosine[np.ix_(subset, subset)][rows, columns]
    target = chart[rows, columns]
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(source, target)
    predicted = model.predict(source)
    residual = float(np.sqrt(np.mean((predicted - target) ** 2)))
    return Calibration(
        model=model,
        observed_min=float(source.min()),
        observed_max=float(source.max()),
        residual_rmse=residual,
    )


def fit_calibration(cosine: np.ndarray, coords: np.ndarray) -> Calibration:
    """Calibration over all corpus pairs."""
    return _fit_calibration(cosine, coords)


def _trilaterate(
    target_distances: np.ndarray,
    coords: np.ndarray,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, float]:
    """Best chart point whose distances to ``coords`` match ``target_distances``.

    Bounded least squares with several starts. The bound is the chart itself:
    a document whose distance profile is beyond anything the chart can express
    is reported at the edge with a large residual, rather than silently placed
    outside the evaluation domain where no grid point exists.
    """

    def residual(point: np.ndarray) -> np.ndarray:
        return np.linalg.norm(coords - point[np.newaxis, :], axis=1) - target_distances

    starts = [
        np.zeros(2),
        coords[int(np.argmin(target_distances))],
        coords[int(np.argmax(target_distances))],
        np.array([bounds[0], bounds[0]]),
        np.array([bounds[1], bounds[1]]),
    ]
    best: Any = None
    for start in starts:
        solution = least_squares(
            residual,
            np.asarray(start, dtype=np.float64),
            bounds=(np.full(2, bounds[0]), np.full(2, bounds[1])),
        )
        if best is None or solution.cost < best.cost:
            best = solution
    assert best is not None
    return np.asarray(best.x, dtype=np.float64), float(best.cost)


def place_document(
    similarity: np.ndarray,
    chart: ChartContext,
    cosine: np.ndarray,
    calibration: Calibration,
) -> dict[str, Any]:
    """Place one document on the chart and report the fit's quality.

    ``similarity`` is the document's L2 inner product against every corpus
    document, in corpus order - the only thing the chart needs to know about it.
    """
    distances = np.sqrt(np.maximum(2.0 - 2.0 * np.asarray(similarity), 0.0))
    calibrated = calibration.chart_distance(distances)
    point, cost = _trilaterate(calibrated, chart.coords)
    fitted = np.linalg.norm(chart.coords - point[np.newaxis, :], axis=1)
    correlation = float(np.asarray(spearmanr(fitted, calibrated))[0])
    rmse = float(np.sqrt(np.mean((fitted - calibrated) ** 2)))
    clipped = int(
        ((distances < calibration.observed_min) | (distances > calibration.observed_max)).sum()
    )
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "cosine_distances": distances,
        "chart_distances": calibrated,
        "fit_cost": cost,
        "fit_rmse": rmse,
        "fit_spearman": correlation,
        "at_chart_edge": bool(np.any(np.abs(point) >= 1.0 - 1e-9)),
        "clipped_distances": clipped,
    }


def leave_one_out_error(
    cosine: np.ndarray, coords: np.ndarray, calibration: Calibration
) -> dict[str, float]:
    """Place every corpus document from the other 119 and measure the error.

    This is the number that decides how much a position means. It re-fits the
    calibration without the held-out document each time, so the estimate
    includes the calibration's own error rather than borrowing it.
    """
    errors: list[float] = []
    for index in range(coords.shape[0]):
        keep = np.ones(coords.shape[0], dtype=bool)
        keep[index] = False
        fold = _fit_calibration(cosine, coords, keep)
        calibrated = fold.chart_distance(cosine[index, keep])
        point, _ = _trilaterate(calibrated, coords[keep])
        errors.append(float(np.linalg.norm(point - coords[index])))
    values = np.asarray(errors, dtype=np.float64)
    chart_distances = pairwise_distances(coords, coords)
    np.fill_diagonal(chart_distances, np.nan)
    spacing = float(np.nanmedian(chart_distances))
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "typical_neighbour_spacing": spacing,
        "documents": int(values.size),
        "calibration_residual_rmse": calibration.residual_rmse,
    }


# --------------------------------------------------------------------------- #
# Objective scoring at one point
# --------------------------------------------------------------------------- #
def percentile_bounds(values: np.ndarray) -> tuple[float, float]:
    """The ``P10``/``P90`` pair ``normalize_percentile`` used at build time."""
    return float(np.percentile(values, 10.0)), float(np.percentile(values, 90.0))


def _normalized(
    value: float, bounds: tuple[float, float]
) -> float:
    """``N_+(z) = clip((z - P10) / (P90 - P10), 0, 1)`` for a scalar."""
    low, high = bounds
    span = high - low
    if span <= 1e-12:
        return 0.0
    return float(np.clip((value - low) / span, 0.0, 1.0))


def score_point(point: np.ndarray, chart: ChartContext) -> dict[str, Any]:
    """Evaluate ``P_boundary``, ``W``, ``G``, and ``S`` at one chart coordinate.

    Same kernels, same bandwidths, same percentile bounds as the grid, so the
    result is comparable with ``manifold_grid.npz::objectives`` point by point.
    """
    location = np.asarray(point, dtype=np.float64).reshape(1, 2)
    # Eq. 9's smoothing length is a function of the query point, so it is
    # recomputed here exactly as ``compute_objectives`` does on the grid.
    weights_global = sph_weights(
        location, chart.coords, sph_bandwidth(location, chart.coords)
    )[0]
    weights_adaptive = sph_weights_adaptive(
        location, chart.coords, chart.h_documents
    )[0]

    views: dict[str, float] = {}
    diagnostics: dict[str, Any] = {"boundary_views": {}, "mixtures": {}}
    for name, k, scheme in (
        ("k5_global", 5, "global"),
        ("k8_global", 8, "global"),
        ("k8_adaptive", 8, "adaptive"),
    ):
        weights = weights_global if scheme == "global" else weights_adaptive
        mixture = normalized_mixture(
            weights[np.newaxis, :], chart.labels[k], k
        )[0]
        entropy, gap = boundary_strength(mixture[np.newaxis, :])
        entropy_value = float(entropy[0])
        gap_value = float(gap[0])
        boundary = 0.65 * _normalized(
            entropy_value, percentile_bounds(chart.raw[f"entropy_{name}"])
        ) + 0.35 * (
            1.0
            - _normalized(gap_value, percentile_bounds(chart.raw[f"gap_{name}"]))
        )
        views[name] = boundary
        diagnostics["boundary_views"][name] = {
            "boundary": round(boundary, 6),
            "entropy_H": round(entropy_value, 6),
            "dominant_gap": round(gap_value, 6),
            "cluster_mixture": [round(float(value), 4) for value in mixture],
        }

    if all(value > 0.0 for value in views.values()):
        p_boundary = 3.0 / sum(1.0 / value for value in views.values())
    else:
        # Eq. 21 is a harmonic mean: one zero view makes the point zero, which
        # is the AND-like behaviour Section 4.2 argues for.
        p_boundary = 0.0

    d_nn = float(np.linalg.norm(chart.coords - location, axis=1).min())
    whitespace = float(band_pass_whitespace(np.array([d_nn]), chart.quantiles)[0])

    # Eq. 13-14: central differences of the interpolated field at delta, the
    # same estimator ``compute_objectives`` applies to the 441 grid points.
    gradient_raw = float(
        semantic_gradient_field(
            location, chart.coords, chart.gram, delta=chart.gradient_delta
        )[0]
    )
    gradient = _normalized(gradient_raw, percentile_bounds(chart.raw["gradient_raw"]))

    n_eff, d5, top_three = support_statistics(
        weights_global[np.newaxis, :], location, chart.coords
    )
    n_neff = _normalized(float(n_eff[0]), percentile_bounds(chart.raw["n_eff"]))
    n_d5 = _normalized(float(d5[0]), percentile_bounds(chart.raw["d5"]))
    n_t3 = _normalized(float(top_three[0]), percentile_bounds(chart.raw["top3"]))
    support = 0.40 * n_neff + 0.30 * (1.0 - n_d5) + 0.30 * (1.0 - n_t3)

    mixture_regime = normalized_mixture(
        weights_global[np.newaxis, :],
        chart.doc_regime,
        int(chart.doc_regime.max()) + 1,
    )[0]

    objectives = np.array([p_boundary, whitespace, gradient, support], dtype=np.float64)
    diagnostics.update(
        {
            "d_nn": d_nn,
            "n_eff": float(n_eff[0]),
            "d5": float(d5[0]),
            "top3": float(top_three[0]),
            "gradient_raw": gradient_raw,
            "regime_mixture": [round(float(value), 4) for value in mixture_regime],
            "sph_weights": weights_global,
        }
    )
    return {
        "objectives": objectives,
        "scores": {
            name: round(float(value), 6)
            for name, value in zip(OBJECTIVE_NAMES, objectives, strict=True)
        },
        "diagnostics": diagnostics,
    }


def grid_percentiles(objectives: np.ndarray, chart: ChartContext) -> dict[str, float]:
    """Where each score falls among the 441 grid points (0-100)."""
    return {
        name: round(float((chart.objectives[:, index] < value).mean() * 100.0), 1)
        for index, (name, value) in enumerate(zip(OBJECTIVE_NAMES, objectives, strict=True))
    }


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def neighbour_evidence(
    similarity: np.ndarray, chart: ChartContext, point: np.ndarray, papers: dict[str, dict]
) -> list[dict[str, Any]]:
    """Corpus documents nearest in chart space and by raw text similarity."""
    chart_distances = np.linalg.norm(chart.coords - point[np.newaxis, :], axis=1)
    chart_order = np.argsort(chart_distances)
    cosine_order = np.argsort(-similarity)
    chart_rank = {int(index): rank for rank, index in enumerate(chart_order)}
    cosine_rank = {int(index): rank for rank, index in enumerate(cosine_order)}
    nearest_chart = {int(index) for index in chart_order[:NEIGHBOUR_COUNT]}
    nearest_cosine = {int(index) for index in cosine_order[:NEIGHBOUR_COUNT]}
    keep = sorted(nearest_chart | nearest_cosine)

    records: list[dict[str, Any]] = []
    for index in keep:
        paper = papers.get(chart.doc_ids[index], {})
        records.append(
            {
                "id": chart.doc_ids[index],
                "title": str(paper.get("title", chart.doc_ids[index])),
                "regime_id": int(chart.doc_regime[index]),
                "regime_label": str(paper.get("regime_label", "")),
                "year": paper.get("year"),
                "chart_distance": round(float(chart_distances[index]), 4),
                "chart_rank": chart_rank[index],
                "cosine_similarity": round(float(similarity[index]), 4),
                "cosine_rank": cosine_rank[index],
                "is_seed": bool(paper.get("is_seed", False)),
                "url": str(paper.get("url", "")),
            }
        )
    records.sort(key=lambda item: item["chart_distance"])
    return records


def neighbour_views(neighbours: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split the neighbour evidence into its two disagreeing orderings.

    ``neighbour_evidence`` returns the union of the chart-nearest and the
    text-nearest documents, so one list can hide the other's winner: a parent
    paper can be rank 1 in text and rank 25 in the chart. Both orderings are
    reported, and neither is presented as "the" neighbourhood.
    """
    by_text = sorted(neighbours, key=lambda item: int(item["cosine_rank"]))
    by_chart = sorted(neighbours, key=lambda item: int(item["chart_rank"]))
    return {
        "by_text": by_text[:NEIGHBOUR_VIEW_SIZE],
        "by_chart": by_chart[:NEIGHBOUR_VIEW_SIZE],
    }


def gap_evidence(
    point: np.ndarray, pareto: dict[str, Any], count: int = 3
) -> list[dict[str, Any]]:
    """Nearest practical Pareto gaps and the archetype each belongs to."""
    archetype_of: dict[int, dict[str, Any]] = {}
    for archetype in pareto.get("archetypes", []):
        for member in archetype.get("members", []):
            archetype_of[int(member["grid_index"])] = archetype

    ranked: list[tuple[float, dict[str, Any]]] = []
    for solution in pareto.get("solutions", []):
        target = np.array([float(solution["x"]), float(solution["y"])])
        ranked.append((float(np.linalg.norm(target - point)), solution))
    ranked.sort(key=lambda item: item[0])

    records: list[dict[str, Any]] = []
    for distance, solution in ranked[:count]:
        archetype = archetype_of.get(int(solution["grid_index"]), {})
        records.append(
            {
                "rank": int(solution.get("rank", 0)),
                "grid_index": int(solution["grid_index"]),
                "x": round(float(solution["x"]), 4),
                "y": round(float(solution["y"]), 4),
                "chart_distance": round(distance, 4),
                "objectives": {
                    name: round(float(value), 6)
                    for name, value in solution.get("objectives", {}).items()
                },
                "ideal_distance": round(float(solution.get("ideal_distance", 0.0)), 6),
                "archetype_id": archetype.get("archetype_id"),
                "archetype_label": archetype.get("label"),
                "archetype_size": archetype.get("size"),
            }
        )
    return records


def gap_reference(
    objectives: np.ndarray,
    ideal_distance: float,
    point: np.ndarray,
    pareto: dict[str, Any],
) -> dict[str, Any]:
    """Where the external point sits relative to the discrete front.

    The front is a finite set of grid candidates ranked by distance to the ideal
    corner, so an external document has no gap rank: it is not a candidate and
    cannot be interleaved into a ranking that only contains candidates. Three
    surrogates are reported instead, each answering a different question:

    - the gap nearest *on the chart* (where a reader's eye lands),
    - the gap nearest *in the four-objective space* (the actually similar
      proposal, which the 2-D projection can put far away),
    - the slot the point *would* take if inserted into the front's own ranking
      by ``d_ideal`` -- labelled hypothetical, because insertion does not make
      it a member.

    The last field is the one that decides membership: a gap whose four
    objectives are all at least as high strictly dominates the point, and a
    dominated point cannot be on the Pareto front no matter how good its
    ``d_ideal`` is.
    """
    solutions = [s for s in pareto.get("solutions", []) if isinstance(s, dict)]
    if not solutions:
        return {}
    names = list(OBJECTIVE_NAMES)
    archetype_of: dict[int, dict[str, Any]] = {}
    for archetype in pareto.get("archetypes", []):
        for member in archetype.get("members", []):
            archetype_of[int(member["grid_index"])] = archetype

    def describe(solution: dict[str, Any], chart_distance: float) -> dict[str, Any]:
        values = np.array(
            [float(solution.get("objectives", {}).get(name, 0.0)) for name in names]
        )
        archetype = archetype_of.get(int(solution["grid_index"]), {})
        return {
            "rank": int(solution.get("rank", 0)),
            "chart_distance": round(float(chart_distance), 4),
            "objective_distance": round(float(np.linalg.norm(values - objectives)), 4),
            "ideal_distance": round(float(solution.get("ideal_distance", 0.0)), 6),
            "objectives": {
                name: round(float(value), 6)
                for name, value in zip(names, values, strict=True)
            },
            "archetype": str(archetype.get("label", "")),
        }

    chart_distances = [
        float(
            np.linalg.norm(
                np.array([float(s["x"]), float(s["y"])], dtype=np.float64) - point
            )
        )
        for s in solutions
    ]
    nearest_chart = min(range(len(solutions)), key=lambda i: chart_distances[i])
    nearest_objective = min(
        range(len(solutions)),
        key=lambda i: float(
            np.linalg.norm(
                np.array(
                    [float(solutions[i].get("objectives", {}).get(n, 0.0)) for n in names]
                )
                - objectives
            )
        ),
    )

    ordered = sorted(solutions, key=lambda s: float(s.get("ideal_distance", 0.0)))
    insertion = int(
        sum(1 for s in solutions if float(s.get("ideal_distance", 0.0)) < ideal_distance)
    ) + 1
    before = ordered[insertion - 2] if insertion > 1 else None
    after = ordered[insertion - 1] if insertion <= len(ordered) else None

    dominators: list[tuple[float, dict[str, Any]]] = []
    for solution in solutions:
        values = np.array(
            [float(solution.get("objectives", {}).get(name, 0.0)) for name in names]
        )
        if bool(np.all(values >= objectives)) and bool(np.any(values > objectives)):
            dominators.append((float(solution.get("ideal_distance", 0.0)), solution))
    dominators.sort(key=lambda item: item[0])
    dominating = describe(dominators[0][1], 0.0) if dominators else None
    if dominating is not None:
        index = solutions.index(dominators[0][1])
        dominating["chart_distance"] = round(chart_distances[index], 4)

    return {
        "nearest_chart": describe(solutions[nearest_chart], chart_distances[nearest_chart]),
        "nearest_objective": describe(
            solutions[nearest_objective], chart_distances[nearest_objective]
        ),
        "insertion": {
            "rank": insertion,
            "of": len(solutions) + 1,
            "before": (
                {
                    "rank": int(before.get("rank", 0)),
                    "ideal_distance": round(float(before.get("ideal_distance", 0.0)), 6),
                }
                if before is not None
                else None
            ),
            "after": (
                {
                    "rank": int(after.get("rank", 0)),
                    "ideal_distance": round(float(after.get("ideal_distance", 0.0)), 6),
                }
                if after is not None
                else None
            ),
        },
        "dominated_by": [
            describe(solution, chart_distances[solutions.index(solution)])
            for _, solution in dominators
        ],
        "on_front": not dominators,
        "note": "Hypothetical only: the point is not a candidate. Insertion into "
        "the d_ideal ordering is arithmetic, not front membership; a gap that "
        "dominates it on all four objectives would keep it off the front.",
    }


OBJECTIVE_MEANINGS = {
    "P_boundary": "border strength: 1 = a hard seam between literatures, 0 = inside one literature",
    "W": "white space: band-pass on the distance to the nearest paper, so 0 can mean "
    "incremental or unsupported and 1 means the healthy window between them",
    "G": "gradient: how fast the map's topics change around this point",
    "S": "support: how much corpus evidence sits nearby to build on",
}


def _band(percentile: float, low: str, middle: str, high: str, *, split: float = 50.0) -> str:
    return high if percentile >= split else (middle if percentile >= split / 2 else low)


def _objective_reading(name: str, value: float, percentile: float, plateau: float) -> str:
    if name == "P_boundary":
        return _band(
            percentile,
            "you sit inside one literature rather than on a seam; the positions that "
            "score higher here are the ones that join two fields",
            "there is a boundary near you but you are not on it, and the clusterings "
            "disagree about where it runs",
            "you sit on a seam between literatures, and more than one clustering agrees "
            "the boundary is real",
        )
    if name == "W":
        if plateau >= 0.05 and value >= 0.999:
            return (
                f"maximum but shared: the band-pass saturates, so {plateau * 100:.0f}% of "
                "the map scores exactly this. Read it as \"the nearest paper is neither "
                "on top of me nor missing\", not as a record"
            )
        if percentile >= 75:
            return "the nearest paper is far enough to leave room and close enough to build on"
        if percentile >= 25:
            return "middle distance to the nearest paper: the space is partly occupied"
        return (
            "either incremental (a paper sits right there) or unsupported (nothing does) "
            "-- W alone cannot tell you which"
        )
    if name == "G":
        return _band(
            percentile,
            "the surrounding literature is uniform, so this area is saturated with similar work",
            "the local topic field is moderately flat: you are in the body of a literature, "
            "not at its edge",
            "the map's topics change fast around this point, which is what a transition "
            "zone looks like",
        )
    return _band(
        percentile,
        "thin support: fewer papers to build on, so claims here rest on less prior art",
        "reasonable support nearby -- enough to anchor claims without being crowded",
        "plenty of corpus evidence nearby: prior work to cite and to build on",
        split=70.0,
    )


def plain_language(
    scores: dict[str, float],
    percentiles: dict[str, float],
    quality: dict[str, Any],
    mixture: dict[str, float],
    plateau_share: float,
    front: dict[str, Any],
    residual: float | None = None,
) -> dict[str, Any]:
    """The same numbers said in sentences, so a reader can check the reading.

    Nothing here is inferred by a model: every clause is selected from the
    artifact's own values by fixed thresholds, which is why it can be verified
    by re-reading the numbers printed beside it. Percentiles are shares of the
    441 grid points, not of published papers.
    """
    objectives = [
        {
            "name": name,
            "value": round(float(scores[name]), 4),
            "percentile": round(float(percentiles[name]), 1),
            "meaning": OBJECTIVE_MEANINGS[name],
            "reading": _objective_reading(
                name, float(scores[name]), float(percentiles[name]), plateau_share
            ),
        }
        for name in OBJECTIVE_NAMES
    ]

    mean_percentile = float(quality.get("mean_distance_percentile", 0.0))
    nearest_percentile = float(quality.get("nearest_distance_percentile", 0.0))
    vocab = float(quality.get("vocabulary_match_rate", 0.0))
    if mean_percentile < 10:
        isolation = (
            "the document is more isolated than almost every paper in the corpus, so it "
            "is not a paraphrase of anything already on the map"
        )
    elif mean_percentile < 40:
        isolation = (
            "the document sits in the corpus's own range of distances: related work "
            "exists, but nothing says the same thing"
        )
    else:
        isolation = (
            "the document is closer to the corpus than most corpus papers are to each "
            "other, which usually reads as an extension of existing work"
        )
    if vocab < 0.7:
        vocabulary = (
            f"only {vocab * 100:.0f}% of its wording is known to the corpus, so part of "
            "the position reflects language the map cannot represent"
        )
    else:
        vocabulary = (
            f"{vocab * 100:.0f}% of its wording is corpus vocabulary, so the position is "
            "earned inside this map rather than imported from another field"
        )
    novelty = (
        "Novelty rests on the text distances, not on W. Mean distance to the corpus "
        f"{quality.get('mean_cosine_distance', 0.0):.3f} ({mean_percentile:.0f}% of corpus "
        f"documents are further), nearest {quality.get('nearest_distance', 0.0):.3f} "
        f"({nearest_percentile:.0f}%). In plain words: {isolation}. {vocabulary[0].upper() + vocabulary[1:]}."
    )

    dominant = max(mixture.items(), key=lambda item: item[1]) if mixture else ("", 0.0)
    boundary_percentile = float(percentiles.get("P_boundary", 0.0))
    if boundary_percentile >= 75:
        stance = (
            "The objectives put you on a seam between literatures, so the contribution "
            "reads as a bridge between two fields."
        )
    elif boundary_percentile >= 40:
        stance = (
            "The objectives put you near a seam but not on it: the shape of a "
            "single-field contribution with cross-field exposure."
        )
    else:
        stance = (
            "The objectives put you inside a single literature, so the contribution has "
            "to be read as depth in that field rather than a bridge between two."
        )
    position = f"The local mixture is {dominant[0]} at {dominant[1] * 100:.0f}%. {stance}"

    dominators = [entry for entry in (front.get("dominated_by") or []) if isinstance(entry, dict)]
    if dominators:
        best = dominators[0]
        theirs = best.get("objectives", {})
        higher = [
            name
            for name in OBJECTIVE_NAMES
            if float(theirs.get(name, 0.0)) > float(scores[name]) + SCORE_TOLERANCE
        ]
        equal = [
            name
            for name in OBJECTIVE_NAMES
            if abs(float(theirs.get(name, 0.0)) - float(scores[name])) <= SCORE_TOLERANCE
        ]
        comparison = ", ".join(
            f"{name} {float(theirs.get(name, 0.0)):.3f} vs {float(scores[name]):.3f}"
            for name in OBJECTIVE_NAMES
        )
        front_text = (
            f"As a front candidate this position would not earn a slot: gap "
            f"#{best.get('rank')} ({best.get('archetype', '')}) scores higher on "
            f"{', '.join(higher)}"
            + (f" and ties on {', '.join(equal)} (a plateau value)" if equal else "")
            + f" -- {comparison}. That compares positions on four axes; it does not say "
            "the work is weak, and most published papers are not front points either."
        )
    else:
        front_text = (
            "No gap scores at least as high on all four axes, so at this exact "
            "coordinate the position would join the front."
        )
        if residual:
            front_text += (
                f" That verdict is knife-edge: the placement's own error is {residual:.2f} "
                "chart units, and non-domination flips inside that radius, so read it "
                "as 'no gap dominates here' rather than as front membership."
            )

    headline = _band(
        float(percentiles.get("S", 0.0)),
        "Thinly supported corner of the map",
        "Supported position",
        "Well-supported position",
        split=70.0,
    )
    headline += _band(
        boundary_percentile,
        " inside a single field",
        " near a seam",
        " on a seam between fields",
        split=75.0,
    )
    headline += (
        ", with textual isolation rather than white space as its novelty evidence."
        if mean_percentile < 40
        else ", and its novelty evidence is weak: it reads as an extension of nearby work."
    )

    return {
        "headline": headline,
        "objectives": objectives,
        "position": position,
        "novelty": novelty,
        "front": front_text,
        "note": "Every sentence is selected from the artifact's own values by fixed "
        "thresholds; nothing is generated. Percentiles are shares of the evaluation "
        "grid, not of published papers.",
    }


def _is_plausible_term(term: str) -> bool:
    """Reject vocabulary entries that are markup or URLs, not subject words.

    Hyphenated technical terms (``trajectory-conditioned``) belong in the list;
    a hyphen chain longer than three segments or a token over 32 characters
    comes from a citation key, a file path, or a code span.
    """
    if not term or len(term) > 32:
        return False
    if term.count("-") > 2:
        return False
    return not any(character.isdigit() for character in term)


def idea_terms(
    z_l1: sp.csr_matrix,
    representation: Representation,
    vocabulary: Vocabulary,
    local: dict[str, Any],
) -> dict[str, Any]:
    """Whole-word terms the document leans on, graded by corpus coverage.

    Reuses the report's vocabulary and its two df regimes (Rule 2: ``df <= 3``
    is "limited source", ``df >= 4`` is an emergent topic), but the share comes
    from the document instead of from a map location, which is the point: this
    is what *the idea* says, not what its neighbourhood says.
    """
    # Aggregate the document's n-gram L1 weights onto whole words with the same
    # n-gram -> family map the corpus vocabulary was built from.
    projection = representation.word_index.feature_family @ sp.csr_matrix(z_l1).T
    share = np.asarray(projection.toarray(), dtype=np.float64).ravel()
    plausible = np.asarray(
        [_is_plausible_term(str(term)) for term in vocabulary.terms], dtype=np.bool_
    )

    candidates = np.flatnonzero(
        vocabulary.usable
        & (share >= MIN_LOCAL_L1_SHARE)
        & plausible
    )
    order = candidates[np.argsort(share[candidates])[::-1]]

    novel: list[dict[str, Any]] = []
    covered: list[dict[str, Any]] = []
    for index in order:
        entry = {
            "term": str(vocabulary.terms[index]),
            "share": round(float(share[index]), 5),
            "df": int(vocabulary.document_frequency[index]),
        }
        if entry["df"] <= LIMITED_SOURCE_DF:
            novel.append(entry)
        else:
            covered.append(entry)
        if len(novel) >= LENS_SIZE and len(covered) >= LENS_SIZE:
            break

    def dedupe(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep one spelling of a term when the corpus also stores its joined form."""
        seen: set[str] = set()
        kept: list[dict[str, Any]] = []
        for entry in entries:
            key = str(entry["term"]).replace("-", "")
            if key in seen:
                continue
            seen.add(key)
            kept.append(entry)
        return kept

    return {
        "not_in_corpus": dedupe(novel[: LENS_SIZE * 2])[:LENS_SIZE],
        "well_covered_by_corpus": dedupe(covered[: LENS_SIZE * 2])[:LENS_SIZE],
        "map_theme_lens": local.get("theme_lens_L2", []),
        "map_concentration_lens": local.get("concentration_lens_L1", []),
        "shared_terms": local.get("shared_terms", []),
    }


# --------------------------------------------------------------------------- #
# Artifact assembly
# --------------------------------------------------------------------------- #
def document_quality(
    raw_text: str,
    cleaned: str,
    representation: Representation,
    chart: ChartContext,
    papers: dict[str, dict],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """How much of the document the chart can actually represent.

    Reported instead of assumed: a short document, a document in vocabulary the
    corpus never used, or a document from a different field all produce a
    confident-looking position that means much less than it looks like it does.
    """
    matched, total = vocabulary_match_rate(cleaned, representation)
    match_rate = (matched / total) if total else 0.0
    distances = np.asarray(evaluation["cosine_distances"], dtype=np.float64)
    similarity = 1.0 - 0.5 * distances**2

    # Corpus baselines, all as cosine *distances* because that is the scale the
    # placement works in: how far each corpus document sits from its nearest
    # neighbour, and how far it sits from the corpus on average.
    cosine = cosine_distance_matrix(representation.z_l2)
    count = cosine.shape[0]
    off_diagonal = np.where(np.eye(count, dtype=bool), np.inf, cosine)
    neighbour_distance = np.asarray(off_diagonal.min(axis=1), dtype=np.float64)
    mean_corpus_distance = np.asarray(
        np.where(np.eye(count, dtype=bool), 0.0, cosine).sum(axis=1) / max(count - 1, 1),
        dtype=np.float64,
    )
    nearest_index = int(np.argmin(distances))

    word_count = len(cleaned.split())
    char_count = len(cleaned)
    mean_distance = float(np.mean(distances))
    nearest_distance = float(distances[nearest_index])

    warnings: list[str] = []
    if word_count < MIN_WORDS:
        warnings.append(
            f"Document has {word_count} words; the corpus sanitation floor is "
            f"{MIN_WORDS}. A short text yields a coarse, vocabulary-heavy profile."
        )
    if match_rate < MIN_VOCABULARY_MATCH:
        warnings.append(
            f"Only {match_rate * 100:.1f}% of the document's n-grams exist in the "
            "corpus vocabulary; the position rests on the terms it shares."
        )
    if nearest_distance > float(neighbour_distance.max()):
        warnings.append(
            "No corpus document is as similar to this one as the corpus's most "
            "isolated document is to its own neighbour: the idea is outside the "
            "chart's density range and the position is extrapolated."
        )
    if evaluation["at_chart_edge"]:
        warnings.append(
            "Placed on the chart border; the best-fitting point is outside the "
            "evaluated domain, so the grid objectives are read at the edge."
        )
    if evaluation["clipped_distances"]:
        warnings.append(
            f"{evaluation['clipped_distances']} of {chart.coords.shape[0]} distances "
            "fell outside the calibrated range and were clipped."
        )
    if evaluation["fit_rmse"] > 0.4 * chart.spacing * 4:
        warnings.append(
            "Placement residual is larger than the calibration's own error "
            f"({evaluation['fit_rmse']:.3f} chart units)."
        )

    return {
        "title": str(papers.get("__title__", "")),
        "word_count": word_count,
        "char_count": char_count,
        "character_share_of_raw": round(char_count / max(len(raw_text), 1), 4),
        "n_grams_total": total,
        "n_grams_in_vocabulary": matched,
        "vocabulary_match_rate": round(match_rate, 4),
        "mean_cosine_distance": round(mean_distance, 4),
        "mean_distance_percentile": round(
            float((mean_corpus_distance > mean_distance).mean() * 100.0), 1
        ),
        "nearest_document": chart.doc_ids[nearest_index],
        "nearest_similarity": round(float(similarity[nearest_index]), 4),
        "nearest_distance": round(nearest_distance, 4),
        "nearest_distance_percentile": round(
            float((neighbour_distance > nearest_distance).mean() * 100.0), 1
        ),
        "corpus_neighbour_distance_median": round(float(np.median(neighbour_distance)), 4),
        "corpus_neighbour_distance_max": round(float(neighbour_distance.max()), 4),
        "warnings": warnings,
    }


def build_artifact(
    config: dict[str, Any],
    paths: Paths,
    document: Path,
    label: str,
) -> dict[str, Any]:
    """Place one external document and return the full evidence record."""
    representation = load_representation(paths.tfidf_matrix, paths.tfidf_vocab)
    chart = load_chart(paths, representation)

    corpus = load_json(paths.corpus, "corpus")
    raw_papers = corpus.get("papers", [])
    papers = {str(paper["id"]): paper for paper in raw_papers}

    raw_text, digest = read_document(document)
    if digest in {
        hashlib.sha256(str(paper.get("full_text", "")).encode("utf-8")).hexdigest()
        for paper in raw_papers
        if paper.get("full_text")
    }:
        raise SystemExit(
            f"{document} is byte-identical to a corpus document; it is not external."
        )

    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        raise SystemExit(f"Document has no prose left after markup removal: {document}")

    z_l2, z_l1 = embed_document(cleaned, representation)
    similarity = np.asarray(
        sp.csr_matrix(representation.z_l2 @ z_l2.T).toarray(), dtype=np.float64
    ).ravel()
    cosine = cosine_distance_matrix(representation.z_l2)
    calibration = fit_calibration(cosine, chart.coords)
    evaluation = place_document(similarity, chart, cosine, calibration)
    point = np.array([evaluation["x"], evaluation["y"]], dtype=np.float64)

    scored = score_point(point, chart)
    objectives = scored["objectives"]

    vocabulary = build_vocabulary(representation)
    location_weights = scored["diagnostics"].pop("sph_weights")
    local = local_lenses(location_weights, vocabulary, raw_papers)

    pareto: dict[str, Any] = {}
    if paths.pareto_json.is_file():
        pareto = load_json(paths.pareto_json, "Pareto front")
    else:
        LOGGER.warning("No %s; gap distances omitted", paths.pareto_json)

    ideal = np.asarray(
        config.get("pareto", {}).get("ideal_point", [1.0, 1.0, 1.0, 1.0]),
        dtype=np.float64,
    )
    ideal_distance = float(np.linalg.norm(objectives - ideal))
    grid_ideal = np.linalg.norm(chart.objectives - ideal, axis=1)

    regime_labels = {
        int(paper["regime_id"]): str(paper.get("regime_label", paper["regime_id"]))
        for paper in raw_papers
    }
    weights = scored["diagnostics"]["regime_mixture"]
    mixture = {
        regime_labels.get(regime, str(regime)): round(float(value), 4)
        for regime, value in zip(sorted(regime_labels), weights, strict=True)
    }

    quality = document_quality(raw_text, cleaned, representation, chart, papers, evaluation)
    quality["title"] = label

    neighbours = neighbour_evidence(similarity, chart, point, papers)
    gaps = gap_evidence(point, pareto) if pareto else []
    loo = leave_one_out_error(cosine, chart.coords, calibration)

    # ``W`` is a band-pass function that saturates between q60 and q85, so a
    # score of 1.0 is not unique: report how much of the grid shares it, which
    # is the only way a reader can tell "maximum" from "rare".
    white_column = int(OBJECTIVE_NAMES.index("W"))
    white_grid = chart.objectives[:, white_column]
    plateau_share = float(
        (white_grid >= float(white_grid.max()) - WHITE_SPACE_PLATEAU_EPSILON).mean()
    )

    # An external point has no gap rank. These three surrogates answer the three
    # questions a reader actually asks: where is it on the map, which gap is it
    # actually similar to, and would it make the front at all.
    front = gap_reference(objectives, ideal_distance, point, pareto) if pareto else {}
    percentiles = grid_percentiles(objectives, chart)

    record: dict[str, Any] = {
        "metadata": {
            "stage": "explorer",
            "purpose": "Place an external document on the finished chart without "
            "altering the corpus, the embedding, the chart, or the front.",
            "reference": "Obayashi et al. (2026) Knowledge Manifold Optimization, "
            "Eqs. 6-21 evaluated out of sample",
            "label": label,
            "document": str(document),
            "document_format": document.suffix.lower() or "text",
            "document_sha256": digest,
            "chart_stress": round(chart.stress, 4),
            "chart_distance_correlation": round(chart.distance_correlation, 4),
            "corpus_documents": int(chart.coords.shape[0]),
            "grid_size": int(chart.grid_size),
            "objective_names": list(OBJECTIVE_NAMES),
            "scoring": "identical kernels, bandwidths and percentile bounds as "
            "manifold_grid.npz; nothing is refitted",
            "placement": "isotonic cosine-to-chart distance calibration plus "
            "bounded least-squares trilateration",
            "profile": str(config.get("profile", "")),
            "distance_quantiles": dict(chart.quantiles),
        },
        "quality": quality,
        "placement": {
            "x": round(float(point[0]), 6),
            "y": round(float(point[1]), 6),
            "fit_rmse": round(evaluation["fit_rmse"], 4),
            "fit_spearman": round(evaluation["fit_spearman"], 4),
            "at_chart_edge": evaluation["at_chart_edge"],
            "calibration_residual_rmse": round(calibration.residual_rmse, 4),
            "leave_one_out": {
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in loo.items()
            },
        },
        "scores": scored["scores"],
        "grid_percentiles": percentiles,
        "plain_language": plain_language(
            scored["scores"],
            percentiles,
            quality,
            mixture,
            plateau_share,
            front,
            float(evaluation["fit_rmse"]),
        ),
        "grid_context": {
            "candidates": int(white_grid.size),
            "white_space_plateau_share": round(plateau_share, 4),
            "white_space_plateau_epsilon": WHITE_SPACE_PLATEAU_EPSILON,
            "note": "W saturates between q60 and q85, so this share of the grid "
            "scores the same value; a plateau score is a healthy distance, not "
            "a unique position.",
        },
        "ideal_distance": round(ideal_distance, 6),
        "ideal_distance_reference": {
            "best_gap": round(float(grid_ideal.min()), 6),
            "grid_median": round(float(np.median(grid_ideal)), 6),
            "hypothetical_rank": int((grid_ideal < ideal_distance).sum() + 1),
            "candidates": int(grid_ideal.size),
            "note": "Hypothetical only: the idea is not a candidate and does not "
            "enter the Pareto search.",
        },
        "diagnostics": scored["diagnostics"],
        "regime_mixture": mixture,
        "gaps": gaps,
        "gap_reference": front,
        "neighbours": neighbours,
        "neighbour_views": neighbour_views(neighbours),
        "terms": idea_terms(z_l1, representation, vocabulary, local),
    }
    return record


def write_markdown(record: dict[str, Any], path: Path) -> None:
    """Render the artifact as a short human-readable report."""
    meta = record["metadata"]
    quality = record["quality"]
    placement = record["placement"]
    lines: list[str] = [
        f"# Idea placement: {meta['label']}",
        "",
        f"- document: `{meta['document']}`",
        f"- sha256: `{meta['document_sha256'][:16]}`",
        f"- chart: {meta['corpus_documents']} documents, "
        f"{meta['grid_size']}x{meta['grid_size']} grid, "
        f"stress {meta['chart_stress']:.2f}, distance correlation "
        f"{meta['chart_distance_correlation']:.3f}",
        "",
        "This document is **not** part of the corpus and is **not** included in any "
        "objective, front, or ranking. It is projected through the stored "
        "vocabulary and scored at its own coordinate.",
        "",
        "## Document",
        "",
        f"- {quality['word_count']} words, {quality['char_count']} characters after "
        f"markdown/LaTeX stripping",
        f"- vocabulary match: {quality['vocabulary_match_rate'] * 100:.1f}% of its "
        "n-grams are known to the corpus vocabulary",
        f"- text distance to the corpus: mean {quality['mean_cosine_distance']:.3f} "
        f"(percentile {quality['mean_distance_percentile']:.1f} of corpus documents), "
        f"nearest {quality['nearest_distance']:.3f} (percentile "
        f"{quality['nearest_distance_percentile']:.1f})",
        *(["- at the edge of the chart: the trilateration hit its bound, so treat the "
            "coordinate as a direction rather than a position"]
          if placement.get("at_chart_edge") else []),
        "",
        "## Position",
        "",
        f"- chart coordinate: **({placement['x']:+.3f}, {placement['y']:+.3f})**",
        f"- placement residual: {placement['fit_rmse']:.3f} chart units "
        f"(Spearman distance fit {placement['fit_spearman']:.3f})",
        f"- leave-one-out position error on this corpus: median "
        f"{placement['leave_one_out']['median']:.3f}, p90 "
        f"{placement['leave_one_out']['p90']:.3f} chart units "
        f"(typical document spacing "
        f"{placement['leave_one_out']['typical_neighbour_spacing']:.3f})",
        "",
        "## In plain words",
        "",
    ]
    plain = record.get("plain_language") or {}
    if plain.get("headline"):
        lines.append(f"**{plain['headline']}**")
        lines.append("")
    for key, label in (
        ("position", "Where it sits"),
        ("novelty", "What the novelty evidence is"),
        ("front", "What the front says"),
    ):
        if plain.get(key):
            lines += [f"- **{label}.** {plain[key]}", ""]
    for entry in plain.get("objectives") or []:
        lines.append(
            f"- **{entry['name']} {entry['value']:.3f}** "
            f"({entry['percentile']:.0f}th) — {entry['meaning']}. "
            f"For this document: {entry['reading']}."
        )
    if plain:
        lines += ["", f"_{plain.get('note', '')}_", ""]
    lines += [
        "## Objectives",
        "",
        f"| objective | value | percentile among "
        f"{record['ideal_distance_reference']['candidates']} grid points |",
        "| :--- | ---: | ---: |",
    ]
    for name in OBJECTIVE_NAMES:
        lines.append(
            f"| {name} | {record['scores'][name]:.3f} | "
            f"{record['grid_percentiles'][name]:.0f} |"
        )
    reference = record["ideal_distance_reference"]
    lines += [
        "",
        f"- distance to the ideal corner (1,1,1,1): **{record['ideal_distance']:.3f}** "
        f"(best map gap {reference['best_gap']:.3f}, grid median "
        f"{reference['grid_median']:.3f}). If this point were a candidate it would "
        f"rank {reference['hypothetical_rank']} of {reference['candidates']} by that "
        "distance; it is not one.",
        (
            f"- `W` plateau: "
            f"{record['grid_context']['white_space_plateau_share'] * 100:.1f}% "
            f"of the {record['grid_context']['candidates']} grid points score the "
            "same `W` value, because the band-pass saturates between q60 and q85. "
            "A plateau score means the nearest document is at a healthy distance; "
            "it does not mean the position is unique."
        ),
        "",
        "## Region",
        "",
        "| regime | share |",
        "| :--- | ---: |",
    ]
    for label, value in record["regime_mixture"].items():
        lines.append(f"| {label} | {value:.3f} |")
    diagnostics = record["diagnostics"]
    lines += [
        "",
        f"- nearest-document distance d_nn = {diagnostics['d_nn']:.3f} "
        "(corpus quantiles "
        + ", ".join(
            f"{name} {value:.3f}"
            for name, value in meta.get("distance_quantiles", {}).items()
        )
        + ")",
        "- persistent boundary views: "
        + ", ".join(
            f"{name} {entry['boundary']:.3f}"
            for name, entry in diagnostics["boundary_views"].items()
        ),
        f"- support: N_eff {diagnostics['n_eff']:.1f}, d5 {diagnostics['d5']:.3f}, "
        f"T3 {diagnostics['top3']:.3f}",
        "",
        "## Nearest corpus documents",
        "",
        (
            "Two orderings, reported separately: the chart distorts text distance "
            f"(stress {meta['chart_stress']:.2f}, distance correlation "
            f"{meta['chart_distance_correlation']:.3f}), so the document nearest "
            "in the chart is not always the document nearest in words."
        ),
        "",
        "### Nearest in text (cosine similarity)",
        "",
        "| text rank | chart rank | cos sim | chart d | regime | title |",
        "| ---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    for neighbour in record["neighbour_views"]["by_text"]:
        lines.append(
            f"| {neighbour['cosine_rank'] + 1} | {neighbour['chart_rank'] + 1} | "
            f"{neighbour['cosine_similarity']:.3f} | "
            f"{neighbour['chart_distance']:.3f} | "
            f"{neighbour['regime_label']} | {neighbour['title']} |"
        )
    lines += [
        "",
        "### Nearest on the map (chart distance)",
        "",
        "| chart rank | text rank | chart d | cos sim | regime | title |",
        "| ---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    for neighbour in record["neighbour_views"]["by_chart"]:
        lines.append(
            f"| {neighbour['chart_rank'] + 1} | {neighbour['cosine_rank'] + 1} | "
            f"{neighbour['chart_distance']:.3f} | "
            f"{neighbour['cosine_similarity']:.3f} | "
            f"{neighbour['regime_label']} | {neighbour['title']} |"
        )
    lines += [
        "",
        "### Full neighbour union",
        "",
        "| chart d | cos sim | regime | title |",
        "| ---: | ---: | :--- | :--- |",
    ]
    for neighbour in record["neighbours"][:NEIGHBOUR_COUNT]:
        lines.append(
            f"| {neighbour['chart_distance']:.3f} | "
            f"{neighbour['cosine_similarity']:.3f} | "
            f"{neighbour['regime_label']} | {neighbour['title']} |"
        )
    if record["gaps"]:
        front = record.get("gap_reference") or {}
        members = len(front.get("dominated_by") or [])
        lines += [
            "",
            "## Nearest discovery gaps",
            "",
            (
                "The idea has no gap rank: the front holds a finite set of "
                "discrete candidates ranked by d_ideal, and this document is not "
                "a candidate. The rows below are geometric neighbours, not a "
                "ranking. Where the document would fall if inserted:"
            ),
            "",
        ]
        if front.get("nearest_chart"):
            nearest = front["nearest_chart"]
            lines.append(
                f"- nearest on the map: gap #{nearest['rank']} "
                f"({nearest['chart_distance']:.3f} chart units, its d_ideal "
                f"{nearest['ideal_distance']:.3f}) -- a projection neighbour."
            )
        if front.get("nearest_objective"):
            similar = front["nearest_objective"]
            lines.append(
                f"- nearest in objectives: gap #{similar['rank']} "
                f"(Delta {similar['objective_distance']:.3f} in P, W, G, S, its "
                f"d_ideal {similar['ideal_distance']:.3f}) -- the actually similar "
                "proposal, which the 2-D projection cannot express."
            )
        if front.get("insertion"):
            slot = front["insertion"]
            lines.append(
                f"- if inserted by d_ideal: #{slot['rank']} of {slot['of']} "
                "(hypothetical: insertion is arithmetic, not front membership)."
            )
        if members:
            dominator = front["dominated_by"][0]
            lines.append(
                f"- front membership: no. Gap #{dominator['rank']} scores at "
                "least as high on all four objectives, so it dominates the "
                "document regardless of its d_ideal."
            )
        else:
            lines.append(
                "- front membership: no gap dominates the document on all four "
                f"objectives -- but read that against the placement residual "
                f"({placement['fit_rmse']:.3f} chart units): non-domination is a "
                "knife-edge property of one 4-D coordinate, and it flips inside "
                "that radius."
            )
        lines += [
            "",
            "| gap | x | y | d_ideal | archetype | distance |",
            "| ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
        for gap in record["gaps"]:
            lines.append(
                f"| #{gap['rank']} | {gap['x']:+.2f} | {gap['y']:+.2f} | "
                f"{gap['ideal_distance']:.3f} | {gap['archetype_label']} | "
                f"{gap['chart_distance']:.3f} |"
            )
    lines += ["", "## Terms", ""]
    novel = record["terms"]["not_in_corpus"]
    lines.append(
        "- vocabulary the corpus barely covers (df <= 3): "
        + (", ".join(f"`{entry['term']}`" for entry in novel) if novel else "none")
    )
    theme = record["terms"]["map_theme_lens"]
    lines.append(
        "- what the map says dominates this location: "
        + (", ".join(f"`{entry['term']}`" for entry in theme) if theme else "none")
    )
    if quality["warnings"]:
        lines += ["", "## Caveats", ""]
        lines += [f"- {warning}" for warning in quality["warnings"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_artifact(
    paths: Paths, config: dict[str, Any], tolerance: float = SCORE_TOLERANCE
) -> bool:
    """Re-derive the artifact from the document and compare with what is saved."""
    if not paths.explorer_json.is_file():
        LOGGER.error("Missing %s", paths.explorer_json)
        return False
    stored = load_json(paths.explorer_json, "explorer artifact")
    meta = stored.get("metadata", {})
    document = Path(str(meta.get("document", "")))
    if not document.is_file():
        LOGGER.error("Idea document recorded in the artifact is gone: %s", document)
        return False

    rebuilt = build_artifact(config, paths, document, str(meta.get("label", "Idea")))

    ok = True
    position_stored = np.array(
        [stored["placement"]["x"], stored["placement"]["y"]], dtype=np.float64
    )
    position_rebuilt = np.array(
        [rebuilt["placement"]["x"], rebuilt["placement"]["y"]], dtype=np.float64
    )
    drift = float(np.linalg.norm(position_stored - position_rebuilt))
    if drift > PLACEMENT_TOLERANCE:
        LOGGER.error("Placement drifted by %.4f chart units", drift)
        ok = False
    else:
        LOGGER.info("Placement reproducible (drift %.2e chart units)", drift)

    for name in OBJECTIVE_NAMES:
        difference = abs(stored["scores"][name] - rebuilt["scores"][name])
        if difference > tolerance:
            LOGGER.error("Objective %s drifted by %.2e", name, difference)
            ok = False
    if ok:
        LOGGER.info("Objectives reproduce within %.0e", tolerance)

    stored_plateau = stored.get("grid_context", {}).get("white_space_plateau_share")
    if stored_plateau is None or abs(
        stored_plateau - rebuilt["grid_context"]["white_space_plateau_share"]
    ) > tolerance:
        LOGGER.error(
            "W plateau share drifted: stored %s, rebuilt %s",
            stored_plateau,
            rebuilt["grid_context"]["white_space_plateau_share"],
        )
        ok = False
    else:
        LOGGER.info(
            "W plateau share reproduces (%.1f%% of %d grid points)",
            stored_plateau * 100,
            rebuilt["grid_context"]["candidates"],
        )

    stored_front = stored.get("gap_reference") or {}
    rebuilt_front = rebuilt.get("gap_reference") or {}
    if stored_front.get("nearest_chart", {}).get("rank") != rebuilt_front.get(
        "nearest_chart", {}
    ).get("rank") or stored_front.get("nearest_objective", {}).get("rank") != rebuilt_front.get(
        "nearest_objective", {}
    ).get("rank"):
        LOGGER.error("Front reference drifted: the nearest gaps changed")
        ok = False
    elif stored_front.get("insertion", {}).get("rank") != rebuilt_front.get(
        "insertion", {}
    ).get("rank"):
        LOGGER.error("Front insertion slot drifted")
        ok = False
    else:
        LOGGER.info(
            "Front reference reproduces (map #%s, objectives #%s, insertion #%s of %s)",
            stored_front.get("nearest_chart", {}).get("rank"),
            stored_front.get("nearest_objective", {}).get("rank"),
            stored_front.get("insertion", {}).get("rank"),
            stored_front.get("insertion", {}).get("of"),
        )

    corpus_papers = load_json(paths.corpus, "corpus").get("papers", [])
    corpus_ids = {str(paper["id"]) for paper in corpus_papers}
    corpus_digests = {
        hashlib.sha256(str(paper.get("full_text", "")).encode("utf-8")).hexdigest()
        for paper in corpus_papers
        if paper.get("full_text")
    }
    if meta.get("document_sha256") in corpus_digests:
        LOGGER.error("The idea document is in the corpus; the overlay is invalid")
        ok = False
    if meta.get("label", "").lower() in {identifier.lower() for identifier in corpus_ids}:
        LOGGER.error("The idea label collides with a corpus document id")
        ok = False

    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place an external idea document on a finished research radar."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Idea document (default: explorer.document from config.yaml).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Display label (default: explorer.label from config.yaml).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-derive the saved idea artifact and check it reproduces.",
    )
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="Skip the leave-one-out placement error (saves ~1 minute).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths: Paths = resolve_paths(config, args.config)

    if not paths.grid_npz.is_file():
        LOGGER.error("Missing %s. Run 'python -m src.manifold' first.", paths.grid_npz)
        return 1
    if not (paths.tfidf_matrix.is_file() and paths.tfidf_vocab.is_file()):
        LOGGER.error("Missing TF-IDF artifacts. Run 'python -m src.embed' first.")
        return 1

    if args.verify_only:
        return 0 if verify_artifact(paths, config) else 1

    explorer_cfg = config.get("explorer", {}) or {}
    document = args.doc or Path(str(explorer_cfg.get("document", "")))
    label = args.label or str(explorer_cfg.get("label", "External idea"))
    if not str(document) or str(document) in {".", ""}:
        LOGGER.error(
            "No idea document. Pass --doc PATH or set explorer.document in %s",
            args.config,
        )
        return 1

    if args.skip_loo:
        LOGGER.info("Skipping leave-one-out calibration (--skip-loo)")
    record = build_artifact(config, paths, Path(document), label)
    if args.skip_loo:
        record["placement"]["leave_one_out"]["median"] = None
        record["placement"]["leave_one_out"]["p90"] = None

    paths.explorer_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.explorer_json.with_suffix(paths.explorer_json.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    temporary.replace(paths.explorer_json)

    write_markdown(record, paths.idea_report)

    LOGGER.info(
        "Placed %r at (%+.3f, %+.3f): P_boundary=%.3f W=%.3f G=%.3f S=%.3f "
        "d_ideal=%.3f",
        record["metadata"]["label"],
        record["placement"]["x"],
        record["placement"]["y"],
        record["scores"]["P_boundary"],
        record["scores"]["W"],
        record["scores"]["G"],
        record["scores"]["S"],
        record["ideal_distance"],
    )
    for warning in record["quality"]["warnings"]:
        LOGGER.warning("Caveat: %s", warning)
    LOGGER.info("Wrote %s and %s", paths.explorer_json, paths.idea_report)

    if "leave_one_out" in record["placement"]:
        LOGGER.info(
            "Leave-one-out position error: median %.3f, p90 %.3f chart units "
            "(typical document spacing %.3f)",
            record["placement"]["leave_one_out"]["median"] or 0.0,
            record["placement"]["leave_one_out"]["p90"] or 0.0,
            record["placement"]["leave_one_out"]["typical_neighbour_spacing"],
        )
    LOGGER.info(
        "The corpus, chart, and front are untouched: run 'python -m src.dashboard' "
        "to overlay the idea."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
