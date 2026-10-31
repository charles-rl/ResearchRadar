"""Interactive map, Markdown report, and the verbatim KMO LLM prompt payload.

Implements PLAN.md Phase 6 for Obayashi et al. (2026), Sections 7.1-7.3 and
Appendix A.1:

* ``output/radar_map.html`` - Plotly chart of the 120 corpus papers coloured by
  regime, with the practical Pareto gaps overlaid as stars and the four discovery
  fields available as togglable heatmaps.
* ``output/RADAR_REPORT.md`` - overview table of every practical Pareto gap, a
  deep dive on the top three archetypes with their top five contributing anchor
  papers and weights, the local vocabulary analysis (L2 theme lens versus L1
  concentration lens, split into emergent shared topics at ``df >= 4`` and
  limited-source hypotheses at ``df <= 3``), and the Appendix A.1 prompt
  pre-filled with the evidence package for the #1 compromise gap.

The Appendix A.1 template is stored verbatim in
``docs/kmo_prompt_template_A1.txt`` and only the placeholder file name is
substituted, so the prompt text cannot drift from the paper.

Usage
-----
    python -m src.report
    python -m src.report --verify-only
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import LOGGER, Paths, configure_logging, load_config, resolve_paths

DEFAULT_CONFIG = Path("config.yaml")
PROMPT_TEMPLATE = Path("docs/kmo_prompt_template_A1.txt")

REGIME_COLOURS = {
    "SysID": "#1f77b4",
    "Sim2Real": "#2ca02c",
    "RL": "#d62728",
}

# Hand-curated prose filter. These are words that carry no topical information in
# an abstract; without it the L1 concentration lens surfaces generic academic
# filler ("drawn", "main", "illustrate") because such words have tiny global mass
# and therefore a huge local/global enrichment ratio.
PROSE_FILLER = frozenset(
    ["approach", "approaches", "article", "based", "best", "better", "carried", "case", "cases", "change", "changes", "clear", "clearly", "common", "commonly", "compare", "compared", "comparing", "conclusion", "conclusions", "consider", "considered", "demonstrate", "demonstrated", "demonstrates", "describe", "described", "different", "discussion", "drawn", "during", "effective", "efficiently", "especially", "even", "example", "examples", "exhibit", "existing", "exploit", "first", "focus", "focused", "following", "found", "furthermore", "future", "general", "given", "gives", "good", "great", "high", "higher", "however", "illustrate", "illustrated", "illustrative", "important", "improve", "improved", "increasing", "introduction", "large", "larger", "latter", "lead", "leading", "less", "like", "low", "lower", "made", "main", "make", "makes", "many", "may", "method", "methods", "might", "much", "must", "new", "newly", "novel", "often", "one", "only", "operations", "operative", "others", "outperform", "outperformed", "outperforms", "paper", "papers", "particular", "particularly", "performed", "possible", "prescriptive", "present", "presented", "presents", "previous", "previously", "primary", "provide", "provides", "purpose", "rather", "recent", "recently", "recognition", "report", "reported", "results", "second", "section", "several", "show", "showed", "shown", "significant", "significantly", "similar", "simple", "simpler", "since", "small", "smaller", "study", "sudden", "suggest", "suggests", "take", "taken", "takes", "terms", "therefore", "third", "though", "three", "tools", "towards", "tutorial", "two", "typically", "understand", "understanding", "unlike", "upon", "used", "useful", "using", "various", "vast", "well", "whether", "within", "without", "work", "works", "would"]
)

# Word tokenizer for the two lenses. Applied through the estimator's
# ``tokenizer`` hook rather than through a regex string, which keeps the
# tokenization rule in one named place and testable on its own.
_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z-]{2,}")


def tokenize_word(text: str) -> list[str]:
    """Split text into lowercase content words of at least three letters."""
    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]


#: Terms that must never enter the lens lists, on top of scikit-learn's stopwords.
MIN_TERM_LENGTH = 3
MIN_LOCAL_L1_SHARE = 0.002
MIN_ENRICHMENT = 1.5
THEME_LENS_SIZE = 12
CONCENTRATION_LENS_SIZE = 12
EMERGENT_DF = 4
LIMITED_SOURCE_DF = 3


# --------------------------------------------------------------------------- #
# Vocabulary and the two lenses
# --------------------------------------------------------------------------- #
@dataclass
class Vocabulary:
    """Corpus TF-IDF matrices in the L2 and L1 normalizations (Section 7.1)."""

    terms: NDArray[np.str_]
    l2: NDArray[np.float64]
    l1: NDArray[np.float64]
    document_frequency: NDArray[np.int64]
    mean_l1: NDArray[np.float64]
    usable: NDArray[np.bool_]
    word_forms: dict[str, list[str]]


def build_vocabulary(papers: list[dict[str, Any]]) -> Vocabulary:
    """Fit TF-IDF on titles+abstracts and derive both normalizations.

    The same vocabulary and term weights are used for both lenses; only the row
    normalization differs (``l2`` for the semantic geometry, ``l1`` for feature
    allocation), which is how the paper separates the two views.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [f"{paper['title']}. {paper['abstract']}" for paper in papers]
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        tokenizer=tokenize_word,
        # ``token_pattern`` is stubbed as ``str`` only, but sklearn documents and
        # accepts None to disable it when a ``tokenizer`` is supplied; leaving it
        # set emits "The parameter 'token_pattern' will not be used" on every run.
        token_pattern=None,  # type: ignore[arg-type]
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = np.asarray(vectorizer.get_feature_names_out())

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
    l1_matrix = matrix.multiply(1.0 / row_sums[:, None]).tocsr()

    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.int64)
    mean_l1 = np.asarray(l1_matrix.mean(axis=0)).ravel().astype(np.float64)

    lower_terms = terms.tolist()
    vocabulary_index = {term: index for index, term in enumerate(lower_terms)}
    word_forms = {
        term: _attested_forms(term, vocabulary_index) for term in lower_terms
    }

    usable = np.array(
        [
            len(term) >= MIN_TERM_LENGTH and term not in PROSE_FILLER
            for term in lower_terms
        ],
        dtype=bool,
    )

    LOGGER.info(
        "Vocabulary: %d terms, %d usable after the prose filter",
        len(terms),
        int(usable.sum()),
    )
    return Vocabulary(
        terms=terms,
        l2=matrix.toarray(),  # type: ignore[assignment]
        l1=l1_matrix.toarray(),  # type: ignore[assignment]
        document_frequency=document_frequency,
        mean_l1=mean_l1,
        usable=usable,
        word_forms=word_forms,
    )


def _attested_forms(term: str, index: dict[str, int]) -> list[str]:
    """Collect surface variants of ``term`` that actually occur in the corpus."""
    stems = {term}
    for suffix in (
        "ations", "ation", "ions", "ion", "ings", "ing", "ies", "ied",
        "es", "ed", "s", "al", "ally", "ive", "ity", "ities",
    ):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stems.add(term[: -len(suffix)])
    stems.add(term + "e")
    stems.add(term + "y")

    forms: set[str] = set()
    for stem in stems:
        for suffix in ("", "s", "es", "d", "ed", "ing", "ion", "ions",
                       "al", "ive", "ity", "ies", "ied"):
            candidate = stem + suffix
            if candidate in index:
                forms.add(candidate)
    forms.add(term)
    return sorted(forms)


def witness_documents(
    column: NDArray[np.float64] | NDArray[np.int64],
    normalized_weights: NDArray[np.float64],
    papers: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Name the documents carrying a term, ordered by contribution weight."""
    order = np.argsort(normalized_weights)[::-1]
    witnesses: list[dict[str, Any]] = []
    for index in order:
        if len(witnesses) >= limit:
            break
        if float(normalized_weights[index]) <= 0.0 or float(column[index]) <= 0.0:
            continue
        witnesses.append(
            {
                "doc": str(papers[int(index)]["id"]),
                "title": str(papers[int(index)]["title"]),
                "weight": round(float(normalized_weights[int(index)]), 4),
            }
        )
    return witnesses


def local_lenses(
    weights: NDArray[np.float64],
    vocabulary: Vocabulary,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the L2 theme lens and L1 concentration lens for one location."""
    total = float(weights.sum())
    normalized = weights / total if total > 0 else np.zeros_like(weights)

    local_l2 = normalized @ vocabulary.l2
    local_l1 = normalized @ vocabulary.l1
    enrichment = local_l1 / np.maximum(vocabulary.mean_l1, 1e-12)

    theme_candidates = np.flatnonzero(
        vocabulary.usable
        & (vocabulary.document_frequency >= 2)
        & (local_l2 > 0.0)
    )
    theme_order = theme_candidates[np.argsort(local_l2[theme_candidates])[::-1]]
    theme_order = theme_order[:THEME_LENS_SIZE]

    concentration_candidates = np.flatnonzero(
        vocabulary.usable
        & (local_l1 >= MIN_LOCAL_L1_SHARE)
        & (enrichment >= MIN_ENRICHMENT)
    )
    concentration_order = concentration_candidates[
        np.argsort(enrichment[concentration_candidates])[::-1]
    ]
    concentration_order = concentration_order[:CONCENTRATION_LENS_SIZE]

    theme_set = {int(index) for index in theme_order}
    theme_lens = [
        {
            "term": str(vocabulary.terms[index]),
            "weight": round(float(local_l2[index]), 5),
            "df": int(vocabulary.document_frequency[index]),
            "word_forms": vocabulary.word_forms[str(vocabulary.terms[index])],
            "in_both_lenses": False,
        }
        for index in theme_order
    ]

    concentration_lens = []
    for index in concentration_order:
        term = str(vocabulary.terms[index])
        entry: dict[str, Any] = {
            "term": term,
            "local_l1": round(float(local_l1[index]), 6),
            "enrichment": round(float(enrichment[index]), 3),
            "df": int(vocabulary.document_frequency[index]),
            "word_forms": vocabulary.word_forms[term],
        }
        if entry["df"] <= LIMITED_SOURCE_DF:
            entry["source_docs"] = witness_documents(
                vocabulary.l1[:, index], normalized, papers
            )
        entry["in_both_lenses"] = int(index) in theme_set
        concentration_lens.append(entry)

    for entry in theme_lens:
        entry["in_both_lenses"] = any(
            candidate["term"] == entry["term"] for candidate in concentration_lens
        )

    return {
        "theme_lens_L2": theme_lens,
        "concentration_lens_L1": concentration_lens,
        "shared_terms": [
            entry["term"]
            for entry in theme_lens
            if entry["in_both_lenses"]
        ],
        "local_l2": local_l2,
        "local_l1": local_l1,
        "enrichment": enrichment,
    }


# --------------------------------------------------------------------------- #
# Evidence package (Section 7.1)
# --------------------------------------------------------------------------- #
def contributing_documents(
    weights: NDArray[np.float64], papers: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    """Highest-weight neighbouring papers for one location."""
    total = float(weights.sum())
    normalized = weights / total if total > 0 else np.zeros_like(weights)
    order = np.argsort(normalized)[::-1][:limit]
    return [
        {
            "doc": str(papers[int(index)]["id"]),
            "title": str(papers[int(index)]["title"]),
            "authors": list(papers[int(index)]["authors"])[:3],
            "year": int(papers[int(index)]["year"]),
            "regime": str(papers[int(index)]["regime_label"]),
            "weight": round(float(normalized[int(index)]), 4),
        }
        for index in order
    ]


def build_evidence_package(
    point: dict[str, Any],
    grid_data: dict[str, NDArray[np.float64] | NDArray[np.int64] | NDArray[np.str_]],
    vocabulary: Vocabulary,
    papers: list[dict[str, Any]],
    regime_labels: list[str],
) -> dict[str, Any]:
    """Assemble the machine-computed evidence package for one Pareto location."""
    index = int(point["grid_index"])
    weights = np.asarray(grid_data["weights_global"][index], dtype=np.float64)
    lenses = local_lenses(weights, vocabulary, papers)

    mixture_global = np.asarray(grid_data["mixture_global"][index], dtype=np.float64)
    mixture_adaptive = np.asarray(grid_data["mixture_adaptive"][index], dtype=np.float64)

    mixture = {
        label: {
            "global": round(float(mixture_global[position]), 4),
            "adaptive": round(float(mixture_adaptive[position]), 4),
        }
        for position, label in enumerate(regime_labels)
    }

    n_eff = float(np.asarray(grid_data["n_eff"])[index])
    top3 = float(np.asarray(grid_data["top3"])[index])
    d5 = float(np.asarray(grid_data["d5"])[index])
    entropy = float(np.asarray(grid_data["h_global_entropy"])[index])
    uncertainty = float(np.asarray(grid_data["uncertainty"])[index])

    return {
        "location": {
            "x": round(float(point["x"]), 2),
            "y": round(float(point["y"]), 2),
        },
        "theme_lens_L2": lenses["theme_lens_L2"],
        "concentration_lens_L1": lenses["concentration_lens_L1"],
        "contributing_documents_top3": contributing_documents(weights, papers, limit=3),
        "contributing_documents_top5": contributing_documents(weights, papers, limit=5),
        "mixture_entropy_H": round(entropy, 4),
        "regime_mixture": mixture,
        "effective_contributing_documents": round(n_eff, 2),
        "relative_uncertainty_u": round(uncertainty, 6),
        "top3_weight_concentration_T3": round(top3, 4),
        "fifth_neighbour_distance_d5": round(d5, 4),
        "objectives": point["objectives"],
        "rank": point["rank"],
        "ideal_distance": point["ideal_distance"],
        "on_outer_boundary": point["on_outer_boundary"],
    }


def prompt_evidence_package(package: dict[str, Any]) -> dict[str, Any]:
    """Trim the evidence package to exactly the fields Appendix A.1 describes."""
    return {
        "location": package["location"],
        "theme_lens_L2": [
            {"term": entry["term"], "weight": entry["weight"], "df": entry["df"],
             "word_forms": entry["word_forms"]}
            for entry in package["theme_lens_L2"]
        ],
        "concentration_lens_L1": [
            {
                "term": entry["term"],
                "df": entry["df"],
                "word_forms": entry["word_forms"],
                "in_both_lenses": entry["in_both_lenses"],
                **(
                    {"source_docs": entry["source_docs"]}
                    if "source_docs" in entry
                    else {}
                ),
            }
            for entry in package["concentration_lens_L1"]
        ],
        "contributing_documents_top3": package["contributing_documents_top3"],
        "mixture_entropy_H": package["mixture_entropy_H"],
        "effective_contributing_documents": package[
            "effective_contributing_documents"
        ],
        "relative_uncertainty_u": package["relative_uncertainty_u"],
    }


# --------------------------------------------------------------------------- #
# Plotly figure
# --------------------------------------------------------------------------- #
def build_figure(
    corpus: dict[str, Any],
    grid_data: dict[str, NDArray[np.float64] | NDArray[np.int64] | NDArray[np.str_]],
    front: dict[str, Any],
    objective_names: list[str],
) -> Any:
    """Build the interactive scatter + Pareto overlay figure."""
    import plotly.graph_objects as go

    papers = corpus["papers"]
    coords = np.asarray(grid_data["doc_coords"], dtype=np.float64)
    grid = np.asarray(grid_data["grid"], dtype=np.float64)
    objectives = np.asarray(grid_data["objectives"], dtype=np.float64)
    grid_size = round(np.sqrt(len(grid)))
    axis = np.unique(np.round(grid[:, 0], 10))

    traces: list[Any] = []
    for label in ("SysID", "Sim2Real", "RL"):
        rows = [
            position
            for position, paper in enumerate(papers)
            if paper["regime_label"] == label
        ]
        if not rows:
            continue
        hover = []
        for position in rows:
            paper = papers[position]
            authors = paper["authors"]
            author_text = ", ".join(authors[:3])
            if len(authors) > 3:
                author_text += " et al."
            hover.append(
                f"<b>{html_module.escape(str(paper['title']))}</b><br>"
                f"{html_module.escape(author_text)}<br>"
                f"{paper['year']} &middot; {paper['regime_label']}<br>"
                f"arXiv {paper['id']}"
            )
        traces.append(
            go.Scatter(
                x=coords[rows, 0],
                y=coords[rows, 1],
                mode="markers",
                name=label,
                marker={
                    "size": 9,
                    "color": REGIME_COLOURS.get(label, "#7f7f7f"),
                    "line": {"width": 0.6, "color": "white"},
                    "opacity": 0.85,
                },
                text=hover,
                hoverinfo="text",
                legendgroup=label,
            )
        )

    stars = front["solutions"]
    star_hover = []
    for solution in stars:
        anchors = solution.get("top_anchors", [])
        anchor_text = "<br>".join(
            f"&nbsp;&nbsp;{anchor['weight']:.3f} &middot; "
            f"{html_module.escape(anchor['title'][:70])}"
            for anchor in anchors[:3]
        )
        scores = " &middot; ".join(
            f"{name} {value:.2f}" for name, value in solution["objectives"].items()
        )
        star_hover.append(
            f"<b>Gap #{solution['rank']}</b> "
            f"(archetype {solution.get('archetype', '?')})<br>"
            f"chart ({solution['x']:+.2f}, {solution['y']:+.2f})<br>"
            f"{scores}<br>"
            f"d<sub>ideal</sub> {solution['ideal_distance']:.3f} &middot; "
            f"N<sub>eff</sub> {solution.get('n_eff', float('nan')):.1f}<br>"
            f"<i>top contributors</i><br>{anchor_text}"
        )

    top = front["solutions"][0]
    traces.append(
        go.Scatter(
            x=[solution["x"] for solution in stars],
            y=[solution["y"] for solution in stars],
            mode="markers",
            name="Practical Pareto gaps",
            marker={
                "symbol": "star",
                "size": 17,
                "color": "#111111",
                "line": {"width": 1.2, "color": "#ffd400"},
            },
            text=star_hover,
            hoverinfo="text",
            legendgroup="pareto",
        )
    )

    # Rank-1 gap called out separately so it stays identifiable at a glance.
    traces.append(
        go.Scatter(
            x=[top["x"]],
            y=[top["y"]],
            mode="markers+text",
            name="#1 compromise",
            marker={"symbol": "star", "size": 24, "color": "#ffd400",
                    "line": {"width": 1.5, "color": "#111111"}},
            text=[f"#1 ({top['x']:+.2f}, {top['y']:+.2f})"],
            textposition="top center",
            textfont={"size": 11, "color": "#111111"},
            hoverinfo="skip",
            legendgroup="pareto",
        )
    )

    base_count = len(traces)
    for position, name in enumerate(objective_names):
        field = objectives[:, position].reshape(grid_size, grid_size)
        traces.append(
            go.Heatmap(
                x=axis,
                y=axis,
                z=field,
                name=name,
                colorscale="Viridis",
                opacity=0.72,
                showscale=True,
                colorbar={"title": name, "thickness": 12, "len": 0.6},
                hovertemplate=f"{name} %{{z:.3f}}<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
                visible=False,
            )
        )

    buttons = [
        {
            "label": "Papers + gaps",
            "method": "update",
            "args": [
                {"visible": [True] * base_count + [False] * len(objective_names)},
                {"title.text": "ResearchRadar — Knowledge Manifold and practical Pareto gaps"},
            ],
        }
    ]
    for position, name in enumerate(objective_names):
        visibility = [True] * base_count + [False] * len(objective_names)
        visibility[base_count + position] = True
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title.text": f"ResearchRadar — {name} field"},
                ],
            }
        )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title={
            "text": "ResearchRadar — Knowledge Manifold and practical Pareto gaps",
            "x": 0.5,
        },
        template="plotly_white",
        width=1000,
        height=820,
        xaxis={
            "title": "Knowledge Manifold x",
            "range": [-1.05, 1.05],
            "showgrid": True,
            "zeroline": True,
            "zerolinecolor": "#cccccc",
            "constrain": "domain",
        },
        yaxis={
            "title": "Knowledge Manifold y",
            "range": [-1.05, 1.05],
            "showgrid": True,
            "zeroline": True,
            "zerolinecolor": "#cccccc",
            "scaleanchor": "x",
            "scaleratio": 1.0,
        },
        legend={"title": {"text": "Regime / front"}, "bgcolor": "rgba(255,255,255,0.85)"},
        hovermode="closest",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.09,
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
        margin={"l": 60, "r": 30, "t": 110, "b": 60},
    )
    return figure


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Escape Markdown table-hostile characters in free text."""
    return text.replace("|", "\\|").replace("\n", " ")


def archive_scalar(
    grid_data: dict[str, Any], key: str, default: float = 0.0
) -> float:
    """Read a single saved scalar from the grid archive.

    Scalars are stored as length-1 arrays by ``np.savez``, so they must be
    unwrapped before ``float()``; ``float()`` on a 1-element array raises
    ``TypeError`` rather than converting.
    """
    value = np.asarray(grid_data.get(key, default)).ravel()
    return float(value[0]) if value.size else float(default)


def _format_term_list(entries: list[dict[str, Any]], limit: int = 12) -> str:
    """Format lens terms with their df for compact prose."""
    if not entries:
        return "*(none)*"
    parts = [f"`{entry['term']}` (df={entry['df']})" for entry in entries[:limit]]
    return ", ".join(parts)


def render_markdown(
    corpus: dict[str, Any],
    grid_data: dict[str, Any],
    front: dict[str, Any],
    vocabulary: Vocabulary,
    packages: dict[int, dict[str, Any]],
    prompt_text: str,
    config: dict[str, Any],
) -> str:
    """Compose the full Markdown report."""
    papers = corpus["papers"]
    metadata = front["metadata"]
    solutions = front["solutions"]
    archetypes = front["archetypes"]
    grid_size = int(metadata["candidates"] ** 0.5)
    manifold_cfg = config.get("manifold", {})
    quantiles = {
        key: float(value)
        for key, value in zip(
            ("q20", "q60", "q85", "q98"),
            np.asarray(
                grid_data.get("whitespace_quantiles", np.zeros(4))
            ).ravel(),
            strict=False,
        )
    }

    lines: list[str] = []
    lines.append("# ResearchRadar — Supported Knowledge Gaps in Robot Learning")
    lines.append("")
    lines.append(
        "Automated Knowledge Manifold Optimization (KMO) over a balanced 120-paper "
        "arXiv corpus spanning three conflicting robotics regimes: classical system "
        "identification, sim-to-real / domain randomization, and deep reinforcement "
        "learning control."
    )
    lines.append("")
    lines.append(
        f"Method: Obayashi et al. (2026), *Knowledge Manifold Optimization*. "
        f"Corpus `{corpus['metadata']['total_papers']}` papers "
        f"({', '.join(f'{k} {v}' for k, v in corpus['metadata']['regime_counts'].items())}), "
        f"embedded with `{config.get('ollama', {}).get('model', 'F2LLM')}` "
        f"({int(np.asarray(grid_data['doc_coords']).shape[0])} documents on the chart)."
    )
    lines.append("")
    lines.append("All novelty statements are **relative to this corpus** and are not claims of global scientific novelty.")
    lines.append("")

    # --- manifold summary ------------------------------------------------
    lines.append("## 1. Knowledge Manifold")
    lines.append("")
    lines.append(
        _markdown_table(
            ["Quantity", "Value"],
            [
                ["Chart", "Ω = [−1, 1]², fixed"],
                ["Documents", str(len(papers))],
                ["Gauge-fixing anchors", "9 (8 perimeter + 1 central)"],
                [
                    "Stress minimization",
                    f"metric stress {archive_scalar(grid_data, 'stress'):.2f}, "
                    + f"Spearman(distance) {archive_scalar(grid_data, 'distance_correlation'):.4f}",
                ],
                ["Evaluation grid", f"{grid_size} × {grid_size} = {metadata['candidates']} candidates"],
                ["SPH bandwidth h (global)", f"{archive_scalar(grid_data, 'sph_h', 0.25):.2f}"],
                ["Adaptive neighbourhood", f"{int(archive_scalar(grid_data, 'nearest_k', 8))} nearest documents"],
                [
                    "White-space quantiles (this corpus)",
                    ", ".join(
                        f"{key} {float(value):.4f}" for key, value in quantiles.items()
                    ),
                ],
                [
                    "White-space quantiles (paper)",
                    ", ".join(
                        f"{key} {float(value):.4f}"
                        for key, value in (manifold_cfg.get("q_whitespace", {}) or {}).items()
                    ),
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        "The band-pass window for `W` is recomputed on this corpus's own "
        "nearest-document distances (`use_empirical_quantiles: true`), because the "
        "paper's quantiles belong to its aerodynamic corpus at a different map scale. "
        "The paper's values are retained here as the documented reference."
    )
    lines.append("")

    # --- pareto overview ------------------------------------------------
    lines.append("## 2. Practical Pareto front")
    lines.append("")
    lines.append(
        f"Exact pairwise dominance over all {metadata['candidates']} candidates gives "
        f"**{metadata['non_dominated']} non-dominated** points; requiring "
        f"`W(x) > {metadata['min_whitespace']}` (Eq. 25) leaves the "
        f"**{metadata['practical_front']} practical gaps** below, ranked by Euclidean "
        f"distance to the ideal corner (1, 1, 1, 1). Points on the chart's outer edge "
        f"stay in the front but are interpreted cautiously (Sec. 4.5)."
    )
    lines.append("")

    headers = ["Rank", "x", "y", "P_boundary", "W", "G", "S", "d_ideal", "Archetype", "N_eff"]
    rows: list[list[str]] = []
    for solution in solutions:
        scores = solution["objectives"]
        archetype = solution.get("archetype", "—")
        rows.append(
            [
                str(solution["rank"]),
                f"{solution['x']:+.2f}",
                f"{solution['y']:+.2f}",
                f"{scores['P_boundary']:.3f}",
                f"{scores['W']:.3f}",
                f"{scores['G']:.3f}",
                f"{scores['S']:.3f}",
                f"{solution['ideal_distance']:.4f}",
                archetype,
                f"{solution.get('n_eff', float('nan')):.1f}",
            ]
        )
    lines.append(_markdown_table(headers, rows))
    lines.append("")

    # --- archetypes -----------------------------------------------------
    lines.append("## 3. Discovery archetypes")
    lines.append("")
    lines.append(
        f"Per Sec. 4.7 the front is a structured set, not a ranked list: nearby points "
        f"represent the same opportunity. Single-linkage grouping at radius "
        f"`{metadata['archetype_radius']}` yields {len(archetypes)} archetypes."
    )
    lines.append("")
    lines.append(
        _markdown_table(
            ["Archetype", "Points", "Representative", "Objectives", "d_ideal"],
            [
                [
                    str(archetype["label"]),
                    str(archetype["size"]),
                    f"({archetype['representative']['x']:+.2f}, {archetype['representative']['y']:+.2f})",
                    ", ".join(
                        f"{name} {value:.2f}"
                        for name, value in archetype["representative"]["objectives"].items()
                    ),
                    f"{archetype['representative']['ideal_distance']:.4f}",
                ]
                for archetype in archetypes
            ],
        )
    )
    lines.append("")

    # --- deep dives -----------------------------------------------------
    lines.append("## 4. Deep dive: top 3 Pareto archetypes")
    lines.append("")
    if not archetypes:
        lines.append("_No archetypes available._")
        lines.append("")
    for archetype in archetypes[:3]:
        representative = archetype["representative"]
        package = packages[representative["grid_index"]]
        scores = representative["objectives"]
        lines.append(
            f"### {archetype['label']} — gap #{representative['rank']} at "
            f"({representative['x']:+.2f}, {representative['y']:+.2f})"
        )
        lines.append("")
        lines.append(
            f"**{archetype['size']}** points in this archetype. Objectives: "
            + ", ".join(f"**{name} {value:.3f}**" for name, value in scores.items())
            + f"; distance to ideal `{representative['ideal_distance']:.4f}`."
        )
        lines.append("")
        lines.append(
            _markdown_table(
                ["Regime", "Global mixture", "Adaptive mixture"],
                [
                    [label, f"{data['global']:.3f}", f"{data['adaptive']:.3f}"]
                    for label, data in package["regime_mixture"].items()
                ],
            )
        )
        lines.append("")
        lines.append(
            f"Mixture entropy `H = {package['mixture_entropy_H']:.3f}`; effective "
            f"contributing documents `N_eff = {package['effective_contributing_documents']:.1f}`; "
            f"top-3 weight concentration `T3 = {package['top3_weight_concentration_T3']:.3f}`; "
            f"5th-neighbour distance `d5 = {package['fifth_neighbour_distance_d5']:.3f}`; "
            f"relative uncertainty `u = {package['relative_uncertainty_u']:.4f}`."
        )
        lines.append("")
        lines.append("##### Top 5 contributing anchor papers")
        lines.append("")
        lines.append(
            _markdown_table(
                ["Document", "Weight", "Regime", "Year", "Title"],
                [
                    [
                        f"`{anchor['doc']}`",
                        f"{anchor['weight']:.4f}",
                        anchor["regime"],
                        str(anchor["year"]),
                        _escape(anchor["title"]),
                    ]
                    for anchor in package["contributing_documents_top5"]
                ],
            )
        )
        lines.append("")
        lines.append(
            f"**L2 theme lens:** {_format_term_list(package['theme_lens_L2'])}"
        )
        lines.append("")
        lines.append(
            f"**L1 concentration lens:** "
            f"{_format_term_list(package['concentration_lens_L1'])}"
        )
        lines.append("")
        lines.append(
            f"Terms present in **both** lenses (strongest evidence): "
            f"{', '.join(f'`{term}`' for term in package['shared_terms']) or '*(none)*'}"
        )
        lines.append("")

    # --- vocabulary -----------------------------------------------------
    lines.append("## 5. Local vocabulary analysis")
    lines.append("")
    lines.append(
        "The L2 lens marks the local research theme (direction of the l2-normalized "
        "TF-IDF field); the L1 lens marks terms whose probability mass is locally "
        "concentrated (feature allocation). Per Sec. 7.1 the two are graded differently: "
        f"`df >= {EMERGENT_DF}` may be read as an emergent shared topic, while "
        f"`df <= {LIMITED_SOURCE_DF}` may only be a limited-source hypothesis that names "
        "its source documents."
    )
    lines.append("")

    primary = packages[solutions[0]["grid_index"]]
    lines.append(
        f"### 5.1 Lens table for gap #1 ({solutions[0]['x']:+.2f}, {solutions[0]['y']:+.2f})"
    )
    lines.append("")
    lines.append(
        _markdown_table(
            ["Lens", "Term", "Score", "df", "Status"],
            [
                [
                    "L2 theme",
                    f"`{entry['term']}`",
                    f"{entry['weight']:.5f}",
                    str(entry["df"]),
                    "both lenses" if entry["in_both_lenses"] else "theme",
                ]
                for entry in primary["theme_lens_L2"]
            ]
            + [
                [
                    "L1 concentration",
                    f"`{entry['term']}`",
                    f"x{entry['enrichment']:.2f}",
                    str(entry["df"]),
                    "emergent shared topic"
                    if entry["df"] >= EMERGENT_DF
                    else "limited-source hypothesis",
                ]
                for entry in primary["concentration_lens_L1"]
            ],
        )
    )
    lines.append("")

    emergent = [
        entry for entry in primary["concentration_lens_L1"] if entry["df"] >= EMERGENT_DF
    ]
    limited = [
        entry for entry in primary["concentration_lens_L1"]
        if entry["df"] <= LIMITED_SOURCE_DF
    ]

    lines.append(f"### 5.2 Emergent shared topics (df >= {EMERGENT_DF})")
    lines.append("")
    if emergent:
        lines.append(
            _markdown_table(
                ["Term", "df", "Enrichment", "Attested word forms", "In L2 lens"],
                [
                    [
                        f"`{entry['term']}`",
                        str(entry["df"]),
                        f"x{entry['enrichment']:.2f}",
                        ", ".join(f"`{form}`" for form in entry["word_forms"]),
                        "yes" if entry["in_both_lenses"] else "no",
                    ]
                    for entry in emergent
                ],
            )
        )
    else:
        lines.append(
            "_No L1 concentration term reaches this df at gap #1._"
        )
    lines.append("")

    lines.append(f"### 5.3 Limited-source hypotheses (df <= {LIMITED_SOURCE_DF})")
    lines.append("")
    if limited:
        lines.append(
            _markdown_table(
                ["Term", "df", "Enrichment", "Source documents"],
                [
                    [
                        f"`{entry['term']}`",
                        str(entry["df"]),
                        f"x{entry['enrichment']:.2f}",
                        "; ".join(
                            f"`{source['doc']}` (w={source['weight']:.3f})"
                            for source in entry.get("source_docs", [])
                        )
                        or "*(not attested in a contributing document)*",
                    ]
                    for entry in limited
                ],
            )
        )
        lines.append("")
        lines.append(
            "These terms must not be presented as established regional knowledge; "
            "each may only be raised as a hypothesis attributed to the listed documents."
        )
    else:
        lines.append("_None at gap #1._")
    lines.append("")

    # Cross-archetype view: gap #1 sits on a boundary whose L1 terms are all rare,
    # so the df >= 4 category is often populated only at other representatives.
    lines.append("### 5.4 Emergent shared topics across the archetype representatives")
    lines.append("")
    cross_rows: list[list[str]] = []
    for archetype in archetypes[:3]:
        representative = archetype["representative"]
        cross_package = packages[representative["grid_index"]]
        cross_emergent = [
            entry
            for entry in cross_package["concentration_lens_L1"]
            if entry["df"] >= EMERGENT_DF
        ]
        cross_rows.append(
            [
                str(archetype["label"]),
                f"({representative['x']:+.2f}, {representative['y']:+.2f})",
                ", ".join(
                    f"`{entry['term']}` (df={entry['df']})"
                    for entry in cross_emergent
                )
                or "*(none)*",
                str(
                    sum(
                        1
                        for entry in cross_package["concentration_lens_L1"]
                        if entry["df"] <= LIMITED_SOURCE_DF
                    )
                ),
            ]
        )
    lines.append(
        _markdown_table(
            ["Archetype", "Location", "Emergent shared topics (df >= 4)",
             "Limited-source terms"],
            cross_rows,
        )
    )
    lines.append("")

    # --- prompt payload -------------------------------------------------
    lines.append("## 6. Evidence-constrained verbalization prompt")
    lines.append("")
    lines.append(
        "The block below is the **verbatim** Appendix A.1 prompt template from "
        "Obayashi et al. (2026), with only the placeholder file name "
        "`evidence_point_<x>_<y>.json` replaced. The attached evidence package for "
        f"gap #1 — chart ({solutions[0]['x']:+.2f}, {solutions[0]['y']:+.2f}) — follows "
        "the template in the same block, so the whole block can be pasted into an LLM "
        "unchanged."
    )
    lines.append("")
    lines.append("```text")
    lines.extend(prompt_text.splitlines())
    lines.append("```")
    lines.append("")
    lines.append(
        "Output expectations from the template: Task A is a 200–280-word virtual "
        "abstract, Task B an exactly 45-word summary, followed by an `Evidence used:` "
        "list. The audit checklist in Appendix A.2 applies unchanged."
    )
    lines.append("")

    # --- verification ---------------------------------------------------
    lines.append("## 7. Reproduce and verify")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python -m src.harvest --verify-only")
    lines.append(".venv/bin/python -m src.embed   --verify-only")
    lines.append(".venv/bin/python -m src.manifold --verify-only")
    lines.append(".venv/bin/python -m src.pareto  --verify-only")
    lines.append(".venv/bin/python -m src.report   --verify-only")
    lines.append("```")
    lines.append("")
    lines.append("## 8. Caveats")
    lines.append("")
    lines.append(
        "- Novelty is **corpus-relative**. These are gaps in a 120-paper arXiv sample, "
        "not proof that the topics are unexplored in the wider literature."
    )
    lines.append(
        "- Abstracts, not full text, feed the vocabulary and the embeddings. The paper "
        "uses full-text character n-grams, so its exact term statistics are not "
        "reproducible at this scale."
    )
    lines.append(
        "- The reliability diagnostic `R` is reported but, per Sec. 4.3, is deliberately "
        "**not** a Pareto objective: maximizing low uncertainty favours already-dense "
        "regions."
    )
    lines.append(
        "- The sim-to-real query returns only ~140 candidates corpus-wide, so its 40 "
        "papers are the top 40 by relevance with no headroom beyond review rejection."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_inputs(paths: Paths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load corpus, grid archive, and Pareto solutions."""
    with paths.corpus.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)
    with paths.pareto_json.open("r", encoding="utf-8") as handle:
        front = json.load(handle)

    with np.load(paths.grid_npz) as archive:
        grid_data: dict[str, Any] = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    return corpus, grid_data, front


def attach_anchor_evidence(
    front: dict[str, Any],
    grid_data: dict[str, Any],
    papers: list[dict[str, Any]],
) -> None:
    """Add top-3 anchor papers, N_eff, and archetype labels to every solution."""
    archetype_of: dict[int, str] = {}
    for archetype in front["archetypes"]:
        for member in archetype["members"]:
            archetype_of[int(member["grid_index"])] = archetype["label"]

    for solution in front["solutions"]:
        index = int(solution["grid_index"])
        weights = np.asarray(grid_data["weights_global"][index], dtype=np.float64)
        solution["top_anchors"] = contributing_documents(weights, papers, limit=3)
        solution["n_eff"] = round(float(np.asarray(grid_data["n_eff"])[index]), 2)
        solution["archetype"] = archetype_of.get(index, "—")


def write_outputs(
    config: dict[str, Any], paths: Paths, template_path: Path
) -> dict[str, Any]:
    """Generate both output artifacts and return a summary of what was written."""
    corpus, grid_data, front = load_inputs(paths)
    papers = corpus["papers"]
    metadata = front["metadata"]
    objective_names = list(metadata["objective_names"])

    if not template_path.is_file():
        raise SystemExit(
            f"Verbatim Appendix A.1 template not found: {template_path}. "
            "Extract it from docs/papers/2610.25662v1.pdf."
        )
    template = template_path.read_text(encoding="utf-8")

    regime_labels = [
        str(regime.get("label", f"Regime {regime['id']}"))
        for regime in config.get("regimes", [])
    ]
    if not regime_labels:
        regime_labels = [str(label) for label in np.unique(
            [paper["regime_label"] for paper in papers]
        )]

    vocabulary = build_vocabulary(papers)
    attach_anchor_evidence(front, grid_data, papers)

    packages: dict[int, dict[str, Any]] = {}
    needed = {int(solution["grid_index"]) for solution in front["solutions"]}
    for archetype in front["archetypes"][:3]:
        needed.add(int(archetype["representative"]["grid_index"]))
    for grid_index in sorted(needed):
        point = next(
            solution
            for solution in front["solutions"]
            if int(solution["grid_index"]) == grid_index
        )
        package = build_evidence_package(
            point, grid_data, vocabulary, papers, regime_labels
        )
        package["shared_terms"] = [
            entry["term"]
            for entry in package["theme_lens_L2"]
            if entry["in_both_lenses"]
        ]
        packages[grid_index] = package

    primary = front["solutions"][0]
    primary_package = packages[int(primary["grid_index"])]
    evidence_json = json.dumps(
        prompt_evidence_package(primary_package), indent=2, ensure_ascii=False
    )
    filename = f"evidence_point_{primary['x']:+.2f}_{primary['y']:+.2f}.json"
    filled = template.replace("evidence_point_<x>_<y>.json", filename)
    # Do not strip ``filled``: the template's final line ends in a space, and
    # rstrip() would make the reproduced prompt differ from Appendix A.1. The
    # attachment is appended after a blank line so the template body stays exact.
    prompt_text = (
        f"{filled}\n"
        f"--- attached: {filename} ---\n"
        f"{evidence_json}\n"
    )

    figure = build_figure(corpus, grid_data, front, objective_names)
    paths.html.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(paths.html, include_plotlyjs="cdn")
    LOGGER.info("Saved %s (%d bytes)", paths.html, paths.html.stat().st_size)

    markdown = render_markdown(
        corpus, grid_data, front, vocabulary, packages, prompt_text, config
    )
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(markdown + "\n", encoding="utf-8")
    LOGGER.info("Saved %s (%d bytes)", paths.report, paths.report.stat().st_size)

    # Persist the paste-ready prompt on its own for direct handing to an LLM.
    prompt_path = paths.report.with_name("KMO_PROMPT_TOP1.txt")
    prompt_path.write_text(prompt_text, encoding="utf-8")
    LOGGER.info("Saved %s (%d bytes)", prompt_path, prompt_path.stat().st_size)

    return {
        "solutions": len(front["solutions"]),
        "archetypes": len(front["archetypes"]),
        "vocabulary_terms": len(vocabulary.terms),
        "prompt_chars": len(prompt_text),
        "evidence_json_chars": len(evidence_json),
        "prompt_path": str(prompt_path),
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_outputs(
    paths: Paths, template_path: Path, min_html_bytes: int = 50_000
) -> bool:
    """Check that both artifacts exist, are populated, and stay truthful."""
    ok = True
    if not paths.html.is_file():
        LOGGER.error("Missing %s", paths.html)
        return False
    if not paths.report.is_file():
        LOGGER.error("Missing %s", paths.report)
        return False

    html_size = paths.html.stat().st_size
    if html_size < min_html_bytes:
        LOGGER.error("HTML looks empty: %d bytes", html_size)
        ok = False
    html_text = paths.html.read_text(encoding="utf-8", errors="replace")
    for needle in ("Plotly.newPlot", "Practical Pareto gaps", "SysID", "Sim2Real"):
        if needle not in html_text:
            LOGGER.error("HTML is missing expected content: %r", needle)
            ok = False

    markdown = paths.report.read_text(encoding="utf-8")
    sections = [
        "## 1. Knowledge Manifold",
        "## 2. Practical Pareto front",
        "## 3. Discovery archetypes",
        "## 4. Deep dive: top 3 Pareto archetypes",
        "## 5. Local vocabulary analysis",
        "## 6. Evidence-constrained verbalization prompt",
    ]
    for section in sections:
        if section not in markdown:
            LOGGER.error("Report is missing section: %s", section)
            ok = False

    # The prompt must contain the verbatim template, not a paraphrase. The only
    # permitted difference is the substituted evidence-file name, so compare line
    # by line and treat that placeholder as a wildcard.
    template_lines = template_path.read_text(encoding="utf-8").splitlines()
    placeholder = re.compile(r"evidence_point_[+-]?[\d.]+_[+-]?[\d.]+\.json")
    missing: list[str] = []
    for line in template_lines:
        if "evidence_point_<x>_<y>.json" in line:
            pattern = re.escape(line).replace(
                re.escape("evidence_point_<x>_<y>.json"),
                placeholder.pattern,
            )
            if not re.search(pattern, markdown):
                missing.append(line)
        elif line.strip() and line not in markdown:
            missing.append(line)

    if missing:
        LOGGER.error(
            "Report deviates from the verbatim Appendix A.1 template in %d line(s); "
            "first: %r",
            len(missing),
            missing[0],
        )
        ok = False
    else:
        LOGGER.info(
            "Verbatim Appendix A.1 template present (%d lines)", len(template_lines)
        )

    # Every required evidence field must appear inside the pasted prompt block.
    for field in (
        "theme_lens_L2",
        "concentration_lens_L1",
        "contributing_documents_top3",
        "mixture_entropy_H",
        "effective_contributing_documents",
        "relative_uncertainty_u",
    ):
        if field not in markdown:
            LOGGER.error("Evidence package missing field: %s", field)
            ok = False

    table_rows = [
        line for line in markdown.splitlines() if line.startswith("| ")
    ]
    if len(table_rows) < 20:
        LOGGER.error("Report tables look under-populated: %d rows", len(table_rows))
        ok = False

    LOGGER.info(
        "HTML: %d bytes; report: %d bytes, %d table rows, %d chars of prompt",
        html_size,
        len(markdown),
        len(table_rows),
        len(markdown.split("## 6.")[1]) if "## 6." in markdown else 0,
    )
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the ResearchRadar HTML map, Markdown report, and LLM prompt."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--template", type=Path, default=PROMPT_TEMPLATE,
        help="Verbatim Appendix A.1 prompt template",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify the rendered outputs"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths = resolve_paths(config, args.config)
    template_path = (
        args.template
        if args.template.is_absolute()
        else paths.config_dir / args.template
    )

    if args.verify_only:
        return 0 if verify_outputs(paths, template_path) else 1

    for required, producer in (
        (paths.corpus, "src.harvest"),
        (paths.grid_npz, "src.manifold"),
        (paths.pareto_json, "src.pareto"),
    ):
        if not required.is_file():
            LOGGER.error("Missing %s. Run 'python -m %s' first.", required, producer)
            return 1

    summary = write_outputs(config, paths, template_path)
    LOGGER.info(
        "Front: %d gaps in %d archetypes; vocabulary %d terms; prompt %d chars",
        summary["solutions"],
        summary["archetypes"],
        summary["vocabulary_terms"],
        summary["prompt_chars"],
    )
    return 0 if verify_outputs(paths, template_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
