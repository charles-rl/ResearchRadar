"""Shared helpers for the ResearchRadar pipeline stages.

Keeps config loading, path resolution, and logging setup in one place so
``harvest``/``embed``/``manifold``/``pareto``/``report`` agree on where artifacts
live and how they are named.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("research_radar")


@dataclass(frozen=True)
class Paths:
    """Resolved artifact paths for one config file."""

    config_dir: Path
    corpus: Path
    corpus_fulltext: Path
    tfidf_matrix: Path
    tfidf_vocab: Path
    embeddings: Path
    grid_npz: Path
    pareto_json: Path
    html: Path
    report: Path
    dashboard: Path
    dashboard_template: Path
    prompt_payload: Path
    prompt_template: Path


def configure_logging(level: str = "INFO") -> None:
    """Install a consistent stdout log format for pipeline stages."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(getattr(logging, str(level).upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load ``config.yaml`` and fail with a clear message on bad input."""
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"Config file is not valid YAML ({config_path}): {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Config file must contain a mapping: {config_path}")
    return config


def resolve_paths(config: dict[str, Any], config_path: Path) -> Paths:
    """Resolve every artifact path relative to the config file's directory."""
    config_dir = config_path.parent if str(config_path.parent) else Path(".")
    corpus_cfg = config.get("corpus", {})
    fulltext_cfg = config.get("fulltext", {})
    tfidf_cfg = config.get("tfidf", {})
    ollama_cfg = config.get("ollama", {})
    manifold_cfg = config.get("manifold", {})
    pareto_cfg = config.get("pareto", {})
    output_cfg = config.get("output", {})

    return Paths(
        config_dir=config_dir,
        corpus=config_dir / str(corpus_cfg.get("output_path", "data/corpus.json")),
        corpus_fulltext=config_dir
        / str(fulltext_cfg.get("output_path", "data/corpus_fulltext.json")),
        tfidf_matrix=config_dir
        / str(tfidf_cfg.get("matrix_path", "data/sparse_tfidf.npz")),
        tfidf_vocab=config_dir
        / str(tfidf_cfg.get("vocab_path", "data/tfidf_vocab.json")),
        embeddings=config_dir
        / str(ollama_cfg.get("embeddings_cache", "data/embeddings.npy")),
        grid_npz=config_dir
        / str(manifold_cfg.get("grid_output", "data/manifold_grid.npz")),
        pareto_json=config_dir
        / str(pareto_cfg.get("output_path", "data/pareto_solutions.json")),
        html=config_dir / str(output_cfg.get("html_path", "output/radar_map.html")),
        report=config_dir
        / str(output_cfg.get("report_path", "output/RADAR_REPORT.md")),
        dashboard=config_dir
        / str(output_cfg.get("dashboard_path", "output/index.html")),
        dashboard_template=config_dir
        / str(output_cfg.get("dashboard_template", "templates/dashboard.html")),
        prompt_payload=config_dir
        / str(output_cfg.get("prompt_path", "output/KMO_PROMPT_TOP1.txt")),
        prompt_template=config_dir
        / str(output_cfg.get("prompt_template", "docs/kmo_prompt_template_A1.txt")),
    )


def paper_source_url(paper: dict[str, Any]) -> str:
    """Public link for one corpus paper, or an empty string when none exists.

    Harvested papers are arXiv entries and always resolve.  A seeding paper is a
    local PDF: it carries an arXiv identifier only when its filename did, and
    free-form filenames (``TD JEPA.pdf``) are given a ``seed:`` identifier that
    no public service knows.  Returning an empty string lets each renderer fall
    back to plain text instead of emitting a dead link.
    """
    identifier = str(paper.get("id", ""))
    if not identifier or identifier.startswith("seed:"):
        return ""
    return f"https://arxiv.org/abs/{identifier}"


def is_seed_paper(paper: dict[str, Any]) -> bool:
    """True for the seeding papers that founded the corpus."""
    return bool(paper.get("is_seed", False))


def paper_provenance(paper: dict[str, Any]) -> str:
    """Short human label for where a paper came from."""
    return "seed" if is_seed_paper(paper) else "harvested"


#: Categorical colours for regimes, assigned by position in the regime list.
#: Nothing downstream may hardcode a regime label: the taxonomy lives in
#: ``config.yaml`` and a taxonomy change must not require a code edit.
REGIME_PALETTE = (
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
)


def corpus_regime_labels(papers: list[dict[str, Any]]) -> list[str]:
    """Regime labels in ``regime_id`` order, as they appear in the corpus."""
    ordered: dict[int, str] = {}
    for paper in papers:
        identifier = int(paper.get("regime_id", 0))
        ordered.setdefault(identifier, str(paper.get("regime_label", "")))
    return [ordered[key] for key in sorted(ordered) if ordered[key]]


def regime_palette(labels: Sequence[str]) -> dict[str, str]:
    """Map each regime label to a stable colour, wrapping if there are more."""
    return {
        label: REGIME_PALETTE[position % len(REGIME_PALETTE)]
        for position, label in enumerate(labels)
    }
