"""Shared helpers for the ResearchRadar pipeline stages.

Keeps config loading, path resolution, and logging setup in one place so
``harvest``/``embed``/``manifold``/``pareto``/``report`` agree on where artifacts
live and how they are named.
"""

from __future__ import annotations

import logging
import sys
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
    embeddings: Path
    grid_npz: Path
    pareto_json: Path
    html: Path
    report: Path
    dashboard: Path
    dashboard_template: Path


def configure_logging(level: str = "INFO") -> None:
    """Install a consistent stdout log format for pipeline stages."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(
            getattr(logging, str(level).upper(), logging.INFO)
        )
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
        raise SystemExit(f"Config file is not valid YAML ({config_path}): {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Config file must contain a mapping: {config_path}")
    return config


def resolve_paths(config: dict[str, Any], config_path: Path) -> Paths:
    """Resolve every artifact path relative to the config file's directory."""
    config_dir = config_path.parent if str(config_path.parent) else Path(".")
    corpus_cfg = config.get("corpus", {})
    ollama_cfg = config.get("ollama", {})
    manifold_cfg = config.get("manifold", {})
    pareto_cfg = config.get("pareto", {})
    output_cfg = config.get("output", {})

    return Paths(
        config_dir=config_dir,
        corpus=config_dir / str(corpus_cfg.get("output_path", "data/corpus.json")),
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
        / str(
            output_cfg.get("dashboard_template", "templates/dashboard.html")
        ),
    )
