"""Shared helpers for the ResearchRadar pipeline stages.

Keeps config loading, path resolution, and logging setup in one place so
``harvest``/``embed``/``manifold``/``pareto``/``report`` agree on where artifacts
live and how they are named.
"""

from __future__ import annotations

import copy
import logging
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("research_radar")

#: Seed-First Expansion: hand-picked PDFs anchor the corpus and a per-regime
#: harvest expands it.  This is the default and the mode the committed
#: artifacts were built in.
PROFILE_SEED_FIRST = "seed_first"

#: Single-Field Exploration (Zero-Seed Mode): one broad query, no seeding PDFs,
#: and the regimes are discovered from the chart itself.
PROFILE_SINGLE_FIELD = "single_field"

PROFILES = (PROFILE_SEED_FIRST, PROFILE_SINGLE_FIELD)

#: A single-field corpus is partitioned at Discovery K=5 and Reference K=8, so
#: it needs enough documents for both partitions to mean anything.  Twelve per
#: Reference cluster is the floor below which a boundary measure is just noise.
MIN_SINGLE_FIELD_PAPERS = 3 * 8


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


#: Repository root, used to find source assets (templates, the Appendix A.1
#: prompt file) when a config lives outside the checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_asset_path(config_dir: Path, value: Any, default: str) -> Path:
    """Locate a source asset such as a template, preferring the config's tree.

    Artifacts are anchored strictly at the config file, but templates are part
    of the checkout, not of a run.  A config kept outside the repository (two
    profiles side by side, or a dry run) must still find
    ``templates/dashboard.html``, so a relative asset falls back to the
    repository root when the config's own tree does not have it.  The
    config-relative path is returned when neither exists, so the error message
    names the place the caller would have expected.

    A path that is already absolute but missing (``resolve_paths`` hands out
    absolute paths) is retried as a repository-relative one, which is what makes
    this work for callers that resolve first and ask questions later.
    """
    path = Path(str(value if value is not None else default)).expanduser()
    if path.is_absolute():
        candidates = [path]
        # ``resolve_paths`` hands out absolute paths, so retry the same relative
        # tail inside the checkout (``templates/dashboard.html``).
        with suppress(ValueError):
            candidates.append(REPO_ROOT / path.relative_to(Path(config_dir)))
    else:
        candidates = [Path(config_dir) / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
    """Load ``config.yaml``, resolve the profile, and fail with a clear message."""
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
    return resolve_profile_config(config, source=config_path)


def active_profile(config: dict[str, Any]) -> str:
    """Which pipeline profile the config selects (``seed_first`` by default)."""
    raw = config.get("profile", PROFILE_SEED_FIRST)
    profile = str(raw).strip().lower()
    if profile not in PROFILES:
        raise SystemExit(
            f"Unknown profile {raw!r}; expected one of {', '.join(PROFILES)}"
        )
    return profile


def is_single_field(config: dict[str, Any]) -> bool:
    """True when the corpus is harvested from one broad query, with no seeds."""
    return active_profile(config) == PROFILE_SINGLE_FIELD


def resolve_profile_config(
    config: dict[str, Any], source: Path | None = None
) -> dict[str, Any]:
    """Return the config every stage should see, with the profile applied.

    Two modes share one pipeline, so rather than teaching each stage about both
    profiles, single-field mode is normalised into the shape the stages already
    understand: one regime, zero seeds, and a fixed lookback.  Downstream code
    (``harvest``, ``embed``, ``manifold``, ``pareto``, ``report``, ``dashboard``)
    therefore needs no branch of its own, and a bug cannot lurk in only one arm.

    The regimes in this mode are *discovered* from the chart by ``src.regimes``
    rather than supplied here, so the single placeholder regime below is only
    the harvest query's carrier; ``src.manifold`` overwrites the per-paper
    labels before any objective is computed.
    """
    profile = active_profile(config)
    if profile == PROFILE_SEED_FIRST:
        return config

    single = config.get("single_field")
    where = f" in {source}" if source else ""
    if not isinstance(single, dict):
        raise SystemExit(
            "profile: single_field requires a 'single_field:' block" + where
        )

    query = str(single.get("query", "")).strip()
    if not query:
        raise SystemExit("single_field.query must be a non-empty arXiv query" + where)

    target = int(single.get("target_total_papers", 0))
    if target < MIN_SINGLE_FIELD_PAPERS:
        raise SystemExit(
            f"single_field.target_total_papers must be at least "
            f"{MIN_SINGLE_FIELD_PAPERS} (Discovery K={int(single.get('discovery_k', 5))} "
            f"and Reference K={int(single.get('reference_k', 8))} both need "
            f"documents to partition){where}"
        )

    years_back = int(single.get("years_back", 5))
    if years_back < 1:
        raise SystemExit(f"single_field.years_back must be positive{where}")

    label = str(single.get("label", "Field")) or "Field"
    resolved = copy.deepcopy(config)
    resolved["profile"] = PROFILE_SINGLE_FIELD

    corpus_cfg = dict(resolved.get("corpus", {}) or {})
    corpus_cfg["seed_count"] = 0
    corpus_cfg["target_total_papers"] = target
    # One query: every paper belongs to it, so the per-regime quota is the whole
    # corpus.  verify_corpus then checks "harvested == target" with no special case.
    corpus_cfg["target_per_regime"] = target
    corpus_cfg["lookback_years"] = years_back
    # No seeds means no seed years, so the offset never applies; set it to 0 so a
    # reader of the config cannot mistake it for an active setting.
    corpus_cfg["seed_year_offset"] = 0
    corpus_cfg["seed_overrides"] = {}
    resolved["corpus"] = corpus_cfg

    resolved["regimes"] = [
        {
            "id": 0,
            "name": str(single.get("name", label)),
            "label": label,
            "description": str(single.get("description", query)),
            "query": query,
            "keywords": [str(word) for word in single.get("keywords", []) or []],
        }
    ]
    return resolved


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


def single_field_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The ``single_field:`` block, or an empty mapping in seed-first mode."""
    block = config.get("single_field")
    return block if isinstance(block, dict) else {}


def config_int(settings: dict[str, Any], key: str, default: int) -> int:
    """Read an integer that may arrive as a string from YAML."""
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def corpus_regime_labels(papers: list[dict[str, Any]]) -> list[str]:
    """Regime labels in ``regime_id`` order, as they appear in the corpus.

    Single-field mode discovers its regimes from the chart, so the corpus, not
    ``config.yaml``, is the authority on what the labels are.  Reading them from
    the papers keeps the legend, the colour assignment, and the per-gap mixture
    table in one place: they cannot disagree, and a taxonomy change cannot leave
    a renderer enumerating a regime the corpus no longer contains.
    """
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
