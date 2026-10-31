"""Build the standalone Knowledge Manifold dashboard.

Renders ``templates/dashboard.html`` into a single self-contained
``output/index.html``: the corpus, the 21x21 discovery fields, and one evidence
package per practical Pareto gap are baked into an inline JSON payload, so the
file opens from ``file://`` with no server, no build step, and no local data
files. Tailwind, Plotly, and Lucide load from CDNs.

The layout follows PLAN.md's reporting goal rather than the reference sketch:
the left panel is the manifold with the four objectives as toggleable overlays,
the right panel is the evidence drawer for whichever gap is selected.

Prompt text is produced by :func:`src.report.render_prompt_text`, the same
function the Markdown report uses, so the prompt shown on screen and the prompt
in ``output/KMO_PROMPT_TOP1.txt`` are byte-identical.

Usage
-----
    python -m src.dashboard
    python -m src.dashboard --verify-only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    LOGGER,
    Paths,
    configure_logging,
    is_seed_paper,
    load_config,
    paper_source_url,
    resolve_paths,
)
from .report import (
    EMERGENT_DF,
    LIMITED_SOURCE_DF,
    PROMPT_TEMPLATE,
    archive_scalar,
    build_gap_packages,
    load_inputs,
    objective_field,
    prompt_evidence_package,
    read_json,
    render_prompt_text,
)

DEFAULT_CONFIG = Path("config.yaml")
PLACEHOLDER = "__RADAR_DATA__"

#: Regime palette. Cyan / amber / emerald keeps the three groups distinguishable
#: against a dark slate field and matches the Markdown report's ordering.
REGIME_COLOURS = {
    "SysID": "#22d3ee",
    "Sim2Real": "#fbbf24",
    "RL": "#34d399",
}

#: Plotly reads ``z[i][j]`` at ``(y=axis[i], x=axis[j])``, so the field matrix is
#: transposed by ``objective_field`` before it reaches the browser.
FIELD_ORDER = ("P_boundary", "W", "G", "S")


# --------------------------------------------------------------------------- #
# Payload assembly
# --------------------------------------------------------------------------- #
def paper_records(
    papers: list[dict[str, Any]], coordinates: np.ndarray
) -> list[dict[str, Any]]:
    """One record per corpus paper, with chart coordinates and an arXiv URL."""
    records: list[dict[str, Any]] = []
    for position, paper in enumerate(papers):
        authors = [str(name) for name in paper.get("authors") or []]
        records.append(
            {
                "id": str(paper["id"]),
                "title": str(paper["title"]),
                "authors": authors[:3],
                "author_count": len(authors),
                "year": int(paper["year"]),
                "regime_id": int(paper["regime_id"]),
                "regime_label": str(paper["regime_label"]),
                "x": round(float(coordinates[position, 0]), 4),
                "y": round(float(coordinates[position, 1]), 4),
                "url": paper_source_url(paper),
                "is_seed": is_seed_paper(paper),
            }
        )
    return records


def field_records(
    grid_data: dict[str, Any], grid_size: int, names: list[str]
) -> list[dict[str, Any]]:
    """The four discovery objectives as browser-ready heatmap matrices."""
    objectives = np.asarray(grid_data["objectives"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    for name in names:
        position = names.index(name)
        matrix = objective_field(objectives, position, grid_size)
        records.append(
            {
                "name": name,
                "z": [[round(float(value), 5) for value in row] for row in matrix],
                "min": round(float(matrix.min()), 4),
                "max": round(float(matrix.max()), 4),
            }
        )
    return records


def gap_records(
    front: dict[str, Any],
    packages: dict[int, dict[str, Any]],
    template: str,
) -> list[dict[str, Any]]:
    """Flatten each gap's evidence package into the shape the page consumes."""
    archetype_of: dict[int, str] = {}
    for archetype in front.get("archetypes", []):
        for member in archetype["members"]:
            archetype_of[int(member["grid_index"])] = str(archetype["label"])

    records: list[dict[str, Any]] = []
    for solution in front["solutions"]:
        index = int(solution["grid_index"])
        package = packages.get(index)
        if package is None:
            raise SystemExit(f"Missing evidence package for grid index {index}")

        x = round(float(solution["x"]), 2)
        y = round(float(solution["y"]), 2)
        filename, prompt_text = render_prompt_text(
            template, x, y, prompt_evidence_package(package)
        )

        concentration = package["concentration_lens_L1"]
        anchors = [
            {
                "doc": anchor["doc"],
                "title": anchor["title"],
                "authors": list(anchor.get("authors") or []),
                "year": int(anchor["year"]),
                "regime": anchor["regime"],
                "weight": float(anchor["weight"]),
                # ``source`` is already an arXiv URL, or empty for a local seed.
                "url": str(anchor.get("source") or ""),
                "is_seed": bool(anchor.get("is_seed", False)),
            }
            for anchor in package["contributing_documents_top5"]
        ]

        records.append(
            {
                "rank": int(solution["rank"]),
                "x": x,
                "y": y,
                "d_ideal": round(float(solution["ideal_distance"]), 4),
                "archetype": archetype_of.get(index, "Unassigned"),
                "on_outer_boundary": bool(solution["on_outer_boundary"]),
                "objectives": {
                    name: round(float(value), 4)
                    for name, value in solution["objectives"].items()
                },
                "n_eff": package["effective_contributing_documents"],
                "d5": package["fifth_neighbour_distance_d5"],
                "t3": package["top3_weight_concentration_T3"],
                "H": package["mixture_entropy_H"],
                "u": package["relative_uncertainty_u"],
                "mixture": package["regime_mixture"],
                "anchors": anchors,
                "theme_lens": [
                    {
                        "term": entry["term"],
                        "weight": entry["weight"],
                        "df": int(entry["df"]),
                        "word_forms": list(entry["word_forms"]),
                        "in_both_lenses": bool(entry["in_both_lenses"]),
                    }
                    for entry in package["theme_lens_L2"]
                ],
                "emergent": [
                    {
                        "term": entry["term"],
                        "df": int(entry["df"]),
                        "enrichment": entry["enrichment"],
                        "word_forms": list(entry["word_forms"]),
                    }
                    for entry in concentration
                    if int(entry["df"]) >= EMERGENT_DF
                ],
                "limited": [
                    {
                        "term": entry["term"],
                        "df": int(entry["df"]),
                        "enrichment": entry["enrichment"],
                        "source_docs": [
                            {
                                "doc": source["doc"],
                                "title": source["title"],
                                "weight": float(source["weight"]),
                                "url": str(source.get("source") or ""),
                                "is_seed": bool(source.get("is_seed", False)),
                            }
                            for source in entry.get("source_docs", [])
                        ],
                    }
                    for entry in concentration
                    if int(entry["df"]) <= LIMITED_SOURCE_DF
                ],
                "filename": filename,
                "prompt": prompt_text,
            }
        )
    records.sort(key=lambda record: record["rank"])
    return records


def build_payload(
    config: dict[str, Any],
    corpus: dict[str, Any],
    grid_data: dict[str, Any],
    front: dict[str, Any],
    template: str,
    paths: Paths,
) -> dict[str, Any]:
    """Assemble the full JSON bundle embedded in the page."""
    metadata = front["metadata"]
    names = [str(name) for name in metadata["objective_names"]]
    grid_size = round(int(metadata["candidates"]) ** 0.5)

    _, packages = build_gap_packages(corpus, grid_data, front, config, paths)
    gaps = gap_records(front, packages, template)

    axis = np.round(
        np.linspace(
            float(np.asarray(grid_data["grid"])[:, 0].min()),
            float(np.asarray(grid_data["grid"])[:, 0].max()),
            grid_size,
        ),
        4,
    )

    regimes = []
    for regime in config.get("regimes", []):
        label = str(regime.get("label", f"Regime {regime['id']}"))
        regimes.append(
            {
                "id": int(regime["id"]),
                "label": label,
                "name": str(regime.get("name", label)),
                "color": REGIME_COLOURS.get(label, "#94a3b8"),
            }
        )
    if not regimes:
        labels = sorted({str(paper["regime_label"]) for paper in corpus["papers"]})
        regimes = [
            {
                "id": position,
                "label": label,
                "name": label,
                "color": REGIME_COLOURS.get(label, "#94a3b8"),
            }
            for position, label in enumerate(labels)
        ]

    return {
        "meta": {
            "papers": len(corpus["papers"]),
            "gaps": len(gaps),
            "archetypes": len(front.get("archetypes", [])),
            "non_dominated": int(metadata.get("non_dominated", 0)),
            "stress": round(archive_scalar(grid_data, "stress"), 4),
            "spearman": round(archive_scalar(grid_data, "distance_correlation"), 4),
            "representation": "char 4-7-gram TF-IDF",
            "seeds": int(corpus["metadata"].get("n_seeds", 0)),
            "grid_size": grid_size,
            "built_from": str(PLACEHOLDER).replace(PLACEHOLDER, "src.dashboard"),
        },
        "regimes": regimes,
        "papers": paper_records(
            corpus["papers"], np.asarray(grid_data["doc_coords"], dtype=np.float64)
        ),
        "axis": [float(value) for value in axis],
        "fields": field_records(grid_data, grid_size, names),
        "field_order": list(FIELD_ORDER),
        "gaps": gaps,
    }


def encode_payload(payload: dict[str, Any]) -> str:
    """Serialize the payload so it cannot terminate the surrounding script tag.

    ``<``, ``>``, and ``&`` become ``\\u003c``-style escapes, which keeps the JSON
    valid while making ``</script>`` (or a stray ``<!--``) inside a paper title
    impossible. Line/paragraph separators are escaped too because they are legal
    inside JSON strings but break some JavaScript parsers.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    for raw, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(raw, escaped)
    return text


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_dashboard(
    config: dict[str, Any], paths: Paths, template_path: Path
) -> dict[str, Any]:
    """Render ``output/index.html`` and return a summary of the result."""
    if not template_path.is_file():
        raise SystemExit(f"Dashboard template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(
            f"Dashboard template {template_path} is missing the {PLACEHOLDER} slot"
        )

    prompt_template_path = paths.config_dir / PROMPT_TEMPLATE
    if not prompt_template_path.is_file():
        raise SystemExit(
            f"Verbatim Appendix A.1 template not found: {prompt_template_path}"
        )
    prompt_template = prompt_template_path.read_text(encoding="utf-8")

    corpus, grid_data, front = load_inputs(paths)
    payload = build_payload(config, corpus, grid_data, front, prompt_template, paths)

    document = template.replace(PLACEHOLDER, encode_payload(payload))
    paths.dashboard.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.dashboard.with_suffix(paths.dashboard.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(paths.dashboard)

    LOGGER.info(
        "Saved %s (%d bytes): %d papers, %d gaps, %d archetypes, %d field layers",
        paths.dashboard,
        paths.dashboard.stat().st_size,
        payload["meta"]["papers"],
        payload["meta"]["gaps"],
        payload["meta"]["archetypes"],
        len(payload["fields"]),
    )
    return {
        "bytes": paths.dashboard.stat().st_size,
        "papers": payload["meta"]["papers"],
        "gaps": payload["meta"]["gaps"],
        "payload_chars": len(encode_payload(payload)),
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
_PAYLOAD_RE = re.compile(
    r'<script id="radar-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def extract_payload(document: str) -> dict[str, Any]:
    """Pull the embedded JSON bundle back out of the rendered page."""
    match = _PAYLOAD_RE.search(document)
    if not match:
        raise SystemExit("Rendered page has no embedded radar-data payload")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Embedded payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Embedded payload must be a JSON object")
    return payload


def verify_dashboard(paths: Paths, min_bytes: int = 150_000) -> bool:
    """Check the page is self-contained, populated, and agrees with the artifacts."""
    if not paths.dashboard.is_file():
        LOGGER.error("Missing %s", paths.dashboard)
        return False

    document = paths.dashboard.read_text(encoding="utf-8", errors="replace")
    size = len(document)
    ok = True

    if size < min_bytes:
        LOGGER.error("Dashboard looks empty: %d bytes", size)
        ok = False
    if PLACEHOLDER in document:
        LOGGER.error("Dashboard still contains the %s placeholder", PLACEHOLDER)
        ok = False

    # Self-contained means CDN assets only: no local script/style/img sources and
    # no relative fetches that would 404 when the file is opened directly.
    # Tailwind is deliberately absent: cdn.tailwindcss.com sends no
    # Access-Control-Allow-Origin header, so an SRI-protected (crossorigin)
    # script tag is blocked outright on any http origin. All layout lives in the
    # inline stylesheet instead, and these three CDNs do send CORS headers.
    for asset in (
        "cdn.plot.ly/plotly-",
        "unpkg.com/lucide@",
        "fonts.googleapis.com",
    ):
        if asset not in document:
            LOGGER.error("Missing expected CDN asset: %s", asset)
            ok = False
    for pattern, description in (
        (r'<script[^>]+src="(?!https?://)', "local script source"),
        (r'<link[^>]+href="(?!https://)', "local stylesheet"),
        (r"fetch\(", "runtime fetch()"),
        (r"XMLHttpRequest", "runtime XHR"),
    ):
        if re.search(pattern, document):
            LOGGER.error("Dashboard is not offline-safe: found %s", description)
            ok = False

    payload = extract_payload(document)

    # Counts must match the artifacts on disk.
    front = read_json(paths.pareto_json)
    with np.load(paths.grid_npz) as archive:
        coordinates = np.asarray(archive["doc_coords"], dtype=np.float64)
        occupancy = np.asarray(archive["objectives"]).shape[0]

    expected_papers = len(coordinates)
    if len(payload["papers"]) != expected_papers:
        LOGGER.error(
            "Payload has %d papers, expected %d",
            len(payload["papers"]),
            expected_papers,
        )
        ok = False
    expected_gaps = len(front["solutions"])
    if len(payload["gaps"]) != expected_gaps:
        LOGGER.error(
            "Payload has %d gaps, expected %d", len(payload["gaps"]), expected_gaps
        )
        ok = False
    grid_size = round(occupancy**0.5)
    if any(len(field["z"]) != grid_size for field in payload["fields"]):
        LOGGER.error("A field matrix is not %d x %d", grid_size, grid_size)
        ok = False
    if [field["name"] for field in payload["fields"]] != list(FIELD_ORDER):
        LOGGER.error(
            "Field layers are %s, expected %s",
            [field["name"] for field in payload["fields"]],
            list(FIELD_ORDER),
        )
        ok = False

    # Coordinates must round-trip: the chart is the whole point of the page.
    drift = max(
        abs(record["x"] - float(coordinates[position, 0]))
        for position, record in enumerate(payload["papers"])
    )
    if drift > 1e-3:
        LOGGER.error("Paper coordinates drifted by %.2e", drift)
        ok = False

    # The field matrix must orient x and y the same way the paper markers do.
    # Plotly reads z[i][j] at (y=axis[i], x=axis[j]), so a transposed matrix would
    # put the field behind the papers instead of under them. Index 3/17 is
    # (x=+0.7, y=-0.7) on a 21-point axis, where this corpus's P_boundary is 0
    # while (x=-0.7, y=+0.7) is 0.62 - a cheap asymmetry to assert on.
    axis = np.asarray(payload["axis"], dtype=np.float64)
    with np.load(paths.grid_npz) as archive:
        raw = np.asarray(archive["objectives"], dtype=np.float64)
    field = np.asarray(payload["fields"][0]["z"], dtype=np.float64)
    expected = float(raw[:, 0].reshape(grid_size, grid_size)[3, 17])
    actual = float(field[17][3])
    if abs(actual - expected) > 1e-4:
        LOGGER.error(
            "Field matrix appears transposed: z[y=+0.7][x=-0.7]=%.4f, expected %.4f",
            actual,
            expected,
        )
        ok = False
    if abs(axis[0] + 1.0) > 1e-6 or abs(axis[-1] - 1.0) > 1e-6:
        LOGGER.error("Axis does not span [-1, 1]: %s..%s", axis[0], axis[-1])
        ok = False

    # Every gap needs a complete, self-consistent evidence package.
    ranks = [gap["rank"] for gap in payload["gaps"]]
    if ranks != sorted(ranks) or ranks[0] != 1:
        LOGGER.error("Gap ranks are not 1..n in order: %s", ranks[:5])
        ok = False
    for gap in payload["gaps"]:
        if (
            not gap["prompt"]
            or "Appendix" not in gap["prompt"]
            and "Rules (binding)" not in gap["prompt"]
        ):
            LOGGER.error("Gap #%d has no prompt", gap["rank"])
            ok = False
            break
        if len(gap["anchors"]) != 5:
            LOGGER.error(
                "Gap #%d has %d anchors, expected 5", gap["rank"], len(gap["anchors"])
            )
            ok = False
        if not gap["theme_lens"]:
            LOGGER.error("Gap #%d has an empty theme lens", gap["rank"])
            ok = False
        if gap["filename"] not in gap["prompt"]:
            LOGGER.error(
                "Gap #%d prompt does not name %s", gap["rank"], gap["filename"]
            )
            ok = False
        if not 0.0 <= gap["objectives"]["W"] <= 1.0:
            LOGGER.error("Gap #%d has W out of range", gap["rank"])
            ok = False

    # The #1 prompt must match the standalone artifact byte for byte.
    standalone = paths.report.with_name("KMO_PROMPT_TOP1.txt")
    if standalone.is_file():
        if standalone.read_text(encoding="utf-8") != payload["gaps"][0]["prompt"]:
            LOGGER.error("Dashboard #1 prompt differs from %s", standalone)
            ok = False
        else:
            LOGGER.info("Dashboard #1 prompt matches %s exactly", standalone.name)
    else:
        LOGGER.warning(
            "Missing %s; skipped the cross-artifact prompt check", standalone
        )

    top = payload["gaps"][0]
    LOGGER.info(
        "Dashboard: %.1f KB, %d papers, %d gaps, %d archetypes, %d field layers, "
        "coordinate drift %.1e, #1 at (%+.2f, %+.2f)",
        size / 1024,
        len(payload["papers"]),
        len(payload["gaps"]),
        payload["meta"]["archetypes"],
        len(payload["fields"]),
        drift,
        top["x"],
        top["y"],
    )
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the standalone ResearchRadar dashboard to output/index.html."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify the rendered dashboard"
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Dashboard template (defaults to output.dashboard_template)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    paths = resolve_paths(config, args.config)
    template_path = (
        args.template if args.template is not None else paths.dashboard_template
    )
    if not template_path.is_absolute() and args.template is not None:
        template_path = paths.config_dir / template_path

    if args.verify_only:
        return 0 if verify_dashboard(paths) else 1

    for required, producer in (
        (paths.corpus, "src.harvest"),
        (paths.embeddings, "src.embed"),
        (paths.grid_npz, "src.manifold"),
        (paths.pareto_json, "src.pareto"),
    ):
        if not required.is_file():
            LOGGER.error("Missing %s. Run 'python -m %s' first.", required, producer)
            return 1

    render_dashboard(config, paths, template_path)
    return 0 if verify_dashboard(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
