from __future__ import annotations

import json
from pathlib import Path
from textwrap import wrap


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "docs" / "submission" / "static_vs_adaptive_divergence.svg"

RUNS = [
    {
        "label": "Llama 3.1 8B",
        "path": REPO_ROOT / "data" / "evals" / "llama31_balanced_interactive100" / "aggregate_summary.json",
        "color": "#d1495b",
    },
    {
        "label": "Qwen 2.5 7B",
        "path": REPO_ROOT / "data" / "evals" / "qwen25_balanced_interactive100" / "aggregate_summary.json",
        "color": "#00798c",
    },
    {
        "label": "Mistral NeMo 12B",
        "path": REPO_ROOT / "data" / "evals" / "mistralnemo_balanced_interactive100" / "aggregate_summary.json",
        "color": "#edae49",
    },
]


def load_points():
    points = []
    for run in RUNS:
        data = json.loads(run["path"].read_text())
        points.append(
            {
                "label": run["label"],
                "color": run["color"],
                "static_accuracy": float(data["static_summary"]["accuracy"]),
                "belief_trajectory_quality": float(
                    data["interactive_summary"]["belief_trajectory_quality"]
                ),
                "source": str(run["path"].relative_to(REPO_ROOT)),
            }
        )
    return points


def map_x(value: float, plot_x: int, plot_w: int) -> float:
    return plot_x + value * plot_w


def map_y(value: float, plot_y: int, plot_h: int) -> float:
    return plot_y + plot_h - value * plot_h


def add_multiline_text(
    parts: list[str],
    *,
    x: float,
    y: float,
    lines: list[str],
    font_size: int,
    fill: str,
    font_weight: str | None = None,
    line_height: int = 18,
) -> None:
    weight = f' font-weight="{font_weight}"' if font_weight else ""
    parts.append(
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{font_size}"{weight} fill="{fill}">'
    )
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        parts.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    parts.append("</text>")


def build_svg(points):
    width = 980
    height = 680
    plot_x = 120
    plot_y = 120
    plot_w = 720
    plot_h = 380

    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        '<text x="120" y="52" font-family="Helvetica, Arial, sans-serif" font-size="28" font-weight="700" fill="#1f2937">AGUS Learning Core: Static Accuracy vs Adaptive Quality</text>',
        '<text x="120" y="82" font-family="Helvetica, Arial, sans-serif" font-size="16" fill="#4b5563">Existing local Learning Core artifacts only. X = static accuracy, Y = belief_trajectory_quality.</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#fffdf8" stroke="#d6d3d1" stroke-width="1.5"/>',
    ]

    for tick in x_ticks:
        x = map_x(tick, plot_x, plot_w)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#ece7df" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_h + 26}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#6b7280">{tick:.1f}</text>'
        )

    for tick in y_ticks:
        y = map_y(tick, plot_y, plot_h)
        parts.append(
            f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#ece7df" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{plot_x - 18}" y="{y + 4:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#6b7280">{tick:.1f}</text>'
        )

    parts.extend(
        [
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#6b7280" stroke-width="1.5"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#6b7280" stroke-width="1.5"/>',
            f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 56}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#374151">Static accuracy</text>',
            f'<text x="42" y="{plot_y + plot_h / 2}" transform="rotate(-90 42 {plot_y + plot_h / 2})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="600" fill="#374151">Belief trajectory quality</text>',
        ]
    )

    parts.append(
        f'<line x1="{plot_x}" y1="{map_y(0.6, plot_y, plot_h):.1f}" x2="{plot_x + plot_w}" y2="{map_y(0.6, plot_y, plot_h):.1f}" stroke="#cbd5e1" stroke-dasharray="6,6" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{plot_x + plot_w + 10}" y="{map_y(0.6, plot_y, plot_h) + 4:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#64748b">BTQ = 0.6 reference</text>'
    )

    label_offsets = {
        "Llama 3.1 8B": (18, -12),
        "Qwen 2.5 7B": (18, -12),
        "Mistral NeMo 12B": (18, 22),
    }

    for point in points:
        cx = map_x(point["static_accuracy"], plot_x, plot_w)
        cy = map_y(point["belief_trajectory_quality"], plot_y, plot_h)
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{point["color"]}" stroke="#111827" stroke-width="1.5"/>'
        )
        dx, dy = label_offsets[point["label"]]
        parts.append(
            f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="#111827">{point["label"]}</text>'
        )
        parts.append(
            f'<text x="{cx + dx:.1f}" y="{cy + dy + 18:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">accuracy {point["static_accuracy"]:.4f}, BTQ {point["belief_trajectory_quality"]:.4f}</text>'
        )

    add_multiline_text(
        parts,
        x=120,
        y=548,
        lines=["Interpretation"],
        font_size=15,
        font_weight="700",
        fill="#111827",
    )
    interpretation = (
        "Llama is strongest on frozen-task correctness, Qwen is strongest on adaptive "
        "trajectory quality, and Mistral lands between them on dynamic quality despite "
        "weak static accuracy."
    )
    add_multiline_text(
        parts,
        x=120,
        y=574,
        lines=wrap(interpretation, width=92),
        font_size=14,
        fill="#374151",
        line_height=20,
    )
    source_lines = [
        "Sources:",
        "data/evals/llama31_balanced_interactive100/aggregate_summary.json",
        "data/evals/qwen25_balanced_interactive100/aggregate_summary.json",
        "data/evals/mistralnemo_balanced_interactive100/aggregate_summary.json",
    ]
    add_multiline_text(
        parts,
        x=120,
        y=628,
        lines=source_lines,
        font_size=12,
        fill="#6b7280",
        line_height=16,
    )
    parts.append("</svg>")

    return "\n".join(parts)


def main():
    points = load_points()
    OUTPUT_PATH.write_text(build_svg(points), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
