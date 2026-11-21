"""Thread-safe chart rendering.

The old code used ``matplotlib.pyplot`` (global state) inside FastAPI's thread
pool, so two simultaneous requests could draw on or close each other's
figures. Here every chart gets its own ``Figure`` object and no pyplot state
is touched.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RISK_UP = "#e11d48"  # increases mortality risk
RISK_DOWN = "#0f9f6e"  # decreases mortality risk
TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def _to_base64(fig: Figure) -> str:
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", facecolor="white")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def contribution_chart(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    value_fmt: str = "{:+.3f}",
) -> str:
    """Horizontal bar chart of signed contributions, largest at the top."""
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    height = max(2.6, 0.42 * len(values) + 1.2)
    fig = Figure(figsize=(7.2, height))
    ax = fig.add_subplot(111)

    colors = [RISK_UP if v > 0 else RISK_DOWN for v in values]
    bars = ax.barh(range(len(values)), values, color=colors, height=0.62)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=10, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_title(title, fontsize=12, color=TEXT, loc="left", pad=12, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)

    span = max((abs(v) for v in values), default=0) or 1e-6
    ax.set_xlim(-span * 1.35, span * 1.35)
    for bar, v in zip(bars, values, strict=False):
        offset = span * 0.03
        x = v + offset if v >= 0 else v - offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(v),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
            color=TEXT,
        )

    fig.legend(
        handles=[
            Patch(color=RISK_UP, label="increases risk"),
            Patch(color=RISK_DOWN, label="decreases risk"),
        ],
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(1.0, -0.02),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _to_base64(fig)
