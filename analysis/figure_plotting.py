from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.collections import Collection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text

from experiment_protocol import PROTOCOL


def save_report_figure(
    figure: Figure, paths: list[Path], dpi: int, scale: float,
    layout_rect: tuple[float, float, float, float] = (0, 0, 1, 1),
) -> None:
    """Scale text, line widths and marker diameters at the same physical size."""
    figure.canvas.draw()
    for text in figure.findobj(Text):
        text.set_fontsize(scale * text.get_fontsize())
    for line in figure.findobj(Line2D):
        line.set_linewidth(scale * line.get_linewidth())
        line.set_markersize(scale * line.get_markersize())
        line.set_markeredgewidth(scale * line.get_markeredgewidth())
    for collection in figure.findobj(Collection):
        collection.set_linewidths(scale * collection.get_linewidths())
        if isinstance(collection, PathCollection):
            collection.set_sizes(scale**2 * collection.get_sizes())
    for axis in figure.axes:
        for spine in axis.spines.values():
            spine.set_linewidth(scale * spine.get_linewidth())
    figure.tight_layout(rect=layout_rect)
    figure.canvas.draw()
    figure.tight_layout(rect=layout_rect)
    for path in paths:
        figure.savefig(path, dpi=dpi)
    plt.close(figure)


def plot_pooled_metrics(
    metric_rows: list[dict], output_dir: Path, scale: float,
) -> None:
    models, clicks = PROTOCOL.models, PROTOCOL.click_counts
    labels = {"base_sam": "SAM", "med_sa_adapter": "Med-SA"}
    methods = ("native", "separate", "pooled", "shared_slope")
    colors = {
        "native": "#8a8a85", "separate": "#2a78d6",
        "pooled": "#eb6834", "shared_slope": "#1baf7a",
    }
    markers = {"native": "s", "separate": "o", "pooled": "^", "shared_slope": "D"}
    names = {
        "native": "Native", "separate": "Separate",
        "pooled": "Pooled", "shared_slope": "Shared slope",
    }
    metrics = {
        (row["model"], int(row["click_count"]), row["method"]): row
        for row in metric_rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey="row")
    for column, model in enumerate(models):
        for row, statistic in enumerate(("mae", "bias")):
            axis = axes[row, column]
            for index, method in enumerate(methods):
                x = np.array(clicks) + (index - 1.5) * 0.09
                values = [metrics[model, k, method] for k in clicks]
                estimates = np.array([float(value[statistic]) for value in values])
                errors = np.array([
                    [float(value[statistic]) - float(value[f"{statistic}_ci_low"])
                     for value in values],
                    [float(value[f"{statistic}_ci_high"]) - float(value[statistic])
                     for value in values],
                ])
                axis.errorbar(
                    x, estimates, yerr=errors, color=colors[method],
                    marker=markers[method], markersize=6, linewidth=2,
                    capsize=0, elinewidth=1.2, label=names[method],
                )
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            if statistic == "bias":
                axis.axhline(0, color="black", linewidth=0.8)
                axis.set_xlabel("Number of\nclicks $k$")
        axes[0, column].set_title(labels[model])
    axes[1, 0].set_xticks(clicks)
    axes[0, 0].set_ylabel("MAE")
    axes[0, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("Signed bias\n(score $-$ IoU)")
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", ncol=2,
        frameon=False, fontsize=8,
    )
    save_report_figure(
        figure,
        [output_dir / "mae_by_click.png", output_dir / "mae_by_click.pdf"],
        dpi=300, layout_rect=(0.04, 0, 1, 0.82), scale=scale,
    )
