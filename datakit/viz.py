"""Charts.

Every function saves an image and returns its path. Colors come from one
validated palette (see ``PALETTE``): categorical hues are assigned in a fixed
order and never cycled, magnitude uses a single-hue ramp, and polarity
(correlation) uses a blue-to-red diverging ramp with a neutral midpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # render to files, never to a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "PALETTE", "apply_theme", "histogram", "bar", "line", "scatter", "box",
    "correlation_heatmap", "missing_matrix", "auto_charts",
]

#: Fixed categorical order plus the ink/surface tokens, per mode.
PALETTE: dict[str, dict[str, Any]] = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                       "#256abf", "#184f95", "#0d366b"],
        "diverging": ["#184f95", "#6da7ec", "#cde2fb", "#f0efec",
                      "#f0a3a3", "#e34948", "#a32b2a"],
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                       "#256abf", "#184f95", "#0d366b"],
        "diverging": ["#184f95", "#6da7ec", "#cde2fb", "#383835",
                      "#f0a3a3", "#e34948", "#a32b2a"],
    },
}

#: Slots that stay distinguishable when every pair can end up side by side
#: (scatter, bubble). Beyond this, extra categories fold into "Other".
_ALL_PAIRS_LIMIT = 3
#: Slots for forms where only neighbours touch (bars, lines, stacks).
_ADJACENT_LIMIT = 8


def theme(dark: bool = False) -> dict[str, Any]:
    """The token set for one mode."""
    return PALETTE["dark" if dark else "light"]


def apply_theme(dark: bool = False) -> dict[str, Any]:
    """Set matplotlib defaults: recessive grid and axes, ink-colored text."""
    tokens = theme(dark)
    plt.rcParams.update({
        "figure.facecolor": tokens["surface"],
        "axes.facecolor": tokens["surface"],
        "savefig.facecolor": tokens["surface"],
        "axes.edgecolor": tokens["axis"],
        "axes.labelcolor": tokens["secondary"],
        "axes.titlecolor": tokens["primary"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": tokens["grid"],
        "grid.linewidth": 0.8,
        "text.color": tokens["primary"],
        "xtick.color": tokens["muted"],
        "ytick.color": tokens["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": ["DejaVu Sans"],
        "figure.dpi": 110,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    })
    return tokens


def sequential_cmap(dark: bool = False) -> LinearSegmentedColormap:
    """One-hue light-to-dark ramp for magnitude."""
    return LinearSegmentedColormap.from_list("dk_seq", theme(dark)["sequential"])


def diverging_cmap(dark: bool = False) -> LinearSegmentedColormap:
    """Blue-to-red ramp through a neutral midpoint, for signed values."""
    return LinearSegmentedColormap.from_list("dk_div", theme(dark)["diverging"])


def _prepare(title: str, dark: bool, figsize: tuple[float, float]):
    tokens = apply_theme(dark)
    figure, axes = plt.subplots(figsize=figsize)
    if title:
        axes.set_title(title)
    return figure, axes, tokens


def _compact(value: float) -> str:
    """``1169432.5`` -> ``'1.17M'`` — readable labels instead of ``1.169e+06``."""
    if value is None or not np.isfinite(value):
        return "-"
    magnitude = abs(value)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= cutoff:
            scaled = value / cutoff
            return f"{scaled:,.1f}{suffix}" if abs(scaled) < 100 else f"{scaled:,.0f}{suffix}"
    if magnitude >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}" if magnitude >= 1 else f"{value:,.3g}"


def _compact_axis(axes, which: str = "x") -> None:
    """Format a numeric axis with :func:`_compact`, dropping the 1e6-style offset."""
    from matplotlib.ticker import FuncFormatter

    axis = axes.xaxis if which == "x" else axes.yaxis
    axis.set_major_formatter(FuncFormatter(lambda value, _: _compact(value)))
    axis.get_offset_text().set_visible(False)


def _finish(figure, path: str | Path, *, tight: bool = True) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def _fold_other(labels: Sequence[Any], limit: int) -> tuple[list[Any], dict[Any, str]]:
    """Keep the first ``limit`` labels; everything else becomes 'Other' in gray."""
    kept = list(labels[:limit])
    overflow = list(labels[limit:])
    mapping = {label: None for label in kept}
    for label in overflow:
        mapping[label] = "Other"
    return kept, mapping


def _series_colors(count: int, dark: bool, limit: int = _ADJACENT_LIMIT) -> list[str]:
    tokens = theme(dark)
    colors = tokens["series"][:limit]
    if count <= len(colors):
        return colors[:count]
    return colors + [tokens["muted"]] * (count - len(colors))


def histogram(df: pd.DataFrame, column: str, *, bins: int = 30, path: str = "histogram.png",
              title: str | None = None, dark: bool = False,
              figsize: tuple[float, float] = (8, 4.5)) -> str:
    """Distribution of one numeric column, with the median marked."""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"histogram: {column!r} has no numeric values")

    figure, axes, tokens = _prepare(title or f"Distribution of {column}", dark, figsize)
    axes.hist(values, bins=bins, color=tokens["series"][0], edgecolor=tokens["surface"],
              linewidth=1.0)
    median = values.median()
    axes.axvline(median, color=tokens["secondary"], linewidth=1.5, linestyle="--")
    axes.annotate(f"median {_compact(median)}", xy=(median, axes.get_ylim()[1] * 0.94),
                  xytext=(6, 0), textcoords="offset points",
                  color=tokens["secondary"], fontsize=9, va="top")
    axes.set_xlabel(column)
    axes.set_ylabel("count")
    axes.grid(axis="x", visible=False)
    _compact_axis(axes, "x")
    _compact_axis(axes, "y")
    return _finish(figure, path)


def bar(df: pd.DataFrame, x: str, y: str | None = None, *, top: int = 15,
        horizontal: bool = True, path: str = "bar.png", title: str | None = None,
        dark: bool = False, label_values: bool = True,
        figsize: tuple[float, float] | None = None) -> str:
    """Ranked bars. With no ``y`` the categories in ``x`` are counted.

    One measure means one series, so every bar shares the first categorical hue —
    color carries no extra meaning here and the axis labels do the identifying.
    """
    if y is None:
        data = df[x].value_counts(dropna=False).head(top)
        labels, values, measure = data.index.astype(str), data.values, "count"
    else:
        grouped = df.groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(top)
        labels, values, measure = grouped.index.astype(str), grouped.values, y

    figsize = figsize or ((8, max(3.0, 0.42 * len(labels) + 1.4)) if horizontal else (9, 5))
    figure, axes, tokens = _prepare(title or f"{measure} by {x}", dark, figsize)
    color = tokens["series"][0]

    if horizontal:
        positions = np.arange(len(labels))[::-1]
        axes.barh(positions, values, color=color, height=0.62)
        axes.set_yticks(positions, labels)
        axes.set_xlabel(measure)
        axes.grid(axis="y", visible=False)
        _compact_axis(axes, "x")
        if label_values and len(labels) <= 25:
            span = max(abs(np.max(values)), 1e-9)
            for position, value in zip(positions, values):
                axes.annotate(_compact(value), (value, position), xytext=(4, 0),
                              textcoords="offset points", va="center", fontsize=9,
                              color=tokens["secondary"])
            axes.set_xlim(right=float(np.max(values)) + span * 0.12)
    else:
        positions = np.arange(len(labels))
        axes.bar(positions, values, color=color, width=0.62)
        axes.set_xticks(positions, labels, rotation=45, ha="right")
        axes.set_ylabel(measure)
        axes.grid(axis="x", visible=False)
        _compact_axis(axes, "y")
        if label_values and len(labels) <= 20:
            for position, value in zip(positions, values):
                axes.annotate(_compact(value), (position, value), xytext=(0, 4),
                              textcoords="offset points", ha="center", fontsize=9,
                              color=tokens["secondary"])
    return _finish(figure, path)


def line(df: pd.DataFrame, x: str, y: str | Iterable[str], *, path: str = "line.png",
         title: str | None = None, dark: bool = False, marker: bool = False,
         figsize: tuple[float, float] = (9, 4.8)) -> str:
    """One or more measures over ``x`` (usually time), all on a single y axis.

    A second axis is deliberately not offered — two scales on one plot invite
    false comparisons. Chart the second measure separately, or index both to a
    common base.
    """
    measures = [y] if isinstance(y, str) else list(y)
    frame = df[[x, *measures]].dropna(subset=[x]).sort_values(x)
    colors = _series_colors(len(measures), dark)

    figure, axes, tokens = _prepare(
        title or f"{', '.join(measures)} over {x}", dark, figsize)
    for measure, color in zip(measures, colors):
        axes.plot(frame[x], pd.to_numeric(frame[measure], errors="coerce"),
                  color=color, label=measure,
                  marker="o" if marker else None, markersize=5,
                  markeredgecolor=tokens["surface"], markeredgewidth=1.2)
        # Direct-label the series end so identity never rests on color alone.
        if len(measures) <= 4:
            present = pd.to_numeric(frame[measure], errors="coerce").dropna()
            if len(present):
                axes.annotate(measure, (frame.loc[present.index[-1], x], present.iloc[-1]),
                              xytext=(6, 0), textcoords="offset points",
                              color=tokens["secondary"], fontsize=9, va="center")
    axes.set_xlabel(x)
    axes.set_ylabel(measures[0] if len(measures) == 1 else "value")
    axes.grid(axis="x", visible=False)
    _compact_axis(axes, "y")
    if len(measures) > 1:
        axes.legend(loc="upper left", ncols=min(len(measures), 4))
    figure.autofmt_xdate()
    return _finish(figure, path)


def scatter(df: pd.DataFrame, x: str, y: str, *, color_by: str | None = None,
            size_by: str | None = None, path: str = "scatter.png",
            title: str | None = None, dark: bool = False, trend: bool = True,
            figsize: tuple[float, float] = (7.5, 5.5)) -> str:
    """Two numeric columns against each other, optionally split by a category.

    Any two groups can end up adjacent here, so only the first three categories
    keep their own hue; the rest fold into a gray "Other".
    """
    frame = df[[x, y] + [c for c in (color_by, size_by) if c]].copy()
    frame[x] = pd.to_numeric(frame[x], errors="coerce")
    frame[y] = pd.to_numeric(frame[y], errors="coerce")
    frame = frame.dropna(subset=[x, y])
    if frame.empty:
        raise ValueError(f"scatter: no rows with numeric {x!r} and {y!r}")

    figure, axes, tokens = _prepare(title or f"{y} vs {x}", dark, figsize)
    sizes = None
    if size_by:
        magnitude = pd.to_numeric(frame[size_by], errors="coerce").fillna(0)
        span = magnitude.max() - magnitude.min()
        sizes = 30 + 220 * ((magnitude - magnitude.min()) / span if span else 0.5)

    if color_by:
        categories = list(frame[color_by].value_counts().index)
        kept, mapping = _fold_other(categories, _ALL_PAIRS_LIMIT)
        colors = _series_colors(len(kept), dark, limit=_ALL_PAIRS_LIMIT)
        for category, color in zip(kept, colors):
            subset = frame[frame[color_by] == category]
            axes.scatter(subset[x], subset[y], s=sizes.loc[subset.index] if sizes is not None else 42,
                         color=color, alpha=0.8, edgecolor=tokens["surface"], linewidth=1.2,
                         label=str(category))
        overflow = [c for c, label in mapping.items() if label == "Other"]
        if overflow:
            subset = frame[frame[color_by].isin(overflow)]
            axes.scatter(subset[x], subset[y],
                         s=sizes.loc[subset.index] if sizes is not None else 42,
                         color=tokens["muted"], alpha=0.65, edgecolor=tokens["surface"],
                         linewidth=1.2, label=f"Other ({len(overflow)})")
        axes.legend(title=color_by, loc="best")
    else:
        axes.scatter(frame[x], frame[y], s=sizes if sizes is not None else 42,
                     color=tokens["series"][0], alpha=0.75,
                     edgecolor=tokens["surface"], linewidth=1.2)

    if trend and len(frame) > 2 and frame[x].nunique() > 1:
        slope, intercept = np.polyfit(frame[x], frame[y], 1)
        span = np.linspace(frame[x].min(), frame[x].max(), 100)
        correlation = frame[x].corr(frame[y])
        axes.plot(span, slope * span + intercept, color=tokens["secondary"],
                  linewidth=1.5, linestyle="--",
                  label=None)
        axes.annotate(f"r = {correlation:+.2f}", xy=(0.98, 0.03), xycoords="axes fraction",
                      ha="right", fontsize=9, color=tokens["secondary"])
    axes.set_xlabel(x)
    axes.set_ylabel(y)
    _compact_axis(axes, "x")
    _compact_axis(axes, "y")
    return _finish(figure, path)


def box(df: pd.DataFrame, columns: str | Iterable[str] | None = None, *, by: str | None = None,
        path: str = "box.png", title: str | None = None, dark: bool = False,
        figsize: tuple[float, float] = (9, 5)) -> str:
    """Spread and outliers, either across numeric columns or split by a category."""
    figure, axes, tokens = _prepare(title or "Distribution", dark, figsize)
    color = tokens["series"][0]
    style = {
        "patch_artist": True,
        "boxprops": {"facecolor": color, "edgecolor": tokens["secondary"], "linewidth": 1.0},
        "medianprops": {"color": tokens["surface"], "linewidth": 2.0},
        "whiskerprops": {"color": tokens["secondary"], "linewidth": 1.0},
        "capprops": {"color": tokens["secondary"], "linewidth": 1.0},
        "flierprops": {"marker": "o", "markersize": 4, "markerfacecolor": tokens["muted"],
                       "markeredgecolor": "none", "alpha": 0.6},
    }

    if by:
        target = columns if isinstance(columns, str) else list(columns or [])[0]
        groups = [(str(name), pd.to_numeric(group[target], errors="coerce").dropna())
                  for name, group in df.groupby(by, dropna=False)]
        groups = [(name, values) for name, values in groups if len(values)][:12]
        axes.boxplot([values for _, values in groups], tick_labels=[name for name, _ in groups],
                     **style)
        axes.set_ylabel(target)
        axes.set_xlabel(by)
    else:
        names = ([columns] if isinstance(columns, str) else list(columns)) if columns is not None \
            else list(df.select_dtypes(include=np.number).columns)[:12]
        data = [pd.to_numeric(df[name], errors="coerce").dropna() for name in names]
        keep = [(name, values) for name, values in zip(names, data) if len(values)]
        axes.boxplot([values for _, values in keep], tick_labels=[name for name, _ in keep],
                     **style)
    axes.grid(axis="x", visible=False)
    _compact_axis(axes, "y")
    plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
    return _finish(figure, path)


def correlation_heatmap(df: pd.DataFrame, *, path: str = "correlations.png",
                        title: str | None = None, dark: bool = False,
                        method: str = "pearson", annotate: bool = True,
                        figsize: tuple[float, float] | None = None) -> str:
    """Correlation matrix on a diverging ramp — negative blue, zero neutral, positive red."""
    numeric = df.select_dtypes(include=np.number)
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        raise ValueError("correlation_heatmap: need at least two varying numeric columns")
    matrix = numeric.corr(method=method)

    size = max(4.5, 0.62 * len(matrix) + 2.2)
    figure, axes, tokens = _prepare(title or "Correlations", dark, figsize or (size, size))
    image = axes.imshow(matrix.values, cmap=diverging_cmap(dark),
                        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    axes.set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    axes.set_yticks(range(len(matrix)), matrix.index)
    axes.grid(False)

    if annotate and len(matrix) <= 12:
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                value = matrix.iloc[row, column]
                ink = tokens["surface"] if abs(value) > 0.55 else tokens["primary"]
                axes.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=8, color=ink)
    bar_ = figure.colorbar(image, ax=axes, shrink=0.75)
    bar_.outline.set_visible(False)
    bar_.ax.tick_params(colors=tokens["muted"], labelsize=8)
    return _finish(figure, path)


def missing_matrix(df: pd.DataFrame, *, path: str = "missing.png", title: str | None = None,
                   dark: bool = False, max_rows: int = 500,
                   figsize: tuple[float, float] | None = None) -> str:
    """Where the nulls are: one column per field, dark marks where a value is missing."""
    sample = df if len(df) <= max_rows else df.sample(max_rows, random_state=0).sort_index()
    mask = sample.isna().to_numpy(dtype=float)

    figure, axes, tokens = _prepare(title or "Missing values", dark,
                                    figsize or (max(6, 0.5 * df.shape[1] + 2), 4.5))
    cmap = LinearSegmentedColormap.from_list(
        "dk_missing", [tokens["grid"], tokens["series"][0]])
    axes.imshow(mask, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    axes.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
    axes.set_ylabel(f"rows (showing {len(sample):,} of {len(df):,})")
    axes.set_yticks([])
    axes.grid(False)
    null_pct = df.isna().mean().mean() * 100
    axes.annotate(f"{null_pct:.1f}% of cells missing — filled marks are nulls",
                  xy=(0, 1.02), xycoords="axes fraction", fontsize=9,
                  color=tokens["secondary"])
    return _finish(figure, path)


def auto_charts(df: pd.DataFrame, outdir: str = "charts", *, dark: bool = False,
                max_charts: int = 12) -> list[str]:
    """Pick sensible charts for this DataFrame and render them all."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    numeric = list(df.select_dtypes(include=np.number).columns)
    datetimes = list(df.select_dtypes(include="datetime").columns)
    categorical = [
        column for column in df.columns
        if column not in numeric and column not in datetimes
        and 1 < df[column].nunique(dropna=True) <= 30
    ]

    if df.isna().any().any():
        made.append(missing_matrix(df, path=outdir_path / "missing_values.png", dark=dark))

    for column in numeric[:4]:
        if len(made) >= max_charts:
            break
        try:
            made.append(histogram(df, column, dark=dark,
                                  path=outdir_path / f"hist_{_slug(column)}.png"))
        except ValueError:
            continue

    for column in categorical[:3]:
        if len(made) >= max_charts:
            break
        made.append(bar(df, column, dark=dark, path=outdir_path / f"bar_{_slug(column)}.png"))

    if len(numeric) >= 2 and len(made) < max_charts:
        made.append(correlation_heatmap(df, dark=dark,
                                        path=outdir_path / "correlations.png"))
        made.append(scatter(df, numeric[0], numeric[1], dark=dark,
                            color_by=categorical[0] if categorical else None,
                            path=outdir_path / "scatter.png"))

    if numeric and len(made) < max_charts:
        made.append(box(df, numeric[:8], dark=dark, path=outdir_path / "box.png"))

    if datetimes and numeric and len(made) < max_charts:
        made.append(line(df, datetimes[0], numeric[0], dark=dark,
                         path=outdir_path / "timeseries.png"))
    return made


def _slug(name: Any) -> str:
    import re

    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower() or "column"
