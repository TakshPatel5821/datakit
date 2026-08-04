"""Dataset profiling: shape, per-column statistics, correlations and a quality score."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import outlier_bounds

__all__ = ["Profile", "profile", "describe", "correlations", "quality_score", "missing_summary"]


def _scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value.item() if hasattr(value, "item") else value


@dataclass
class Profile:
    """Everything :func:`profile` found out about a DataFrame."""

    summary: dict = field(default_factory=dict)
    columns: dict[str, dict] = field(default_factory=dict)
    correlations: pd.DataFrame | None = None
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "columns": self.columns,
            "quality": self.quality,
            "correlations": (
                self.correlations.round(3).to_dict() if self.correlations is not None else None
            ),
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            from pathlib import Path

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_frame(self) -> pd.DataFrame:
        """Per-column facts as a DataFrame, one row per column."""
        return pd.DataFrame(self.columns).T

    def to_text(self, max_columns: int | None = None) -> str:
        lines = []
        summary = self.summary
        lines.append("=" * 72)
        lines.append(f"DATA PROFILE  —  {summary.get('rows', 0):,} rows x "
                     f"{summary.get('columns', 0)} columns")
        lines.append("=" * 72)
        lines.append(f"Memory        : {summary.get('memory', '?')}")
        lines.append(f"Duplicate rows: {summary.get('duplicate_rows', 0):,} "
                     f"({summary.get('duplicate_pct', 0):.1f}%)")
        lines.append(f"Missing cells : {summary.get('missing_cells', 0):,} "
                     f"({summary.get('missing_pct', 0):.1f}%)")
        lines.append(f"Quality score : {self.quality.get('score', 0):.1f}/100 "
                     f"({self.quality.get('grade', '?')})")
        for note in self.quality.get("notes", []):
            lines.append(f"  ! {note}")

        lines.append("")
        lines.append("-" * 72)
        lines.append("COLUMNS")
        lines.append("-" * 72)
        items = list(self.columns.items())
        if max_columns:
            items = items[:max_columns]
        for name, info in items:
            lines.append(f"\n{name}  ({info['dtype']}, {info['kind']})")
            lines.append(f"    non-null {info['non_null']:,}   null {info['nulls']:,} "
                         f"({info['null_pct']:.1f}%)   unique {info['unique']:,}")
            if info["kind"] == "numeric":
                lines.append(
                    f"    min {_fmt(info['min'])}  q1 {_fmt(info['q1'])}  "
                    f"median {_fmt(info['median'])}  q3 {_fmt(info['q3'])}  "
                    f"max {_fmt(info['max'])}"
                )
                lines.append(
                    f"    mean {_fmt(info['mean'])}  std {_fmt(info['std'])}  "
                    f"skew {_fmt(info['skew'])}  zeros {info['zeros']:,}  "
                    f"outliers {info['outliers']:,}"
                )
            elif info["kind"] == "datetime":
                lines.append(f"    range {info['min']} .. {info['max']}")
            else:
                top = info.get("top_values", {})
                preview = ", ".join(f"{key!r}: {count:,}" for key, count in list(top.items())[:5])
                if preview:
                    lines.append(f"    top: {preview}")
                if info.get("avg_length") is not None:
                    lines.append(f"    avg length {info['avg_length']:.1f} chars")
        if max_columns and len(self.columns) > max_columns:
            lines.append(f"\n... {len(self.columns) - max_columns} more column(s)")

        if self.correlations is not None and not self.correlations.empty:
            strong = strongest_correlations(self.correlations)
            if strong:
                lines.append("")
                lines.append("-" * 72)
                lines.append("STRONGEST CORRELATIONS")
                lines.append("-" * 72)
                for a, b, value in strong:
                    lines.append(f"    {a} ~ {b}: {value:+.2f}")
        return "\n".join(lines)

    def __str__(self) -> str:  # so print(profile(df)) just works
        return self.to_text()


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000 or (value and abs(value) < 0.01):
            return f"{value:,.3g}"
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def profile(df: pd.DataFrame, *, correlation_method: str = "pearson",
            top_n: int = 10) -> Profile:
    """Profile a DataFrame end to end."""
    rows = len(df)
    cells = max(rows * max(df.shape[1], 1), 1)
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum()) if rows else 0

    summary = {
        "rows": rows,
        "columns": int(df.shape[1]),
        "memory": _human_bytes(df.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicates,
        "duplicate_pct": duplicates / rows * 100 if rows else 0.0,
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / cells * 100,
        "numeric_columns": int(df.select_dtypes(include=np.number).shape[1]),
        "datetime_columns": int(df.select_dtypes(include="datetime").shape[1]),
    }

    columns = {str(name): _profile_column(df[name], top_n=top_n) for name in df.columns}
    corr = correlations(df, method=correlation_method)
    result = Profile(summary=summary, columns=columns, correlations=corr)
    result.quality = quality_score(df, result)
    return result


def _profile_column(series: pd.Series, *, top_n: int = 10) -> dict:
    non_null = int(series.notna().sum())
    nulls = int(series.isna().sum())
    total = max(len(series), 1)

    info: dict[str, Any] = {
        "dtype": str(series.dtype),
        "kind": _kind(series),
        "non_null": non_null,
        "nulls": nulls,
        "null_pct": nulls / total * 100,
        "unique": int(series.nunique(dropna=True)),
        "unique_pct": series.nunique(dropna=True) / max(non_null, 1) * 100,
        "constant": bool(non_null and series.nunique(dropna=True) <= 1),
    }

    if info["kind"] == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        present = numeric.dropna()
        low, high = outlier_bounds(present)
        outliers = int(((present < low) | (present > high)).sum()) if low is not None else 0
        info.update({
            "min": _scalar(present.min()) if len(present) else None,
            "max": _scalar(present.max()) if len(present) else None,
            "mean": _scalar(present.mean()) if len(present) else None,
            "std": _scalar(present.std()) if len(present) else None,
            "median": _scalar(present.median()) if len(present) else None,
            "q1": _scalar(present.quantile(0.25)) if len(present) else None,
            "q3": _scalar(present.quantile(0.75)) if len(present) else None,
            "skew": _scalar(present.skew()) if len(present) > 2 else None,
            "kurtosis": _scalar(present.kurtosis()) if len(present) > 3 else None,
            "zeros": int((present == 0).sum()),
            "negatives": int((present < 0).sum()),
            "outliers": outliers,
        })
    elif info["kind"] == "datetime":
        present = series.dropna()
        info.update({
            "min": str(present.min()) if len(present) else None,
            "max": str(present.max()) if len(present) else None,
            "span_days": (present.max() - present.min()).days if len(present) > 1 else None,
        })
    else:
        counts = series.value_counts(dropna=True).head(top_n)
        info["top_values"] = {str(key): int(value) for key, value in counts.items()}
        text = series.dropna().astype(str)
        info["avg_length"] = float(text.str.len().mean()) if len(text) else None
        info["max_length"] = int(text.str.len().max()) if len(text) else None
    return info


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    return "text"


def _human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """A compact per-column overview as a DataFrame (handy in notebooks)."""
    rows = []
    for name in df.columns:
        series = df[name]
        info = _profile_column(series, top_n=1)
        rows.append({
            "column": name,
            "dtype": info["dtype"],
            "kind": info["kind"],
            "non_null": info["non_null"],
            "nulls": info["nulls"],
            "null_%": round(info["null_pct"], 1),
            "unique": info["unique"],
            "min": info.get("min"),
            "median": info.get("median"),
            "max": info.get("max"),
            "mean": info.get("mean"),
        })
    return pd.DataFrame(rows).set_index("column")


def correlations(df: pd.DataFrame, *, method: str = "pearson",
                 min_columns: int = 2) -> pd.DataFrame | None:
    """Correlation matrix of the numeric columns, or None if there are too few."""
    numeric = df.select_dtypes(include=np.number)
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < min_columns:
        return None
    return numeric.corr(method=method)


def strongest_correlations(matrix: pd.DataFrame, *, threshold: float = 0.5,
                           limit: int = 10) -> list[tuple[str, str, float]]:
    """Distinct column pairs whose correlation exceeds ``threshold``, strongest first."""
    pairs = []
    columns = list(matrix.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            value = matrix.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((str(a), str(b), float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs[:limit]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Null counts per column, worst first."""
    counts = df.isna().sum()
    frame = pd.DataFrame({
        "nulls": counts,
        "null_%": (counts / max(len(df), 1) * 100).round(2),
    })
    return frame[frame["nulls"] > 0].sort_values("nulls", ascending=False)


def quality_score(df: pd.DataFrame, prof: Profile | None = None) -> dict:
    """Score the dataset 0-100 on completeness, uniqueness and column health."""
    prof = prof or profile(df)
    rows = max(len(df), 1)
    notes: list[str] = []

    completeness = 100.0 - prof.summary["missing_pct"]
    uniqueness = 100.0 - prof.summary["duplicate_pct"]

    constant_columns = [name for name, info in prof.columns.items() if info["constant"]]
    high_null_columns = [name for name, info in prof.columns.items() if info["null_pct"] > 50]
    id_like = [name for name, info in prof.columns.items()
               if info["kind"] == "text" and info["unique_pct"] > 95 and rows > 20]

    penalty = 8 * len(constant_columns) + 5 * len(high_null_columns)
    structure = max(0.0, 100.0 - penalty)

    score = 0.45 * completeness + 0.25 * uniqueness + 0.30 * structure

    if prof.summary["duplicate_rows"]:
        notes.append(f"{prof.summary['duplicate_rows']:,} duplicate row(s)")
    if high_null_columns:
        notes.append(f"over half missing in: {', '.join(high_null_columns)}")
    if constant_columns:
        notes.append(f"single-valued column(s): {', '.join(constant_columns)}")
    if id_like:
        notes.append(f"looks like an identifier: {', '.join(id_like)}")
    outlier_columns = [name for name, info in prof.columns.items()
                       if info.get("outliers", 0) > 0.05 * rows]
    if outlier_columns:
        notes.append(f">5% outliers in: {', '.join(outlier_columns)}")

    grade = ("excellent" if score >= 90 else "good" if score >= 75
             else "fair" if score >= 60 else "poor")
    return {
        "score": round(score, 1),
        "grade": grade,
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "structure": round(structure, 1),
        "notes": notes,
    }
