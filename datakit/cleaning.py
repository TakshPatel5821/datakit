"""Cleaning operations. Every function takes a DataFrame and returns a new one.

The individual steps are useful on their own, and :func:`clean` runs the common
ones in a sensible order.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

__all__ = [
    "standardize_columns",
    "strip_whitespace",
    "drop_empty",
    "drop_duplicates",
    "fill_missing",
    "coerce_types",
    "remove_outliers",
    "replace_values",
    "clean",
    "CleaningReport",
]

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


class CleaningReport(dict):
    """What each cleaning step changed, keyed by step name."""

    def to_text(self) -> str:
        if not self:
            return "No changes made."
        width = max(len(key) for key in self)
        return "\n".join(f"  {key:<{width}}  {value}" for key, value in self.items())


def snake_case(name: str) -> str:
    """``"Total Sales ($)"`` -> ``"total_sales"``."""
    text = _CAMEL_BOUNDARY.sub("_", str(name).strip())
    text = _NON_ALNUM.sub("_", text)
    return text.strip("_").lower() or "column"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake_case column names, de-duplicating any collisions."""
    out = df.copy()
    seen: dict[str, int] = {}
    names = []
    for column in out.columns:
        name = snake_case(column)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    out.columns = names
    return out


def strip_whitespace(df: pd.DataFrame, *, collapse_inner: bool = True) -> pd.DataFrame:
    """Trim leading/trailing whitespace in text columns (and collapse runs of spaces)."""
    out = df.copy()
    for column in out.select_dtypes(include=["object", "string"]).columns:
        values = out[column].astype("string").str.strip()
        if collapse_inner:
            values = values.str.replace(r"\s+", " ", regex=True)
        # Empty strings carry no more information than a null.
        out[column] = values.replace("", pd.NA)
    return out


def drop_empty(df: pd.DataFrame, *, rows: bool = True, columns: bool = True,
               column_threshold: float = 1.0) -> pd.DataFrame:
    """Drop all-null rows, and columns that are at least ``column_threshold`` null."""
    out = df.copy()
    if rows:
        out = out.dropna(how="all")
    if columns and len(out):
        null_ratio = out.isna().mean()
        keep = null_ratio[null_ratio < column_threshold].index
        out = out[list(keep)]
    return out


def drop_duplicates(df: pd.DataFrame, subset: Iterable[str] | None = None,
                    keep: Literal["first", "last", False] = "first") -> pd.DataFrame:
    """Remove duplicate rows, optionally comparing only ``subset`` columns."""
    return df.drop_duplicates(subset=list(subset) if subset else None, keep=keep)


def fill_missing(df: pd.DataFrame, strategy: str | dict[str, Any] = "auto", *,
                 value: Any = None) -> pd.DataFrame:
    """Fill nulls.

    ``strategy`` is either one strategy for every column or a per-column mapping::

        fill_missing(df, "median")
        fill_missing(df, {"age": "mean", "city": "unknown", "score": 0})

    Strategies: ``mean``, ``median``, ``mode``, ``ffill``, ``bfill``, ``zero``,
    ``drop`` (drop rows with a null in that column), ``auto`` (median for
    numbers, mode for everything else). Anything else is used as a literal.
    """
    out = df.copy()
    mapping = strategy if isinstance(strategy, dict) else {c: strategy for c in out.columns}

    for column, how in mapping.items():
        if column not in out.columns:
            continue
        series = out[column]
        if not series.isna().any():
            continue

        if how == "auto":
            how = "median" if pd.api.types.is_numeric_dtype(series) else "mode"

        if how == "drop":
            out = out[out[column].notna()]
            continue
        if how == "mean":
            filler = series.mean() if pd.api.types.is_numeric_dtype(series) else None
        elif how == "median":
            filler = series.median() if pd.api.types.is_numeric_dtype(series) else None
        elif how == "mode":
            modes = series.mode(dropna=True)
            filler = modes.iloc[0] if len(modes) else None
        elif how == "zero":
            filler = 0
        elif how in ("ffill", "bfill"):
            out[column] = series.ffill() if how == "ffill" else series.bfill()
            continue
        elif how == "value":
            filler = value
        else:
            filler = how  # a literal such as "unknown" or 0

        if filler is not None:
            out[column] = series.fillna(filler)
    return out


def coerce_types(df: pd.DataFrame, types: dict[str, str] | None = None, *,
                 infer: bool = True, dayfirst: bool = False) -> pd.DataFrame:
    """Convert column types.

    With an explicit ``types`` mapping (``{"age": "int", "joined": "datetime"}``)
    only those columns are touched. With ``infer=True`` every object column is
    tested against numeric, then datetime, and converted when the whole column
    (ignoring nulls) parses cleanly.
    """
    out = df.copy()
    aliases = {"int": "Int64", "integer": "Int64", "float": "float64", "str": "string",
               "string": "string", "bool": "boolean", "boolean": "boolean",
               "category": "category"}

    for column, target in (types or {}).items():
        if column not in out.columns:
            continue
        if target in ("datetime", "date", "timestamp"):
            out[column] = pd.to_datetime(out[column], errors="coerce", dayfirst=dayfirst)
        elif target in ("int", "integer"):
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = numeric.round().astype("Int64")
        else:
            out[column] = out[column].astype(aliases.get(target, target), errors="ignore")

    if infer:
        explicit = set(types or {})
        for column in out.columns:
            if column in explicit or not _is_text_like(out[column]):
                continue
            out[column] = _infer_column(out[column], dayfirst=dayfirst)
    return out


def _is_text_like(series: pd.Series) -> bool:
    """True for columns worth re-inspecting — object and string dtypes alike.

    (pandas 2 reads text as ``object``, pandas 3 as ``str``; both land here.)
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _infer_column(series: pd.Series, *, dayfirst: bool = False,
                  threshold: float = 0.95) -> pd.Series:
    """Return the series converted to a better dtype, or unchanged."""
    present = series.dropna()
    if present.empty:
        return series

    cleaned = present.astype(str).str.strip().str.replace(r"[,$%\s]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().mean() >= threshold:
        converted = pd.to_numeric(
            series.astype(str).str.strip().str.replace(r"[,$%\s]", "", regex=True),
            errors="coerce",
        )
        if (converted.dropna() % 1 == 0).all():
            return converted.astype("Int64")
        return converted

    # Pure digit strings are identifiers (zip codes, ids) more often than dates.
    if not present.astype(str).str.fullmatch(r"\d+").all():
        parsed = pd.to_datetime(present, errors="coerce", dayfirst=dayfirst, format="mixed")
        if parsed.notna().mean() >= threshold:
            return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, format="mixed")

    lowered = present.astype(str).str.lower()
    if lowered.isin({"true", "false", "yes", "no", "y", "n", "0", "1"}).all():
        mapping = {"true": True, "yes": True, "y": True, "1": True,
                   "false": False, "no": False, "n": False, "0": False}
        return series.astype(str).str.lower().map(mapping).astype("boolean")
    return series


def remove_outliers(df: pd.DataFrame, columns: Iterable[str] | None = None, *,
                    method: Literal["iqr", "zscore"] = "iqr", factor: float = 1.5,
                    action: Literal["drop", "clip", "null"] = "drop") -> pd.DataFrame:
    """Drop, clip or null out extreme numeric values.

    ``iqr`` flags values outside ``Q1 - factor*IQR .. Q3 + factor*IQR``;
    ``zscore`` flags values more than ``factor`` standard deviations from the mean
    (pass ``factor=3`` for the usual rule).
    """
    out = df.copy()
    targets = list(columns) if columns else list(out.select_dtypes(include=np.number).columns)
    mask = pd.Series(False, index=out.index)

    for column in targets:
        series = pd.to_numeric(out[column], errors="coerce")
        low, high = outlier_bounds(series, method=method, factor=factor)
        if low is None:
            continue
        if action == "clip":
            out[column] = series.clip(low, high)
        else:
            flagged = (series < low) | (series > high)
            if action == "null":
                out.loc[flagged.fillna(False), column] = pd.NA
            else:
                mask |= flagged.fillna(False)

    if action == "drop":
        out = out[~mask]
    return out


def outlier_bounds(series: pd.Series, *, method: str = "iqr",
                   factor: float = 1.5) -> tuple[float | None, float | None]:
    """Lower/upper cutoffs for a numeric series, or ``(None, None)`` if not applicable."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None, None
    if method == "zscore":
        std = numeric.std()
        if not std or np.isnan(std):
            return None, None
        mean = numeric.mean()
        return mean - factor * std, mean + factor * std
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    spread = q3 - q1
    if spread == 0:
        return None, None
    return q1 - factor * spread, q3 + factor * spread


def replace_values(df: pd.DataFrame, mapping: dict[str, dict[Any, Any]] | dict[Any, Any], *,
                   columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Replace values everywhere, or per column with a nested mapping."""
    out = df.copy()
    if mapping and all(isinstance(value, dict) for value in mapping.values()):
        for column, column_map in mapping.items():
            if column in out.columns:
                out[column] = out[column].replace(column_map)
        return out
    targets = list(columns) if columns else list(out.columns)
    out[targets] = out[targets].replace(mapping)
    return out


def clean(df: pd.DataFrame, *, standardize: bool = True, strip: bool = True,
          drop_dupes: bool = True, drop_empties: bool = True,
          missing: str | dict[str, Any] | None = None, types: dict[str, str] | None = None,
          infer_types: bool = True, outliers: str | None = None,
          outlier_factor: float = 1.5,
          report: CleaningReport | None = None) -> pd.DataFrame:
    """Run the standard cleaning pipeline.

    Pass a :class:`CleaningReport` as ``report`` to find out what each step did.
    """
    log = report if report is not None else CleaningReport()
    out = df

    if standardize:
        renamed = standardize_columns(out)
        changed = sum(1 for a, b in zip(out.columns, renamed.columns) if a != b)
        if changed:
            log["standardize_columns"] = f"renamed {changed} column(s)"
        out = renamed
    if strip:
        out = strip_whitespace(out)
        log["strip_whitespace"] = "trimmed text columns"
    if drop_empties:
        before_rows, before_cols = out.shape
        out = drop_empty(out)
        dropped_rows = before_rows - out.shape[0]
        dropped_cols = before_cols - out.shape[1]
        if dropped_rows or dropped_cols:
            log["drop_empty"] = f"dropped {dropped_rows} empty row(s), {dropped_cols} empty column(s)"
    if drop_dupes:
        before = len(out)
        out = drop_duplicates(out)
        if before - len(out):
            log["drop_duplicates"] = f"removed {before - len(out)} duplicate row(s)"
    if types or infer_types:
        before_types = out.dtypes.astype(str).to_dict()
        out = coerce_types(out, types, infer=infer_types)
        changed = [c for c, t in out.dtypes.astype(str).items() if before_types.get(c) != t]
        if changed:
            log["coerce_types"] = f"converted {', '.join(changed)}"
    if missing is not None:
        before = int(out.isna().sum().sum())
        out = fill_missing(out, missing)
        after = int(out.isna().sum().sum())
        log["fill_missing"] = f"filled {before - after} missing value(s)"
    if outliers:
        before = len(out)
        action = "clip" if outliers == "clip" else "drop"
        method = "zscore" if outliers == "zscore" else "iqr"
        out = remove_outliers(out, method=method, factor=outlier_factor, action=action)
        log["remove_outliers"] = (
            f"clipped extremes ({method})" if action == "clip"
            else f"dropped {before - len(out)} outlier row(s) ({method})"
        )

    return out.reset_index(drop=True)
