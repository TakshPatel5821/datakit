"""Data operations: select, filter, sort, aggregate, join, reshape, derive.

Each function is a plain DataFrame -> DataFrame transformation. :func:`apply_steps`
runs a list of them described as data, which is what the CLI and the pipeline
runner use.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "select", "drop", "rename", "filter_rows", "sort", "limit", "sample",
    "group_by", "aggregate", "join", "pivot", "unpivot", "derive", "bin_column",
    "value_counts", "top_n", "resample_time", "rolling", "rank", "cumulative",
    "apply_steps", "STEP_OPS",
]


class TransformError(ValueError):
    """Raised when an operation is misconfigured."""


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], op: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise TransformError(
            f"{op}: column(s) not found: {', '.join(missing)}. "
            f"Available: {', '.join(map(str, df.columns))}"
        )


def select(df: pd.DataFrame, columns: str | Iterable[str]) -> pd.DataFrame:
    """Keep only these columns, in this order."""
    names = _as_list(columns)
    _require_columns(df, names, "select")
    return df[names].copy()


def drop(df: pd.DataFrame, columns: str | Iterable[str]) -> pd.DataFrame:
    """Remove these columns (missing ones are ignored)."""
    names = [c for c in _as_list(columns) if c in df.columns]
    return df.drop(columns=names)


def rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns via ``{"old": "new"}``."""
    return df.rename(columns=mapping)


def filter_rows(df: pd.DataFrame, query: str | None = None, *, column: str | None = None,
                op: str | None = None, value: Any = None) -> pd.DataFrame:
    """Filter rows.

    Either a pandas query string (``"amount > 100 and region == 'EU'"``) or an
    explicit ``column``/``op``/``value`` triple, where ``op`` is one of
    ``== != > >= < <= in not_in contains startswith endswith isnull notnull between``.
    """
    if query:
        try:
            return df.query(query).copy()
        except Exception as exc:
            raise TransformError(f"filter: could not evaluate {query!r} ({exc})") from exc

    if not column or not op:
        raise TransformError("filter: give either a query or column/op/value")
    _require_columns(df, [column], "filter")
    series = df[column]

    if op == "==":
        mask = series == value
    elif op == "!=":
        mask = series != value
    elif op == ">":
        mask = series > value
    elif op == ">=":
        mask = series >= value
    elif op == "<":
        mask = series < value
    elif op == "<=":
        mask = series <= value
    elif op == "in":
        mask = series.isin(_as_list(value) if isinstance(value, str) else list(value))
    elif op == "not_in":
        mask = ~series.isin(_as_list(value) if isinstance(value, str) else list(value))
    elif op == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    elif op == "startswith":
        mask = series.astype(str).str.startswith(str(value), na=False)
    elif op == "endswith":
        mask = series.astype(str).str.endswith(str(value), na=False)
    elif op == "isnull":
        mask = series.isna()
    elif op == "notnull":
        mask = series.notna()
    elif op == "between":
        low, high = value
        mask = series.between(low, high)
    else:
        raise TransformError(f"filter: unknown operator {op!r}")
    return df[mask.fillna(False)].copy()


def sort(df: pd.DataFrame, by: str | Iterable[str], *, ascending: bool | Sequence[bool] = True,
         na_position: str = "last") -> pd.DataFrame:
    """Sort by one or more columns."""
    columns = _as_list(by)
    _require_columns(df, columns, "sort")
    return df.sort_values(columns, ascending=ascending, na_position=na_position)


def limit(df: pd.DataFrame, n: int = 10, *, offset: int = 0) -> pd.DataFrame:
    """Take ``n`` rows starting at ``offset``."""
    return df.iloc[offset: offset + n].copy()


def sample(df: pd.DataFrame, n: int | None = None, *, fraction: float | None = None,
           seed: int | None = None) -> pd.DataFrame:
    """Random sample of rows."""
    if fraction is not None:
        return df.sample(frac=fraction, random_state=seed)
    return df.sample(n=min(n or 10, len(df)), random_state=seed)


def aggregate(df: pd.DataFrame, aggregations: dict[str, str | list[str]] | None = None,
              *, by: str | Iterable[str] | None = None) -> pd.DataFrame:
    """Group and aggregate.

    ``aggregate(df, {"revenue": "sum", "units": ["mean", "max"]}, by="region")``
    produces ``revenue_sum``, ``units_mean`` and ``units_max`` — result columns
    are always ``<column>_<func>``, so adding a second function to a column never
    silently renames the first one out from under a later step.

    With no ``aggregations`` it counts rows per group; with no ``by`` it
    aggregates the whole frame into one row.
    """
    keys = _as_list(by)
    if keys:
        _require_columns(df, keys, "group_by")

    if not aggregations:
        if not keys:
            return pd.DataFrame({"count": [len(df)]})
        out = df.groupby(keys, dropna=False).size().reset_index(name="count")
        return out.sort_values("count", ascending=False, ignore_index=True)

    normalized = {
        column: (aggs if isinstance(aggs, list) else [aggs])
        for column, aggs in aggregations.items()
    }
    _require_columns(df, [c for c in normalized if c != "*"], "aggregate")

    if not keys:
        row = {}
        for column, funcs in normalized.items():
            for func in funcs:
                row[f"{column}_{func}"] = getattr(df[column], func)()
        return pd.DataFrame([row])

    grouped = df.groupby(keys, dropna=False).agg(normalized)
    grouped.columns = [f"{column}_{func}" for column, func in grouped.columns]
    return grouped.reset_index()


#: ``group_by`` reads better at call sites that lead with the grouping keys.
def group_by(df: pd.DataFrame, by: str | Iterable[str],
             aggregations: dict[str, str | list[str]] | None = None) -> pd.DataFrame:
    """Group by ``by`` and aggregate (counts rows when no aggregations are given)."""
    return aggregate(df, aggregations, by=by)


def join(left: pd.DataFrame, right: pd.DataFrame, *, on: str | Iterable[str] | None = None,
         left_on: str | Iterable[str] | None = None, right_on: str | Iterable[str] | None = None,
         how: str = "inner", suffixes: tuple[str, str] = ("", "_right")) -> pd.DataFrame:
    """Join two DataFrames (``inner``, ``left``, ``right``, ``outer``, ``cross``)."""
    if on:
        keys = _as_list(on)
        _require_columns(left, keys, "join (left)")
        _require_columns(right, keys, "join (right)")
        return left.merge(right, on=keys, how=how, suffixes=suffixes)
    if how == "cross":
        return left.merge(right, how="cross", suffixes=suffixes)
    if not (left_on and right_on):
        raise TransformError("join: give on=, or both left_on= and right_on=")
    return left.merge(right, left_on=_as_list(left_on), right_on=_as_list(right_on),
                      how=how, suffixes=suffixes)


def pivot(df: pd.DataFrame, *, index: str | Iterable[str], columns: str,
          values: str, agg: str = "sum", fill_value: Any = None) -> pd.DataFrame:
    """Long -> wide, aggregating duplicates with ``agg``."""
    rows = _as_list(index)
    _require_columns(df, rows + [columns, values], "pivot")
    table = pd.pivot_table(df, index=rows, columns=columns, values=values,
                           aggfunc=agg, fill_value=fill_value, observed=False)
    table.columns = [str(column) for column in table.columns]
    return table.reset_index()


def unpivot(df: pd.DataFrame, *, id_columns: str | Iterable[str],
            value_columns: str | Iterable[str] | None = None,
            var_name: str = "variable", value_name: str = "value") -> pd.DataFrame:
    """Wide -> long (``melt``)."""
    ids = _as_list(id_columns)
    _require_columns(df, ids, "unpivot")
    values = _as_list(value_columns) or [c for c in df.columns if c not in ids]
    return df.melt(id_vars=ids, value_vars=values, var_name=var_name, value_name=value_name)


def derive(df: pd.DataFrame, expressions: dict[str, str]) -> pd.DataFrame:
    """Add computed columns from expressions over the existing ones.

    ``derive(df, {"margin": "revenue - cost", "pct": "margin / revenue * 100"})``.
    Expressions are evaluated in order, so later ones can use earlier results.
    """
    out = df.copy()
    for name, expression in expressions.items():
        try:
            out[name] = out.eval(expression)
        except Exception:
            # eval() does not cover string methods or function calls; fall back to
            # a restricted python expression with the columns in scope.
            scope = {column: out[column] for column in out.columns}
            scope.update({"np": np, "pd": pd, "len": len, "abs": abs, "round": round,
                          "min": min, "max": max})
            try:
                out[name] = eval(expression, {"__builtins__": {}}, scope)  # noqa: S307
            except Exception as exc:
                raise TransformError(
                    f"derive: could not evaluate {name} = {expression!r} ({exc})"
                ) from exc
    return out


def bin_column(df: pd.DataFrame, column: str, *, bins: int | list[float] = 5,
               labels: list[str] | None = None, target: str | None = None,
               quantile: bool = False) -> pd.DataFrame:
    """Bucket a numeric column into ranges (equal width, or equal frequency)."""
    _require_columns(df, [column], "bin")
    out = df.copy()
    series = pd.to_numeric(out[column], errors="coerce")
    name = target or f"{column}_bin"
    if quantile:
        out[name] = pd.qcut(series, q=bins if isinstance(bins, int) else len(bins),
                            labels=labels, duplicates="drop")
    else:
        out[name] = pd.cut(series, bins=bins, labels=labels)
    return out


def value_counts(df: pd.DataFrame, column: str, *, normalize: bool = False,
                 top: int | None = None) -> pd.DataFrame:
    """Frequency table for one column."""
    _require_columns(df, [column], "value_counts")
    counts = df[column].value_counts(normalize=normalize, dropna=False)
    if top:
        counts = counts.head(top)
    name = "share" if normalize else "count"
    return counts.rename(name).reset_index().rename(columns={"index": column})


def top_n(df: pd.DataFrame, column: str, n: int = 10, *, largest: bool = True) -> pd.DataFrame:
    """The ``n`` rows with the largest (or smallest) value in ``column``."""
    _require_columns(df, [column], "top_n")
    series = pd.to_numeric(df[column], errors="coerce")
    order = series.sort_values(ascending=not largest, na_position="last")
    return df.loc[order.index[:n]].copy()


def resample_time(df: pd.DataFrame, *, on: str, freq: str = "M",
                  aggregations: dict[str, str] | None = None) -> pd.DataFrame:
    """Roll a time series up to a period (``D``, ``W``, ``M``, ``Q``, ``Y``)."""
    _require_columns(df, [on], "resample")
    out = df.copy()
    out[on] = pd.to_datetime(out[on], errors="coerce")
    out = out.dropna(subset=[on]).set_index(on)
    if aggregations:
        result = out.resample(freq).agg(aggregations)
    else:
        numeric = out.select_dtypes(include=np.number)
        result = numeric.resample(freq).sum()
        result["count"] = out.resample(freq).size()
    return result.reset_index()


def rolling(df: pd.DataFrame, column: str, *, window: int = 7, func: str = "mean",
            target: str | None = None) -> pd.DataFrame:
    """Add a rolling-window statistic over ``column``."""
    _require_columns(df, [column], "rolling")
    out = df.copy()
    out[target or f"{column}_{func}_{window}"] = getattr(
        out[column].rolling(window, min_periods=1), func
    )()
    return out


def cumulative(df: pd.DataFrame, column: str, *, func: str = "sum",
               target: str | None = None) -> pd.DataFrame:
    """Add a running total (or running max/min/product) of ``column``."""
    _require_columns(df, [column], "cumulative")
    out = df.copy()
    out[target or f"{column}_cum{func}"] = getattr(out[column], f"cum{func}")()
    return out


def rank(df: pd.DataFrame, column: str, *, ascending: bool = False,
         method: str = "dense", target: str | None = None) -> pd.DataFrame:
    """Add a rank column."""
    _require_columns(df, [column], "rank")
    out = df.copy()
    out[target or f"{column}_rank"] = out[column].rank(ascending=ascending, method=method)
    return out


#: Operation name -> callable, used by :func:`apply_steps`, the CLI and pipelines.
STEP_OPS = {
    "select": select,
    "drop": drop,
    "rename": rename,
    "filter": filter_rows,
    "sort": sort,
    "limit": limit,
    "head": limit,
    "sample": sample,
    "aggregate": aggregate,
    "group_by": group_by,
    "pivot": pivot,
    "unpivot": unpivot,
    "melt": unpivot,
    "derive": derive,
    "bin": bin_column,
    "value_counts": value_counts,
    "top_n": top_n,
    "resample": resample_time,
    "rolling": rolling,
    "cumulative": cumulative,
    "rank": rank,
}


def apply_steps(df: pd.DataFrame, steps: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Run a list of operations described as dicts.

        apply_steps(df, [
            {"op": "filter", "query": "amount > 0"},
            {"op": "derive", "expressions": {"net": "amount - fee"}},
            {"op": "group_by", "by": "region", "aggregations": {"net": "sum"}},
            {"op": "sort", "by": "net", "ascending": false},
        ])

    Operations from :mod:`datakit.clean` (``clean``, ``fill_missing``,
    ``drop_duplicates``, ``coerce_types``, ``remove_outliers``) are available too.
    """
    from . import cleaning as clean_module

    extra = {
        "clean": clean_module.clean,
        "fill_missing": clean_module.fill_missing,
        "drop_duplicates": clean_module.drop_duplicates,
        "coerce_types": clean_module.coerce_types,
        "remove_outliers": clean_module.remove_outliers,
        "standardize_columns": clean_module.standardize_columns,
    }

    out = df
    for index, step in enumerate(steps, start=1):
        step = dict(step)
        name = step.pop("op", None) or step.pop("operation", None)
        if not name:
            raise TransformError(f"step {index}: missing 'op'")
        func = STEP_OPS.get(name) or extra.get(name)
        if func is None:
            known = sorted(set(STEP_OPS) | set(extra))
            raise TransformError(f"step {index}: unknown op {name!r}. Known ops: {', '.join(known)}")
        step.pop("name", None)  # allow a human label on each step
        try:
            out = func(out, **step)
        except TypeError as exc:
            raise TransformError(f"step {index} ({name}): bad arguments — {exc}") from exc
    return out
