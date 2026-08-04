"""A chainable wrapper around a DataFrame — the friendly face of the toolkit.

    from datakit import Dataset

    (Dataset.read("sales.csv")
        .clean(missing="median")
        .filter("amount > 0")
        .derive(margin="revenue - cost")
        .group_by("region", {"margin": "sum"})
        .sort("margin", ascending=False)
        .save("out/by_region.xlsx"))

Every method returns a new :class:`Dataset`, so nothing mutates underneath you.
The underlying DataFrame is always available as ``.df``.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from . import cleaning as _clean
from . import io as _io
from . import profiling as _profiling
from . import transform as _transform
from . import validation as _validate
from . import viz as _viz

__all__ = ["Dataset"]


class Dataset:
    """A DataFrame plus a fluent set of data-management operations."""

    def __init__(self, df: pd.DataFrame, *, name: str | None = None):
        self.df = df
        self.name = name
        #: Set by :meth:`clean` — what each cleaning step changed.
        self.last_report: _clean.CleaningReport | None = None

    # ------------------------------------------------------------------ io

    @classmethod
    def read(cls, source: str, **kwargs: Any) -> "Dataset":
        """Load a file, database table/query or HTTP endpoint. See :func:`datakit.io.load`."""
        return cls(_io.load(source, **kwargs), name=str(source))

    @classmethod
    def from_records(cls, records: Iterable[dict], *, name: str | None = None) -> "Dataset":
        """Build a Dataset from a list of dicts."""
        return cls(pd.DataFrame(list(records)), name=name)

    def save(self, target: str, **kwargs: Any) -> str:
        """Write to a file or database table. Returns the target."""
        return _io.save(self.df, target, **kwargs)

    # -------------------------------------------------------------- cleaning

    def clean(self, **kwargs: Any) -> "Dataset":
        """Run the standard cleaning pipeline (see :func:`datakit.clean.clean`)."""
        report = _clean.CleaningReport()
        out = self._wrap(_clean.clean(self.df, report=report, **kwargs))
        out.last_report = report
        return out

    def standardize_columns(self) -> "Dataset":
        """snake_case every column name."""
        return self._wrap(_clean.standardize_columns(self.df))

    def fill_missing(self, strategy: str | dict[str, Any] = "auto", **kwargs: Any) -> "Dataset":
        """Fill nulls, globally or per column."""
        return self._wrap(_clean.fill_missing(self.df, strategy, **kwargs))

    def drop_duplicates(self, subset: Iterable[str] | None = None, **kwargs: Any) -> "Dataset":
        """Remove duplicate rows."""
        return self._wrap(_clean.drop_duplicates(self.df, subset, **kwargs))

    def coerce_types(self, types: dict[str, str] | None = None, **kwargs: Any) -> "Dataset":
        """Convert column types, explicitly or by inference."""
        return self._wrap(_clean.coerce_types(self.df, types, **kwargs))

    def remove_outliers(self, columns: Iterable[str] | None = None, **kwargs: Any) -> "Dataset":
        """Drop, clip or null extreme numeric values."""
        return self._wrap(_clean.remove_outliers(self.df, columns, **kwargs))

    # ------------------------------------------------------------ operations

    def select(self, columns: str | Iterable[str]) -> "Dataset":
        """Keep only these columns."""
        return self._wrap(_transform.select(self.df, columns))

    def drop(self, columns: str | Iterable[str]) -> "Dataset":
        """Remove these columns."""
        return self._wrap(_transform.drop(self.df, columns))

    def rename(self, mapping: dict[str, str] | None = None, **pairs: str) -> "Dataset":
        """Rename columns: ``.rename(old_name="new_name")`` or with a dict."""
        return self._wrap(_transform.rename(self.df, {**(mapping or {}), **pairs}))

    def filter(self, query: str | None = None, **kwargs: Any) -> "Dataset":
        """Keep matching rows: a query string, or column/op/value."""
        return self._wrap(_transform.filter_rows(self.df, query, **kwargs))

    def sort(self, by: str | Iterable[str], **kwargs: Any) -> "Dataset":
        """Sort by one or more columns."""
        return self._wrap(_transform.sort(self.df, by, **kwargs))

    def limit(self, n: int = 10, **kwargs: Any) -> "Dataset":
        """Take the first ``n`` rows."""
        return self._wrap(_transform.limit(self.df, n, **kwargs))

    def sample(self, n: int | None = None, **kwargs: Any) -> "Dataset":
        """Random sample of rows."""
        return self._wrap(_transform.sample(self.df, n, **kwargs))

    def group_by(self, by: str | Iterable[str],
                 aggregations: dict[str, str | list[str]] | None = None) -> "Dataset":
        """Group and aggregate (counts rows when no aggregations are given)."""
        return self._wrap(_transform.group_by(self.df, by, aggregations))

    def aggregate(self, aggregations: dict[str, str | list[str]] | None = None,
                  **kwargs: Any) -> "Dataset":
        """Aggregate the whole frame, or per group with ``by=``."""
        return self._wrap(_transform.aggregate(self.df, aggregations, **kwargs))

    def derive(self, expressions: dict[str, str] | None = None, **named: str) -> "Dataset":
        """Add computed columns: ``.derive(margin="revenue - cost")``."""
        return self._wrap(_transform.derive(self.df, {**(expressions or {}), **named}))

    def join(self, other: "Dataset | pd.DataFrame", **kwargs: Any) -> "Dataset":
        """Join another Dataset or DataFrame."""
        right = other.df if isinstance(other, Dataset) else other
        return self._wrap(_transform.join(self.df, right, **kwargs))

    def pivot(self, **kwargs: Any) -> "Dataset":
        """Long to wide."""
        return self._wrap(_transform.pivot(self.df, **kwargs))

    def unpivot(self, **kwargs: Any) -> "Dataset":
        """Wide to long."""
        return self._wrap(_transform.unpivot(self.df, **kwargs))

    def bin(self, column: str, **kwargs: Any) -> "Dataset":
        """Bucket a numeric column into ranges."""
        return self._wrap(_transform.bin_column(self.df, column, **kwargs))

    def top_n(self, column: str, n: int = 10, **kwargs: Any) -> "Dataset":
        """The ``n`` rows with the largest values in ``column``."""
        return self._wrap(_transform.top_n(self.df, column, n, **kwargs))

    def value_counts(self, column: str, **kwargs: Any) -> "Dataset":
        """Frequency table for one column."""
        return self._wrap(_transform.value_counts(self.df, column, **kwargs))

    def resample(self, **kwargs: Any) -> "Dataset":
        """Roll a time series up to a period."""
        return self._wrap(_transform.resample_time(self.df, **kwargs))

    def rolling(self, column: str, **kwargs: Any) -> "Dataset":
        """Add a rolling-window statistic."""
        return self._wrap(_transform.rolling(self.df, column, **kwargs))

    def cumulative(self, column: str, **kwargs: Any) -> "Dataset":
        """Add a running total."""
        return self._wrap(_transform.cumulative(self.df, column, **kwargs))

    def rank(self, column: str, **kwargs: Any) -> "Dataset":
        """Add a rank column."""
        return self._wrap(_transform.rank(self.df, column, **kwargs))

    def steps(self, steps: Iterable[dict[str, Any]]) -> "Dataset":
        """Run a list of operations described as dicts (the pipeline format)."""
        return self._wrap(_transform.apply_steps(self.df, steps))

    def apply(self, func) -> "Dataset":
        """Apply any ``DataFrame -> DataFrame`` function."""
        return self._wrap(func(self.df))

    # -------------------------------------------------------------- analysis

    def profile(self, **kwargs: Any) -> _profiling.Profile:
        """Full profile: shape, per-column stats, correlations, quality score."""
        return _profiling.profile(self.df, **kwargs)

    def describe(self) -> pd.DataFrame:
        """Compact per-column overview."""
        return _profiling.describe(self.df)

    def missing(self) -> pd.DataFrame:
        """Null counts per column, worst first."""
        return _profiling.missing_summary(self.df)

    def correlations(self, **kwargs: Any) -> pd.DataFrame | None:
        """Correlation matrix of the numeric columns."""
        return _profiling.correlations(self.df, **kwargs)

    def validate(self, schema: dict | str) -> _validate.ValidationReport:
        """Check the data against a schema."""
        return _validate.validate(self.df, schema)

    def infer_schema(self, **kwargs: Any) -> dict:
        """Build a starter schema from the current data."""
        return _validate.infer_schema(self.df, **kwargs)

    # ---------------------------------------------------------------- charts

    def chart(self, kind: str, *args: Any, **kwargs: Any) -> str:
        """Render one chart: ``hist``, ``bar``, ``line``, ``scatter``, ``box``,
        ``correlations`` or ``missing``. Returns the saved path."""
        charts = {
            "hist": _viz.histogram, "histogram": _viz.histogram,
            "bar": _viz.bar, "line": _viz.line, "scatter": _viz.scatter,
            "box": _viz.box, "correlations": _viz.correlation_heatmap,
            "heatmap": _viz.correlation_heatmap, "missing": _viz.missing_matrix,
        }
        if kind not in charts:
            raise ValueError(f"unknown chart {kind!r}; try one of {sorted(charts)}")
        return charts[kind](self.df, *args, **kwargs)

    def auto_charts(self, outdir: str = "charts", **kwargs: Any) -> list[str]:
        """Render a sensible set of charts for this data."""
        return _viz.auto_charts(self.df, outdir, **kwargs)

    # ----------------------------------------------------------------- misc

    def head(self, n: int = 5) -> pd.DataFrame:
        """First ``n`` rows as a plain DataFrame."""
        return self.df.head(n)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def _wrap(self, df: pd.DataFrame) -> "Dataset":
        return Dataset(df, name=self.name)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key):
        result = self.df[key]
        return Dataset(result, name=self.name) if isinstance(result, pd.DataFrame) else result

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        rows, columns = self.df.shape
        return f"<Dataset{label} {rows:,} rows x {columns} columns>"

    def _repr_html_(self) -> str:  # notebook display
        return f"<p>{self!r}</p>" + self.df.head().to_html()
