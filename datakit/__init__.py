"""datakit — data management, operations and analysis for tabular data.

Load anything (CSV, Excel, JSON, Parquet, SQL, HTTP APIs), clean and validate it,
run operations on it, profile it, and chart it — from Python or the command line.

    from datakit import Dataset

    data = Dataset.read("sales.csv").clean(missing="median")
    print(data.profile())
    data.filter("amount > 0").group_by("region", {"amount": "sum"}).save("out.xlsx")

Or from a shell::

    datakit profile sales.csv
    datakit query sales.csv --group-by region --agg amount:sum -o by_region.csv
"""

from __future__ import annotations

__version__ = "0.1.0"

from .cleaning import (
    clean,
    coerce_types,
    drop_duplicates,
    drop_empty,
    fill_missing,
    remove_outliers,
    standardize_columns,
    strip_whitespace,
)
from .dataset import Dataset
from .io import list_tables, load, read_api, read_sql, save, write_sql
from .pipeline import run_pipeline
from .profiling import Profile, correlations, describe, missing_summary, profile, quality_score
from .transform import (
    aggregate,
    apply_steps,
    bin_column,
    derive,
    drop,
    filter_rows,
    group_by,
    join,
    limit,
    pivot,
    rename,
    resample_time,
    sample,
    select,
    sort,
    top_n,
    unpivot,
    value_counts,
)
from .validation import ValidationReport, infer_schema, validate
from .viz import auto_charts, bar, box, correlation_heatmap, histogram, line, missing_matrix, scatter

__all__ = [
    "__version__",
    # core
    "Dataset",
    # io
    "load", "save", "read_sql", "write_sql", "read_api", "list_tables",
    # clean
    "clean", "standardize_columns", "strip_whitespace", "drop_empty", "drop_duplicates",
    "fill_missing", "coerce_types", "remove_outliers",
    # validate
    "validate", "infer_schema", "ValidationReport",
    # profile
    "profile", "Profile", "describe", "correlations", "missing_summary", "quality_score",
    # transform
    "select", "drop", "rename", "filter_rows", "sort", "limit", "sample", "group_by",
    "aggregate", "join", "pivot", "unpivot", "derive", "bin_column", "value_counts",
    "top_n", "resample_time", "apply_steps",
    # viz
    "histogram", "bar", "line", "scatter", "box", "correlation_heatmap", "missing_matrix",
    "auto_charts",
    # pipeline
    "run_pipeline",
]
