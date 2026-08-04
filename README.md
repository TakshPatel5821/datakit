# datakit

Data management, data operations and analysis for tabular data — a Python library
and a command line tool over the same engine.

- **Read anything** — CSV, TSV, Excel, JSON, JSONL, Parquet, Feather, any
  SQLAlchemy database, and HTTP/JSON APIs.
- **Clean and validate** — standardize names, fix types, fill nulls, drop
  duplicates and outliers; then check the result against a schema.
- **Operate** — filter, derive, sort, group, aggregate, join, pivot, resample,
  bin, rank, roll.
- **Analyse** — full statistical profile, correlations, missing-value map and a
  0–100 data-quality score.
- **Chart** — histograms, bars, lines, scatter, box plots, correlation heatmaps,
  from one validated color palette.
- **Repeat** — describe the whole job in a YAML pipeline and run it on a schedule.

---

## Install

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -e .
```

On macOS/Linux the activate line is `source .venv/bin/activate`. That installs
the `datakit` command; `python -m datakit` works too.

## 60-second tour

```bash
python examples/make_sample_data.py              # writes a deliberately messy CSV
datakit info examples/data/sales_raw.csv         # shape, dtypes, nulls, preview
datakit profile examples/data/sales_raw.csv      # full stats + quality score
datakit run examples/sales_pipeline.yaml -v      # clean -> transform -> save -> chart
```

---

## The library

```python
from datakit import Dataset

sales = (Dataset.read("sales.csv")
         .clean(missing="median", outliers="iqr")
         .filter("revenue > 0")
         .derive(margin="revenue - cost",
                 margin_pct="(revenue - cost) / revenue * 100"))

print(sales.profile())                   # readable profile report
print(sales.missing())                   # null counts per column

by_region = (sales
             .group_by("region", {"margin": "sum", "revenue": ["sum", "mean"]})
             .sort("margin_sum", ascending=False))

by_region.save("out/by_region.xlsx")
by_region.chart("bar", x="region", y="margin_sum", path="out/margin.png")
```

Every `Dataset` method returns a new `Dataset` — nothing mutates in place. The
underlying DataFrame is always `.df`, so you can drop into plain pandas whenever
you want and come back with `Dataset(df)`.

The module-level functions work directly on DataFrames if you prefer:

```python
import datakit

df = datakit.load("sqlite:///warehouse.db#orders")
df = datakit.clean(df, missing="median")
report = datakit.validate(df, "schema.yaml")
prof = datakit.profile(df)
datakit.auto_charts(df, "charts/")
datakit.save(df, "orders.parquet")
```

## Sources

`load()` and `save()` decide what to do from the source string:

| Source | Example |
|---|---|
| Flat files | `load("sales.csv")`, `load("sales.tsv", sep="\|")` |
| Excel | `load("book.xlsx", sheet_name="Q1")` |
| JSON / NDJSON | `load("events.jsonl")`, `load("nested.json", record_path="data.rows")` |
| Parquet / Feather | `load("big.parquet")` |
| Database table | `load("sqlite:///shop.db#customers")` |
| SQL query | `load("postgresql://user:pw@host/db", query="select * from orders")` |
| HTTP API | `load("https://api.example.com/users", record_path="data")` |

Any extra keyword goes straight to the underlying reader. `save()` mirrors it —
`save(df, "sqlite:///warehouse.db#daily")` writes a table, `save(df, "out.xlsx")`
writes a workbook.

## Cleaning

```python
from datakit import cleaning

cleaning.clean(df,
    standardize=True,        # "Total Sales ($)" -> total_sales
    strip=True,              # trim whitespace, blank strings become nulls
    drop_empties=True,       # drop all-null rows and columns
    drop_dupes=True,
    infer_types=True,        # "$1,200.50" -> 1200.5, "2024-03-02" -> datetime
    types={"joined": "datetime"},
    missing="median",        # or {"age": "mean", "city": "unknown"}
    outliers="iqr",          # or "zscore" / "clip"
)
```

Pass a `CleaningReport` to find out exactly what each step did:

```python
data = Dataset.read("raw.csv").clean(missing="median")
print(data.last_report.to_text())
#   standardize_columns  renamed 12 column(s)
#   drop_duplicates      removed 25 duplicate row(s)
#   coerce_types         converted order_date, revenue
#   fill_missing         filled 96 missing value(s)
```

Type inference is deliberately conservative: it strips currency and thousands
separators, and it leaves all-digit strings (zip codes, IDs) alone rather than
reading them as dates.

## Validation

A schema is plain YAML or JSON, so it lives in version control next to the data:

```yaml
min_rows: 1
strict: false                 # true -> unexpected columns are errors
columns:
  order_id:  {type: str, unique: true, nullable: false, regex: '^ORD-\d{6}$'}
  order_date: {type: datetime, nullable: false}
  region:    {type: str, allowed: [North, South, East, West, Central]}
  units_sold: {type: int, min: 1, max: 1000}
  channel:   {type: str, max_null_fraction: 0.05}
```

```bash
datakit validate sales.csv --schema schema.yaml --infer-types
```

```
Validation: 1 error(s), 0 warning(s) across 1200 row(s) and 10 checked column(s)
  ERROR   [units_sold] values above maximum 1000 (6 row(s)) e.g. [6232, 6447, 5885]
```

It exits `1` when anything fails, so it drops straight into CI or a scheduled job.
Use `--infer-types` for CSV and JSON, which store no dtypes of their own.

Don't have a schema yet? Generate one from data you trust and edit it down:

```bash
datakit infer-schema good_data.csv -o schema.yaml
```

## Operations

Every operation exists as a function, a `Dataset` method, and a pipeline step.

| Op | What it does |
|---|---|
| `select` / `drop` / `rename` | pick and rename columns |
| `filter` | `"amount > 100 and region == 'EU'"`, or `column`/`op`/`value` |
| `derive` | computed columns, evaluated in order |
| `sort` / `limit` / `sample` / `top_n` | ordering and slicing |
| `group_by` / `aggregate` | rollups, one or many functions per column |
| `join` | inner / left / right / outer / cross |
| `pivot` / `unpivot` | long ↔ wide |
| `bin` | equal-width or equal-frequency buckets |
| `resample` | roll a time series up to D/W/M/Q/Y |
| `rolling` / `cumulative` / `rank` | window and running calculations |

Aggregation output columns are **always** `<column>_<function>` —
`{"revenue": "sum"}` produces `revenue_sum`. Adding a second function later never
silently renames the first one out from under a downstream step.

## Profiling

```python
prof = Dataset.read("sales.csv").profile()
print(prof)                     # the text report
prof.to_json("profile.json")    # machine-readable
prof.to_frame()                 # one row per column, as a DataFrame
```

You get shape and memory, duplicate and null counts, per-column statistics
(quartiles, skew, zeros, outlier counts, top values, date ranges), a correlation
matrix with the strongest pairs called out, and a quality score with notes:

```
Quality score : 78/100 (good)
  ! 25 duplicate row(s)
  ! over half missing in: notes
  ! single-valued column(s): legacy_flag
```

## Charts

```python
from datakit import viz

viz.histogram(df, "revenue", path="charts/revenue.png")
viz.bar(df, "region", "revenue", path="charts/by_region.png")
viz.line(df, "date", ["revenue", "cost"], path="charts/trend.png")
viz.scatter(df, "revenue", "cost", color_by="category", path="charts/rc.png")
viz.correlation_heatmap(df, path="charts/corr.png")
viz.auto_charts(df, "charts/")          # picks a sensible set for this data
```

Pass `dark=True` to any of them for a dark-surface version.

The palette is fixed and validated rather than picked per chart: categorical hues
are assigned in a set order and never cycled (extra categories fold into a gray
"Other"), magnitude uses a single-hue ramp, and signed values like correlations
use a blue↔red diverging ramp through a neutral midpoint. There is deliberately
no dual-axis option — two scales on one plot invite false comparisons; chart the
second measure separately or index both to a common base.

## Pipelines

Describe the whole job once and run it whenever:

```yaml
name: sales by region
source: data/sales_raw.csv

clean:
  missing: median
  outliers: iqr
  outlier_factor: 3.0

steps:
  - op: filter
    query: "revenue > 0 and units_sold > 0"
  - op: derive
    expressions:
      margin: "revenue - cost"
  - op: group_by
    by: [region, category]
    aggregations:
      revenue: [sum, mean]
      margin: sum
  - op: sort
    by: revenue_sum
    ascending: false

validate: schemas/output.yaml
fail_on_validation_error: true

output:
  - out/by_region.csv
  - sqlite:///out/warehouse.db#by_region

charts:
  outdir: out/charts
  auto: true

profile: out/profile.json
```

```bash
datakit run pipeline.yaml -v
datakit new-pipeline -o pipeline.yaml     # scaffold a starter file
```

A run reports what it did and returns exit code `1` if validation failed, so it
composes with schedulers and CI.

## Command line

```
datakit info      <source>                    shape, dtypes, nulls, preview
datakit head      <source> -n 20
datakit profile   <source> [--json p.json] [--charts DIR]
datakit clean     <source> -o out.parquet [--missing median] [--outliers iqr]
datakit validate  <source> --schema s.yaml [--infer-types]      exit 1 on error
datakit infer-schema <source> -o schema.yaml
datakit query     <source> [--filter EXPR] [--derive N=EXPR] [--group-by COLS]
                           [--agg COL:FUNC] [--sort COLS] [--desc] [--limit N]
datakit convert   <source> <target>           any format or database to any other
datakit chart     <source> --kind bar --x region --y revenue -o chart.png
datakit sql       <url> --query "select ..." | --list-tables
datakit run       <pipeline.yaml> -v
```

Every command that reads data takes the same `<source>` forms as `load()`, plus
`--option key=value` for reader arguments and `--query-sql` for database sources.
Commands that produce a table print it, or write it with `-o`.

```bash
# ad-hoc analysis without writing any code
datakit query sales.csv --filter "revenue > 1000" \
    --derive "margin=revenue - cost" \
    --group-by region --agg margin:sum --agg revenue:mean \
    --sort margin_sum --desc
```

```
region      margin_sum    revenue_mean
--------  ------------  --------------
North       464,145.08        4,849.53
South       332,880.84        4,589.06
West        322,387.19        4,299.32
East        276,161.47        3,895.83
Central     251,022.71        4,508.01
```

## Layout

```
datakit/
  io.py          load / save across files, databases and APIs
  cleaning.py    cleaning operations and the standard clean() pipeline
  validation.py  schema rules, validation reports, schema inference
  transform.py   data operations + the declarative step runner
  profiling.py   statistics, correlations, quality score
  viz.py         charts and the color palette
  dataset.py     Dataset — the chainable API
  pipeline.py    YAML/JSON pipeline runner
  cli.py         the datakit command
examples/        sample-data generator, example pipeline and schema
tests/           60 tests covering every module
```

## Tests

```bash
pytest -q
```

Requires Python 3.10+. Verified on Python 3.13 with pandas 3.0.5; the code avoids
pandas-3-only APIs and declares `pandas>=2.0`, but only 3.0 has been run here.
