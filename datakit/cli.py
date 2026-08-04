"""Command line interface.

    datakit info      sales.csv
    datakit profile   sales.csv --json out/profile.json
    datakit clean     sales.csv -o clean.parquet --missing median --outliers iqr
    datakit validate  sales.csv --schema schema.yaml
    datakit query     sales.csv --filter "amount > 100" --group-by region --agg amount:sum
    datakit chart     sales.csv --kind bar --x region --y amount -o charts/region.png
    datakit convert   sales.csv warehouse.db#sales
    datakit run       pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from . import cleaning as _clean
from . import io as _io
from . import pipeline as _pipeline
from . import profiling as _profiling
from . import transform as _transform
from . import validation as _validate
from . import viz as _viz


def _print_frame(df: pd.DataFrame, *, max_rows: int = 30) -> None:
    """Print a DataFrame as a table, using tabulate when it is available."""
    shown = df.head(max_rows)
    try:
        from tabulate import tabulate

        print(tabulate(shown, headers="keys", tablefmt="simple", showindex=False,
                       floatfmt=",.2f"))
    except ImportError:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(shown.to_string(index=False))
    if len(df) > len(shown):
        print(f"... {len(df) - len(shown):,} more row(s)")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    options: dict[str, Any] = {}
    for item in getattr(args, "option", None) or []:
        key, _, value = item.partition("=")
        options[key.strip()] = _coerce(value.strip())
    if getattr(args, "query_sql", None):
        options["query"] = args.query_sql
    if getattr(args, "sheet", None):
        options["sheet_name"] = _coerce(args.sheet)
    return _io.load(args.source, **options)


def _coerce(text: str) -> Any:
    """Turn a CLI string into an int/float/bool/None where that is unambiguous."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def _parse_agg(items: list[str] | None) -> dict[str, list[str]]:
    """``["amount:sum", "amount:mean", "qty:max"]`` -> ``{"amount": ["sum","mean"], ...}``."""
    aggregations: dict[str, list[str]] = {}
    for item in items or []:
        column, _, func = item.partition(":")
        aggregations.setdefault(column.strip(), []).append((func or "sum").strip())
    return aggregations


def _parse_pairs(items: list[str] | None) -> dict[str, str]:
    """``["margin=revenue - cost"]`` -> ``{"margin": "revenue - cost"}``."""
    pairs: dict[str, str] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def _emit(df: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write to --output if given, otherwise print."""
    if getattr(args, "output", None):
        target = _io.save(df, args.output)
        print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to {target}")
    else:
        _print_frame(df, max_rows=getattr(args, "max_rows", 30))


# ------------------------------------------------------------------ commands


def cmd_info(args: argparse.Namespace) -> int:
    df = _load(args)
    print(f"{args.source}")
    print(f"  {len(df):,} rows x {df.shape[1]} columns   "
          f"{_profiling._human_bytes(df.memory_usage(deep=True).sum())}")
    print()
    _print_frame(_profiling.describe(df).reset_index(), max_rows=100)
    print()
    print("First rows:")
    _print_frame(df.head(args.rows))
    return 0


def cmd_head(args: argparse.Namespace) -> int:
    df = _load(args)
    _print_frame(df.head(args.rows), max_rows=args.rows)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    df = _load(args)
    prof = _profiling.profile(df)
    if args.json:
        prof.to_json(args.json)
        print(f"Wrote profile to {args.json}")
    if args.charts:
        made = _viz.auto_charts(df, args.charts, dark=args.dark)
        print(f"Rendered {len(made)} chart(s) into {args.charts}")
    if not args.quiet:
        print(prof.to_text(max_columns=args.max_columns))
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    df = _load(args)
    report = _clean.CleaningReport()
    types = _parse_pairs(args.type)
    out = _clean.clean(
        df,
        missing=args.missing,
        types=types or None,
        infer_types=not args.no_infer_types,
        drop_dupes=not args.keep_duplicates,
        standardize=not args.keep_column_names,
        outliers=args.outliers,
        outlier_factor=args.outlier_factor,
        report=report,
    )
    print(f"Cleaned {len(df):,} -> {len(out):,} rows")
    print(report.to_text())
    _emit(out, args)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    df = _load(args)
    if args.infer_types:
        # CSV and JSON carry no dtypes, so 'type' rules would fail on text alone.
        df = _clean.coerce_types(df, infer=True)
    report = _validate.validate(df, args.schema)
    print(report.to_text())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2, default=str),
                                   encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0 if report.ok else 1


def cmd_infer_schema(args: argparse.Namespace) -> int:
    df = _load(args)
    schema = _validate.infer_schema(df)
    if args.output:
        print(f"Wrote {_validate.save_schema(schema, args.output)}")
    else:
        print(json.dumps(schema, indent=2, default=str))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    df = _load(args)
    steps: list[dict[str, Any]] = []
    if args.filter:
        for expression in args.filter:
            steps.append({"op": "filter", "query": expression})
    if args.derive:
        steps.append({"op": "derive", "expressions": _parse_pairs(args.derive)})
    if args.group_by:
        steps.append({"op": "group_by", "by": args.group_by,
                      "aggregations": _parse_agg(args.agg) or None})
    elif args.agg:
        steps.append({"op": "aggregate", "aggregations": _parse_agg(args.agg)})
    if args.select:
        steps.append({"op": "select", "columns": args.select})
    if args.sort:
        steps.append({"op": "sort", "by": args.sort, "ascending": not args.desc})
    if args.limit:
        steps.append({"op": "limit", "n": args.limit})

    out = _transform.apply_steps(df, steps)
    _emit(out, args)
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    df = _load(args)
    target = _io.save(df, args.target)
    print(f"Converted {args.source} -> {target} ({len(df):,} rows x {df.shape[1]} columns)")
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    df = _load(args)
    if args.kind == "auto":
        made = _viz.auto_charts(df, args.output or "charts", dark=args.dark)
        print("\n".join(f"Wrote {path}" for path in made))
        return 0

    needs_x = {"hist", "histogram", "bar", "line", "scatter", "box"}
    needs_y = {"line", "scatter"}
    if args.kind in needs_x and not args.x:
        print(f"error: --kind {args.kind} needs --x", file=sys.stderr)
        return 2
    if args.kind in needs_y and not args.y:
        print(f"error: --kind {args.kind} needs --y", file=sys.stderr)
        return 2

    output = args.output or f"{args.kind}.png"
    common = {"path": output, "dark": args.dark, "title": args.title}
    if args.kind in ("hist", "histogram"):
        path = _viz.histogram(df, args.x, bins=args.bins, **common)
    elif args.kind == "bar":
        path = _viz.bar(df, args.x, args.y, top=args.top, **common)
    elif args.kind == "line":
        path = _viz.line(df, args.x, args.y.split(","), **common)
    elif args.kind == "scatter":
        path = _viz.scatter(df, args.x, args.y, color_by=args.color_by, **common)
    elif args.kind == "box":
        path = _viz.box(df, args.x, by=args.color_by, **common)
    elif args.kind in ("correlations", "heatmap"):
        path = _viz.correlation_heatmap(df, **common)
    elif args.kind == "missing":
        path = _viz.missing_matrix(df, **common)
    else:
        print(f"Unknown chart kind: {args.kind}", file=sys.stderr)
        return 2
    print(f"Wrote {path}")
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    if args.list_tables:
        tables = _io.list_tables(args.url)
        print("\n".join(tables) if tables else "(no tables)")
        return 0
    if not args.query:
        print("Give --query or --list-tables", file=sys.stderr)
        return 2
    df = _io.read_sql(args.url, args.query)
    _emit(df, args)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    result = _pipeline.run_pipeline(args.pipeline, verbose=args.verbose)
    print(result.to_text())
    if result.validation is not None and not result.validation.ok:
        return 1
    return 0


def cmd_new_pipeline(args: argparse.Namespace) -> int:
    template = """# datakit pipeline
name: my pipeline
source: data/input.csv

clean:
  missing: median          # mean | median | mode | zero | drop | a literal
  outliers: null           # iqr | zscore | clip

steps:
  - op: filter
    query: "amount > 0"
  - op: group_by
    by: region
    aggregations:
      amount: [sum, mean]
  - op: sort
    by: amount_sum
    ascending: false

# validate: schema.yaml
# fail_on_validation_error: true

output:
  - out/result.csv

charts:
  outdir: out/charts
  auto: true

profile: out/profile.json
"""
    path = Path(args.output)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


# -------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datakit",
        description="Load, clean, validate, transform, analyse and chart tabular data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"datakit {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_source(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("source", help="file path, database URL (sqlite:///db.sqlite#table) "
                                        "or http(s) endpoint")
        sub.add_argument("--option", action="append", metavar="KEY=VALUE",
                         help="extra reader option, repeatable (e.g. --option sep=;)")
        sub.add_argument("--sheet", help="Excel sheet name or index")
        sub.add_argument("--query-sql", metavar="SQL", help="SQL to run against a database source")

    def add_output(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-o", "--output", help="write the result here instead of printing")
        sub.add_argument("--max-rows", type=int, default=30, help="rows to print (default 30)")

    info = subparsers.add_parser("info", help="shape, dtypes, null counts and a preview")
    add_source(info)
    info.add_argument("-n", "--rows", type=int, default=5)
    info.set_defaults(func=cmd_info)

    head = subparsers.add_parser("head", help="print the first rows")
    add_source(head)
    head.add_argument("-n", "--rows", type=int, default=10)
    head.set_defaults(func=cmd_head)

    profile = subparsers.add_parser("profile", help="full statistical profile and quality score")
    add_source(profile)
    profile.add_argument("--json", help="also write the profile as JSON")
    profile.add_argument("--charts", metavar="DIR", help="also render charts into DIR")
    profile.add_argument("--dark", action="store_true", help="dark-mode charts")
    profile.add_argument("--max-columns", type=int, default=None,
                         help="limit how many columns are described")
    profile.add_argument("-q", "--quiet", action="store_true", help="suppress the text report")
    profile.set_defaults(func=cmd_profile)

    clean = subparsers.add_parser("clean", help="standardize, de-duplicate, fix types, fill nulls")
    add_source(clean)
    add_output(clean)
    clean.add_argument("--missing", help="fill strategy: mean|median|mode|zero|drop|<literal>")
    clean.add_argument("--type", action="append", metavar="COL=TYPE",
                       help="force a column type, repeatable (e.g. --type joined=datetime)")
    clean.add_argument("--no-infer-types", action="store_true")
    clean.add_argument("--keep-duplicates", action="store_true")
    clean.add_argument("--keep-column-names", action="store_true",
                       help="do not snake_case the column names")
    clean.add_argument("--outliers", choices=["iqr", "zscore", "clip"],
                       help="drop or clip extreme numeric values")
    clean.add_argument("--outlier-factor", type=float, default=1.5)
    clean.set_defaults(func=cmd_clean)

    validate = subparsers.add_parser("validate", help="check data against a schema (exit 1 on error)")
    add_source(validate)
    validate.add_argument("--schema", required=True, help="schema .yaml/.json file")
    validate.add_argument("--json", help="write the report as JSON")
    validate.add_argument("--infer-types", action="store_true",
                          help="infer dtypes first — use for CSV/JSON, which store none")
    validate.set_defaults(func=cmd_validate)

    infer = subparsers.add_parser("infer-schema", help="draft a schema from trusted data")
    add_source(infer)
    infer.add_argument("-o", "--output", help="write to this .yaml/.json file")
    infer.set_defaults(func=cmd_infer_schema)

    query = subparsers.add_parser("query", help="filter, derive, group, sort, limit")
    add_source(query)
    add_output(query)
    query.add_argument("--select", help="comma-separated columns to keep")
    query.add_argument("--filter", action="append", metavar="EXPR",
                       help="pandas query expression, repeatable")
    query.add_argument("--derive", action="append", metavar="NAME=EXPR",
                       help="computed column, repeatable")
    query.add_argument("--group-by", metavar="COLS", help="comma-separated grouping columns")
    query.add_argument("--agg", action="append", metavar="COL:FUNC",
                       help="aggregation, repeatable (e.g. --agg amount:sum)")
    query.add_argument("--sort", metavar="COLS", help="comma-separated sort columns")
    query.add_argument("--desc", action="store_true", help="sort descending")
    query.add_argument("--limit", type=int, help="keep only N rows")
    query.set_defaults(func=cmd_query)

    convert = subparsers.add_parser("convert", help="move data between formats and databases")
    add_source(convert)
    convert.add_argument("target", help="destination file or sqlite:///db.sqlite#table")
    convert.set_defaults(func=cmd_convert)

    chart = subparsers.add_parser("chart", help="render a chart")
    add_source(chart)
    chart.add_argument("--kind", default="auto",
                       choices=["auto", "hist", "histogram", "bar", "line", "scatter", "box",
                                "correlations", "heatmap", "missing"])
    chart.add_argument("--x", help="x column (or the column to chart)")
    chart.add_argument("--y", help="y column(s), comma-separated for line charts")
    chart.add_argument("--color-by", help="split by this categorical column")
    chart.add_argument("--bins", type=int, default=30)
    chart.add_argument("--top", type=int, default=15, help="bars to show")
    chart.add_argument("--title")
    chart.add_argument("--dark", action="store_true")
    chart.add_argument("-o", "--output", help="output png (or directory for --kind auto)")
    chart.set_defaults(func=cmd_chart)

    sql = subparsers.add_parser("sql", help="run SQL against a database")
    sql.add_argument("url", help="SQLAlchemy URL, e.g. sqlite:///warehouse.db")
    sql.add_argument("--query", help="SQL to run")
    sql.add_argument("--list-tables", action="store_true")
    add_output(sql)
    sql.set_defaults(func=cmd_sql)

    run = subparsers.add_parser("run", help="run a pipeline file")
    run.add_argument("pipeline", help="pipeline .yaml/.json")
    run.add_argument("-v", "--verbose", action="store_true")
    run.set_defaults(func=cmd_run)

    new = subparsers.add_parser("new-pipeline", help="write a starter pipeline file")
    new.add_argument("-o", "--output", default="pipeline.yaml")
    new.add_argument("--force", action="store_true")
    new.set_defaults(func=cmd_new_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (_io.LoadError, _transform.TransformError, _pipeline.PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
