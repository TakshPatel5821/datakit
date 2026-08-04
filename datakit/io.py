"""Reading and writing data: files, SQL databases and HTTP APIs.

Everything goes through :func:`load` and :func:`save`, which pick the right
backend from the source string:

    load("sales.csv")                     # csv / tsv / txt
    load("sales.xlsx", sheet_name="Q1")   # excel
    load("events.jsonl")                  # json / ndjson
    load("big.parquet")                   # parquet
    load("sqlite:///shop.db#customers")   # a table
    load("sqlite:///shop.db", query="select * from orders where total > 100")
    load("https://api.example.com/users", record_path="data")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

__all__ = [
    "load",
    "save",
    "read_sql",
    "write_sql",
    "list_tables",
    "read_api",
    "source_kind",
]

#: Extensions we know how to read, mapped to a short format name.
_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
}


class LoadError(RuntimeError):
    """Raised when a source cannot be read."""


def source_kind(source: str) -> str:
    """Classify a source string as ``file``, ``sql`` or ``api``."""
    parsed = urlparse(str(source))
    if parsed.scheme in ("http", "https"):
        return "api"
    # A scheme longer than one character means a database URL. (Single-character
    # schemes are Windows drive letters: "C:\\data\\sales.csv".)
    if len(parsed.scheme) > 1:
        return "sql"
    return "file"


def _file_format(path: Path, override: str | None = None) -> str:
    if override:
        return override
    fmt = _EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        raise LoadError(
            f"Unknown file type {path.suffix!r} for {path}. "
            f"Pass format= explicitly (one of {sorted(set(_EXTENSIONS.values()))})."
        )
    return fmt


def _split_fragment(source: str) -> tuple[str, str | None]:
    """Split ``sqlite:///shop.db#customers`` into url and table name."""
    if "#" in source:
        url, _, fragment = source.rpartition("#")
        return url, fragment or None
    return source, None


def load(source: str, *, format: str | None = None, **kwargs: Any) -> pd.DataFrame:
    """Load any supported source into a DataFrame.

    Extra keyword arguments are forwarded to the underlying reader, so
    ``load("a.csv", sep=";", nrows=100)`` works as expected.
    """
    kind = source_kind(source)
    if kind == "api":
        return read_api(source, **kwargs)
    if kind == "sql":
        url, table = _split_fragment(str(source))
        query = kwargs.pop("query", None)
        table = kwargs.pop("table", table)
        return read_sql(url, query or table, **kwargs)

    path = Path(source)
    if not path.exists():
        raise LoadError(f"File not found: {path}")

    fmt = _file_format(path, format)
    if fmt == "csv":
        sep = kwargs.pop("sep", "\t" if path.suffix.lower() == ".tsv" else None)
        # sep=None with the python engine sniffs the delimiter automatically.
        engine = kwargs.pop("engine", "python" if sep is None else "c")
        return pd.read_csv(path, sep=sep, engine=engine, **kwargs)
    if fmt == "excel":
        return pd.read_excel(path, **kwargs)
    if fmt == "json":
        return _read_json_file(path, **kwargs)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(path, **kwargs)
    if fmt == "feather":
        return pd.read_feather(path, **kwargs)
    raise LoadError(f"Unsupported format: {fmt}")


def _read_json_file(path: Path, *, record_path: str | None = None, **kwargs: Any) -> pd.DataFrame:
    """Read a JSON file, digging into ``record_path`` when the rows are nested."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _json_to_frame(payload, record_path=record_path, **kwargs)


def _json_to_frame(payload: Any, *, record_path: str | None = None, **kwargs: Any) -> pd.DataFrame:
    if record_path:
        for key in record_path.split("."):
            if not isinstance(payload, dict) or key not in payload:
                raise LoadError(f"record_path {record_path!r} not found in the response")
            payload = payload[key]
    if isinstance(payload, dict):
        # A bare object: either a wrapper around a list, or a single record.
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) == 1:
            payload = lists[0]
        else:
            payload = [payload]
    return pd.json_normalize(payload, **kwargs)


def read_api(url: str, *, record_path: str | None = None, params: dict | None = None,
             headers: dict | None = None, timeout: float = 30.0, **kwargs: Any) -> pd.DataFrame:
    """Fetch a URL and turn the response into a DataFrame.

    JSON responses are flattened with :func:`pandas.json_normalize`; CSV
    responses are parsed as CSV.
    """
    import requests

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")

    if "csv" in content_type or url.endswith(".csv"):
        from io import StringIO

        return pd.read_csv(StringIO(response.text), **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:  # not JSON either
        raise LoadError(f"{url} returned {content_type!r}, which is not JSON or CSV") from exc
    return _json_to_frame(payload, record_path=record_path, **kwargs)


def _engine(url: str):
    from sqlalchemy import create_engine

    return create_engine(url)


def read_sql(url: str, query_or_table: str | None = None, **kwargs: Any) -> pd.DataFrame:
    """Run a query (or read a whole table) from any SQLAlchemy-supported database."""
    if not query_or_table:
        tables = list_tables(url)
        raise LoadError(
            f"No table or query given for {url}. Available tables: {', '.join(tables) or '(none)'}"
        )
    engine = _engine(url)
    looks_like_sql = query_or_table.strip().lower().startswith(("select", "with"))
    with engine.connect() as connection:
        if looks_like_sql:
            from sqlalchemy import text

            return pd.read_sql_query(text(query_or_table), connection, **kwargs)
        return pd.read_sql_table(query_or_table, connection, **kwargs)


def write_sql(df: pd.DataFrame, url: str, table: str, *, if_exists: str = "replace",
              index: bool = False, **kwargs: Any) -> int:
    """Write a DataFrame to a database table. Returns the number of rows written."""
    engine = _engine(url)
    with engine.begin() as connection:
        df.to_sql(table, connection, if_exists=if_exists, index=index, **kwargs)
    return len(df)


def list_tables(url: str) -> list[str]:
    """List the tables visible in a database."""
    from sqlalchemy import inspect

    return sorted(inspect(_engine(url)).get_table_names())


def save(df: pd.DataFrame, target: str, *, format: str | None = None, **kwargs: Any) -> str:
    """Write a DataFrame to a file or database table. Returns the target written."""
    kind = source_kind(target)
    if kind == "api":
        raise LoadError("Writing to an HTTP endpoint is not supported")
    if kind == "sql":
        url, table = _split_fragment(str(target))
        table = kwargs.pop("table", table)
        if not table:
            raise LoadError("No table name given; use sqlite:///file.db#tablename")
        write_sql(df, url, table, **kwargs)
        return target

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _file_format(path, format)
    index = kwargs.pop("index", False)

    if fmt == "csv":
        sep = kwargs.pop("sep", "\t" if path.suffix.lower() == ".tsv" else ",")
        df.to_csv(path, sep=sep, index=index, **kwargs)
    elif fmt == "excel":
        df.to_excel(path, index=index, **kwargs)
    elif fmt == "json":
        df.to_json(path, orient=kwargs.pop("orient", "records"), indent=kwargs.pop("indent", 2),
                   date_format=kwargs.pop("date_format", "iso"), **kwargs)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, date_format="iso", **kwargs)
    elif fmt == "parquet":
        df.to_parquet(path, index=index, **kwargs)
    elif fmt == "feather":
        df.reset_index(drop=True).to_feather(path, **kwargs)
    else:
        raise LoadError(f"Unsupported format: {fmt}")
    return str(path)
