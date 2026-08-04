"""Schema and data-quality validation.

A schema is plain data (dict, JSON or YAML), so it can live next to the data::

    columns:
      order_id:  {type: int, required: true, unique: true}
      email:     {type: str, nullable: false, regex: '^[^@]+@[^@]+\\.[^@]+$'}
      quantity:  {type: int, min: 1, max: 1000}
      status:    {type: str, allowed: [new, shipped, cancelled]}
    strict: false          # true -> unexpected columns are errors
    min_rows: 1
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

__all__ = ["Issue", "ValidationReport", "validate", "infer_schema", "load_schema", "save_schema"]

_TYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "integer": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_float_dtype,
    "number": pd.api.types.is_numeric_dtype,
    "numeric": pd.api.types.is_numeric_dtype,
    "str": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
    "bool": pd.api.types.is_bool_dtype,
    "boolean": pd.api.types.is_bool_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "date": pd.api.types.is_datetime64_any_dtype,
    "category": lambda s: isinstance(s.dtype, pd.CategoricalDtype),
}


@dataclass
class Issue:
    """One rule violation."""

    level: str  # "error" or "warning"
    column: str | None
    rule: str
    message: str
    count: int = 0
    examples: list[Any] = field(default_factory=list)

    def __str__(self) -> str:
        where = f"[{self.column}] " if self.column else ""
        detail = f" ({self.count} row(s))" if self.count else ""
        sample = f" e.g. {self.examples}" if self.examples else ""
        return f"{self.level.upper():<7} {where}{self.message}{detail}{sample}"


@dataclass
class ValidationReport:
    """The result of :func:`validate`."""

    issues: list[Issue] = field(default_factory=list)
    rows: int = 0
    checked_columns: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing failed at error level."""
        return not self.errors

    def to_text(self) -> str:
        header = (
            f"Validation: {len(self.errors)} error(s), {len(self.warnings)} warning(s) "
            f"across {self.rows} row(s) and {self.checked_columns} checked column(s)"
        )
        if not self.issues:
            return header + "\n  All checks passed."
        lines = [header] + [f"  {issue}" for issue in self.issues]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "checked_columns": self.checked_columns,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [issue.__dict__ for issue in self.issues],
        }

    def __bool__(self) -> bool:
        return self.ok


def load_schema(path: str | Path) -> dict:
    """Read a schema from a ``.yaml``/``.yml`` or ``.json`` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def save_schema(schema: dict, path: str | Path) -> str:
    """Write a schema to YAML or JSON, chosen by extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    return str(path)


def infer_schema(df: pd.DataFrame, *, max_allowed: int = 15) -> dict:
    """Build a starter schema from data you trust. Edit it, then check future loads against it."""
    columns: dict[str, dict] = {}
    for name in df.columns:
        series = df[name]
        rule: dict[str, Any] = {"type": _dtype_name(series)}
        rule["nullable"] = bool(series.isna().any())
        if series.notna().any() and series.nunique(dropna=True) == series.notna().sum():
            rule["unique"] = True
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            rule["min"] = _scalar(series.min())
            rule["max"] = _scalar(series.max())
        elif rule["type"] == "str" and 0 < series.nunique(dropna=True) <= max_allowed:
            rule["allowed"] = sorted(str(value) for value in series.dropna().unique())
        columns[str(name)] = rule
    return {"columns": columns, "strict": False, "min_rows": 1}


def _dtype_name(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "str"


def _scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python so schemas serialize cleanly."""
    return value.item() if hasattr(value, "item") else value


def _examples(series: pd.Series, mask: pd.Series, limit: int = 3) -> list[Any]:
    values = series[mask.fillna(False)].dropna().unique()[:limit]
    return [_scalar(value) for value in values]


def validate(df: pd.DataFrame, schema: dict | str | Path) -> ValidationReport:
    """Check a DataFrame against a schema and collect every violation."""
    if isinstance(schema, (str, Path)):
        schema = load_schema(schema)

    rules: dict[str, dict] = schema.get("columns", {}) or {}
    report = ValidationReport(rows=len(df), checked_columns=len(rules))
    issues = report.issues

    min_rows = schema.get("min_rows")
    if min_rows is not None and len(df) < min_rows:
        issues.append(Issue("error", None, "min_rows",
                            f"expected at least {min_rows} row(s), found {len(df)}"))

    if schema.get("strict"):
        unexpected = [str(c) for c in df.columns if str(c) not in rules]
        if unexpected:
            issues.append(Issue("error", None, "strict",
                                f"unexpected column(s): {', '.join(unexpected)}"))

    if schema.get("no_duplicate_rows", True):
        duplicates = int(df.duplicated().sum())
        if duplicates:
            issues.append(Issue("warning", None, "duplicate_rows",
                                "duplicate rows present", duplicates))

    for column, rule in rules.items():
        rule = rule or {}
        if column not in df.columns:
            level = "error" if rule.get("required", True) else "warning"
            issues.append(Issue(level, column, "missing_column", "column is missing"))
            continue

        series = df[column]
        nulls = int(series.isna().sum())

        expected = rule.get("type")
        if expected:
            check = _TYPE_CHECKS.get(str(expected).lower())
            if check is None:
                issues.append(Issue("warning", column, "type",
                                    f"unknown type {expected!r} in schema"))
            elif not check(series):
                issues.append(Issue("error", column, "type",
                                    f"expected {expected}, found {series.dtype}"))

        if rule.get("nullable") is False and nulls:
            issues.append(Issue("error", column, "nullable", "nulls not allowed", nulls,
                                []))
        elif nulls and rule.get("max_null_fraction") is not None:
            fraction = nulls / max(len(df), 1)
            if fraction > rule["max_null_fraction"]:
                issues.append(Issue("error", column, "max_null_fraction",
                                    f"{fraction:.1%} null exceeds "
                                    f"{rule['max_null_fraction']:.1%}", nulls))

        if rule.get("unique"):
            duplicated = series.duplicated(keep=False) & series.notna()
            if duplicated.any():
                issues.append(Issue("error", column, "unique", "values must be unique",
                                    int(duplicated.sum()), _examples(series, duplicated)))

        if "min" in rule or "max" in rule:
            numeric = pd.to_numeric(series, errors="coerce")
            if "min" in rule:
                below = numeric < rule["min"]
                if below.any():
                    issues.append(Issue("error", column, "min",
                                        f"values below minimum {rule['min']}",
                                        int(below.sum()), _examples(series, below)))
            if "max" in rule:
                above = numeric > rule["max"]
                if above.any():
                    issues.append(Issue("error", column, "max",
                                        f"values above maximum {rule['max']}",
                                        int(above.sum()), _examples(series, above)))

        if "allowed" in rule:
            allowed = set(map(str, rule["allowed"]))
            invalid = series.notna() & ~series.astype(str).isin(allowed)
            if invalid.any():
                issues.append(Issue("error", column, "allowed",
                                    f"values outside {sorted(allowed)}",
                                    int(invalid.sum()), _examples(series, invalid)))

        if "regex" in rule:
            pattern = re.compile(rule["regex"])
            text = series.dropna().astype(str)
            bad = ~text.str.fullmatch(pattern)
            if bad.any():
                invalid = pd.Series(False, index=series.index)
                invalid.loc[text.index] = bad
                issues.append(Issue("error", column, "regex",
                                    f"values do not match {rule['regex']!r}",
                                    int(bad.sum()), _examples(series, invalid)))

        if "length" in rule or "max_length" in rule:
            limit = rule.get("max_length", rule.get("length"))
            lengths = series.dropna().astype(str).str.len()
            too_long = lengths > limit
            if too_long.any():
                issues.append(Issue("error", column, "max_length",
                                    f"values longer than {limit} characters",
                                    int(too_long.sum())))

    return report


def check_columns(df: pd.DataFrame, required: Iterable[str]) -> ValidationReport:
    """Shorthand: assert a set of columns exists."""
    return validate(df, {"columns": {name: {"required": True} for name in required}})
