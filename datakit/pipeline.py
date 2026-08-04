"""Declarative pipelines: describe a job in YAML/JSON, run it repeatedly.

    name: monthly sales
    source: data/sales.csv
    clean:
      missing: median
      outliers: iqr
    validate: schemas/sales.yaml
    steps:
      - op: filter
        query: "amount > 0"
      - op: derive
        expressions: {margin: "revenue - cost"}
      - op: group_by
        by: region
        aggregations: {margin: sum, amount: [sum, mean]}
      - op: sort
        by: margin_sum
        ascending: false
    output:
      - out/by_region.csv
      - sqlite:///out/warehouse.db#by_region
    charts:
      outdir: out/charts
      auto: true
    profile: out/profile.json

Run it with ``datakit run pipeline.yaml`` or :func:`run_pipeline`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import cleaning as _clean
from . import io as _io
from . import profiling as _profiling
from . import transform as _transform
from . import validation as _validate
from . import viz as _viz

__all__ = ["PipelineResult", "run_pipeline", "load_pipeline"]


class PipelineError(RuntimeError):
    """Raised when a pipeline is malformed or a required stage fails."""


@dataclass
class PipelineResult:
    """What a pipeline run produced."""

    name: str = "pipeline"
    rows_in: int = 0
    rows_out: int = 0
    columns_out: int = 0
    steps_run: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    cleaning: dict = field(default_factory=dict)
    validation: _validate.ValidationReport | None = None
    profile_path: str | None = None
    seconds: float = 0.0
    df: pd.DataFrame | None = None

    def to_text(self) -> str:
        lines = [f"Pipeline '{self.name}' finished in {self.seconds:.2f}s",
                 f"  rows {self.rows_in:,} -> {self.rows_out:,} ({self.columns_out} columns)"]
        for key, value in self.cleaning.items():
            lines.append(f"  clean: {key} — {value}")
        for step in self.steps_run:
            lines.append(f"  step: {step}")
        if self.validation is not None:
            status = "passed" if self.validation.ok else "FAILED"
            lines.append(f"  validation {status}: {len(self.validation.errors)} error(s), "
                         f"{len(self.validation.warnings)} warning(s)")
            for issue in self.validation.issues[:10]:
                lines.append(f"      {issue}")
        for output in self.outputs:
            lines.append(f"  wrote: {output}")
        for chart in self.charts:
            lines.append(f"  chart: {chart}")
        if self.profile_path:
            lines.append(f"  profile: {self.profile_path}")
        return "\n".join(lines)


def load_pipeline(path: str | Path) -> dict:
    """Read a pipeline definition from YAML or JSON."""
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Pipeline file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise PipelineError(f"{path} does not contain a pipeline object")
    return config


def run_pipeline(config: dict | str | Path, *, verbose: bool = False) -> PipelineResult:
    """Execute a pipeline definition and return what it did."""
    if isinstance(config, (str, Path)):
        config = load_pipeline(config)

    started = time.perf_counter()
    result = PipelineResult(name=str(config.get("name", "pipeline")))

    source = config.get("source")
    if not source:
        raise PipelineError("pipeline needs a 'source'")
    source_options = dict(config.get("options") or {})
    if isinstance(source, dict):
        source_options.update({k: v for k, v in source.items() if k != "path"})
        source = source["path"]

    df = _io.load(str(source), **source_options)
    result.rows_in = len(df)
    _say(verbose, f"loaded {len(df):,} rows from {source}")

    # Optional joins with other sources, before the row-level work.
    for spec in config.get("join") or []:
        spec = dict(spec)
        right = _io.load(str(spec.pop("source")), **(spec.pop("options", None) or {}))
        df = _transform.join(df, right, **spec)
        result.steps_run.append(f"join ({len(df):,} rows)")

    clean_config = config.get("clean")
    if clean_config:
        report = _clean.CleaningReport()
        options = {} if clean_config is True else dict(clean_config)
        df = _clean.clean(df, report=report, **options)
        result.cleaning = dict(report)
        _say(verbose, f"cleaned -> {len(df):,} rows")

    steps = config.get("steps") or []
    for index, step in enumerate(steps, start=1):
        label = step.get("name") or step.get("op") or f"step {index}"
        df = _transform.apply_steps(df, [step])
        result.steps_run.append(f"{label} ({len(df):,} rows)")
        _say(verbose, f"{label} -> {len(df):,} rows")

    schema = config.get("validate")
    if schema:
        result.validation = _validate.validate(df, schema)
        _say(verbose, result.validation.to_text())
        if not result.validation.ok and config.get("fail_on_validation_error", False):
            raise PipelineError(
                f"validation failed with {len(result.validation.errors)} error(s):\n"
                + result.validation.to_text()
            )

    result.rows_out, result.columns_out = len(df), df.shape[1]

    outputs = config.get("output") or []
    if isinstance(outputs, (str, dict)):
        outputs = [outputs]
    for output in outputs:
        options = {}
        if isinstance(output, dict):
            options = {k: v for k, v in output.items() if k != "path"}
            output = output["path"]
        result.outputs.append(_io.save(df, str(output), **options))
        _say(verbose, f"wrote {output}")

    charts = config.get("charts")
    if charts:
        result.charts = _render_charts(df, charts)
        _say(verbose, f"rendered {len(result.charts)} chart(s)")

    profile_target = config.get("profile")
    if profile_target:
        prof = _profiling.profile(df)
        path = Path(str(profile_target))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            prof.to_json(str(path))
        else:
            path.write_text(prof.to_text(), encoding="utf-8")
        result.profile_path = str(path)

    result.seconds = time.perf_counter() - started
    result.df = df
    return result


def _render_charts(df: pd.DataFrame, config: Any) -> list[str]:
    """Render either ``auto`` charts or an explicit list of chart specs."""
    if isinstance(config, list):
        config = {"list": config}
    outdir = Path(str(config.get("outdir", "charts")))
    dark = bool(config.get("dark", False))
    made: list[str] = []

    if config.get("auto"):
        made.extend(_viz.auto_charts(df, str(outdir), dark=dark))

    for spec in config.get("list") or []:
        spec = dict(spec)
        kind = spec.pop("kind", None)
        if not kind:
            raise PipelineError("each chart needs a 'kind'")
        renderer = {
            "hist": _viz.histogram, "histogram": _viz.histogram, "bar": _viz.bar,
            "line": _viz.line, "scatter": _viz.scatter, "box": _viz.box,
            "correlations": _viz.correlation_heatmap, "missing": _viz.missing_matrix,
        }.get(kind)
        if renderer is None:
            raise PipelineError(f"unknown chart kind {kind!r}")
        spec.setdefault("dark", dark)
        spec["path"] = str(outdir / spec.get("path", f"{kind}.png"))
        made.append(renderer(df, **spec))
    return made


def _say(verbose: bool, message: str) -> None:
    if verbose:
        print(f"  {message}")
