import json

import pandas as pd
import pytest

from datakit import Dataset, io as dio, pipeline as dp, viz
from datakit.cli import main


# ------------------------------------------------------------------------ io


@pytest.mark.parametrize("filename", ["out.csv", "out.tsv", "out.json", "out.jsonl",
                                      "out.parquet", "out.xlsx"])
def test_save_then_load_round_trips(tidy, tmp_path, filename):
    target = tmp_path / filename
    dio.save(tidy, str(target))
    out = dio.load(str(target))
    assert len(out) == len(tidy)
    assert set(out.columns) == set(tidy.columns)


def test_sqlite_round_trip(tidy, tmp_path):
    url = f"sqlite:///{(tmp_path / 'warehouse.db').as_posix()}"
    dio.write_sql(tidy, url, "sales")
    assert "sales" in dio.list_tables(url)

    whole = dio.load(f"{url}#sales")
    assert len(whole) == len(tidy)

    filtered = dio.read_sql(url, "select region, revenue from sales where revenue > 500")
    assert list(filtered.columns) == ["region", "revenue"]
    assert (filtered["revenue"] > 500).all()


def test_source_kind_classifies_windows_paths():
    assert dio.source_kind(r"C:\data\sales.csv") == "file"
    assert dio.source_kind("sqlite:///db.sqlite") == "sql"
    assert dio.source_kind("https://api.example.com/rows") == "api"


def test_load_missing_file_raises():
    with pytest.raises(dio.LoadError, match="not found"):
        dio.load("no_such_file.csv")


def test_load_unknown_extension_raises(tmp_path):
    path = tmp_path / "data.weird"
    path.write_text("x")
    with pytest.raises(dio.LoadError, match="Unknown file type"):
        dio.load(str(path))


# ------------------------------------------------------------------- Dataset


def test_dataset_chain(tidy, tmp_path):
    target = tmp_path / "by_region.csv"
    out = (Dataset(tidy)
           .clean()
           .filter("revenue > 100")
           .derive(margin="revenue - cost")
           .group_by("region", {"margin": "sum"})
           .sort("margin_sum", ascending=False))
    out.save(str(target))

    assert target.exists()
    assert out.columns == ["region", "margin_sum"]
    assert len(out) == tidy["region"].nunique()
    assert repr(out).startswith("<Dataset")


def test_dataset_profile_and_validate(tidy):
    data = Dataset(tidy)
    assert data.profile().summary["rows"] == len(tidy)
    assert data.validate(data.infer_schema()).ok


# -------------------------------------------------------------------- charts


def test_charts_render(tidy, tmp_path):
    assert viz.histogram(tidy, "revenue", path=tmp_path / "h.png")
    assert viz.bar(tidy, "region", "revenue", path=tmp_path / "b.png")
    assert viz.line(tidy, "date", ["revenue", "cost"], path=tmp_path / "l.png")
    assert viz.scatter(tidy, "revenue", "cost", color_by="region", path=tmp_path / "s.png")
    assert viz.box(tidy, ["revenue", "cost"], path=tmp_path / "x.png")
    assert viz.correlation_heatmap(tidy, path=tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0


def test_auto_charts_picks_a_set(tidy, tmp_path):
    made = viz.auto_charts(tidy, str(tmp_path / "charts"))
    assert len(made) >= 4
    assert all(str(path).endswith(".png") for path in made)


# ------------------------------------------------------------------ pipeline


def test_pipeline_end_to_end(tidy, tmp_path):
    source = tmp_path / "in.csv"
    tidy.to_csv(source, index=False)

    config = {
        "name": "test",
        "source": str(source),
        "clean": {"missing": "median"},
        "steps": [
            {"op": "filter", "query": "revenue > 100"},
            {"op": "derive", "expressions": {"margin": "revenue - cost"}},
            {"op": "group_by", "by": "region", "aggregations": {"margin": "sum"}},
        ],
        "output": [str(tmp_path / "out.csv"),
                   f"sqlite:///{(tmp_path / 'w.db').as_posix()}#result"],
        "charts": {"outdir": str(tmp_path / "charts"),
                   "list": [{"kind": "bar", "x": "region", "y": "margin_sum",
                             "path": "region.png"}]},
        "profile": str(tmp_path / "profile.json"),
    }
    result = dp.run_pipeline(config)

    assert result.rows_in == len(tidy)
    assert result.rows_out == tidy["region"].nunique()
    assert len(result.outputs) == 2
    assert len(result.charts) == 1
    assert (tmp_path / "out.csv").exists()
    assert json.loads((tmp_path / "profile.json").read_text())["summary"]["rows"] == result.rows_out
    assert "test" in result.to_text()


def test_pipeline_can_fail_on_validation_error(tidy, tmp_path):
    source = tmp_path / "in.csv"
    tidy.to_csv(source, index=False)
    config = {
        "source": str(source),
        "validate": {"columns": {"missing_column": {"required": True}}},
        "fail_on_validation_error": True,
    }
    with pytest.raises(dp.PipelineError, match="validation failed"):
        dp.run_pipeline(config)


# ----------------------------------------------------------------------- cli


def test_cli_info_and_head(tidy, tmp_path, capsys):
    source = tmp_path / "in.csv"
    tidy.to_csv(source, index=False)

    assert main(["info", str(source)]) == 0
    assert "rows" in capsys.readouterr().out

    assert main(["head", str(source), "-n", "3"]) == 0


def test_cli_query_writes_output(tidy, tmp_path):
    source = tmp_path / "in.csv"
    target = tmp_path / "out.csv"
    tidy.to_csv(source, index=False)

    code = main(["query", str(source), "--filter", "revenue > 100",
                 "--derive", "margin=revenue - cost",
                 "--group-by", "region", "--agg", "margin:sum",
                 "--sort", "margin_sum", "--desc", "-o", str(target)])
    assert code == 0
    out = pd.read_csv(target)
    assert list(out.columns) == ["region", "margin_sum"]
    assert out["margin_sum"].is_monotonic_decreasing


def test_cli_clean_and_convert(messy, tmp_path):
    source = tmp_path / "messy.csv"
    cleaned = tmp_path / "clean.csv"
    messy.to_csv(source, index=False)

    assert main(["clean", str(source), "-o", str(cleaned), "--missing", "median"]) == 0
    out = pd.read_csv(cleaned)
    assert "order_id" in out.columns

    parquet = tmp_path / "clean.parquet"
    assert main(["convert", str(cleaned), str(parquet)]) == 0
    assert parquet.exists()


def test_cli_validate_exit_codes(tidy, tmp_path):
    source = tmp_path / "in.csv"
    tidy.to_csv(source, index=False)
    schema = tmp_path / "schema.json"

    schema.write_text(json.dumps({"columns": {"region": {"type": "str"}}}))
    assert main(["validate", str(source), "--schema", str(schema)]) == 0

    schema.write_text(json.dumps({"columns": {"nope": {"required": True}}}))
    assert main(["validate", str(source), "--schema", str(schema)]) == 1


def test_cli_profile_and_chart(tidy, tmp_path):
    source = tmp_path / "in.csv"
    tidy.to_csv(source, index=False)

    assert main(["profile", str(source), "--json", str(tmp_path / "p.json"), "-q"]) == 0
    assert (tmp_path / "p.json").exists()

    assert main(["chart", str(source), "--kind", "bar", "--x", "region", "--y", "revenue",
                 "-o", str(tmp_path / "bar.png")]) == 0
    assert (tmp_path / "bar.png").exists()


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    assert main(["head", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err
