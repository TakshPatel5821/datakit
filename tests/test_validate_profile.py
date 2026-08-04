import numpy as np
import pandas as pd

from datakit import profiling, validation as vd


# ------------------------------------------------------------------ validate

SCHEMA = {
    "columns": {
        "id": {"type": "int", "unique": True, "nullable": False},
        "email": {"type": "str", "regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
        "age": {"type": "int", "min": 0, "max": 120},
        "status": {"type": "str", "allowed": ["active", "inactive"]},
    },
    "min_rows": 1,
}


def good_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": pd.array([1, 2, 3], dtype="Int64"),
        "email": ["a@x.com", "b@x.com", "c@x.com"],
        "age": pd.array([30, 41, 22], dtype="Int64"),
        "status": ["active", "inactive", "active"],
    })


def test_valid_data_passes():
    report = vd.validate(good_frame(), SCHEMA)
    assert report.ok
    assert report.errors == []


def test_duplicate_key_is_an_error():
    df = good_frame()
    df.loc[2, "id"] = 1
    report = vd.validate(df, SCHEMA)
    assert not report.ok
    assert any(issue.rule == "unique" for issue in report.errors)


def test_out_of_range_and_disallowed_values_are_reported():
    df = good_frame()
    df.loc[0, "age"] = 500
    df.loc[1, "status"] = "pending"
    report = vd.validate(df, SCHEMA)
    rules = {issue.rule for issue in report.errors}
    assert {"max", "allowed"} <= rules


def test_regex_violation_is_reported_with_examples():
    df = good_frame()
    df.loc[0, "email"] = "not-an-email"
    report = vd.validate(df, SCHEMA)
    issue = next(i for i in report.errors if i.rule == "regex")
    assert issue.count == 1
    assert "not-an-email" in issue.examples


def test_missing_required_column_is_an_error():
    report = vd.validate(good_frame().drop(columns=["email"]), SCHEMA)
    assert any(issue.rule == "missing_column" for issue in report.errors)


def test_nulls_rejected_when_not_nullable():
    df = good_frame()
    df.loc[0, "id"] = pd.NA
    report = vd.validate(df, SCHEMA)
    assert any(issue.rule == "nullable" for issue in report.errors)


def test_strict_mode_flags_unexpected_columns():
    df = good_frame()
    df["extra"] = 1
    report = vd.validate(df, {**SCHEMA, "strict": True})
    assert any(issue.rule == "strict" for issue in report.errors)


def test_infer_schema_round_trips(tmp_path):
    schema = vd.infer_schema(good_frame())
    assert schema["columns"]["id"]["type"] == "int"
    assert schema["columns"]["status"]["allowed"] == ["active", "inactive"]

    path = tmp_path / "schema.yaml"
    vd.save_schema(schema, path)
    assert vd.load_schema(path) == schema
    assert vd.validate(good_frame(), schema).ok


# ------------------------------------------------------------------- profile


def test_profile_summary_counts(tidy):
    prof = profiling.profile(tidy)
    assert prof.summary["rows"] == len(tidy)
    assert prof.summary["columns"] == tidy.shape[1]
    assert prof.summary["missing_cells"] == 0
    assert prof.quality["score"] > 60


def test_profile_numeric_column_stats(tidy):
    info = profiling.profile(tidy).columns["revenue"]
    assert info["kind"] == "numeric"
    assert info["min"] <= info["median"] <= info["max"]
    assert info["nulls"] == 0


def test_profile_flags_quality_problems():
    df = pd.DataFrame({
        "constant": [1] * 20,
        "mostly_null": [np.nan] * 18 + [1, 2],
        "value": list(range(20)),
    })
    df = pd.concat([df, df.head(3)], ignore_index=True)
    prof = profiling.profile(df)
    notes = " ".join(prof.quality["notes"])
    assert "duplicate" in notes
    assert "constant" in notes
    assert "mostly_null" in notes


def test_profile_text_and_json_render(tidy):
    prof = profiling.profile(tidy)
    assert "DATA PROFILE" in prof.to_text()
    assert '"summary"' in prof.to_json()
    assert not profiling.missing_summary(tidy).shape[0]


def test_describe_returns_one_row_per_column(tidy):
    out = profiling.describe(tidy)
    assert len(out) == tidy.shape[1]
    assert "null_%" in out.columns
