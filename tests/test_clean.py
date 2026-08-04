import numpy as np
import pandas as pd

from datakit import cleaning as ck


def test_standardize_columns_snake_cases_and_deduplicates():
    df = pd.DataFrame(columns=["Order ID", "Revenue ($)", "orderId", "order id"])
    out = ck.standardize_columns(df)
    assert list(out.columns) == ["order_id", "revenue", "order_id_1", "order_id_2"]


def test_strip_whitespace_trims_and_nulls_empty_strings():
    df = pd.DataFrame({"a": ["  x  ", "y   z", "", "  "]})
    out = ck.strip_whitespace(df)
    assert out["a"].tolist()[:2] == ["x", "y z"]
    assert out["a"].isna().sum() == 2


def test_drop_empty_removes_all_null_rows_and_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
    out = ck.drop_empty(df)
    assert list(out.columns) == ["a"]
    assert len(out) == 2


def test_coerce_types_parses_currency_text_as_numbers():
    df = pd.DataFrame({"amount": ["1,200.50", "$800", "950.25"]})
    out = ck.coerce_types(df)
    assert pd.api.types.is_numeric_dtype(out["amount"])
    assert out["amount"].iloc[0] == 1200.50


def test_coerce_types_leaves_digit_only_identifiers_alone_as_dates():
    df = pd.DataFrame({"zip": ["02134", "94110", "10001"]})
    out = ck.coerce_types(df)
    assert not pd.api.types.is_datetime64_any_dtype(out["zip"])


def test_coerce_types_explicit_datetime():
    df = pd.DataFrame({"joined": ["2024-01-05", "2024-02-11"]})
    out = ck.coerce_types(df, {"joined": "datetime"}, infer=False)
    assert pd.api.types.is_datetime64_any_dtype(out["joined"])


def test_fill_missing_per_column_strategies():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0], "t": ["a", None, "a"]})
    out = ck.fill_missing(df, {"n": "median", "t": "unknown"})
    assert out["n"].tolist() == [1.0, 2.0, 3.0]
    assert out["t"].tolist() == ["a", "unknown", "a"]


def test_fill_missing_auto_uses_median_then_mode():
    df = pd.DataFrame({"n": [1.0, np.nan, 5.0], "t": ["a", None, "a"]})
    out = ck.fill_missing(df, "auto")
    assert out["n"].isna().sum() == 0
    assert out["t"].tolist() == ["a", "a", "a"]


def test_remove_outliers_drop_and_clip():
    df = pd.DataFrame({"v": [10, 11, 12, 13, 14, 5000]})
    dropped = ck.remove_outliers(df, method="iqr", action="drop")
    assert 5000 not in dropped["v"].tolist()
    clipped = ck.remove_outliers(df, method="iqr", action="clip")
    assert len(clipped) == len(df) and clipped["v"].max() < 5000


def test_clean_pipeline_end_to_end(messy):
    report = ck.CleaningReport()
    out = ck.clean(messy, missing="median", outliers="iqr", outlier_factor=3.0, report=report)

    assert "order_id" in out.columns
    assert "empty" not in out.columns          # all-null column dropped
    assert len(out) < len(messy)               # duplicate row removed
    assert pd.api.types.is_numeric_dtype(out["revenue"])
    assert out["units"].isna().sum() == 0
    assert report                              # steps were recorded
    assert "drop_duplicates" in report
