import pandas as pd
import pytest

from datakit import transform as tf


def test_select_and_drop(tidy):
    assert list(tf.select(tidy, "region,revenue").columns) == ["region", "revenue"]
    assert "revenue" not in tf.drop(tidy, ["revenue"]).columns


def test_select_reports_missing_columns(tidy):
    with pytest.raises(tf.TransformError, match="not found"):
        tf.select(tidy, "nope")


def test_filter_with_query_and_with_operator(tidy):
    by_query = tf.filter_rows(tidy, "revenue > 500")
    by_op = tf.filter_rows(tidy, column="revenue", op=">", value=500)
    assert len(by_query) == len(by_op) > 0
    assert (by_query["revenue"] > 500).all()


def test_filter_contains_is_case_insensitive(tidy):
    out = tf.filter_rows(tidy, column="region", op="contains", value="nort")
    assert set(out["region"]) == {"North"}


def test_group_by_counts_when_no_aggregations(tidy):
    out = tf.group_by(tidy, "region")
    assert set(out.columns) == {"region", "count"}
    assert out["count"].sum() == len(tidy)


def test_group_by_names_every_result_column_after_its_function(tidy):
    out = tf.group_by(tidy, "region", {"revenue": ["sum", "mean"], "units": "sum"})
    assert {"revenue_sum", "revenue_mean", "units_sum"} <= set(out.columns)
    assert round(out["revenue_sum"].sum(), 2) == round(tidy["revenue"].sum(), 2)


def test_derive_uses_earlier_results(tidy):
    out = tf.derive(tidy, {"margin": "revenue - cost", "margin_pct": "margin / revenue * 100"})
    assert "margin_pct" in out.columns
    assert (out["margin"] == tidy["revenue"] - tidy["cost"]).all()


def test_derive_falls_back_for_non_eval_expressions(tidy):
    out = tf.derive(tidy, {"upper": "region.str.upper()"})
    assert out["upper"].iloc[0] == "NORTH"


def test_join_on_key():
    left = pd.DataFrame({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    right = pd.DataFrame({"id": [1, 3], "w": [10, 30]})
    assert len(tf.join(left, right, on="id")) == 2
    assert len(tf.join(left, right, on="id", how="left")) == 3


def test_pivot_and_unpivot_round_trip(tidy):
    wide = tf.pivot(tidy, index="region", columns="category", values="revenue", agg="sum")
    assert "Hardware" in wide.columns
    long = tf.unpivot(wide, id_columns="region", var_name="category", value_name="revenue")
    assert set(long.columns) == {"region", "category", "revenue"}


def test_top_n_returns_largest(tidy):
    out = tf.top_n(tidy, "revenue", 3)
    assert len(out) == 3
    assert out["revenue"].tolist() == sorted(tidy["revenue"], reverse=True)[:3]


def test_resample_time_rolls_up_to_months(tidy):
    out = tf.resample_time(tidy, on="date", freq="ME", aggregations={"revenue": "sum"})
    assert len(out) < len(tidy)
    assert round(out["revenue"].sum(), 2) == round(tidy["revenue"].sum(), 2)


def test_bin_column_adds_buckets(tidy):
    out = tf.bin_column(tidy, "revenue", bins=4)
    assert out["revenue_bin"].nunique() <= 4


def test_apply_steps_runs_a_sequence(tidy):
    out = tf.apply_steps(tidy, [
        {"op": "filter", "query": "revenue > 200"},
        {"op": "derive", "expressions": {"margin": "revenue - cost"}},
        {"op": "group_by", "by": "region", "aggregations": {"margin": "sum"}},
        {"op": "sort", "by": "margin_sum", "ascending": False},
    ])
    assert list(out.columns) == ["region", "margin_sum"]
    assert out["margin_sum"].is_monotonic_decreasing


def test_apply_steps_rejects_unknown_op(tidy):
    with pytest.raises(tf.TransformError, match="unknown op"):
        tf.apply_steps(tidy, [{"op": "teleport"}])
