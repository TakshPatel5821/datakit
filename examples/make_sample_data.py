"""Generate a deliberately messy sales file so the toolkit has something to chew on.

    python examples/make_sample_data.py

Writes ``examples/data/sales_raw.csv`` with inconsistent column names, text
numbers, mixed date formats, whitespace, duplicates, nulls and a few outliers.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data" / "sales_raw.csv"

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Hardware", "Software", "Services", "Support"]
CHANNELS = ["online", "retail", "partner"]


def build(rows: int = 1200, seed: int = 7) -> pd.DataFrame:
    random.seed(seed)
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-01", periods=rows, freq="6h")
    region = rng.choice(REGIONS, rows, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    category = rng.choice(CATEGORIES, rows)
    units = rng.integers(1, 40, rows)
    unit_price = np.round(rng.gamma(shape=4.0, scale=45.0, size=rows) + 15, 2)
    revenue = np.round(units * unit_price, 2)
    cost = np.round(revenue * rng.uniform(0.45, 0.85, rows), 2)

    df = pd.DataFrame({
        "Order ID": [f"ORD-{100000 + i}" for i in range(rows)],
        "Order Date": [d.strftime("%Y-%m-%d") for d in dates],
        "  Region  ": region,
        "Category": category,
        "Channel": rng.choice(CHANNELS, rows),
        "Units Sold": units,
        "Unit Price": unit_price,
        "Revenue ($)": revenue,
        "Cost": cost,
        "Customer Email": [f"user{i % 400}@example.com" for i in range(rows)],
        "Notes": [""] * rows,
    })

    # --- now make it messy -------------------------------------------------
    # currency-formatted text in a numeric column
    money_rows = rng.choice(rows, size=rows // 4, replace=False)
    df["Revenue ($)"] = df["Revenue ($)"].astype(object)
    df.loc[money_rows, "Revenue ($)"] = [f"${value:,.2f}" for value in revenue[money_rows]]

    # untrimmed and inconsistently-cased text
    ragged = rng.choice(rows, size=rows // 6, replace=False)
    df.loc[ragged, "  Region  "] = [f"  {value}  " for value in region[ragged]]

    # a second date format
    other_format = rng.choice(rows, size=rows // 8, replace=False)
    df.loc[other_format, "Order Date"] = [dates[i].strftime("%d/%m/%Y") for i in other_format]

    # missing values
    for column, fraction in [("Cost", 0.06), ("Channel", 0.03), ("Unit Price", 0.02)]:
        blanks = rng.choice(rows, size=int(rows * fraction), replace=False)
        df.loc[blanks, column] = np.nan

    # outliers
    extremes = rng.choice(rows, size=6, replace=False)
    df.loc[extremes, "Units Sold"] = rng.integers(3000, 9000, len(extremes))

    # duplicate rows
    duplicates = df.sample(25, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    # an entirely empty column survives to be dropped by clean()
    df["Legacy Flag"] = np.nan
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} — {len(df):,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
