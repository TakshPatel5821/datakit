import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy() -> pd.DataFrame:
    """A small frame with every problem the cleaner is supposed to fix."""
    return pd.DataFrame({
        "Order ID": ["A-1", "A-2", "A-3", "A-3", "A-4", "A-5"],
        "  Region  ": ["  North ", "South", "north", "north", "East", None],
        "Revenue ($)": ["1,200.50", "800", "$2,000.00", "$2,000.00", "950.25", "1,100"],
        "Units": [3, 5, np.nan, np.nan, 9999, 4],
        "Joined": ["2024-01-05", "2024-02-11", "2024-03-02", "2024-03-02",
                   "2024-04-19", "2024-05-30"],
        "Empty": [None, None, None, None, None, None],
    })


@pytest.fixture
def tidy() -> pd.DataFrame:
    """A clean frame for transform/profile tests."""
    rng = np.random.default_rng(0)
    rows = 60
    return pd.DataFrame({
        "region": ["North", "South", "East"] * 20,
        "category": (["Hardware", "Software"] * 30),
        "revenue": np.round(rng.uniform(100, 900, rows), 2),
        "cost": np.round(rng.uniform(50, 500, rows), 2),
        "units": rng.integers(1, 25, rows),
        "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
    })
