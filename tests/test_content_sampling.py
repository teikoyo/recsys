"""Tests for src.content.sampling -- column profiling and description generation."""

import numpy as np
import pandas as pd
import pytest

from src.content.sampling import profile_column, col_to_description


# ---------- profile_column ----------

def test_profile_numeric_column():
    """Numeric series is detected as dtype='numeric'."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cs = profile_column(series, "price")
    assert cs.dtype == "numeric"


def test_profile_categorical_column():
    """Short-string column with few unique values is categorical."""
    series = pd.Series(["red", "blue", "green", "red", "blue", "red"])
    cs = profile_column(series, "color")
    assert cs.dtype == "categorical"


def test_profile_datetime_column():
    """ISO date strings are detected as datetime."""
    series = pd.Series([
        "2023-01-01", "2023-02-15", "2023-03-20",
        "2023-04-10", "2023-05-05", "2023-06-30",
    ])
    cs = profile_column(series, "created_at")
    assert cs.dtype == "datetime"


# ---------- col_to_description ----------

def test_col_to_description_numeric():
    """Numeric description contains 'numeric' and range info."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    cs = profile_column(series, "score")
    desc = col_to_description(cs)
    assert "numeric" in desc.lower()
    assert "10" in desc
    assert "50" in desc


def test_col_to_description_categorical():
    """Categorical description contains unique count."""
    series = pd.Series(["a", "b", "c", "a", "b", "a"])
    cs = profile_column(series, "category")
    desc = col_to_description(cs)
    assert "3" in desc  # 3 unique values
    assert "categorical" in desc.lower()
