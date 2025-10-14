import numpy as np
import pandas as pd

from utils import last_or_default, to_series_1d


def test_to_series_from_ndarray():
    arr = np.arange(6, dtype=float).reshape(-1, 1)
    series = to_series_1d(arr, name="ndarray")
    assert isinstance(series, pd.Series)
    assert series.shape == (6,)
    assert series.iloc[-1] == 5.0
    assert last_or_default(arr) == 5.0


def test_to_series_from_dataframe():
    df = pd.DataFrame({"value": [1.0, np.nan, 3.0, 4.0]})
    series = to_series_1d(df, name="dataframe")
    assert isinstance(series, pd.Series)
    assert series.tolist() == [1.0, 3.0, 4.0]
    assert last_or_default(df) == 4.0
