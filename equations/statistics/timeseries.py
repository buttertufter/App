"""Time series analysis functions.

This module contains pure functions for analyzing time series data,
including exponential moving averages, rolling calculations, and trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional


def ema(s: pd.Series, span: int = 4) -> pd.Series:
    """Calculate exponential moving average of a time series.
    
    The EMA gives more weight to recent observations while still considering
    historical values. It's smoother than a simple moving average and more
    responsive to recent changes.
    
    Parameters
    ----------
    s : pd.Series
        Input time series data.
    span : int, default=4
        Number of periods for the EMA span. Corresponds to roughly
        (span + 1) / 2 periods of decay.
        
    Returns
    -------
    pd.Series
        Exponential moving average with same index as input.
        
    Notes
    -----
    - Uses pandas ewm() with adjust=False (standard recursive formula)
    - Alpha = 2 / (span + 1)
    - For span=4, alpha ≈ 0.4, giving ~40% weight to most recent value
    
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> ema(s, span=3)  # doctest: +SKIP
    0    1.000000
    1    1.500000
    2    2.250000
    3    3.125000
    4    4.062500
    """
    return s.ewm(span=span, adjust=False).mean()


def ttm_sum(q_series: pd.Series, window_quarters: int = 4) -> pd.Series:
    """Calculate trailing twelve month (TTM) sum from quarterly data.
    
    Sums the last N quarters (typically 4) to get a rolling annual total.
    Commonly used for revenue, expenses, cash flow, etc.
    
    Parameters
    ----------
    q_series : pd.Series
        Quarterly time series data.
    window_quarters : int, default=4
        Number of quarters to sum. Use 4 for TTM (trailing twelve months),
        8 for rolling 2-year, etc.
        
    Returns
    -------
    pd.Series
        Rolling sum with same index as input. First (window_quarters - 1)
        values will be NaN.
        
    Notes
    -----
    - First TTM value appears at index (window_quarters - 1)
    - Units: Same as input (e.g., if input is millions, output is millions)
    - Assumes input is in chronological order
    
    Examples
    --------
    >>> q = pd.Series([100, 110, 120, 130, 140])
    >>> ttm_sum(q, window_quarters=4)  # doctest: +SKIP
    0     NaN
    1     NaN
    2     NaN
    3    460.0  # 100+110+120+130
    4    500.0  # 110+120+130+140
    """
    return q_series.rolling(window_quarters).sum()


def slope_log(series: pd.Series, window: int = 4, eps: float = 1e-6) -> pd.Series:
    """Calculate OLS slope of log-transformed series over rolling window.
    
    Fits a linear regression to log(series) over a rolling window and
    returns the slope coefficient. This gives the per-period log-change,
    which approximates the growth rate.
    
    For small growth rates: slope ≈ growth rate
    For larger rates: actual_growth = exp(slope) - 1
    
    Parameters
    ----------
    series : pd.Series
        Input time series. Should be positive values.
    window : int, default=4
        Number of periods for the rolling window.
    eps : float, default=1e-6
        Small constant to prevent log(0).
        
    Returns
    -------
    pd.Series
        Rolling log-slope values. First (window - 1) values are NaN.
        Units: log-change per period (approximately growth rate).
        
    Notes
    -----
    - Uses ordinary least squares (OLS) regression on log-transformed data
    - Slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    - Returns NaN if denominator is zero (flat time index)
    - More robust to outliers than simple percentage change
    
    Examples
    --------
    >>> s = pd.Series([100, 110, 121, 133.1])  # ~10% growth
    >>> slope_log(s, window=3)  # doctest: +SKIP
    0      NaN
    1      NaN
    2    0.095  # ≈ ln(1.10) ≈ 0.095
    3    0.095
    """
    series = series.copy()
    series = series.replace(0, eps)
    s = np.log(np.maximum(series, eps))
    
    idx = np.arange(len(s))
    out = pd.Series(index=series.index, dtype=float)
    
    for t in range(len(s)):
        if t + 1 < window:
            out.iloc[t] = np.nan
            continue
            
        x = idx[t - window + 1: t + 1].astype(float)
        y = s.iloc[t - window + 1: t + 1].astype(float).values
        
        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm) ** 2)
        
        if denom == 0:
            out.iloc[t] = np.nan
        else:
            out.iloc[t] = float(np.sum((x - xm) * (y - ym)) / denom)
    
    return out


def stdev_over(series: pd.Series, window: int = 4) -> pd.Series:
    """Calculate rolling standard deviation.
    
    Measures the volatility or dispersion of values over a rolling window.
    Higher values indicate more variability.
    
    Parameters
    ----------
    series : pd.Series
        Input time series.
    window : int, default=4
        Number of periods for the rolling window.
        
    Returns
    -------
    pd.Series
        Rolling standard deviation. First (window - 1) values are NaN.
        Units: Same as input series.
        
    Notes
    -----
    - Uses sample standard deviation (Bessel's correction, ddof=1)
    - Useful for measuring volatility, risk, consistency
    - Returns NaN for windows with < 2 observations
    
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> stdev_over(s, window=3)  # doctest: +SKIP
    0         NaN
    1         NaN
    2    1.000000
    3    1.000000
    4    1.000000
    """
    return series.rolling(window).std()


def to_series_1d(x, name: Optional[str] = None) -> pd.Series:
    """Convert various data types to a 1D pandas Series.
    
    Handles conversion from None, Series, DataFrame, arrays, and scalars
    to a clean 1D pandas Series with numeric values.
    
    Parameters
    ----------
    x : None, pd.Series, pd.DataFrame, array-like, or scalar
        Input data to convert.
    name : str, optional
        Name to assign to the output series.
        
    Returns
    -------
    pd.Series
        1D Series with numeric values (NaN values dropped).
        Returns empty Series if input is None or cannot be converted.
        
    Notes
    -----
    - DataFrame inputs: if single column, uses that column; otherwise takes row mean
    - Multi-dimensional arrays are squeezed to 1D
    - Non-numeric values are coerced to NaN and then dropped
    - Preserves index from Series/DataFrame inputs when possible
    
    Examples
    --------
    >>> to_series_1d([1, 2, 3])  # doctest: +SKIP
    0    1.0
    1    2.0
    2    3.0
    dtype: float64
    
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> to_series_1d(df, name='values')  # doctest: +SKIP
    0    1.0
    1    2.0
    2    3.0
    Name: values, dtype: float64
    """
    if x is None:
        return pd.Series(dtype="float64", name=name)
    
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = pd.Series(x.iloc[:, 0].values.flatten(), index=x.index, name=name)
        else:
            s = x.mean(axis=1)
    else:
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        s = pd.Series(arr, name=name, dtype="float64")
    
    s = pd.to_numeric(s, errors="coerce").dropna()
    
    if name is not None:
        s.name = name
    
    return s


def last_or_default(x, default=np.nan) -> float:
    """Get the last value from a series or return a default.
    
    Parameters
    ----------
    x : array-like
        Input data (converted to Series internally).
    default : float, default=np.nan
        Value to return if series is empty.
        
    Returns
    -------
    float
        Last value in the series, or default if empty.
        
    Examples
    --------
    >>> last_or_default([1, 2, 3])
    3.0
    >>> last_or_default([])
    nan
    >>> last_or_default([], default=0.0)
    0.0
    """
    s = to_series_1d(x)
    return float(s.iloc[-1]) if len(s) > 0 else default


def nth_from_end_or_default(x, n: int = 1, default=np.nan) -> float:
    """Get the nth value from the end of a series or return a default.
    
    Parameters
    ----------
    x : array-like
        Input data (converted to Series internally).
    n : int, default=1
        Position from end (1 = last, 2 = second-to-last, etc.).
    default : float, default=np.nan
        Value to return if series doesn't have enough values.
        
    Returns
    -------
    float
        nth value from end, or default if not enough values.
        
    Examples
    --------
    >>> nth_from_end_or_default([1, 2, 3, 4], n=1)
    4.0
    >>> nth_from_end_or_default([1, 2, 3, 4], n=2)
    3.0
    >>> nth_from_end_or_default([1, 2], n=5, default=0.0)
    0.0
    """
    if n <= 0:
        return default
    
    s = to_series_1d(x)
    
    if len(s) < n:
        return default
    
    return float(s.iloc[-n])
