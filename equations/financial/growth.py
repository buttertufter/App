"""Financial growth and momentum calculations.

This module contains pure functions for calculating growth rates,
momentum, and acceleration metrics from financial time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from equations.statistics.timeseries import slope_log, ttm_sum


def calculate_revenue_growth(
    revenue_series: pd.Series,
    window: int = 4,
    fallback: float = 0.0
) -> Tuple[pd.Series, float]:
    """Calculate revenue growth rate using log-linear regression.
    
    Computes the per-quarter log-growth rate of revenue using OLS regression
    on log-transformed revenue over a rolling window.
    
    Parameters
    ----------
    revenue_series : pd.Series
        Quarterly revenue time series.
    window : int, default=4
        Number of quarters for rolling window.
    fallback : float, default=0.0
        Default growth rate if calculation fails.
        
    Returns
    -------
    growth_series : pd.Series
        Rolling growth rate series (log-change per quarter).
    latest_growth : float
        Most recent growth rate value.
        
    Notes
    -----
    - Returns log-growth rate: for small rates, ≈ actual growth rate
    - Units: per-quarter log-change (e.g., 0.05 ≈ 5% quarterly growth)
    - Uses TTM (trailing twelve months) sums before calculating growth
    
    Examples
    --------
    >>> revenue = pd.Series([100, 105, 110.25, 115.76])  # ~5% growth
    >>> growth_series, latest = calculate_revenue_growth(revenue, window=3)
    """
    # Calculate TTM revenue first
    revenue_ttm = ttm_sum(revenue_series, window_quarters=4)
    
    # Forward fill any gaps
    revenue_ttm = revenue_ttm.ffill()
    
    # Calculate log-linear growth
    growth_series = slope_log(revenue_ttm, window=window)
    
    # Get latest value
    valid_growth = growth_series.dropna()
    if len(valid_growth) > 0:
        latest_growth = float(valid_growth.iloc[-1])
    else:
        latest_growth = fallback
    
    if not np.isfinite(latest_growth):
        latest_growth = fallback
    
    return growth_series, latest_growth


def calculate_revenue_acceleration(
    revenue_series: pd.Series,
    window: int = 4,
    fallback: float = 0.0
) -> Tuple[pd.Series, float]:
    """Calculate revenue acceleration (change in growth rate).
    
    Computes the second derivative of log-revenue, measuring how
    the growth rate itself is changing over time.
    
    Parameters
    ----------
    revenue_series : pd.Series
        Quarterly revenue time series.
    window : int, default=4
        Number of quarters for rolling window.
    fallback : float, default=0.0
        Default acceleration if calculation fails.
        
    Returns
    -------
    accel_series : pd.Series
        Rolling acceleration series.
    latest_accel : float
        Most recent acceleration value.
        
    Notes
    -----
    - Positive acceleration means growth rate is increasing
    - Negative acceleration means growth rate is decreasing (but could still be growing)
    - Units: change in log-growth per quarter
    
    Examples
    --------
    >>> revenue = pd.Series([100, 105, 111, 118])  # accelerating growth
    >>> accel_series, latest = calculate_revenue_acceleration(revenue, window=3)
    """
    # First get growth rate
    growth_series, _ = calculate_revenue_growth(revenue_series, window=window)
    
    # Forward fill gaps
    growth_filled = growth_series.ffill()
    
    # Calculate acceleration (slope of growth)
    accel_series = slope_log(growth_filled, window=window)
    
    # Get latest value
    valid_accel = accel_series.dropna()
    if len(valid_accel) > 0:
        latest_accel = float(valid_accel.iloc[-1])
    else:
        latest_accel = fallback
    
    if not np.isfinite(latest_accel):
        latest_accel = fallback
    
    return accel_series, latest_accel


def calculate_ttm_metrics(
    revenue: pd.Series,
    operating_expenses: pd.Series,
    operating_cash_flow: pd.Series,
    window: int = 4,
    revenue_fallback: float = 1e6,
    expense_fallback_pct: float = 0.75,
    ocf_fallback_pct: float = 0.15
) -> Tuple[float, float, float]:
    """Calculate trailing twelve month (TTM) metrics.
    
    Sums the last 4 quarters to get TTM totals for revenue,
    operating expenses, and operating cash flow.
    
    Parameters
    ----------
    revenue : pd.Series
        Quarterly revenue.
    operating_expenses : pd.Series
        Quarterly operating expenses.
    operating_cash_flow : pd.Series
        Quarterly operating cash flow.
    window : int, default=4
        Number of quarters to sum (4 = TTM).
    revenue_fallback : float, default=1e6
        Fallback revenue value if data missing.
    expense_fallback_pct : float, default=0.75
        Fallback expenses as % of revenue.
    ocf_fallback_pct : float, default=0.15
        Fallback OCF as % of revenue.
        
    Returns
    -------
    ttm_revenue : float
        Trailing twelve month revenue.
    ttm_expenses : float
        Trailing twelve month operating expenses.
    ttm_ocf : float
        Trailing twelve month operating cash flow.
        
    Notes
    -----
    - Uses fallbacks if data is insufficient or invalid
    - All values returned in the same units as input
    - Designed to handle missing or sparse data gracefully
    """
    # Revenue TTM
    revenue_ttm_series = ttm_sum(revenue, window_quarters=window)
    valid_revenue = revenue_ttm_series.dropna()
    
    if len(valid_revenue) > 0:
        ttm_revenue = float(valid_revenue.iloc[-1])
    else:
        # Try last raw value
        valid_revenue_raw = revenue.dropna()
        if len(valid_revenue_raw) > 0:
            ttm_revenue = float(valid_revenue_raw.iloc[-1])
        else:
            ttm_revenue = revenue_fallback
    
    if not np.isfinite(ttm_revenue):
        ttm_revenue = revenue_fallback
    
    # Expenses TTM
    expenses_ttm_series = ttm_sum(operating_expenses, window_quarters=window)
    valid_expenses = expenses_ttm_series.dropna()
    
    if len(valid_expenses) > 0:
        ttm_expenses = float(valid_expenses.iloc[-1])
    else:
        # Try last raw value
        valid_expenses_raw = operating_expenses.dropna()
        if len(valid_expenses_raw) > 0:
            ttm_expenses = float(valid_expenses_raw.iloc[-1])
        else:
            ttm_expenses = ttm_revenue * expense_fallback_pct
    
    if not np.isfinite(ttm_expenses):
        ttm_expenses = ttm_revenue * expense_fallback_pct
    
    # OCF TTM
    ocf_ttm_series = ttm_sum(operating_cash_flow, window_quarters=window)
    valid_ocf = ocf_ttm_series.dropna()
    
    if len(valid_ocf) > 0:
        ttm_ocf = float(valid_ocf.iloc[-1])
    else:
        # Try last raw value
        valid_ocf_raw = operating_cash_flow.dropna()
        if len(valid_ocf_raw) > 0:
            ttm_ocf = float(valid_ocf_raw.iloc[-1])
        else:
            ttm_ocf = ttm_revenue * ocf_fallback_pct
    
    if not np.isfinite(ttm_ocf):
        ttm_ocf = ttm_revenue * ocf_fallback_pct
    
    return ttm_revenue, ttm_expenses, ttm_ocf
