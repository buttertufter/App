"""Capital productivity and efficiency calculations.

This module contains functions for analyzing how efficiently a company
uses its capital (debt, credit, financing) to generate revenue growth.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

from equations.statistics.timeseries import slope_log, ttm_sum


EPS = 1e-6


def calculate_financial_support(
    cash: float,
    debt: float,
    undrawn_credit: float,
    financing_in: float,
    revenue: float,
    lambda_debt: float = 0.5,
    lambda_credit: float = 0.6,
    lambda_financing: float = 0.8,
    eps: float = EPS
) -> float:
    """Calculate financial support index (inverse of capital intensity).
    
    Measures how much capital the company needs relative to revenue.
    Lower capital intensity (higher support) is generally better.
    
    Formula:
        I_eff = (Cash + λ_d * Debt + λ_c * UndrawnCredit + λ_f * FinancingIn) / Revenue
        Support = 1/I_eff - 1
    
    Parameters
    ----------
    cash : float
        Current cash balance.
    debt : float
        Current debt balance.
    undrawn_credit : float
        Available undrawn credit lines.
    financing_in : float
        Recent financing inflows.
    revenue : float
        Trailing twelve month revenue.
    lambda_debt : float, default=0.5
        Weight for debt in capital calculation.
    lambda_credit : float, default=0.6
        Weight for undrawn credit.
    lambda_financing : float, default=0.8
        Weight for financing inflows.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    float
        Financial support index. Higher values indicate less capital-intensive business.
        
    Notes
    -----
    - Negative values indicate high capital intensity
    - Positive values indicate efficient capital usage
    - Typical range: -2.0 to 2.0
    """
    I_eff = (cash + lambda_debt * debt + lambda_credit * undrawn_credit + 
             lambda_financing * financing_in) / max(revenue, eps)
    
    support = 1.0 / max(I_eff, eps) - 1.0
    
    return float(support)


def calculate_capital_productivity(
    revenue_series: pd.Series,
    cash_series: pd.Series,
    debt_series: pd.Series,
    undrawn_credit_series: pd.Series,
    financing_in_series: pd.Series,
    revenue_ttm: float,
    window: int = 4,
    lag: int = 1,
    lambda_debt: float = 0.5,
    lambda_credit: float = 0.6,
    lambda_financing: float = 0.8,
    eps: float = EPS
) -> Tuple[float, List[str]]:
    """Calculate capital productivity: revenue growth per unit of capital investment.
    
    Measures how effectively additional capital translates into revenue growth.
    Higher values indicate better capital efficiency.
    
    Formula:
        CP = ΔlnR / ΔI
        where ΔlnR is log-change in revenue and ΔI is change in capital intensity
    
    Parameters
    ----------
    revenue_series : pd.Series
        Quarterly revenue time series.
    cash_series : pd.Series
        Quarterly cash balances.
    debt_series : pd.Series
        Quarterly debt balances.
    undrawn_credit_series : pd.Series
        Quarterly undrawn credit availability.
    financing_in_series : pd.Series
        Quarterly financing inflows.
    revenue_ttm : float
        Current trailing twelve month revenue for normalization.
    window : int, default=4
        Number of quarters for change calculation.
    lag : int, default=1
        Number of quarters to lag capital changes (causal relationship).
    lambda_debt : float, default=0.5
        Weight for debt in capital calculation.
    lambda_credit : float, default=0.6
        Weight for undrawn credit.
    lambda_financing : float, default=0.8
        Weight for financing inflows.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    capital_productivity : float
        Capital productivity ratio. Higher is better.
        Returns 0.0 if insufficient data or invalid calculation.
    warnings : list of str
        Any issues encountered during calculation.
        
    Notes
    -----
    - Requires at least (window + lag + 1) quarters of data
    - Positive CP indicates capital drives growth
    - Negative CP indicates capital doesn't translate to growth
    - Very high absolute values may indicate data quality issues
    """
    warnings = []
    
    # Check data sufficiency
    min_length = window + lag + 1
    if len(revenue_series) < min_length:
        warnings.append(
            f"Insufficient data for capital productivity (need {min_length} quarters, "
            f"have {len(revenue_series)})"
        )
        return 0.0, warnings
    
    # Calculate capital intensity over time
    I_hist = (
        cash_series + 
        lambda_debt * debt_series + 
        lambda_credit * undrawn_credit_series.fillna(0.0) + 
        lambda_financing * financing_in_series.fillna(0.0)
    ) / max(revenue_ttm, eps)
    
    # Lag the capital series (today's capital influences future growth)
    I_lagged = I_hist.shift(lag)
    
    # Calculate change in capital intensity over window
    dI_series = I_lagged - I_lagged.shift(window)
    dI_latest = dI_series.dropna()
    
    if len(dI_latest) == 0:
        warnings.append("No valid capital intensity changes found")
        return 0.0, warnings
    
    dI = float(dI_latest.iloc[-1])
    
    # Check if capital change is meaningful
    if abs(dI) < eps:
        warnings.append("Capital intensity change negligible; productivity set to neutral")
        return 0.0, warnings
    
    # Calculate revenue log-change over same window
    revenue_log = np.log(np.maximum(revenue_series, eps))
    dlnR_series = revenue_log - revenue_log.shift(window)
    dlnR_latest = dlnR_series.dropna()
    
    if len(dlnR_latest) == 0:
        warnings.append("No valid revenue changes found")
        return 0.0, warnings
    
    dlnR = float(dlnR_latest.iloc[-1])
    
    # Calculate productivity
    capital_productivity = dlnR / dI
    
    # Sanity check
    if not np.isfinite(capital_productivity):
        warnings.append("Capital productivity calculation resulted in invalid value")
        return 0.0, warnings
    
    return capital_productivity, warnings


def calculate_liquidity_ratio(
    cash: float,
    operating_expenses: float,
    eps: float = EPS
) -> float:
    """Calculate liquidity ratio (cash runway in expense units).
    
    Measures how many periods of operating expenses the company can cover with cash.
    
    Parameters
    ----------
    cash : float
        Current cash balance.
    operating_expenses : float
        Trailing operating expenses (annual).
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    float
        Liquidity ratio. Higher values indicate more cash runway.
        
    Notes
    -----
    - Value of 1.0 means cash = annual expenses
    - Value of 0.5 means cash = 6 months of expenses
    - Typical healthy range: 0.5 to 3.0
    
    Examples
    --------
    >>> calculate_liquidity_ratio(1000000, 4000000)
    0.25
    """
    return float(cash / max(operating_expenses, eps))


def calculate_net_debt_ratio(
    debt: float,
    cash: float,
    revenue: float,
    eps: float = EPS
) -> float:
    """Calculate net debt as a ratio of revenue.
    
    Net debt = max(Debt - Cash, 0) / Revenue
    
    Parameters
    ----------
    debt : float
        Total debt.
    cash : float
        Total cash.
    revenue : float
        Trailing twelve month revenue.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    float
        Net debt ratio. Lower is better. Zero if cash > debt.
        
    Notes
    -----
    - Only counts net debt (debt minus cash), floored at zero
    - Normalized by revenue for comparability
    - Typical range: 0.0 to 2.0
    - > 1.0 indicates debt exceeds annual revenue
    
    Examples
    --------
    >>> calculate_net_debt_ratio(1000, 400, 5000)
    0.12
    >>> calculate_net_debt_ratio(400, 1000, 5000)  # Cash > Debt
    0.0
    """
    net_debt = max(debt - cash, 0.0)
    return float(net_debt / max(revenue, eps))
