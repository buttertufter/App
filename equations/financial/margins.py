"""Margin and profitability calculations.

This module contains pure functions for calculating various margin
metrics from financial data.
"""

import numpy as np
from typing import Tuple


EPS = 1e-6  # Small epsilon to prevent division by zero


def calculate_operating_margin(
    revenue: float,
    operating_expenses: float,
    eps: float = EPS
) -> float:
    """Calculate operating margin.
    
    Operating margin = (Revenue - Operating Expenses) / Revenue
    
    Parameters
    ----------
    revenue : float
        Total revenue.
    operating_expenses : float
        Total operating expenses.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    float
        Operating margin as a ratio (0.15 = 15%).
        
    Notes
    -----
    - Returns the margin as a decimal (not percentage)
    - Higher values indicate better operational efficiency
    - Typical range: 0.05 to 0.30 (5% to 30%)
    
    Examples
    --------
    >>> calculate_operating_margin(1000, 750)
    0.25
    >>> calculate_operating_margin(1000, 900)
    0.1
    """
    return float((revenue - operating_expenses) / max(revenue, eps))


def calculate_ocf_margin(
    operating_cash_flow: float,
    revenue: float,
    eps: float = EPS
) -> float:
    """Calculate operating cash flow margin.
    
    OCF margin = Operating Cash Flow / Revenue
    
    Parameters
    ----------
    operating_cash_flow : float
        Operating cash flow.
    revenue : float
        Total revenue.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    float
        OCF margin as a ratio.
        
    Notes
    -----
    - Measures cash generation efficiency
    - Can be negative if company is burning cash
    - Healthy companies often have OCF margin > 15%
    - Higher than operating margin if working capital is improving
    
    Examples
    --------
    >>> calculate_ocf_margin(200, 1000)
    0.2
    >>> calculate_ocf_margin(-50, 1000)
    -0.05
    """
    return float(operating_cash_flow / max(revenue, eps))


def calculate_margins(
    revenue: float,
    operating_expenses: float,
    operating_cash_flow: float,
    eps: float = EPS
) -> Tuple[float, float]:
    """Calculate both operating and OCF margins.
    
    Parameters
    ----------
    revenue : float
        Total revenue.
    operating_expenses : float
        Total operating expenses.
    operating_cash_flow : float
        Operating cash flow.
    eps : float, default=1e-6
        Small constant to prevent division by zero.
        
    Returns
    -------
    operating_margin : float
        Operating margin (Revenue - OpEx) / Revenue.
    ocf_margin : float
        Operating cash flow margin OCF / Revenue.
        
    Examples
    --------
    >>> op_margin, ocf_margin = calculate_margins(1000, 750, 200)
    >>> op_margin
    0.25
    >>> ocf_margin
    0.2
    """
    op_margin = calculate_operating_margin(revenue, operating_expenses, eps)
    ocf_margin = calculate_ocf_margin(operating_cash_flow, revenue, eps)
    
    return op_margin, ocf_margin
