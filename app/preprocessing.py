"""Data validation and preprocessing utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from constants import (
    REQUIRED_FIELDS,
    DEFAULT_QUARTERS_WINDOW,
    MIN_QUARTERS_DATA,
    MIN_ANALYSIS_QUARTERS,
    FALLBACKS
)

def validate_financials(
    df: pd.DataFrame,
    min_quarters: int = MIN_QUARTERS_DATA
) -> Tuple[bool, List[str]]:
    """
    Validate financial data quality and coverage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Quarterly financial data
    min_quarters : int
        Minimum required quarters of data
        
    Returns
    -------
    bool : Whether data is valid
    list : List of validation issues found
    """
    issues = []
    
    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        issues.append(f"Missing required fields: {', '.join(missing)}")
        return False, issues
        
    # Convert to numeric
    df_clean = df[REQUIRED_FIELDS].apply(pd.to_numeric, errors='coerce')
    
    # Check data coverage
    n_valid = df_clean.dropna(how='all').shape[0]
    if n_valid < min_quarters:
        issues.append(
            f"Insufficient data: found {n_valid} quarters, need {min_quarters}"
        )
        return False, issues
        
    # Check specific requirements
    if df_clean["Revenue"].dropna().empty:
        issues.append("No valid revenue data found")
        return False, issues
        
    return True, issues

def preprocess_financials(
    df: pd.DataFrame,
    window: int = DEFAULT_QUARTERS_WINDOW
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess financial data for analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw quarterly financial data
    window : int
        Rolling window size for calculations
        
    Returns
    -------
    pd.DataFrame : Preprocessed data
    list : Processing notes and warnings
    """
    notes = []
    
    # Work on copy
    df = df.copy()
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Basic cleaning
    df = df.dropna(how='all')
    df = df.sort_index()
    
    # Handle missing values
    if df["Revenue"].isna().any():
        notes.append("Some revenue values missing")
        df["Revenue"] = df["Revenue"].ffill()
        
    if df["OperatingExpenses"].isna().any():
        notes.append("Imputing missing operating expenses")
        df["OperatingExpenses"] = df["Revenue"] * (1 - FALLBACKS["margin"])
        
    if df["OperatingCashFlow"].isna().any():
        notes.append("Imputing missing operating cash flow")
        df["OperatingCashFlow"] = df["Revenue"] * FALLBACKS["cash_flow_margin"]
        
    # Balance sheet items
    if df["Cash"].isna().any():
        notes.append("Forward-filling missing cash values")
        df["Cash"] = df["Cash"].ffill()
        
    if df["Debt"].isna().any():
        notes.append("Forward-filling missing debt values")
        df["Debt"] = df["Debt"].ffill()
        
    return df, notes

def compute_derived_metrics(
    df: pd.DataFrame,
    window: int = DEFAULT_QUARTERS_WINDOW
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute derived financial metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean quarterly financial data
    window : int 
        Rolling window size
        
    Returns
    -------
    pd.DataFrame : Data with additional metrics
    list : Computation notes
    """
    notes = []
    
    # Work on copy
    df = df.copy()
    
    # TTM calculations
    for col in ["Revenue", "OperatingExpenses", "OperatingCashFlow"]:
        df[f"{col}_TTM"] = df[col].rolling(window).sum()
        
    # Growth rates
    df["Revenue_Growth"] = df["Revenue_TTM"].pct_change(periods=window)
    df["OCF_Growth"] = df["OperatingCashFlow_TTM"].pct_change(periods=window)
    
    # Margins
    df["Operating_Margin"] = 1 - (df["OperatingExpenses"] / df["Revenue"])
    df["OCF_Margin"] = df["OperatingCashFlow"] / df["Revenue"]
    
    # Stability metrics 
    df["Margin_Stability"] = df["Operating_Margin"].rolling(window).std()
    df["OCF_Stability"] = df["OCF_Margin"].rolling(window).std()
    
    # Investment metrics
    df["Net_Debt"] = df["Debt"] - df["Cash"]
    df["Net_Debt_to_Revenue"] = df["Net_Debt"] / df["Revenue_TTM"]
    
    return df, notes