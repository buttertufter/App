"""File I/O utilities for reading and writing data.

This module provides utilities for working with CSV, JSON, and
other file formats, including cache validation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any

import pandas as pd


def file_age_days(path: Union[str, Path]) -> float:
    """Get the age of a file in days.
    
    Parameters
    ----------
    path : str or Path
        Path to the file.
        
    Returns
    -------
    float
        Age in days. Returns infinity if file doesn't exist.
        
    Examples
    --------
    >>> age = file_age_days('/path/to/file.csv')
    >>> if age < 7:
    ...     print("File is less than a week old")
    """
    p = Path(path)
    if not p.exists():
        return float("inf")
    
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age.total_seconds() / 86400.0


def is_file_fresh(path: Union[str, Path], max_age_days: int = 30) -> bool:
    """Check if a file exists and is fresh (not too old).
    
    Parameters
    ----------
    path : str or Path
        Path to the file.
    max_age_days : int, default=30
        Maximum age in days to consider file fresh.
        
    Returns
    -------
    bool
        True if file exists and is younger than max_age_days.
        
    Examples
    --------
    >>> if is_file_fresh('cache/data.csv', max_age_days=7):
    ...     # Use cached data
    ...     pass
    """
    p = Path(path)
    return p.exists() and file_age_days(p) <= max_age_days


def read_local_csv(
    path: Union[str, Path],
    index_col: Optional[str] = None
) -> pd.DataFrame:
    """Read CSV file into DataFrame.
    
    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    index_col : str, optional
        Column to use as index. If provided and contains date-like values,
        will parse as dates.
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame, or empty DataFrame if file doesn't exist or read fails.
        
    Examples
    --------
    >>> df = read_local_csv('data/prices.csv', index_col='date')
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    
    try:
        parse_dates = [index_col] if index_col else None
        df = pd.read_csv(p, parse_dates=parse_dates)
        
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        
        return df
    except Exception:
        return pd.DataFrame()


def write_local_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    index_name: Optional[str] = None
) -> bool:
    """Write DataFrame to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str or Path
        Destination path.
    index_name : str, optional
        Name for the index column if writing index.
        
    Returns
    -------
    bool
        True if write succeeded, False otherwise.
        
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3]})
    >>> success = write_local_csv(df, 'output/data.csv')
    """
    if df is None or df.empty:
        return False
    
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        if index_name and df.index.name is None:
            df.index.name = index_name
        
        df.to_csv(p, index=True if df.index.name else False)
        return True
    except Exception:
        return False


def read_local_json(path: Union[str, Path]) -> Optional[dict]:
    """Read JSON file into dictionary.
    
    Parameters
    ----------
    path : str or Path
        Path to JSON file.
        
    Returns
    -------
    dict or None
        Parsed JSON data, or None if file doesn't exist or parse fails.
        
    Examples
    --------
    >>> data = read_local_json('config/settings.json')
    >>> if data:
    ...     print(data['key'])
    """
    p = Path(path)
    if not p.exists():
        return None
    
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def write_local_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2
) -> bool:
    """Write data to JSON file.
    
    Parameters
    ----------
    data : Any
        Data to serialize to JSON (must be JSON-serializable).
    path : str or Path
        Destination path.
    indent : int, default=2
        Indentation level for pretty-printing.
        
    Returns
    -------
    bool
        True if write succeeded, False otherwise.
        
    Examples
    --------
    >>> data = {'key': 'value', 'numbers': [1, 2, 3]}
    >>> success = write_local_json(data, 'output/config.json')
    """
    if data is None:
        return False
    
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(p, 'w') as f:
            json.dump(data, f, indent=indent)
        
        return True
    except Exception:
        return False


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str or Path
        Directory path.
        
    Returns
    -------
    Path
        Path object for the directory.
        
    Examples
    --------
    >>> cache_dir = ensure_dir('cache/data')
    >>> file_path = cache_dir / 'myfile.csv'
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
