"""Statistical transformation functions.

This module contains pure mathematical transformation functions
used throughout the application. All functions are stateless and side-effect free.
"""

import numpy as np
from typing import Optional, Union


EPS = 1e-6  # Small epsilon to prevent division by zero


def logistic_sigmoid(x: Union[float, np.ndarray], kappa: float = 4.0) -> Union[float, np.ndarray]:
    """Apply logistic sigmoid transformation to map values to [0, 1].
    
    The logistic sigmoid function is defined as:
        σ(x) = 1 / (1 + exp(-kappa * x))
    
    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) to transform. Can be a scalar or array.
    kappa : float, default=4.0
        Steepness parameter. Higher values create a steeper S-curve.
        Range: typically [1.0, 10.0]
    
    Returns
    -------
    float or np.ndarray
        Transformed value(s) in range [0, 1].
        
    Notes
    -----
    - Input is clipped to [-50, 50] to prevent numerical overflow
    - kappa = 4.0 provides a good balance for scoring applications
    - Higher kappa values create more binary outputs (closer to 0 or 1)
    
    Examples
    --------
    >>> logistic_sigmoid(0.0)
    0.5
    >>> logistic_sigmoid(1.0, kappa=4.0)
    0.9820...
    >>> logistic_sigmoid(-1.0, kappa=4.0)
    0.0179...
    """
    x_clipped = np.clip(x, -50, 50)
    result = 1.0 / (1.0 + np.exp(-kappa * x_clipped))
    
    # Return scalar if input was scalar
    if np.isscalar(x):
        return float(result)
    return result


def geo_mean(
    xs: Union[list, np.ndarray],
    ws: Optional[Union[list, np.ndarray]] = None,
    eps: float = EPS
) -> float:
    """Compute weighted geometric mean of positive values.
    
    The geometric mean is defined as:
        GM(x₁, ..., xₙ) = (x₁^w₁ * ... * xₙ^wₙ)^(1/Σwᵢ)
    
    Or equivalently in log space:
        ln(GM) = (Σ wᵢ * ln(xᵢ)) / Σ wᵢ
    
    Parameters
    ----------
    xs : array-like
        Input values. Should be positive numbers.
    ws : array-like, optional
        Weights for each value. If None, all weights are 1.0 (unweighted geometric mean).
    eps : float, default=1e-6
        Small constant added to prevent log(0).
        
    Returns
    -------
    float
        Weighted geometric mean of the input values.
        
    Notes
    -----
    - Computation done in log-space for numerical stability
    - Small epsilon prevents log(0) errors
    - All values should be non-negative
    - Returns geometric mean, which is always ≤ arithmetic mean
    
    Examples
    --------
    >>> geo_mean([1, 4, 16])
    4.0
    >>> geo_mean([2, 8], ws=[1, 1])
    4.0
    >>> geo_mean([2, 8], ws=[3, 1])
    2.828...
    """
    xs = np.asarray(xs, dtype=float)
    
    if ws is None:
        ws = np.ones_like(xs)
    ws = np.asarray(ws, dtype=float)
    
    # Compute in log space for stability
    log_result = np.sum(ws * np.log(xs + eps)) / np.sum(ws)
    return float(np.exp(log_result))


def pct_clip(x: float, lo: float = -0.99, hi: float = 0.99) -> float:
    """Clip percentage values to prevent extreme outliers.
    
    Parameters
    ----------
    x : float
        Value to clip (typically a percentage or ratio).
    lo : float, default=-0.99
        Lower bound.
    hi : float, default=0.99
        Upper bound.
        
    Returns
    -------
    float
        Clipped value in range [lo, hi].
        
    Notes
    -----
    Useful for clipping growth rates, returns, and other percentage metrics
    to prevent extreme outliers from dominating calculations.
    
    Examples
    --------
    >>> pct_clip(0.5)
    0.5
    >>> pct_clip(1.5)
    0.99
    >>> pct_clip(-1.5)
    -0.99
    """
    return float(np.clip(x, lo, hi))


# Convenience aliases
sig = logistic_sigmoid
geo = geo_mean
