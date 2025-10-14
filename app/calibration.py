"""Empirical model calibration framework for Bridge Dashboard."""

import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

from data import fetch_quarterlies, fetch_prices, fetch_peer_prices, discover_peers, get_universe
from model import compute_modules
from utils import logistic_sigmoid, geo_mean, EPS

# Type hints
ParamVector = np.ndarray
OptimResult = Tuple[Dict[str, float], float, Dict[str, Dict]]

# Parameter bounds and initial values
PARAM_BOUNDS = {
    "kappa": (1.0, 10.0),       # Sigmoid steepness
    "gamma": (0.8, 2.0),        # Wave exponent
    "lambda_d": (0.0, 1.0),     # Debt weight
    "lambda_c": (0.0, 1.0),     # Credit weight  
    "lambda_f": (0.0, 1.0),     # Financing weight
    "tau": (0.01, 0.20),        # Volatility threshold
    "theta_cp": (0.0, 0.5),     # Capital productivity neutral point
    # Module weights (must sum to 1 within each module)
    "w_a_supp": (0.0, 1.0),     # Support weight in Module A
    "w_a_ops": (0.0, 1.0),      # Operations weight in Module A 
    "w_a_cp": (0.0, 1.0),       # Capital productivity weight in Module A
    "w_a_mom": (0.0, 1.0),      # Momentum weight in Module A
    "w_a_acc": (0.0, 1.0),      # Acceleration weight in Module A
    "w_b_stab": (0.0, 1.0),     # Stability weight in Module B
    "w_b_coh": (0.0, 1.0),      # Coherence weight in Module B
    "w_b_liq": (0.0, 1.0),      # Liquidity weight in Module B
    "w_b_nd": (0.0, 1.0),       # Net debt weight in Module B
}

INITIAL_VALUES = {
    "kappa": 4.0,
    "gamma": 1.15,
    "lambda_d": 0.5,
    "lambda_c": 0.6,
    "lambda_f": 0.8,
    "tau": 0.08,
    "theta_cp": 0.2,
    "w_a_supp": 0.222,  # 2/9
    "w_a_ops": 0.222,   # 2/9
    "w_a_cp": 0.222,    # 2/9
    "w_a_mom": 0.167,   # 1.5/9
    "w_a_acc": 0.167,   # 1.5/9
    "w_b_stab": 0.4,    # 2/5
    "w_b_coh": 0.2,     # 1/5
    "w_b_liq": 0.2,     # 1/5
    "w_b_nd": 0.2,      # 1/5
}

def normalize_weights(params: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights within each module to sum to 1."""
    normalized = params.copy()
    
    # Module A weights
    w_a_sum = sum(params[k] for k in ["w_a_supp", "w_a_ops", "w_a_cp", "w_a_mom", "w_a_acc"])
    for k in ["w_a_supp", "w_a_ops", "w_a_cp", "w_a_mom", "w_a_acc"]:
        normalized[k] = params[k] / (w_a_sum + EPS)
        
    # Module B weights  
    w_b_sum = sum(params[k] for k in ["w_b_stab", "w_b_coh", "w_b_liq", "w_b_nd"])
    for k in ["w_b_stab", "w_b_coh", "w_b_liq", "w_b_nd"]:
        normalized[k] = params[k] / (w_b_sum + EPS)
        
    return normalized

def compute_target_correlation(
    symbol: str,
    params: Dict[str, float],
    start_date: str,
    end_date: str,
    forward_months: int = 12,
    min_obs: int = 6
) -> float:
    """
    Compute correlation between model Wave score and forward returns.
    
    Returns:
    --------
    float : Correlation coefficient between -1 and 1
    """
    try:
        # Fetch required data
        q_fin, _ = fetch_quarterlies(symbol, max_quarters=12)
        price, _ = fetch_prices(symbol, years=3)
        universe = get_universe()
        peers, _ = discover_peers(symbol, universe)
        peer_prices, _ = fetch_peer_prices(peers, years=3)
        
        # Compute Wave score with given parameters
        modules, wave, _ = compute_modules(
            q_fin=q_fin,
            price=price,
            peer_prices=peer_prices,
            lambdas=(params["lambda_d"], params["lambda_c"], params["lambda_f"]),
            kappa=params["kappa"],
            theta_cp=params["theta_cp"]
        )
        
        if wave is None:
            return 0.0
            
        # Compute forward returns
        returns = compute_forward_returns(price, months=forward_months)
        
        # Align Wave scores with forward returns
        wave_series = pd.Series(wave, index=price.index)
        aligned = pd.concat([wave_series, returns], axis=1).dropna()
        
        if len(aligned) < 4:
            return 0.0
            
        return float(np.corrcoef(aligned.iloc[:,0], aligned.iloc[:,1])[0,1])
        
    except Exception as e:
        print(f"Error computing correlation for {symbol}: {str(e)}")
        return 0.0

def compute_forward_returns(price: pd.Series, months: int = 12) -> pd.Series:
    """Compute forward total returns over specified months."""
    return price.shift(-months) / price - 1

def objective_function(param_vector: ParamVector,
                      param_names: List[str],
                      symbols: List[str],
                      start_date: str,
                      end_date: str) -> float:
    """
    Objective function to minimize:
    -1 * mean(correlation(Wave, forward_returns))
    """
    # Convert parameter vector to dictionary
    params = {name: value for name, value in zip(param_names, param_vector)}
    params = normalize_weights(params)
    
    # Compute correlation for each symbol
    correlations = []
    for symbol in symbols:
        corr = compute_target_correlation(symbol, params, start_date, end_date)
        if not np.isnan(corr):
            correlations.append(corr)
            
    if not correlations:
        return 0.0
        
    # Return negative mean correlation (since we minimize)
    return -1 * np.mean(correlations)

def calibrate_model(
    symbols: List[str],
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31", 
    max_iter: int = 100,
    min_valid_tickers: int = 3
) -> OptimResult:
    """
    Calibrate model parameters using historical data.
    
    Parameters:
    -----------
    symbols : List[str]
        List of stock symbols to use for calibration
    start_date : str
        Start date for historical data
    end_date : str
        End date for historical data 
    max_iter : int
        Maximum iterations for optimizer
        
    Returns:
    --------
    tuple:
        - Optimized parameters dictionary
        - Final objective value 
        - Diagnostics dictionary
    """
    # Setup optimization
    param_names = list(INITIAL_VALUES.keys())
    x0 = np.array([INITIAL_VALUES[p] for p in param_names])
    bounds = optimize.Bounds(
        [PARAM_BOUNDS[p][0] for p in param_names],
        [PARAM_BOUNDS[p][1] for p in param_names]
    )
    
    # Run optimization with multiple restarts
    results = []
    scores = []
    
    # Define different starting points
    start_points = [
        x0,  # Original starting point
        x0 * 0.8,  # Lower values
        x0 * 1.2,  # Higher values
        np.random.uniform(low=0.8, high=1.2, size=len(x0)) * x0  # Random perturbation
    ]
    
    for x_start in start_points:
        result = optimize.minimize(
            objective_function,
            x_start,
            args=(param_names, symbols, start_date, end_date),
            method="L-BFGS-B", 
            bounds=bounds,
            options={
                "maxiter": max_iter,
                "ftol": 1e-6,
                "gtol": 1e-6
            }
        )
        results.append(result)
        scores.append(result.fun)
    
    # Select best result
    best_idx = np.argmin(scores)
    result = results[best_idx]
    
    # Extract results
    optimized_params = {
        name: float(value) 
        for name, value in zip(param_names, result.x)
    }
    optimized_params = normalize_weights(optimized_params)
    
    # Compute final statistics
    final_corr = -1 * objective_function(
        result.x,
        param_names,
        symbols,
        start_date,
        end_date
    )
    
    diagnostics = {
        "optimization_success": bool(result.success),
        "iterations": int(result.nit),
        "final_correlation": float(final_corr),
        "message": str(result.message)
    }
    
    return optimized_params, final_corr, diagnostics

def save_calibration(params: Dict[str, float], 
                    metrics: Dict,
                    output_dir: str = "calibration") -> None:
    """Save calibration results to JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save parameters
    with open(out_dir / f"params_{timestamp}.json", "w") as f:
        json.dump(params, f, indent=2)
        
    # Save metrics
    with open(out_dir / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
if __name__ == "__main__":
    # Example calibration run
    universe = get_universe()
    symbols = universe["symbol"].tolist()[:50]  # Start with top 50
    
    print(f"Calibrating model using {len(symbols)} symbols...")
    
    params, correlation, diag = calibrate_model(
        symbols=symbols,
        start_date="2018-01-01",
        end_date="2023-12-31"
    )
    
    print("\nCalibration Results:")
    print(f"Mean correlation: {correlation:.3f}")
    print("\nOptimized Parameters:")
    for k, v in params.items():
        print(f"{k}: {v:.3f}")
        
    save_calibration(params, diag)