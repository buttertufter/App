"""Validation and testing framework for model calibration."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from pathlib import Path
import json

from data import fetch_quarterlies, fetch_prices, get_universe
from model import compute_modules
from preprocessing import validate_financials, preprocess_financials
from constants import MIN_QUARTERS_DATA, DEFAULT_QUARTERS_WINDOW

class ValidationResult:
    """Container for validation metrics."""
    
    def __init__(self):
        self.correlations = []
        self.errors = []
        self.coverage = []
        self.param_stability = {}
        
    def add_fold(
        self,
        correlation: float,
        error: float,
        coverage: float,
        params: Dict[str, float]
    ):
        """Add results from one validation fold."""
        self.correlations.append(correlation)
        self.errors.append(error)
        self.coverage.append(coverage)
        
        # Track parameter stability
        for param, value in params.items():
            if param not in self.param_stability:
                self.param_stability[param] = []
            self.param_stability[param].append(value)
            
    @property
    def mean_correlation(self) -> float:
        """Mean correlation across folds."""
        return float(np.mean(self.correlations))
        
    @property
    def std_correlation(self) -> float:
        """Standard deviation of correlations."""
        return float(np.std(self.correlations))
        
    @property 
    def mean_error(self) -> float:
        """Mean absolute error across folds."""
        return float(np.mean(self.errors))
        
    @property
    def mean_coverage(self) -> float:
        """Mean data coverage ratio."""
        return float(np.mean(self.coverage))
        
    @property
    def param_means(self) -> Dict[str, float]:
        """Mean parameter values across folds."""
        return {
            param: float(np.mean(values))
            for param, values in self.param_stability.items()
        }
        
    @property
    def param_stds(self) -> Dict[str, float]:
        """Parameter standard deviations across folds."""
        return {
            param: float(np.std(values))
            for param, values in self.param_stability.items()
        }
        
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            "correlation": {
                "mean": self.mean_correlation,
                "std": self.std_correlation
            },
            "error": {
                "mean": self.mean_error
            },
            "coverage": {
                "mean": self.mean_coverage
            },
            "parameters": {
                param: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                for param, values in self.param_stability.items()
            }
        }
        
def validate_model(
    params: Dict[str, float],
    symbols: List[str],
    start_date: str,
    end_date: str,
    n_splits: int = 5
) -> ValidationResult:
    """
    Validate model performance using k-fold cross validation.
    
    Parameters
    ----------
    params : dict
        Model parameters to validate
    symbols : list
        Stock symbols to use
    start_date : str
        Start of validation period
    end_date : str
        End of validation period 
    n_splits : int
        Number of validation folds
        
    Returns
    -------
    ValidationResult
        Validation metrics and statistics
    """
    # Get validation data
    validation_data = {}
    for symbol in symbols:
        try:
            q_fin, _ = fetch_quarterlies(symbol, max_quarters=12)
            price, _ = fetch_prices(symbol, years=3)
            
            # Validate data quality
            valid, issues = validate_financials(q_fin, MIN_QUARTERS_DATA)
            if not valid:
                print(f"Skipping {symbol}: {issues}")
                continue
                
            # Preprocess data
            q_fin, notes = preprocess_financials(
                q_fin, 
                window=DEFAULT_QUARTERS_WINDOW
            )
            
            validation_data[symbol] = (q_fin, price)
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {str(e)}")
            continue
            
    # Create cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Run validation
    result = ValidationResult()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(symbols)):
        train_symbols = [symbols[i] for i in train_idx]
        test_symbols = [symbols[i] for i in test_idx]
        
        # Compute validation metrics
        correlations = []
        errors = []
        n_valid = 0
        
        for symbol in test_symbols:
            if symbol not in validation_data:
                continue
                
            q_fin, price = validation_data[symbol]
            
            try:
                # Compute Wave score
                modules, wave, issues = compute_modules(
                    q_fin=q_fin,
                    price=price,
                    peer_prices=pd.DataFrame(),  # Skip peer comparison for validation
                    lambdas=(
                        params["lambda_d"],
                        params["lambda_c"],
                        params["lambda_f"]
                    ),
                    kappa=params["kappa"],
                    theta_cp=params["theta_cp"]
                )
                
                if wave is not None:
                    # Compute forward returns
                    fwd_return = price.pct_change(periods=12).shift(-12)
                    
                    # Align series
                    wave_series = pd.Series(wave, index=price.index) 
                    aligned = pd.concat(
                        [wave_series, fwd_return],
                        axis=1
                    ).dropna()
                    
                    if len(aligned) >= 6:
                        correlation = aligned.corr().iloc[0,1]
                        error = np.mean(np.abs(aligned.iloc[:,0] - aligned.iloc[:,1]))
                        
                        correlations.append(correlation)
                        errors.append(error)
                        n_valid += 1
                        
            except Exception as e:
                print(f"Error validating {symbol}: {str(e)}")
                continue
                
        if correlations:
            result.add_fold(
                correlation=np.mean(correlations),
                error=np.mean(errors),
                coverage=n_valid / len(test_symbols),
                params=params
            )
            
    return result
    
def save_validation_results(
    results: ValidationResult,
    output_dir: str = "calibration"
) -> None:
    """Save validation results to JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    with open(out_dir / f"validation_{timestamp}.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)
        
if __name__ == "__main__":
    # Example validation run
    from constants import INITIAL_PARAMS
    
    universe = get_universe()
    symbols = universe["symbol"].tolist()[:20]  # Test with 20 stocks
    
    print(f"Validating model using {len(symbols)} symbols...")
    
    results = validate_model(
        params=INITIAL_PARAMS,
        symbols=symbols,
        start_date="2018-01-01",
        end_date="2023-12-31"
    )
    
    print("\nValidation Results:")
    print(f"Mean correlation: {results.mean_correlation:.3f} ± {results.std_correlation:.3f}")
    print(f"Mean abs error: {results.mean_error:.3f}")
    print(f"Mean coverage: {results.mean_coverage:.1%}")
    
    print("\nParameter Stability:")
    for param, values in results.param_stability.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{param}: {mean:.3f} ± {std:.3f}")
        
    save_validation_results(results)