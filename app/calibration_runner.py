"""Calibration runner and analysis tools."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime

from calibration import calibrate_model
from data import get_universe, fetch_quarterlies, fetch_prices
from model import compute_modules
from constants import INITIAL_PARAMS, MODULE_INFO

def analyze_parameter_sensitivity(
    base_params: Dict[str, float],
    param_name: str,
    values: List[float],
    test_symbols: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Analyze how changing a single parameter affects model performance.
    
    Returns DataFrame with parameter values and resulting correlations.
    """
    results = []
    for value in values:
        test_params = base_params.copy()
        test_params[param_name] = value
        
        correlations = []
        for symbol in test_symbols:
            corr = compute_target_correlation(
                symbol, 
                test_params,
                start_date,
                end_date
            )
            if not np.isnan(corr):
                correlations.append(corr)
                
        results.append({
            "param_value": value,
            "mean_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations)
        })
        
    return pd.DataFrame(results)

def plot_sensitivity(df: pd.DataFrame, param_name: str, output_dir: str = "calibration"):
    """Plot parameter sensitivity analysis results."""
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df["param_value"],
        df["mean_correlation"],
        yerr=df["std_correlation"],
        fmt="o-",
        capsize=5
    )
    plt.xlabel(param_name)
    plt.ylabel("Mean Correlation")
    plt.title(f"Sensitivity Analysis: {param_name}")
    plt.grid(True)
    
    out_dir = Path(output_dir)
    plt.savefig(out_dir / f"sensitivity_{param_name}.png")
    plt.close()

def generate_calibration_report(
    params: Dict[str, float],
    metrics: Dict,
    output_dir: str = "calibration"
) -> None:
    """Generate a markdown report summarizing calibration results."""
    out_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Bridge Dashboard Model Calibration Report
Generated: {timestamp}

## Optimization Results
- Final Mean Correlation: {metrics['final_correlation']:.3f}
- Optimization Success: {metrics['optimization_success']}
- Iterations: {metrics['iterations']}
- Message: {metrics['message']}

## Calibrated Parameters
| Parameter | Value | Change from Initial |
|-----------|-------|-------------------|
"""
    
    from constants import INITIAL_PARAMS
    for param, value in params.items():
        initial = INITIAL_VALUES[param]
        pct_change = (value - initial) / initial * 100
        report += f"| {param} | {value:.3f} | {pct_change:+.1f}% |\n"
    
    report += "\n## Parameter Interpretations\n\n"
    
    # Add interpretations
    interpretations = {
        "kappa": "Sigmoid steepness - controls sensitivity of logistic transforms",
        "gamma": "Wave exponent - amplifies signal strength in final score",
        "lambda_d": "Debt contribution to effective investment capacity",
        "lambda_c": "Credit line contribution to investment capacity",
        "lambda_f": "Financing inflow contribution to investment capacity",
        "tau": "Volatility comfort threshold",
        "theta_cp": "Neutral point for capital productivity",
        "w_a_supp": "Support score weight in Growth module",
        "w_a_ops": "Operations score weight in Growth module", 
        "w_a_cp": "Capital productivity weight in Growth module",
        "w_a_mom": "Momentum weight in Growth module",
        "w_a_acc": "Acceleration weight in Growth module",
        "w_b_stab": "Stability weight in Resilience module",
        "w_b_coh": "Coherence weight in Resilience module", 
        "w_b_liq": "Liquidity weight in Resilience module",
        "w_b_nd": "Net debt weight in Resilience module"
    }
    
    for param, value in params.items():
        report += f"### {param}\n"
        report += f"- Value: {value:.3f}\n"
        report += f"- {interpretations[param]}\n\n"
        
    # Save report
    with open(out_dir / "calibration_report.md", "w") as f:
        f.write(report)

def run_calibration_suite(
    n_symbols: int = 100,
    n_validation: int = 20,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "calibration"
) -> None:
    """Run full calibration process with analysis and reporting."""
    
    print("Starting calibration suite...")
    
    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get universe and sample symbols
    universe = get_universe()
    all_symbols = universe["symbol"].tolist()
    
    # Select calibration and validation sets
    np.random.seed(42)
    calibration_symbols = np.random.choice(
        all_symbols,
        size=min(n_symbols, len(all_symbols)),
        replace=False
    ).tolist()
    
    remaining_symbols = list(set(all_symbols) - set(calibration_symbols))
    validation_symbols = np.random.choice(
        remaining_symbols,
        size=min(n_validation, len(remaining_symbols)),
        replace=False
    ).tolist()
    
    print(f"Calibrating model using {len(calibration_symbols)} symbols...")
    print(f"Will validate on {len(validation_symbols)} additional symbols...")
    
    # Run main calibration
    params, correlation, diag = calibrate_model(
        symbols=calibration_symbols,
        start_date=start_date,
        end_date=end_date,
        max_iter=150  # Increased iterations for better convergence
    )
    
    print(f"\nCalibration complete. Mean correlation: {correlation:.3f}")
    
    # Save calibration results
    save_calibration(params, diag, output_dir)
    
    # Run validation on held-out set
    print("\nValidating on held-out data...")
    validation_results = validate_model(
        params=params,
        symbols=validation_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    print("\nValidation Results:")
    print(f"Mean correlation: {validation_results.mean_correlation:.3f} ± {validation_results.std_correlation:.3f}")
    print(f"Mean abs error: {validation_results.mean_error:.3f}")
    print(f"Mean coverage: {validation_results.mean_coverage:.1%}")
    
    # Save validation results
    save_validation_results(validation_results, output_dir)
    
    # Generate sensitivity analysis
    print("\nRunning sensitivity analysis...")
    test_symbols = symbols[:20]  # Use subset for sensitivity
    
    for param_name in ["kappa", "gamma", "tau", "theta_cp"]:
        base = params[param_name]
        values = np.linspace(
            base * 0.5,
            base * 1.5,
            10
        )
        sensitivity_df = analyze_parameter_sensitivity(
            params,
            param_name,
            values,
            test_symbols,
            start_date,
            end_date
        )
        plot_sensitivity(sensitivity_df, param_name, output_dir)
        
    # Generate report
    print("\nGenerating calibration report...")
    generate_calibration_report(params, diag, output_dir)
    
    print("\nCalibration suite complete!")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    run_calibration_suite()