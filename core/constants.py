# app/constants.py

"""Constants and default values for Bridge Dashboard."""

import numpy as np

# Analysis windows
DEFAULT_QUARTERS_WINDOW = 4
DEFAULT_YEARS_WINDOW = 3
MIN_QUARTERS_DATA = 4
MIN_ANALYSIS_QUARTERS = 2

# Financial thresholds 
NUMERIC_EPSILON = 1e-9
DEFAULT_CONFIDENCE = 0.95
MIN_OBSERVATIONS = 20

# Wave model parameters
KAPPA_BOUNDS = (1.0, 10.0)
GAMMA_BOUNDS = (0.8, 2.0)
LAMBDA_BOUNDS = (0.0, 1.0)
TAU_BOUNDS = (0.01, 0.20) 
THETA_CP_BOUNDS = (0.0, 0.5)

# Initial model parameters
INITIAL_PARAMS = {
    "kappa": 4.0,        # Sigmoid steepness
    "gamma": 1.15,       # Wave exponent
    "lambda_d": 0.5,     # Debt weight
    "lambda_c": 0.6,     # Credit weight
    "lambda_f": 0.8,     # Financing weight
    "tau": 0.08,         # Volatility threshold
    "theta_cp": 0.2,     # Capital productivity neutral point
    
    # Module A weights
    "w_a_supp": 0.222,   # Support weight (2/9)
    "w_a_ops": 0.222,    # Operations weight (2/9)
    "w_a_cp": 0.222,     # Capital productivity weight (2/9) 
    "w_a_mom": 0.167,    # Momentum weight (1.5/9)
    "w_a_acc": 0.167,    # Acceleration weight (1.5/9)
    
    # Module B weights
    "w_b_stab": 0.4,     # Stability weight (2/5)
    "w_b_coh": 0.2,      # Coherence weight (1/5)
    "w_b_liq": 0.2,      # Liquidity weight (1/5)
    "w_b_nd": 0.2,       # Net debt weight (1/5)
}

# Parameter bounds for optimization
PARAM_BOUNDS = {
    "kappa": KAPPA_BOUNDS,
    "gamma": GAMMA_BOUNDS,
    "lambda_d": LAMBDA_BOUNDS,
    "lambda_c": LAMBDA_BOUNDS,
    "lambda_f": LAMBDA_BOUNDS,
    "tau": TAU_BOUNDS,
    "theta_cp": THETA_CP_BOUNDS,
    
    # Module weights must sum to 1 within each module
    "w_a_supp": LAMBDA_BOUNDS,
    "w_a_ops": LAMBDA_BOUNDS,
    "w_a_cp": LAMBDA_BOUNDS,
    "w_a_mom": LAMBDA_BOUNDS,
    "w_a_acc": LAMBDA_BOUNDS,
    
    "w_b_stab": LAMBDA_BOUNDS,
    "w_b_coh": LAMBDA_BOUNDS,
    "w_b_liq": LAMBDA_BOUNDS,
    "w_b_nd": LAMBDA_BOUNDS,
}

# Module names and descriptions
MODULE_INFO = {
    "A": {
        "name": "Growth & Acceleration",
        "components": ["Support", "Operations", "Capital Productivity", "Momentum", "Acceleration"],
        "weights": [2, 2, 2, 1, 1]
    },
    "B": {
        "name": "Stability & Resilience", 
        "components": ["Stability", "Coherence", "Liquidity", "Net Debt"],
        "weights": [2, 1, 1, 1]
    },
    "C": {
        "name": "Leadership Alignment",
        "components": ["Insider Activity", "Compensation", "Guidance", "Buybacks"],
        "weights": [1, 1, 1, 1]
    },
    "D": {
        "name": "Sector & Network",
        "components": ["Sector Momentum", "Alignment"],
        "weights": [1, 1]
    },
    "E": {
        "name": "Market Uncertainty",
        "components": ["Idiosyncratic Vol", "Price-Growth Alignment"],
        "weights": [1, 1]
    }
}

# Validation configurations
VALIDATION_CONFIG = {
    # Leadership scoring weights
    "leadership_weights": {
        "tenure": 0.2,        # CEO/management tenure
        "compensation": 0.2,  # Compensation alignment
        "insider": 0.3,      # Insider trading patterns
        "performance": 0.3    # Historical performance
    },
    
    # Fallback thresholds
    "min_data_points": 4,    # Minimum data points for reliable analysis
    "stale_data_days": 30,   # Maximum age of cached data
    
    # Score calibration
    "neutral_growth": 0.10,  # Baseline growth rate (10%)
    "neutral_margin": 0.15,  # Baseline operating margin (15%)
    "growth_scaling": 0.25,  # Growth rate scaling factor
    "margin_scaling": 0.15,  # Margin scaling factor
    
    # Uncertainty handling
    "max_missing_ratio": 0.3,  # Maximum allowed missing data ratio
    "min_confidence": 0.8,     # Minimum confidence threshold
}

# Fallback values for missing data
FALLBACKS = {
    # Growth and revenue metrics
    "revenueGrowth": 0.0,            # Neutral growth
    "revenueAcceleration": 0.0,      # No acceleration
    "operatingMargin": 0.15,         # Typical margin
    "margin": 0.25,                  # Operating margin for legacy code
    "cash_flow_margin": 0.15,        # Operating cash flow margin
    
    # Leadership and management
    "leadershipScore": 0.5,          # Neutral leadership
    "insiderConfidence": 0.5,        # Neutral insider sentiment
    
    # Market metrics
    "sectorMomentum": 0.0,           # Neutral sector trend
    "priceMomentum": 0.0,            # Neutral price trend
    "volatility": 0.15,              # Typical volatility
    
    # Financial metrics
    "debtCoverage": 1.0,             # Neutral coverage
    "liquidityRatio": 1.0,           # Neutral liquidity
    "cashFlow": 0.0,                 # Neutral cash flow
    "credit_line": 0.0,              # Undrawn credit (legacy)
    "financing": 0.0,                # Net financing inflow (legacy)
    
    # Core metrics with min/max values
    "minMargin": -0.5,               # Minimum acceptable margin
    "maxMargin": 0.5,                # Maximum realistic margin
    "minGrowth": -0.5,               # Minimum growth rate
    "maxGrowth": 1.0                 # Maximum realistic growth
}

# Required data fields
REQUIRED_FIELDS = [
    "Revenue",
    "OperatingExpenses", 
    "OperatingCashFlow",
    "Cash",
    "Debt"
]

# Optional data fields
OPTIONAL_FIELDS = [
    "UndrawnCredit",
    "FinancingIn"
]