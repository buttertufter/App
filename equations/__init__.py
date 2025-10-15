"""Equations module initialization.

Exports all equation functions for easy importing.
"""

# Statistical transforms
from equations.statistics.transforms import (
    logistic_sigmoid,
    geo_mean,
    pct_clip,
    sig,  # alias
    geo,  # alias
    EPS,
)

# Time series
from equations.statistics.timeseries import (
    ema,
    ttm_sum,
    slope_log,
    stdev_over,
    to_series_1d,
    last_or_default,
    nth_from_end_or_default,
)

# Financial metrics
from equations.financial.growth import (
    calculate_revenue_growth,
    calculate_revenue_acceleration,
    calculate_ttm_metrics,
)

from equations.financial.margins import (
    calculate_operating_margin,
    calculate_ocf_margin,
    calculate_margins,
)

from equations.financial.capital import (
    calculate_financial_support,
    calculate_capital_productivity,
    calculate_liquidity_ratio,
    calculate_net_debt_ratio,
)

__all__ = [
    # Transforms
    'logistic_sigmoid',
    'geo_mean',
    'pct_clip',
    'sig',
    'geo',
    'EPS',
    # Time series
    'ema',
    'ttm_sum',
    'slope_log',
    'stdev_over',
    'to_series_1d',
    'last_or_default',
    'nth_from_end_or_default',
    # Financial
    'calculate_revenue_growth',
    'calculate_revenue_acceleration',
    'calculate_ttm_metrics',
    'calculate_operating_margin',
    'calculate_ocf_margin',
    'calculate_margins',
    'calculate_financial_support',
    'calculate_capital_productivity',
    'calculate_liquidity_ratio',
    'calculate_net_debt_ratio',
]
