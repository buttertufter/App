"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_quarterly_data():
    """Generate sample quarterly financial data for testing."""
    dates = pd.date_range('2020-01-01', periods=20, freq='Q')
    
    data = {
        'date': dates,
        'Revenue': np.linspace(100, 150, 20) * 1e6,  # Growing revenue
        'OperatingExpenses': np.linspace(60, 90, 20) * 1e6,  # Growing expenses
        'OperatingCashFlow': np.linspace(15, 25, 20) * 1e6,  # Growing OCF
        'Cash': np.linspace(50, 80, 20) * 1e6,  # Growing cash
        'Debt': np.linspace(30, 25, 20) * 1e6,  # Declining debt
        'UndrawnCredit': np.full(20, 20) * 1e6,  # Constant credit line
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    return df


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range('2020-01-01', periods=250, freq='D')
    
    # Generate random walk with upward drift
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    series = pd.Series(prices, index=dates, name='Close')
    return series


@pytest.fixture
def sample_peer_prices():
    """Generate sample peer price data for testing."""
    dates = pd.date_range('2020-01-01', periods=250, freq='D')
    
    peers = {}
    np.random.seed(42)
    
    for peer in ['PEER1', 'PEER2', 'PEER3']:
        returns = np.random.normal(0.0008, 0.018, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        peers[peer] = prices
    
    df = pd.DataFrame(peers, index=dates)
    return df


@pytest.fixture
def golden_data_path():
    """Path to golden test data."""
    return Path(__file__).parent / 'fixtures' / 'golden_data.json'


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API access"
    )
