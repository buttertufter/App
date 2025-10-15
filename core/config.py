"""Application configuration management.

Centralizes all configuration including API keys, paths, and settings.
All secrets are loaded from environment variables.
"""

import os
from pathlib import Path
from typing import Optional

# Try to load dotenv (optional dependency)
_DOTENV_LOADED = False
try:
    from dotenv import load_dotenv
    
    # Load from .env file in project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
        _DOTENV_LOADED = True
except ImportError:
    pass


# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_STORE = PROJECT_ROOT / "data_store"
ASSETS_DIR = PROJECT_ROOT / "assets"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_STORE.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API Configuration
# =============================================================================

# Financial Modeling Prep API
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_MAX_QUARTERS = 400

# Finnhub API
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Alpha Vantage API (if needed in future)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()

# SEC API
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Company Research Tool")


# =============================================================================
# Data Fetching Configuration
# =============================================================================

# Cache settings
STALE_DAYS = 30  # Number of days before cached data is considered stale
FORCE_OFFLINE = os.getenv("FORCE_OFFLINE", "0") == "1"
OFFLINE_MODE = FORCE_OFFLINE or not _DOTENV_LOADED

# HTTP settings
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_FACTOR = 0.5
HTTP_TIMEOUT = 30  # seconds
HTTP_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]


# =============================================================================
# Data Validation
# =============================================================================

REQUIRED_COLUMNS = [
    "Revenue",
    "OperatingExpenses",
    "OperatingCashFlow",
    "Cash",
    "Debt",
]

CANONICAL_COLUMNS = [
    "Revenue",
    "OperatingExpenses",
    "OperatingCashFlow",
    "Cash",
    "Debt",
    "UndrawnCredit",
]


# =============================================================================
# Utility Functions
# =============================================================================

def is_dotenv_loaded() -> bool:
    """Check if dotenv was successfully loaded."""
    return _DOTENV_LOADED


def is_offline_mode() -> bool:
    """Check if application is running in offline mode."""
    return OFFLINE_MODE


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service.
    
    Parameters
    ----------
    service : str
        Service name: 'fmp', 'finnhub', 'alphavantage'
        
    Returns
    -------
    str or None
        API key if available, None otherwise
    """
    keys = {
        'fmp': FMP_API_KEY,
        'finnhub': FINNHUB_API_KEY,
        'alphavantage': ALPHA_VANTAGE_API_KEY,
    }
    
    key = keys.get(service.lower())
    return key if key else None


def validate_config() -> dict:
    """Validate configuration and return status.
    
    Returns
    -------
    dict
        Configuration status including missing keys, paths, etc.
    """
    status = {
        'dotenv_loaded': _DOTENV_LOADED,
        'offline_mode': OFFLINE_MODE,
        'api_keys': {
            'fmp': bool(FMP_API_KEY),
            'finnhub': bool(FINNHUB_API_KEY),
        },
        'paths': {
            'project_root': PROJECT_ROOT.exists(),
            'cache': CACHE_DIR.exists(),
            'data_store': DATA_STORE.exists(),
            'assets': ASSETS_DIR.exists(),
        }
    }
    
    return status


# =============================================================================
# Environment Setup
# =============================================================================

# Print configuration status on import (optional, for debugging)
if os.getenv("DEBUG_CONFIG", "0") == "1":
    print("=" * 60)
    print("Configuration Status")
    print("=" * 60)
    for key, value in validate_config().items():
        print(f"{key}: {value}")
    print("=" * 60)
