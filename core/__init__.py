"""Core module initialization.

Exports commonly used utilities, configuration, and constants.
"""

# Configuration
from core.config import (
    PROJECT_ROOT,
    CACHE_DIR,
    DATA_STORE,
    ASSETS_DIR,
    FMP_API_KEY,
    FINNHUB_API_KEY,
    OFFLINE_MODE,
    STALE_DAYS,
    REQUIRED_COLUMNS,
    is_dotenv_loaded,
    is_offline_mode,
    validate_config,
)

# Constants
from core.constants import (
    DEFAULT_QUARTERS_WINDOW,
    DEFAULT_YEARS_WINDOW,
    MIN_QUARTERS_DATA,
    INITIAL_PARAMS,
    MODULE_INFO,
    FALLBACKS,
)

# HTTP utilities
from core.utils.http_utils import requests_session

# File utilities
from core.utils.file_utils import (
    file_age_days,
    is_file_fresh,
    read_local_csv,
    write_local_csv,
    read_local_json,
    write_local_json,
    ensure_dir,
)

__all__ = [
    # Config
    'PROJECT_ROOT',
    'CACHE_DIR',
    'DATA_STORE',
    'ASSETS_DIR',
    'FMP_API_KEY',
    'FINNHUB_API_KEY',
    'OFFLINE_MODE',
    'STALE_DAYS',
    'REQUIRED_COLUMNS',
    'is_dotenv_loaded',
    'is_offline_mode',
    'validate_config',
    # Constants
    'DEFAULT_QUARTERS_WINDOW',
    'DEFAULT_YEARS_WINDOW',
    'MIN_QUARTERS_DATA',
    'INITIAL_PARAMS',
    'MODULE_INFO',
    'FALLBACKS',
    # Utilities
    'requests_session',
    'file_age_days',
    'is_file_fresh',
    'read_local_csv',
    'write_local_csv',
    'read_local_json',
    'write_local_json',
    'ensure_dir',
]
