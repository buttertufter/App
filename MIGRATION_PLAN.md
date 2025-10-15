# Migration Plan: Code Reorganization

## Proposed Folder Structure

```
App/
├── README.md                    # Main documentation
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore patterns
├── requirements.txt             # Python dependencies
├── setup.py                     # Package configuration
├── Makefile                     # Dev automation commands
├── pyproject.toml              # Modern Python project config
│
├── app.py                      # Minimal application bootstrap
│
├── ui/                         # User Interface Layer
│   ├── __init__.py
│   ├── streamlit_app.py       # Main Streamlit UI (from main.py)
│   ├── components/            # Reusable UI components
│   │   ├── __init__.py
│   │   ├── gauge.py           # Gauge visualization
│   │   └── ticker_selector.py # Ticker selection widget
│   └── utils/                 # UI-specific utilities
│       ├── __init__.py
│       └── formatting.py      # Color schemes, formatters
│
├── data/                      # Data Access Layer
│   ├── __init__.py
│   ├── clients/               # API client wrappers
│   │   ├── __init__.py
│   │   ├── fmp_client.py     # Financial Modeling Prep API
│   │   ├── finnhub_client.py # Finnhub API
│   │   └── yfinance_client.py # Yahoo Finance wrapper
│   ├── sources/               # Data source modules
│   │   ├── __init__.py
│   │   ├── financials.py     # Quarterly/annual data (from data.py)
│   │   ├── prices.py         # Price data fetchers
│   │   ├── leadership.py     # Leadership data (from leadership.py)
│   │   ├── insiders.py       # Insider trading data
│   │   └── universe.py       # Universe/peer discovery
│   ├── models/                # Data Transfer Objects
│   │   ├── __init__.py
│   │   ├── company.py        # Company profile DTOs
│   │   └── financials.py     # Financial data DTOs
│   ├── cache/                 # Caching logic
│   │   ├── __init__.py
│   │   ├── file_cache.py     # Local file cache
│   │   └── cache_utils.py    # Cache validation
│   └── adapters/              # Data transformation adapters
│       ├── __init__.py
│       └── financial_adapter.py
│
├── equations/                 # Business Logic & Math (Pure Functions)
│   ├── __init__.py
│   ├── financial/             # Financial calculations
│   │   ├── __init__.py
│   │   ├── growth.py         # Growth metrics (TTM, slope, etc.)
│   │   ├── margins.py        # Margin calculations
│   │   ├── cash_flow.py      # Cash flow metrics
│   │   └── capital.py        # Capital productivity
│   ├── scoring/               # Scoring algorithms
│   │   ├── __init__.py
│   │   ├── modules.py        # Module scoring logic (from model.py)
│   │   ├── wave.py           # Wave score computation
│   │   └── leadership.py     # Leadership scoring
│   ├── statistics/            # Statistical functions
│   │   ├── __init__.py
│   │   ├── transforms.py     # Sigmoid, geometric mean, etc.
│   │   ├── timeseries.py     # EMA, slope, stdev, etc.
│   │   └── normalization.py  # Clipping, normalization
│   └── validation/            # Data validation equations
│       ├── __init__.py
│       └── validators.py     # Validation logic
│
├── core/                      # Shared Infrastructure
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── constants.py          # Application constants (from constants.py)
│   ├── logging_config.py     # Logging setup
│   ├── errors.py             # Custom exceptions
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── datetime_utils.py # Date/time helpers
│       ├── file_utils.py     # File I/O helpers (from utils.py)
│       └── http_utils.py     # HTTP session, retries (from utils.py)
│
├── tests/                     # Test Suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest fixtures
│   ├── fixtures/             # Test data fixtures
│   │   ├── __init__.py
│   │   ├── golden_data.json  # Reference test data
│   │   └── mock_responses.py # Mock API responses
│   ├── unit/                 # Unit tests
│   │   ├── __init__.py
│   │   ├── test_equations.py # Test all equation modules
│   │   ├── test_scoring.py   # Test scoring logic
│   │   └── test_data.py      # Test data adapters
│   └── integration/          # Integration tests
│       ├── __init__.py
│       └── test_pipeline.py  # Test end-to-end flows
│
├── scripts/                   # Development Scripts
│   ├── __init__.py
│   ├── dev.py                # Run development server
│   ├── pipeline.py           # Background data pipeline (from pipeline.py)
│   ├── prefetch.py           # Data prefetch utility (from prefetch.py)
│   ├── mock_data.py          # Generate mock test data
│   ├── migrate_db.py         # DB migration (from migrate_to_duckdb_postgres.py)
│   └── api_server.py         # FastAPI server (from service.py)
│
├── assets/                    # Static assets
│   └── tickers.csv
│
├── data_store/               # Local data cache (gitignored)
├── cache/                    # Temporary cache (gitignored)
└── calibration/              # Model calibration data (gitignored)
```

## Migration Steps

### Phase 1: Create New Structure (Step 1)
- Create all new folders and `__init__.py` files
- Set up proper Python package structure

### Phase 2: Extract Equations (Step 2)
**Source Files:**
- `app/utils.py` → Extract pure math functions
- `app/model.py` → Extract scoring algorithms
- `app/preprocessing.py` → Extract validation logic

**Target Structure:**
```
equations/
├── statistics/
│   ├── transforms.py      # logistic_sigmoid, geo_mean
│   └── timeseries.py      # ema, ttm_sum, slope_log, stdev_over
├── financial/
│   ├── growth.py          # Growth calculations
│   ├── margins.py         # Margin calculations
│   └── capital.py         # Capital productivity
└── scoring/
    ├── modules.py         # compute_modules logic
    └── wave.py            # Wave score computation
```

### Phase 3: Reorganize Data Layer (Step 3)
**Source Files:**
- `app/data.py` → Split into clients/, sources/, cache/
- `app/leadership.py` → Move to data/sources/

**Target Structure:**
```
data/
├── clients/
│   ├── fmp_client.py      # FMP API wrapper
│   ├── finnhub_client.py  # Finnhub wrapper
│   └── yfinance_client.py # YFinance wrapper
├── sources/
│   ├── financials.py      # fetch_quarterlies, fetch_prices
│   ├── leadership.py      # Leadership data fetchers
│   └── universe.py        # Universe management
└── cache/
    └── file_cache.py      # Caching utilities
```

### Phase 4: Reorganize UI (Step 4)
**Source Files:**
- `app/main.py` → Move to `ui/streamlit_app.py`

**Extractions:**
- `render_gauge()` → `ui/components/gauge.py`
- `ticker_options()` → `ui/components/ticker_selector.py`
- `score_to_color()` → `ui/utils/formatting.py`

### Phase 5: Create Core Module (Step 5)
**Source Files:**
- `app/constants.py` → `core/constants.py`
- `app/utils.py` (config parts) → `core/config.py`
- `app/utils.py` (file/http) → `core/utils/`

### Phase 6: Move Scripts (Step 6)
**Relocations:**
- `pipeline.py` → `scripts/pipeline.py`
- `prefetch.py` → `scripts/prefetch.py`
- `app/service.py` → `scripts/api_server.py`
- `app/calibration_runner.py` → `scripts/calibration_runner.py`

### Phase 7: Create Tests (Step 7)
- Move existing test files to `tests/unit/`
- Create golden data fixtures
- Write comprehensive equation tests
- Add mock data generator

### Phase 8: Update All Imports (Step 8)
- Update all imports to use new paths
- Use absolute imports from package root
- Add `__init__.py` exports for clean interfaces

### Phase 9: Add Tooling (Step 9)
- Create `requirements.txt`
- Create `setup.py` and `pyproject.toml`
- Add `Makefile` with dev/test/lint commands
- Configure black, ruff, mypy

### Phase 10: Documentation (Step 10)
- Create comprehensive README.md
- Create .env.example
- Document architecture
- Migration report

### Phase 11: Validation (Step 11)
- Run all tests
- Verify imports
- Test application startup
- Validate CLI commands

## Import Path Examples

**Before:**
```python
from data import fetch_quarterlies
from model import compute_modules
from utils import logistic_sigmoid
```

**After:**
```python
from data.sources.financials import fetch_quarterlies
from equations.scoring.modules import compute_modules
from equations.statistics.transforms import logistic_sigmoid
```

**Or with clean `__init__.py` exports:**
```python
from data import fetch_quarterlies
from equations import compute_modules, logistic_sigmoid
```

## Key Design Decisions

1. **Pure Functions in equations/**: All math/business logic as testable pure functions
2. **Clean Separation**: UI never imports from data clients directly
3. **Adapter Pattern**: Data adapters transform API responses to clean DTOs
4. **Configuration**: All API keys, URLs, settings in `core/config.py` from env
5. **Caching**: Centralized cache logic in `data/cache/`
6. **Type Safety**: Add type hints throughout, validate with mypy
7. **Testing**: Unit tests for equations, mocked tests for data layer

## Breaking Changes

None expected - all functionality preserved, just reorganized.

## TODO Items After Migration

1. Add more comprehensive type hints
2. Implement proper dependency injection for data sources
3. Add API rate limiting middleware
4. Create Docker containerization
5. Add CI/CD pipeline configuration
6. Performance profiling and optimization
