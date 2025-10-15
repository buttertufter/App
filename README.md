# Bridge Dashboard 🌉# App

A comprehensive financial analysis platform that combines growth metrics, resilience indicators, leadership assessment, sector positioning, and market uncertainty into a unified "wave score" for investment analysis.

## ✨ Features

- **Multi-Module Scoring**: Combines 5 key dimensions (Growth, Resilience, Leadership, Sector, Uncertainty)
- **Real-Time Data**: Integrates with Financial Modeling Prep, Finnhub, and Yahoo Finance APIs
- **Offline Mode**: Cached data support for offline analysis
- **Interactive UI**: Streamlit-based dashboard with real-time calculations
- **REST API**: FastAPI server for programmatic access
- **Extensible**: Modular architecture makes it easy to add new metrics

## 📁 Project Structure

```
App/
├── ui/                     # User Interface Layer
├── data/                   # Data Access Layer (API clients, caching)
├── equations/              # Business Logic (pure mathematical functions)
├── core/                   # Shared Infrastructure (config, constants, utilities)
├── tests/                  # Test Suite (unit, integration, fixtures)
├── scripts/                # Development Scripts (pipeline, prefetch, API server)
├── data_store/            # Local data cache (gitignored)
└── cache/                 # Temporary cache (gitignored)
```

See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for detailed architecture.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) API keys for FMP and Finnhub

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Run the application
make dev
# or: streamlit run ui/streamlit_app.py
```

### Using Make Commands

```bash
make help       # Show all available commands
make install    # Install dependencies
make dev        # Run Streamlit UI
make run-api    # Run FastAPI server
make test       # Run tests
make lint       # Run linters
make format     # Format code
make prefetch   # Prefetch data for offline use
```

## 📊 Scoring Methodology

### Module A: Growth & Acceleration
Growth metrics, momentum, and acceleration indicators

### Module B: Stability & Resilience  
Volatility, coherence, liquidity, and debt metrics

### Module C: Leadership Alignment
Insider trading, compensation alignment, management quality

### Module D: Sector & Network
Industry trends and peer comparison

### Module E: Market Uncertainty
Volatility and valuation alignment

### Wave Score
```
Wave = (A × B × C × D × E)^(1/5) ^ 1.15
```

## 🧪 Testing

```bash
make test                              # Run all tests
pytest tests/unit/ -v                  # Unit tests
pytest tests/integration/ -v           # Integration tests
pytest --cov=. --cov-report=html      # Coverage report
```

## 📚 Documentation

- **Migration Guide**: [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - Detailed reorganization plan
- **Migration Report**: [MIGRATION_REPORT.md](MIGRATION_REPORT.md) - Progress and status
- **API Docs**: Start server and visit http://localhost:8000/docs
- **Code Docs**: Inline docstrings in NumPy format

## 🔧 Configuration

Create `.env` file (copy from `.env.example`):

```bash
FMP_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
FORCE_OFFLINE=0
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run `make format && make lint && make test`
5. Submit pull request

## 📝 Code Style

- Formatting: Black (100 char line)
- Linting: Ruff
- Type hints: Required for public functions
- Docstrings: NumPy style

## 🐛 Troubleshooting

See detailed troubleshooting in full README at top of this file.

## 📞 Support

- **Issues**: GitHub issue tracker
- **Documentation**: See MD files in repository
- **Migration Questions**: See MIGRATION_REPORT.md

---

**Version**: 2.0.0-alpha  
**Status**: Active Development - Code Reorganization in Progress  
**Last Updated**: 2025-10-15
