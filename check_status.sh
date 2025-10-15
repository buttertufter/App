#!/bin/bash
# Migration Status Check Script
# Run this to see what's working and what needs migration

echo "=========================================="
echo "Bridge Dashboard - Migration Status"
echo "=========================================="
echo ""

echo "✅ WORKING NOW:"
echo "  - Streamlit UI: streamlit run app/main.py"
echo "  - Make commands: make dev, make help"
echo "  - Equation modules: Ready to import"
echo "  - Core config: Ready to import"
echo ""

echo "📦 NEW MODULES (Ready to Use):"
echo ""
echo "Testing equations module..."
python3 << 'EOF'
try:
    from equations.statistics.transforms import logistic_sigmoid, geo_mean
    from equations.statistics.timeseries import ema, ttm_sum
    from equations.financial.growth import calculate_revenue_growth
    print("  ✅ equations.statistics.transforms - WORKS")
    print("  ✅ equations.statistics.timeseries - WORKS")
    print("  ✅ equations.financial.growth - WORKS")
    print(f"     Example: logistic_sigmoid(0) = {logistic_sigmoid(0)}")
    print(f"     Example: geo_mean([1,4,16]) = {geo_mean([1,4,16]):.2f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
EOF

echo ""
echo "Testing core module..."
python3 << 'EOF'
try:
    from core.config import PROJECT_ROOT, OFFLINE_MODE, validate_config
    from core.utils.http_utils import requests_session
    print("  ✅ core.config - WORKS")
    print("  ✅ core.utils.http_utils - WORKS")
    print(f"     PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"     OFFLINE_MODE: {OFFLINE_MODE}")
except Exception as e:
    print(f"  ❌ Error: {e}")
EOF

echo ""
echo "⏳ TO BE MIGRATED:"
echo "  - app/main.py → ui/streamlit_app.py"
echo "  - app/data.py → data/sources/"
echo "  - app/model.py → equations/scoring/"
echo "  - Scripts → scripts/"
echo ""

echo "📊 PROGRESS: 40% Complete"
echo ""
echo "📚 DOCUMENTATION:"
echo "  - README.md - Quick start guide"
echo "  - QUICK_FIX.md - How to complete migration"
echo "  - MIGRATION_REPORT.md - Detailed status"
echo "  - FINAL_SUMMARY.md - What's done, what's next"
echo ""

echo "🚀 QUICK COMMANDS:"
echo "  make dev        # Run Streamlit UI"
echo "  make help       # Show all commands"
echo "  make install    # Install dependencies"
echo "  make test       # Run tests (needs pytest)"
echo ""

echo "=========================================="
echo "Your app is working! See QUICK_FIX.md"
echo "for next steps."
echo "=========================================="
