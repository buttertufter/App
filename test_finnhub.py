#!/usr/bin/env python3
"""Test script to debug Finnhub data fetching."""

import sys
sys.path.insert(0, '/workspaces/App')
sys.path.insert(0, '/workspaces/App/app')

from app.data import _fetch_finnhub_quarterlies

# Test with MSFT
print("Testing Finnhub data fetch for MSFT...")
df = _fetch_finnhub_quarterlies("MSFT")

print("\n" + "="*60)
print("FINAL RESULT:")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns) if not df.empty else 'Empty'}")
if not df.empty:
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print("\nFirst few rows:")
    print(df.head())
