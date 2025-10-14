"""Offline-first data access utilities for Bridge Dashboard."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from utils import (
    file_age_days,
    read_local_csv,
    read_local_json,
    requests_session,
    to_series_1d,
    write_local_csv,
    write_local_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_STORE = PROJECT_ROOT / "data_store"
DATA_STORE.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

DOTENV_OK = False
try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(str(PROJECT_ROOT / ".env"))
    DOTENV_OK = True
except Exception:  # pragma: no cover - dotenv optional
    DOTENV_OK = False

FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
OFFLINE_MODE = os.getenv("FORCE_OFFLINE", "0") == "1" or not DOTENV_OK
SESSION = requests_session()

STALE_DAYS = 30
REQUIRED_COLUMNS = [
    "Revenue",
    "OperatingExpenses",
    "OperatingCashFlow",
    "Cash",
    "Debt",
]

DEFAULT_UNIVERSE = pd.DataFrame(
    [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corporation"),
        ("AMZN", "Amazon.com, Inc."),
        ("TSLA", "Tesla, Inc."),
        ("NVDA", "NVIDIA Corporation"),
    ],
    columns=["symbol", "name"],
)

CANONICAL_COLUMNS = [
    "Revenue",
    "OperatingExpenses",
    "OperatingCashFlow",
    "Cash",
    "Debt",
    "UndrawnCredit",
]


def dotenv_status() -> bool:
    return DOTENV_OK


def _local_path(kind: str, symbol: str, ext: str) -> Path:
    return DATA_STORE / kind / f"{symbol.upper()}.{ext}"


def _is_fresh(path: Path, ttl_days: int = STALE_DAYS) -> bool:
    return path.exists() and file_age_days(path) <= ttl_days


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CANONICAL_COLUMNS]

def fill_with_peer_medians(q_fin: pd.DataFrame, symbol: str, universe: pd.DataFrame, depth: int = 0) -> pd.DataFrame:
    if depth > 0:  # Don't fetch peer data for peers (avoid recursion)
        return q_fin
        
    sector = universe.loc[universe["symbol"] == symbol, "sector"].values[0]
    peers = universe[(universe["sector"] == sector) & (universe["symbol"] != symbol)]["symbol"].tolist()
    peer_frames = []
    
    for peer in peers:
        try:
            local_path = _local_path("quarterlies", peer, "csv")
            if local_path.exists():
                df = read_local_csv(local_path)
                if not df.empty:
                    df.index = pd.to_datetime(df["date"]) if "date" in df.columns else df.index
                    peer_frames.append(df)
        except Exception:
            continue
            
    if not peer_frames:
        return q_fin
        
    peer_concat = pd.concat(peer_frames)
    medians = peer_concat.median()
    
    for col in q_fin.columns:
        if q_fin[col].isnull().all():
            q_fin[col] = medians.get(col, 0)
    
    return q_fin


def get_universe() -> pd.DataFrame:
    universe_path = DATA_STORE / "universe.csv"
    if _is_fresh(universe_path):
        df = read_local_csv(universe_path)
        if not df.empty:
            return df
    assets_seed = ASSETS_DIR / "tickers.csv"
    if assets_seed.exists():
        try:
            df = pd.read_csv(assets_seed)
            return df
        except Exception:
            pass
    write_local_csv(DEFAULT_UNIVERSE, universe_path)
    return DEFAULT_UNIVERSE


def sector_etf_map() -> Dict[str, str]:
    return {
        "Communication Services": "XLC",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Financials": "XLF",
        "Health Care": "XLV",
        "Industrials": "XLI",
        "Information Technology": "XLK",
        "Technology": "XLK",
        "Materials": "XLB",
        "Real Estate": "IYR",
        "Utilities": "XLU",
    }


def _fetch_fmp_quarterlies(symbol: str) -> pd.DataFrame:
    """Fetch quarterly financial data from Financial Modeling Prep API."""
    if not FMP_API_KEY:
        print("No FMP API key found")
        return pd.DataFrame()
    
    print(f"\n=== Fetching FMP data for {symbol} ===")
    try:
        # Fetch income statement
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&limit=400&apikey={FMP_API_KEY}"
        balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=quarter&limit=400&apikey={FMP_API_KEY}"
        cashflow_url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=quarter&limit=400&apikey={FMP_API_KEY}"
        
        income_resp = SESSION.get(income_url)
        income_data = income_resp.json()
        print(f"\nFMP Income Statement:")
        print(f"Response: {income_resp.status_code}")
        if isinstance(income_data, list):
            print(f"Quarters: {len(income_data)}")
            if len(income_data) > 0:
                print(f"Date range: {income_data[-1].get('date', 'N/A')} to {income_data[0].get('date', 'N/A')}")
        
        balance_resp = SESSION.get(balance_url)
        balance_data = balance_resp.json()
        print(f"\nFMP Balance Sheet:")
        print(f"Response: {balance_resp.status_code}")
        if isinstance(balance_data, list):
            print(f"Quarters: {len(balance_data)}")
            if len(balance_data) > 0:
                print(f"Date range: {balance_data[-1].get('date', 'N/A')} to {balance_data[0].get('date', 'N/A')}")
        
        cashflow_resp = SESSION.get(cashflow_url)
        cashflow_data = cashflow_resp.json()
        print(f"\nFMP Cash Flow:")
        print(f"Response: {cashflow_resp.status_code}")
        if isinstance(cashflow_data, list):
            print(f"Quarters: {len(cashflow_data)}")
            if len(cashflow_data) > 0:
                print(f"Date range: {cashflow_data[-1].get('date', 'N/A')} to {cashflow_data[0].get('date', 'N/A')}")
        
        # Convert to DataFrames
        income_df = pd.DataFrame(income_data).set_index('date') if isinstance(income_data, list) and income_data else pd.DataFrame()
        balance_df = pd.DataFrame(balance_data).set_index('date') if isinstance(balance_data, list) and balance_data else pd.DataFrame()
        cashflow_df = pd.DataFrame(cashflow_data).set_index('date') if isinstance(cashflow_data, list) and cashflow_data else pd.DataFrame()
        
        if income_df.empty and balance_df.empty:
            return pd.DataFrame()
        
        # Get all dates and sort
        all_dates = sorted(set(income_df.index) | set(balance_df.index) | set(cashflow_df.index))
        
        # Create final dataframe
        result = pd.DataFrame(index=all_dates)
        result.index = pd.to_datetime(result.index)
        
        # Map FMP fields to our standard names
        result['Revenue'] = income_df['revenue'] if not income_df.empty else np.nan
        result['OperatingExpenses'] = income_df['operatingExpenses'] if not income_df.empty else np.nan
        result['OperatingCashFlow'] = cashflow_df['operatingCashFlow'] if not cashflow_df.empty else np.nan
        result['Cash'] = balance_df['cashAndCashEquivalents'] if not balance_df.empty else np.nan
        result['Debt'] = balance_df['totalDebt'] if not balance_df.empty else np.nan
        
        return result.sort_index()
    except Exception as e:
        print(f"FMP API error: {str(e)}")
        return pd.DataFrame()

def _fetch_yahoo_quarterlies(symbol: str) -> pd.DataFrame:
    print(f"\n=== Fetching Yahoo Finance data for {symbol} ===")
    ticker = yf.Ticker(symbol)
    def frame(tbl: Optional[pd.DataFrame]) -> pd.DataFrame:
        if tbl is None or tbl.empty:
            return pd.DataFrame()
        df = tbl.transpose()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna(how="all")
        print(f"Quarters: {len(df)}")
        if not df.empty:
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df

    income = frame(ticker.quarterly_financials)
    balance = frame(ticker.quarterly_balance_sheet)
    cashflow = frame(ticker.quarterly_cashflow)
    if income.empty and balance.empty:
        return pd.DataFrame()

    index = income.index
    for df in (balance.index if not balance.empty else [], cashflow.index if not cashflow.empty else []):
        index = index.union(df)
    index = index.sort_values()

    def pick(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
        if df.empty:
            return pd.Series(np.nan, index=index)
        for name in names:
            if name in df.columns:
                series = df[name]
                series.index = pd.to_datetime(series.index, errors="coerce")
                return to_series_1d(series).reindex(index)
        return pd.Series(np.nan, index=index)

    revenue = pick(income, ["Total Revenue", "Revenue", "Operating Revenue"])
    op_income = pick(income, ["Operating Income"])
    cogs = pick(income, ["Cost Of Revenue", "Cost of Revenue", "Cost of Goods Sold"])
    op_expense = pick(income, ["Operating Expense", "Operating Expenses"])
    if op_expense.dropna().empty:
        op_expense = revenue - op_income.fillna(0.0) - cogs.fillna(0.0)
        op_expense = op_expense.fillna(revenue * 0.75)

    ocf = pick(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    cash = pick(balance, ["Cash And Cash Equivalents", "Cash And Short Term Investments", "Cash"])
    short_debt = pick(balance, ["Short Long Term Debt", "Short Term Debt", "Current Debt"])
    long_debt = pick(balance, ["Long Term Debt", "LongTerm Debt", "Long Term Debt Noncurrent"])
    total_debt = pick(balance, ["Total Debt"])
    debt = total_debt
    if debt.dropna().empty:
        debt = short_debt.fillna(0.0) + long_debt.fillna(0.0)

    frame = pd.DataFrame(
        {
            "Revenue": revenue,
            "OperatingExpenses": op_expense,
            "OperatingCashFlow": ocf,
            "Cash": cash,
            "Debt": debt,
            "UndrawnCredit": np.nan,
        },
        index=index,
    )
    frame.index.name = "date"
    return frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def fetch_quarterlies(
    symbol: str,
    force_refresh: bool = False,
    max_quarters: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("quarterlies", symbol, "csv")
    warnings: List[str] = []

    if not force_refresh and _is_fresh(local_path):
        df = read_local_csv(local_path)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            df = _ensure_columns(df)
            if max_quarters:
                df = df.tail(max_quarters)
            return df, warnings

    if OFFLINE_MODE:
        if local_path.exists():
            df = read_local_csv(local_path)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
                df = _ensure_columns(df)
                if max_quarters:
                    df = df.tail(max_quarters)
                warnings.append("Offline mode: using stored fundamentals.")
                return df, warnings
        raise ValueError(f"Offline mode: no local fundamentals for {symbol}.")

    # Initialize empty DataFrame to store merged data
    merged_df = pd.DataFrame()
    
    # Get FMP data
    try:
        fmp_df = _fetch_fmp_quarterlies(symbol)
        if not fmp_df.empty:
            warnings.append(f"FMP data: {len(fmp_df)} quarters ({fmp_df.index.min()} to {fmp_df.index.max()})")
            merged_df = merged_df._append(fmp_df)
    except Exception as exc:
        warnings.append(f"FMP data fetch failed: {exc}")
    
    # Get Yahoo Finance data
    try:
        yahoo_df = _fetch_yahoo_quarterlies(symbol)
        if not yahoo_df.empty:
            warnings.append(f"Yahoo data: {len(yahoo_df)} quarters ({yahoo_df.index.min()} to {yahoo_df.index.max()})")
            merged_df = merged_df._append(yahoo_df)
    except Exception as exc:
        warnings.append(f"Yahoo data fetch failed: {exc}")
    
    if not merged_df.empty:
        # Remove any duplicate indices (dates) by keeping the first occurrence
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        # Sort by date
        merged_df = merged_df.sort_index()
        # Fill missing values forward and backward within each source
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        
        warnings.append(f"Combined data: {len(merged_df)} quarters ({merged_df.index.min()} to {merged_df.index.max()})")
        
        # Apply max_quarters filter if specified
        if max_quarters:
            merged_df = merged_df.tail(max_quarters)
        
        # Cache the merged data
        export = merged_df.reset_index()
        write_local_csv(export, local_path)
        
        return merged_df, warnings
    else:
        warnings.append(f"No quarterly fundamentals available for {symbol}.")
        return pd.DataFrame(), warnings

    df = _ensure_columns(df)
    df = df.dropna(how="all")
    if max_quarters:
        df = df.tail(max_quarters)

    if df.dropna(how="all").shape[0] < 4:
        raise ValueError(f"Quarterly data insufficient for {symbol} (need >=4 rows).")

    # Fill with peer medians only for the primary stock request
    df = fill_with_peer_medians(df, symbol, get_universe(), depth=0)

    export = df.reset_index()
    write_local_csv(export, local_path)
    return df, warnings


def fetch_prices(
    symbol: str,
    years: int = 5,
    interval: str = "1wk",
    force_refresh: bool = False,
) -> Tuple[pd.Series, List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("prices", symbol, "csv")
    warnings: List[str] = []

    if not force_refresh and _is_fresh(local_path):
        df = read_local_csv(local_path)
        if not df.empty and "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            series = pd.Series(df["close"].values, index=df["date"], name=str(symbol))
            return series, warnings

    if OFFLINE_MODE:
        if local_path.exists():
            df = read_local_csv(local_path)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                series = pd.Series(df["close"].values, index=df["date"], name=str(symbol))
                warnings.append("Offline mode: using stored prices.")
                return series, warnings
        raise ValueError(f"Offline mode: no local price data for {symbol}.")

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No price data for {symbol}")
    
    # Convert to series and ensure 1D
    series = pd.Series(data["Close"].values.flatten(), index=data.index, name=str(symbol))
    series = series.dropna()
    
    export = pd.DataFrame({"date": series.index, "close": series.values})
    write_local_csv(export, local_path)
    return series, warnings


def fetch_peer_prices(peers: Sequence[str], years: int = 5, force_refresh: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    matrix = []
    warnings: List[str] = []
    for peer in peers or []:
        try:
            series, warn = fetch_prices(peer, years=years, force_refresh=force_refresh)
            warnings.extend(warn)
            if not series.empty:
                matrix.append(series.rename(peer))
        except Exception as exc:
            warnings.append(f"Peer {peer}: {exc}")
    if not matrix:
        return pd.DataFrame(), warnings
    df = pd.concat(matrix, axis=1).dropna(how="all")
    return df, warnings


def fetch_profile(symbol: str, force_refresh: bool = False) -> Tuple[Optional[dict], List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("profiles", symbol, "json")
    warnings: List[str] = []

    if not force_refresh and _is_fresh(local_path):
        data = read_local_json(local_path)
        if data:
            return data, warnings

    if OFFLINE_MODE:
        data = read_local_json(local_path)
        if data:
            warnings.append("Offline mode: using stored profile.")
            return data, warnings
        return None, ["Offline mode: no local profile data."]

    try:
        info = yf.Ticker(symbol).get_info()
    except Exception as exc:
        return None, [f"Yahoo profile failed: {exc}"]

    profile = {
        "symbol": symbol,
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "beta": info.get("beta"),
        "sharesOutstanding": info.get("sharesOutstanding"),
        "updated": datetime.utcnow().isoformat(),
    }
    write_local_json(profile, local_path)
    return profile, warnings


def discover_peers(symbol: str, universe: Optional[pd.DataFrame] = None, force_refresh: bool = False) -> Tuple[List[str], List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("peers", symbol, "csv")
    warnings: List[str] = []

    if not force_refresh and _is_fresh(local_path):
        df = read_local_csv(local_path)
        if not df.empty and "peer" in df.columns:
            return df["peer"].dropna().astype(str).str.upper().tolist(), warnings

    universe_df = universe if universe is not None else get_universe()
    peers: List[str] = []

    profile, profile_w = fetch_profile(symbol, force_refresh=force_refresh and not OFFLINE_MODE)
    warnings.extend(profile_w)
    sector = profile.get("sector") if profile else None

    if sector and "symbol" in universe_df and "sector" in universe_df:
        matches = universe_df[(universe_df["symbol"].str.upper() != symbol) & (universe_df["sector"].fillna("").str.lower() == sector.lower())]
        peers = matches.head(5)["symbol"].str.upper().tolist()
    else:
        peers = universe_df[universe_df["symbol"].str.upper() != symbol].head(5)["symbol"].str.upper().tolist()

    if peers:
        write_local_csv(pd.DataFrame({"peer": peers}), local_path)
    else:
        warnings.append("Unable to determine peers; please specify manually.")
    return peers, warnings


def fetch_sector_series(symbol: str) -> Tuple[Optional[pd.Series], List[str]]:
    profile, warnings = fetch_profile(symbol)
    if not profile or not profile.get("sector"):
        warnings.append("Sector unavailable; cannot load sector ETF.")
        return None, warnings
    etf = sector_etf_map().get(profile["sector"]) or sector_etf_map().get(profile["sector"].title())
    if not etf:
        warnings.append(f"No sector ETF mapping for {profile['sector']}.")
        return None, warnings
    try:
        series, price_warn = fetch_prices(etf, years=5)
        warnings.extend(price_warn)
        return series.rename(etf), warnings
    except Exception as exc:
        warnings.append(f"Failed to load sector ETF {etf}: {exc}")
        return None, warnings


def fetch_estimates(symbol: str, force_refresh: bool = False) -> Tuple[Optional[dict], List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("estimates", symbol, "json")
    if not force_refresh:
        data = read_local_json(local_path)
        if data:
            return data, []
    if OFFLINE_MODE or not FINNHUB_API_KEY:
        return None, ["Estimates unavailable offline."]
    
    warnings = []
    try:
        url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = SESSION.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                write_local_json(data, local_path)
                return data, warnings
    except Exception as e:
        warnings.append(f"Failed to fetch estimates: {str(e)}")
    
    return None, warnings


def fetch_insiders(symbol: str, profile: Optional[dict] = None, force_refresh: bool = False) -> Tuple[Optional[dict], List[str]]:
    symbol = symbol.upper()
    local_path = _local_path("insiders", symbol, "json")
    if not force_refresh:
        data = read_local_json(local_path)
        if data:
            return data, []
    if OFFLINE_MODE or not FINNHUB_API_KEY:
        return None, ["Insider data unavailable offline."]
    
    warnings = []
    try:
        url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_API_KEY}"
        response = SESSION.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data:
                write_local_json(data, local_path)
                return data, warnings
    except Exception as e:
        warnings.append(f"Failed to fetch insider data: {str(e)}")
    
    return None, warnings


__all__ = [
    "dotenv_status",
    "get_universe",
    "sector_etf_map",
    "discover_peers",
    "fetch_profile",
    "fetch_quarterlies",
    "fetch_prices",
    "fetch_peer_prices",
    "fetch_sector_series",
    "fetch_estimates",
    "fetch_insiders",
]
