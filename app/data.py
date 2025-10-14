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
    """Fetch up to 400 quarters of financial data from FMP (Financial Modeling Prep)."""
    if not FMP_API_KEY:
        print("No FMP API key found")
        return pd.DataFrame()
    
    print(f"\n=== Fetching FMP data for {symbol} ===")
    try:
        base_url = "https://financialmodelingprep.com/api/v3"
        params = {"apikey": FMP_API_KEY, "limit": 400}
        
        # Fetch income statement
        income_url = f"{base_url}/income-statement/{symbol}"
        income_resp = SESSION.get(income_url, params={**params, "period": "quarter"})
        print(f"FMP Income Statement: Response: {income_resp.status_code}")
        if income_resp.status_code == 403:
            print(f"FMP API key may be invalid or expired. Consider checking your subscription at https://financialmodelingprep.com/")
            print(f"Response: {income_resp.text[:200]}")
            return pd.DataFrame()
        
        # Fetch balance sheet
        balance_url = f"{base_url}/balance-sheet-statement/{symbol}"
        balance_resp = SESSION.get(balance_url, params={**params, "period": "quarter"})
        print(f"FMP Balance Sheet: Response: {balance_resp.status_code}")
        if balance_resp.status_code == 403:
            return pd.DataFrame()
        
        # Fetch cash flow
        cashflow_url = f"{base_url}/cash-flow-statement/{symbol}"
        cashflow_resp = SESSION.get(cashflow_url, params={**params, "period": "quarter"})
        print(f"FMP Cash Flow: Response: {cashflow_resp.status_code}")
        if cashflow_resp.status_code == 403:
            return pd.DataFrame()
        
        # Parse responses
        income_data = income_resp.json() if income_resp.status_code == 200 else []
        balance_data = balance_resp.json() if balance_resp.status_code == 200 else []
        cashflow_data = cashflow_resp.json() if cashflow_resp.status_code == 200 else []
        
        if not income_data:
            print("No income data from FMP")
            return pd.DataFrame()
        
        # Create dictionaries indexed by date for easy merging
        balance_dict = {item['date']: item for item in balance_data} if balance_data else {}
        cashflow_dict = {item['date']: item for item in cashflow_data} if cashflow_data else {}
        
        # Combine all data
        quarters = []
        for income in income_data:
            date = income.get('date')
            if not date:
                continue
                
            balance = balance_dict.get(date, {})
            cashflow = cashflow_dict.get(date, {})
            
            quarter_data = {
                'date': date,
                'Revenue': income.get('revenue', np.nan),
                'OperatingExpenses': income.get('operatingExpenses', np.nan),
                'OperatingCashFlow': cashflow.get('operatingCashFlow', np.nan),
                'Cash': balance.get('cashAndCashEquivalents', np.nan),
                'Debt': balance.get('totalDebt', np.nan)
            }
            quarters.append(quarter_data)
        
        if not quarters:
            print("No quarterly data parsed from FMP")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(quarters)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        print(f"FMP data retrieved:")
        print(f"Quarters: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"FMP API error: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def _fetch_finnhub_quarterlies(symbol: str) -> pd.DataFrame:
    """Fetch quarterly financial data from Finnhub."""
    if not FINNHUB_API_KEY:
        print("No Finnhub API key found")
        return pd.DataFrame()
    
    print(f"\n=== Fetching Finnhub data for {symbol} ===")
    try:
        # Construct API URLs for different financial statements
        base_url = "https://finnhub.io/api/v1"
        headers = {"X-Finnhub-Token": FINNHUB_API_KEY}
        
        # Fetch income statement (up to 4 years of quarterly data)
        income_url = f"{base_url}/stock/financials-reported?symbol={symbol}&freq=quarterly"
        income_resp = SESSION.get(income_url, headers=headers)
        print(f"Income statement response: {income_resp.status_code}")
        
        if income_resp.status_code != 200:
            print(f"Finnhub API error: {income_resp.text}")
            return pd.DataFrame()
            
        response_data = income_resp.json()
        print(f"Finnhub response keys: {list(response_data.keys())}")
        
        data = response_data.get('data', [])
        if not data:
            print(f"No data returned from Finnhub. Full response: {response_data}")
            return pd.DataFrame()
        
        print(f"Number of reports: {len(data)}")
            
        # Process the quarterly reports
        quarters = []
        for i, report in enumerate(data):
            report_date = report.get('endDate') or report.get('period') or report.get('year')
            if not report_date:
                continue
                
            # Try to get the report structure
            report_obj = report.get('report', {})
            if not report_obj:
                continue
                
            # Look for different possible statement keys
            statement = report_obj.get('ic', []) or report_obj.get('is', [])  # Income statement
            bs = report_obj.get('bs', [])  # Balance sheet
            cf = report_obj.get('cf', [])  # Cash flow
            
            # Parse list-based data structure
            def find_value(statement_list, labels):
                if not isinstance(statement_list, list):
                    return np.nan
                for item in statement_list:
                    if isinstance(item, dict):
                        item_label = item.get('label', '')
                        if item_label in labels:
                            return item.get('value', np.nan)
                return np.nan
            
            quarter_data = {
                'date': report_date,
                'Revenue': find_value(statement, ['Revenue', 'Revenues', 'Total Revenue']),
                'OperatingExpenses': find_value(statement, [
                    'Operating Expenses', 
                    'Operating expenses',
                    'Total operating expenses',
                    'Research and Development Expense, Total',
                    'Selling and Marketing Expense, Total',
                    'General and Administrative Expense, Total'
                ]),
                'OperatingCashFlow': find_value(cf, [
                    'Net Cash Provided by (Used in) Operating Activities',
                    'Net cash from operating activities',
                    'Operating Cash Flow',
                    'Cash flows from operating activities'
                ]),
                'Cash': find_value(bs, [
                    'Cash and Cash Equivalents, at Carrying Value, Total',
                    'Cash and cash equivalents',
                    'Cash'
                ]),
                'Debt': find_value(bs, [
                    'Long-Term Debt, Total',
                    'Long-term debt',
                    'Total debt',
                    'Debt, Total'
                ])
            }
            quarters.append(quarter_data)
        
        if not quarters:
            print("No quarterly data found in Finnhub response")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(quarters)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Drop rows where all financial data is NaN
        df = df.dropna(how='all', subset=['Revenue', 'OperatingExpenses', 'OperatingCashFlow', 'Cash', 'Debt'])
        
        if not df.empty:
            print(f"Finnhub data retrieved:")
            print(f"Quarters: {len(df)}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)}")
        else:
            print("All Finnhub data was NaN after parsing")
        
        return df
        
    except Exception as e:
        print(f"Finnhub API error: {str(e)}")
        import traceback
        traceback.print_exc()
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
    for df_index in (balance.index if not balance.empty else [], cashflow.index if not cashflow.empty else []):
        index = index.union(df_index)
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

    dfs = []
    
    # Get FMP data
    try:
        fmp_df = _fetch_fmp_quarterlies(symbol)
        if not fmp_df.empty:
            print(f"\nFMP Data loaded:")
            print(f"Shape: {fmp_df.shape}")
            print(f"Columns: {list(fmp_df.columns)}")
            print(f"Date range: {fmp_df.index.min()} to {fmp_df.index.max()}")
            warnings.append(f"FMP data: {len(fmp_df)} quarters ({fmp_df.index.min()} to {fmp_df.index.max()})")
            dfs.append(fmp_df)
    except Exception as exc:
        print(f"FMP error: {exc}")
        warnings.append(f"FMP data fetch failed: {exc}")
    
    # Get Finnhub data
    try:
        finnhub_df = _fetch_finnhub_quarterlies(symbol)
        if not finnhub_df.empty:
            print(f"\nFinnhub Data loaded:")
            print(f"Shape: {finnhub_df.shape}")
            print(f"Columns: {list(finnhub_df.columns)}")
            print(f"Date range: {finnhub_df.index.min()} to {finnhub_df.index.max()}")
            warnings.append(f"Finnhub data: {len(finnhub_df)} quarters ({finnhub_df.index.min()} to {finnhub_df.index.max()})")
            dfs.append(finnhub_df)
    except Exception as exc:
        print(f"Finnhub error: {exc}")
        warnings.append(f"Finnhub data fetch failed: {exc}")

    # Get Yahoo Finance data
    try:
        yahoo_df = _fetch_yahoo_quarterlies(symbol)
        if not yahoo_df.empty:
            print(f"\nYahoo Data loaded:")
            print(f"Shape: {yahoo_df.shape}")
            print(f"Columns: {list(yahoo_df.columns)}")
            print(f"Date range: {yahoo_df.index.min()} to {yahoo_df.index.max()}")
            warnings.append(f"Yahoo data: {len(yahoo_df)} quarters ({yahoo_df.index.min()} to {yahoo_df.index.max()})")
            dfs.append(yahoo_df)
    except Exception as exc:
        print(f"Yahoo error: {exc}")
        warnings.append(f"Yahoo data fetch failed: {exc}")
    
    if dfs:
        # Merge all dataframes, preferring FMP data over Finnhub for duplicates
        merged_df = pd.concat(dfs, axis=0, join='outer')
        # Remove duplicates by date, preferring FMP data over Finnhub
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        # Sort by date
        merged_df = merged_df.sort_index()
        
        # Ensure we have the required columns
        merged_df = _ensure_columns(merged_df)
        
        print(f"\nMerged Data Summary:")
        print(f"Total quarters: {len(merged_df)}")
        print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"Columns: {list(merged_df.columns)}")
        
        warnings.append(f"Combined data: {len(merged_df)} quarters ({merged_df.index.min()} to {merged_df.index.max()})")
        
        # Apply max_quarters filter if specified
        if max_quarters:
            merged_df = merged_df.tail(max_quarters)
            print(f"After max_quarters filter ({max_quarters}): {len(merged_df)} quarters")
        
        # Cache the merged data
        export = merged_df.reset_index()
        export.rename(columns={'date': 'date'}, inplace=True)
        write_local_csv(export, local_path)
        
        return merged_df, warnings
    else:
        warnings.append(f"No quarterly fundamentals available for {symbol}.")
        return pd.DataFrame(), warnings


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
