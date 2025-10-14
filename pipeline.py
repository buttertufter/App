"""Background data pipeline for refreshing cached data."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import Iterable, List, Tuple

import duckdb
import pandas as pd

from app.data import (
    fetch_estimates,
    fetch_insiders,
    fetch_prices,
    fetch_profile,
    fetch_quarterlies,
    get_universe,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data.duckdb"


def _open_connection() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quarterlies (
            ticker TEXT,
            date DATE,
            Revenue DOUBLE,
            OperatingExpenses DOUBLE,
            OperatingCashFlow DOUBLE,
            Cash DOUBLE,
            Debt DOUBLE,
            UndrawnCredit DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            date DATE,
            close DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profile (
            ticker TEXT,
            name TEXT,
            sector TEXT,
            industry TEXT,
            marketCap DOUBLE,
            beta DOUBLE,
            sharesOutstanding DOUBLE,
            updated TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS estimates (
            ticker TEXT,
            asof TIMESTAMP,
            eps_disp DOUBLE,
            rev_disp DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS insiders (
            ticker TEXT,
            asof TIMESTAMP,
            insider_net_12m DOUBLE,
            insider_net_pct DOUBLE
        )
        """
    )
    return conn


def _write_frame(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, ticker: str) -> None:
    if df.empty:
        return
    conn.execute(f"DELETE FROM {table} WHERE ticker = ?", [ticker])
    conn.register("tmp_df", df)
    conn.execute(f"INSERT INTO {table} SELECT * FROM tmp_df")
    conn.unregister("tmp_df")


def _process_symbol(symbol: str, max_quarters: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    max_q = None if max_quarters == "Max" else int(max_quarters) if str(max_quarters).isdigit() else None
    try:
        q_fin, q_warn = fetch_quarterlies(symbol, force_refresh=True, max_quarters=max_q)
        warnings.extend(q_warn)
    except Exception as exc:
        warnings.append(f"{symbol} quarterlies failed: {exc}")
        q_fin = pd.DataFrame()

    try:
        price, price_warn = fetch_prices(symbol, years=15, force_refresh=True)
        warnings.extend(price_warn)
    except Exception as exc:
        warnings.append(f"{symbol} prices failed: {exc}")
        price = pd.Series(dtype=float)

    profile, profile_warn = fetch_profile(symbol, force_refresh=True)
    warnings.extend(profile_warn)

    estimates, est_warn = fetch_estimates(symbol, force_refresh=True)
    warnings.extend(est_warn)

    insiders, ins_warn = fetch_insiders(symbol, profile, force_refresh=True)
    warnings.extend(ins_warn)

    conn = _open_connection()
    try:
        if not q_fin.empty:
            q_out = q_fin.reset_index().rename(columns={"date": "date"})
            q_out["ticker"] = symbol
            q_out = q_out[["ticker", "date", "Revenue", "OperatingExpenses", "OperatingCashFlow", "Cash", "Debt", "UndrawnCredit"]]
            _write_frame(conn, "quarterlies", q_out, symbol)

        if not price.empty:
            p_out = price.reset_index().rename(columns={price.name or symbol: "close", "index": "date"})
            p_out.columns = ["date", "close"]
            p_out["ticker"] = symbol
            p_out = p_out[["ticker", "date", "close"]]
            _write_frame(conn, "prices", p_out, symbol)

        if profile:
            prof_df = pd.DataFrame([
                {
                    "ticker": symbol,
                    "name": profile.get("name"),
                    "sector": profile.get("sector"),
                    "industry": profile.get("industry"),
                    "marketCap": profile.get("marketCap"),
                    "beta": profile.get("beta"),
                    "sharesOutstanding": profile.get("sharesOutstanding"),
                    "updated": pd.Timestamp.utcnow(),
                }
            ])
            _write_frame(conn, "profile", prof_df, symbol)

        if estimates:
            est_df = pd.DataFrame([
                {
                    "ticker": symbol,
                    "asof": pd.to_datetime(estimates.get("asof")),
                    "eps_disp": estimates.get("eps_disp"),
                    "rev_disp": estimates.get("rev_disp"),
                }
            ])
            _write_frame(conn, "estimates", est_df, symbol)

        if insiders:
            ins_df = pd.DataFrame([
                {
                    "ticker": symbol,
                    "asof": pd.to_datetime(insiders.get("asof")),
                    "insider_net_12m": insiders.get("insider_net_12m"),
                    "insider_net_pct": insiders.get("insider_net_pct"),
                }
            ])
            _write_frame(conn, "insiders", ins_df, symbol)
    finally:
        conn.close()

    return symbol, warnings


def backfill(universe: Iterable[str], max_quarters: str = "Max") -> None:
    symbols = list(universe)
    stubs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process_symbol, symbol, max_quarters): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol, warnings = future.result()
            if warnings:
                print(f"[pipeline] {symbol}:\n  - " + "\n  - ".join(warnings))
            else:
                print(f"[pipeline] {symbol}: refreshed")


def main():
    parser = argparse.ArgumentParser(description="Backfill market data into DuckDB cache")
    parser.add_argument("--universe", type=str, default=str(TICKERS_CSV := (PROJECT_ROOT / "assets" / "tickers.csv")), help="Path to universe CSV")
    parser.add_argument("--max-quarters", type=str, default="Max", help="Max quarters to retain (e.g., 12)")
    args = parser.parse_args()

    if Path(args.universe).exists():
        df = pd.read_csv(args.universe)
        symbols = df["symbol"].astype(str).str.upper().tolist()
    else:
        symbols = get_universe()["symbol"].astype(str).str.upper().tolist()

    backfill(symbols, max_quarters=args.max_quarters)


if __name__ == "__main__":
    main()
