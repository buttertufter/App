import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data import (
    discover_peers,
    dotenv_status,
    fetch_estimates,
    fetch_insiders,
    fetch_peer_prices,
    fetch_prices,
    fetch_profile,
    fetch_quarterlies,
    fetch_sector_series,
    get_universe,
    OFFLINE_MODE,      # <-- ADD THIS
    REQUIRED_COLUMNS,  # <-- ADD THIS
    DATA_STORE,        # <-- ADD THIS
    PROJECT_ROOT,      # <-- BEST PRACTICE
    CACHE_DIR,         # <-- BEST PRACTICE
)
from model import compute_modules  # noqa: E402  # isort:skip
from utils import file_age_days

st.set_page_config(page_title="Bridge Dashboard", page_icon="B", layout="wide")

st.title("Bridge Dashboard")
st.caption(
    "Press compute to assemble a confirmation wave across growth, resilience, leadership, "
    "sector posture, and market uncertainty."
)

if not dotenv_status():
    st.warning(
        "Optional dependency `python-dotenv` is not installed. "
        "Environment variables will be read from the OS only. "
        "Run `pip install python-dotenv` to load settings from the project .env file."
    )

if OFFLINE_MODE:
    st.warning("Running in OFFLINE MODE - data loaded from local storage only.")
else:
    st.info("Online mode: pulling live data and caching locally.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

HISTORY_PRESETS = {
    "1y (4q)": {"quarters": 4, "years": 1},
    "3y (12q)": {"quarters": 12, "years": 3},
    "5y (20q)": {"quarters": 20, "years": 5},
    "10y (40q)": {"quarters": 40, "years": 10},
    "Max (All Data)": {"quarters": None, "years": None},  # Use ALL available data
}

UNIVERSE_DF = get_universe()


def ticker_options():
    rows: List[str] = []
    symbols = []
    if not UNIVERSE_DF.empty and "symbol" in UNIVERSE_DF:
        names = UNIVERSE_DF.get("name", pd.Series(index=UNIVERSE_DF.index, dtype=str))
        for symbol, name in zip(UNIVERSE_DF["symbol"], names):
            label = f"{symbol.upper()} — {name}" if str(name) not in {None, "nan"} else symbol.upper()
            rows.append(label)
            symbols.append(symbol.upper())
    if not rows:
        defaults = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("AMZN", "Amazon.com, Inc."),
            ("TSLA", "Tesla, Inc."),
            ("NVDA", "NVIDIA Corporation"),
        ]
        for symbol, name in defaults:
            rows.append(f"{symbol} — {name}")
            symbols.append(symbol)
    default_label = next((label for label in rows if label.startswith("AAPL")), rows[0])
    return rows, {label: sym for label, sym in zip(rows, symbols)}, rows.index(default_label)


def score_to_color(score: float) -> str:
    pct = max(0.0, min(1.0, float(score)))
    if pct < 0.6:
        return "#f87171"  # red
    elif pct < 0.8:
        return "#fbbf24"  # amber
    else:
        return "#34d399"  # green


def render_gauge(label: str, score: float, blurb: str):
    color = score_to_color(score)
    pct = max(0.0, min(1.0, score)) * 100
    st.markdown(
        f"""
        <div style="border:1px solid #d4d4d8;border-radius:12px;padding:14px;">
          <div style="font-weight:600;margin-bottom:6px;">{label}</div>
          <div style="height:10px;background:#f4f4f5;border-radius:6px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:10px;background:{color};transition:width 0.4s ease;"></div>
          </div>
          <div style="font-size:12px;color:#71717a;margin-top:6px;">{score:.2f}</div>
          <div style="font-size:11px;color:#9ca3af;margin-top:4px;">{blurb}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def log_result(row: dict):
    log_path = Path(__file__).parent / "logs" / "compute_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


ticker_labels, label_to_symbol, default_index = ticker_options()

col_main, col_peers = st.columns([2, 1])
with col_main:
    selected = st.selectbox(
        "Company",
        ticker_labels,
        index=default_index,
        help="Type to search for a company.",
    )
    manual_symbol = st.text_input(
        "Or type a symbol",
        value="",
        help="Provide a custom symbol to override the list.",
    ).strip().upper()
    history_choice = st.selectbox(
        "History",
        list(HISTORY_PRESETS.keys()),
        index=4,  # Default to "Max (All Data)"
        help="Controls how many recent quarters are analysed. Use 'Max (All Data)' for complete historical analysis.",
    )
    force_refresh = st.checkbox(
        "🔄 Force refresh data (ignore cache)",
        value=False,
        help="⚠️ Check this to fetch fresh data from APIs. Otherwise cached data (up to 30 days old) will be used. IMPORTANT: If you're seeing limited data, enable this!",
    )
    if OFFLINE_MODE and force_refresh:
        st.warning("Offline mode: force refresh disabled.")
        force_refresh = False

with col_peers:
    peers_raw = st.text_input(
        "Peer tickers (optional)",
        value="",
        help="Comma-separated tickers. Leave blank to rely on automatic peers.",
    )

symbol = manual_symbol or label_to_symbol.get(selected, selected.split("—")[0].strip().upper())
config = HISTORY_PRESETS[history_choice]
max_quarters = config["quarters"]  # None means use all available
price_years = config["years"] if config["years"] is not None else 20  # Default to 20 years if None

# Show cache status after symbol is defined
if not force_refresh and symbol:
    from data import _local_path, _is_fresh
    cache_path = _local_path("quarterlies", symbol, "csv")
    if _is_fresh(cache_path):
        from utils import file_age_days
        age = file_age_days(cache_path)
        st.caption(f"ℹ️ Using cached data ({age:.0f} days old). Check '🔄 Force refresh' for latest data.")

advanced = st.expander("Advanced parameters")
with advanced:
    risk_tolerance = st.slider(
        "Risk tolerance (0 = very conservative, 1 = aggressive)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    lambda_cols = st.columns(3)
    lambda_debt = lambda_cols[0].number_input(
        "Debt weight lambda_1",
        value=0.5,
        min_value=0.0,
        max_value=2.0,
        step=0.1,
    )
    lambda_credit = lambda_cols[1].number_input(
        "Credit weight lambda_2",
        value=0.6,
        min_value=0.0,
        max_value=2.0,
        step=0.1,
    )
    lambda_finin = lambda_cols[2].number_input(
        "Financing weight lambda_3",
        value=0.8,
        min_value=0.0,
        max_value=2.0,
        step=0.1,
    )
    analysis_depth_map = {
        "Quick (8 quarters)": {"quarters": 8, "years": 2},
        "Standard (12 quarters)": {"quarters": 12, "years": 3},
        "Deep (16 quarters)": {"quarters": 16, "years": 4}
    }
    analysis_depth_selection = st.selectbox(
        "Analysis depth",
        options=list(analysis_depth_map.keys()),
        index=1,
    )
    analysis_depth = analysis_depth_map[analysis_depth_selection]

run = st.button("Compute", type="primary")

if run:
    if not symbol:
        st.warning("Enter a ticker symbol to begin analysis.")
        st.stop()

    warnings: List[str] = []

    profile, profile_warnings = fetch_profile(symbol, force_refresh=force_refresh)
    warnings.extend(profile_warnings)

    # Fetch quarterly financials with proper depth
    # Use the history choice config instead of analysis depth for financials
    max_quarters = config["quarters"]  # From history_choice
    price_years = config["years"]
    
    try:
        q_fin, q_warn = fetch_quarterlies(symbol, force_refresh=force_refresh, max_quarters=max_quarters)
        warnings.extend(q_warn)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in q_fin.columns]
    if missing_cols:
        st.error(f"Missing fundamental columns for {symbol}: {', '.join(missing_cols)}. Please refresh data.")
        st.stop()

    # Fetch price history matching the analysis depth
    try:
        price, price_warn = fetch_prices(symbol, years=price_years, force_refresh=force_refresh)
        warnings.extend(price_warn)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    if price.size < 30:
        warnings.append(f"Price history for {symbol} is shorter than 30 rows; results may be unstable.")

    sector_series, sector_warn = fetch_sector_series(symbol)
    warnings.extend(sector_warn)

    manual_peers = [p.strip().upper() for p in peers_raw.split(",") if p.strip()]
    if manual_peers:
        peers = manual_peers
        peer_warn = []
    else:
        peers, peer_warn = discover_peers(symbol, UNIVERSE_DF)
    warnings.extend(peer_warn)

    # Fetch peer prices with consistent time window
    peer_prices, peer_price_warn = fetch_peer_prices(peers, years=price_years, force_refresh=force_refresh)
    warnings.extend(peer_price_warn)
    
    # Fall back to sector ETF if peer data is unavailable
    if peer_prices.empty:
        sector_series, sector_warn = fetch_sector_series(symbol)
        warnings.extend(sector_warn)
        if sector_series is not None:
            column_name = sector_series.name or "SectorETF"
            peer_prices = pd.DataFrame({column_name: sector_series})
            warnings.append("Peer prices unavailable; using sector ETF as proxy.")

    estimates, estimate_warn = fetch_estimates(symbol, force_refresh=force_refresh)
    warnings.extend(estimate_warn)

    insiders, insider_warn = fetch_insiders(symbol, profile, force_refresh=force_refresh)
    warnings.extend(insider_warn)

    depth_cfg = {
        "Quick (8 quarters)": {"Wq": 3, "Lq": 1},
        "Standard (12 quarters)": {"Wq": 4, "Lq": 1},
        "Deep (16 quarters)": {"Wq": 6, "Lq": 2},
    }.get(analysis_depth_selection, {"Wq": 4, "Lq": 1})

    wave_kappa = 3.0 + (1.0 - risk_tolerance) * 3.5

    if warnings:
        st.warning("\n".join(sorted(set(warnings))))

    modules, wave, issues = compute_modules(
        q_fin,
        price,
        peer_prices,
        symbol=symbol,
        lambdas=(float(lambda_debt), float(lambda_credit), float(lambda_finin)),
        Wq=depth_cfg["Wq"],
        Lq=depth_cfg["Lq"],
        kappa=wave_kappa
    )

    if modules is None:
        for msg in issues or ["Unable to compute modules with the available data."]:
            st.warning(msg)
        st.stop()

    if issues:
        st.warning("\n".join(issues))

    if profile:
        header = profile.get("name") or symbol
        sub = " • ".join(filter(None, [profile.get("sector"), profile.get("industry")]))
        st.subheader(f"{header} ({symbol})")
        if sub:
            st.caption(sub)

    badge_parts = []
    q_cache = DATA_STORE / "quarterlies" / f"{symbol}.csv"
    if q_cache.exists():
        badge_parts.append(f"Quarterlies cache: {file_age_days(q_cache):.1f}d old")
    p_cache = DATA_STORE / "prices" / f"{symbol}.csv"
    if p_cache.exists():
        badge_parts.append(f"Prices cache: {file_age_days(p_cache):.1f}d old")
    if badge_parts:
        st.caption(" | ".join(badge_parts))

    st.markdown(
        f"""
        <div style="margin-top:12px;padding:20px;border-radius:16px;background:{score_to_color(wave)};color:white;">
          <div style="font-size:14px;opacity:0.85;">{symbol} Wave Score</div>
          <div style="font-size:42px;font-weight:700;margin-top:4px;">{wave:.2f}</div>
          <div style="font-size:12px;opacity:0.9;">
            0 = fragile, 1 = resilient. Adjusted for the selected risk tolerance.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    module_titles = {
        "A": "Engine (Growth)",
        "B": "Core (Stability)",
        "C": "Captain (Leadership)",
        "D": "Neighbors (Sector)",
        "E": "Wobble (Uncertainty)",
    }
    module_help = {
        "A": "Revenue growth, acceleration, and capital support efficiency.",
        "B": "Variability of growth, cash coherence, and liquidity buffers.",
        "C": "Leadership alignment metrics (placeholder for future data).",
        "D": "Peer sector momentum and alignment with company performance.",
        "E": "Market volatility and idiosyncratic risk inverted.",
    }

    module_cols = st.columns(5)
    for col, key in zip(module_cols, ["A", "B", "C", "D", "E"]):
        with col:
            val = modules.get(key, {}).get("score")
            if val is None:
                st.info(f"{module_titles[key]}\\n\\n_coming soon_", icon="ℹ️")
            else:
                render_gauge(module_titles[key], float(val), module_help[key])

    tags = []
    for module in modules.values():
        tags.extend(module.get("tags", []))

    st.subheader("Notes and tags")
    if tags:
        st.write(", ".join(sorted(set(tags))))
    else:
        st.write("No qualitative tags surfaced for this run.")

    st.subheader("📊 Financial Performance Analysis")
    
    if not q_fin.empty:
        # Show data overview with prominent warning if data is limited
        col1, col2, col3 = st.columns(3)
        with col1:
            quarters_available = len(q_fin)
            st.metric("Quarters Available", quarters_available)
            # Warning if data seems limited
            if quarters_available < 10:
                st.warning(f"⚠️ Only {quarters_available} quarters found. Enable '🔄 Force refresh data' above to fetch more historical data from Finnhub API (~35+ quarters).")
        with col2:
            st.metric("Start Date", q_fin.index.min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("End Date", q_fin.index.max().strftime('%Y-%m-%d'))
            
        # Display data source info
        if warnings:
            with st.expander("📁 Data Source Information"):
                for warning in warnings:
                    if "data:" in warning:
                        st.write(f"✓ {warning}")
        
        # Prepare financial data
        fin_data = q_fin.copy()
        fin_data.index = pd.to_datetime(fin_data.index)
        fin_data = fin_data.sort_index()
        
        # Calculate TTM (Trailing Twelve Months) values
        ttm_data = pd.DataFrame(index=fin_data.index)
        for col in ["Revenue", "OperatingCashFlow", "OperatingExpenses"]:
            if col in fin_data.columns:
                ttm_data[col] = fin_data[col].rolling(4, min_periods=1).sum()
        
        # Calculate key financial ratios
        metrics_data = pd.DataFrame(index=fin_data.index)
        
        # 1. Operating Cash Flow to Revenue Ratio (Cash Conversion)
        if "Revenue" in ttm_data.columns and "OperatingCashFlow" in ttm_data.columns:
            metrics_data["OCF/Revenue %"] = (ttm_data["OperatingCashFlow"] / ttm_data["Revenue"] * 100).replace([np.inf, -np.inf], np.nan)
        
        # 2. Operating Margin
        if "Revenue" in ttm_data.columns and "OperatingExpenses" in ttm_data.columns:
            metrics_data["Operating Margin %"] = ((ttm_data["Revenue"] - ttm_data["OperatingExpenses"]) / ttm_data["Revenue"] * 100).replace([np.inf, -np.inf], np.nan)
        
        # 3. Cash to Debt Ratio (Liquidity)
        if "Cash" in fin_data.columns and "Debt" in fin_data.columns:
            metrics_data["Cash/Debt Ratio"] = (fin_data["Cash"] / fin_data["Debt"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        
        # 4. Net Debt (Debt - Cash)
        if "Cash" in fin_data.columns and "Debt" in fin_data.columns:
            metrics_data["Net Debt ($B)"] = (fin_data["Debt"] - fin_data["Cash"]) / 1e9
        
        # 5. Revenue Growth Rate (YoY)
        if "Revenue" in ttm_data.columns:
            metrics_data["Revenue Growth %"] = (ttm_data["Revenue"].pct_change(4) * 100).replace([np.inf, -np.inf], np.nan)
        
        # 6. Cash Flow Growth Rate (YoY)
        if "OperatingCashFlow" in ttm_data.columns:
            metrics_data["OCF Growth %"] = (ttm_data["OperatingCashFlow"].pct_change(4) * 100).replace([np.inf, -np.inf], np.nan)
        
        # Chart 1: Revenue and Operating Cash Flow (TTM)
        st.subheader("💰 Revenue & Operating Cash Flow (TTM)")
        chart1_data = pd.DataFrame()
        if "Revenue" in ttm_data.columns:
            chart1_data["Revenue ($B)"] = ttm_data["Revenue"] / 1e9
        if "OperatingCashFlow" in ttm_data.columns:
            chart1_data["Operating Cash Flow ($B)"] = ttm_data["OperatingCashFlow"] / 1e9
        if not chart1_data.empty:
            st.line_chart(chart1_data)
        
        # Chart 2: Profitability Metrics
        st.subheader("📈 Profitability & Efficiency Metrics")
        chart2_data = pd.DataFrame()
        if "OCF/Revenue %" in metrics_data.columns:
            chart2_data["Cash Conversion %"] = metrics_data["OCF/Revenue %"]
        if "Operating Margin %" in metrics_data.columns:
            chart2_data["Operating Margin %"] = metrics_data["Operating Margin %"]
        if not chart2_data.empty:
            st.line_chart(chart2_data)
            col1, col2 = st.columns(2)
            with col1:
                latest_conv = chart2_data["Cash Conversion %"].dropna().iloc[-1] if "Cash Conversion %" in chart2_data.columns else 0
                st.metric("Latest Cash Conversion", f"{latest_conv:.1f}%", 
                         help="Operating Cash Flow / Revenue. Higher is better (>20% is good)")
            with col2:
                latest_margin = chart2_data["Operating Margin %"].dropna().iloc[-1] if "Operating Margin %" in chart2_data.columns else 0
                st.metric("Latest Operating Margin", f"{latest_margin:.1f}%",
                         help="(Revenue - Operating Expenses) / Revenue. Higher is better")
        
        # Chart 3: Growth Rates
        st.subheader("🚀 Year-over-Year Growth Rates")
        chart3_data = pd.DataFrame()
        if "Revenue Growth %" in metrics_data.columns:
            chart3_data["Revenue Growth %"] = metrics_data["Revenue Growth %"]
        if "OCF Growth %" in metrics_data.columns:
            chart3_data["OCF Growth %"] = metrics_data["OCF Growth %"]
        if not chart3_data.empty:
            st.line_chart(chart3_data)
        
        # Chart 4: Balance Sheet Health
        st.subheader("💼 Balance Sheet Health")
        col1, col2 = st.columns(2)
        
        with col1:
            # Cash vs Debt
            chart4a_data = pd.DataFrame()
            if "Cash" in fin_data.columns:
                chart4a_data["Cash ($B)"] = fin_data["Cash"] / 1e9
            if "Debt" in fin_data.columns:
                chart4a_data["Debt ($B)"] = fin_data["Debt"] / 1e9
            if not chart4a_data.empty:
                st.line_chart(chart4a_data)
        
        with col2:
            # Net Debt and Cash/Debt Ratio
            chart4b_data = pd.DataFrame()
            if "Net Debt ($B)" in metrics_data.columns:
                chart4b_data["Net Debt ($B)"] = metrics_data["Net Debt ($B)"]
            if "Cash/Debt Ratio" in metrics_data.columns:
                chart4b_data["Cash/Debt Ratio"] = metrics_data["Cash/Debt Ratio"]
            if not chart4b_data.empty:
                st.line_chart(chart4b_data)
        
        # Key Metrics Summary
        st.subheader("📊 Latest Quarter Metrics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_revenue = (ttm_data["Revenue"].iloc[-1] / 1e9) if "Revenue" in ttm_data.columns else 0
            st.metric("Revenue (TTM)", f"${latest_revenue:.2f}B")
        
        with col2:
            latest_ocf = (ttm_data["OperatingCashFlow"].iloc[-1] / 1e9) if "OperatingCashFlow" in ttm_data.columns else 0
            st.metric("Operating Cash Flow", f"${latest_ocf:.2f}B")
        
        with col3:
            latest_cash = (fin_data["Cash"].iloc[-1] / 1e9) if "Cash" in fin_data.columns else 0
            st.metric("Cash", f"${latest_cash:.2f}B")
        
        with col4:
            latest_debt = (fin_data["Debt"].iloc[-1] / 1e9) if "Debt" in fin_data.columns else 0
            st.metric("Debt", f"${latest_debt:.2f}B")
        
        # Raw data in expander
        with st.expander("🔍 View Raw Financial Data"):
            st.dataframe(fin_data.sort_index(ascending=False))
            st.write("### Summary Statistics")
            st.dataframe(fin_data.describe())
    else:
        st.caption("Insufficient quarterly data to chart revenue and cash flows")

    if not peer_prices.empty:
        st.subheader(f"Peer comparison ({price_years} years normalized)")
        # Normalize from start of period and handle missing data
        normalized = peer_prices.copy()
        for col in normalized.columns:
            clean_series = normalized[col].dropna()
            if not clean_series.empty:
                normalized[col] = normalized[col] / clean_series.iloc[0]
        normalized = normalized.dropna(how="all")
        
        # Add main symbol price for comparison
        if not price.empty:
            clean_price = price.dropna()
            if not clean_price.empty:
                normalized[symbol] = price / clean_price.iloc[0]
                
        st.line_chart(normalized)
    else:
        st.info(
            "Peer price data was unavailable for the selected window; sector ETF proxy may be used instead.",
            icon="i",
        )

    momentum = price.pct_change().rolling(4).mean()
    acceleration = momentum.diff()
    inflection = acceleration[(acceleration.shift(1) * acceleration) < 0].dropna().tail(6)
    with st.expander("Inflection points in price momentum"):
        if inflection.empty:
            st.write("No clear momentum inflections detected in the selected window.")
        else:
            df_inflect = pd.DataFrame(
                {
                    "Momentum": momentum.loc[inflection.index],
                    "Acceleration": acceleration.loc[inflection.index],
                }
            )
            st.table(df_inflect.style.format("{:.4f}"))

    with st.expander("Quarterly data (detail)"):
        st.dataframe(q_fin)

    log_result(
        {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "ticker": symbol,
            "history_window": history_choice,
            "max_quarters": max_quarters if max_quarters is not None else "Max",
            "price_years": price_years,
            "peers": ";".join(peers),
            "risk_tolerance": risk_tolerance,
            "lambda_debt": lambda_debt,
            "lambda_credit": lambda_credit,
            "lambda_finin": lambda_finin,
            "analysis_depth": analysis_depth,
            "wave": wave,
            "estimates": estimates or {},
            "insiders": insiders or {},
        }
    )
