import os
import sys
from pathlib import Path
from typing import List

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
    "Max": {"quarters": None, "years": 15},
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
    r = int(255 * (1 - pct))
    g = int(180 * pct + 60 * (1 - pct))
    b = int(120 * pct + 40 * (1 - pct))
    return f"#{r:02x}{g:02x}{b:02x}"


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
        index=1,
        help="Controls how many recent quarters are analysed.",
    )
    force_refresh = st.checkbox(
        "Force refresh fundamentals (ignore cache)",
        value=False,
        help="Bypass cached data and fetch fresh results.",
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
max_quarters = config["quarters"]
price_years = config["years"] or 15

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
    analysis_depth = st.selectbox(
        "Analysis depth",
        options=["Quick (8 quarters)", "Standard (12 quarters)", "Deep (16 quarters)"],
        index=1,
    )

run = st.button("Compute", type="primary")

if run:
    if not symbol:
        st.warning("Enter a ticker symbol to begin analysis.")
        st.stop()

    warnings: List[str] = []

    profile, profile_warnings = fetch_profile(symbol, force_refresh=force_refresh)
    warnings.extend(profile_warnings)

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

    peer_prices, peer_price_warn = fetch_peer_prices(peers, years=price_years, force_refresh=force_refresh)
    warnings.extend(peer_price_warn)
    if peer_prices.empty and sector_series is not None:
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
    }.get(analysis_depth, {"Wq": 4, "Lq": 1})

    wave_kappa = 3.0 + (1.0 - risk_tolerance) * 3.5

    if warnings:
        st.warning("\n".join(sorted(set(warnings))))

    modules, wave, issues = compute_modules(
        q_fin=q_fin,
        price=price,
        peer_prices=peer_prices,
        lambdas=(lambda_debt, lambda_credit, lambda_finin),
        kappa=wave_kappa,
        **depth_cfg,
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
                st.info(f"{module_titles[key]}\\n\\n_coming soon_", icon="i")
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

    st.subheader("Revenue vs cash & financing signals")
    rev_frame = q_fin[["Revenue", "OperatingCashFlow"]].copy()
    if "FinancingIn" in q_fin.columns:
        rev_frame["FinancingIn"] = q_fin["FinancingIn"]
    rev_frame = rev_frame.dropna(how="all")
    if not rev_frame.empty:
        st.line_chart(rev_frame)
    else:
        st.caption("Insufficient quarterly data to chart revenue and inflows.")

    if not peer_prices.empty:
        st.subheader("Peer comparison (normalized)")
        normalized = peer_prices.divide(peer_prices.iloc[0]).dropna(how="all")
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
