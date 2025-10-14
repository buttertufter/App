# app/model.py

import numpy as np
import pandas as pd

try:
    from utils import (
        logistic_sigmoid as lgsig,
        geo_mean as gmean,
        ttm_sum,
        slope_log,
        stdev_over,
        ema,
        EPS,
        last_or_default,
        nth_from_end_or_default,
        to_series_1d,
    )
except ImportError:  # pragma: no cover - fallback when utils is packaged differently
    from utils import (
        logistic_sigmoid as lgsig,
        geo_mean as gmean,
        ttm_sum,
        slope_log,
        stdev_over,
        ema,
        EPS,
        last_or_default,
        nth_from_end_or_default,
        to_series_1d,
    )

from constants import (
    FALLBACKS,
    MIN_QUARTERS_DATA,
    DEFAULT_QUARTERS_WINDOW,
    VALIDATION_CONFIG,
    MODULE_INFO  # Add MODULE_INFO import
)


DEBUG_SHAPES = False


from leadership import fetch_ceo_data, fetch_insider_analysis, compute_leadership_score

def compute_modules(
    q_fin: pd.DataFrame,
    price: pd.Series,
    peer_prices: pd.DataFrame,
    symbol: str = None,
    lambdas=(0.5, 0.6, 0.8),  # debt, credit, finin
    rho=0.6,  # unused placeholder for future modelling
    theta_cp=0.2,
    kappa=4.0,
    Wq: int = 4,   # quarters window for TTM/derivatives
    Lq: int = 1,   # lag (quarters) for causal checks
):
    """Compute scoring modules and overall wave score with adaptive windows.

    Returns
    -------
    modules : dict | None
        Module score breakdown or ``None`` if computation failed.
    wave : float | None
        Overall wave score in [0, 1] or ``None`` on failure.
    issues : list[str]
        Non-fatal issues encountered during the computation.
    """

    issues: list[str] = []

    if not isinstance(q_fin, pd.DataFrame):
        return None, None, ["Quarterly financials are unavailable."]

    qf = q_fin.copy()
    qf = qf.apply(pd.to_numeric, errors="coerce")
    qf = qf.dropna(how="all")

    n_rows = len(qf)
    if n_rows < 4:
        return None, None, [f"At least 4 quarterly observations required; found {n_rows}."]

    Wq_eff = max(2, min(Wq, n_rows - 1))
    Lq_eff = max(1, min(Lq if Lq > 0 else 1, max(1, n_rows // 4)))
    if Wq_eff != Wq or Lq_eff != Lq:
        issues.append(f"Adaptive windows in use (Wq={Wq_eff}, Lq={Lq_eff}).")

    required_cols = ["Revenue", "OperatingExpenses", "OperatingCashFlow", "Cash", "Debt"]
    for col in required_cols:
        if col not in qf.columns:
            return None, None, [f"{col} series missing from quarterly data."]

    window_ttm = max(Wq_eff, 1)

    def _debug(label: str, obj):
        if DEBUG_SHAPES:
            try:
                arr = np.asarray(obj)
                print(f"[debug] {label}: type={type(obj).__name__}, shape={arr.shape}")
            except Exception:
                print(f"[debug] {label}: type={type(obj).__name__}, shape=unknown")

    def _series(label: str, obj):
        series = to_series_1d(obj, name=label)
        if DEBUG_SHAPES:
            print(f"[debug] {label} -> Series len={len(series)}")
        return series

    def _latest(label: str, obj, default=np.nan):
        return last_or_default(_series(label, obj), default=default)

    R_ttm = ttm_sum(qf["Revenue"], window_ttm)
    R = _latest("Revenue_TTM", R_ttm, default=np.nan)
    if not np.isfinite(R):
        R = _latest("Revenue_raw", qf["Revenue"], default=np.nan)
    if not np.isfinite(R):
        return None, None, ["Revenue history is too sparse to compute trailing totals."]

    E_ttm = ttm_sum(qf["OperatingExpenses"], window_ttm)
    E = _latest("OperatingExp_TTM", E_ttm, default=np.nan)
    if not np.isfinite(E):
        E = _latest("OperatingExp_raw", qf["OperatingExpenses"], default=np.nan)
    if not np.isfinite(E):
        issues.append("Operating expenses history is sparse; using revenue-based fallback.")
        E = R * 0.75

    OCF_ttm = ttm_sum(qf["OperatingCashFlow"], window_ttm)
    OCF = _latest("OperatingCF_TTM", OCF_ttm, default=np.nan)
    if not np.isfinite(OCF):
        OCF = _latest("OperatingCF_raw", qf["OperatingCashFlow"], default=np.nan)
    if not np.isfinite(OCF):
        issues.append("Operating cash flow history is sparse; using revenue-based fallback.")
        OCF = R * 0.15

    # Use newer pandas fillna style with validated fallback values
    g_R_source = R_ttm.ffill()
    if g_R_source.dropna().shape[0] >= MIN_QUARTERS_DATA:
        g_R = slope_log(g_R_source, window=Wq_eff)
    else:
        g_R = pd.Series(FALLBACKS["revenueGrowth"], index=g_R_source.index)
    g_R_last = _latest("g_R", g_R, default=FALLBACKS["revenueGrowth"])
    if not np.isfinite(g_R_last):
        issues.append("Revenue momentum insufficient; set to neutral.")
        g_R_last = FALLBACKS["revenue_growth"]

    # Use newer pandas fillna style with validated fallback values
    a_R_source = g_R.ffill()
    if a_R_source.dropna().shape[0] >= MIN_QUARTERS_DATA:
        a_R = slope_log(a_R_source, window=Wq_eff)
    else:
        a_R = pd.Series(FALLBACKS["revenueAcceleration"], index=a_R_source.index)
    a_R_last = _latest("a_R", a_R, default=FALLBACKS["revenueAcceleration"])
    if not np.isfinite(a_R_last):
        issues.append("Revenue acceleration insufficient; set to neutral.")
        a_R_last = 0.0

    m_op = (R - E) / max(R, EPS)
    m_ocf = OCF / max(R, EPS)

    cash = _latest("Cash", qf["Cash"], default=0.0)
    debt = _latest("Debt", qf["Debt"], default=0.0)
    undrawn = _latest("UndrawnCredit", qf.get("UndrawnCredit"), default=0.0)
    if not np.isfinite(undrawn):
        issues.append("Undrawn credit history unavailable; treating as zero.")
        undrawn = 0.0

    if "FinancingIn" in qf.columns:
        fin_slice = qf["FinancingIn"].tail(max(Wq_eff, 1)).fillna(0.0)
        finin = float(fin_slice.mean()) if not fin_slice.empty else 0.0
    else:
        finin = 0.0

    ld, lc, lf = lambdas
    I_eff = (cash + ld * debt + lc * undrawn + lf * finin) / max(R, EPS)
    S_supp = 1.0 / max(I_eff, EPS) - 1.0

    R_log = np.log(np.maximum(g_R_source, EPS))
    dlnR_series = R_log - R_log.shift(Wq_eff)
    dlnR_W = _latest("dlnR_W", dlnR_series, default=np.nan)
    if not np.isfinite(dlnR_W):
        issues.append("Unable to compute revenue growth over the analysis window; using neutral value.")
        dlnR_W = 0.0

    CP = 0.0
    tail_len = Wq_eff + Lq_eff + 1
    tail_slice = qf.tail(tail_len)
    if tail_slice.shape[0] < tail_len:
        issues.append("Not enough quarterly history to evaluate capital productivity dynamics; treating as neutral.")
    else:
        cash_tail = tail_slice["Cash"].astype(float)
        debt_tail = tail_slice["Debt"].astype(float)
        undr_tail = tail_slice.get("UndrawnCredit", pd.Series(0.0, index=tail_slice.index)).fillna(0.0).astype(float)
        fin_tail = tail_slice.get("FinancingIn", pd.Series(0.0, index=tail_slice.index)).fillna(0.0).astype(float)
        I_hist = (cash_tail + ld * debt_tail + lc * undr_tail + lf * fin_tail) / max(R, EPS)
        I_lag = I_hist.shift(Lq_eff)
        dI_series = I_lag - I_lag.shift(Wq_eff)
        dI_W = _latest("dI_W", dI_series, default=np.nan)
        if not np.isfinite(dI_W) or abs(dI_W) < EPS:
            issues.append("Investment change is unavailable or too small; capital productivity treated as neutral.")
        else:
            CP = dlnR_W / dI_W

    def safe_corr(x: pd.Series, y: pd.Series) -> float:
        x = x.dropna()
        y = y.dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < max(4, Wq_eff):
            return 0.0
        x_v = x.loc[idx].astype(float).values
        y_v = y.loc[idx].astype(float).values
        xv = x_v - x_v.mean()
        yv = y_v - y_v.mean()
        denom = np.sqrt((xv ** 2).sum() * (yv ** 2).sum())
        return float((xv * yv).sum() / denom) if denom > 0 else 0.0

    window_sigma = min(Wq_eff, len(g_R)) if len(g_R) >= 2 else 2
    sigma_series = stdev_over(g_R.fillna(0.0), window=window_sigma)
    sigma_R = _latest("sigma_R", sigma_series, default=np.nan)
    if not np.isfinite(sigma_R):
        issues.append("Insufficient history to estimate growth volatility; assuming zero volatility.")
        sigma_R = 0.0
    tau = 0.08

    ocf_source = OCF_ttm.ffill()  # Using ffill() instead of fillna(method="ffill")
    if ocf_source.dropna().shape[0] >= Wq_eff:
        dln_ocf = slope_log(ocf_source, window=Wq_eff)
    else:
        issues.append("Operating cash flow history insufficient for slope analysis; coherence defaults to zero.")
        dln_ocf = pd.Series(dtype=float)
    corr_cash = safe_corr(g_R, dln_ocf)
    corr_fin = 0.0  # placeholder for future FinancingIn coherence

    price_clean = price.dropna()
    if price_clean.empty:
        return None, None, ["Price history is empty for the selected window."]
    if len(price_clean) >= 2:
        price_window = min(12, max(2, len(price_clean)))
        g_P = slope_log(price_clean, window=price_window)
    else:
        issues.append("Not enough price history to evaluate momentum trend; treating momentum as neutral.")
        g_P = pd.Series(dtype=float)
    gP_lag_val = nth_from_end_or_default(g_P, n=2, default=0.0)

    if peer_prices is None or peer_prices.empty:
        K = 0.0
    else:
        peer_slopes = []
        for col in peer_prices.columns:
            series = peer_prices[col].dropna()
            if len(series) < 2:
                issues.append(f"Peer {col} lacks sufficient price history for slope analysis; skipping.")
                continue
            peer_window = min(12, max(2, len(series)))
            peer_slopes.append(slope_log(series, window=peer_window).rename(col))
        if peer_slopes:
            peer_stack = pd.concat(peer_slopes, axis=1)
            peer_mean = peer_stack.clip(lower=0.0).mean(axis=1)
            K = _latest("peer_alignment", peer_mean, default=0.0)
        else:
            issues.append("Peer price history insufficient; sector alignment defaults to neutral.")
            K = 0.0

    pct_changes = price_clean.pct_change().dropna()
    if len(pct_changes) >= 2:
        vol_window = min(12, max(2, len(pct_changes)))
        idio_series = pct_changes.rolling(vol_window).std()
        idio_vol = _latest("idio_vol", idio_series, default=np.nan)
        if not np.isfinite(idio_vol):
            issues.append("Unable to compute idiosyncratic volatility; treating uncertainty as neutral.")
            idio_vol = 0.0
    else:
        issues.append("Not enough price history to compute idiosyncratic volatility; treating uncertainty as neutral.")
        idio_vol = 0.0

    def s(z: float) -> float:
        return lgsig(z, kappa)

    Q_supp = s(S_supp)
    Q_ops = gmean([s(m_op), s(m_ocf)], [1, 1])
    Q_cp = s(CP - theta_cp)
    Q_mom = s(g_R_last)
    Q_acc = s(a_R_last)
    module_a_weights = MODULE_INFO["A"]["weights"]
    A = gmean([Q_supp, Q_ops, Q_cp, Q_mom, Q_acc], module_a_weights)

    Q_stab = s(tau - sigma_R)
    Q_coh = gmean([(1 + corr_cash) / 2, (1 + corr_fin) / 2], [1, 1])
    liq = cash / max(E, EPS)
    nd = max(debt - cash, 0.0) / max(R, EPS)
    Q_liq, Q_nd = s(liq), s(-nd)
    module_b_weights = MODULE_INFO["B"]["weights"]
    B = gmean([Q_stab, Q_coh, Q_liq, Q_nd], module_b_weights)

    C = None

    Q_sector = s(K)
    Q_align = 1.0 - max(0.0, s(gP_lag_val) - s(g_R_last))
    D = float(np.sqrt(max(Q_sector, 0.0) * max(Q_align, 0.0)))

    module_e_weights = MODULE_INFO["E"]["weights"]
    E_mod = gmean([s(-idio_vol), Q_align], module_e_weights)

    # Use leadership score with fallback
    leadership_score = FALLBACKS.get("leadershipScore", 0.5)
    if symbol:
        try:
            ceo_data, ceo_warns = fetch_ceo_data(symbol)
            insider_data, insider_warns = fetch_insider_analysis(symbol)
            
            issues.extend(ceo_warns)
            issues.extend(insider_warns)
            
            # Pass through normalized growth and margin data
            historical_growth = g_R_last if np.isfinite(g_R_last) else None
            profit_margin = m_op if np.isfinite(m_op) else None
            
            leadership_score, leadership_warns = compute_leadership_score(
                ceo_data,
                insider_data,
                historical_growth,
                profit_margin
            )
            issues.extend(leadership_warns)
            
        except Exception as e:
            issues.append(f"Leadership analysis failed: {str(e)}")
    
    C = leadership_score
    # All modules have equal weight in the final wave score
    module_weights = [1] * 5  # [A, B, C, D, E] weights
    wave_core = gmean([A, B, C, D, E_mod], module_weights)
    wave = wave_core ** 1.15

    # Compute leadership score if symbol is provided
    leadership_score = 0.5  # neutral default
    leadership_tags = []
    leadership_explains = ["Tenure", "Growth", "Margins", "Insiders"]
    
    if symbol:
        try:
            # Fetch leadership and insider data
            ceo_data, ceo_warnings = fetch_ceo_data(symbol)
            insider_data, insider_warnings = fetch_insider_analysis(symbol)
            
            # Add any warnings to issues list
            issues.extend(ceo_warnings)
            issues.extend(insider_warnings)
            
            # Calculate historical growth and profit margins
            historical_growth = g_R_last if np.isfinite(g_R_last) else None
            profit_margin = m_op if np.isfinite(m_op) else None
            
            # Compute leadership score
            leadership_score, leadership_warns = compute_leadership_score(
                ceo_data, 
                insider_data,
                historical_growth,
                profit_margin
            )
            issues.extend(leadership_warns)
            
            # Add relevant tags
            if ceo_data and ceo_data.get("ceo_name"):
                leadership_tags.append(f"CEO: {ceo_data['ceo_name']}")
            if insider_data and insider_data.get("insider_confidence", {}).get("net_transactions", 0) > 0:
                leadership_tags.append("Net insider buying")
            elif insider_data and insider_data.get("insider_confidence", {}).get("net_transactions", 0) < 0:
                leadership_tags.append("Net insider selling")
                
        except Exception as e:
            issues.append(f"Leadership score computation failed: {str(e)}")
            leadership_score = 0.5  # fallback to neutral

    modules = {
        "A": {"score": round(A, 4), "explain": ["Growth", "Accel", "Margins", "CP", "Support"], "tags": []},
        "B": {"score": round(B, 4), "explain": ["Stability", "Coherence", "Liquidity", "NetDebt"], "tags": []},
        "C": {"score": round(leadership_score, 4), "explain": leadership_explains, "tags": leadership_tags},
        "D": {"score": round(D, 4), "explain": ["Sector", "Alignment"], "tags": []},
        "E": {"score": round(E_mod, 4), "explain": ["IdioVol", "Align"], "tags": []},
    }
    return modules, round(float(wave), 4), issues
