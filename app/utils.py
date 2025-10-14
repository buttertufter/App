from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EPS = 1e-6
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF = 0.5

_session: Optional[Session] = None


def requests_session() -> Session:
    """Return a shared requests session with retry/backoff."""
    global _session
    if _session is None:
        session = Session()
        retry = Retry(
            total=HTTP_MAX_RETRIES,
            backoff_factor=HTTP_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _session = session
    return _session


def logistic_sigmoid(x, kappa=4.0):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-kappa * x))


def geo_mean(xs, ws=None, eps=1e-6):
    xs = np.asarray(xs, dtype=float)
    if ws is None:
        ws = np.ones_like(xs)
    ws = np.asarray(ws, dtype=float)
    return float(np.exp(np.sum(ws * np.log(xs + eps)) / np.sum(ws)))


def ema(s: pd.Series, span: int = 4):
    return s.ewm(span=span, adjust=False).mean()


def ttm_sum(q_series: pd.Series, window_quarters: int = 4):
    return q_series.rolling(window_quarters).sum()


def slope_log(series: pd.Series, window: int = 4):
    """OLS slope of log(series) over rolling window; returns per-period log-change."""
    series = series.copy()
    series = series.replace(0, EPS)
    s = np.log(np.maximum(series, EPS))
    idx = np.arange(len(s))
    out = pd.Series(index=series.index, dtype=float)
    for t in range(len(s)):
        if t + 1 < window:
            out.iloc[t] = np.nan
            continue
        x = idx[t - window + 1 : t + 1].astype(float)
        y = s.iloc[t - window + 1 : t + 1].astype(float).values
        xm, ym = x.mean(), y.mean()
        denom = np.sum((x - xm) ** 2)
        out.iloc[t] = np.nan if denom == 0 else float(
            np.sum((x - xm) * (y - ym)) / denom
        )
    return out


def stdev_over(series: pd.Series, window: int = 4):
    return series.rolling(window).std()


def pct_clip(x, lo=-0.99, hi=0.99):
    return float(np.clip(x, lo, hi))


sig = logistic_sigmoid
geo = geo_mean


def to_series_1d(x, name: Optional[str] = None) -> pd.Series:
    if x is None:
        return pd.Series(dtype="float64", name=name)
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = pd.Series(x.iloc[:, 0].values.flatten(), index=x.index, name=name)
        else:
            s = x.mean(axis=1)
    else:
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        s = pd.Series(arr, name=name, dtype="float64")
    s = pd.to_numeric(s, errors="coerce").dropna()
    if name is not None:
        s.name = name
    return s


def last_or_default(x, default=np.nan):
    s = to_series_1d(x)
    return float(s.iloc[-1]) if len(s) else default


def nth_from_end_or_default(x, n=1, default=np.nan):
    if n <= 0:
        return default
    s = to_series_1d(x)
    if len(s) < n:
        return default
    return float(s.iloc[-n])


def file_age_days(path) -> float:
    p = Path(path)
    if not p.exists():
        return float("inf")
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age.total_seconds() / 86400.0


def load_cached_frame(path, index_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        parse_dates = [index_col] if index_col else None
        df = pd.read_csv(p, parse_dates=parse_dates)
    except Exception:
        return None
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def save_cached_frame(path, df: pd.DataFrame, index_name: Optional[str] = None) -> None:
    if df is None or df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy().sort_index().reset_index()
    if index_name:
        first_col = export.columns[0]
        export = export.rename(columns={first_col: index_name})
    export.to_csv(p, index=False)


def read_local_csv(path: str | Path) -> pd.DataFrame:
    try:
        path = Path(path)
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
    except Exception as exc:
        print(f"[warn] failed reading {path}: {exc}")
    return pd.DataFrame()


def write_local_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        print(f"[warn] failed saving {path}: {exc}")


def read_local_json(path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f"[warn] failed reading {path}: {exc}")
    return {}


def write_local_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    except Exception as exc:
        print(f"[warn] failed saving {path}: {exc}")
