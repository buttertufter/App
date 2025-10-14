"""Leadership and management analysis utilities."""
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import requests
import yfinance as yf

from utils import requests_session, read_local_json, write_local_json, file_age_days
from data import _local_path, FINNHUB_API_KEY, FMP_API_KEY, OFFLINE_MODE, SESSION

def fetch_ceo_data(symbol: str, force_refresh: bool = False) -> Tuple[Optional[dict], List[str]]:
    """Fetch CEO information and tenure."""
    symbol = symbol.upper()
    local_path = _local_path("leadership", symbol, "json")
    warnings: List[str] = []
    
    if not force_refresh:
        data = read_local_json(local_path)
        if data:
            return data, []

    try:
        # Get basic info from yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        leadership_data = {
            "ceo_name": info.get("ceo"),
            "company_age": datetime.now().year - int(info.get("foundedYear", datetime.now().year)),
            "employee_count": info.get("fullTimeEmployees"),
            "sector": info.get("sector"),
            "industry": info.get("industry")
        }

        # Try to get additional data from OpenCorporates API (free tier)
        try:
            oc_response = SESSION.get(f"https://api.opencorporates.com/v0.4/companies/search?q={symbol}")
            if oc_response.status_code == 200:
                oc_data = oc_response.json()
                if oc_data.get("results"):
                    company = oc_data["results"]["companies"][0]
                    leadership_data["incorporation_date"] = company.get("incorporation_date")
        except Exception as e:
            warnings.append(f"OpenCorporates data unavailable: {str(e)}")

        # Add historical performance metrics
        financials = ticker.financials
        if not financials.empty:
            leadership_data["historical_performance"] = {
                "revenue_growth": financials.loc["Total Revenue"].pct_change().mean(),
                "profit_margins": (financials.loc["Net Income"] / financials.loc["Total Revenue"]).mean()
            }

        write_local_json(leadership_data, local_path)
        return leadership_data, warnings

    except Exception as e:
        warnings.append(f"Failed to fetch CEO data: {str(e)}")
        return None, warnings

def fetch_insider_analysis(symbol: str, force_refresh: bool = False) -> Tuple[Optional[dict], List[str]]:
    """Analyze insider trading patterns and SEC filings."""
    symbol = symbol.upper()
    local_path = _local_path("insider_analysis", symbol, "json")
    warnings: List[str] = []
    
    if not force_refresh:
        data = read_local_json(local_path)
        if data:
            return data, []

    try:
        # Get Finnhub insider trading data
        if FINNHUB_API_KEY:
            url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_API_KEY}"
            response = SESSION.get(url)
            if response.status_code == 200:
                insider_data = response.json()
                
                # Analyze insider trading patterns
                buys = [t for t in insider_data.get("data", []) if t.get("change") > 0]
                sells = [t for t in insider_data.get("data", []) if t.get("change") < 0]
                
                analysis = {
                    "insider_confidence": {
                        "buy_count": len(buys),
                        "sell_count": len(sells),
                        "net_transactions": len(buys) - len(sells),
                        "buy_volume": sum(t.get("change", 0) for t in buys),
                        "sell_volume": abs(sum(t.get("change", 0) for t in sells)),
                    }
                }
                
                # Try to get SEC filings from SEC API
                try:
                    sec_response = SESSION.get(
                        f"https://data.sec.gov/submissions/CIK{symbol}.json",
                        headers={"User-Agent": "Company Research Tool - Contact: your@email.com"}
                    )
                    if sec_response.status_code == 200:
                        filings = sec_response.json()
                        analysis["recent_filings"] = filings.get("filings", {}).get("recent", [])
                except Exception as e:
                    warnings.append(f"SEC filing data unavailable: {str(e)}")

                write_local_json(analysis, local_path)
                return analysis, warnings

    except Exception as e:
        warnings.append(f"Failed to analyze insider data: {str(e)}")
        return None, warnings

    return None, warnings

def compute_leadership_score(
    ceo_data: Optional[dict],
    insider_data: Optional[dict],
    historical_growth: Optional[float],
    profit_margin: Optional[float]
) -> Tuple[float, List[str]]:
    """Compute leadership competency score based on multiple factors.
    
    Parameters
    ----------
    ceo_data : dict | None
        CEO and company information including age, tenure, etc.
    insider_data : dict | None
        Insider trading analysis with transaction history
    historical_growth : float | None
        Historical revenue growth rate
    profit_margin : float | None
        Operating profit margin
        
    Returns
    -------
    score : float
        Leadership score in [0, 1]
    warns : list[str]
        Warnings encountered during computation
    """
    warns: List[str] = []
    
    # Component scores with weights from validation config
    weights = {
        "tenure": 0.2,
        "growth": 0.3,
        "margin": 0.2,
        "insider_trans": 0.15,
        "insider_vol": 0.15
    }
    
    scores = {}
    
    # 1. CEO Tenure and Company Maturity
    if ceo_data:
        company_age = ceo_data.get("company_age", 0)
        if company_age > 0:
            # Prefer companies 10-30 years old
            age_factor = min(1.0, company_age / 30.0) if company_age <= 30 else (40 - company_age) / 10.0
            scores["tenure"] = max(0.0, age_factor)
        else:
            warns.append("Company age data unavailable")
    else:
        warns.append("CEO data unavailable")

    # 2. Historical Growth
    if historical_growth is not None:
        # Center around typical growth rate of 10%
        scores["growth"] = min(1.0, max(0.0, 0.5 + historical_growth / 0.25))
    else:
        warns.append("Historical growth data unavailable")
        
    # 3. Profit Efficiency
    if profit_margin is not None:
        # Center around typical margin of 15%
        scores["margin"] = min(1.0, max(0.0, 0.5 + profit_margin / 0.15))
    else:
        warns.append("Profit margin data unavailable")

    # 4. Insider Trading Sentiment
    if insider_data and "insider_confidence" in insider_data:
        conf = insider_data["insider_confidence"]
        
        # Transaction count sentiment
        total_trans = conf.get("buy_count", 0) + conf.get("sell_count", 0)
        if total_trans > 0:
            scores["insider_trans"] = min(1.0, max(0.0, 
                0.5 + float(conf.get("buy_count", 0) - conf.get("sell_count", 0)) / max(5, total_trans)
            ))
            
        # Transaction volume sentiment
        buy_vol = conf.get("buy_volume", 0.0)
        sell_vol = conf.get("sell_volume", 0.0)
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            scores["insider_vol"] = min(1.0, max(0.0,
                0.5 + (buy_vol - sell_vol) / total_vol
            ))
    else:
        warns.append("Insider trading data unavailable")

    # Combine weighted scores, defaulting to neutral (0.5) for missing components
    final_score = 0.0
    total_weight = 0.0
    
    for component, weight in weights.items():
        if component in scores:
            final_score += scores[component] * weight
            total_weight += weight
    
    # Add neutral weight for missing components
    if total_weight < 1.0:
        remaining_weight = 1.0 - total_weight
        final_score += 0.5 * remaining_weight
        warns.append(f"Using neutral values for {remaining_weight*100:.0f}% of components")
    
    return min(1.0, max(0.0, final_score)), warns