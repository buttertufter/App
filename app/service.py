from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from data import (
    fetch_quarterlies,
    fetch_prices,
    fetch_peer_prices,
    get_universe,
    discover_peers
)
from model import compute_modules

app = FastAPI()

class TickerResponse(BaseModel):
    asOf: str
    modules: dict
    wave: float

@app.get("/api/ticker/{symbol}", response_model=TickerResponse)
def get_ticker(symbol: str):
    universe = get_universe()
    try:
        q_fin, _ = fetch_quarterlies(symbol, max_quarters=12)
        price, _ = fetch_prices(symbol, years=3)
        peers, _ = discover_peers(symbol, universe)
        peer_prices, _ = fetch_peer_prices(peers, years=3)
        modules, wave, issues = compute_modules(q_fin, price, peer_prices)
        if modules is None:
            raise HTTPException(status_code=404, detail="Insufficient data for this symbol.")
        return TickerResponse(
            asOf=pd.Timestamp.now().strftime("%Y-%m"),
            modules=modules,
            wave=wave
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
