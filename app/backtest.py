import pandas as pd
from data import get_universe, fetch_quarterlies, fetch_prices, fetch_peer_prices, discover_peers
from model import compute_modules

def backtest_wave(start_year=2016, end_year=2024, top_decile=0.1):
    universe = get_universe()
    results = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dt = pd.Timestamp(year=year, month=month, day=1)
            scores = []
            for symbol in universe["symbol"]:
                try:
                    q_fin, _ = fetch_quarterlies(symbol, max_quarters=12)
                    price, _ = fetch_prices(symbol, years=3)
                    peers, _ = discover_peers(symbol, universe)
                    peer_prices, _ = fetch_peer_prices(peers, years=3)
                    modules, wave, _ = compute_modules(q_fin, price, peer_prices)
                    scores.append((symbol, wave))
                except Exception:
                    continue
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_n = int(len(scores) * top_decile)
            held = [s[0] for s in scores[:top_n]]
            results.append({"date": dt, "held": held, "waves": scores})
    return results

if __name__ == "__main__":
    results = backtest_wave()
    # Save or plot results as needed
    print(results[:2])
