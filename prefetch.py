"""Prefetch utility to populate the local data_store before running offline."""

from app.data import fetch_prices, fetch_quarterlies

SYMBOLS = ["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]


def main():
    for symbol in SYMBOLS:
        print(f"Prefetching {symbol}...")
        try:
            q_fin, q_warn = fetch_quarterlies(symbol, force_refresh=True)
            if q_warn:
                print("  quarterlies:", "; ".join(q_warn))
        except Exception as exc:
            print(f"  quarterlies failed: {exc}")
        try:
            price, p_warn = fetch_prices(symbol, years=5, force_refresh=True)
            if p_warn:
                print("  prices:", "; ".join(p_warn))
        except Exception as exc:
            print(f"  prices failed: {exc}")


if __name__ == "__main__":
    main()
