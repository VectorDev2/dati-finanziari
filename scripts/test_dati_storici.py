import yfinance as yf
import argparse

def get_last_10_days_data(ticker: str):
    df = yf.download(ticker, period="10d")
    if df.empty:
        print(f"Errore: nessun dato trovato per il ticker {ticker}")
        return
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, help="Simbolo del titolo da scaricare")
    args = parser.parse_args()

    get_last_10_days_data(args.ticker)
