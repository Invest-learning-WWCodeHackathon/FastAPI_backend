import yfinance as yf
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor  # Add ThreadPoolExecutor import

class Stock:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)

    def get_info(self):
        return self.ticker.info

# ... [rest of the Stock class methods]

class StockSearchEngine:
    def search(self, query):
        stock = Stock(query)
        return stock

    def list_all_tickers(self, page=1, per_page=5):
        tickers = get_nasdaq_symbols()
        # Implementing pagination
        start = (page - 1) * per_page
        end = start + per_page
        return tickers[start:end]

    def get_multiple_stock_info(self, tickers):
        data = yf.download(tickers, group_by="ticker")
        return {ticker: data[ticker]['Close'].iloc[-1] for ticker in tickers}

def get_nasdaq_symbols():
    data = pd.read_csv(r"ticker_symbols.csv")
    rows = data.apply(lambda row: row.tolist(), axis=1)
    return rows

def get_stats(ticker):
    info = yf.Tickers(ticker).tickers[ticker].info
    print(f"{ticker} {info['currentPrice']} {info['marketCap']}")

def main():
    engine = StockSearchEngine()

    action = input("Choose action (search/list/multi-search/stats): ").strip().lower()

    if action == "search":
        query = input("Enter stock ticker to search: ")
        stock = engine.search(query)
        # Example usage to print out stock information
        print(stock.get_info())
    elif action == "list":
        page = int(input("Enter page number (default 1): ") or 1)
        per_page = int(input("Enter number of tickers per page (default 5): ") or 5)
        tickers = engine.list_all_tickers(page=page, per_page=per_page)
        print(tickers)
    elif action == "multi-search":
        tickers = input("Enter stock tickers separated by comma (e.g., AAPL,MSFT): ").split(",")
        data = engine.get_multiple_stock_info(tickers)
        print(data)
    elif action == "stats":  # Add a new action for fetching stock statistics
        ticker_list = input("Enter stock tickers separated by comma: ").split(",")
        with ThreadPoolExecutor() as executor:
            executor.map(get_stats, ticker_list)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
