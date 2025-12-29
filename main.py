import pandas as pd
import numpy as np
import threading
import requests
import pytz 
from datetime import datetime
import matplotlib.pyplot as plt
import ccxt # CCXT library for cryptocurrency trading and market data
import utils

import os
from pathlib import Path
os.chdir(r"C:/Users/Jiawe.JIAWEI/OneDrive/Desktop/Coding/Python/cryptostatarb")


start = datetime(2015,1,1, tzinfo=pytz.utc)
end = datetime(2025,1,1,tzinfo=pytz.utc)

top20_tickers = [
    "BTC","ETH","XRP","SOL","ADA","DOGE","LTC","BCH","LINK","DOT",
    "ATOM","AVAX","UNI","XLM","ETC","FIL","AAVE","ALGO","MKR","EOS"
]
 #top 20 by market cap excluding USDT, USDC

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def make_exchange(exchange_id: str):
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    ex.load_markets()
    print("num markets:", len(ex.markets), "num symbols:", len(ex.symbols))
    return ex

def available_tickers(exchange, candidates: list[str]) -> list[str]:
    bases = set(candidates)
    out = []
    for m in exchange.markets.values():
        if m.get("spot", False) and m.get("quote") == "USDT" and m.get("base") in bases:
            out.append(m["symbol"])
    return sorted(set(out))

def fetch_ohlcv_full(exchange, ticker: str, timeframe: str, start_dt: datetime, end_dt: datetime, limit: int = 1000, tries: int = 0) -> pd.DataFrame:
    since = _ms(start_dt)
    end_ms = _ms(end_dt)
    rows = []

    while since < end_ms:
        batch = exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    df = df[(df.index >= start_dt) & (df.index < end_dt)]
    return df

def get_histories(exchange_id: str, timeframe: str, start_dt: datetime, end_dt: datetime, candidates: list[str]):
    ex_for_markets = make_exchange(exchange_id)
    tickers = available_tickers(ex_for_markets, top20_tickers)

    dfs = [None] * len(tickers)

    def _helper(i):
        t = tickers[i]
        print(tickers[i])
        try: 
            ex = make_exchange(exchange_id)
            dfs[i] = fetch_ohlcv_full(ex, t, timeframe, start_dt, end_dt)
        except Exception as err:
            dfs[i] = pd.DataFrame()

    threads = [threading.Thread(target=_helper, args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]

    return tickers, dfs


def get_ticker_dfs(exchange_id: str, timeframe: str):
    from utils import load_pickle, save_pickle
    fname = f"dataset_{exchange_id}_{timeframe}.obj"
    try:
        tickers, ticker_dfs = load_pickle(fname)
        return tickers, ticker_dfs
    except Exception as err:
        tickers, dfs = get_histories(exchange_id, timeframe, start, end, top20_tickers)
        ticker_dfs = {t: d for t, d in zip(tickers, dfs)}
        save_pickle(fname, (tickers, ticker_dfs))
        return tickers, ticker_dfs

def main():
   exchange_id = "okx"
   timeframe = "1d"
   tickers, ticker_dfs = get_ticker_dfs(exchange_id, timeframe)
   print("available tickers:", len(tickers))
   print('testing...')
   if tickers:
       t0 = tickers[0]
       print(t0, ticker_dfs[t0].head())

if __name__ == '__main__':
    main()