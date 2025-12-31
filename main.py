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
os.chdir(r"address")


start = datetime(2020,1,1, tzinfo=pytz.utc)
end = datetime(2025,1,1,tzinfo=pytz.utc)

tickers = [
    "BTC/USDT","ETH/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","HYPE/USDT","BCH/USDT","LINK/USDT","ZEC/USDT",
    "SHIB/USDT","XMR/USDT","USDE/USDT","XLM/USDT","LTC/USDT","SUI/USDT","AVAX/USDT","CC/USDT","BNB/USDT","TRX/USDT"
]#top 20 by market cap excluding USDT, USDC. List is filtered accordingly with CoinMarketCap, if coin not available in Kucoin then next ranking will be chosen


exchange = ccxt.kucoin({"enableRateLimit": True})
kucoin_market = exchange.load_markets()

##symbols = [item for item in kucoin_market if item.endswith('USDT')] #gather all USDT pairs, this will only be used after testing finish top20_tickers for simplicity sake

"""
for i in range(len(tickers)): #to check whether ticker exist 
    if tickers[i] in symbols:
        continue 
    else:
        print(tickers[i])
"""

def get_history(ticker, start, end, timeframe ='1d', tries =0):
    try:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        since = start_ms
        rows = []
        limit = 1500 

        while True:
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since,limit=limit)
            if not ohlcv:
                break 
            rows.extend(ohlcv)
            last = ohlcv[-1][0]
            if last >= end_ms:
                break 
            since = last + 1

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns="timestamp")
        df = df.loc[(df.index >= start) & (df.index < end)]
        df = df[~df.index.duplicated(keep="first")]
        return df

    except Exception as err:
        if tries < 5:
            return get_history(ticker, start, end, timeframe ='1d', tries=tries+1)
        print(f"Failed {ticker}: {err}")
        return pd.DataFrame()
    
def get_histories(tickers, start, end, timeframe='1d'):
    dfs = [None] * len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(tickers[i], start, end, timeframe=timeframe)
        dfs[i] = df
    #note that for 20 tickers threading may not neccessarily be needed, however if tested on larger then it will be needed for efficiency   
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]

    kept = [(tickers[i], dfs[i]) for i in range(len(tickers)) if dfs[i] is not None and not dfs[i].empty]
    tickers = [t for t, _ in kept]
    dfs = [df for _, df in kept]
    return tickers, dfs


def get_ticker_dfs(tickers,start,end):  
    from utils import load_pickle, save_pickle
    try:
        tickers_saved, ticker_dfs = load_pickle('dataset.obj') #load previously saved data if exist 
    except Exception as err:
        tickers_kept,dfs = get_histories(tickers,start,end,timeframe='1d')
        ticker_dfs = {ticker:df for ticker, df in zip(tickers_kept,dfs)} #dict of {ticker : dataframe}   
        save_pickle('dataset.obj',(tickers_kept,ticker_dfs)) #save data for future use      
        return tickers_kept, ticker_dfs 
    return tickers_saved, ticker_dfs


def main():
    tickers_kept, ticker_dfs = get_ticker_dfs(tickers,start,end)

    for t in tickers_kept:
        print(f"\n==== {t} ====")
        print(ticker_dfs[t].head())


if __name__ == '__main__':
    main()
<<<<<<< HEAD

#current work, but still missing a few pair due to exchange, date problem
=======
>>>>>>> c9a0f4ea6840851e00cc808bd1753d27494e2cee
