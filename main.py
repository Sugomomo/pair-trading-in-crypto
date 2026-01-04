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
os.chdir(r"/Users/jiawei/Desktop/Code/Python/cryptopairtrading")

start = datetime(2020,1,1)
end = datetime(2025,1,1)

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
        
        price_cols = ["open", "high", "low", "close"]
        df[price_cols]= df[price_cols].replace(0,np.nan)
        df[price_cols] = np.log(df[price_cols])

       
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[(df["datetime"] >= pd.Timestamp(start, tz="UTC")) &
        (df["datetime"] <  pd.Timestamp(end, tz="UTC"))]
        df["datetime"] = df["datetime"].dt.tz_convert(None)
        df = df.set_index("datetime").drop(columns="timestamp")
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
    from pairselect import run_pair
    #from pairselect import plot_spread
    tickers_kept, ticker_dfs = get_ticker_dfs(tickers,start,end)

    log_prices = pd.concat(
    {t.split("/")[0]: df["close"] for t, df in ticker_dfs.items()},
    axis=1).sort_index()
    log_prices = log_prices[~log_prices.index.duplicated(keep="first")]

    pairs, train_df, test_df = run_pair(
    log_prices, 
    train_start="2020-01-01",
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2025-01-01",
)
    print("Selected pairs:", len(pairs))
    for (y, x), stats in pairs.items():print(
        f"{y} ~ {x} | "
        f"beta={stats['beta']:.3f}, "
        f"p={stats['adf_pvalue']:.4f}, "
        f"half-life={stats['half_life']:.1f} days"
    )
    
    pairs_df = (pd.DataFrame.from_dict(pairs, orient="index").rename_axis(index=["y","x"]).sort_values("adf_pvalue"))
    pairs_df.to_csv('selected_pairs.csv')

    selected_pairs = pairs
    selected_assets = sorted(set([a for (y, x) in selected_pairs.keys() for a in (y, x)]))
    test_start="2023-01-01"
    test_end="2025-01-01"
    dfs_for_alpha = {}
    for t, df in ticker_dfs.items():
        asset = t.split("/")[0]
        if asset in selected_assets:
            dfs_for_alpha[asset] = df[["close"]].copy()
    from alpha1 import Alpha1
    alpha1 = Alpha1(
        insts=selected_assets,
        dfs=dfs_for_alpha,
        start=test_start,
        end=test_end,
        pairs=selected_pairs,
        portfolio_vol=0.20,
        entry_z=2.0,
        exit_z=0.5,
        lookback_mode="half_life",
        fixed_lookback=60,
        min_lookback=20,
        max_lookback=180,
        use_prev_close=True
    )
    df1 = alpha1.run_simulation()
    print(df1["capital"].iloc[-1])


    


    """
    example_pair = list(pairs.items())[:5] #top 5 pair 

    for (pair, stats) in example_pair:
        plot_spread(log_prices, pair, stats['beta'], train_start="2020-01-01",train_end="2022-12-31",test_start="2023-01-01",test_end="2025-01-01",)

    """


"""
    for t in tickers_kept:
        print(f"\n==== {t} ====")
        print(ticker_dfs[t].head())
"""

if __name__ == '__main__':
    main()

#current work, but still missing a few pair due to exchange, date problem
#individual alpha are to be created
