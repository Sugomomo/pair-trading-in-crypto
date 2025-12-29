import pandas as pd
import numpy as np
import threading
import requests
import pytz 
from datetime import datetime
import matplotlib.pyplot as plt
#import ccxt - CCXT library for cryptocurrency trading and market data, will be using yfinance easier
import yfinance
import utils

import os
from pathlib import Path
os.chdir(r"C:/Users/Jiawe.JIAWEI/OneDrive/Desktop/Coding/Python/cryptostatarb")


start = datetime(2015,1,1, tzinfo=pytz.utc)
end = datetime(2025,1,1,tzinfo=pytz.utc)

tickers = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD", "USDC-USD", "SOL-USD", "TRX-USD", "STETH-USD", "WTRX-USD", "DOGE-USD", 'ADA-USD', "BCH-USD",\
           "WSTETH-USD", "WBTC-USD", "WBETH-USD", "WETH-USD", "AETHWETH-USD", "USDS33039-USD", "WEETH-USD"] 