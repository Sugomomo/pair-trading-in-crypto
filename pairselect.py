import pandas as pd 
import numpy as np
import statsmodels.api as sm
from itertools import combinations 
from statsmodels.tsa.stattools import adfuller 
from copy import deepcopy


def train_test_split(log_prices:pd.DataFrame, train_start:str, train_end:str, test_start:str, test_end:str):
    train = deepcopy(log_prices.loc[train_start:train_end]) #ensure no modification to original data
    test = deepcopy(log_prices.loc[test_start:test_end])

    if train.empty or test.empty:
        return ValueError("Data invalid")
    
    common = train.columns.intersection(test.columns)
    train = train[common] #filter
    test = test[common]
    return train, test

def engle_granger(y:pd.Series, x:pd.Series, adf_thres=0.05, min_obs=180):
    df = pd.concat([y, x], axis=1).dropna()
    if len(df)<min_obs:
        return None 
    
    y = df.iloc[:,0] #first col
    x = df.iloc[:,1]

    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    beta = float(model.params.iloc[1])
    spread = y - beta * x

    adf, pvalue, *_ = adfuller(spread) #only adf and pvalue needed
    if pvalue<adf_thres: #only significant pairs, standard pvalue<0.05
        return {
            'beta': beta,
            'adf_pvalue': float(pvalue),
            'spread': spread
        }
    return None

def estimate_half_life(spread:pd.Series) -> float:
    spread = spread.dropna()
    if len(spread)<60:
        return np.inf
    
    delta = spread.diff().dropna()
    lagged = spread.shift(1).dropna()
    delta = delta.loc[lagged.index]

    x = sm.add_constant(lagged)
    res = sm.OLS(delta, x).fit()
    b = float(res.params.iloc[1])

    if b>=0:
        return np.inf
    
    return float(np.log(2) / (-b))

def cointegration_pair(train_log: pd.DataFrame, adf_thres=0.05, min_half = 2, max_half=90, min_obs=180, rank:str='adf_pvalue'):
    chosen = {}

    for x_asset, y_asset in combinations(train_log.columns,2):
        x = train_log[x_asset]
        y = train_log[y_asset]

        res = engle_granger(y, x, adf_thres=adf_thres, min_obs=min_obs)
        if res is None:
            continue 
        half_life = estimate_half_life(res['spread'])
        if not (min_half <= half_life <= max_half): 
            continue

        chosen[(y_asset, x_asset)] = {
            'beta': res['beta'],
            'adf_pvalue': res['adf_pvalue'],
            'half_life': half_life
        }

    if rank in {'adf_pvalue', 'half_life'}:
        chosen = dict(sorted(chosen.items(), key=lambda item: item[1][rank]))
    return chosen

def run_pair(log_prices: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    *,
    adf_thres = 0.05,
    min_half = 2,
    max_half = 90,
    min_obs = 180,
    rank_by: str = 'adf_pvalue',
):
    train, test = train_test_split(log_prices, train_start, train_end, test_start, test_end)
    pairs = cointegration_pair(
        train,
        adf_thres=adf_thres,
        min_half=min_half,
        max_half=max_half,
        min_obs=min_obs,
        rank=rank_by
    )
    return pairs, train, test 