import lzma  # compress files, higher comression ratio than gzip
import os
from pathlib import Path 
import dill as pickle
import pandas as pd 
import numpy as np
import random
from datetime import timedelta  # for run sim purposes, check again
from copy import deepcopy
from collections import defaultdict
from timeme import timeme
os.chdir(r"C:/Users/Jiawe.JIAWEI/OneDrive/Desktop/Coding/Python/cryptostatarb")



def load_pickle(path): #fast cache reload for large datasets and intermediate results.
    with lzma.open(path, 'rb') as fp:
        file = pickle.load(fp)
    return file 


def save_pickle(path, obj):
    with lzma.open(path, 'wb') as fp:
        pickle.dump(obj, fp)

class AbstractImplementationException(Exception):
    pass


class Alpha():
    def __init__(self, insts, dfs, start, end, portfolio_vol = 0.20):
        self.insts = insts
        self.dfs = deepcopy(dfs) #ensure each pair get own data
        self.start = start 
        self.end = end
        self.portfolio_vol = portfolio_vol

    def pre_compute(self,trade_range):
        pass

    def post_compute(self,trade_range):
        pass

    def compute_signal_distribution(self, eligibles, date):
        raise AbstractImplementationException('No concrete implemental for signal generation')
    
    #the following 3 function will be overide by their own respective alpha1,2,3

    def get_strat_scaler(self, target_vol, ewmas, ewstrats):
        ann_realized_vol = np.sqrt(ewmas[-1] * 365) #annualized daily variance then sqrt to annualized vol
        return target_vol / ann_realized_vol * ewstrats[-1] #scaling factor based on strategy recent performance
    

    def compute_meta_info(self, trade_range):
        self.pre_compute(trade_range=trade_range)
        def is_any_one(x):
            return int(np.any(x))
        
        closes, eligibles, vols , rets = [], [], [], []

        for inst in self.insts:
            df = pd.DataFrame(index=trade_range)
            self.dfs[inst] = df.join(self.dfs[inst]).ffill()
            self.dfs[inst]['ret'] = -1 + self.dfs[inst]['close'] / self.dfs[inst]['close'].shift(1)
            inst_vol = self.dfs[inst]['ret'].rolling(30).std()
            self.dfs[inst]['vol'] = inst_vol
            self.dfs[inst]['vol'] = self.dfs[inst]['vol'].ffill().fillna(0)
            self.dfs[inst]['vol'] = np.where(self.dfs[inst]['vol'] < 0.005, 0.005, self.dfs[inst]['vol'])
            sampled = self.dfs[inst]['close'] != self.dfs[inst]['close'].shift(1)
            sampled = sampled.fillna(False)
            eligible = sampled.rolling(5).apply(is_any_one, raw=True).fillna(0)
            eligible = eligible.astype(int) & self.dfs[inst]['close'].notna().astype(int)
            eligibles.append(eligible.astype(int) & (self.dfs[inst]['close'] > 0).astype(int))

            closes.append(self.dfs[inst]['close'])
            vols.append(self.dfs[inst]['vol'])
            rets.append(self.dfs[inst]['ret'])

        self.eligiblesdf = pd.concat(eligibles,axis=1)
        self.eligiblesdf.columns = self.insts
        self.closedf = pd.concat(closes,axis=1)
        self.closedf.columns = self.insts
        self.voldf = pd.concat(vols,axis=1)
        self.voldf.columns = self.insts
        self.retdf = pd.concat(rets,axis=1)
        self.retdf.columns = self.insts

        self.post_compute(trade_range=trade_range) 
        return 
    

    def run_simulation(self):
        start = self.start
        end = self.end

        date_range= pd.date_range(start,end,freq='D')

        self.compute_meta_info(trade_range=date_range)


        units_held, weights_held = [], []
        close_prev = None
        ewmas, ewstrats = [0.01], [1]
        strat_scalars = []
        capitals, nominal_rets, capital_rets = [10000.0], [0.0], [0.0]
        nominals, leverages = [], []

        for data in self.zip_data_generator():
            portfolio_i = data['portfolio_i']
            ret_i = data['ret_i']
            ret_row = data['ret_row']
            close_row = data['close_row']
            eligibles_row = data['eligibles_row']
            vol_row = data['vol_row']

            strat_scalar = 2

            if portfolio_i != 0 and close_prev is not None:
                strat_scalar = self.get_strat_scaler(
                    target_vol = self.portfolio_vol,
                    ewmas = ewmas,
                    ewstrats = ewstrats
                )
            
                day_pnl, nominal_ret, capital_ret = get_pnl_stats(
                    last_weights = weights_held[-1], 
                    last_units = units_held[-1], 
                    prev_close = close_prev,  
                    ret_row = ret_row, 
                    leverages= leverages
                )

                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(nominal_ret)
                capital_rets.append(capital_ret)
                ewmas.append(0.06 * (capital_ret**2) + 0.94 * ewmas[-1] if capital_ret != 0 else ewmas[-1])
                ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1] if capital_ret != 0 else ewstrats[-1])
            else:
                capitals.append(capitals[-1])
                nominal_rets.append(0.0)
                capital_rets.append(0.0)
                ewmas.append(ewmas[-1])
                ewstrats.append(ewstrats[-1])
                 
            strat_scalars.append(strat_scalar)
            forecasts = self.compute_signal_distribution(eligibles_row, ret_i) #use ret_i as date
            if type(forecasts) == pd.Series: forecasts = forecasts.values
            forecasts = forecasts / eligibles_row #forces all ineligible entries into inf/NaN, making it impossible for a stray value to survive
            forecasts = np.nan_to_num(forecasts, nan=0,posinf=0,neginf=0)
            forecast_chips = np.sum(np.abs(forecasts))

            vol_target = (self.portfolio_vol / np.sqrt(253)) * capitals[-1]
            positions = strat_scalar * forecasts/forecast_chips * vol_target / (vol_row * close_row) if forecast_chips != 0 else np.zeros(len(self.insts)) 

            positions = np.nan_to_num(positions, nan=0,posinf=0,neginf=0)
            nominal_tot = np.linalg.norm(positions * close_row, ord=1)
            units_held.append(positions)
            weights = positions * close_row / nominal_tot if nominal_tot != 0 else np.zeros(len(self.insts))
            weights = np.nan_to_num(weights, nan=0,posinf=0,neginf=0)
            weights_held.append(weights)
            
            nominals.append(nominal_tot)
            leverages.append(nominal_tot/capitals[-1])

            close_prev = close_row

        units_df = pd.DataFrame(data = units_held, index=date_range, columns=[inst + 'units' for inst in self.insts])
        weights_df = pd.DataFrame(data = weights_held, index=date_range, columns=[inst + 'w' for inst in self.insts])
        nom_ser = pd.Series(data=nominals, index= date_range, name='nominal_tot')
        lev_ser = pd.Series(data=leverages, index= date_range, name='leverages')
        cap_ser = pd.Series(data=capitals[1:], index= date_range, name='capital')
        nomret_ser = pd.Series(data=nominal_rets[1:], index= date_range, name='nominal_ret')
        capret_ser = pd.Series(data=capital_rets[1:], index= date_range, name='capital_ret')
        scaler_ser = pd.Series(data=strat_scalars, index= date_range, name='strat_scalar')
        portfolio_df = pd.concat([units_df, weights_df, lev_ser, scaler_ser, nom_ser, nomret_ser, capret_ser, cap_ser], axis=1)
        return portfolio_df
    
    def zip_data_generator(self):
        for (portfolio_i),(ret_i, ret_row),(close_i, close_row), \
            (eligibles_i, eligibles_row),(vol_i, vol_row) in zip (range(len(self.retdf)),
            self.retdf.iterrows(), self.closedf.iterrows(), self.eligiblesdf.iterrows(), self.voldf.iterrows()
            ): #iterrows() - like a df for each index, zip walk one index at a time
            yield {
                "portfolio_i": portfolio_i,
                "ret_i": ret_i,
                "ret_row": ret_row.values,
                "close_row": close_row.values,
                "eligibles_row": eligibles_row.values,
                "vol_row": vol_row.values,
            }

class Portfolio(Alpha):

    def __init__(self, insts, dfs, start, end, stratdfs):
        super().__init__(insts, dfs, start,end)
        self.stratdfs = stratdfs
    
    def post_compute(self, trade_range):
        self.positions = {}
        for inst in self.insts:
            inst_weight=pd.DataFrame(index=trade_range)
            for i in range(len(self.stratdfs)):
                inst_weight[i] = self.stratdfs[i]["{} w".format(inst)]\
                    *self.stratdfs[i]['leverage']
                inst_weight[i] = inst_weight[i].ffill().fillna(0)
            self.positions[inst] = inst_weight

    def compute_signal_distribution(self, eligibles, date):
        forecasts = defaultdict(float)
        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                forecasts[inst] += self.positions[inst].at[date,i] * (1/len(self.stratdfs)) #parity risk allocation
        return forecasts, np.sum(np.abs(list(forecasts.values())))