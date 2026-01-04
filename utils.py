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
os.chdir(r"/Users/jiawei/Desktop/Code/Python/cryptopairtrading")



def load_pickle(path): #fast cache reload for large datasets and intermediate results.
    with lzma.open(path, 'rb') as fp:
        file = pickle.load(fp)
    return file 


def save_pickle(path, obj):
    with lzma.open(path, 'wb') as fp:
        pickle.dump(obj, fp)

def get_pnl_stats(last_weights, last_units, prev_close,ret_row,last_leverages):
    ret_row = np.nan_to_num(ret_row, nan=0, posinf=0,neginf=0)
    simple_ret = np.expm1(ret_row)
    day_pnl = np.sum(last_units * prev_close * simple_ret)
    nominal_ret = np.dot(last_weights, simple_ret)
    capital_ret = nominal_ret * last_leverages
    return day_pnl, nominal_ret, capital_ret


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
            self.dfs[inst]['ret'] = self.dfs[inst]['close'].diff()
            inst_vol = self.dfs[inst]['ret'].rolling(30).std()
            self.dfs[inst]['vol'] = inst_vol
            self.dfs[inst]['vol'] = self.dfs[inst]['vol'].ffill().fillna(0)
            self.dfs[inst]['vol'] = np.where(self.dfs[inst]['vol'] < 0.005, 0.005, self.dfs[inst]['vol'])
            eligibles.append(self.dfs[inst]['close'].notna().astype(int))


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
        date_range = pd.date_range(start, end, freq="D")
        self.compute_meta_info(trade_range=date_range)

        units_held, weights_held = [], []
        close_prev = None

        ewmas, ewstrats = [0.01], [1.0]
        strat_scalars = []

        capitals = [10000.0]
        nominal_rets, capital_rets = [0.0], [0.0]
        nominals, leverages = [], []
        
        for data in self.zip_data_generator():
            portfolio_i = data["portfolio_i"]
            ret_i = data["ret_i"]

            ret_row = data["ret_row"]
            log_close_row = data["close_row"]
            eligibles_row = data["eligibles_row"]
            vol_row = data["vol_row"]

            px_row = np.exp(log_close_row)

            strat_scalar = 2.0
            if portfolio_i != 0 and close_prev is not None:
                strat_scalar = self.get_strat_scaler(
                    target_vol=self.portfolio_vol,
                    ewmas=ewmas,
                    ewstrats=ewstrats
                )
                last_leverages = leverages[-1] if len(leverages) > 0 else 0.0
                day_pnl, nominal_ret, capital_ret = get_pnl_stats(
                    last_weights=weights_held[-1],
                    last_units=units_held[-1],
                    prev_close=close_prev,
                    ret_row=ret_row,
                    last_leverages=last_leverages
                )
                capitals.append(capitals[-1] + day_pnl)
                nominal_rets.append(float(nominal_ret))
                capital_rets.append(float(capital_ret))
                if capital_ret != 0:
                    ewmas.append(0.06 * (capital_ret ** 2) + 0.94 * ewmas[-1])
                    ewstrats.append(0.06 * strat_scalar + 0.94 * ewstrats[-1])
                else:
                    ewmas.append(ewmas[-1])
                    ewstrats.append(ewstrats[-1])
            else:
                capitals.append(capitals[-1])
                nominal_rets.append(0.0)
                capital_rets.append(0.0)
                ewmas.append(ewmas[-1])
                ewstrats.append(ewstrats[-1])
            
            strat_scalars.append(float(strat_scalar))
            forecasts = self.compute_signal_distribution(eligibles_row, ret_i)
            if isinstance(forecasts, pd.Series):
                forecasts = forecasts.values
            forecasts = np.asarray(forecasts, dtype=float)
            forecasts = np.where(eligibles_row.astype(bool), forecasts, 0.0)
            forecasts = np.nan_to_num(forecasts, nan=0.0, posinf=0.0, neginf=0.0)
            forecast_chips = float(np.sum(np.abs(forecasts)))
            """ #debug
            if portfolio_i < 5: 
                print("day", portfolio_i,
                      "sum|forecast| =", np.sum(np.abs(forecasts)),
                      "eligibles =", np.sum(eligibles_row))
            """
            vol_target = (self.portfolio_vol / np.sqrt(365.0)) * capitals[-1]

            if forecast_chips != 0:
                positions = (
                    strat_scalar
                    * (forecasts / forecast_chips)
                    * vol_target
                    / (vol_row * px_row)
                )
            else:
                positions = np.zeros(len(self.insts), dtype=float)
            positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
            nominal_tot = float(np.linalg.norm(positions * px_row, ord=1))
            units_held.append(positions)
            
            if nominal_tot != 0:
                weights = (positions * px_row) / nominal_tot
            else:
                weights = np.zeros(len(self.insts), dtype=float)
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights_held.append(weights)

            nominals.append(nominal_tot)
            leverages.append(nominal_tot / capitals[-1] if capitals[-1] != 0 else 0.0)

            close_prev = px_row
        units_df = pd.DataFrame(units_held, index=date_range, columns=[f"{inst}units" for inst in self.insts])
        weights_df = pd.DataFrame(weights_held, index=date_range, columns=[f"{inst}w" for inst in self.insts])

        nom_ser = pd.Series(nominals, index=date_range, name="nominal_tot")
        lev_ser = pd.Series(leverages, index=date_range, name="leverages")
        scaler_ser = pd.Series(strat_scalars, index=date_range, name="strat_scalar")

        cap_ser = pd.Series(capitals[1:], index=date_range, name="capital")
        nomret_ser = pd.Series(nominal_rets[1:], index=date_range, name="nominal_ret")
        capret_ser = pd.Series(capital_rets[1:], index=date_range, name="capital_ret")

        portfolio_df = pd.concat(
            [units_df, weights_df, lev_ser, scaler_ser, nom_ser, nomret_ser, capret_ser, cap_ser],
            axis=1
        )
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
                inst_weight[i] = self.stratdfs[i][f"{inst}w"] * self.stratdfs[i]["leverages"]
                inst_weight[i] = inst_weight[i].ffill().fillna(0)
            self.positions[inst] = inst_weight

    def compute_signal_distribution(self, eligibles, date):
        forecasts = np.zeros(len(self.insts))
        for j, inst in enumerate(self.insts):
            for i in range(len(self.stratdfs)):
                forecasts[j] += self.positions[inst].at[date, i] * (1/len(self.stratdfs))
        return forecasts