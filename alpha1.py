from utils import Alpha 
import pandas as pd 
import numpy as np

class Alpha1(Alpha):
    def __init__(
        self,
        insts,
        dfs,
        start,
        end,
        pairs,
        portfolio_vol=0.20,
        entry_z=2.0,
        exit_z=0.0,
        lookback_mode="half_life",
        fixed_lookback=60,
        min_lookback=20,
        max_lookback=180,
        use_prev_close=True,
    ):
        super().__init__(insts, dfs, start, end, portfolio_vol)
        self.pairs = pairs
        self.entry_z = entry_z
        self.exit_z = exit_z

        self.inst_index = {inst: i for i, inst in enumerate(insts)}
        self.spreads = {}
        self.zscores = {}

    def pre_compute(self, trade_range):
        for (y, x), stats in self.pairs.items():
            beta = stats["beta"]

            spread = (
                self.dfs[y]["close"]
                - beta * self.dfs[x]["close"]
            )

            lookback = int(stats["half_life"])
            lookback = max(20, min(lookback, 180))

            mean = spread.rolling(lookback).mean()
            std = spread.rolling(lookback).std()
            z = (spread - mean) / std

            self.spreads[(y, x)] = spread
            self.zscores[(y, x)] = z

    def compute_signal_distribution(self, eligibles, date):
        forecasts = np.zeros(len(self.insts))

        for (y, x), stats in self.pairs.items():
            z = self.zscores[(y, x)].get(date, np.nan)
            if np.isnan(z):
                continue

            iy = self.inst_index[y]
            ix = self.inst_index[x]
            beta = stats["beta"]

            if z > self.entry_z:
                forecasts[iy] -= 1.0
                forecasts[ix] += beta

            elif z < -self.entry_z:
                forecasts[iy] += 1.0
                forecasts[ix] -= beta

        return forecasts
