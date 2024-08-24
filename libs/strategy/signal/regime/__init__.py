'''
Author: J , jwsun1987@gmail.com
Date: 2023-02-01 01:56:16
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
from typing import Union, Dict, List

from utils.data import *
from utils.dates import *

from strategy.signal import Signal
from strategy.signal.enum import Signals


class Regime(Signal):
    def __init__(
        self,
        name: str = Signals.WEI,
        regimes: pd.DataFrame = None,
        regime_dict: dict = None,
        **kwargs
    ) -> None:
        super().__init__(name)
        self.regimes = regimes
        if self.regimes is not None:
            self.regimes = self.roll_to_business_day(regimes)
        self.regime_dict = regime_dict
        
        for k, v in kwargs.items():
            setattr(self, k, v)


    def roll_to_business_day(self, data: pd.DataFrame):
        data.index = pd.to_datetime(data.index.tolist())
        data.index = [
            date
            if date.weekday() in [0, 1, 2, 3, 4]
            else date - timedelta(days=date.weekday() - 4)
            for date in data.index.tolist()
        ]
        return data

    def regime_change(self, date: Union[str, pd.Timestamp] = None):
        if date is not None:
            regimes = self.regimes.loc[self.regimes.index <= pd.to_datetime(date)]
        else:
            regimes = self.regimes

        if regimes.Regime.iloc[-1] != regimes.Regime.iloc[-2]:
            self.logger.info(
                str(regimes.index[-2].date())
                + ": "
                + self.regime_dict[regimes.Regime.iloc[-2]]
                + "-> "
                + str(regimes.index[-1].date())
                + ": "
                + self.regime_dict[regimes.Regime.iloc[-1]]
            )
            return True
        else:
            return False

    def start_dates(self, regimes: Union[List[int], int] = None):
        if regimes is None:
            regimes = list(self.regime_dict.keys())
        regimes = to_list(regimes)

        regime_dates = {}
        for regime in regimes:
            periods = find_nonzero_runs(
                (self.regimes.Regime == regime).astype(int).values
            )

            if len(periods) == 0:
                continue
            regime_dates.update(
                {self.regime_dict[regime]: [self.regimes.index[p[0]] for p in periods]}
            )
        return regime_dates

    def duration(self, regimes: Union[List[int], int] = None):
        if regimes is None:
            regimes = list(self.regime_dict.keys())
        regimes = to_list(regimes)

        regime_duration = {}
        for regime in regimes:
            periods = find_nonzero_runs(
                (self.regimes.Regime == regime).astype(int).values
            )
            if len(periods) == 0:
                continue
            regime_duration.update(
                {self.regime_dict[regime]: [p[1] - p[0] for p in periods]}
            )
        return regime_duration

    def get_regime(self, date: Union[str, pd.Timestamp] = None):
        if date is not None:
            regimes = self.regimes.loc[self.regimes.index <= pd.to_datetime(date)]
        else:
            regimes = self.regimes
        return regimes.iloc[-1].Regime

    def get_regime_data(
        self,
        data: Union[pd.DataFrame, pd.Series],
        regimes: Union[List[int], int] = None,
    ):
        if regimes is None:
            regimes = list(self.regime_dict.keys())
        regimes = to_list(regimes)

        data = pd.DataFrame(data) if isinstance(data, pd.Series) else data
        #data = data.groupby(pd.Grouper(freq=self.frequency)).last()
        #data = self.roll_to_business_day(data)

        regime_data = {}
        for regime in regimes:
            periods = find_nonzero_runs(
                (self.regimes.Regime == regime).astype(int).values
            )
            if len(periods) == 0:
                continue

            period_data = []
            for period in periods:
                if period[0] <= data.shape[0]:
                    periodDf = data[period[0] : period[1]]
                    if periodDf.index.max() == self.regimes.index.max():
                        continue
                    period_data.append(periodDf)
            regime_data.update({self.regime_dict[regime]: period_data})
        return regime_data

    def historical_analysis(self, universe_data=None):
        if universe_data is None:
            if hasattr(self, 'universe_data'):
                universe_data = self.universe_data
            else:
                self.logger.error('Non universe data available')

    def signal(self, universe_data, regime: int, signal_type: str, **kwargs):
        return NotImplementedError("Not implemented...")
        """
        if regime not in self.regime_dict:
            raise ValueError("regime not available...")

        params = {}
        for k, v in kwargs.items():
            params.update({k: v})
        # get the regime within a given periods
        since = params.get("since", self.regimes.index.min())
        # get data for the universe
        # TODO: add the computation of hit ratio / returns etc
        universe_data = universe_data.loc[universe_data.index >= since]
        # get the stats of the universe under each regime
        periods = params.get("periods", None)
        regime_data = self.get_regime_data(universe_data, regime)[
            self.regime_dict[regime]
        ]
        if periods is not None:
            regime_data = regime_data[-periods:]

        data_stats = []
        for data in regime_data:
            data_stats.append(data.mean(axis=0))
        return pd.DataFrame(
            pd.concat(data_stats, axis=1).mean(axis=1).rename(self.regime_dict[regime])
        )  # .sort_values(by=self.regime_dict[regime], ascending=False)
        """