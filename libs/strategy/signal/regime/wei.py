'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from datetime import datetime as dt
import pandas as pd
from pandas.tseries.offsets import BMonthEnd

from data.market import bloomberg as bbg
from strategy.signal.enum import Signals

from strategy.signal.regime import Regime


class WEI(Regime):
    def __init__(self, name: str = Signals.WEI, regimes: pd.DataFrame = None, regime_dict: dict = None, **kwargs) -> None:
        if regimes is None:
            regimes = (
                bbg.BloombergServer()
                .request(
                    security=[".WEILONG F Index", ".WEIREGIM F Index"],
                    start="1970-01-01",
                    end=dt.today(),
                    periodicitySelection="M",
                )
                .fillna(method="ffill")
                .groupby(pd.Grouper(freq="M"))
                .last()
            )
            regimes.index = [BMonthEnd().rollback(d) for d in regimes.index]
            regimes.columns = ["Score", "Regime"]
        if regime_dict is None:
            regime_dict = {
                0: "Contraction",
                1: "Slowdown",
                2: "Stationary",
                3: "Expansion",
                4: "Recovery",
            }
        super().__init__(name, regimes, regime_dict, **kwargs)