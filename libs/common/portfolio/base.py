'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:27:14
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd

from datetime import datetime as dt
from collections import defaultdict
from typing import Dict, Union, List

from utils.logger import *
from .account import Account
from .position import Position
from .positions import Positions


class Portfolio:
    def __init__(
        self,
        positions: Union[Dict, Dict[str, Position]] = None,
        id: str = None,
        date: Union[str, pd.Timestamp, dt] = str(dt.today().date()),
        **kwargs,
    ) -> None:
        self.logger = config_logger(self.__class__.__name__)
        self._id = id if id is not None else str(1).zfill(3)
        self._date = date
        self._previous_trade = date        
        #self._positions = Positions(positions=positions, date=pd.to_datetime(date))
        self._account = Account()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    @property
    def account(self):
        return self._account

    @account.setter
    def account(self, account: Account):
        self._account = account

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value: Union[str, pd.Timestamp]):
        self._date = (
            pd.to_datetime(value, infer_datetime_format=True)
            if isinstance(value, str)
            else value
        )

    @property
    def previous_trade(self):
        return self._previous_trade

    @previous_trade.setter
    def previous_trade(self, date: Union[str, pd.Timestamp, dt]):
        self._previous_trade = date

    def reset(self):
        self._id = None
        self._account = None
        self._positions = None
        self._last_trade = None

    @property
    def positions(self):
        return self._positions.to_dict()

    @positions.setter
    def positions(
        self,
        positions: Union[Dict, Position],
        date: Union[str, pd.Timestamp, dt] = dt.today().date(),
    ):
        self._positions = (
            Positions(positions=positions, date=date)
            if isinstance(positions, Dict)
            else positions
        )

    def cash(self):
        return self.positions.cash

    def prices(self):
        return self.positions._prices

    def weights(self):
        return self.positions._weights

    def total_weight(self):
        return self.positions._total_weight

    def values(self):
        return self.positions._values

    def total_value(self):
        return self.positions._total_value

    def amounts(self):
        return self.positions._amounts

    def total_amount(self):
        return self.positions._total_amount

    def costs(self):
        return self.positions._costs

    def total_cost(self):
        return self.positions._total_cost

    def buy_prices(self):
        return self.positions._buy_prices

    def buy_dates(self):
        return self.positions._buy_dates

    def update(self, date, orders: List = None, **kwargs):
        # TODO: for backtesting only: download all price data all in once
        if pd.to_datetime(date) >= self.last_trade:
            if orders is not None:
                self.last_trade = date
                self.positions.update(date, orders, **kwargs)
            else:
                self.positions.update(date, **kwargs)
            # self.history.add(date, self.positions)
            # self.performances.update(self.history)
        # elif pd.to_datetime(date) == prev_date:
        #    self.history.update(date)

    """
    def export(self):  # instrument=None,attributes=None,
        if start is None:
            start = self.dates[0]
        if end is None:
            end = self.dates[-1]
        data_to_export = {}
        # if attributes is None:
        #    attributes = ['price', 'amount', 'value', 'cost']
        data_to_export.update({"price": self.price_history(start, end)})
        data_to_export.update({"amount": self.amount_history(start, end)})
        data_to_export.update({"valaue": self.value_history(start, end)})
        data_to_export.update({"weight": self.weight_history(start, end)})
        data_to_export.update({"cost": self.cost_history(start, end)})
        return data_to_export
    """
