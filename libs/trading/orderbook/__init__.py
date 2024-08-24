'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from enum import IntEnum
import numpy as np
import pandas as pd
from typing import Dict, List, Union


class OrderType(IntEnum):
    SELL_MARKET = -1
    SELL = -2
    BUY_MARKET = 1
    BUY = 2


class Order:
    """the trade order made by the system/strategy
    An example:
    'Instrument':
    {
        'action': 'BUY'/'SELL'/'BUYN'/'SELLN',
        'amount': 200,
        'price': 100 | market price,
        'start': '2022-03-07 00:00:00',
        'end': '2022-03-15 00:00:00',
        'exchange': 'US'
    }
    """

    def __init__(
        self,
        instrument: str,
        action: Union[str, int],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        value: Union[int, float],
        price: float,
        cost: float == 0,
    ) -> None:
        self.instrument = instrument
        if isinstance(action, str):
            self.action = self.translate_action(action)
        else:
            self.action = action
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.value = value
        self.price = price
        self.amount = (self.value / self.price) * np.sign(self.action)
        self.cost = cost

    def translate_action(self, action: str):
        if action.casefold() == "sell":
            return OrderType.SELL
        elif action.casefold() == "sell_market":
            return OrderType.SELL_MARKET
        elif action.casefold() == "buy":
            return OrderType.BUY
        elif action.casefold() == "buy_market":
            return OrderType.BUY_MARKET
        else:
            raise ValueError("%s not supported..." % action)

    ## TODO: if the order is not an instaneous order, then we need to compute when and the amount of the order
    def deal_price(self, price, exchange):
        return

    def type(self):
        return NotImplementedError

    def status(self):
        return NotImplementedError