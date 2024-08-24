'''
Author: J , jwsun1987@gmail.com
Date: 2023-01-31 19:10:10
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import os
from pathlib import Path
from datetime import datetime
from datetime import time as Time
from abc import ABC, abstractmethod
from typing import List, Dict

import yaml

from portfolio.account import Account
from portfolio.positions import Position


class Trader(ABC):
    def __init__(self) -> None:
        super().__init__()

    def config(self):
        return NotImplementedError("Close is not implemented yet")

    def close(self):
        return NotImplementedError("Close is not implemented yet")

    def get_order(self):
        return NotImplementedError("Get Order is not implemented yet")

    def place_order(self):
        return NotImplementedError("Close is not implemented yet")

    def cancel_order(self):
        return NotImplementedError("Close is not implemented yet")

    def get_broker_balance(self):
        """Get broker balance"""
        raise NotImplementedError("[get_broker_balance] has not been implemented")

    def get_broker_position(self):
        """Get broker position"""
        raise NotImplementedError("[get_broker_position] has not been implemented")

    def get_all_broker_positions(self):
        """Get all broker positions"""
        raise NotImplementedError("[get_all_broker_positions] has not been implemented")

    def get_all_orders(self):
        """Get all orders (sent by current algo)"""
        all_orders = []
        for orderid, order in self.orders.queue.items():
            order.orderid = orderid
            all_orders.append(order)
        return all_orders

    def get_all_deals(self):
        """Get all deals (sent by current algo and got executed)"""
        all_deals = []
        for dealid, deal in self.deals.queue.items():
            deal.dealid = dealid
            all_deals.append(deal)
        return all_deals

    @property
    def trade_mode(self):
        return self._trade_mode

    @trade_mode.setter
    def trade_mode(self, trade_mode):
        self._trade_mode = trade_mode

    def get_quote(self, security):
        """Get quote"""
        raise NotImplementedError("[get_quote] has not been implemented")

    def get_orderbook(self, security):
        """Get orderbook"""
        raise NotImplementedError("[get_orderbook] has not been implemented")

    def req_historical_bars(
        self,
        security,
        periods: int,
        freq: str,
        cur_datetime: datetime = None,
        daily_open_time: Time = None,
        daily_close_time: Time = None,
    ):
        """request historical bar data."""
        raise NotImplementedError("[req_historical_bars] has not been implemented")

    def subscribe(self):
        """Subscribe market data (quote and orderbook, and ohlcv)"""
        raise NotImplementedError("[subscribe] has not been implemented")

    def unsubscribe(self):
        """Unsubscribe market data (quote and orderbook, and ohlcv)"""
        raise NotImplementedError("[unsubscribe] has not been implemented")
