'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:25:34
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
from utils.logger import *
from collections import defaultdict

from engine.base import EngineBase
from portfolio import Portfolio

class Manager(EngineBase):
    def __init__(self, id, type: str) -> None:
        super().__init__(id)
        self.logger = config_logger(self.__name__)
        self.portfolios = defaultdict()
        self.trades = defaultdict()
        self._aum = 0

    
    def add_portfolio(self, portfolio:Portfolio):        
        if portfolio.id not in self.portfolios:
            self.portfolios.update({portfolio.id: portfolio})
            self.logger.info('%s now managed by manager %s'%(portfolio.id, self.id))

    def remove_portfolio(self, portfolio:Portfolio):
        if portfolio.id in self.portfolios:
            self.portfolios.pop(portfolio.id)
        else:
            self.logger.warning('%s not managed by manager %s'%(portfolio.id, self.id))

    def get_signals(self, strategy, date: Union[str, dt, pd.Timestamp] = dt.today()):
        pass

    def get_orders(self, signals, portfolio):
        pass

    def place_orders(self, orders, trader):
        pass

    def get_positions(self, portfolio):
        return

    def update_portfolio(self, portfolio, orders, date: Union[str, dt, pd.Timestamp] = dt.today()):
        if portfolio.id not in self.portfolios:
            self.logger.error('Portfolio %s not managed by manager %s' %(portfolio.id, self.id))
        # 1. update positions of the portfolio
        self.portfolios.get(portfolio.id).update(orders, date)
        # 2. update account of the portfolio

    def rebalance(self, date: Union[str, dt, pd.Timestamp] = dt.today()):
        for pf in self.portfolios:
            if date in self.trading_calendar:
                signals = self.get_signals()
                orders = self.get_signals(signals)
                positions = self.get_positions(orders)
                self.update_portfolio(pf, positions, date)
            else:
                self.update_portfolio()

    def get_portfolio_performance(self, portfolio, date: Union[str, dt, pd.Timestamp] = dt.today()):
        return
        
    def summary(self, portfolio: Portfolio = None):
        # account info for all portfolios managed by the manager
        if portfolio is None:
            portfolios = self.portfolios
        else:
            portfolios = [portfolio]
        
        pm_summary = {}
        for pf in portfolios:
            pm_summary.update({pf.id: pf.account})
        
        return pm_summary

    def get_manager_performance(self):
        return

    def get_aum_source(self):
        return
    
    def get_aum(self):
        return
    
    def get_macro_exposure(self, portfolio):
        return
