'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:23:00
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


class Engine:
    def __init__(self, data, strategy, trader, portfolio, analyzer) -> None:
        self.data = data
        self.strategy = strategy
        self.trader = trader
        self.portfolio = portfolio
        self.analyzer = analyzer

    def run(self):
        return NotImplementedError("Run not implemented")
        """
        for trading_day in self.trader.trading_sessions:
            orders = self.strategy.get_orders(trading_day)
            self.trader.place_orders(self.portfolio, orders)
        """
