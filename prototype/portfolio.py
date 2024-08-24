'''
Author: J , jwsun1987@gmail.com
Date: 2021-11-25 01:23:46
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

class portfolio:
    def __init__(self, instruments, weights, budgets, returns) -> None:
        self.universe_budgets = budgets
        self.universe_returns = returns
        self.instruments = instruments
        self.instrument_returns = returns.loc[:, instruments.Asset]
        self.weights = weights.T

        self.open_positions = {}

        self.initial_capital = 100
        self.position_budgets = weights.reset_index().merge(budgets, left_on='Asset', right_on='Asset').set_index('Asset')

        self._dates = []
        self._portfolio_values = []
        self._positions = []

        self.get_portfolio_bench_performance()
        #self.logger
    
    def get_portfolio_bench_performance(self):
        self.pf_bench_returns = self.weights.values * self.instrument_returns
        self.pf_bench_returns['Portfolio'] = self.pf_bench_returns.sum(axis=1)
        self.pf_bench_cumulative_returns = (1 + self.pf_bench_returns).cumprod()*self.initial_capital
