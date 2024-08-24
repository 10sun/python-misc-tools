'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


"""
def get_positions(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)
        else:
            self.logger.error("no record for %s" % str(date))
"""

def cash(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions.cash
    else:
        if date in self.history.dates:
            return self.history.positions.get(date).cash
        else:
            self.logger.error("no record for %s" % str(date))

def prices(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._prices
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._prices
        else:
            self.logger.error("no record for %s" % str(date))

def price_history(self, start=None, end=None):
    if start is None:
        start = self.history.dates[0]
    if end is None:
        end = self.history.dates[-1]
    history_dates = [d for d in self.history.dates if (d >= start) and (d <= end)]
    prices = [self.prices(date) for date in history_dates]
    return pd.concat(prices, axis=0)

def weights(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._weights
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._weights
        else:
            self.logger.error("no record for %s" % str(date))

def weight_history(self, start=None, end=None):
    if start is None:
        start = self.history.dates[0]
    if end is None:
        end = self.history.dates[-1]
    history_dates = [d for d in self.history.dates if (d >= start) and (d <= end)]
    weights = [self.weights(date) for date in history_dates]
    return pd.concat(weights, axis=0)

def total_weight(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._total_weight
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._total_weight
        else:
            self.logger.error("no record for %s" % str(date))

def values(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._values
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._values
        else:
            self.logger.error("no record for %s" % str(date))

def value_history(self, start=None, end=None):
    if start is None:
        start = self.history.dates[0]
    if end is None:
        end = self.history.dates[-1]
    history_dates = [d for d in self.history.dates if (d >= start) and (d <= end)]
    values = [self.values(date) for date in history_dates]
    return pd.concat(values, axis=0)

def total_value(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._total_value
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._total_value
        else:
            self.logger.error("no record for %s" % str(date))

def amounts(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._amounts
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._amounts
        else:
            self.logger.error("no record for %s" % str(date))

def amount_history(self, start=None, end=None):
    if start is None:
        start = self.history.dates[0]
    if end is None:
        end = self.history.dates[-1]
    history_dates = [d for d in self.history.dates if (d >= start) and (d <= end)]
    amounts = [self.amounts(date) for date in history_dates]
    return pd.concat(amounts, axis=0)

def total_amount(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._total_amount
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._total_amount
        else:
            self.logger.error("no record for %s" % str(date))

def costs(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._costs
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._costs
        else:
            self.logger.error("no record for %s" % str(date))

def cost_history(self, start=None, end=None):
    if start is None:
        start = self.history.dates[0]
    if end is None:
        end = self.history.dates[-1]
    history_dates = [d for d in self.history.dates if (d >= start) and (d <= end)]
    costs = [self.costs(date) for date in history_dates]
    return pd.concat(costs, axis=0)

def total_cost(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._total_cost
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._total_cost
        else:
            self.logger.error("no record for %s" % str(date))

def buy_prices(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._buy_prices
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._buy_prices
        else:
            self.logger.error("no record for %s" % str(date))

def buy_dates(self, date: Union[str, dt, pd.Timestamp] = None):
    if date is None:
        return self.positions._buy_dates
    else:
        if date in self.history.dates:
            return self.history.positions.get(date)._buy_dates
        else:
            self.logger.error("no record for %s" % str(date))

def update(self, date, orders: List = None, **kwargs):
    # TODO: for backtesting only: download all price data all in once
    prev_date = self.history.dates[-1]
    if pd.to_datetime(date) >= prev_date:
        if orders is not None:
            self.last_trade = date
        self.positions.update(date, orders, **kwargs)
        self.history.add(date, self.positions)
        # self.performances.update(self.history)
    # elif pd.to_datetime(date) == prev_date:
    #    self.history.update(date)

def export(self, start=None, end=None):  # instrument=None,attributes=None,
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
