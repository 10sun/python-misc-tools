'''
Author: J , jwsun1987@gmail.com
Date: 2022-12-07 01:01:06
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''
from typing import Dict, Union, List
import pandas as pd
import copy
from datetime import datetime as dt

from .position import Position
from utils.logger import config_logger

class Positions:
    def __init__(
        self,
        positions: Dict[str, Position] = None,
        date: Union[str, pd.Timestamp, dt] = dt.today().date(),
    ) -> None:
        """initialize the position by cash and positions (a dictionary consisting of instrument id and related info)

        Args:
            cash (float, optional): _description_. Defaults to 0.0.
            positions (Dict[str, Dict[str, float]], optional): _description_. Defaults to {}.
        """
        self.logger = config_logger(self.__class__.__name__)
        self._date = date
        if positions is not None:
            self._positions = {k:Position(v) for k,v in positions.items()}

    def _update(
        self, action: str, instrument, amount: Union[float, int], price: float = None
    ):
        if action == "BUY":
            if instrument not in self.instruments:
                self.instruments.update(
                    {
                        instrument: {
                            "amount": amount,
                            "buy_price": price,
                            "buy_date": self.date,
                        }
                    }
                )
            else:
                try:
                    prev_cost = self.cost(instrument)
                    prev_amount = self.amount(instrument)
                    new_buy_price = (prev_cost + amount * price) / (
                        prev_amount + amount
                    )
                    self.instruments[instrument]["amount"] += amount
                    self.instruments[instrument]["buy_price"] = new_buy_price
                except Exception as e:
                    self.logger.warning(instrument)
                    return
        elif action == "SELL":
            if instrument not in self.instruments:
                self.instruments.update({instrument: {"amount": -amount}})
            else:
                if np.isclose(self.instruments[instrument]["amount"], amount):
                    self._delete(instrument)
                else:
                    self.instruments[instrument]["amount"] -= amount

    def _delete(self, instrument):
        if instrument in self.positions:
            del self.positions[instrument]
        else:
            raise ValueError("%s not in the portfolio" % instrument)

    def _buy(
        self,
        instrument,
        trade_value: Union[int, float],
        cost: Union[int, float],
        trade_price: Union[int, float],
    ):
        trade_amount = trade_value / trade_price
        self._update("BUY", instrument, trade_amount, trade_price)

        self.cash -= trade_value + cost

    def _sell(
        self,
        instrument,
        trade_value: Union[int, float],
        cost: Union[int, float],
        trade_price: Union[int, float],
    ):
        trade_amount = trade_value / trade_price
        # if instrument not in self.instruments:
        #    self.logger.error("%s not in the portfolio" % instrument)
        # else:
        self._update("SELL", instrument, trade_amount, trade_price)

        self.cash += trade_value - cost
        # receivable = trade_value - cost
        # if self._settle_type == self.ST_CASH:
        #    self.instruments["receivable"] = receivable
        # elif self._settle_type == self.ST_NO:
        #    self.instruments["cash"] += receivable
        # else:
        #    self.logger.error("%s settlement is not supported..." % self._settle_type)
    """
    def _update_instruments(self, orders=None, prices=None, last_dates=None):
        if orders is not None:
            for order in orders:
                if order.action == 2:
                    # TODO: buy without a price instrumction, then use the market price
                    self._buy(order.instrument, order.value, order.cost, order.price)
                elif order.action == -2:
                    self._sell(order.instrument, order.value, order.cost, order.price)

        positions = copy.deepcopy(self.instruments)

        # TODO: check if self.date is a business day
        if prices is not None:
            if self.date in prices.index:
                ins_price = prices.loc[prices.index == self.date].dropna(axis=1)
            else:
                ins_price = (
                    pd.DataFrame(prices.loc[prices.index <= self.date].iloc[-1])
                    .dropna(axis=0)
                    .T
                )
        else:
            ins_price = bbg.BloombergServer().request(
                security=list(set(positions.keys())), start=self.date, end=self.date
            )
        
        #if last_dates is not None:
        #    ins_last_dates = last_dates
        #else:
        #    ins_last_dates = (
        #        bbg.BloombergServer()
        #        .request(
        #            request_type="Reference",
        #            security=list(set(positions.keys())),
        #            fields="LAST_UPDATE",
        #        )
        #    )
        #    display(ins_last_dates)

        if isinstance(ins_price, (pd.DataFrame, pd.Series)):
            if pd.DataFrame(ins_price).columns.nlevels > 1:
                ins_price.columns = ins_price.columns.get_level_values(0)
        else:
            ins_price = pd.DataFrame()

        for ins, pos in positions.items():
            # if ins not in ins_price.columns and ins_last_dates[ins].values[0] < pd.to_datetime(self.date):
            #    self._instruments.pop(ins)
            #    continue
            # elif ins in ins_price.columns:
            for k, v in copy.copy(pos).items():
                if k.islower():
                    continue
                self._instruments[ins][k.casefold()] = v
                self._instruments[ins].pop(k)

            if ins in ins_price.columns:
                self._instruments[ins]["price"] = ins_price[ins].values[0]

            if orders is not None:
                for k, v in pos.items():
                    if k.casefold() in [
                        "amount",
                        "buy_date",
                        "buy_price",
                    ]:
                        self._instruments[ins][k] = v

        if not self.amount():
            if self.weight():
                self.init_amount()
            else:
                return

        if orders is not None or not self.cost():
            self.update_cost()
        self.update_value()
        self.update_weight()
    """

    def has(self, instrument):
        return instrument in self.instruments

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        self._cash = value
        self.weight()
    
    @property
    def receivable(self):
        if hasattr(self, "_receivable"):
            return self._receivable
        else:
            return 0

    @receivable.setter
    def receivable(self, value):
        self._receivable = value
        self.weight()

    """
    # @property
    # def current_value(self):
    #    return self.current_value

    # @current_value.setter
    # def current_value(self, value):
    #    self.current_value = value
    """

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value: Union[str, pd.Timestamp, dt]):
        self._date = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value: Dict[str, Position]):
        self._positions = value

        """
        self._instruments = copy.deepcopy(positions)
        print("set portfolio instruments")
        for instrument in self._instruments:
            print(instrument)
            if isinstance(self.instruments[instrument], int):
                if "amount" not in self.instruments:
                    continue
                self.instruments[instrument] = {"amount": self.instruments[instrument]}
            else:
                self.update()
        """
    #                self._update_instruments(instrument, positions)
    #        self.weight()

    def init_positions(self):
        for ins, pos in self.instruments.items():
            if "weight" in pos and "price" in pos:
                self.instruments[ins]["amount"] = (
                    pos["weight"] * self._initial_value / pos["price"]
                )

    def update_value(self):
        for ins, pos in self.instruments.items():
            if "price" not in pos:
                continue
            self.instruments[ins]["value"] = pos["amount"] * pos["price"]

    def update_weight(self):
        curr_value = self.total_value().values[0][0] + self.cash
        for ins, pos in self.instruments.items():
            if "value" not in pos:
                continue
            self.instruments[ins]["weight"] = pos["value"] / curr_value

    def update_cost(self):
        # TODO: if there is a cost related
        for ins, pos in self.instruments.items():
            if "price" not in pos:
                continue

            if "buy_price" not in pos:
                self.instruments[ins]["buy_price"] = pos["price"]
                self.instruments[ins]["buy_date"] = self.date

            self.instruments[ins]["cost"] = (
                self.instruments[ins]["buy_price"] * self.instruments[ins]["amount"]
            )

    def update(self, date=None, orders: List = None, **kwargs):
        if date is not None:
            self.date = date

        params = {}
        for k, v in kwargs.items():
            params.update({k: v})

        self._update_instruments(
            orders,
            prices=params.get("prices", None),
            last_dates=params.get("last_dates", None),
        )

        self._prices = (
            pd.DataFrame.from_dict(self.price(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._amounts = (
            pd.DataFrame.from_dict(self.amount(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._values = (
            pd.DataFrame.from_dict(self.value(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._weights = (
            pd.DataFrame.from_dict(self.weight(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._costs = (
            pd.DataFrame.from_dict(self.cost(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._buy_dates = (
            pd.DataFrame.from_dict(self.buy_date(), orient="index")
            .rename(columns={0: self.date})
            .T
        )
        self._buy_prices = (
            pd.DataFrame.from_dict(self.buy_price(), orient="index")
            .rename(columns={0: self.date})
            .T
        )

        self._total_amount = self._amounts.sum(axis=1)
        self._total_value = self._values.sum(axis=1)
        self._total_weight = self._weights.sum(axis=1)
        self._total_cost = self._costs.sum(axis=1)
        # return self


    def quantity(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = list(self.positions.keys())
        
        amounts = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            amounts.update({self.positions[ins].asset_id: self.positions[ins].quantity})
        return amounts

    def total_quantity(self):
        return pd.DataFrame(
            pd.DataFrame.from_dict(self.quantity(), orient="index")
            .rename(columns={0: self.date})
            .T.sum(axis=1)
        )

    def price(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = list(self.positions.keys())
        prices = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            prices.update({self.positions[ins].asset_id: self.positions[ins].price})
        return prices
    
    def value(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = self.positions.keys()

        values = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            values.update({self.positions[ins].asset_id: self.positions[ins].quantity*self.positions[ins].price})
        return values

    def total_value(self):
        return pd.DataFrame(
            pd.DataFrame.from_dict(self.value(), orient="index")
            .rename(columns={0: self.date})
            .T.sum(axis=1)
        )

    def weight(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = self.positions.keys()

        weights = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            weights.update({self.positions[ins].asset_id: self.positions[ins].weight})
        return weights

    def total_weight(self):
        return pd.DataFrame(
            pd.DataFrame.from_dict(self.weight(), orient="index")
            .rename(columns={0: self.date})
            .T.sum(axis=1)
        )

    def cost(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = self.positions.keys()
        costs = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            costs.update({self.positions[ins].asset_id: self.positions[ins].cost})
        return costs

    def total_cost(self):
        return pd.DataFrame(
            pd.DataFrame.from_dict(self.cost(), orient="index")
            .rename(columns={0: self.date})
            .T.sum(axis=1)
        )

    def buy_date(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = self.positions.keys()
        buy_dates = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            buy_dates.update({self.positions[ins].asset_id: self.positions[ins].date})
        return buy_dates

    def buy_price(self, instruments: Union[List[str], str] = None):
        if instruments is None:
            instruments = self.positions.keys()

        buy_prices = {}
        for ins in instruments:
            if ins not in self.positions:
                continue
            buy_prices.update({self.positions[ins].asset_id: self.positions[ins].price})
        return buy_prices

    def holding_period(self, instrument):
        return NotImplementedError(
            "not implemented yet"
        )  # self.instruments[instrument]["holding_period"]

    def settle(self):
        self.value()
        return NotImplementedError("not implemented yet")

    def all_instruments(self):
        return pd.DataFrame(list(set(self.positions.keys())))

    def to_table(self) -> pd.DataFrame:
        return

    @classmethod
    def from_dict(cls, positions: Dict, date: Union[str, pd.Timestamp, dt], **kwargs):
        return

    @classmethod
    def from_dict(cls, positions: pd.DataFrame, date: Union[str, pd.Timestamp, dt], **kwargs):
        return
    
    @classmethod
    def from_list(cls, positions: List[str], date: Union[str, pd.Timestamp, dt], **kwargs):
        return