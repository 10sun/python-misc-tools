'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from typing import List, Dict, Union
from .enum import Assets

class Position:
    """Position

    an example of the position is:
    {
        <Insturment_ID>:{
            #'holding_period': days of the holding period,
            'amount': the amount of the instrument,
            'price': the close price of the instrument in the last trading day,
            'value': the current value of the instrument,
            'weight': the weight of the instrument of total position value,
            'date': the date of buying the instrument,
            'price': the price of buying the instrument,
            'cost': the cost of buying the instrument
            'exchange': the exchange
        }
    }
    """

    def __init__(
        self,
        identifier: str,
        name: str = None,
        asset_id: str = None,
        asset_class: str = None,
        weight: float = None,
        quantity: float = None,
        date: str = None,
        price: float = None,
        tags: Dict = None
    ) -> None:
        self._identifier = identifier
        self._name = name
        self._asset_id = asset_id
        self._asset_class = asset_class
        self._weight = weight
        self._quantity = quantity
        self._dates = [date]
        self._prices = [price]
        self._tags = tags

    def __eq__(self, other) -> bool:
        if not isinstance(self, other):
            return False
        for attr in ['asset_id', 'weight', 'quantity']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.asset_id) ^ hash(self.identifier)
    
    @property
    def identifier(self):
        return self._identifier
    
    @identifier.setter
    def identifier(self, value: str):
        self._identifier = value

    @property 
    def name(self):
        return self._name

    @name.setter
    def name(self, value:str):
        self._name = value

    @property
    def asset_id(self):
        return self._asset_id

    @asset_id.setter
    def asset_id(self, value:str):
        self._asset_id = value

    @property
    def asset_class(self):
        return self._asset_class

    @asset_class.setter
    def asset_class(self, value: Union[str, Assets]):
        self._asset_class = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = value

    @property
    def quantity(self):
        return self._quantity
    
    @quantity.setter
    def quantity(self, value: float):
        self._quantity = value

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value: Dict):
        self._tags = value
    
    @property
    def dates(self):
        return self._dates

    @dates.setter
    def dates(self, date:str):
        self.dates.append(date)

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, price:float):
        self._prices.append(price)

    def to_dict(self) -> dict:
        position_dict = dict(
            identifier=self.identifier,
            name=self.name,
            asset_id = self.asset_id,
            asset_class = self.asset_class,
            weight=self.weight,
            quantity=self.quantity,
            dates=self.dates,
            prices = self.prices
        )

        return {k:v for k,v in position_dict.items() if v is not None}