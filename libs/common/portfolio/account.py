'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:20:48
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from dataclasses import field, dataclass
from typing import Union, List, Dict
from .enum import Currency


@dataclass
class Account:
    base_currency: str = Currency.USD
    cash: float = 0.0
    cash_by_currency: Dict[str, float] = field(
        default_factory=lambda: {
            Currency.USD: 0,
            Currency.EUR: 0,
            Currency.GBP: 0,
            Currency.CHF: 0,
            Currency.JPY: 0,
            Currency.CNY: 0,
        }
    )
    available_cash: float = 0.0
    balance: float = 0.0
    balance_by_currency: Dict[str, float] = field(
        default_factory=lambda:{
            Currency.USD: 0,
            Currency.EUR: 0,
            Currency.GBP: 0,
            Currency.CHF: 0,
            Currency.JPY: 0,
            Currency.CNY: 0,
        }
    )
    max_power_short: float = None
    net_cash_power: float = None
    maintenance_margin: float = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
