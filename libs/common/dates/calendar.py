'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-18 00:57:40
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from enum import Enum, EnumMeta

import datetime as dt
import numpy as np
from typing import Tuple, Union, Optional

from cachetools import TTLCache

_calendar_cache = TTLCache(maxsize=128, ttl=600)


class TradingCalendar:
    """A class consists of trading/business calendars

    Returns:
        _type_: _description_
    """
    DATE_LOW_LIMIT = dt.date(1900, 1, 1)
    DATE_HIGH_LIMIT = dt.date(2052, 12, 31)
    DEFAULT_WEEK_MASK = "1111100"

    def __init__(self, calendars) -> None:
        if isinstance(calendars, ()):
            calendars = (calendars,)
        if calendars is None:
            calendars = ()
        self.__calendars = calendars
        self.__business_day_calendars = {}

    # def get(calendars: Union[str, Tuple]):
    #    return
    @staticmethod
    def reset():
        _calendar_cache.clear()

    @property
    def holidays(self) -> Tuple[dt.date]:
        holidays = []
        return NotImplementedError("Not implemented yet...")

    @staticmethod
    def get(calendars):
        return TradingCalendar(calendars)

    def calendars(self) -> Tuple:
        return self.__calendars

    @staticmethod
    def is_currency(currency) -> bool:
        return NotImplementedError("Not implemented yet...")

    def business_day_calendar(self, week_mask: Optional[str]):
        return self.__business_day_calendars.setdefault(
            week_mask,
            np.busdaycalendar(
                weekmask=week_mask or self.DEFAULT_WEEK_MASK,
                holidays=tuple([np.datetime64(d.isoformat()) for d in self.holidays]),
            ),
        )
