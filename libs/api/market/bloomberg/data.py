'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-16 18:24:18
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
from typing import Union, List, Dict, Optional

from data.base import DataContainer
from .instruments import BBGTickers
from .enum import *

#TODO: custimized bloomber data class
class BBGData(DataContainer):
    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        data_type: Optional[str],
        frequency: Optional[str],
        tickers: Union[Dict, pd.DataFrame] = None,
    ) -> None:
        super().__init__(data, data_type)
        self.frequency = frequency
        if tickers is not None:
            if isinstance(tickers, pd.DataFrame):
                self.tickers = BBGTickers(tickers).tickers
            elif isinstance(tickers, dict):
                self.tickers = tickers
        if data is not None:
            self._split_raw_data()

    def _split_raw_data(self, data: pd.DataFrame = None, fields: str = None):
        if data is None:
            data = self.data

        if fields is None:
            fields_lvl = list(data.columns.names).index("Field")

            fields = list(
                set(list(data.columns.get_level_values(fields_lvl).astype(str)))
            )

        if all("" == s or s.isspace() for s in fields):
            return

        for field in fields:
            field_df = data.loc[:, (slice(None), field, slice(None))]
            field_df.columns = field_df.columns.get_level_values("Instrument").astype(
                str
            )
            field_df.columns = [
                self.tickers[ticker.split(":")[1]].get(
                    "Description", ticker.split(":")[1]
                )
                if ":" in ticker
                else self.tickers[ticker.split(":")[0]].get(
                    "Description", ticker.split(":")[0]
                )
                for ticker in field_df.columns
            ]
            self._assign_data_attribute(field_df, field)

        if hasattr(self, "price") and hasattr(self, "EPS"):
            self.set_trailing_pe()

        if (
            hasattr(self, "fwd_EPS_1M")
            and hasattr(self, "fwd_EPS_3M")
            and hasattr(self, "fwd_EPS_u")
            and hasattr(self, "fwd_EPS_d")
        ):
            self.set_earnings_momentum()

        if hasattr(self, "fwd_EPS_growth") and hasattr(self, "fwd_PE"):
            self.set_fwd_peg()

    def _assign_data_attribute(
        self,
        field_df: Union[pd.DataFrame, pd.Series],
        field: str,
    ):
        if FIELD_DESCRIPTION.get(field, "None") in DESCRIPTION_ATTRIBUTE.keys():
            attr_name = DESCRIPTION_ATTRIBUTE[FIELD_DESCRIPTION.get(field, "None")]
            if hasattr(self, attr_name):
                new_df = pd.concat([getattr(self, attr_name), field_df], axis=1)
            else:
                new_df = field_df
            setattr(self, attr_name, new_df.astype(float))
        else:
            self.logger.info(field + " not in fields dictionary, passed...")

    def get_data(self, attr: str):
        if attr.casefold() in [
            desc.casefold() for desc in DESCRIPTION_ATTRIBUTE.keys()
        ]:
            for desc in DESCRIPTION_ATTRIBUTE.keys():
                if desc.casefold() == attr.casefold():
                    attr_desc = desc
            return getattr(self, DESCRIPTION_ATTRIBUTE[attr_desc])
        else:
            self.logger.error(attr + " not in fields dictionary, passed...")

    def get_ticker_attr(data: dict, value, key):
        for v in data.values():
            if value in v.values():
                return v.get(key, None)

    def export(
        self, attributes: Union[list, str] = None, params: dict = {"type": pd.DataFrame}
    ):
        data_to_export = {}
        if attributes is not None:
            if isinstance(attributes, str):
                attributes = [attributes]
            for attr in attributes:
                data_to_export.update({attr: getattr(self, attr)})
        else:
            types = params.get("type", pd.DataFrame)
            for attr, data in self.__dict__.items():
                if isinstance(data, types):
                    data_to_export.update({attr: data})

        if params.get("path", None) is None:
            return data_to_export
        else:
            self.logging.error("not implemented yet")
