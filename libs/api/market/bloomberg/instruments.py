'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-16 18:24:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd

from typing import Union, Dict, List

from api.market.instruments import Instruments

class BBGTickers(Instruments):
    def __init__(self, universe: Union[pd.DataFrame, pd.Series]) -> None:
        super().__init__(universe)
        self.prepare_tickers()

    def prepare_tickers(self, ticker_universe: Union[pd.DataFrame, dict] = None):
        """[summary]

        Args:
            ticker_universe (Union[dict, pd.DataFrame], optional): [description]. Defaults to None.
        """
        if ticker_universe is None:
            ticker_universe = self.universe
            
        if isinstance(ticker_universe, pd.DataFrame):
            if "Fields".casefold() not in [
                ticker.casefold() for ticker in list(ticker_universe.columns)
            ]:
                self.logger.error("no fields provided...")

            tickers = {}
            for ticker in ticker_universe.Ticker:
                tickers.update(
                    {
                        ticker: ticker_universe.loc[
                            ticker_universe.Ticker == ticker, :
                        ].to_dict("records")[0]
                    }
                )
            self.tickers = tickers
        else:
            self.logger.error("ticker universe has to be not in fields dictionary...")

        try:
            for ticker, ticker_params in self.tickers.items():
                if "Fields" not in ticker_params.keys():
                    self.logger.error("no fields provided...")

                ticker_fields = [
                    field.replace(" ", "")
                    for field in ticker_params.get("Fields", None).split(",")
                ]
                # if ticker_params.get('Fields', None) != '' else []

                self.tickers[ticker]["Fields"] = ticker_fields
                
        except Exception as e:
            self.logger.error(e)
