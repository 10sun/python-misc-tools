'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-17 22:56:38
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import numpy as np

from contextlib import contextmanager
from typing import Union, Dict, List, Optional

import blpapi
from blpapi import Element

from api.server import DataServerBase

from common.data import *
from common.dates import *

from .enum import *

class Server(DataServerBase):
    def __init__(self) -> None:
        super().__init__()
        try:
            self.connect()
        except Exception as e:
            self.logger.warning(e)

    def _name(self, element):
        """Extract name of the element

        Args:
            element (_type_): element

        Returns:
            _type_: the name of the element
        """
        return str(element.name())

    def _value(self, value):
        """return value as string

        Args:
            value (_type_): _description_

        Returns:
            _type_: _description_
        """
        return value.getValueAsString() if isinstance(value, Element) else value

    def _values(self, element, fn=None):
        """Extract values from elemenet, support two layers nested elements

        Args:
            fn (_type_, optional): _description_. Defaults to None.
        """
        return [
            {self._name(i): self._value(i) for i in v.elements()}
            if isinstance(v, Element)
            else {fn: v}
            for v in element.values()
        ]

    def _create_session(self, event_handler=None) -> "blpapi.Session":
        options = blpapi.SessionOptions()
        options.setServerHost("localhost")
        options.setServerPort(8194)
        session = blpapi.Session(options, eventHandler=event_handler)
        session.start()
        return session

    @contextmanager
    def connect(self) -> "blpapi.Session":
        self.session = self._create_session()
        try:
            yield self.session
        finally:
            self.session.stop()

    def disconnect(self):
        return super().disconnect()

    def create_request(
        self, session, service, request_type, overrides: dict = None, **kwargs
    ):
        if session is None:
            session = self.session

        params = {}
        for k, v in kwargs.items():
            params.update({k: v})

        session.openService(service)
        if params.get("recurrance_daily", False):
            request_name = "IntradayBarDateTimeChoiceRequest"
        else:
            request_name = REQUEST_TYPE.get(request_type)

        request = session.getService(service).createRequest(request_name)

        # historicalData
        if request_type == "History":
            start_date, end_date = start_end(
                start=params.get("start", None),
                end=params.get("end", None),
                n_days=params.get("n_days", 365),
                fmt=params.get("format", "%Y%m%d"),
            )
            # set the periodicitySelection
            if (
                params.get("periodicitySelection", "D").upper() not in FREQUENCY.keys()
                and params.get("periodicitySelection", "D").upper()
                not in FREQUENCY.values()
            ):
                raise ValueError(
                    "Invalid periodicity %s ..."
                    % params.get("periodicitySelection", "D")
                )
            elif params.get("periodicitySelection", "D").upper() in FREQUENCY.keys():
                frequency = FREQUENCY.get(
                    params.get("periodicitySelection", "D").upper()
                )
            elif params.get("periodicitySelection", "D").upper() in FREQUENCY.values():
                frequency = params.get("periodicitySelection", "D").upper()

            if "periodicitySelection" not in params.keys():
                params.update({"periodicitySelection": frequency})
            else:
                params["periodicitySelection"] = frequency

            #if "days" not in params.keys():
            #    params.update({"days": "T"})
                
            # set the startDate
            # if params.get("start", None) is None:
            if "start" in params.keys():
                params.pop("start")
            params.update({"startDate": start_date})
            # set the endDate
            if "end" in params.keys():
                params.pop("end")
            params.update({"endDate": end_date})
            # set the fields
            if params.get("fields", None) is None:
                params.update({"fields": ["PX_LAST"]})

            if "security" in params.keys():
                params.update({"securities": params["security"]})
                params.pop("security")
        elif request_type == "Reference":
            if "security" in params.keys():
                params.update({"securities": params["security"]})
                params.pop("security")
        # intraDayTickData
        elif request_type == "IntraDayTick":
            start_date, end_date = start_end(
                start=params.get("start", None),
                end=params.get("end", None),
                n_days=params.get("n_days", 3),
                tz=params.get("tz", "Europe/Zurich"),
                intraday=params.get("intraday", True),
            )
            # set the startDateTime
            if "start" in params.keys():
                params.pop("start")
            params.update({"startDateTime": start_date})
            if "end" in params.keys():
                params.pop("end")
            params.update({"endDateTime": end_date})

            # set the eventTypes
            if "eventTypes" not in params.keys():
                params.update({"eventTypes": "TRADE"})
            if "securities" in params.keys():
                params.update({"security": params["securities"]})
                params.pop("securities")
        elif request_type == "IntraDayBar":
            start_date, end_date = start_end(
                start=params.get("start", None),
                end=params.get("end", None),
                n_days=params.get("n_days", 3),
                tz=params.get("tz", "Europe/Zurich"),
                intraday=params.get("intraday", True),
            )
            if "start" in params.keys():
                params.pop("start")
            if "end" in params.keys():
                params.pop("end")
            if params.get("recurrance_daily", False):
                duration = (end_date - start_date).seconds
                element = request.getElement("dateTimeInfo").getElement(
                    "startDateDuration"
                )
                for d in pd.date_range(start_date, end_date):
                    element.getElement("rangeStartDateTimeList").appendValue(d)
                element.setElement("duration", duration)
            else:
                params.update({"startDateTime": start_date})
                params.update({"endDateTime": end_date})

            if "eventTypes" in params.keys():
                params.update({"eventType": params["eventTypes"]})
                params.pop("eventTypes")
            if "eventType" not in params.keys():
                params.update({"eventType": "TRADE"})
            if "securities" in params.keys():
                params.update({"security": params["securities"]})
                params.pop("securities")
            if "interval" not in params.keys():
                params.update({"interval": 10})


        for k, v in params.items():
            if k in ["securities", "fields"]:
                for s in to_list(v):
                    request.getElement(k).appendValue(s)
            elif k == "eventTypes":
                for s in [s.upper() for s in to_list(v)]:
                    request.getElement(k).appendValue(s)
            else:
                request.set(k, v)

        if overrides is not None:
            ovrd = request.getElement("overrides")
            for k, v in overrides.items():
                ovrd_i = ovrd.appendElement()
                ovrd_i.setElement("fieldId", k)
                ovrd_i.setElement("value", v)
        #print(request)
        return request

    def _extract_history_data(self, session, tickers, squeeze_from):
        if session is None:
            session = self.session

        res = {}
        while True:
            event = session.nextEvent(2000)
            if event.eventType() in [
                blpapi.Event.RESPONSE,
                blpapi.Event.PARTIAL_RESPONSE,
            ]:
                for msg in event:
                    # each message contains one 'securityData'
                    if msg.hasElement("responseError"):
                        self.logger.error(
                            msg.getElement("responseError")
                            .getElement("message")
                            .getValueAsString()
                        )
                        continue

                    security_data = msg.getElement("securityData")
                    security = security_data.getElement("security").getValue()
                    # each security Data contains multiple fieldData
                    df = pd.DataFrame(
                        self._values(security_data.getElement("fieldData"))
                    )
                    res[security] = df if df.empty else df.set_index("date")
                if event.eventType() == blpapi.Event.RESPONSE:
                    break

        res_sorted = {k: res[k] for k in tickers if k in res.keys()}
        res = pd.concat(res_sorted.values(), keys=res_sorted.keys(), axis=1)
        if res.empty:
            return None
        res.columns = res.columns.set_names(["Ticker", "Fields"])
        if "+" in res.index[-1]:
            res.index = pd.to_datetime(res.index, format="%Y-%m-%d+%H:%M")
        else:
            res.index = pd.to_datetime(res.index)
        res = res.astype(
            {
                c: HIST_DTYPES[c[-1]] if c[-1] in HIST_DTYPES else "float"
                for c in res.columns
            },
            errors="ignore",
        )
        res = res.sort_index()

        # remove last row
        last = res.iloc[-1].where(
            res.columns.get_level_values(-1) != "PX_VOLUME",
            res.iloc[-1].replace(0, np.nan),
        )
        if last.isna().all():
            res = res.iloc[:-1].copy()
        # res.columns = squeeze_index(res.columns, squeeze_from)
        return res

    def _extract_reference_data(self, session, tickers, fields, squeeze: bool = True):
        if session is None:
            session = self.session

        res = {}
        while True:
            event = session.nextEvent(1000)
            if event.eventType() in [
                blpapi.Event.RESPONSE,
                blpapi.Event.PARTIAL_RESPONSE,
            ]:
                for msg in event:
                    # each message contains multiple securityData
                    for security_data in msg.getElement("securityData").values():
                        # each securityData contains multiple fieldData
                        for fn, f in [
                            (self._name(i), i)
                            for i in security_data.getElement("fieldData").elements()
                        ]:
                            if fn not in res:
                                res[fn] = {}
                            security = security_data.getElement("security").getValue()
                            res[fn][security] = pd.DataFrame(self._values(f, fn))
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        res_final = {}
        for i, d in res.items():
            d = pd.concat(d)
            d = d.set_axis(d.index.droplevel(1))
            res_final[i] = d

        if squeeze and len(set(r.shape[0] for r in res_final.values())) == 1:
            res_final = pd.concat(res_final.values(), axis=1)
            return res_final.loc[
                [s for s in tickers if s in res_final.index],
                #[f for f in fields if f in res_final.columns],
            ]
        return res_final

    def _extract_ticker_derived_data(self, session, data_type, sub_date_type, tz):
        if session is None:
            session = self.session
        res_all = []
        while True:
            event = session.nextEvent(1000)
            if event.eventType() in [
                blpapi.Event.RESPONSE,
                blpapi.Event.PARTIAL_RESPONSE,
            ]:
                for msg in event:
                    if msg.hasElement("responseError"):
                        self.logger.error(
                            msg.getElement("responseError")
                            .getElement("message")
                            .getValueAsString()
                        )
                    else:
                        res = pd.DataFrame(
                            self._values(
                                msg.getElement(data_type).getElement(sub_date_type)
                            )
                        )
                        res = res if res.empty else res.set_index("time")
                        # noinspection PyTypeChecker
                        date = pd.to_datetime(res.index)
                        res.index = (
                            (date + tz_diff(date=date, to_tz=tz))
                            if tz != "UTC"
                            else date
                        )
                        res_all.append(res)
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        return pd.concat(res_all) if res_all else None

    def request(
        self,
        service: str = REF_DATA,
        request_type: str = "History",
        overrides=None,
        **kwargs,
    ):
        """request data from bloomberg server

        Args:
            tickers (_type_): _description_
            fields (Union[str, List[str]], optional): _description_. Defaults to "PX_LAST".
            request_type (str, optional): _description_. Defaults to "HistoricalDataRequest".
            event_type (Union[str, List[str]], optional): _description_. Defaults to "TRADE".
            overrides (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        params = {}
        for k, v in kwargs.items():
            params.update({k: v})

        tickers = unique(to_list(params.get("security")))
        if params.get("fields", None) is not None:
            fields = unique(to_list(params.get('fields', None))) 

        with self.connect() as session:
            request = self.create_request(
                session=session,
                service=service,
                request_type=request_type,
                overrides=overrides,
                **kwargs,
            )

            session.sendRequest(request)

            if request_type == "History":
                res = self._extract_history_data(
                    session, tickers, params.get("squeeze_from", "outer")
                )
            elif request_type == "Reference":
                res = self._extract_reference_data(session, tickers, fields)
            elif request_type == "IntraDayTick":
                res = self._extract_ticker_derived_data(
                    session, "tickData", "tickData", params.get("tz", "Europe/Zurich")
                )
                res = (
                    res.astype({k: v for k, v in TICK_DTYPES.items() if k in res})
                    if res is not None
                    else None
                )
            elif request_type == "IntraDayBar":
                res = self._extract_ticker_derived_data(
                    session, "barData", "barTickData", params.get("tz", "Europe/Zurich")
                )
                if res is None or res.empty:
                    return None
                else:
                    res = res[["open", "high", "low", "close", "volume"]].astype(
                        "float"
                    )
            return res
