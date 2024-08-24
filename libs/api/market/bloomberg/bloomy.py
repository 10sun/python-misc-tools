'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-17 22:05:24
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import blpapi
import numpy as np
import pandas as pd

from blpapi import Element
from contextlib import contextmanager
from datetime import datetime
from warnings import warn_explicit
from typing import List, Union, Dict, Optional
from datetime import timedelta

all__ = [
    'get_historical_data', 'get_bar_data', 'get_tick_data', 'get_ref_data'
]

HIST_DTYPES = {}
TICK_DTYPES = {'value': 'float', 'size': 'int', 'tradeTime': 'datetime64[ns]'}
REF_SERVICE = '//blp/refdata'
MKT_SERVICE = '//blp/mktdata'


def tz_swap(date, from_tz='UTC', to_tz='Europe/Zurich'):
    """[summary]

    Args:
        date ([type]): [description]
        from_tz (str, optional): [description]. Defaults to 'UTC'.
        to_tz (str, optional): [description]. Defaults to 'Europe/Zurich'.
    """
    if from_tz == to_tz:
        return pd.Timestamp(date)
    return pd.Timestamp(date).tz_localize(from_tz).tz_convert(to_tz).replace(
        tzinfo=None)


def tz_diff(date=None,
            from_tz='UTC',
            to_tz='Europe/Zurich') -> 'Union[Timedelta, TimedeltaIndex]':
    if isinstance(date, pd.DatetimeIndex):
        return date.tz_localize(from_tz) - date.tz_localize(to_tz).tz_convert(
            from_tz)
    now = pd.Timestamp(date) if date else pd.Timestamp.now()
    return now - now.tz_localize(to_tz).tz_convert(from_tz).replace(
        tzinfo=None)


def squeeze_index(index: 'pd.Index', squeeze_from):
    if isinstance(index, pd.MultiIndex) and squeeze_from:
        for level in ([1, 0] if squeeze_from == 'inner' else [0, 1]):
            if len(unique(index.get_level_values(level))) == 1:
                return index.droplevel(level)
    return index


def _create_session(event_handler=None) -> 'blpapi.Session':
    options = blpapi.SessionOptions()
    options.setServerHost('localhost')
    options.setServerPort(8194)
    session = blpapi.Session(options, eventHandler=event_handler)
    session.start()
    return session


@contextmanager
def create_session() -> 'blpapi.Session':
    session = _create_session()
    try:
        yield session
    finally:
        session.stop()


def create_request(session, service, request, overrides=None, **kwargs):
    session.openService(service)
    request = session.getService(service).createRequest(request)
    for key, vals in kwargs.items():
        if key in ['securities', 'fields', 'eventTypes']:
            for s in ([vals] if isinstance(vals, str) else vals):
                request.getElement(key).appendValue(s)
        else:
            request.set(key, vals)
    if overrides:
        ovrd = request.getElement("overrides")
        for key, value in overrides.items():
            ovrd_i = ovrd.appendElement()
            ovrd_i.setElement("fieldId", key)
            ovrd_i.setElement("value", value)
    return request


def _name(element):
    """ Extract name of the element """
    return str(element.name())


def _value(val):
    return val.getValueAsString() if isinstance(val, Element) else val


def _values(element, fn=None):
    """ Extract values from element, support two layers nested elements """
    return [{_name(i): _value(i)
             for i in v.elements()} if isinstance(v, Element) else {
                 fn: v
             } for v in element.values()]


def _start_end(start, end, n_days, fmt=None, tz=None, intraday=False):
    start = pd.Timestamp(start) if start else (pd.Timestamp.now().normalize() -
                                               pd.offsets.Day(n_days))
    end = pd.Timestamp(end) if end else pd.Timestamp.now()
    if end.normalize() == end and intraday:
        end = end + pd.Timedelta('23:59:59')
    start = tz_swap(start, tz, 'UTC') if tz else start
    end = tz_swap(end, tz, 'UTC') if tz else end
    return (start.strftime(fmt), end.strftime(fmt)) if fmt else (start, end)


def _construct_ticker(res, tks, field='FUT_CUR_GEN_TICKER'):
    if not isinstance(tks, str):
        return (res[tks[0]][field].str.replace(' ', '_') +
                res[tks[1]][field].str.replace(' ', '_')).to_frame()
    return res[tks]


# noinspection PyDefaultArgument
def get_historical_data(securities: Union[str, List[str]],
                        fields: Union[str, List[str]] = 'PX_LAST',
                        start: Union[str, datetime] = None,
                        end: Union[str, datetime] = None,
                        periodicity='DAILY',
                        squeeze_from: 'Optional[str]' = 'outer',
                        n_days=365,
                        overrides=None,
                        **kwargs):
    """

    Parameters
    ----------
    securities :
    fields :
    start :
    end :
    periodicity :
    squeeze_from :
    n_days :
    overrides :
    kwargs :

    Returns
    -------

    """
    securities = unique(
        [securities] if isinstance(securities, str) else securities)
    fields = fields if isinstance(fields, str) else fields
    start, end = _start_end(start, end, n_days, fmt='%Y%m%d')
    periodicity = periodicity.upper()
    if periodicity not in [
            'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY',
            'YEARLY'
    ]:
        raise ValueError('Invlid periodicity')
    with create_session() as session:
        request = create_request(session=session,
                                 service=REF_SERVICE,
                                 request='HistoricalDataRequest',
                                 securities=securities,
                                 fields=fields,
                                 periodicitySelection=periodicity,
                                 startDate=start,
                                 endDate=end,
                                 overrides=overrides,
                                 **kwargs)
        session.sendRequest(request)
        res = {}
        while True:
            event = session.nextEvent(2000)
            if event.eventType() in [
                    blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE
            ]:
                for msg in event:
                    # each message contains one 'securityData'
                    security_data = msg.getElement('securityData')
                    security = security_data.getElement('security').getValue()
                    # each securityData contains multiple fieldData
                    df = pd.DataFrame(
                        _values(security_data.getElement('fieldData')))
                    res[security] = df if df.empty else df.set_index('date')
                if event.eventType() == blpapi.Event.RESPONSE:
                    break

        res_sorted = {k: res[k] for k in securities if k in res}
        res = pd.concat(res_sorted.values(), keys=res_sorted.keys(), axis=1)
        if res.empty:
            return None
        res.columns = res.columns.set_names(['Security', 'Field'])
        # noinspection PyTypeChecker
        res.index = pd.to_datetime(res.index)
        res = res.astype({
            c: HIST_DTYPES[c[-1]]
            for c in res.columns if c[-1] in HIST_DTYPES
        })
        res = res.astype(
            {c: 'float'
             for c in res.columns if c[-1] not in HIST_DTYPES},
            errors='ignore')
        res = res.sort_index()
        # remove last row
        last = res.iloc[-1].where(
            res.columns.get_level_values(-1) != 'PX_VOLUME',
            res.iloc[-1].replace(0, np.nan))
        if last.isna().all():
            res = res.iloc[:-1].copy()
        res.columns = squeeze_index(res.columns, squeeze_from)
        return res


# noinspection PyDefaultArgument
def get_ref_data(securities,
                 fields,
                 overrides=None,
                 squeeze=False,
                 **kwargs) -> 'Union[Dict[str, pd.DataFrame], pd.DataFrame]':
    """
    Get Bloomberg reference data

    Parameters
    ----------
    securities :
        List of Bloomberg ticker
    fields :
        List of Bloomberg fields
    overrides :
        Overrides
    squeeze :
        Return a data frame when length of fields is 1
    kwargs :
        Other key-value parameters...
    Returns
    -------

    """
    fields = [fields] if isinstance(fields, str) else fields
    with create_session() as session:
        request = create_request(session=session,
                                 service=REF_SERVICE,
                                 request='ReferenceDataRequest',
                                 securities=securities,
                                 fields=fields,
                                 overrides=overrides,
                                 **kwargs)
        session.sendRequest(request)
        res = {}
        while True:
            event = session.nextEvent(1000)
            if event.eventType() in [
                    blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE
            ]:
                for msg in event:
                    # each message contains multiple securityData
                    for security_data in msg.getElement(
                            'securityData').values():
                        # each securityData contains multiple fieldData
                        for fn, f in [(_name(i), i)
                                      for i in security_data.getElement(
                                          'fieldData').elements()]:
                            if fn not in res:
                                res[fn] = {}
                            security = security_data.getElement(
                                'security').getValue()
                            res[fn][security] = pd.DataFrame(_values(f, fn))
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        res_final = {}
        for i, d in res.items():
            d = pd.concat(d)
            d = d.set_axis(d.index.droplevel(1))
            res_final[i] = d
        if squeeze and len(set(r.shape[0] for r in res_final.values())) == 1:
            res_final = pd.concat(res_final.values(), axis=1)
            fields = fields
            return res_final.loc[
                [s for s in securities if s in res_final.index],
                [f for f in fields if f in res_final.columns]]
        return res_final


def _extract_tick_derived_data(session, data_type, sub_date_type, tz):
    res_all = []
    while True:
        event = session.nextEvent(1000)
        if event.eventType() in [
                blpapi.Event.RESPONSE, blpapi.Event.PARTIAL_RESPONSE
        ]:
            for msg in event:
                if msg.hasElement('responseError'):
                    warn(
                        msg.getElement('responseError').getElement(
                            'message').getValueAsString())
                else:
                    res = pd.DataFrame(
                        _values(
                            msg.getElement(data_type).getElement(
                                sub_date_type)))
                    res = res if res.empty else res.set_index('time')
                    # noinspection PyTypeChecker
                    date = pd.to_datetime(res.index)
                    res.index = (
                        date +
                        tz_diff(date=date, to_tz=tz)) if tz != 'UTC' else date
                    res_all.append(res)
            if event.eventType() == blpapi.Event.RESPONSE:
                break
    return pd.concat(res_all) if res_all else None


def get_tick_data(security: 'str',
                  event_types='TRADE',
                  start=None,
                  end=None,
                  tz='Europe/Zurich',
                  n_days=3,
                  overrides=None,
                  **kwargs) -> 'pd.DataFrame':
    """

    Parameters
    ----------
    security :
        String, security to query
    event_types :
        Possible types are 'TRADE', 'BID', 'ASK', 'BID_BEST', 'ASK_BEST', 'BID_YIELD', 'ASK_YIELD', 'MID_PRICE',
        'AT_TRADE'
    start :
        Start date(time) of the query, default to end - n_days.
    end :
        End date(time) of the query, default to now/today. For tick/bar data, end need to be set to a time after market
        close to get all data of the end date.
    tz :
        Parameter for tick/bar data, time zone
    n_days :
        used to derived start if it is not given and defaults to 365 for historical data and 3 for tick/bar data
    overrides:
        Bloomberg overrides, given as dict
    kwargs :
        Passed to Bloomberg Request. Check bloomberg API doc for available options

    Returns
    -------
    """

    start, end = _start_end(start, end, n_days, tz=tz, intraday=True)
    with create_session() as session:
        request = create_request(
            session=session,
            service=REF_SERVICE,
            request='IntradayTickRequest',
            security=security,
            eventTypes=[
                s.upper()
                for s in ([event_types] if isinstance(event_types, str
                                                      ) else event_types)
            ],
            startDateTime=start,
            endDateTime=end,
            overrides=overrides,
            **kwargs)
        session.sendRequest(request)
        res = _extract_tick_derived_data(session, 'tickData', 'tickData', tz)
        return res.astype({k: v
                           for k, v in TICK_DTYPES.items()
                           if k in res}) if res is not None else None


def get_bar_data(security: 'str',
                 event_type='TRADE',
                 start=None,
                 end=None,
                 interval=10,
                 recur_daily=False,
                 tz='Europe/Zurich',
                 n_days=3,
                 **kwargs):
    """

    Parameters
    ----------
    security :
        String, security to query
    event_type :
        Possible types are 'TRADE', 'BID', 'ASK' for bar data
    start :
        Start date(time) of the query, default to end - n_days.
    end :
        End date(time) of the query, default to now/today. For tick/bar data, end need to be set to a time after market
        close to get all data of the end date.
    interval :
        Parameter for bar data, resample interval. One minute is the lowerst possible interval according to BBG
    recur_daily :
        Parameter for bar data.  True the bar series is imported for the same time period every day between the
        specified start and end dates
    tz :
        Parameter for tick/bar data, time zone
    n_days :
        used to derived start if it is not given and defaults to 365 for historical data and 3 for tick/bar data
    kwargs :
        Passed to Bloomberg Request. Check bloomberg API doc for available options

    Returns
    -------
    """
    start, end = _start_end(start, end, n_days, tz=tz, intraday=True)
    request_name = 'IntradayBarDateTimeChoiceRequest' if recur_daily else 'IntradayBarRequest'
    with create_session() as session:
        request = create_request(session=session,
                                 service=REF_SERVICE,
                                 request=request_name,
                                 security=security,
                                 eventType=event_type.upper(),
                                 interval=interval,
                                 **kwargs)
        if recur_daily:
            duration = (end - start).seconds
            element = request.getElement('dateTimeInfo').getElement(
                'startDateDuration')
            for d in pd.date_range(start, end):
                element.getElement('rangeStartDateTimeList').appendValue(d)
            element.setElement('duration', duration)
        else:
            request.set("startDateTime", start)
            request.set("endDateTime", end)
        session.sendRequest(request)
        res = _extract_tick_derived_data(session, 'barData', 'barTickData', tz)
        return None if res is None or res.empty else res[[
            'open', 'high', 'low', 'close', 'volume'
        ]].astype('float')

