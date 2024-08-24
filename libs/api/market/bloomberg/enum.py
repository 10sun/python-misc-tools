'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

MKT_DATA = "//blp/mktdata"
REF_DATA = "//blp/refdata"
SRC_SERVICE = "//blp/srcref"
MKT_DEPTH = "//blp/mktdepthdata"
MKT_BAR = "//blp/mktbar"
MKT_LIST = "//blp/mktlist"
API_FIELDS = "//blp/apiflds"
BBG_INS = "//blp/instruments"
PAGE_DATA = "//blp/pagedata"
TECH_ANALYSIS = "//blp/tasvc"
CURVE_TOOLS = "blp/irdctk3"

FREQUENCY = {
    "D": "DAILY",
    "W": "WEEKLY",
    "M": "MONTHLY",
    "Q": "QUARTERLY",
    "S": "SEMI_ANNUALLY",
    "Y": "YEARLY",
}

REQUEST_TYPE = {
    "History": "HistoricalDataRequest",
    "IntraDayTick": "IntradayTickRequest",
    "IntraDayBar": "IntradayBarRequest",
    "Reference": "ReferenceDataRequest",
    "ListChain": "chain",
    "ListSec": "secids"
}

HIST_DTYPES = {}
TICK_DTYPES = {
    "value": "float",
    "size": "int",
    "tradeTime": "datetime64[ns]",
}



FIELD_DESCRIPTION = {
    'PX_LAST': 'Price',
}

DESCRIPTION_ATTRIBUTE = {
    'Price': 'price',
}

