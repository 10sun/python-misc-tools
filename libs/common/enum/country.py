'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-22 01:06:18
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
import pycountry


MAJOR_MARKETS = [
    "US",
    "China",
    "EU",
    "United Kingdom",
    "Japan",
    "EM",
    "LatAm",
]

COUNTRY_OF_INTEREST = [
    "US",
    "China",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "United Kingdom",
    "Japan",
    "South Korea",
    "Switzerland",
    "Brazil",
    "Russia",
    "India",
    "South Africa",
    "Mexico",
    "Canada",
    "Australia",
    "Netherlands",
    "Indonesia",
    "Greece",
    "Czech",
    "Poland",
    "Hungary",
    "Turkey",
    "Argentina",
    "Taiwan",
    "Belgium",
    "Ireland",
    "Israel",
    "Saudi Arab",
    "United Arab Emirates",
    "Hong Kong",
    "Singapore",
    "Thailand",
    "Chile",
    "Colombia",
]


REGION_OF_INTREST = [
    "EU",
    "EM",
]

REGION_WEIGHTS = {
    "EU": {
        "Germany": 0.29,
        "France": 0.21,
        "Italy": 0.14,
        "Spain": 0.1,
    },
    "DM": {
        "US": 0.487,
        "Japan": 0.115,
        "Germany": 0.087,
        "United Kingdom": 0.064,
        "France": 0.062,
        "Italy": 0.045,
        "Canada": 0.039,
        "Spain": 0.032,
        "Australia": 0.032,
        "Netherlands": 0.021,
        "Switzerland": 0.016,
    },
    "DMxUS": {
        "Japan": 0.215,
        "Germany": 0.17,
        "United Kingdom": 0.125,
        "France": 0.120,
        "Italy": 0.089,
        "Canada": 0.077,
        "Spain": 0.064,
        "Australia": 0.062,
        "Netherlands": 0.040,
        "Switzerland": 0.031,
    },
    "EM": {
        "China": 0.533,
        "India": 0.107,
        "Brazil": 0.068,
        "Russia": 0.063,
        "South Korea": 0.061,
        "Mexico": 0.047,
        "Indonesia": 0.042,
        "Turkey": 0.028,
        "Taiwan": 0.022,
        "Argentina": 0.017,
        "South Africa": 0.013,
    },
    "EMxCN": {
        "India": 0.229,
        "Brazil": 0.146,
        "Russia": 0.135,
        "South Korea": 0.131,
        "Mexico": 0.100,
        "Indonesia": 0.089,
        "Turkey": 0.060,
        "Taiwan": 0.046,
        "Argentina": 0.036,
        "South Africa": 0.028,
    },
    "LatAm": {
        "Brazil": 0,
        "Argentina": 0,
        "Mexico": 0,
        "Chile": 0,
        "Colombia": 0,
    },
}


def get_country_attr(country: str, attribute: str = "name"):
    country_info = pycountry.countries.search_fuzzy(country)[0]
    if attribute.casefold() == "name".casefold():
        if hasattr(country_info, "common_name"):
            country_attr = country_info.common_name
        else:
            country_attr = country_info.name
    else:
        country_attr = getattr(country_info, attribute)
    return country_attr


def weight_sum_countries(data: pd.DataFrame, region: str, countries: dict):
    data.columns = [get_country_attr(cnt, "name") for cnt in data.columns]

    country_weights = pd.DataFrame.from_dict(countries, orient="index").T

    common_cnt = data.columns.intersection(country_weights.columns)

    region_data = (
        pd.DataFrame(
            (
                data[common_cnt].astype(float).mul(country_weights[common_cnt].values)
            ).sum(axis=1)
        )
        .div(country_weights[common_cnt].sum(axis=1).values)
        .dropna()
    )
    region_data.columns = [region]
    return region_data
