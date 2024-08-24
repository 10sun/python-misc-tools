'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-04 01:30:36
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from typing import List
import pandas as pd
import numpy as np
import decimal
import datetime
from datetime import datetime as dt, timedelta
from xbbg import blp
import logging
from pathlib import Path
import copy
from sklearn import linear_model

import dataIO


def find_nonzero_runs(data: np.array) -> list:
    """get the coordinats of non-zero subarrays

    Args:
        data (np.array): [description]

    Returns:
        list: the list of indices of the first and last element of non-zero subarrays
    """
    isnonzero = np.concatenate(([0], (np.asarray(data) != 0).view(np.int8), [0]))

    abs_diff = np.abs(np.diff(isnonzero))
    ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
    return ranges


def get_ticker_data(ticker, col_name=None, params={}) -> pd.DataFrame:
    """[summary]

    Args:
        ticker ([type]): [description]
        col_namae ([type], optional): [description]. Defaults to None.
        field (str, optional): [description]. Defaults to 'PX_LAST'.
    """
    tickerDf = blp.bdh(
        tickers=ticker,
        flds=params.get("field", ["PX_LAST"]),
        Per=params.get("per", "M"),
        Days=params.get("days", "T"),
        start_date=params.get("start_date", "2001-01-01"),
        end_date=params.get("end_date", str(dt.today().date())),
    )
    if col_name is not None:
        tickerDf.columns = [col_name]
    tickerDf.index.names = ["Date"]
    tickerDf.index = pd.to_datetime(tickerDf.index)
    return tickerDf


def get_universe_prices(asset_universe: pd.DataFrame, params={}) -> pd.DataFrame:
    ticker_prices = []
    ticker_returns = []
    for ticker in asset_universe.Ticker:
        logging.info(
            str(asset_universe.loc[asset_universe.Ticker == ticker].AssetClass.values)
            + ": "
            + ticker
            + ", "
            + asset_universe.loc[asset_universe.Ticker == ticker].Method.values[0]
        )
        ticker_price = get_ticker_data(
            ticker,
            col_name=asset_universe.loc[asset_universe.Ticker == ticker].Asset.values[
                0
            ],
            params=params,
        )
        ticker_prices.append(ticker_price)

        if (
            asset_universe.loc[asset_universe.Ticker == ticker].Method.values[0]
            == "pct"
        ):
            ticker_return = ticker_price.pct_change(params.get("window", 1))
        elif (
            asset_universe.loc[asset_universe.Ticker == ticker].Method.values[0]
            == "diff"
        ):
            ticker_return = ticker_price.diff(params.get("window", 1))
        else:
            raise ("no computation method provided...")
            return
        ticker_returns.append(ticker_return)
    return pd.concat(ticker_prices, axis=1), pd.concat(ticker_returns, axis=1)


def get_universe_returns(
    asset_universe: pd.DataFrame, asset_prices: pd.DataFrame, params={}
) -> pd.DataFrame:
    ticker_returns = []
    for ticker in asset_universe.Ticker:
        ticker_price = asset_prices[
            [asset_universe.loc[asset_universe.Ticker == ticker].Asset.values[0]]
        ]
        if (
            asset_universe.loc[asset_universe.Ticker == ticker].Method.values[0]
            == "pct"
        ):
            ticker_return = ticker_price.pct_change(params.get("window", 1))
        elif (
            asset_universe.loc[asset_universe.Ticker == ticker].Method.values[0]
            == "diff"
        ):
            ticker_return = ticker_price.diff(params.get("window", 1))
        else:
            raise ("no computation method provided...")
            return
        ticker_returns.append(ticker_return)
    return pd.concat(ticker_returns, axis=1)


def get_asset_group_data(asset_groups: dict, asset_data: pd.DataFrame):
    """get both the absolute and relative data provided a dict of asset groups

    Args:
        asset_groups (dict): [description]
        asset_data (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    all_group_data = {}
    for group, group_list in asset_groups.items():
        logging.info(group)
        # get the raw prices of all asset in the group
        group_raw_data = asset_data[group_list.Asset]
        all_group_data.update({group: {"all": group_raw_data}})

        # get the relative prices
        if "Reference" in group_list.columns:
            # reference_prices = asset_prices[asset_list.Reference]
            group_data = {}
            for asset_group in list(set(list(group_list.AssetGroup))):
                # get the asset list of the asset group
                asset_group_list = group_list.loc[group_list.AssetGroup == asset_group]
                # get the absolute and relative data of the asset group
                abs_data = asset_data[asset_group_list.Asset]
                rel_data = (
                    asset_data[asset_group_list.Asset]
                    - asset_data[asset_group_list.Reference].values
                )
                # check if the reference asset is already in the asset group
                ref_not_included = [
                    c
                    for c in list(set(list(asset_group_list.Reference)))
                    if c not in list(asset_group_list.Asset)
                ]

                if ref_not_included:
                    abs_data = pd.concat(
                        [abs_data, asset_data[ref_not_included]], axis=1
                    )
                    rel_ref_not_included = asset_data[ref_not_included]
                    rel_ref_not_included[~rel_ref_not_included.isnull()] = 0
                    rel_data = pd.concat([rel_data, rel_ref_not_included], axis=1)

                # include the data of the asset group, both absolute and relative
                group_data.update(
                    {asset_group: {"Absolute": abs_data, "Relative": rel_data}}
                )
            all_group_data.get(group, ValueError("no data available...")).update(
                {"groups": group_data}
            )
        # asset_group_prices.get(group,
        #                       ValueError('no data available...')).update({
        #                           'AbsoluteStats':
        #                           asset_list.AbsoluteStats[0]
        #                       })
    return all_group_data


"""
def get_asset_returns(asset_groups: dict, asset_returns: pd.DataFrame) -> dict:
    asset_group_returns = {}
    for group, asset_list in asset_groups.items():
        logging.info(group)
        # get the absolute returns
        group_abs_returns = asset_returns[asset_list.Asset]
        asset_group_returns.update({group: {'all': group_abs_returns}})
        # get the relative returns
        if 'Reference' in asset_list.columns:
            group_returns = {}
            for asset_group in list(set(list(asset_list.AssetGroup))):
                asset_group_list = asset_list.loc[asset_list.AssetGroup ==
                                                  asset_group]
                rel_returns = asset_returns[
                    asset_group_list.Asset] - asset_returns[
                        asset_group_list.Reference].values
                group_returns.update({
                    asset_group: {
                        'Absolute': asset_returns[asset_group_list.Asset],
                        'Relative': rel_returns
                    }
                })
            asset_group_returns.get(
                group, ValueError('no returns available...')).update(
                    {'groups': group_returns})
        #asset_group_returns.get(group,
        #                        ValueError('no returns available...')).update({
        #                            'AbsoluteStats':
        #                            asset_list.AbsoluteStats[0]
        #                        })
    return asset_group_returns
"""


def cum_returns_final(returns, starting_value=0):
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, pd.DataFrame):
        result = (returns + 1).prod()
    else:
        result = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value

    return result


def annual_return(returns, period="M", annualization=None):
    if len(returns) < 1:
        return np.nan

    if period == "D":
        ann_factor = 252
    elif period == "W":
        ann_factor = 52
    elif period == "M":
        ann_factor = 12
    num_years = len(returns) / ann_factor
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)
    return ending_value ** (1 / num_years) - 1


def get_data_slope(dataDf):
    reg = linear_model.LinearRegression()
    dataDf = dataDf.reset_index()
    dataDf.columns = ["index", "value"]
    reg.fit(dataDf.index.values.reshape(-1, 1), dataDf["value"].values)
    return reg.coef_


def get_tableauDf(
    statsDf: pd.DataFrame,
    kpi_str: str,
    index_name: str,
    col_name: str,
    additional_cols: dict,
    kpi_name="KPI",
    value_name="Value",
) -> pd.DataFrame:
    """[summary]

    Args:
        statsDf (pd.DataFrame): [description]
        kpi_str (str): [description]
        index_name (str): [description]
        col_name (str): [description]
        additional_cols ([type], optional): [description]. Defaults to None.

    Returns:
        pd.DataFrame: [description]
    """
    regime_tableau = dataIO.df_to_tableau(
        statsDf,
        kpi_str=kpi_str,
        index_name=index_name,
        col_name=col_name,
        kpi_name=kpi_name,
        value_name=value_name,
        additional_cols=additional_cols,
    )
    return regime_tableau


def regime_to_tableau(regimeDf: pd.DataFrame, regimes: dict, params={}) -> pd.DataFrame:
    # regime_str = pd.concat([pd.DataFrame(regimeDf.Regime.apply(lambda x: regimes[x]), columns={'Regime'}), regimeDf['Score']], axis=1)
    regime_all = []
    for regime, regimeStr in regimes.items():
        regimeTmp = pd.DataFrame((regimeDf.Regime == regime).astype(int))
        regimeTmp.columns = [regimeStr]
        regime_all.append(regimeTmp)

    regime_tableau = dataIO.df_to_tableau(
        pd.concat(regime_all, axis=1),
        index_name=params.get("index_name", "Date"),
        col_name=params.get("col_name", "Asset"),
        kpi_str=params.get("kpi_str", "Regime"),
        additional_cols=params.get("additional_cols", {}),
    )
    regime_score = pd.DataFrame(regimeDf.Score)
    regime_tableau = pd.concat(
        [
            regime_tableau,
            dataIO.df_to_tableau(
                regime_score,
                index_name=params.get("index_name", "Date"),
                col_name=params.get("col_name", "Asset"),
                kpi_str="RegimeScore",
                additional_cols=params.get("additional_cols", {}),
            ),
        ],
        axis=0,
    )
    return regime_tableau


def get_hit_ratio(dataDf, index_str=None, perf_ref=0):
    """[summary]

    Args:
        dataDf ([type]): [description]
        index_str ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    hitRatio = {}
    for col in dataDf.columns:
        N = len(dataDf[col].dropna())
        hit = (dataDf[col].dropna() > perf_ref).astype(int).sum()
        hitRatio.update({col: hit / N})
    hitDf = pd.DataFrame.from_dict(hitRatio, orient="index")
    hitDf.columns = [index_str]
    return hitDf.T


def get_asset_info(
    asset_list: pd.DataFrame,
    tableauDf: pd.DataFrame,
    info_list=None,
    on_cols=["Asset", "AssetGroup"],
) -> pd.DataFrame:
    try:
        if info_list is None:
            info_list = [
                "Asset",
                "AssetGroup",
                "AssetClass",
                "Reference",
                "Ticker",
                "Method",
            ]
        on_list = [c for c in on_cols if c in info_list and c in tableauDf.columns]
        index_name = tableauDf.index.names[0]
        new_tableauDf = (
            tableauDf.reset_index()
            .merge(asset_list[info_list], on=on_list)
            .set_index(index_name)
        )
        """
        tableauDf['AssetClass'] = tableauDf['Asset'].apply(
            lambda x: asset_list.loc[asset_list.Asset == x].AssetClass.values)
        tableauDf['AssetGroup'] = tableauDf['Asset'].apply(
            lambda x: asset_list.loc[asset_list.Asset == x].AssetGroup.values)
        tableauDf['Reference'] = tableauDf['Asset'].apply(
            lambda x: asset_list.loc[asset_list.Asset == x].Reference.values)
        """
    except Exception as e:
        logging.error(e)
        return tableauDf
    return new_tableauDf


def get_regime_current_info(
    regimeDf: pd.DataFrame, regimes: dict, regime_index="WEI", params={}
):
    """[summary]

    Args:
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]
        regime_index (str, optional): [description]. Defaults to 'WEI'.
        params (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """
    # get the current month info
    regime_str = pd.DataFrame(
        regimeDf.Regime.apply(lambda x: regimes[x]), columns=["Regime"]
    )

    current_regime = regime_str.loc[regime_str.Regime == regime_str.iloc[-1].values[0]]
    current_regime = current_regime.reset_index()
    if any(
        (current_regime["Date"] - current_regime["Date"].shift(1))
        > datetime.timedelta(days=31)
    ):
        current_regime_start_date = current_regime.index[
            (current_regime["Date"] - current_regime["Date"].shift(1))
            > datetime.timedelta(days=31)
        ][-1]
    else:
        current_regime_start_date = current_regime.index.min()

    current_regime_month_ind = (
        current_regime["Date"].index[-1] - current_regime_start_date + 1
    )
    current_regimeDf = pd.DataFrame(
        [
            str(regime_str.iloc[-1].values[0]),
            str(current_regime.iloc[current_regime_start_date].Date.date()),
            str(current_regime.iloc[-1].Date.date()),
            str(current_regime_month_ind),
        ]
    ).T
    current_regimeDf.columns = [
        "CurrentRegime",
        "StartMonth",
        "CurrentMonth",
        "MonthInd",
    ]
    current_regimeDf["Date"] = [str(regime_str.index[-1])]

    if regime_index == "Inflation":
        current_regimeDf["RegimeMean"] = np.round(regimeDf["Score"].mean(), 2)
        current_regimeDf["RegimeStd"] = np.round(regimeDf["Score"].std(), 2)
        current_regimeDf["RegimeUp"] = np.round(
            regimeDf["Score"].mean() + regimeDf["Score"].std(), 2
        )
        current_regimeDf["RegimeDown"] = np.round(
            regimeDf["Score"].mean() - regimeDf["Score"].std(), 2
        )
    current_regimeDf = current_regimeDf.set_index("Date")
    # convert regime df into tableau applicable format
    current_regime_tableau = get_tableauDf(
        current_regimeDf,
        kpi_str=params.get("kpi_str", "CurrentRegime"),
        index_name=params.get("index_name", "Date"),
        col_name=params.get("col_name", "Asset"),
        value_name="Info",
        additional_cols=params.get("additional_cols", {}),
    )
    return current_regime_tableau


def get_regime_duration(regimeDf: pd.DataFrame, regimes: dict, regime_index: str):
    regime_duration = {}
    for regime_key, regime in regimes.items():
        # get the indexes of each period within each regime
        logging.debug(regimes[regime_key])
        periods = find_nonzero_runs((regimeDf.Regime == regime_key).astype(int).values)
        if len(periods) == 0:
            continue
        # get the data for each period
        period_durations = []
        for period in periods:
            # get period data
            if period[0] <= regimeDf.shape[0]:
                period_regime = regimeDf.iloc[period[0] : period[1]]
                start_month = period_regime.index.min()
                end_month = period_regime.index.max()
                if end_month == regimeDf.index.max() and regime_index != "Decade":
                    continue
                period_duration = pd.DataFrame(
                    [
                        (end_month.year - start_month.year) * 12
                        + (end_month.month - start_month.month)
                        + 1
                    ],
                    columns=[regimes[regime_key]],
                    index=[start_month],
                )
                period_durations.append(period_duration)
            # elif period[0] == regimeDf.shape[0]:
            #    period_durations.append(pd.DataFrame([1], columns=[regimes[regime_key]], index=[start_month]))
            else:
                logging.error("period out of index...")
                continue
        regime_duration.update(
            {regime: {"Duration": pd.concat(period_durations, axis=0)}}
        )
    return regime_duration


def get_regime_data(
    dataDf: pd.DataFrame,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    last: bool = False,
) -> dict:
    """[summary]

    Args:
        dataDf (pd.DataFrame): [description]
        regimeDf (pd.DataFrame): [description]
        regime_dict (dict): [description]

    Returns:
        dict: [description]
    """
    regime_data = {}
    for regime_key, regime in regimes.items():
        logging.debug(regime + ": " + str(regime_key))
        # get the indexes of each period within each regime
        periods = find_nonzero_runs((regimeDf.Regime == regime_key).astype(int).values)
        if len(periods) == 0:
            continue
        # get the data for each period
        period_data = []
        for period in periods:
            # get period data
            if period[0] <= dataDf.shape[0]:
                periodDf = dataDf[period[0] : period[1]]
                if (
                    periodDf.index.max() == regimeDf.index.max()
                    and regime_index != "Decade"
                    and not last
                ):
                    continue
                period_data.append(periodDf)
            else:
                logging.error("period out of index...")
                continue
        regime_data.update({regime: period_data})
    return regime_data


def get_stats(
    regime_performances: dict,
    index_name="Regime",
    col_name="Asset",
    kpi_dict=None,
    statistic=["mean", "median", "max", "min"],
    additional_cols=None,
    tableau=True,
):
    """[summary]

    Args:
        regime_performances ([type]): [description]
        index_name (str, optional): [description]. Defaults to 'Cycle'.
        col_name (str, optional): [description]. Defaults to 'Asset'.
        statistic (list, optional): [description]. Defaults to ['mean', 'median', 'max', 'min'].

    Returns:
        [type]: [description]
    """
    if kpi_dict is None:
        kpi_dict = {
            "AverageMonthlyReturn": "The mean of monthly return of each period within the same regime",
            "CumulativeReturn": "The mean of cumulative return of each period within the same regime",
            "AnnualizedReturn": "The mean of annulized return of each period within the same regim",
            "Correlation": "The mean correlation between the return of the base asset and that of the selected asset",
            "Duration": "The number of months within the same regime",
            "HitRatio": "The hit ratio of an asset's return within a regime relative to a reference asset (by default 0)",
        }
    else:
        kpi_dict = {
            **kpi_dict,
            **{
                "AverageMonthlyReturn": "The mean of monthly return of each period within the same regime",
                "CumulativeReturn": "The mean of cumulative return of each period within the same regime",
                "AnnualizedReturn": "The mean of annulized return of each period within the same regim",
                "Correlation": "The mean correlation between the return of the base asset and that of the selected asset",
                "Duration": "The number of months within the same regime",
                "HitRatio": "The hit ratio of an asset's return within a regime relative to a reference asset (by default 0)",
            },
        }

    regime_stats = []
    for regime, performances in regime_performances.items():
        # for each regime
        for kpi, kpiDf in performances.items():
            # kpi: ['AverageMonthlyReturn', 'CumulativeReturn', 'AnnualizedReturn']
            for s in statistic:
                if additional_cols is None:
                    additional_cols = {"Statistic": s, "Description": kpi_dict[kpi]}
                else:
                    additional_cols = {
                        **additional_cols,
                        **{"Statistic": s, "Description": kpi_dict[kpi]},
                    }
                # s: ['mean', 'median', 'max', 'min']
                try:
                    if tableau:
                        regime_stats.append(
                            get_tableauDf(
                                pd.DataFrame(kpiDf.agg(s, axis=0), columns=[regime]).T,
                                kpi_str=kpi,
                                index_name=index_name,
                                col_name=col_name,
                                additional_cols=additional_cols,
                            )
                        )
                    else:
                        regime_stats.append(
                            pd.DataFrame(kpiDf.agg(s, axis=0), columns=[regime]).T
                        )
                except Exception as e:
                    logging.error(e)
    return pd.concat(regime_stats, axis=0)


def get_period_stats(period_returns: pd.DataFrame, hitratio=True) -> pd.DataFrame:
    """[summary]

    Args:
        period_returns (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    avg_monthly_return = pd.DataFrame(period_returns.mean(axis=0)).T
    avg_monthly_return.index = [period_returns.index.min()]

    cumulative_return = pd.DataFrame(period_returns.sum(axis=0, min_count=1)).T
    cumulative_return.index = [period_returns.index.min()]

    annualized_return = pd.DataFrame(annual_return(period_returns, period="M")).T
    annualized_return.index = [period_returns.index.min()]

    period_performance = pd.concat(
        [avg_monthly_return, cumulative_return, annualized_return], axis=0
    )
    period_performance["KPI"] = [
        "AverageMonthlyReturn",
        "CumulativeReturn",
        "AnnualizedReturn",
    ]

    if hitratio:
        hit_ratio = get_hit_ratio(period_returns)
        hit_ratio.index = [period_returns.index.min()]
        hit_ratio["KPI"] = ["HitRatio"]
        period_performance = pd.concat([period_performance, hit_ratio], axis=0)
    return period_performance


def get_regime_performances(regime_data: dict, hitratio=True) -> dict:
    """[summary]

    Args:
        regime_data (dict): [description]
        absolute_stats (bool, optional): [description]. Defaults to True.

    Returns:
        dict: [description]
    """
    regime_perf = {}
    for regime, data in regime_data.items():
        # print(regime)
        period_performances = pd.concat(
            [get_period_stats(returns, hitratio) for returns in data], axis=0
        )
        kpis = list(set(list(period_performances.KPI)))
        for kpi in kpis:
            if regime not in regime_perf.keys():
                regime_perf.update(
                    {
                        regime: {
                            kpi: period_performances.loc[
                                period_performances.KPI == kpi,
                                period_performances.columns != "KPI",
                            ]
                        }
                    }
                )
            else:
                regime_perf[regime].update(
                    {
                        kpi: period_performances.loc[
                            period_performances.KPI == kpi,
                            period_performances.columns != "KPI",
                        ]
                    }
                )
    return regime_perf


def get_regime_transition_matrix(
    regimeDf: pd.DataFrame, regimes: dict, additional_cols=None
):
    """[summary]

    Args:
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]

    Returns:
        [type]: [description]
    """
    regimes_str = pd.DataFrame(regimeDf.Regime.apply(lambda x: regimes[x]))
    regimes_str.index.names = ["Date"]
    regime_mat = pd.crosstab(
        pd.Series(regimes_str.Regime.values[:-1]),
        pd.Series(regimes_str.Regime.values[1:]),
    )
    regime_trans_mat = regime_mat.div(regime_mat.sum(axis=1), axis=0)
    if additional_cols is None:
        additional_cols = {
            "Description": "The probability of the regime moving into another regime"
        }
    else:
        additional_cols = {
            **additional_cols,
            **{
                "Description": "The probability of the regime moving into another regime"
            },
        }
    regime_trans_mat_tableau = get_tableauDf(
        regime_trans_mat,
        kpi_str="Probability",
        index_name="Regime",
        col_name="Asset",
        additional_cols=additional_cols,
    )
    return regime_trans_mat_tableau


def get_regime_historical_data(
    asset_groups: dict,
    asset_prices: dict,
    asset_returns: dict,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    since_date=None,
):
    """[summary]

    Args:
        asset_groups (dict): [description]
        asset_prices (dict): [description]
        asset_returns (dict): [description]
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]
        regime_index (str): [description]

    Returns:
        [type]: [description]
    """
    all_data_tableau = []
    for group in asset_prices.keys():
        # print(group)
        group_tableau = []
        # absolute = bool(group_prices.get('AbsoluteStats', True))
        group_list = asset_groups.get(group, ValueError("no asset list available..."))
        group_prices = asset_prices.get(
            group, ValueError("no price data available...")
        ).get("groups", ValueError("no data available..."))
        group_returns = asset_returns.get(
            group, ValueError("no return data available...")
        ).get("groups", ValueError("no return data available..."))
        if since_date is None:
            since_date = str(
                asset_prices.get(group, ValueError("no price data available..."))
                .get("all", ValueError("no data available..."))
                .index.min()
                .date()
            )
        else:
            since_date = str(since_date)
        # if absolute else group_prices.get(
        # group, ValueError('no data available'))
        for asset_group in group_prices.keys():
            additional_cols = {
                "Group": group,
                "AssetGroup": asset_group,
                "RegimeIndex": regime_index,
                "Since": since_date,
            }

            # get price data in place
            for abs_rel, prices in group_prices[asset_group].items():
                prices = prices.loc[prices.index.intersection(regimeDf.index)].dropna(
                    how="all", axis=1
                )

                prices_tableau = get_tableauDf(
                    prices,
                    kpi_str="Price",
                    index_name="Date",
                    col_name="Asset",
                    additional_cols={**additional_cols, **{"Performance": abs_rel}},
                )

                group_tableau.append(get_asset_info(group_list, prices_tableau))

            # get return data in place
            for abs_rel, returns in group_returns[asset_group].items():
                returns = returns.loc[
                    returns.index.intersection(regimeDf.index)
                ].dropna(how="all", axis=1)
                rebased_cum_returns = ((1 + returns).cumprod()) * 100
                cumulative_returns = rebased_cum_returns.loc[
                    rebased_cum_returns.index.intersection(regimeDf.index)
                ]
                returns_tableau = get_tableauDf(
                    cumulative_returns,
                    kpi_str="RebasedReturn",
                    index_name="Date",
                    col_name="Asset",
                    additional_cols={**additional_cols, **{"Performance": abs_rel}},
                )
                group_tableau.append(get_asset_info(group_list, returns_tableau))

                asset_group_regimeDf = pd.DataFrame(
                    regimeDf.loc[regimeDf.index.intersection(returns.index)]
                )
                group_tableau.append(
                    regime_to_tableau(
                        asset_group_regimeDf,
                        regimes,
                        params={
                            "additional_cols": {
                                **additional_cols,
                                **{"Performance": abs_rel},
                            }
                        },
                    )
                )

                # get the current regime information
                group_tableau.append(
                    get_regime_current_info(
                        asset_group_regimeDf,
                        regimes,
                        regime_index,
                        {
                            "additional_cols": {
                                **additional_cols,
                                **{"Performance": abs_rel},
                            }
                        },
                    )
                )
        all_data_tableau.append(pd.concat(group_tableau, axis=0))
    return pd.concat(all_data_tableau, axis=0)


def get_regime_historical_stats(
    asset_groups: dict,
    asset_returns: dict,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    next_n: List[int] = [0, 1],
    since_date=None,
) -> pd.DataFrame:
    """[summary]

    Args:
        asset_groups (dict): [description]
        asset_returns (dict): [description]
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]
        regime_index (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    regime_stats = []

    for group, group_returns in asset_returns.items():
        logging.info(group)
        # get the start date of the data
        if since_date is None:
            since_date = str(
                group_returns.get("all", ValueError("no return data available..."))
                .index.min()
                .date()
            )
        else:
            since_date = str(since_date)

        # get regime duration stats
        group_regimeDf = regimeDf.loc[
            regimeDf.index.intersection(group_returns["all"].index)
        ]

        if regime_index != "Decade":
            group_regime_duration = get_regime_duration(
                group_regimeDf, regimes, regime_index
            )
            regime_stats.append(
                get_stats(
                    group_regime_duration,
                    additional_cols={
                        "RegimeIndex": regime_index,
                        "Group": group,
                        "Since": since_date,
                    },
                )
            )

            # get regime transition matrix
            regime_stats.append(
                get_regime_transition_matrix(
                    group_regimeDf,
                    regimes,
                    additional_cols={
                        "RegimeIndex": regime_index,
                        "Group": group,
                        "Since": since_date,
                    },
                )
            )

        # absolute = bool(group_returns.get('AbsoluteStats', True))
        group_list = asset_groups[group]
        group_stats = []
        # get returns
        asset_group_returns = group_returns.get(
            "groups", ValueError("no return data available...")
        )
        for asset_group, returns in asset_group_returns.items():
            logging.info(asset_group)
            additional_cols = {
                "Group": group,
                "AssetGroup": asset_group,
                "RegimeIndex": regime_index,
                "Since": since_date,
            }
            for abs_rel, grp_returns in returns.items():
                logging.debug(abs_rel)
                # for shift_n, shift_n_str in {0: 'Current', 1: 'Next 1M', 3:'Next 3M'}.items():
                for shift_n in next_n:  # 1
                    if shift_n != 0:
                        grp_returns = grp_returns.copy().shift(-shift_n)
                    grp_returns = grp_returns.loc[
                        grp_returns.index.intersection(group_regimeDf.index)
                    ]
                    grp_regimeDf = group_regimeDf.loc[
                        group_regimeDf.index.intersection(grp_returns.index)
                    ]
                    # get regime data
                    regime_data = get_regime_data(
                        grp_returns, grp_regimeDf, regimes, regime_index
                    )
                    # get performance stats
                    regime_performances = get_regime_performances(regime_data)
                    regime_perf_stats = get_stats(
                        regime_performances,
                        additional_cols={
                            **additional_cols,
                            **{
                                "Performance": abs_rel,
                                "NextMonth": shift_n,
                            },
                        },
                    )

                    group_asset_list = group_list.loc[
                        group_list.AssetGroup == asset_group
                    ]
                    group_asset_list = (
                        pd.concat(
                            [
                                group_asset_list,
                                pd.DataFrame(
                                    [list(set(group_asset_list.Reference))[0]],
                                    columns=["Asset"],
                                ),
                            ],
                            axis=0,
                        )
                        .fillna(method="ffill")
                        .reset_index(drop=True)
                    )
                    group_stats.append(
                        get_asset_info(group_asset_list, regime_perf_stats)
                    )
        # get altogether
        regime_stats.append(pd.concat(group_stats, axis=0))
    return pd.concat(regime_stats, axis=0)


def get_regime_asset_correlation(
    asset_returns: dict,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    since_date=None,
) -> pd.DataFrame:
    """[summary]

    Args:
        asset_returns (dict): [description]
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]
        regime_index (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    statistics = ["mean", "median", "max", "min"]
    all_asset_corr = []
    universe_returns = asset_returns.get("all", ValueError("no data available..."))
    universe_returns = universe_returns.loc[
        universe_returns.index.intersection(regimeDf.index)
    ]
    universe_regimeDf = regimeDf.loc[
        regimeDf.index.intersection(universe_returns.index)
    ]
    if since_date is None:
        since_date = str(universe_regimeDf.index.min().date())
    else:
        since_date = str(since_date)
    universe_regime_data = get_regime_data(
        universe_returns, universe_regimeDf, regimes, regime_index
    )
    for regime, period_returns in universe_regime_data.items():
        logging.debug(regime)
        regime_asset_corr = {}
        for returns in period_returns:
            if returns.shape[0] <= 2:
                continue
            period_corr = returns.corr()
            for asset in period_corr.index:
                asset_corr = period_corr.loc[period_corr.index == asset]
                asset_corr.index = [returns.index.min()]
                # display(asset_corr)
                if asset not in regime_asset_corr:
                    regime_asset_corr.update({asset: [asset_corr]})
                else:
                    regime_asset_corr[asset].append(asset_corr)

        # display(regime_asset_corr)
        asset_corr_tableau = []
        for asset, asset_corr in regime_asset_corr.items():
            asset_corrDf = pd.concat(asset_corr, axis=0).dropna(how="all", axis=0)
            for s in statistics:
                asset_corr_tableau.append(
                    get_tableauDf(
                        pd.DataFrame(asset_corrDf.agg(s, axis=0), columns=[asset]).T,
                        kpi_str="Correlation",
                        index_name="Reference",
                        col_name="Asset",
                        additional_cols={
                            "Group": "universe",
                            "RegimeIndex": regime_index,
                            "Regime": regime,
                            "Statistic": s,
                            "Since": since_date,
                            "Description": "Correlation of one asset with another",
                        },
                    )
                )
        if len(asset_corr_tableau) == 0:
            logging.warning("empty data in regime " + regime + "...")
            continue
        all_asset_corr.append(pd.concat(asset_corr_tableau, axis=0))
    return pd.concat(all_asset_corr, axis=0)


def get_forward_transition_matrix(
    regime_str: dict, start_month_ind: int, start_ind: list, forward_months: list
) -> pd.DataFrame:
    """[summary]

    Args:
        regime_str (dict): [description]
        start_month_ind (int): [description]
        start_ind (list): [description]
        forward_months (list): [description]

    Returns:
        pd.DataFrame: [description]
    """
    month_fwd_transition_mat = []
    for fwd_month in forward_months:
        fwd_transition_mat = pd.crosstab(
            pd.Series(
                regime_str.Regime.values[
                    [m for m in start_ind if m + fwd_month < regime_str.shape[0]]
                ]
            ),
            pd.Series(
                regime_str.Regime.values[
                    [
                        m + fwd_month
                        for m in start_ind
                        if m + fwd_month < regime_str.shape[0]
                    ]
                ]
            ),
        )
        fwd_transition_matDf = fwd_transition_mat.div(
            fwd_transition_mat.sum(axis=1), axis=0
        )
        if fwd_transition_matDf.empty:
            continue
        fwd_transition_matDf.index = [fwd_month]
        month_fwd_transition_mat.append(fwd_transition_matDf)

    if not month_fwd_transition_mat:
        return None
    month_fwd_transition_mat = pd.concat(month_fwd_transition_mat, axis=0)
    additional_cols = {
        "Month": start_month_ind,
        "Regime": regime_str.Regime.values[start_ind[0]],
        "Description": "The probablity of the next n month in a certain regime",
    }
    month_fwd_transition_tableau = get_tableauDf(
        month_fwd_transition_mat,
        kpi_str="Probability",
        index_name="ForwardMonth",
        col_name="NextRegime",
        additional_cols=additional_cols,
    )
    return month_fwd_transition_tableau


def get_forward_month_performances(
    returns: pd.DataFrame, start_months: list, forward_months: list
) -> dict:
    """[summary]

    Args:
        returns (pd.DataFrame): [description]
        start_ind (list): [description]
        forward_months (list): [description]

    Returns:
        dict: [description]
    """
    fwd_month_returns = {}
    for start_month in start_months:
        for fwd_month in forward_months:
            if start_month + fwd_month >= returns.shape[0]:
                continue
            if fwd_month not in fwd_month_returns:
                fwd_month_returns.update(
                    {fwd_month: [returns.iloc[start_month : start_month + fwd_month]]}
                )
            else:
                fwd_month_returns[fwd_month].append(
                    returns.iloc[start_month : start_month + fwd_month]
                )
    return get_regime_performances(fwd_month_returns, hitratio=False)


def get_regime_forward_stats(
    asset_list: pd.DataFrame,
    asset_returns: pd.DataFrame,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    since_date=None,
    forward_months=range(1, 13),
) -> pd.DataFrame:
    """[summary]

    Args:
        asset_returns (pd.DataFrame): [description]
        regimeDf (pd.DataFrame): [description]
        regimes (dict): [description]
        regime_index (str): [description]
        forward_months ([type], optional): [description]. Defaults to range(1, 13).

    Returns:
        pd.DataFrame: [description]
    """
    fwd_regimeDf = regimeDf.loc[regimeDf.index.intersection(asset_returns.index)]
    asset_returns = asset_returns.loc[
        asset_returns.index.intersection(fwd_regimeDf.index)
    ]
    if since_date is None:
        since_date = str(asset_returns.index.min().date())
    else:
        since_date = str(since_date)
    regime_str = pd.DataFrame(
        fwd_regimeDf.Regime.apply(lambda x: regimes[x]), columns={"Regime"}
    )

    fwd_current_regime_info = get_regime_current_info(regimeDf, regimes, regime_index)
    curr_regime = (
        pd.DataFrame(
            fwd_current_regime_info.loc[
                fwd_current_regime_info.Asset == "CurrentRegime"
            ].Info
        )
        .reset_index(drop=True)
        .rename(columns={"Info": "CurrRegime"})
    )
    curr_regime_monthInd = (
        pd.DataFrame(
            fwd_current_regime_info.loc[
                fwd_current_regime_info.Asset == "MonthInd"
            ].Info
        )
        .reset_index(drop=True)
        .rename(columns={"Info": "CurrMonthInd"})
    )
    curr_regime_info = pd.concat([curr_regime, curr_regime_monthInd], axis=1)

    fwd_stats = []
    for regime, regimeStr in regimes.items():
        logging.info(regimeStr)
        # get the starting and ending index of each period of the regime
        periods = find_nonzero_runs((fwd_regimeDf.Regime == regime).astype(int).values)

        # period_duration is the duration of each period in the cycle
        period_duration = []
        for period in periods:
            period_duration.append(period[1] - period[0])

        # max_period is the maximum duration of periods of the cycle
        if len(period_duration) == 0:
            continue

        max_month_ind_in_perdiod = max(period_duration)

        for month_ind in range(max_month_ind_in_perdiod):
            additional_cols = {"Month": (month_ind + 1), "Regime": regimeStr}
            start_month_list = [
                period[0] + month_ind
                for period in periods
                if period[0] + month_ind < period[1]
            ]

            fwd_month_perf = get_forward_month_performances(
                asset_returns, start_month_list, forward_months
            )

            # display(fwd_month_perf)

            try:
                fwd_stats.append(
                    get_asset_info(
                        asset_list,
                        get_stats(
                            fwd_month_perf,
                            index_name="ForwardMonth",
                            col_name="Asset",
                            additional_cols=additional_cols,
                        ),
                        ["Asset", "Ticker", "Method", "AssetClass"],
                    )
                )
            except Exception as e:
                logging.error(e)
                logging.info(regimeStr + ": " + str(month_ind))
                # display(fwd_month_perf)
                break

            fwd_stats.append(
                get_forward_transition_matrix(
                    regime_str, (month_ind + 1), start_month_list, forward_months
                )
            )

    if len(fwd_stats) == 0:
        logging.error("no forward looking stats...")
        return
    fwd_stats = pd.concat(fwd_stats, axis=0)
    # fwd_stats = pd.concat([fwd_stats])
    curr_regime_info.index.names = fwd_stats.index.names
    fwd_stats = pd.concat([fwd_stats, curr_regime_info], axis=0)
    fwd_stats["RegimeIndex"] = regime_index
    fwd_stats["Since"] = since_date
    return fwd_stats  # get_asset_info(asset_list, fwd_stats)


def regime_analysis(
    asset_groups: dict,
    asset_prices: dict,
    asset_returns: dict,
    regime_info: dict,
    next_n: List[int] = [0, 1],
    since_dates=[],
) -> dict:
    """[summary]

    Args:
        asset_groups (dict): [description]
        asset_prices (dict): [description]
        asset_returns (dict): [description]
        regime_info (dict): [description]

    Returns:
        dict: [description]
    """
    regime_results = {}
    for regime, regime_data in regime_info.items():
        regimeDf = regime_data.get("data", ValueError("no regime data available..."))
        regimes = regime_data.get(
            "dict", ValueError("no regime dictioanry available...")
        )
        params = regime_data.get("params", {})

        if len(since_dates) == 0:# or not params.get("since", True):
            analysis_since_dates = [regimeDf.index.min().date()]
        else:
            analysis_since_dates = [regimeDf.index.min().date()] + [
                pd.to_datetime(date).date()
                for date in since_dates
                if pd.to_datetime(date).date() > regimeDf.index.min().date()
            ]
        
        print(analysis_since_dates)

        for since in analysis_since_dates:
            logging.info("reimge index: " + regime + " since " + str(since))

            regimeDf = regimeDf.loc[regimeDf.index >= pd.to_datetime(since)]

            # get regime data
            if params.get("data", True):
                logging.info("get regime data...")
                regime_hist_data = get_regime_historical_data(
                    asset_groups,
                    asset_prices,
                    asset_returns,
                    regimeDf,
                    regimes,
                    regime,
                    since,
                )
                if "data" not in regime_results:
                    regime_results.update({"data": regime_hist_data})
                else:
                    regime_results["data"] = pd.concat(
                        [regime_results["data"], regime_hist_data], axis=0
                    )

            # get perf stats
            logging.info("get regime historical stats...")
            regime_perf_stats = get_regime_historical_stats(
                asset_groups, asset_returns, regimeDf, regimes, regime, next_n, since
            )
            if "stats" not in regime_results:
                regime_results.update({"stats": regime_perf_stats})
            else:
                regime_results["stats"] = pd.concat(
                    [regime_results["stats"], regime_perf_stats], axis=0
                )

            # get correlation
            if params.get("correlation", True):
                logging.info("get asset correlation...")
                corr_asset_returns = asset_returns["universe"]
                regime_corr_stats = get_regime_asset_correlation(
                    corr_asset_returns, regimeDf, regimes, regime, since
                )
                if "correlation" not in regime_results:
                    regime_results.update({"correlation": regime_corr_stats})
                else:
                    regime_results["correlation"] = pd.concat(
                        [regime_results["correlation"], regime_corr_stats], axis=0
                    )

            # get forward looking stats
            if params.get("forward", False):
                logging.info("get regime forward stats...")
                regime_fwd_stats = get_regime_forward_stats(
                    asset_groups.get(
                        "universe", ValueError("No asset list available...")
                    ),
                    asset_returns.get(
                        "universe", ValueError("No available data...")
                    ).get("all", ValueError("No available data..")),
                    regimeDf,
                    regimes,
                    regime,
                    since_date=since,
                )
                if "forward" not in regime_results:
                    regime_results.update({"forward": regime_fwd_stats})
                else:
                    regime_results["forward"] = pd.concat(
                        [regime_results["forward"], regime_fwd_stats], axis=0
                    )
    return regime_results


def regime_analysis_to_tableau(
    regime_results: dict, path_dir: str, file_name: str, overwrite: bool = True
):
    results = []
    try:
        for key, result in regime_results.items():
            if key == "forward":
                result.to_csv(Path(path_dir) / (file_name + "_" + key + "_tableau.csv"))
                logging.info(Path(path_dir) / (file_name + "_" + key + "_tableau.csv"))
            else:
                results.append(result.reset_index())
        pd.concat(results, axis=0).to_csv(
            Path(path_dir) / (file_name + "_analysis_tableau.csv")
        )
        logging.info(Path(path_dir) / (file_name + "_analysis_tableau.csv"))
    except Exception as e:
        logging.error(e)


def convert_asset_to_assetClass(assets: pd.DataFrame, asset_universe: pd.DataFrame):
    return


def get_regime_signal_stats(
    asset_universe: pd.DataFrame,
    asset_groups: dict,
    asset_returns: dict,
    regimeDf: pd.DataFrame,
    regimes: dict,
    regime_index: str,
    next_month: bool = False,
    params={
        "statistic": ["mean"],
        "KPI": ["AverageMonthlyReturn", "HitRatio"],
        "performance": ["Absolute", "Relative"],
        "positionN": 1,
    },
    asset_perf_stats=None,
) -> dict:
    if asset_perf_stats is None:
        asset_perf_stats = get_regime_historical_stats(
            asset_groups, asset_returns, regimeDf, regimes, regime_index
        )

    group_signals = {}
    for group in asset_groups.keys():
        asset_group_signals = {}
        for KPI in params.get("KPI", ["AverageMonthlyReturn", "HitRatio"]):  #
            for s in params.get("statistic", ["mean"]):
                next_n = 1 if next_month else 0
                asset_group_sig = asset_perf_stats.loc[
                    (asset_perf_stats.KPI == KPI)
                    & (asset_perf_stats.Statistic == s)
                    & (asset_perf_stats.Group == group)
                    & (asset_perf_stats.NextMonth == next_n)
                ]

                for asset_group in list(set(list(asset_groups[group].AssetGroup))):
                    for perf in params.get("performance", ["Absolute", "Relative"]):
                        tmp = asset_group_sig.loc[
                            (asset_group_sig.AssetGroup == asset_group)
                            & (asset_group_sig.Performance == perf)
                        ]
                        asset_group_class = list(set(tmp.AssetClass))
                        if len(asset_group_class) > 1:
                            logging.error(
                                "more than one asset class identified..."
                                + asset_group_class
                            )
                        else:
                            asset_group_class = asset_group_class[0]

                        tmp["Ranking"] = tmp.groupby("Regime")["Value"].rank(
                            ascending=False, method="first", na_option="bottom"
                        )
                        try:
                            tmp = tmp.reset_index().pivot(
                                index="Ranking", columns="Regime", values="Asset"
                            )
                        except Exception as e:
                            logging.error(e)

                        long = tmp.iloc[: params.get("positionN", 1)]

                        # if group == 'l1':
                        #    long = long.applymap(lambda x: asset_universe.loc[asset_universe.Asset == x].AssetClass.values[0])
                        #    long.index = [asset_group]
                        #    long.index.names = ['AssetGroup']
                        # else:
                        #    long.index = [asset_group_class]
                        #    long['AssetGroup'] = asset_group
                        #    long.index.names = ['Group']
                        long.index = [asset_group]
                        long.index.names = ["AssetGroup"]

                        short = tmp.iloc[-params.get("positionN", 1) :]
                        # if group == 'l1':
                        #    short = short.applymap(lambda x: asset_universe.loc[asset_universe.Asset == x].AssetClass.values[0])
                        #    short.index = [asset_group]
                        #    short.index.names = ['AssetGroup']
                        # else:
                        # short.index = [asset_group_class]
                        # short['AssetGroup'] = asset_group
                        # short.index.names = ['Group']
                        short.index = [asset_group]
                        short.index.names = ["AssetGroup"]
                        # tmp['AssetClass'] =

                        tmp["AssetGroup"] = asset_group
                        tmp["Group"] = asset_group_class
                        tmp = tmp.reset_index().set_index("Group")
                        if KPI not in asset_group_signals:
                            asset_group_signals.update(
                                {
                                    KPI: {
                                        perf: {
                                            "long": long,
                                            "short": short,
                                            "ranking": tmp,
                                        }
                                    }
                                }
                            )
                        else:
                            if perf not in asset_group_signals[KPI]:
                                asset_group_signals[KPI].update(
                                    {
                                        perf: {
                                            "long": long,
                                            "short": short,
                                            "ranking": tmp,
                                        }
                                    }
                                )
                            else:
                                asset_group_signals[KPI][perf]["long"] = pd.concat(
                                    [asset_group_signals[KPI][perf]["long"], long],
                                    axis=0,
                                )
                                asset_group_signals[KPI][perf]["short"] = pd.concat(
                                    [asset_group_signals[KPI][perf]["short"], short],
                                    axis=0,
                                )
                                asset_group_signals[KPI][perf]["ranking"] = pd.concat(
                                    [asset_group_signals[KPI][perf]["ranking"], tmp],
                                    axis=0,
                                )
        group_signals.update({group: asset_group_signals})
    return group_signals


def get_level_budgets(level_assets, level, portfolio):
    assets = list(set(list(level_assets.Asset) + list(level_assets.Reference)))
    level_budget = {}
    taa_budget = portfolio.position_budgets[
        ["Weight", "AssetClassBudget", "AssetBudget"]
    ]
    for asset in assets:
        if asset in taa_budget.index:
            if level == "l1":
                budget = taa_budget.loc[
                    taa_budget.index == asset
                ].AssetClassBudget.values[0]
            else:
                budget = taa_budget.loc[taa_budget.index == asset].min(axis=1).values[0]
        else:
            if level == "l1":
                budget = portfolio.universe_budgets.loc[
                    portfolio.universe_budgets.Asset == asset
                ].AssetClassBudget.values[0]
            else:
                budget = portfolio.universe_budgets.loc[
                    portfolio.universe_budgets.Asset == asset
                ].AssetBudget.values[0]
        budget = decimal.Decimal(
            decimal.Decimal(budget).quantize(
                decimal.Decimal(".001"), rounding=decimal.ROUND_HALF_DOWN
            )
        )
        # if level == 'l1':
        #    level_budget.update({portfolio.universe_budgets.loc[portfolio.universe_budgets.Asset==asset].AssetClass.values[0]:budget})
        # else:
        level_budget.update({asset: budget})
    level_budget = pd.DataFrame.from_dict(level_budget, orient="index")
    level_budget.columns = ["Budget"]
    return level_budget


def get_level_signals(signals, level, KPI, performance, budgets, regimes):
    level_long = signals[level][KPI][performance]["long"]
    level_short = signals[level][KPI][performance]["short"]
    level_signals = {}
    display(level_short)
    for regime in list(set(regimes.Regime)):
        logging.info(regime)
        regime_short = pd.DataFrame(level_short[regime].value_counts())
        regime_short.columns = ["short"]
        regime_short = regime_short.merge(budgets, left_index=True, right_index=True)
        regime_short["position"] = regime_short.Budget / regime_short.short
        regime_long = {}
        for asset_group in level_long.index:
            funding_source = (
                level_short[regime]
                .loc[level_short[regime].index == asset_group]
                .values[0]
            )
            funding_source_position = regime_short[
                regime_short.index == funding_source
            ].position.values[0]
            long_asset = (
                level_long[regime]
                .loc[level_long[regime].index == asset_group]
                .values[0]
            )
            if long_asset not in regime_long:
                regime_long.update({long_asset: funding_source_position})
            else:
                regime_long[long_asset] = (
                    regime_long[long_asset] + funding_source_position
                )
        regime_long = pd.DataFrame.from_dict(regime_long, orient="index")
        regime_long.columns = ["long"]
        regime_short = regime_short[["Budget"]]
        regime_short.columns = ["short"]
        regime_signal = pd.concat([regime_long, regime_short], axis=1).fillna(0)
        regime_signal["Overall"] = regime_signal.long - regime_signal.short
        level_signals.update({regime: regime_signal})

    level_long["Position"] = "long"
    level_long = level_long.reset_index().set_index(["AssetGroup", "Position"])

    level_short["Position"] = "short"
    level_short = level_short.reset_index().set_index(["AssetGroup", "Position"])
    level_signals_df = pd.concat([level_long, level_short], axis=0).sort_index()

    return level_signals_df, level_signals


def get_level_active_positions(
    asset_groups, perf_stats_signals, level, KPI, performance, portfolio, regimes
):
    # get the budget for each instrument for that level
    level_budgets = get_level_budgets(asset_groups[level], level, portfolio)

    # get the active signal for each regime
    level_signals_df, level_signals = get_level_signals(
        perf_stats_signals, level, KPI, performance, level_budgets, regimes
    )

    level_positions = []
    for regime, regime_perf_stats_signals in level_signals.items():
        tmp = pd.DataFrame(regime_perf_stats_signals["Overall"]).T
        tmp.index = [regime]
        level_positions.append(tmp)
    return level_signals_df, pd.concat(level_positions, axis=0)


def get_regime_switch_time(regimeDf):
    regime_switch_index = []
    for regime_class in list(set(regimeDf.Regime)):
        start_index = [
            ind[0] for ind in find_nonzero_runs(regimeDf.Regime == regime_class)
        ]
        regime_switch_index.append(regimeDf.iloc[start_index].loc[:, "RegimeStr"])
    regime_switch_time = (
        pd.DataFrame(pd.concat(regime_switch_index, axis=0))
        .sort_index()
        .rename(columns={"RegimeStr": "Regime"})
    )
    if regime_switch_time.index[0] == regimeDf.index[0]:
        regime_switch_time = regime_switch_time.iloc[1:]
    return regime_switch_time


def get_portfolio_weights(open_weights, regime_weights):
    new_pf_instruments = list(set().union(regime_weights.columns, open_weights.columns))
    new_pf_instruments.remove("Regime")

    regime_weights[
        [c for c in new_pf_instruments if c not in regime_weights.columns]
    ] = 0
    regime_weights = regime_weights[new_pf_instruments]

    open_weights[[c for c in new_pf_instruments if c not in open_weights.columns]] = 0
    open_weights = open_weights[new_pf_instruments]

    open_weights = open_weights.shift(1)
    open_weights = open_weights.loc[
        open_weights.index.intersection(regime_weights.index)
    ].dropna(how="all", axis=0)
    open_weights = open_weights.reindex(regime_weights.index).fillna(0)

    return open_weights, (open_weights + regime_weights)


def get_strategy_performances(
    portfolio, open_weights, regime_weights, trades, start_date=None, end_date=None
):
    if start_date is None:
        start_date = regime_weights.index.min() - timedelta(days=1)
    if end_date is None:
        end_date = regime_weights.index.max()

    strat_trades = (
        trades.loc[(trades.index >= start_date) & (trades.index < end_date)]
        .reset_index()
        .set_index(["Date", "Regime"])
    )
    strat_turnover = pd.DataFrame(strat_trades.count()).rename(
        columns={0: "Transactions"}
    )

    # get the weight
    open_weights, weights = get_portfolio_weights(
        open_weights.loc[
            (open_weights.index > start_date) & (open_weights.index <= end_date)
        ],
        regime_weights.loc[
            (regime_weights.index > start_date) & (regime_weights.index <= end_date)
        ],
    )

    returns = portfolio.universe_returns[weights.columns]
    returns = returns.loc[returns.index.intersection(weights.index)]
    weights = weights.loc[weights.index.intersection(returns.index)]

    # strategy portfolio performances
    strat_returns = returns * weights
    strat_returns["Portfolio"] = strat_returns.sum(axis=1)
    strat_cum_returns = (1 + strat_returns).cumprod() * 100
    strat_ann_return = pd.DataFrame(annual_return(strat_returns)).T

    # org/benchmark portfolio performances
    bench_returns = portfolio.pf_bench_returns.loc[
        (portfolio.pf_bench_returns.index > start_date)
        & (portfolio.pf_bench_returns.index <= end_date)
    ]
    bench_cum_returns = (1 + bench_returns).cumprod() * 100
    bench_ann_return = pd.DataFrame(annual_return(bench_returns)).T

    # strategy active performances
    act_returns = (
        open_weights
        * portfolio.universe_returns[open_weights.columns].loc[
            (portfolio.universe_returns[open_weights.columns].index > start_date)
            & (portfolio.universe_returns[open_weights.columns].index <= end_date)
        ]
    )
    act_returns["Portfolio"] = act_returns.sum(axis=1)

    act_cum_returns = (1 + act_returns).cumprod() * 100
    act_ann_return = pd.DataFrame(annual_return(act_returns)).T

    return {
        "transaction": {
            "trades": strat_trades,
            "turnover": strat_turnover,
        },
        "strategy": {
            "returns": strat_returns,
            "cumulative_returns": strat_cum_returns,
            "annualized_return": strat_ann_return,
        },
        "active": {
            "returns": act_returns,
            "cumulative_returns": act_cum_returns,
            "annualized_return": act_ann_return,
        },
        "bench": {
            "returns": bench_returns,
            "cumulative_returns": bench_cum_returns,
            "annualized_return": bench_ann_return,
        },
    }


def backtest_level_signal(
    asset_groups,
    perf_stats_signals,
    level,
    KPI,
    performance,
    portfolio,
    regimeDf,
    out_sample_date,
):
    # get the regime switching time
    regime_switch_time = get_regime_switch_time(regimeDf)
    # trading_time = regime_switch_time.shift(1, freq='M')

    # get the time with active positions
    regime_active_time = regime_switch_time.reindex(
        regimeDf.loc[regimeDf.index >= regime_switch_time.index.min()].index,
        method="ffill",
    )

    # regimes is the regimeDf with the regime name
    regimes = pd.DataFrame(regimeDf.RegimeStr)
    regimes.columns = ["Regime"]

    # weights of the benchmark portfolio, i.e. the original long-only taa portfolio
    regime_bench_weights = copy.deepcopy(regimes)
    regime_bench_weights[portfolio.weights.columns] = portfolio.weights.values[0]

    # get the signal dataframe, summarizing the weights and signals
    active_signals_df, active_weights = get_level_active_positions(
        asset_groups, perf_stats_signals, level, KPI, performance, portfolio, regimes
    )
    # get the active weights for the regime
    regime_open_weights = (
        regime_active_time.merge(
            active_weights.astype(float), left_on="Regime", right_index=True
        )
        .sort_index()
        .fillna(0)
    )

    # all the trades
    trades = regime_switch_time.merge(
        active_weights, left_on="Regime", right_index=True
    ).sort_index()  # .set_index(['Date', 'Regime'])

    results = {"signal": active_signals_df}

    # overall
    results.update(
        {
            "overall": get_strategy_performances(
                portfolio, regime_open_weights, regime_bench_weights, trades
            )
        }
    )

    # in sample
    results.update(
        {
            "in": get_strategy_performances(
                portfolio,
                regime_open_weights,
                regime_bench_weights,
                trades,
                end_date=out_sample_date,
            )
        }
    )

    # out of sample
    results.update(
        {
            "out": get_strategy_performances(
                portfolio,
                regime_open_weights,
                regime_bench_weights,
                trades,
                start_date=out_sample_date,
            )
        }
    )

    return results, trades  # for level, level_signals in group_signals.items():


"""
def get_regime_hitratio(regime_data: dict,
                        absolute=True,
                        index_name='Cycle',
                        col_name='Asset') -> pd.DataFrame:
    hitratio_dict = {
        'AbsoluteHitRatio':
        'The hit ratio of an asset\'s return within a regime relative to 0',
        'RelativeHitRatio':
        'The hit ratio of an asset\'s return within a regime relative to the benchmark asset'
    }
    hit_ratio = []
    kpi = 'AbsoluteHitRatio' if absolute else 'RelativeHitRatio'
    for regime, data in regime_data.items():
        dataDf = pd.concat([returns for returns in data], axis=0)
        hit_ratio.append(
            get_tableauDf(get_hit_ratio(dataDf, regime),
                          kpi_str=kpi,
                          index_name=index_name,
                          col_name=col_name,
                          additional_cols={'Description': hitratio_dict[kpi]}))
    return pd.concat(hit_ratio, axis=0)
"""
"""
def get_period_duration(period_returns: pd.DataFrame) -> pd.DataFrame:
    period_duration = pd.DataFrame([
        (period_returns.index.max().year - period_returns.index.min().year) *
        12 +
        (period_returns.index.max().month - period_returns.index.min().month)
    ],
                                   columns=['Duration'])
    period_duration.index = [period_returns.index.min()]
    return period_duration

def get_regime_duration(regime_data: dict) -> dict:
    regime_duration = {}
    for regime, data in regime_data.items():
        period_durations = pd.concat(
            [get_period_duration(returns) for returns in data], axis=0)
        period_durations.columns = [regime]
        regime_duration.update({regime: {'Duration': period_durations}})
    return regime_duration
"""
"""
def getCycleMetricStats(cycleStats, metric):
    metricDict = {'AverageMonthlyReturn':'The mean of monthly return of each period under the same regime', \
                  'CumulativeReturn':'The mean of cumulative return of each period under the same regime', \
                  'AnnualizedReturn':'The mean of annulized return of each period under the same regim', \
                  'Correlation':'The mean correlation between the return of the base asset and that of the selected asset'}
    metricResults = {}
    metricMean = {}
    metricMax = {}
    metricMin = {}
    metricMedian = {}
    #metricHitRatio = {}
    for cycleStr, cycleDf in cycleStats.items():
        metricMean.update({
            cycleStr:
            pd.concat([
                cycleDf['Duration'],
                cycleDf.loc[:, cycleDf.columns.str.contains('-' + metric)]
            ],
                      axis=1).mean()
        })
        metricMax.update({
            cycleStr:
            pd.concat([
                cycleDf['Duration'],
                cycleDf.loc[:, cycleDf.columns.str.contains('-' + metric)]
            ],
                      axis=1).max()
        })
        metricMin.update({
            cycleStr:
            pd.concat([
                cycleDf['Duration'],
                cycleDf.loc[:, cycleDf.columns.str.contains('-' + metric)]
            ],
                      axis=1).min()
        })
        metricMedian.update({
            cycleStr:
            pd.concat([
                cycleDf['Duration'],
                cycleDf.loc[:, cycleDf.columns.str.contains('-' + metric)]
            ],
                      axis=1).median()
        })
        #metricHitRatio.update({cycleStr:hitRatio(cycleDf.loc[:, cycleDf.columns.str.contains('-'+metric)])})

    metricMeanDf = pd.DataFrame.from_dict(metricMean, orient='index')
    metricMeanDf.index.names = ['Cycle']
    metricMeanDf.columns = [
        c.replace('-' + metric, '') for c in metricMeanDf.columns
    ]
    metricResults.update({metric + 'Mean': metricMeanDf})

    metricMaxDf = pd.DataFrame.from_dict(metricMax, orient='index')
    metricMaxDf.index.names = ['Cycle']
    metricMaxDf.columns = [
        c.replace('-' + metric, '') for c in metricMaxDf.columns
    ]
    metricResults.update({metric + 'Max': metricMaxDf})

    metricMinDf = pd.DataFrame.from_dict(metricMin, orient='index')
    metricMinDf.index.names = ['Cycle']
    metricMinDf.columns = [
        c.replace('-' + metric, '') for c in metricMinDf.columns
    ]
    metricResults.update({metric + 'Min': metricMinDf})

    metricMedianDf = pd.DataFrame.from_dict(metricMedian, orient='index')
    metricMedianDf.index.names = ['Cycle']
    metricMedianDf.columns = [
        c.replace('-' + metric, '') for c in metricMedianDf.columns
    ]
    metricResults.update({metric + 'Median': metricMedianDf})

    metricHitRatioDf = pd.DataFrame.from_dict(metricHitRatio, orient='index')
    metricHitRatioDf.index.names = ['Cycle']
    metricHitRatioDf.columns = [c.replace('-'+metric, '') for c in metricHitRatioDf.columns]
    metricResults.update({metric+'HitRatio':metricHitRatioDf})

    metricResults = {metric+'Mean':metricMeanDf, metric+'Max':metricMaxDf, metric+'Min':metricMinDf, \
                     metric+'Median':metricMedianDf} #, metric+'HitRatio':metricHitRatioDf
    metricTableauDf = pd.concat(
        [
            dataIO.df_to_tableau(metricMeanDf, 'Mean', index_name='Cycle'),
            dataIO.df_to_tableau(metricMaxDf, 'Max', index_name='Cycle'),
            dataIO.df_to_tableau(metricMinDf, 'Min', index_name='Cycle'),
            dataIO.df_to_tableau(metricMedianDf, 'Median', index_name='Cycle'),
            #dataIO.df_to_tableau(metricHitRatioDf, 'HitRatio', index_name='Cycle')
        ],
        axis=0)
    metricTableauDf['Metric'] = metric
    metricTableauDf['Description'] = metricDict[metric]
    return metricTableauDf
"""
"""
def get_relative_data(prices, returns, reference_prices, reference_returns):
    relative_prices = prices - reference_prices.values
    relative_returns = returns - reference_returns.values
    return relative_prices, relative_returns
"""
"""
def cycleHistoricalStats(assetList,
                         prices,
                         returns,
                         regimeData,
                         regimeDict,
                         regimeIndex,
                         hitRatio=True):
    cycleStats = {}
    cycleData = {}
    regimeDf = []
    for cycle, cycleStr in regimeDict.items():
        print(cycleStr + ': ' + str(cycle))
        cyclePeriodData = []
        cyclePeriodStats = []
        # 1. get the indexes of each period within each cycle/regime
        rgmDfTmp = pd.DataFrame(regimeData.Regime)
        rgmDfTmp = (rgmDfTmp == cycle).astype(int)
        # nonzeroInd: starting and ending index of each period
        nonzeroInd = find_nonzero_runs(rgmDfTmp.Regime.values)
        rgmDfTmp.columns = [cycleStr]
        # append the indexes of this cycle
        regimeDf.append(rgmDfTmp)
        # get the return data for each period

        for period in nonzeroInd:
            # compute period cumulative return
            if period[0] < returns.shape[0]:
                pDf = returns[period[0]:period[1]]
            elif period[0] == returns.shape[0]:
                pDf = pd.DataFrame(returns.iloc[-1]).T
            else:
                continue

            pCumRet = pd.DataFrame(pDf.sum(axis=0, min_count=1)).T
            pCumRet.columns = [
                c + '-CumulativeReturn' for c in pCumRet.columns
            ]

            # compute period average monthly return
            pAvgMRet = pd.DataFrame(pDf.mean(axis=0)).T
            pAvgMRet.columns = [
                c + '-AverageMonthlyReturn' for c in pAvgMRet.columns
            ]

            # compute period annualzed return
            pAnnRet = pd.DataFrame(annual_return(pDf, period='M')).T
            pAnnRet[pAnnRet.columns.difference(
                pd.DataFrame(pDf).dropna(how='all', axis=1).columns)] = np.nan
            pAnnRet.columns = [
                c + '-AnnualizedReturn' for c in pAnnRet.columns
            ]

            # get the starting and ending date of each period
            if period[1] != regimeData.shape[0]:
                pDate = pd.DataFrame(
                    [
                        str(regimeData.index[period[0]].date()),
                        str(regimeData.index[period[1] - 1].date()),
                        (12 * (regimeData.index[period[1]].year -
                               regimeData.index[period[0]].year) +
                         (regimeData.index[period[1]].month -
                          regimeData.index[period[0]].month))
                    ],
                    index=['Start Date', 'End Date', 'Duration']).T
            else:
                pDate = pd.DataFrame(
                    [str(regimeData.index[period[0]].date()), 'Ongoing', None],
                    index=['Start Date', 'End Date', 'Duration']).T

            pStatsDf = pd.concat([pDate, pAvgMRet, pAnnRet, pCumRet],
                                 axis=1)  #pDfMean, pDfMedian, pDfMax, pDfMin

            cyclePeriodData.append(pDf)
            cyclePeriodStats.append(pStatsDf)
        

        # cycleDf consists of the date, duration, average monthly return, annualized return, and cumulative return of each period within a cycle
        if len(cyclePeriodData) == 0:
            continue
        cycleDf = pd.concat(cyclePeriodData, axis=0)
        cycleData.update({cycleStr: cycleDf})
        cycleStatsDf = pd.concat(cyclePeriodStats, axis=0)
        cycleStatsDf['Start Date'] = pd.to_datetime(cycleStatsDf['Start Date'])
        cycleStatsDf = cycleStatsDf.set_index('Start Date')
        cycleStats.update({cycleStr: cycleStatsDf})

    statsTableauDf = pd.concat([getCycleMetricStats(cycleStats, 'CumulativeReturn'), \
                                getCycleMetricStats(cycleStats, 'AverageMonthlyReturn'), \
                                getCycleMetricStats(cycleStats, 'AnnualizedReturn')], \
                               axis = 0)

    if hitRatio:
        logging.info('get absolute hit ratio...')
        hrList = {}
        for cycle, cycleDf in cycleData.items():
            hrList.update({cycle: hit_ratio(cycleDf)})
        hrDf = pd.DataFrame.from_dict(hrList, orient='index')
        hrTableau = pd.concat(
            [
                dataIO.df_to_tableau(hrDf, 'Mean', index_name='Cycle'),
                dataIO.df_to_tableau(hrDf, 'Median', index_name='Cycle'),
                #dataIO.df_to_tableau(hrDf, 'Max', index_name='Cycle'),
                #dataIO.df_to_tableau(hrDf, 'Min', index_name='Cycle')
            ],
            axis=0)
        hrTableau['Metric'] = 'Absolute Hit Ratio'

        logging.info('get relative hit ratio...')
        relHrList = {}
        for cycle, cycleDf in cycleData.items():
            ref_list = [
                assetList.loc[assetList.Description == c]
                ['Reference'].values[0] for c in cycleDf
            ]
            print(ref_list)
            refDf = cycleDf - (cycleDf[ref_list].values)
            relHrList.update({cycle: hit_ratio(refDf)})
        relHrDf = pd.DataFrame.from_dict(relHrList, orient='index')
        relHrTableau = pd.concat(
            [
                dataIO.df_to_tableau(relHrDf, 'Mean', index_name='Cycle'),
                dataIO.df_to_tableau(relHrDf, 'Median', index_name='Cycle'),
                #dataIO.df_to_tableau(relHrDf, 'Max', index_name='Cycle'),
                #dataIO.df_to_tableau(relHrDf, 'Min', index_name='Cycle')
            ],
            axis=0)
        relHrTableau['Metric'] = 'Relative Hit Ratio'

        statsTableauDf = pd.concat([statsTableauDf, hrTableau, relHrTableau],
                                   axis=0)

    # dictionary to map description to assets and tickers
    assetDict = {}
    for ind in list(set(list(assetList.Description))):
        assetDict.update(
            {ind: assetList.loc[assetList.Description == ind].Asset.values[0]})
    for cycle, cycleStr in regimeDict.items():
        assetDict.update({cycleStr: 'Cycle'})
    assetDict.update({'Duration': 'Cycle'})

    # compute the transmition probability
    regimeStrDf = pd.DataFrame(
        regimeData.Regime.apply(lambda x: regimeDict[x]))
    regimeStrDf.index.names = ['date']
    regimeMat = pd.crosstab(pd.Series(regimeStrDf.Regime.values[:-1]),
                            pd.Series(regimeStrDf.Regime.values[1:]))
    regimeMatDf = regimeMat.div(regimeMat.sum(axis=1), axis=0)

    statsTableauDf = pd.concat([
        statsTableauDf,
        dataIO.df_to_tableau(regimeMatDf, 'Probability', index_name='Cycle')
    ],
                               axis=0)
    statsTableauDf['Asset'] = statsTableauDf['Index'].apply(
        lambda x: assetDict[x] if x in assetDict.keys() else x)

    # asset price and return data
    assetPrice = prices.loc[prices.index.intersection(regimeData.index)]
    assetReturn = returns.loc[returns.index.intersection(regimeData.index)]
    rebasedReturn = ((1 + returns).cumprod()) * 100
    assetCumReturn = rebasedReturn.loc[rebasedReturn.index.intersection(
        regimeData.index)]

    # regime data
    regimeDf = pd.concat(regimeDf, axis=1)

    tickerDict = {}
    for ind in list(set(list(assetList.Description))):
        tickerDict.update(
            {ind: assetList.loc[assetList.Description == ind].Index.values[0]})
    for cycle, cycleStr in regimeDict.items():
        tickerDict.update({cycleStr: 'Cycle'})
    tickerDict.update({'Duration': 'Cycle'})

    regimeData['Duration'] = 0
    dataTableauDf = pd.concat([
        dataIO.df_to_tableau(assetPrice, 'Price'),
        dataIO.df_to_tableau(assetReturn, 'Return'),
        dataIO.df_to_tableau(assetCumReturn, 'RebasedReturn'),
        dataIO.df_to_tableau(
            regimeData.loc[regimeData.index.intersection(assetPrice.index)],
            'Score'),
        dataIO.df_to_tableau(
            regimeDf.loc[regimeDf.index.intersection(assetPrice.index)],
            'Cycle')
    ],
                              axis=0)
    dataTableauDf['Asset'] = dataTableauDf['Index'].apply(
        lambda x: assetDict[x] if x in assetDict.keys() else x)
    dataTableauDf['Ticker'] = dataTableauDf['Index'].apply(
        lambda x: tickerDict[x] if x in tickerDict.keys() else x)

    # get the current month info
    regSubDf = regimeStrDf.loc[regimeStrDf.Regime ==
                               regimeStrDf.iloc[-1].values[0]]
    regSubDfDelta = regSubDf.reset_index()
    if any((regSubDfDelta['date'] -
            regSubDfDelta['date'].shift(1)) > datetime.timedelta(days=31)):
        currRegStartD = regSubDfDelta.index[(
            regSubDfDelta['date'] -
            regSubDfDelta['date'].shift(1)) > datetime.timedelta(days=31)][-1]
    else:
        currRegStartD = regSubDfDelta.index.min()
    currRegMonthInd = regSubDfDelta['date'].index[-1] - currRegStartD + 1
    currRegMonthDf = pd.DataFrame(
        [regimeStrDf.iloc[-1].values[0], currRegMonthInd]).T
    currRegMonthDf.columns = ['Regime', 'MonthInd']
    currRegMonthDf['date'] = [str(regimeStrDf.index[-1].date())]
    if regimeIndex == 'Inflation':
        currRegMonthDf['Mean'] = np.round(
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].mean(), 2)
        currRegMonthDf['Std'] = np.round(
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].std(), 2)
        currRegMonthDf['Up'] = np.round(
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].mean() +
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].std(), 2)
        currRegMonthDf['Down'] = np.round(
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].mean() -
            regimeData.loc[regimeData.index.intersection(assetPrice.index),
                           'Score'].std(), 2)

    return {
        'data': dataTableauDf,
        'stats': statsTableauDf,
        'currInfo': currRegMonthDf
    }
"""
"""
def cycleFwdLookingStats(assetList,
                         returns,
                         regimeData,
                         regimeDict,
                         metric,
                         transitionMatFlag=False,
                         forwardMonth=range(1, 25)):
    metricDict = {'AverageMonthlyReturn':'The mean of monthly return of each period under the same regime', \
                  'CumulativeReturn':'The mean of cumulative return of each period under the same regime', \
                  'AnnualizedReturn':'The mean of annulized return of each period under the same regim', \
                  'Correlation':'The mean correlation between the return of the base asset and that of the selected asset'}
    logging.info(metric)
    cycleForwardStats = {}
    #regimeDf = []
    regimeStrDf = pd.DataFrame(
        regimeData.Regime.apply(lambda x: regimeDict[x]))
    # for each cycle
    for cycle, cycleStr in regimeDict.items():
        print(cycleStr + ': ' + str(cycle))
        cycleData = []
        rgmDfTmp = pd.DataFrame(regimeData.Regime)
        rgmDfTmp = (rgmDfTmp == cycle).astype(int)

        #get the starting and ending index of each period of the cycle
        nonzeroInd = find_nonzero_runs(rgmDfTmp.Regime.values)
        #rgmDfTmp.columns = [cycleStr]
        #regimeDf.append(rgmDfTmp)

        # cyclePeriods is the duration of each period in the cycle
        cyclePeriods = []
        for period in nonzeroInd:
            cyclePeriods.append(period[1] - period[0])

        # cycleRange is the maximum duration of periods of the cycle
        if len(cyclePeriods) == 0:
            continue
        cycleRange = max(cyclePeriods)

        # variables for cycle related stats
        cycleMonthRetMean = []
        cycleMonthRetMin = []
        cycleMonthRetMax = []
        cycleMonthRetMedian = []
        cycleMonthRetHit = []

        if transitionMatFlag:
            cycleMonthMat = []

        # from the 1st month to the maximum month of a cycle
        for cycleMonth in range(cycleRange):
            #cycleMonthDate.update({cycleMonth:[p[0]+cycleMonth for p in nonzeroInd if p[0]+cycleMonth <p[1]]})
            # the starting index of the nth month of a cycle in its original dataframe
            monthStartInd = [
                p[0] + cycleMonth for p in nonzeroInd
                if p[0] + cycleMonth < p[1]
            ]
            # monthIndRet is the return df for all forward m months return
            monthIndRet = []

            # for each period in the history
            for monthInd in monthStartInd:
                forwardRet = {}
                # the forward looking window
                for fM in forwardMonth:
                    if monthInd + fM >= returns.shape[0]:
                        continue
                    if metric == 'CumulativeReturn':
                        fMRet = returns[monthInd:monthInd + fM].sum(
                            axis=0, min_count=1)
                    elif metric == 'AverageMonthlyReturn':
                        fMRet = returns[monthInd:monthInd + fM].mean(axis=0)
                    elif metric == 'AnnualizedReturn':
                        fMRet = pd.DataFrame(
                            annual_return(returns[monthInd:monthInd + fM],
                                          period='M')).T
                        fMRet[fMRet.columns.difference(
                            pd.DataFrame(returns[period[0]:period[1]]).dropna(
                                how='all', axis=1).columns)] = np.nan
                        fMRet = fMRet.iloc[0]
                    forwardRet.update({fM: fMRet})
                forwardRetDf = pd.DataFrame.from_dict(forwardRet,
                                                      orient='index')
                monthIndRet.append(forwardRetDf)

            # compute the stats for the nth month of this cycle
            monthIndRetMean = pd.concat(monthIndRet).groupby(level=0).mean()
            monthIndRetMean['monthInd'] = cycleMonth + 1
            monthIndRetMean.index.names = ['forwardMonth']
            monthIndRetMean = monthIndRetMean.reset_index()
            monthIndRetMean = monthIndRetMean.set_index(
                ['monthInd', 'forwardMonth'])

            monthIndRetMin = pd.concat(monthIndRet).groupby(level=0).min()
            monthIndRetMin['monthInd'] = cycleMonth + 1
            monthIndRetMin.index.names = ['forwardMonth']
            monthIndRetMin = monthIndRetMin.reset_index()
            monthIndRetMin = monthIndRetMin.set_index(
                ['monthInd', 'forwardMonth'])

            monthIndRetMax = pd.concat(monthIndRet).groupby(level=0).max()
            monthIndRetMax['monthInd'] = cycleMonth + 1
            monthIndRetMax.index.names = ['forwardMonth']
            monthIndRetMax = monthIndRetMax.reset_index()
            monthIndRetMax = monthIndRetMax.set_index(
                ['monthInd', 'forwardMonth'])

            monthIndRetMedian = pd.concat(monthIndRet).groupby(
                level=0).median()
            monthIndRetMedian['monthInd'] = cycleMonth + 1
            monthIndRetMedian.index.names = ['forwardMonth']
            monthIndRetMedian = monthIndRetMedian.reset_index()
            monthIndRetMedian = monthIndRetMedian.set_index(
                ['monthInd', 'forwardMonth'])

            cycleMonthRetMean.append(monthIndRetMean)
            cycleMonthRetMedian.append(monthIndRetMedian)
            cycleMonthRetMin.append(monthIndRetMin)
            cycleMonthRetMax.append(monthIndRetMax)

            if transitionMatFlag:
                # for the transmition probability matrix
                monthIndMat = []
                for fM in forwardMonth:
                    #if monthInd + fM >= returns.shape[0]:
                    #    continue
                    regimeMat = pd.crosstab(pd.Series(regimeStrDf.Regime.values[[m for m in monthStartInd if m+fM < regimeStrDf.shape[0]]]), \
                                            pd.Series(regimeStrDf.Regime.values[[m+fM for m in monthStartInd if m+fM < regimeStrDf.shape[0]]]))
                    regimeMatDf = regimeMat.div(regimeMat.sum(axis=1), axis=0)
                    if regimeMatDf.empty:
                        continue
                    regimeMatDf.index = [fM]
                    monthIndMat.append(regimeMatDf)
                if not monthIndMat:
                    continue
                monthIndMat = pd.concat(monthIndMat, axis=0)
                monthIndMat['monthInd'] = cycleMonth + 1
                monthIndMat['cyclePeriods'] = len(nonzeroInd)
                monthIndMat['monthIndOccurance'] = len(monthStartInd)
                monthIndMat.index.names = ['forwardMonth']
                monthIndMat = monthIndMat.reset_index()
                monthIndMat = monthIndMat.set_index(
                    ['monthInd', 'forwardMonth'])
                cycleMonthMat.append(monthIndMat)

        if cycleStr not in cycleForwardStats.keys():
            cycleForwardStats.update(
                {cycleStr: {
                    'Mean': pd.concat(cycleMonthRetMean, axis=0)
                }})
        else:
            cycleForwardStats[cycleStr].update(
                {'Mean': pd.concat(cycleMonthRetMean, axis=0)})
        cycleForwardStats[cycleStr].update(
            {'Median': pd.concat(cycleMonthRetMedian, axis=0)})
        cycleForwardStats[cycleStr].update(
            {'Max': pd.concat(cycleMonthRetMax, axis=0)})
        cycleForwardStats[cycleStr].update(
            {'Min': pd.concat(cycleMonthRetMin, axis=0)})
        if transitionMatFlag:
            cycleForwardStats[cycleStr].update(
                {'RegimeTransitionProb': pd.concat(cycleMonthMat, axis=0)})

    # convert to tableau data
    logging.info('convert to tableau readable dataformat...')
    fwdStats = []
    for c in cycleForwardStats.keys():
        for kpi in cycleForwardStats[c].keys():
            #print(kpi)
            for i in list(
                    set(cycleForwardStats[c][kpi].index.get_level_values(
                        'monthInd'))):
                tmp = dataIO.df_to_tableau(cycleForwardStats[c][kpi].loc[cycleForwardStats[c][kpi].index.get_level_values('monthInd') == i].reset_index().set_index('forwardMonth').drop('monthInd', axis=1),\
                                             kpi, index_name='ForwardMonth')
                tmp['Cycle'] = c
                tmp['MonthInd'] = i
                fwdStats.append(tmp)
    fwdDf = pd.concat(fwdStats, axis=0)
    fwdDf['Metric'] = metric
    fwdDf['Description'] = metricDict[metric]
    # dictionary to map description to assets and tickers
    assetDict = {}
    for ind in list(set(list(assetList.Description))):
        assetDict.update(
            {ind: assetList.loc[assetList.Description == ind].Asset.values[0]})
    for cycle, cycleStr in regimeDict.items():
        assetDict.update({cycleStr: 'Cycle'})
    assetDict.update({'Duration': 'Cycle'})
    
    tickerDict = {}
    for ind in list(set(list(assetList.Description))):
        tickerDict.update({ind:assetList.loc[assetList.Description == ind].Index.values[0]})
    for cycle, cycleStr in regimeDict.items():
        tickerDict.update({cycleStr:'Cycle'})
    tickerDict.update({'Duration':'Cycle'})
    
    fwdDf['Asset'] = fwdDf['Index'].apply(lambda x: assetDict[x]
                                          if x in assetDict.keys() else x)
    return fwdDf
"""
"""
def cycleRelativeHistoricalStats(assetList, prices, returns, regimeData,
                                 regimeDict, regimeIndex, relativeGroups):
    relativeCycleData = {}
    relativeCycleStats = {}
    for ref, indList in relativeGroups.items():
        logging.info(ref)
        relP, relR = get_relative_data(prices, returns, indList['index'],
                                       indList['reference'])
        relAssetList = copy.deepcopy(assetList)
        relAssetList = relAssetList.loc[relAssetList.Description.isin(
            indList['index'])]
        display(relAssetList)
        relAssetList['Reference'] = indList['reference'][0]
        display(relAssetList)
        relResults = cycleHistoricalStats(assetList,
                                          relP,
                                          relR,
                                          regimeData,
                                          regimeDict,
                                          regimeIndex,
                                          hitRatio=True)
        relData = relResults['data']
        relData['Reference'] = indList['reference'][0]
        relStats = relResults['stats']
        relStats['Reference'] = indList['reference'][0]
        relativeCycleData.update({ref: relResults['data']})
        relativeCycleStats.update({ref: relResults['stats']})

    relativeCycleDataTableau = []
    relativeCycleStatsTableau = []
    for k, kDf in relativeCycleData.items():
        kDf['Group'] = k
        relativeCycleDataTableau.append(kDf)

    for k, kDf in relativeCycleStats.items():
        kDf['Group'] = k
        relativeCycleStatsTableau.append(kDf)

    assetDict = {}
    for ind in list(set(list(assetList.Description))):
        assetDict.update(
            {ind: assetList.loc[assetList.Description == ind].Asset.values[0]})
    for cycle, cycleStr in regimeDict.items():
        assetDict.update({cycleStr: 'Cycle'})
    assetDict.update({'Duration': 'Cycle'})

    relativeCycleDataTableauDf = pd.concat(relativeCycleDataTableau, axis=0)
    relativeCycleStatsTableauDf = pd.concat(relativeCycleStatsTableau, axis=0)
    relativeCycleStatsTableauDf['Asset'] = relativeCycleStatsTableauDf[
        'Index'].apply(lambda x: assetDict[x] if x in assetDict.keys() else x)
    return {
        'data': relativeCycleDataTableauDf,
        'stats': relativeCycleStatsTableauDf
    }


def getCorrelation(returns, aoiList=None, reference=None):
    if reference is None:
        reference = list(returns.columns)
    retCorrelation = {}
    for ref in reference:
        if aoiList is None:
            aoiList = [c for c in returns.columns if c != ref]
        refCorr = pd.DataFrame(
            returns[aoiList].apply(lambda x: x.corr(returns[ref]))).T
        refCorr.index = [ref]
        retCorrelation.update({ref: refCorr})
    return retCorrelation


def indexCorr2Tableau(cycleStr, ind, indCorr, metric):
    if metric == 'Mean':
        indCorrMetric = pd.DataFrame(pd.concat(indCorr, axis=0).mean(axis=0))
    elif metric == 'Max':
        indCorrMetric = pd.DataFrame(pd.concat(indCorr, axis=0).max(axis=0))
    elif metric == 'Min':
        indCorrMetric = pd.DataFrame(pd.concat(indCorr, axis=0).min(axis=0))
    elif metric == 'Median':
        indCorrMetric = pd.DataFrame(pd.concat(indCorr, axis=0).median(axis=0))

    indCorrMetric.index.names = ['Index']
    indCorrMetric.columns = ['Value']
    indCorrMetric['Reference'] = ind
    indCorrMetric['KPI'] = metric
    indCorrMetric['Cycle'] = cycleStr
    indCorrMetric = indCorrMetric.reset_index().set_index('Cycle')
    return indCorrMetric


def cycleCorrelationHistoricalStats(returns, regimeData, regimeDict):
    cycleCorrStats = []
    for cycle, cycleStr in regimeDict.items():
        print(cycleStr + ': ' + str(cycle))
        cycleData = {}
        # 1. get the indexes of each period within each cycle/regime
        rgmDfTmp = pd.DataFrame(regimeData.Regime)
        rgmDfTmp = (rgmDfTmp == cycle).astype(int)
        # nonzeroInd: starting and ending index of each period
        nonzeroInd = find_nonzero_runs(rgmDfTmp.Regime.values)

        # get the return data for each period
        for period in nonzeroInd:
            if period[0] < returns.shape[0]:
                pDf = returns[period[0]:period[1]]
            elif period[0] == returns.shape[0]:
                pDf = pd.DataFrame(returns.iloc[-1]).T
            else:
                continue

            # compute period cumulative return
            pRetCorr = getCorrelation(pDf)

            # get the starting and ending date of each period
            if period[1] != regimeData.shape[0]:
                pDate = pd.DataFrame(
                    [
                        str(regimeData.index[period[0]].date()),
                        str(regimeData.index[period[1] - 1].date()),
                        (12 * (regimeData.index[period[1]].year -
                               regimeData.index[period[0]].year) +
                         (regimeData.index[period[1]].month -
                          regimeData.index[period[0]].month))
                    ],
                    index=['Start Date', 'End Date', 'Duration']).T
            else:
                pDate = pd.DataFrame(
                    [str(regimeData.index[period[0]].date()), 'Ongoing', None],
                    index=['Start Date', 'End Date', 'Duration']).T

            for k, kDf in pRetCorr.items():
                pDf = pd.concat([pDate, kDf],
                                axis=1)  #pDfMean, pDfMedian, pDfMax, pDfMin
                if k not in cycleData.keys():
                    cycleData.update({k: [pDf]})
                else:
                    cycleData[k].append(pDf)

        cycleStats = []
        for ind, indCorr in cycleData.items():
            indCorrStats = []
            indCorrStats.append(
                indexCorr2Tableau(cycleStr, ind, indCorr, 'Mean'))
            indCorrStats.append(
                indexCorr2Tableau(cycleStr, ind, indCorr, 'Max'))
            indCorrStats.append(
                indexCorr2Tableau(cycleStr, ind, indCorr, 'Min'))
            indCorrStats.append(
                indexCorr2Tableau(cycleStr, ind, indCorr, 'Median'))
            indCorrDf = pd.concat(indCorrStats, axis=0)
            cycleStats.append(indCorrDf)
        if len(cycleStats) == 0:
            continue
        cycleStatsDf = pd.concat(cycleStats, axis=0)
        cycleCorrStats.append(cycleStatsDf)
    cycleCorrStatsDf = pd.concat(cycleCorrStats, axis=0)
    cycleCorrStatsDf['Metric'] = 'Correlation'
    cycleCorrStatsDf[
        'Description'] = 'The mean correlation between the return of the base asset and that of the selected asset'
    return cycleCorrStatsDf
"""
"""
def get_regime_perf(assetList,
                    prices,
                    returns,
                    regimeData,
                    relativeGroups=None,
                    filename=None,
                    save_file=True):
    regimeStats = {}
    fwdStats = []
    #relativeResults = {}
    for rgmStr, rgmData in regimeData.items():
        logging.info(rgmStr)
        regimeDf = rgmData['data']
        regimeDict = rgmData['dict']
        forwardLooking = rgmData['forwardLooking']
        assetPrices = prices.loc[prices.index.intersection(regimeDf.index)]
        assetReturns = returns.loc[returns.index.intersection(regimeDf.index)]
        #display(assetPrices)
        #display(assetReturns)

        # historical stats for each regime
        logging.info('historical stats of assets in each regime')
        histCycleResults = cycleHistoricalStats(assetList, assetPrices,
                                                assetReturns, regimeDf,
                                                regimeDict, rgmStr)

        #histCycleResults['stats'] = pd.concat([histCycleResults['stats'], histCycleCorrDf], axis=0)
        for k, kDf in histCycleResults.items():
            kDf['MacroIndex'] = rgmStr
            kDf['Perf'] = 'Absolute'
            #histCycleResults[k] = kDf
            if k not in regimeStats.keys():
                regimeStats.update({k: kDf})
            else:
                regimeStats[k] = pd.concat([regimeStats[k], kDf], axis=0)

        # historical correlation for each group
        if relativeGroups is None:
            relativeGroups = {'CountryEquity':{'reference':['Global Equities'], 'index':['US', 'Europe', 'UK', 'Japan', 'China', 'Emerging Markets', 'LatAm']}, \
                        'SectorEquity':{'reference':['Global Equities'], 'index':['Consumer Discretionary', 'Consumer Staples', 'Communication Services', 'Financials', 'Health Care', 'Industrials', 'IT', 'Materials', 'Energy', 'Utilities', 'Real Estate']}, \
                        'StylesEquity':{'reference':['Global Value'], 'index':['Global Growth', 'Global Quality', 'Global Min Volatility', 'Global Momentum']}, \
                        'USEquity':{'reference':['US'], 'index':['US Cyclicals', 'US Defensives']}, \
                        'SovereignBond':{'reference':['Global Sovereign Bonds'], 'index':['US 2Y Treasuries', 'US 10Y Treasuries', 'US Treasuries - Broad', 'Euro Sovereign Bond', 'UK Government Bond', 'Swiss Government Bond', 'China Local Bonds', 'EM Hard Currency Debt']}, \
                        'CorporateBond':{'reference':['USD Corporates'], 'index':['US Corporate Financial Bond', 'US Corporate Non-Financial Bond', 'Euro Corporate Financial Bond', 'Euro Corporate Non-Financial Bond', 'Swiss Corporate Bond']}, \
                        'InflationLinkedBond':{'reference':['Global Inflation Linked bonds ($ hdg)'], 'index':['US Inflation Linked', 'Euro Inflation Linked', 'UK Inflation Linked']}, \
                        'Commodities':{'reference':['Broad Commodities'], 'index':['Gold', 'Oil', 'Industrial Metals']}, \
                        'HedgeFund':{'reference':['Hedge Fund - Broad'], 'index':['HF Event Driven', 'HF Eq L/S', 'HF Relative Value', 'HF Macro']}, \
                        'Infrastructure':{'reference':['Global Infrastructure'], 'index':['REITs US', 'REITs EU', 'Real Estate']}}

        # get historical group stats
        
        logging.info('historical group stats in each regime')
        relativeCycleResults = cycleRelativeHistoricalStats(assetList, assetPrices, assetReturns, regimeDf, regimeDict, rgmStr, relativeGroups)
        for k, kDf in relativeCycleResults.items():
            kDf['MacroIndex'] = rgmStr
            kDf['Perf'] = 'Relative'
            #display(kDf)
            #relativeResults[k] = kDf
            if k not in regimeStats.keys():
                regimeStats.update({k:kDf})
            else:
                regimeStats[k] = pd.concat([regimeStats[k],kDf], axis=0)
        
    if filename:
        fn = filename + '_Perf_Stats_tableau.xlsx'
    else:
        fn = 'MultiAsset_Perf_Stats_tableau.xlsx'

    if save_file:
        dataIO.export_to_excel(
            regimeStats,
            Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI') / fn,
            {'indexName': None})
        #dataIO.dataIO(relativeResults, Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI')/('MultiAsset_relativePerf_Stats_tableau.xlsx'), {'indexName':None})

    return regimeStats
"""
"""
def get_taa_group_stats(all_rgm_stats: pd.DataFrame, taa_groups: dict):
    all_rgm_stats['TAA'] = 'All'
    taa_group_list = []
    for group, group_list in taa_groups.items():
        group_assets = group_list['reference'] + group_list['index']
        group_stats = all_rgm_stats.loc[all_rgm_stats.Index.isin(group_assets)]
        group_stats['TAA'] = group
        taa_group_list.append(group_stats)
    return pd.concat(
        [all_rgm_stats, pd.concat(taa_group_list, axis=0)], axis=0)
"""
"""
def getRegimeStats(assetList,
                   prices,
                   returns,
                   regimeData,
                   taaGroups,
                   relativeGroups=None,
                   filename=None,
                   save_file=True):
    regimeStats = {}
    fwdStats = []
    #relativeResults = {}
    for rgmStr, rgmData in regimeData.items():
        logging.info(rgmStr)
        regimeDf = rgmData['data']
        regimeDict = rgmData['dict']
        forwardLooking = rgmData['forwardLooking']
        assetPrices = prices.loc[prices.index.intersection(regimeDf.index)]
        assetReturns = returns.loc[returns.index.intersection(regimeDf.index)]

        # historical stats for each regime
        logging.info('historical stats of assets in each regime')
        histCycleResults = cycleHistoricalStats(assetList, assetPrices,
                                                assetReturns, regimeDf,
                                                regimeDict, rgmStr)

        histCycleResults['stats'] = get_taa_group_stats(
            histCycleResults['stats'], taaGroups)

        logging.info('historical correlation between assets in each regime')
        histCycleCorrDf = cycleCorrelationHistoricalStats(
            assetReturns, regimeDf, regimeDict)
        histCycleResults['stats'] = pd.concat(
            [histCycleResults['stats'], histCycleCorrDf], axis=0)
        for k, kDf in histCycleResults.items():
            kDf['MacroIndex'] = rgmStr
            kDf['Perf'] = 'Absolute'
            #histCycleResults[k] = kDf
            if k not in regimeStats.keys():
                regimeStats.update({k: kDf})
            else:
                regimeStats[k] = pd.concat([regimeStats[k], kDf], axis=0)

        # forward looking stats for each regime
        if forwardLooking:
            logging.info('forward looking stats of assets in each regime')
            fwdAvgMRetStats = cycleFwdLookingStats(assetList,
                                                   assetReturns,
                                                   regimeDf,
                                                   regimeDict,
                                                   'AverageMonthlyReturn',
                                                   transitionMatFlag=True)
            fwdAnnRetStats = cycleFwdLookingStats(assetList,
                                                  assetReturns,
                                                  regimeDf,
                                                  regimeDict,
                                                  'AnnualizedReturn',
                                                  transitionMatFlag=False)
            fwdCumRetStats = cycleFwdLookingStats(assetList,
                                                  assetReturns,
                                                  regimeDf,
                                                  regimeDict,
                                                  'CumulativeReturn',
                                                  transitionMatFlag=False)
            fwdOverallStats = pd.concat(
                [fwdAvgMRetStats, fwdAnnRetStats, fwdCumRetStats], axis=0)
            fwdOverallStats['MacroIndex'] = rgmStr
            fwdStats.append(fwdOverallStats)
            #fwdOverallStats.to_csv(Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI')/('MultiAsset_fwdStats_'+rgmStr+'_tableau.csv'))
            #logging.info(Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI')/('MultiAsset_fwdStats_'+rgmStr+'_tableau.csv'))

        # historical correlation for each group
        if relativeGroups is None:
            relativeGroups = {'CountryEquity':{'reference':['Global Equities'], 'index':['US', 'Europe', 'UK', 'Japan', 'China', 'Emerging Markets', 'LatAm']}, \
                        'SectorEquity':{'reference':['Global Equities'], 'index':['Consumer Discretionary', 'Consumer Staples', 'Communication Services', 'Financials', 'Health Care', 'Industrials', 'IT', 'Materials', 'Energy', 'Utilities', 'Real Estate']}, \
                        'StylesEquity':{'reference':['Global Value'], 'index':['Global Growth', 'Global Quality', 'Global Min Volatility', 'Global Momentum']}, \
                        'USEquity':{'reference':['US'], 'index':['US Cyclicals', 'US Defensives']}, \
                        'SovereignBond':{'reference':['Global Sovereign Bonds'], 'index':['US 2Y Treasuries', 'US 10Y Treasuries', 'US Treasuries - Broad', 'Euro Sovereign Bond', 'UK Government Bond', 'Swiss Government Bond', 'China Local Bonds', 'EM Hard Currency Debt']}, \
                        'CorporateBond':{'reference':['USD Corporates'], 'index':['US Corporate Financial Bond', 'US Corporate Non-Financial Bond', 'Euro Corporate Financial Bond', 'Euro Corporate Non-Financial Bond', 'Swiss Corporate Bond']}, \
                        'InflationLinkedBond':{'reference':['Global Inflation Linked bonds ($ hdg)'], 'index':['US Inflation Linked', 'Euro Inflation Linked', 'UK Inflation Linked']}, \
                        'Commodities':{'reference':['Broad Commodities'], 'index':['Gold', 'Oil', 'Industrial Metals']}, \
                        'HedgeFund':{'reference':['Hedge Fund - Broad'], 'index':['HF Event Driven', 'HF Eq L/S', 'HF Relative Value', 'HF Macro']}, \
                        'Infrastructure':{'reference':['Global Infrastructure'], 'index':['REITs US', 'REITs EU', 'Real Estate']}}

        # get historical group stats
        logging.info('historical group stats in each regime')
        relativeCycleResults = cycleRelativeHistoricalStats(
            assetList, assetPrices, assetReturns, regimeDf, regimeDict, rgmStr,
            relativeGroups)
        for k, kDf in relativeCycleResults.items():
            kDf['MacroIndex'] = rgmStr
            kDf['Perf'] = 'Relative'
            #display(kDf)
            #relativeResults[k] = kDf
            if k not in regimeStats.keys():
                regimeStats.update({k: kDf})
            else:
                regimeStats[k] = pd.concat([regimeStats[k], kDf], axis=0)

    if len(fwdStats) != 0:
        fwdStatsDf = pd.concat(fwdStats, axis=0)
        if filename:
            fn = filename + '_Perf_fwdStats_tableau.csv'
        else:
            fn = 'MultiAsset_Perf_fwdStats_tableau.csv'
        fwdStatsDf.to_csv(
            Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI') / fn)
        logging.info(
            Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI') / fn)

    if filename:
        fn = filename + '_Perf_Stats_tableau.xlsx'
    else:
        fn = 'MultiAsset_Perf_Stats_tableau.xlsx'

    if save_file:
        dataIO.export_to_excel(
            regimeStats,
            Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI') / fn,
            {'index_name': None})
        #dataIO.dataIO(relativeResults, Path(r'\\merlin\lib_isg\28.Alternative Data\Projects\WEI')/('MultiAsset_relativePerf_Stats_tableau.xlsx'), {'indexName':None})

    return regimeStats
"""
"""
def cycleHistoricalData(assetList, prices, returns, regimeData, regimeDict):
    cycleStats = {}
    cycleData = {}
    regimeDf = []
    for cycle, cycleStr in regimeDict.items():
        logging.info(cycleStr + ': ' + str(cycle))
        cyclePeriodData = []
        cyclePeriodStats = []
        # 1. get the indexes of each period within each cycle/regime
        rgmDfTmp = pd.DataFrame(regimeData.Regime)
        rgmDfTmp = (rgmDfTmp == cycle).astype(int)
        # nonzeroInd: starting and ending index of each period
        nonzeroInd = find_nonzero_runs(rgmDfTmp.Regime.values)
        rgmDfTmp.columns = [cycleStr]
        # append the indexes of this cycle
        regimeDf.append(rgmDfTmp)
        # get the return data for each period
        for period in nonzeroInd:
            # compute period cumulative return
            if period[0] < returns.shape[0]:
                pDf = returns[period[0]:period[1]]
            elif period[0] == returns.shape[0]:
                pDf = pd.DataFrame(returns.iloc[-1]).T
            else:
                continue

            # compute period average monthly return
            pAvgMRet = pd.DataFrame(pDf.mean(axis=0)).T
            pAvgMRet.columns = [
                c + '-AverageMonthlyReturn' for c in pAvgMRet.columns
            ]

            # get the starting and ending date of each period
            if period[1] != regimeData.shape[0]:
                pDate = pd.DataFrame(
                    [
                        str(regimeData.index[period[0]].date()),
                        str(regimeData.index[period[1] - 1].date()),
                        (12 * (regimeData.index[period[1]].year -
                               regimeData.index[period[0]].year) +
                         (regimeData.index[period[1]].month -
                          regimeData.index[period[0]].month))
                    ],
                    index=['Start Date', 'End Date', 'Duration']).T
            else:
                pDate = pd.DataFrame(
                    [str(regimeData.index[period[0]].date()), 'Ongoing', None],
                    index=['Start Date', 'End Date', 'Duration']).T

            pStatsDf = pd.concat([pDate, pAvgMRet],
                                 axis=1)  #pDfMean, pDfMedian, pDfMax, pDfMin
            cyclePeriodData.append(pDf)
            cyclePeriodStats.append(pStatsDf)

        # cycleDf consists of the date, duration, average monthly return, annualized return, and cumulative return of each period within a cycle
        if len(cyclePeriodData) == 0:
            continue
        cycleDf = pd.concat(cyclePeriodData, axis=0)
        cycleStatsDf = pd.concat(cyclePeriodStats, axis=0)
        cycleStatsDf['Start Date'] = pd.to_datetime(cycleStatsDf['Start Date'])
        cycleStatsDf = cycleStatsDf.set_index('Start Date')
        cycleStats.update({cycleStr: cycleStatsDf})
    return cycleStats
"""
