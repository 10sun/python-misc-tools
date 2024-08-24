'''
Author: J , jwsun1987@gmail.com
Date: 2024-04-17 21:49:14
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
import numpy as np
from pandas.tseries.offsets import BMonthBegin, BMonthEnd
from datetime import datetime as dt
import dateutil
from copy import deepcopy
from common.dates import *


def get_dd_dates(data, dates, m=36, get_max_date=False, stats=True):  # dd15 = None,
    dd_info = {}
    dd_duration_info = {}
    for d in dates:
        if pd.to_datetime(d) < data.index.min() or pd.to_datetime(d) > data.index.max():
            continue
        print(d)
        dd_data = data.loc[
            (data.index >= d)
            & (
                data.index
                < pd.to_datetime(d) + dateutil.relativedelta.relativedelta(months=m)
            )
        ]
        if dd_data.empty:
            continue
        dd_min = dd_data.min()
        dd_date = dd_data.idxmin()
        dd_date_str = str(pd.to_datetime(dd_date.values[0]).date())
        dd_peak = dd_data.loc[dd_data.index <= dd_date_str].max()
        peak_date = dd_data.loc[dd_data.index <= dd_date_str].idxmax()
        max_dd = ((dd_min - dd_peak) / dd_peak).values[0]
        dd_duration = ((dd_date - peak_date) / np.timedelta64(1, "M")).values[0]

        if get_max_date:
            dd_max_date = (
                data.loc[
                    (data.index >= d)
                    & (
                        data.index
                        < pd.to_datetime(d)
                        + dateutil.relativedelta.relativedelta(months=12)
                    )
                ]
                .idxmax()
                .values[0]
            )
        else:
            dd_max_date = str(pd.to_datetime(peak_date.values[0]).date())

        recession_d = dates[d]
        if recession_d is None:
            recession_start_month = None
            recession_end_month = None
            recession_duration = None
            peak_to_recession = None
            recession_to_trough = None
            peak_to_recession_dd = None
            trough_to_end = None
            trough_to_end_ret = None
            peak_to_end = None
        else:
            recession_start = pd.to_datetime(recession_d[0])
            recession_start_month = (
                str(recession_start.year) + "-" + str(recession_start.month).zfill(2)
            )
            recession_end = pd.to_datetime(recession_d[1])
            recession_end_month = (
                str(recession_end.year) + "-" + str(recession_end.month).zfill(2)
            )
            recession_duration = (recession_end.year - recession_start.year) * 12 + (
                recession_end.month - recession_start.month
            )
            peak_to_recession = (recession_start.year - pd.to_datetime(d).year) * 12 + (
                recession_start.month - pd.to_datetime(d).month
            )
            recession_to_trough = (
                pd.to_datetime(dd_date.values[0]).year - recession_start.year
            ) * 12 + (pd.to_datetime(dd_date.values[0]).month - recession_start.month)
            peak_to_recession_dd = (
                data.loc[data.index == recession_start].values[0][0]
                - data.loc[data.index == d].values[0][0]
            ) / data.loc[data.index == d].values[0][0]
            trough_to_end = (
                recession_end.year - pd.to_datetime(dd_date.values[0]).year
            ) * 12 + (recession_end.month - pd.to_datetime(dd_date.values[0]).month)
            trough_to_end_ret = (
                data.loc[data.index == recession_end].values[0][0]
                - data.loc[data.index == dd_date.values[0]].values[0][0]
            ) / data.loc[data.index == dd_date.values[0]].values[0][0]
            peak_to_end = (recession_end.year - pd.to_datetime(d).year) * 12 + (
                recession_end.month - pd.to_datetime(d).month
            )

        dd_recovery = data.loc[data.index >= d]
        if dd_recovery[dd_recovery.gt(dd_recovery.values[0])].dropna().empty:
            dd_recovery_date_str = None
            dd_recovery_duration = None
        else:
            dd_recovery_date = (
                dd_recovery[dd_recovery.gt(dd_recovery.values[0])].dropna().index[0]
            )
            dd_recovery_date_str = str(pd.to_datetime(dd_recovery_date).date())
            dd_recovery_duration = (
                dd_recovery_date - dd_date.values[0]
            ) / np.timedelta64(1, "M")

        """
        if dd15 is not None:
            dd_15 = dd15.get(d)
        else:
            dd_15 = pd.to_datetime(d)+dateutil.relativedelta.relativedelta(months=3)
            
        dd_15_duration = (pd.to_datetime(dd_15) - pd.to_datetime(d))
        dd_15_trough = (pd.to_datetime(dd_date.values[0]) - pd.to_datetime(dd_15))

        #'Date at 15% DD': dd_15, 'Peak to 15% DD Duration': dd_15_duration, '15% DD to Trough Duration': dd_15_trough,  
        """
        dd_duration_info.update(
            {
                d: {
                    "Peak Date": dd_max_date,
                    "Recession Start": recession_start_month,
                    "Date at Max DD": dd_date_str,
                    "Recession End": recession_end_month,
                    "Date to Previous High": dd_recovery_date_str,
                    "Peak to Trough Duration (M)": dd_duration,
                    "Peak to Recession (M)": peak_to_recession,
                    "Recession Start to Trough (M)": recession_to_trough,
                    "Trough to Recession End (M)": trough_to_end,
                    "Trough to Previous High (M)": dd_recovery_duration,
                    "Peak to Recession End (M)": peak_to_end,
                }
            }
        )
        dd_info.update(
            {
                d: {
                    "Peak to Recession Start DD": peak_to_recession_dd,
                    "Max DD": max_dd,
                    "Trough to Recession End Return": trough_to_end_ret,
                }
            }
        )
    dd_info = pd.DataFrame.from_dict(dd_info, orient="index")
    dd_info.index.names = ["Market Peak Date"]
    dd_info_mean = pd.DataFrame(dd_info.mean(axis=0).rename("Mean")).T
    dd_info_median = pd.DataFrame(dd_info.median(axis=0).rename("Median")).T

    dd_duration_info = pd.DataFrame.from_dict(dd_duration_info, orient="index")
    dd_duration_info.index.names = ["Market Peak Date"]
    dd_duration_info_mean = pd.DataFrame(
        dd_duration_info.mean(axis=0, numeric_only=True).rename("Mean")
    ).T
    dd_duration_info_median = pd.DataFrame(
        dd_duration_info.median(axis=0, numeric_only=True).rename("Median")
    ).T

    if stats:
        dd_summary = pd.concat(
            [
                pd.concat(
                    [
                        pd.concat(
                            [
                                dd_duration_info,
                                dd_duration_info_mean,
                                dd_duration_info_median,
                            ],
                            axis=0,
                        )
                    ],
                    keys=["Duration"],
                    axis=1,
                ),
                pd.concat(
                    [pd.concat([dd_info, dd_info_mean, dd_info_median], axis=0)],
                    keys=["Price"],
                    axis=1,
                ),
            ],
            axis=1,
        )
        return dd_summary
    else:
        return pd.concat(
            [
                pd.concat([dd_duration_info], keys=["Duration"], axis=1),
                pd.concat([dd_info], keys=["Price"], axis=1),
            ],
            axis=1,
        )


def get_changes(data, dd_info, pct=True):
    dd = dd_info.copy()
    stats_summary = []
    for recession in list(dict.fromkeys(dd_info.index.get_level_values(0))):
        data_dd_stats = {}
        dd_tmp = dd.loc[dd.index.get_level_values(0) == recession]
        # print(dd_tmp.index.get_level_values(1).tolist())
        for d in dd_tmp.index.get_level_values(1).tolist():
            if d in ["Mean", "Median"]:
                continue
            if (
                pd.to_datetime(d) < data.index.min()
                or pd.to_datetime(d) > data.index.max()
            ):
                continue
            peak_date = dd_tmp.loc[
                dd_tmp.index.get_level_values(1) == d,
                ("Duration", "Peak Date"),
            ].values[0]
            print(d + ": " + peak_date)
            recession_start = BMonthEnd().rollforward(
                dd_tmp.loc[
                    dd_tmp.index.get_level_values(1) == d,
                    ("Duration", "Recession Start"),
                ].values[0]
            )
            trough_date = dd_tmp.loc[
                dd_tmp.index.get_level_values(1) == d,
                ("Duration", "Date at Max DD"),
            ].values[0]
            recession_end = BMonthEnd().rollforward(
                dd_tmp.loc[
                    dd_tmp.index.get_level_values(1) == d,
                    ("Duration", "Recession End"),
                ].values[0]
            )

            if pct:
                data_peak_to_trough = (
                    data.loc[data.index <= trough_date].iloc[-1]
                    - data.loc[data.index <= peak_date].iloc[-1]
                ) / data.loc[data.index <= peak_date].iloc[-1]
                data_dd_stats.update(
                    {d: {"Peak to Trough (%)": data_peak_to_trough.values[0]}}
                )
            else:
                data_peak_to_trough = (
                    data.loc[data.index <= trough_date].iloc[-1]
                    - data.loc[data.index <= peak_date].iloc[-1]
                )
                data_dd_stats.update(
                    {d: {"Peak to Trough (diff)": data_peak_to_trough.values[0]}}
                )

            if isinstance(recession_start, pd.Timestamp):
                if pct:
                    data_peak_to_recession_start = (
                        data.loc[data.index <= recession_start].iloc[-1]
                        - data.loc[data.index <= peak_date].iloc[-1]
                    ) / data.loc[data.index <= peak_date].iloc[-1]
                    data_trough_to_recession_end = (
                        data.loc[data.index <= recession_end].iloc[-1]
                        - data.loc[data.index <= trough_date].iloc[-1]
                    ) / data.loc[data.index <= trough_date].iloc[-1]
                    data_dd_stats[d] = {
                        **data_dd_stats[d],
                        **{
                            "Peak to Recession Start (%)": data_peak_to_recession_start.values[
                                0
                            ],
                            "Trough to Recession End (%)": data_trough_to_recession_end.values[
                                0
                            ],
                        },
                    }
                else:
                    data_peak_to_recession_start = (
                        data.loc[data.index <= recession_start].iloc[-1]
                        - data.loc[data.index <= peak_date].iloc[-1]
                    )
                    data_trough_to_recession_end = (
                        data.loc[data.index <= recession_end].iloc[-1]
                        - data.loc[data.index <= trough_date].iloc[-1]
                    )
                    data_dd_stats[d] = {
                        **data_dd_stats[d],
                        **{
                            "Peak to Recession Start (diff)": data_peak_to_recession_start.values[
                                0
                            ],
                            "Trough to Recession End (diff)": data_trough_to_recession_end.values[
                                0
                            ],
                        },
                    }
        data_dd_stats = pd.DataFrame.from_dict(data_dd_stats, orient="index")
        data_dd_stats_mean = pd.DataFrame(data_dd_stats.mean(axis=0).rename("Mean")).T
        data_dd_stats_median = pd.DataFrame(
            data_dd_stats.median(axis=0).rename("Median")
        ).T
        stats_summary.append(
            pd.concat(
                [
                    pd.concat(
                        [data_dd_stats, data_dd_stats_mean, data_dd_stats_median],
                        axis=0,
                    )
                ],
                keys=[recession],
                axis=0,
            )
        )
    return pd.concat(stats_summary, axis=0)


def get_data_at_dates(data, dd_info):
    dd = dd_info.copy()
    stats_summary = []
    for recession in list(dict.fromkeys(dd_info.index.get_level_values(0))):
        data_dd_stats = {}
        dd_tmp = dd.loc[dd.index.get_level_values(0) == recession]
        for d in dd_tmp.index.get_level_values(1).tolist():
            if d in ["Mean", "Median"]:
                continue
            if (
                pd.to_datetime(d) < data.index.min()
                or pd.to_datetime(d) > data.index.max()
            ):
                continue
            peak_date = d
            recession_start = BMonthEnd().rollforward(
                dd_tmp.loc[
                    dd_tmp.index.get_level_values(1) == peak_date,
                    ("Duration", "Recession Start"),
                ].values[0]
            )
            trough_date = dd_tmp.loc[
                dd_tmp.index.get_level_values(1) == peak_date,
                ("Duration", "Date at Max DD"),
            ].values[0]
            recession_end = BMonthEnd().rollforward(
                dd_tmp.loc[
                    dd_tmp.index.get_level_values(1) == peak_date,
                    ("Duration", "Recession End"),
                ].values[0]
            )

            data_peak = data.loc[data.index <= peak_date].iloc[-1]
            data_trough = data.loc[data.index <= trough_date].iloc[-1]
            data_dd_stats.update(
                {
                    d: {
                        "Peak": data_peak.values[0],
                        "Trough": data_trough.values[0],
                        "Peak to Trough (diff)": data_trough.values[0]
                        - data_peak.values[0],
                    }
                }
            )

            if isinstance(
                recession_start, pd.Timestamp
            ):  # pd.isnull(np.datetime64(recession_start.astype(int))):
                data_recession_start = data.loc[data.index <= recession_start].iloc[-1]
                data_recession_end = data.loc[data.index <= recession_end].iloc[-1]
                data_dd_stats[d] = {
                    **data_dd_stats[d],
                    **{
                        "Recession Start": data_recession_start.values[0],
                        "Recession End": data_recession_end.values[0],
                    },
                }
        data_dd_stats = pd.DataFrame.from_dict(data_dd_stats, orient="index")
        data_dd_stats_mean = pd.DataFrame(data_dd_stats.mean(axis=0).rename("Mean")).T
        data_dd_stats_median = pd.DataFrame(
            data_dd_stats.median(axis=0).rename("Median")
        ).T

        stats_summary.append(
            pd.concat(
                [
                    pd.concat(
                        [data_dd_stats, data_dd_stats_mean, data_dd_stats_median],
                        axis=0,
                    )
                ],
                keys=[recession],
                axis=0,
            )
        )
    return pd.concat(stats_summary, axis=0)


def rebase_data(data, pct=True):
    if pct:
        rebased = (
            (
                1 + np.log1p(data.pct_change(1)).reset_index(drop=True).fillna(0)
            ).cumprod()
        ) * 100
    else:
        rebased = data - data.iloc[0].values
    rebased.index = data.index
    return rebased


def get_data(data, dates, month, return_flag=True, after=True):
    exp_data = {}
    data_freq = get_date_frequency(data.index)

    for d in dates:
        if after:
            start_date = d
            if data_freq != "D":
                end_date = BMonthEnd().rollforward(
                    pd.to_datetime(d)
                    + dateutil.relativedelta.relativedelta(months=month)
                )
            elif isinstance(month, float):
                print(month)
                end_date = pd.to_datetime(d) + dateutil.relativedelta.relativedelta(
                    days=round(7 * 4 * month)
                )
            else:
                end_date = pd.to_datetime(d) + dateutil.relativedelta.relativedelta(
                    months=month
                )
        else:
            if data_freq != "D":
                start_date = BMonthEnd().rollback(
                    pd.to_datetime(d)
                    + dateutil.relativedelta.relativedelta(months=month)
                )
            elif isinstance(month, float):
                start_date = pd.to_datetime(d) + dateutil.relativedelta.relativedelta(
                    days=round(7 * 4 * month)
                )
            else:
                start_date = pd.to_datetime(d) + dateutil.relativedelta.relativedelta(
                    months=month
                )
            end_date = d

        if return_flag:
            exp_data.update(
                {d: data.loc[(data.index > start_date) & (data.index <= end_date)]}
            )
        else:
            exp_data.update(
                {d: data.loc[(data.index >= start_date) & (data.index <= end_date)]}
            )
    return exp_data

def get_actual_return(data, return_flag=True, spread=False):
    if return_flag:
        ending_value = (data + 1).prod()
        actual_return = ending_value - 1
    else:
        if spread:
            actual_return = data.iloc[-1] - data.iloc[0]
        else:
            actual_return = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
    return actual_return


def get_return_average(data_list, ref=None, return_flag=True, spread=False):
    actual_ret = []
    for d, data in data_list.items():
        if data.empty:
            continue
        actual_ret.append(
            pd.DataFrame(
                get_actual_return(
                    pd.DataFrame(data).dropna(how="all", axis=1),
                    return_flag=return_flag,
                    spread=spread,
                )
                .rename(d)
                .T
            )
        )  # .rename(start_date)
    actual_ret = pd.concat(actual_ret, axis=1)
    #print(actual_ret)
    if ref is not None:
        if ref in actual_ret.index:
            actual_ret = actual_ret.sub(actual_ret.loc[actual_ret.index == ref].values)
        else:
            print(ref + " not in the returns...")
    actual_ret_mean = actual_ret.mean(axis=1)
    actual_ret_median = actual_ret.median(axis=1)
    actual_ret_high = actual_ret.max(axis=1)
    actual_ret_low = actual_ret.min(axis=1)
    actual_ret_std = actual_ret.std(axis=1)
    actual_ret_stats = pd.concat(
        [
            actual_ret_mean,
            actual_ret_median,
            actual_ret_high,
            actual_ret_low,
            actual_ret_mean + actual_ret_std,
            actual_ret_mean - actual_ret_std,
        ],
        axis=1,
    )
    actual_ret_stats.columns = ["Mean", "Median", "High", "Low", "+1std", "-1std"]
    return {"stats": actual_ret_stats, "data": actual_ret}

def get_vol_average(data_list, ref=None, return_flag=True, spread=False):
    actual_vol = {}
    for d, data in data_list.items():
        if data.empty:
            continue
        vol = pd.DataFrame(data).dropna(how="all", axis=1).std()
        actual_vol.update({d:vol.values[0]})
    actual_vol = pd.DataFrame.from_dict(actual_vol, orient='index').rename(columns={0:'Vol'}).T
    if ref is not None:
        if ref in actual_vol.index:
            actual_vol = actual_vol.sub(actual_vol.loc[actual_vol.index == ref].values)
        else:
            print(ref + " not in the returns...")
    actual_vol_mean = actual_vol.mean(axis=1)
    actual_vol_median = actual_vol.median(axis=1)
    actual_vol_high = actual_vol.max(axis=1)
    actual_vol_low = actual_vol.min(axis=1)
    actual_vol_std = actual_vol.std(axis=1)
    actual_vol_stats = pd.concat(
        [
            actual_vol_mean,
            actual_vol_median,
            actual_vol_high,
            actual_vol_low,
            actual_vol_mean + actual_vol_std,
            actual_vol_mean - actual_vol_std,
        ],
        axis=1,
    )
    actual_vol_stats.columns = ["Mean", "Median", "High", "Low", "+1std", "-1std"]
    return {"stats": actual_vol_stats, "data": actual_vol}

def get_perf_stats(
    data,
    assets,
    dates,
    regime_str,
    ref=None,
    return_flag=True,
    after=True,
    spread=False,
    windows=None,
):
    if windows is None:
        if after:
            windows = [0.25, 1, 3, 6, 9, 12, 18, 24, 30, 36]
        else:
            windows = [-12, -9, -6, -3, -1]

    data = data[assets]

    details = {}
    stats_summary = []
    for w in windows:
        #print(w)
        data_snippets = get_data(data, dates, w, return_flag=return_flag, after=after)
        if not data_snippets:
            continue
        data_avg_returns = get_return_average(
            data_snippets, ref=ref, return_flag=return_flag, spread=spread
        )

        #print(data_avg_returns)
        #print(data_avg_returns["stats"])
        #print(data_avg_returns["stats"].reindex(index=[assets]))
        w_tmp = data_avg_returns["stats"]#.reindex(index=[assets])
        """
        data_avg_returns = get_vol_average(
            data_snippets, ref=ref, return_flag=return_flag, spread=spread
        )
        w_tmp = data_avg_returns["stats"]
        w_tmp.index = [assets]
        """
        stats_summary.append(pd.concat([w_tmp], keys=[str(w) + "M"], axis=1))
        details.update({(str(w) + "M"): data_avg_returns["data"]})#.reindex([assets])

    if details:
        stats_summary = pd.concat(
            [pd.concat(stats_summary, axis=1)], keys=[regime_str], axis=0
        )

        asset_details = {}
        for w_str in details:
            w_data = details[w_str].T
            for asset in w_data.columns:
                if asset not in asset_details:
                    asset_details.update({asset: [w_data[asset].rename(w_str)]})
                else:
                    asset_details[asset].append(w_data[asset].rename(w_str))
        for asset in asset_details:
            asset_details[asset] = pd.concat(
                [pd.concat(asset_details[asset], axis=1)], keys=[regime_str], axis=0
            )
        return {"stats": stats_summary, "data": asset_details}


def get_perf_stats_summary(
    data, dates_dict, ref=None, assets=None, after=True, return_flag=True, spread=False
):
    if assets is None:
        assets = list(data.columns)

    cross_summary = []
    dates_results = {}
    for k, v in dates_dict.items():
        #print(k)
        v_results = get_perf_stats(
            data,
            assets,
            v,
            k,
            ref=ref,
            return_flag=return_flag,
            after=after,
            spread=spread,
        )
        cross_summary.append(v_results["stats"])
        dates_results.update({k: v_results["data"]})

    # out_results = get_perf_stats(data[assets], th_no_dates, 'No Recession Afterwards', ref=ref, return_flag=return_flag, after=after)

    table_to_export = {}

    cross_summary_df = pd.concat(cross_summary, axis=0)  # .T[assets].T
    # display(cross_summary)
    # table_to_export.update({'overall':cross_summary})
    #print(assets)
    #print(dates_results)
    for asset in assets:
        tmp = [v[asset] for k, v in dates_results.items()]
        table_to_export.update({asset: pd.concat(tmp, axis=0)})
    return {"chart": cross_summary_df, "table": table_to_export}


def get_price_perf_around_dates(
    data, dates_dict, ref=None, assets=None, return_flag=True, spread=False
):
    if assets is None:
        assets = list(data.columns)

    stats_details = {}

    stats_before = get_perf_stats_summary(
        data,
        dates_dict=dates_dict,
        ref=ref,
        assets=assets,
        after=False,
        return_flag=return_flag,
        spread=spread,
    )

    stats_after = get_perf_stats_summary(
        data,
        dates_dict=dates_dict,
        ref=ref,
        assets=assets,
        after=True,
        return_flag=return_flag,
        spread=spread,
    )

    overall_summary = pd.concat([stats_before["chart"], stats_after["chart"]], axis=1)
    stats_details.update({"summary": overall_summary})

    brief_summary = overall_summary.loc[
        :, overall_summary.columns.isin(["Mean", "Median"], level=1)
    ]
    stats_details.update({"brief summary": brief_summary})

    for asset, asset_stats in stats_before["table"].items():
        stats_details.update(
            {asset: pd.concat([asset_stats, stats_after["table"][asset]], axis=1)}
        )

    return stats_details


def rebase_at_date(data, dates, prior_w=252 * 2, post_w=252 * 2, pct=True, prior=False):
    rebased_data = []
    for d in dates:
        if d in data.index:
            ind_d = [str(d.date()) for d in data.index].index(d)
        else:
            ind_d = data.loc[data.index <= d].shape[0]

        prior_tmp = data.iloc[ind_d - prior_w : ind_d + 1]
        post_tmp = data.iloc[ind_d : ind_d + post_w]

        # prior_tmp_returns = np.log1p(prior_tmp.pct_change(1))
        if pct:
            prior_tmp_dcum = (
                np.log1p(prior_tmp.pct_change(1)).fillna(0).add(1).cumprod()
            )
            prior_tmp_returns = (
                (prior_tmp_dcum.div(prior_tmp_dcum.iloc[-1]) - 1)
                .reset_index(drop=True)
                .fillna(0)
            )
            post_tmp_returns = (
                1 + np.log1p(post_tmp.pct_change(1)).reset_index(drop=True).fillna(0)
            ).cumprod() - 1  # *100
        else:
            prior_tmp_returns = prior_tmp - prior_tmp.iloc[-1].values
            post_tmp_returns = post_tmp - post_tmp.iloc[0].values
        tmp_returns = pd.concat(
            [prior_tmp_returns, post_tmp_returns.iloc[1:]], axis=0
        ).reset_index(drop=True)

        rebased_data.append(tmp_returns.rename(columns={data.columns[0]: d}))
    rebased_data = pd.concat(rebased_data, axis=1)
    rebased_min = rebased_data.min(axis=1)
    rebased_max = rebased_data.max(axis=1)
    rebased_mean = rebased_data.mean(axis=1)
    rebased_median = rebased_data.median(axis=1)
    rebased_20 = rebased_data.quantile(0.2, axis=1)
    rebased_80 = rebased_data.quantile(0.8, axis=1)
    rebased_stats = pd.concat(
        [
            rebased_min,
            rebased_max,
            rebased_mean,
            rebased_median,
            rebased_20,
            rebased_80,
        ],
        axis=1,
    )
    rebased_stats.columns = ["Down", "Up", "Mean", "Median", "20%", "80%"]
    return pd.concat([rebased_data, rebased_stats], axis=1)


def get_rebased_data(
    prices,
    dates: dict,
    current="2022-01-03",
    prior_w=262 * 2,
    post_w=262 * 2,
    pct=True,
    prior=False,
    rebase100=True,
):
    dates_rebase = []
    for d in dates:
        d_rebase = rebase_at_date(
            prices, dates[d], pct=pct, prior_w=prior_w, post_w=post_w, prior=prior
        )
        dates_rebase.append(pd.concat([d_rebase], keys=[d], axis=1))
    current_rebase = rebase_at_date(
        prices, [current], pct=pct, prior_w=prior_w, post_w=post_w, prior=prior
    )

    summary = pd.concat(
        dates_rebase + [pd.concat([current_rebase], keys=["Current Market"], axis=1)],
        axis=1,
    )
    if rebase100:
        summary = (1 + summary) * 100
    return summary


def get_steepest_perf(data, dates, perf_w=5, trough_w=365):
    best_worst_info = {}
    for d in dates:
        if d == "2022-01-03":
            continue
        tmp = data.loc[
            (data.index >= d)
            & (
                data.index
                < pd.to_datetime(d) + dateutil.relativedelta.relativedelta(months=36)
            )
        ]
        dd_min = tmp.min()
        dd_date = tmp.idxmin().values[0]
        down_data = data.loc[(data.index >= d) & (data.index <= dd_date)]
        up_data = data.loc[
            (data.index >= dd_date)
            & (
                data.index
                < (
                    pd.to_datetime(dd_date)
                    + dateutil.relativedelta.relativedelta(days=trough_w)
                )
            )
        ]

        down_returns = np.log1p(down_data.pct_change(perf_w))
        up_returns = np.log1p(up_data.pct_change(perf_w))

        max_down_return = down_returns.min()
        max_down_return_ind = down_returns.index.tolist().index(
            down_returns.idxmin().values[0]
        )
        max_down_return_date = down_data.index[max_down_return_ind - perf_w]

        max_up_return = up_returns.max()
        # print(d)
        # print(max_up_return)
        max_up_return_ind = up_returns.index.tolist().index(
            up_returns.idxmax().values[0]
        )

        # display(up_data.iloc[max_up_return_ind-5:max_up_return_ind+5])
        # display(up_returns.iloc[max_up_return_ind-5:max_up_return_ind+5])
        max_up_return_date = up_data.index[max_up_return_ind - perf_w]
        # print(max_up_return_date)
        # max_up_return_date = pd.to_datetime(up_returns.idxmax().values[0]) + dateutil.relativedelta.relativedelta(days=-perf_w-2)
        best_worst_info.update(
            {
                d: {
                    "Worst Start Date": max_down_return_date,
                    "Best Start Date": max_up_return_date,
                    "Interval": (max_up_return_date - max_down_return_date),
                    "Worst " + str(perf_w) + "D Perf": max_down_return.values[0],
                    "Best " + str(perf_w) + "D Perf": max_up_return.values[0],
                }
            }
        )
    return best_worst_info


def get_extremes(data, threshold=None):
    if threshold is None:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        up = mean + 2 * std
        down = mean - 2 * std
    else:
        up = threshold
        down = -threshold
    print(up)
    print(down)
    return data[(data >= up) | (data <= down)]


def next_extreme_pos_duration(worst_dates, best_dates):
    duration_stats = []
    worst_dates = sorted(worst_dates.tolist())
    best_dates = sorted(best_dates.tolist())

    for worst_d in worst_dates:
        for best_d in best_dates:
            if best_d > worst_d:
                duration_stats += [best_d - worst_d]
                break
    return pd.DataFrame(duration_stats)


def avoid_extreme_returns(returns, interval="Year", perf_w=1, perf_unit="D"):
    avoid_stats = {}
    avoid_best_returns = []
    avoid_worst_returns = []
    avoid_both_returns = []
    extreme_returns = []
    duration_stats = []

    if interval == "Decade":
        avoid_intervals = (
            returns.groupby((returns.index.year // 10) * 10).sum().index.tolist()
        )
        avoid_intervals.append(avoid_intervals[-1] + 10)
    elif interval == "Year":
        avoid_intervals = list(
            range(returns.index.min().year, returns.index.max().year + 1)
        )
        avoid_intervals.append(avoid_intervals[-1] + 1)

    for ind, y in enumerate(avoid_intervals[:-1]):
        tmp = deepcopy(
            returns.loc[
                (returns.index.year >= y)
                & (returns.index.year < avoid_intervals[ind + 1])
            ]
        )
        avoid_tmp = deepcopy(tmp)
        avoid_worst = deepcopy(tmp)
        avoid_best = deepcopy(tmp)
        only_both = pd.DataFrame(
            np.zeros(returns.shape[0]), index=returns.index
        )  # .rename(columns={0:'Best & Worst'})

        worst_dates = avoid_tmp.sort_values(avoid_tmp.columns[0]).iloc[:perf_w].index
        best_dates = avoid_tmp.sort_values(avoid_tmp.columns[0]).iloc[-perf_w:].index

        worst_best_duration = next_extreme_pos_duration(worst_dates, best_dates)
        duration_stats.append(
            pd.DataFrame.from_dict(
                {
                    y: {
                        "min": worst_best_duration.min().values[0],
                        "mean": worst_best_duration.mean().values[0],
                    }
                },
                orient="index",
            )
        )

        for d in worst_dates:
            avoid_tmp.loc[avoid_tmp.index == d] = 0
            avoid_worst.loc[avoid_worst.index == d] = 0
            only_both.loc[only_both.index == d] = tmp.loc[tmp.index == d]

        for d in best_dates:
            avoid_tmp.loc[avoid_tmp.index == d] = 0
            avoid_best.loc[avoid_best.index == d] = 0
            only_both.loc[only_both.index == d] = tmp.loc[tmp.index == d]

        avoid_worst_cum_ret = (1 + avoid_worst).cumprod().iloc[-1].values[0] - 1
        avoid_best_cum_ret = (1 + avoid_best).cumprod().iloc[-1].values[0] - 1
        avoid_both_cum_ret = (1 + avoid_tmp).cumprod().iloc[-1].values[0] - 1
        bench_cum_ret = (1 + tmp).cumprod().iloc[-1].values[0] - 1

        best_str = "Excluding Best " + str(perf_w) + perf_unit
        worst_str = "Excluding Worst " + str(perf_w) + perf_unit
        both_str = "Excluding Best & Worst " + str(perf_w) + perf_unit

        avoid_stats.update(
            {
                y: {
                    "Price Return": bench_cum_ret,
                    worst_str: avoid_worst_cum_ret,
                    best_str: avoid_best_cum_ret,
                    both_str: avoid_both_cum_ret,
                },
            }
        )

        summary = pd.DataFrame.from_dict(avoid_stats, orient="index").T
        avoid_best_returns.append(avoid_best)
        avoid_worst_returns.append(avoid_worst)
        avoid_both_returns.append(avoid_tmp)
        extreme_returns.append(
            only_both.rename(columns={0: "Best & Worst " + str(perf_w) + perf_unit})
        )

    avoid_best_returns = pd.concat(avoid_best_returns, axis=0).rename(
        columns={returns.columns[0]: best_str}
    )
    avoid_worst_returns = pd.concat(avoid_worst_returns, axis=0).rename(
        columns={returns.columns[0]: worst_str}
    )
    avoid_both_returns = pd.concat(avoid_both_returns, axis=0).rename(
        columns={returns.columns[0]: both_str}
    )

    return {
        "summary": summary,
        "returns": pd.concat(
            [returns, avoid_best_returns, avoid_worst_returns, avoid_both_returns],
            axis=1,
        ),
        "extremes": pd.concat(extreme_returns, axis=0),
        "duration": pd.concat(duration_stats, axis=0),
    }  # avoid_returns.rename(columns={'SPX':'Avoid'})#((1 + avoid_returns.fillna(0)).cumprod()*100).rename(columns={returns.columns[0]:'Avoid '+str(perf_w)+'D'})
