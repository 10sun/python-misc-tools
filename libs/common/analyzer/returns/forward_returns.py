'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 00:29:22
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd


## TODO 1. check if forward returns are computed properly
def compute_forward_returns(
    prices,
    date_freq="D",
    periods=(1, 5, 10),
    filter_zscore=None,
    cumulative_returns=True,
):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.


    Args:
        factor (_type_): _description_
        prices (_type_): _description_
        periods (tuple, optional): _description_. Defaults to (1, 5, 10).
        filter_zscore (_type_, optional): _description_. Defaults to None.
        cumulative_returns (bool, optional): _description_. Defaults to True.
    """
    factor_dateindex = factor.index.levels[0]
    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError(
            "The timezone of 'factor' is not the "
            "same as the timezone of 'prices'. See "
            "the pandas methods tz_localize and "
            "tz_convert."
        )

    freq = infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError(
            "Factor and prices indices don't match: make sure "
            "they have the same convention in terms of datetimes "
            "and symbol-names"
        )

    # chop prices down to only the assets we care about (= unique assets in
    # `factor`).  we could modify `prices` in place, but that might confuse
    # the caller.
    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    column_list = []

    for period in sorted(periods):
        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:
            mask = abs(forward_returns - forward_returns.mean()) > (
                filter_zscore * forward_returns.std()
            )
            forward_returns[mask] = np.nan

        #
        # Find the period length, which will be the column name. We'll test
        # several entries in order to find out the most likely period length
        # (in case the user passed inconsinstent data)
        #
        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq, date_freq)
            days_diffs.append(period_len.components.days)

        delta_days = period_len.components.days - stats.mode(days_diffs).mode[0]
        period_len -= pd.Timedelta(days=delta_days)
        label = timedelta_to_string(period_len)

        column_list.append(label)

        raw_values_dict[label] = np.concatenate(forward_returns.values)

    df = pd.DataFrame.from_dict(raw_values_dict)
    df.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns], names=["date", "asset"]
        ),
        inplace=True,
    )
    df = df.reindex(factor.index)

    # now set the columns correctly
    df = df[column_list]

    df.index.levels[0].freq = freq
    df.index.set_names(["date", "asset"], inplace=True)
    return df


def backshift_returns_series(series, N):
    """Shift a multi-indexed series backwards by N observations in
    the first level.

    This can be used to convert backward-looking returns into a
    forward-returns series.
    """
    ix = series.index
    dates, sids = ix.levels
    date_labels, sid_labels = map(np.array, ix.labels)

    # Output date labels will contain the all but the last N dates.
    new_dates = dates[:-N]

    # Output data will remove the first M rows, where M is the index of the
    # last record with one of the first N dates.
    cutoff = date_labels.searchsorted(N)
    new_date_labels = date_labels[cutoff:] - N
    new_sid_labels = sid_labels[cutoff:]
    new_values = series.values[cutoff:]

    assert new_date_labels[0] == 0

    new_index = pd.MultiIndex(
        levels=[new_dates, sids],
        labels=[new_date_labels, new_sid_labels],
        sortorder=1,
        names=ix.names,
    )

    return pd.Series(data=new_values, index=new_index)


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe was 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values("date")

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(grouper)[cols].transform(
        lambda x: x - x.mean()
    )

    return factor_data
