'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-01 22:03:26
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import copy
from math import e
import warnings

from numpy.lib.function_base import quantile
from pandas.core.indexes.base import Index

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime as dt

import os
from pathlib import Path
from IPython.display import display
import logging

logging.basicConfig(
    format="%(asctime)s.%(msecs)d [%(levelname)s] %(message)s", level=logging.INFO
)  #%(name)s
logging.getLogger("matplotlib.font_manager").disabled = True

import performance as perf
import visualization as vis
import preprocess as prep
import signals as signal

baseN = 100


def configure_params(params: dict = None) -> dict:
    """[summary]

    Args:
        params (dict, optional): [description]. Defaults to None.

    Returns:
        dict: [description]
    """
    if not params:
        params = {
            "portfolio": {"weights": None, "assets": None},
            "trading": {
                "day": "FRI",
                "frequency": "M",
                "strategy": "ls",
                "position": "equal",
            },
            "performance": {  # todo 'performance' -> 'analysis'
                "tearsheet": False,
                "stats": False,
                "visualization": False,
                "signalStats": False,
                "save": False,
                "rootDir": os.getcwd(),
            },
            "data": {
                "operation": "mean",
                "signalWindow": 20,
                "historyWindow": 261,
                "emdWindow": 261 * 5,
                "source": "org",
                "signal": "simple",
                "saveFig": False,
            },
        }

    # configure data parameters
    if "portfolio" not in params.keys():
        params.update({"portfolio": {"weights": None, "assets": None}})
    else:
        if "weights" not in params["portfolio"].keys():
            params["portfolio"].update({"weights": None})
        if "assets" not in params["portfolio"].keys():
            params["portfolio"].update({"assets": None})

    # configure parameters for trading
    if "trading" not in params.keys():
        params.update(
            {
                "trading": {
                    "day": "FRI",
                    "frequency": "M",
                    "strategy": "ls",
                    "position": "zeroCost",
                }
            }
        )
    else:
        if "day" not in params["trading"].keys():
            params["trading"].update({"day": "FRI"})
        if "frequency" not in params["trading"].keys():
            params["trading"].update({"frequency": "M"})
        if "strategy" not in params["trading"].keys():
            params["trading"].update({"strategy": "ls"})
        if "position" not in params["trading"].keys():
            params["trading"].update({"position": "equal"})

    # configure parameters for performance analysis
    if "performance" not in params.keys():
        params.update(
            {
                "performance": {
                    "tearsheet": False,
                    "stats": False,
                    "visualization": False,
                    "signalStats": False,
                    "save": False,
                    "rootDir": os.getcwd(),
                }
            }
        )
    else:
        if "tearsheet" not in params["performance"].keys():
            params["performance"].update({"tearsheet": False})
        if "stats" not in params["performance"].keys():
            params["performance"].update({"stats": False})
        if "visualization" not in params["performance"].keys():
            params["performance"].update({"visualization": False})
        if "signalStats" not in params["performance"].keys():
            params["performance"].update({"signalStats": False})
        if "save" not in params["performance"].keys():
            params["performance"].update({"save": False})
        if "rootDir" not in params["performance"].keys():
            params["performance"].update({"rootDir": os.getcwd()})

    # configure data parameters
    if "data" not in params.keys():
        params.update({"data": signal.configure_params(params["portfolio"]["assets"])})
    else:
        params["data"] = signal.configure_params(
            params["portfolio"]["assets"], params["data"]
        )

    if (
        params["data"]["signal"] == "aggregation"
        or params["data"]["signal"] == "differentiation"
    ):
        if "aggSignal" not in params.keys():
            params.update(
                {
                    "aggSignal": {
                        "source": ["org", "ols", "emd"],
                        "signalWindow": [65, 195],
                        "threshold": 4,
                    }
                }
            )
        if "source" not in params["aggSignal"].keys():
            params["aggSignal"].update({"source": ["org", "ols", "emd"]})
        if "signalWindow" not in params["aggSignal"].keys():
            params["aggSignal"].update({"signalWindow": [20, 130]})
        if "threshold" not in params["aggSignal"].keys():
            th = 2 * (len(params["aggSignal"]["source"]) - 1)
            params["aggSignal"].update({"threshold": th})

    params["performance"]["rootDir"] = Path(params["performance"]["rootDir"]) / params["strategyPrefix"]
    if not os.path.exists(params["performance"]["rootDir"]):
            os.makedirs(params["performance"]["rootDir"])
    # set the exp code
    params["performance"].update({"expCode": configure_expCode(params)})
    params["performance"].update({"dir": configure_path(params["performance"])})
    if not os.path.exists(params["performance"]["dir"]):
            os.makedirs(params["performance"]["dir"])
    return params


def configure_expCode(params: dict) -> str:
    if (
        params["data"]["signal"] == "differentiation"
        or params["data"]["signal"] == "aggregation"
    ):
        expCode = (
            params["data"]["operation"]
            + "-"
            + "-".join(params["aggSignal"]["source"])
            + "-"
            + params["data"]["signal"]
            + "-"
            + params["trading"]["strategy"]
            + "-"
            + str(params["aggSignal"]["signalWindow"][0])
            + "-"
            + str(params["aggSignal"]["signalWindow"][1])
            + "D"
        )
    else:
        expCode = (
            params["data"]["operation"]
            + "-"
            + params["data"]["source"]
            + "-"
            + params["data"]["signal"]
            + "-"
            + params["trading"]["strategy"]
            + "-"
            + str(params["data"]["signalWindow"])
            + "D"
        )
    if params["data"]["signal"] == "ranking":
        expCode = expCode + "-th" + str(params["data"]["threshold"]) + "-top" + str(params["data"]["topN"])
    if params["data"]["signal"] == "simple":
        if params["data"]["source"] == "ols":
            expCode = (
                expCode
                + "-th"
                + ("-").join([str(c) for c in params["data"]["threshold"].values()])
            )
        else:
            expCode = expCode + "-th" + str(params["data"]["threshold"])
        if params["data"]["source"] != "ols" and params["data"]["threshold"] != 0:
            expCode = expCode + "-H" + str(params["data"]["historyWindow"]) + "D"
    elif (
        params["data"]["signal"] == "differentiation"
        or params["data"]["signal"] == "aggregation"
    ):
        expCode = expCode + "-th" + str(params["aggSignal"]["threshold"])
    # Todos: endfix for other time series momentum methods
    if "targetVol" in params["trading"].keys():
        expCode = expCode + "-Vol" + str(params["trading"]["targetVol"])
    expCode = params['strategyPrefix'] + '-' + expCode + "-" + params["trading"]["frequency"]
    return expCode.replace("/", "").replace(" ", "")


def configure_path(params):
    fileDir = Path(params["rootDir"]) / (params["expCode"])
    if (
        params.get("save", False)
        or params.get("stats", False)
        or params.get("visualization", False)
    ):
        if not os.path.exists(params["rootDir"]):
            os.makedirs(params["rootDir"])
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)
    return fileDir


def get_trading_signal(signalData, params=None):
    """
    get trading signal of selected strategies

    Parameters
    ----------
    signalData: daily return of assets
    params: signalData: the return data to generate MoM signals

    Returns
    -------
    tradingSignal: +1 -> long, -1 -> short, 0 -> neutral
    """
    signalData = signalData.loc[
        :,
        [
            c
            for c in signalData.columns
            if "date" not in c and "Day" not in c and "Date" not in c
        ],
    ]
    signalData = signalData.select_dtypes(exclude=["datetime"])
    signalData = pd.DataFrame(signalData.loc[:, params["portfolio"]["assets"]])

    ### get the trading signal
    if params["data"]["signal"] == "ranking":
        tradingSignal = signal.ranking(signalData, params=params)
    elif params["data"]["signal"] == "rankingReversion":
        tradingSignal = signal.ranking_reversion(signalData, params=params)
    elif params["data"]["signal"] == "multi_lvl_Ranking":
        tradingSignal = signal.multi_lvl_ranking(signalData, params=params)
    # elif params['tradingSignal'] == 'quantile':
    # 	quantileSig, quantileDf = quantileSignal(signalData, quantileNo=params['quantileNo'])
    elif params["data"]["signal"] == "simple":
        tradingSignal = signal.simple(signalData, params=params)
    elif params["data"]["signal"] == "simpleReversion":
        tradingSignal = signal.simple_reversion(signalData, params=params)
    elif params["data"]["signal"] == "MACD":
        tradingSignal = signal.macd(signalData, params=params)
    # elif params['data']['signal'] == 'ols':
    # 	tradingSignal = signals.olsMoM(signalData, params=params)
    elif params["data"]["signal"] == "aggregation":
        tradingSignal = signal.aggregate_signal(signalData, params=params)
    elif params["data"]["signal"] == "differentiation":
        tradingSignal = signal.differentiate_signal(signalData, params=params)
    else:
        logging.error(
            "please provide a valid signal type: simple, simpleReversion, ranking, rankingReversion, ols"
        )
    tradingSignal.index = pd.to_datetime(tradingSignal.index)
    return tradingSignal


def zerocost_position(trading_signal: pd.DataFrame, budget=None) -> pd.DataFrame:
    negPosNumber = (trading_signal[trading_signal == -1].abs().sum(axis=1))
    negPos = (
        trading_signal[trading_signal == -1]
        .div(100).multiply(budget) #trading_signal[trading_signal == -1].abs().sum(axis=1), axis=0
        .fillna(0)
    )
    posPos = (
        trading_signal[trading_signal == 1]
        .multiply(negPosNumber*0.01/trading_signal[trading_signal == 1].sum(axis=1), axis=0)
        .fillna(0)
    )
    position = posPos + negPos
    return position


def long_position(trading_signal: pd.DataFrame) -> pd.DataFrame:
    return (
        trading_signal[trading_signal == 1]
        .div(trading_signal[trading_signal == 1].sum(axis=1), axis=0)
        .fillna(0)
    )


def short_position(trading_signal: pd.DataFrame) -> pd.DataFrame:
    return (
        trading_signal[trading_signal == -1]
        .div(trading_signal[trading_signal == -1].sum(axis=1), axis=0)
        .fillna(0)
    )


def low_lvl_position(high_lvl_position, low_lvl_signal):
    high_lvl_long = high_lvl_position.apply(lambda x: x if x > 0 else 0)
    high_lvl_short = high_lvl_position.apply(lambda x: x if x < 0 else 0)
    low_lvl_pos = (
        zerocost_position(low_lvl_signal).mul(high_lvl_position, axis=0)
        + long_position(low_lvl_signal).mul(high_lvl_long, axis=0)
        + short_position(low_lvl_signal).mul(high_lvl_short, axis=0)
    )
    return low_lvl_pos


def get_trading_position(
    portfolioData: pd.DataFrame, tradingSignal, params: dict
) -> pd.DataFrame:
    """configure the position for trading:
                    1. data parameter
                    2. portfolio parameters
                    3. trading parameters
                    4. experiment parameters
    Args:
            portfolioData:
            tradingSignal:
            params:
    Returns:
            position:
    """
    if params["trading"]["position"] == "zeroCost":
        if params["data"]["signal"] != "multi_lvl_Ranking":
            position = zerocost_position(
                tradingSignal,
                params['trading'].get('budget', 1)
            )  # + pd.DataFrame(np.zeros(tradeSignal.shape), columns=tradeSignal.columns, index=tradeSignal.index)#tradeSignal.apply(lambda row: row)
        else:
            high_lvl_position = zerocost_position(tradingSignal["high"])
            fi_position = low_lvl_position(
                high_lvl_position["Government Bonds"], tradingSignal["Fixed Income"]
            )
            eq_position = low_lvl_position(
                high_lvl_position["Global (Thematic)"], tradingSignal["Equities"]
            )
            position = pd.concat(
                [
                    eq_position,
                    fi_position,
                    high_lvl_position.loc[
                        :,
                        [
                            c
                            for c in high_lvl_position.columns
                            if c not in ["Government Bonds", "Global (Thematic)"]
                        ],
                    ],
                ],
                axis=1,
            )
            position = position[portfolioData.columns]
    elif params["trading"]["position"] == "volAdjust":
        priceVol = prep.get_return_vol(
            portfolioData, params
        )  # prep.getPriceGarchVol(portfolioData)#
        targetVol = (
            pd.DataFrame(
                params["trading"]["targetVol"],
                index=priceVol.index,
                columns=priceVol.columns,
            )
            if "targetVol" in params["trading"]
            else prep.get_return_vol(pd.DataFrame(portfolioData.mean(axis=1)), params)
        )  # prep.getPriceGarchVol(pd.DataFrame(portfolioData.mean(axis=1)))#
        priceNormVol = priceVol.div(targetVol.values, axis=0)  # 0.4
        for c in priceNormVol.columns:
            ind = priceNormVol[c].index.get_loc(priceNormVol[c].first_valid_index())
            priceNormVol[c].iloc[ind:].fillna(method="ffill", inplace=True)
        priceNormVol = priceNormVol.loc[
            priceNormVol.index.intersection(tradingSignal.index)
        ]
        for c in portfolioData.columns:
            ind = portfolioData[c].index.get_loc(portfolioData[c].first_valid_index())
            portfolioData[c].iloc[ind:].fillna(method="ffill", inplace=True)
        priceData = portfolioData.loc[
            portfolioData.index.intersection(tradingSignal.index)
        ]
        position = pd.DataFrame(tradingSignal / priceNormVol).div(
            priceData.count(axis=1), axis=0
        )
    elif params["trading"]["position"] == "assigned":
        position = tradingSignal * params["portfolio"]["weights"].values
    """
    elif params['trading']['position'] == 'equal':
        priceData = portfolioData.loc[portfolioData.index.intersection(
            tradingSignal.index)]
        position = pd.DataFrame(tradingSignal).div(priceData.count(axis=1),
                                                   axis=0)
    """
    return position


def backtest(tradingPrice, tradingPosition, params):
    if params["portfolio"]["weights"] is None:
        # if portfolio weights are not specified, we assume the base portfolio is an equally weighted portfolio
        pfW = pd.DataFrame(
            np.ones((1, tradingPosition.shape[1])), columns=tradingPosition.columns
        )
        pfW = pfW.divide(pfW.sum(axis=1), axis=0)
    else:
        pfW = params["portfolio"]["weights"]

    portfolioWeights = (
        pd.DataFrame(1, index=tradingPrice.index, columns=pfW.columns) * pfW.values
    )
    # get the return of assets for the portfolio
    display(tradingPrice)
    priceDf = tradingPrice[list(portfolioWeights.columns)]
    priceReturn = priceDf.pct_change()  # .shift(-1)
    display(priceReturn)
    priceReturn.iloc[0, :] = (priceDf.iloc[0, :] - 100) / 100
    # priceReturn = priceReturn.dropna(how='all', axis=0)

    # compute the return of the base portfolio
    """
    if params['trading']['strategy'] == 'ls':
        benchReturn = pd.DataFrame(
            np.zeros((priceReturn.shape[0], priceReturn.shape[1] + 1)),
            index=priceReturn.index,
            columns=list(priceReturn.columns) + ['Portfolio'])
    else:
    """
    benchReturn = pd.concat(
        [
            priceReturn * portfolioWeights.values,
            (priceReturn * portfolioWeights.values).sum(axis=1),
        ],
        axis=1,
    )
    benchReturn.columns = list(priceReturn.columns) + ["Portfolio"]
    benchCumReturn = baseN * (1 + benchReturn).cumprod()

    # compute the active return of the strategy
    activeWeights = tradingPosition.shift(1).fillna(0)
    for c in portfolioWeights.columns:
        if c not in activeWeights.columns:
            activeWeights[c] = 0
    activeWeights = activeWeights[portfolioWeights.columns]
    stratWeights = portfolioWeights + activeWeights
    stratReturn = stratWeights * priceReturn
    stratReturn["Portfolio"] = stratReturn.sum(axis=1)
    stratCumReturn = pd.DataFrame(baseN * (1 + stratReturn).cumprod())

    # compute the performance of the strategy on the portfolio
    activeReturn = pd.DataFrame(stratReturn - benchReturn).dropna(how="all", axis=1)
    activeCumReturn = pd.DataFrame(baseN * (1 + activeReturn).cumprod())

    
    returns = {
        "strategy": stratReturn,
        "benchmark": benchReturn,
        "active": activeReturn,
    }

    cumReturns = {
        "strategy": stratCumReturn,
        "benchmark": benchCumReturn,
        "active": activeCumReturn,
    }

    results = {
        "positions": stratWeights,
        "trades": pd.concat([tradingPosition[tradingPosition!=0].dropna(how='all', axis=0), tradingPosition[tradingPosition != 0].dropna(how='all', axis=0).count(axis=1)], axis=1),
        "return": returns,
        "cumReturn": cumReturns,
        "summary": pd.concat(
            [
                benchReturn.rename(columns=lambda x: x + "-bench"),
                stratReturn.rename(columns=lambda x: x + "-strategy"),
                activeReturn.rename(columns=lambda x: x + "-active"),
            ],
            axis=1,
        ),
        "cumSummary": pd.concat(
            [
                benchCumReturn.rename(columns=lambda x: x + "-bench"),
                stratCumReturn.rename(columns=lambda x: x + "-strategy"),
                activeCumReturn.rename(columns=lambda x: x + "-active"),
            ],
            axis=1,
        ),
    }
    return results


def strategy_experiment(
    #strategyPrefix: str,
    returnData: pd.DataFrame,
    signalData: pd.DataFrame,
    params: dict = None,
):
    # configure the parameters
    if not params:
        params = {
            "portfolio": {
                "assets": list(
                    set(list(returnData.columns)).intersection(
                        set(list(signalData.columns))
                    )
                )
            }
        }
    elif "portfolio" not in params.keys():
        params.update(
            {
                "portfolio": {
                    "assets": list(
                        set(list(returnData.columns)).intersection(
                            set(list(signalData.columns))
                        )
                    )
                }
            }
        )
    elif "assets" not in params["portfolio"].keys():
        params["portfolio"].update(
            {
                "assets": list(
                    set(list(returnData.columns)).intersection(
                        set(list(signalData.columns))
                    )
                )
            }
        )

    params = configure_params(params)
    logging.info(params["strategyPrefix"] + ": " + params["performance"]["expCode"])

    signalData.index = pd.to_datetime(signalData.index)
    # get trading signal
    signal_assets = params['signal']['assets']
    tradingSignal = get_trading_signal(signalData, params=params)
    tradingSignal = tradingSignal[[asset for asset in signal_assets if asset in tradingSignal.columns ]]
    display(tradingSignal)

    # get asset price based on trading frequency
    cumReturnData = (((1 + returnData).cumprod()) * 100).fillna(method="ffill")
    cumReturnData.index = pd.to_datetime(cumReturnData.index)

    if isinstance(tradingSignal, pd.DataFrame):
        tradingPrice = pd.DataFrame(
            cumReturnData.loc[
                cumReturnData.index.intersection(tradingSignal.index),
                :,#params["portfolio"]["assets"],
            ]
        )
    elif isinstance(tradingSignal, dict):
        tradingPrice = pd.DataFrame(
            cumReturnData.loc[
                cumReturnData.index.intersection(
                    tradingSignal.get("high", ValueError("No data available")).index
                ),
                params["portfolio"]["assets"],
            ]
        )

    # get trading position
    tradingPosition = get_trading_position(returnData, tradingSignal, params=params)

    # backtest
    expResults = backtest(tradingPrice, tradingPosition, params)
    # print(perf.getTearSheet(expResults['returns'], tradingPosition, params))

    # get the summary
    expStats = perf.get_result_summary(expResults["return"], params)
    expStats_summary = expStats["summary"].rename_axis(
        (params["trading"]["strategy"] + "-" + params["data"]["signal"])
    )

    """
    expStats = pd.concat([
        expStats,
        perf.get_signal_performance(returnData, signalData, tradingSignal,
                                  params)
    ],
                         axis=1)
    display(expStats)
    """

    ## todo: add benchmark to the cumulative returns
    expCumRet = pd.DataFrame(expResults["cumReturn"]["strategy"]["Portfolio"])
    expCumRet.columns = [params["performance"]["expCode"]]
    benchCumRet = expResults["cumReturn"]["benchmark"]["Portfolio"]
    benchCumRet.columns = ['Benchmark']

    exp_returns = {}
    for k, v in expResults["return"].items():
        exp_returns.update({k + " returns": v})
    exp_cum_returns = {}
    for k, v in expResults["cumReturn"].items():
        exp_cum_returns.update({k + " cumulative returns": v})

    stratExp = {
        "summary": expStats_summary,
        "stats": expStats["stats"],
        "positions": expResults["positions"],
        "trades": expResults["trades"],
        "portfolio returns": pd.concat([expCumRet, benchCumRet], axis=1),
        **exp_returns,
        **exp_cum_returns,
    }

    # get the tearsheet
    if params["performance"]["tearsheet"]:
        tearsheetDf = pd.concat(
            [
                perf.tearsheet_to_tableau(
                    perf.get_perf_tearsheet(
                        expResults["return"]["strategy"]["Portfolio"],
                        tradingPosition,
                        params,
                    ),
                    "Strategy",
                    params,
                ),
                # perf.tearsheet_to_tableau(
                #    perf.get_perf_tearsheet(
                #        expResults['return']['benchmark']['Portfolio'],
                #        tradingPosition, params), 'Benchmark', params)
            ],
            axis=0,
        )
        tearsheetDf["Signal"] = params["data"]["signal"]
        stratExp.update({"tearsheet": tearsheetDf})
    if params["performance"]["visualization"]:
        expFiles = vis.visualize_results(
            tradingPrice, tradingSignal, expResults, params
        )
        stratExp.update({"results_visualization": expFiles})
    if params["data"]["saveFig"]:
        vis.visualizeTradingSignal(tradingPrice, signalData, params)
    if params["performance"]["signalStats"]:
        expSignalStats = perf.getSignalDuration(tradingSignal, params)
        stratExp.update({"signal": expSignalStats})
    return stratExp, params["performance"]["dir"]
    """
	# get trading position
	tradingPosition = getTradingPosition(tradingPrice, tradingSignal, params=params)

	# backtest
	expResults = backTest(tradingPrice, tradingPosition, params)

	#print(perf.getTearSheet(expResults['returns'], tradingPosition, params))
	# get the summary
	expStats = perf.resultSummary(expResults['return'], params).rename_axis((params['trading']['strategy']+'-'+params['data']['signal']))

	# get the tearsheet
	if params['performance']['tearsheet']:
		expSheet = perf.getTearSheet(expResults['return'], tradingPosition, params)
		return expStats
	if params['performance']['statsFig']:
		expFiles = vis.visualizeResults(tradingPrice, tradingSignal, expResults, params)
	if params['data']['saveFig']:
		vis.visualizeTradingSignal(tradingPrice, rawTradingSignal, params)
	"""
