'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-01 16:58:40
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from concurrent.futures import process
import warnings
from numpy.lib.function_base import quantile

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(
    format="%(asctime)s.%(msecs)d [%(levelname)s] %(message)s", level=logging.INFO
)  #%(name)s
logging.getLogger("matplotlib.font_manager").disabled = True

import numpy as np
import pandas as pd
from datetime import datetime as dt
from IPython.display import display

# import matplotlib.pyplot as plt
import copy
from pathlib import Path
import sys
"""
try:
    from arch import arch_model
    from arch.univariate import LS
    import emd
except ImportError:
    sys.path.append(r"\\merlin\lib_dpm\code\libs\arch-master")
    sys.path.append(r"\\merlin\lib_dpm\code\libs\emd-0.4.0")
    from arch import arch_model
    from arch.univariate import LS
    import emd
"""
# from PyEMD import EMD as emd

## TODO: move to trader module
def get_trading_date(dataDf: pd.DataFrame, params: dict) -> pd.Series:
    """get the trading date given the trading frequency
    Args:
            dataDf:
    """
    if params["trading"]["frequency"] == "D":
        trading_date = pd.DataFrame(
            list(dataDf.index), index=dataDf.index, columns=["tradingDate"]
        )
    elif params["trading"]["frequency"] == "W":
        trading_date = pd.DataFrame(
            dataDf.groupby(pd.Grouper(freq="W-" + params["trading"]["day"]))
            .last()
            .index.values,
            columns=["tradingDate"],
            index=dataDf.groupby(pd.Grouper(freq="W-" + params["trading"]["day"]))
            .last()
            .index,
        )
    elif params["trading"]["frequency"] == "M":
        trading_date = pd.DataFrame(
            dataDf.groupby(pd.Grouper(freq="BM")).last().index.values,
            columns=["tradingDate"],
            index=dataDf.groupby(pd.Grouper(freq="BM")).last().index,
        )
    elif params["trading"]["frequency"] == "Q":
        trading_date = pd.DataFrame(
            dataDf.groupby(pd.Grouper(freq="BQ")).last().index.values,
            columns=["tradingDate"],
            index=dataDf.groupby(pd.Grouper(freq="BQ")).last().index,
        )
    return trading_date.iloc[:-1]


def preprocess_data(dataDf, params):
    """
	if params['data']['source'] == 'Z':
		if params['data']['operation'] == 'mean':
			processedDf = (dataDf.rolling(window=params['data']['signalWindow']).mean() \
				- dataDf.rolling(window=params['data']['historyWindow']).mean())\
				/dataDf.rolling(window=params['data']['historyWindow']).std()
		elif params['data']['operation'] == 'sum':
			dataOp = dataDf.rolling(window=params['data']['signalWindow']).sum()
			processedDf = (dataOp \
				- dataOp.rolling(window=params['data']['historyWindow']).mean())\
				/dataOp.rolling(window=params['data']['historyWindow']).std()
		elif params['data']['operation'] == 'None':
			processedDf = (dataDf \
				- dataDf.rolling(window=params['data']['historyWindow']).mean())\
				/dataDf.rolling(window=params['data']['historyWindow']).std()
		elif params['data']['operation'] == 'return':
			dataOp = dataDf/dataDf.shift(params['data']['signalWindow']) - 1
			processedDf = (dataOp \
				- dataOp.rolling(window=params['data']['historyWindow'], min_periods=int(np.ceil(params['data']['historyWindow']/2))).mean())\
				/dataOp.rolling(window=params['data']['historyWindow'], min_periods=int(np.ceil(params['data']['historyWindow']/2))).std()
	el
	"""
    tradingDate = get_trading_date(dataDf, params)
    if params["data"]["source"] == "org":
        if params["data"]["operation"] == "mean":
            processedDataDf = dataDf.rolling(
                window=params["data"]["signalWindow"]
            ).mean()
        elif params["data"]["operation"] == "sum":
            processedDataDf = dataDf.rolling(
                window=params["data"]["signalWindow"]
            ).sum()
        elif params["data"]["operation"] == "None":
            processedDataDf = dataDf
        elif params["data"]["operation"] == "return":
            rebasedDataDf = ((1 + dataDf).cumprod()) * 100
            processedDataDf = (
                rebasedDataDf / rebasedDataDf.shift(params["data"]["signalWindow"]) - 1
            )
        elif params["data"]["operation"] == "compReturn":
            processedDataDf = dataDf / dataDf.shift(params["data"]["signalWindow"]) - 1
        elif params["data"]["operation"] == "volAdjReturn":
            rebasedDataDf = ((1 + dataDf).cumprod()) * 100
            prepDataDf = (
                rebasedDataDf / rebasedDataDf.shift(params["data"]["signalWindow"]) - 1
            )
            dataDfVol = get_return_vol(dataDf, params)  # getPriceGarchVol(dataDf) #
            if "targetVol" in params["trading"]:
                logging.info("target volatility: " + str(params["trading"]["targetVol"]))
                targetVol = pd.DataFrame(
                    params["trading"]["targetVol"],
                    index=dataDfVol.index,
                    columns=dataDfVol.columns,
                )
            else:
                dataDfBench = (
                    pd.DataFrame(dataDf.mean(axis=1), params)
                    if params["portfolio"]["weights"] is None
                    else (dataDf * params["portfolio"]["weights"].values).sum(axis=1)
                )
                targetVol = get_return_vol(
                    dataDfBench, params
                )  # getPriceGarchVol(pd.DataFrame(dataDf.mean(axis=1))) #
            #dataRelVol = dataDfVol.div(targetVol.values, axis=0)  # portVol.values
            processedDataDf = prepDataDf.div(dataDfVol.values, axis=0)#prepDataDf / dataRelVol
        elif params["data"]["operation"] == "volAdjRelativeReturn":
            dataDfBench = (
                pd.DataFrame(dataDf.mean(axis=1), params)
                if params["portfolio"]["weights"] is None
                else pd.DataFrame(
                    (dataDf * params["portfolio"]["weights"].values).sum(axis=1)
                )
            )
            relDataDf = (
                dataDf - dataDfBench.values
            )  # .subtract(dataDfBench.values, axis=1)
            rebasedDataDf = ((1 + relDataDf).cumprod()) * 100
            prepDataDf = (
                rebasedDataDf / rebasedDataDf.shift(params["data"]["signalWindow"]) - 1
            )
            dataDfVol = get_return_vol(relDataDf, params)  # getPriceGarchVol(dataDf) #
            targetVol = pd.DataFrame(
                1.0, index=dataDfVol.index, columns=dataDfVol.columns
            )  # get_return_vol(dataDfBench, params) #getPriceGarchVol(pd.DataFrame(dataDf.mean(axis=1))) #
            dataRelVol = dataDfVol.div(targetVol.values, axis=0)  # portVol.values
            processedDataDf = prepDataDf / dataRelVol
        elif params["data"]["operation"] == "relativeReturn":
            dataDfBench = (
                pd.DataFrame(dataDf.mean(axis=1), params)
                if params["portfolio"]["weights"] is None
                else pd.DataFrame(
                    (dataDf * params["portfolio"]["weights"].values).sum(axis=1)
                )
            )
            relDataDf = (
                dataDf - dataDfBench.values
            )  # .subtract(dataDfBench.values, axis=1)
            rebasedDataDf = ((1 + relDataDf).cumprod()) * 100
            processedDataDf = (
                rebasedDataDf / rebasedDataDf.shift(params["data"]["signalWindow"]) - 1
            )
        else:
            print('method not available...')

        if params["data"].get("threshold", 0) != 0:
            processedDf = (
                processedDataDf
                - processedDataDf.rolling(
                    window=params["data"]["historyWindow"],
                    min_periods=int(np.ceil(params["data"]["historyWindow"] / 2)),
                ).mean()
            ) / processedDataDf.rolling(
                window=params["data"]["historyWindow"],
                min_periods=int(np.ceil(params["data"]["historyWindow"] / 2)),
            ).std()
        else:
            processedDf = processedDataDf
        processedDf = processedDf.dropna(how='all', axis=1)

        for c in processedDf.columns:
            if processedDf[[c]].empty:
                processedDf[c] = 0
                continue
            ind = processedDf[c].index.get_loc(processedDf[c].first_valid_index())
            processedDf[c].iloc[ind:].fillna(method="ffill", inplace=True)
        
        processedDf = pd.DataFrame(
            processedDf.loc[processedDf.index.intersection(tradingDate.index), :]
        )
    elif params["data"]["source"] == "emd":
        if params["data"]["operation"] == "return":
            emdDataDf = ((1 + dataDf).cumprod()) * 100
        else:
            emdDataDf = dataDf
        # get trading date
        processedDf = emdDataDf #getEMDTrend(emdDataDf, tradingDate, params)
    elif params["data"]["source"] == "ols":
        if params["data"]["operation"] == "return":
            emaData = ((1 + dataDf).cumprod()) * 100
        elif params["data"]["operation"] == "compReturn":
            emaData = dataDf
        processedDf = getOLSData(emaData, tradingDate, params)
    return processedDf

"""
def getEMDResidual(dataSeries, maxIMF=3):
    imf = emd.sift.sift(dataSeries, max_imfs=maxIMF)  # , imf_opts={'energy_thresh':50}
    residual = dataSeries
    for i in range(imf.shape[1]):
        residual = residual - imf[:, i]
    residualDf = pd.DataFrame(residual, columns=[dataSeries.name])
    return residualDf
"""
"""
def getEMDTrend(dataDf, tradingDate, params, maxIMF=3):
    emdTrendList = []
    logReturnDf = np.log(dataDf)  # .dropna(how='all', axis=0)
    for c in dataDf.columns:
        logging.debug(c)
        cList = []
        preProcessedDf = logReturnDf[c].reindex(dataDf.index, fill_value=np.nan)
        for d in tradingDate.tradingDate:
            if (
                d < preProcessedDf.dropna().index.min()
                or d > preProcessedDf.index.max()
            ):
                continue
            ind = preProcessedDf.index.get_loc(d)
            if ind - params["data"]["emdWindow"] < preProcessedDf.index.get_loc(
                preProcessedDf.dropna().index.min()
            ):
                continue
            
			#emdModel = emd.EMD()
			#emdModel.emd(preProcessedDf.iloc[ind-params['data']['emdWindow']:ind].dropna().values, max_imf=maxIMF)
			#emdIMFs, emdTrend = emdModel.get_imfs_and_residue()
			#emdTrend = pd.DataFrame(emdTrend, index = preProcessedDf.iloc[ind-params['data']['emdWindow']:ind].dropna().index, columns = [c])
			
            emdTrend = getEMDResidual(
                preProcessedDf.iloc[ind - params["data"]["emdWindow"] : ind].dropna(),
                maxIMF,
            )
            emdTrend = emdTrend.reindex(
                preProcessedDf.iloc[ind - params["data"]["emdWindow"] : ind].index,
                fill_value=np.nan,
            ).fillna(method="ffill")
            emdDf = (
                pd.DataFrame(emdTrend)
                / pd.DataFrame(emdTrend).shift(params["data"]["signalWindow"])
                - 1
            )
            if params["data"]["threshold"] != 0:
                emdDf = (
                    emdDf
                    - emdDf.rolling(
                        window=params["data"]["historyWindow"],
                        min_periods=int(
                            np.cell(params["data"]["historyWindow"] / 2)
                        ).mean(),
                    )
                ) / emdDf.rolling(
                    window=params["data"]["historyWindow"],
                    min_periods=int(np.cell(params["data"]["historyWindow"] / 2)).std(),
                )
            cList.append(pd.DataFrame(emdDf.iloc[-1]).T)
            cDf = pd.concat(cList, axis=0)
        emdTrendList.append(cDf)
    emdTrendDf = pd.concat(emdTrendList, axis=1)
    # emdTrendDf.index = dataDf.index
    return emdTrendDf
"""

def getOLSData(dataDf, tradingDate, params):
    dataRes = []
    dataR2 = []
    dataT = []
    emaData = dataDf.ewm(
        span=params["data"]["signalWindow"],
        min_periods=np.ceil(params["data"]["signalWindow"]),
    ).mean()
    for c in dataDf.columns:
        logging.debug(c)
        colRes = []
        colR2 = []
        colT = []
        colDate = []
        preProcessedDf = emaData[c].reindex(dataDf.index, fill_value=np.nan)
        for d in tradingDate.tradingDate:
            if (
                d < preProcessedDf.dropna().index.min()
                or d > preProcessedDf.index.max()
            ):
                continue
            ind = preProcessedDf.index.get_loc(d)
            if ind - params["data"]["signalWindow"] < preProcessedDf.index.get_loc(
                preProcessedDf.dropna().index.min()
            ):
                continue
            # for ind in range(0, dataDf.loc[:, c].shape[0]-params['data']['signalWindow']+1):
            lsResults = LS(
                preProcessedDf.iloc[ind - params["data"]["signalWindow"] : ind],
                np.array(list(range(1, params["data"]["signalWindow"] + 1))),
                constant=True,
            ).fit()
            colRes.append(lsResults.params["x0"])
            colR2.append(lsResults.rsquared)
            colT.append(lsResults.tvalues["x0"])
            colDate.append(d)
        dataRes.append(pd.DataFrame(colRes, columns=[c], index=colDate))
        dataR2.append(pd.DataFrame(colR2, columns=[c], index=colDate))
        dataT.append(pd.DataFrame(colT, columns=[c], index=colDate))
    return {
        "res": pd.concat(dataRes, axis=1),
        "r2": pd.concat(dataR2, axis=1),
        "t": pd.concat(dataT, axis=1),
    }


def regressBeta(beta, flowPct):
    # prepare data for (-1,1)
    nanInd = flowPct[flowPct.isnull()].index.tolist()
    x = beta[flowPct[~flowPct.isnull()].index].values
    y = flowPct[~flowPct.isnull()].values
    regressor = LR()
    regressor.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
    k = regressor.coef_
    res = regressor.intercept_
    predY = regressor.predict(np.array(x).reshape(-1, 1))
    adjFlowPct = res + (np.array(y).reshape(-1, 1) - predY)
    adjFlow = pd.DataFrame(
        adjFlowPct, index=flowPct[~flowPct.isnull()].index, columns={flowPct.name}
    ).T
    adjFlow[nanInd] = np.nan
    adjFlow = adjFlow[flowPct.index.tolist()]
    return adjFlow


def adjustBeta(price, flow, params=None):
    beta = price.rolling(window=params["data"]["signalWindow"]).std().dropna()
    flowData = copy.deepcopy(flow)
    flowData = flowData.reset_index()
    flowData = flowData.drop("date", 1)
    flowData["date"] = pd.to_datetime(flowData["DayEnding"], format="%Y%m%d")
    flowData = flowData.set_index("date")
    flowData = flowData.loc[beta.index.intersection(flowData.index), beta.columns]
    beta = beta.loc[beta.index.intersection(flow.index)]
    adjustList = []
    for r in range(beta.shape[0]):
        rFlow = flowData.iloc[r]
        rBeta = beta.iloc[r]
        adjustList.append(regressBeta(rBeta, rFlow))
    adjustFlow = pd.concat(
        adjustList, axis=0
    )  # pd.DataFrame(np.array(adjustList), columns=beta.columns, index=beta.index)
    return adjustFlow  # adjustFlow


def get_return_vol(data, params):
    if params["trading"]["frequency"] == "D":
        returnScale = 261
        stdScale = np.sqrt(261)
    elif params["trading"]["frequency"] == "W":
        returnScale = 52
        stdScale = np.sqrt(52)
    elif params["trading"]["frequency"] == "M":
        returnScale = 12
        stdScale = np.sqrt(12)
    elif params["trading"]["frequency"] == "Q":
        returnScale = 4
        stdScale = np.sqrt(4)

    # returns = np.log(data.astype(float)) - np.log(data.shift(1).astype(float))
    # returns = returns.loc[:, ~returns.columns.duplicated()]
    day_vol = data.ewm(
        ignore_na=False, adjust=True, com=params["data"]["signalWindow"], min_periods=0
    ).std(bias=False)
    # vol = day_vol * np.sqrt(261) # annualise

    volDf = day_vol * np.sqrt(
        261
    )  # np.sqrt(data.rolling(window=params['data']['signalWindow']).std())*stdScale
    return volDf

"""
def garch11(dataDf):
    # compute garch11 coefficients
    coefList = []
    for i in range(0, dataDf.shape[1]):
        try:
            garch11 = arch_model(dataDf.iloc[:, i], mean="Zero", vol="GARCH", p=1, q=1)
            res = garch11.fit(update_freq=5, disp="off", show_warning=False)
            coef = pd.DataFrame.from_dict(
                {
                    "delta": res.params["alpha[1]"],
                    "gamma": res.params["beta[1]"],
                    "omega": res.params["omega"],
                },
                orient="index",
                columns=[dataDf.columns[i]],
            )
            coefList.append(coef)
        except Exception as e:
            logging.error(e)
            return
    garchCoef = pd.concat(coefList, axis=1)

    # historical mean of each index
    h0 = pd.DataFrame(
        dataDf.pow(2, axis=1).mean(axis=0), columns=["historicalMean"]
    ).T  # historical variance
    # expVar(1, :) =  coef(3,:) + coef(1,:).*h0.^2 + coef(2,:).*h0
    expVarDf = pd.DataFrame(
        h0.apply(
            lambda row: garchCoef[row.name].T["omega"]
            + garchCoef[row.name].T["delta"] * row.historicalMean * row.historicalMean
            + garchCoef[row.name].T["gamma"] * row.historicalMean,
            axis=0,
        )
    ).T
    expVarDf["date"] = dataDf.index[0]
    expVarDf = expVarDf.set_index("date")
    # tmpDf = garchCoef.loc['omega'] + dataDf.pow(2)*garchCoef.loc['delta'] + dataDf*garchCoef.loc['gamma']
    # tmpDf = tmpDf.iloc[1:-1]
    expVar = [expVarDf]

    for i in range(0, dataDf.shape[0] - 1):
        tmpDf = pd.DataFrame(
            garchCoef.loc["omega"].values
            + garchCoef.loc["delta"].values * (dataDf.iloc[i].values ** 2)
            + garchCoef.loc["gamma"].values * expVar[i].values
        )
        tmpDf["date"] = dataDf.index[i + 1]
        tmpDf = tmpDf.set_index("date")
        tmpDf.columns = dataDf.columns
        expVar.append(tmpDf)
    expVar = pd.concat(expVar, axis=0)
    # expVar = pd.concat([tmpDf.shift(1), expVar], axis=0)
    expHistVolatility = expVar.pow(0.5)

    expVolatility = pd.DataFrame(
        garchCoef.pow(0.5).loc["omega"].values
        + garchCoef.loc["delta"].values * (dataDf.iloc[-1].values ** 2)
        + garchCoef.loc["gamma"].values * expVar.iloc[-1].values
    ).T
    expVolatility.columns = dataDf.columns

    return {
        "coef": garchCoef,
        "expectedVol": expVolatility,
        "volatility": expHistVolatility,
    }


def get_return_garch_vol(dataDf):
    dataVol = []
    for c in dataDf.columns:
        garchDict = garch11(pd.DataFrame(dataDf[c].dropna()))
        dataVol.append(garchDict["volatility"])
    dataVol = pd.concat(dataVol, axis=1)
    display(dataVol)
    # dataVol.index = dataDf.dropna().index
    return dataVol
"""


def get_price_to_trend(dataDf, params):
    EMA = dataDf.ewm(
        span=params["data"]["signalWindow"], min_periods=params["data"]["signalWindow"]
    ).mean()
    # MA = dataDf.rolling(window=params['data']['signalWindow']).mean().dropna()

    dev = dataDf - EMA
    z = dev / dev.std(axis=0)
    return z
