'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-17 18:01:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import numpy as np
from numpy.lib.function_base import quantile
import pandas as pd
from math import e
from datetime import datetime as dt
#from datetime import timedelta
from plotly import express as px

from pathlib import Path
import sys

import preprocess as prep
import backtesting as test
import signals as signal
import dataIO
try:
	import pyfolio as pf
	import empyrical as ep
except ImportError:
	sys.path.append(r"\\merlin\lib_dpm\code\libs\empyrical-0.5.5")
	sys.path.append(r"\\merlin\lib_dpm\code\libs\pyfolio-0.9.2")
	import pyfolio as pf
	import empyrical as ep

from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib import figure
#from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
from IPython.display import display


def plot_annual_returns(returns, benchReturns=None, ax=None, **kwargs):
	"""
	Plots a bar graph of returns by year.

	Parameters
	----------
	returns : pd.Series
		Daily returns of the strategy, noncumulative.
		 - See full explanation in tears.create_full_tear_sheet.
	ax : matplotlib.Axes, optional
		Axes upon which to plot.
	**kwargs, optional
		Passed to plotting function.

	Returns
	-------
	ax : matplotlib.Axes
		The axes that were plotted on.
	"""

	if ax is None:
		ax = plt.gca()

	x_axis_formatter = FuncFormatter(pf.utils.percentage)
	ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
	ax.tick_params(axis='x', which='major')

	ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, 'yearly'))

	ax.axvline(100 * ann_ret_df.values.mean(),
			   color='coral',
			   linestyle='--',
			   lw=4,
			   alpha=0.7)
	(100 * ann_ret_df.sort_index(ascending=False)).plot(ax=ax,
														color='coral',
														kind='barh',
														alpha=0.70,
														legend=True,
														**kwargs)

	if benchReturns is not None:
		bench_ret_df = pd.DataFrame(
			ep.aggregate_returns(benchReturns, 'yearly'))
		ax.axvline(100 * bench_ret_df.values.mean(),
				   color='b',
				   linestyle='--',
				   lw=4,
				   alpha=0.7)
		(100 * bench_ret_df.sort_index(ascending=False)).plot(ax=ax,
															  color='b',
															  kind='barh',
															  alpha=0.60,
															  legend=True,
															  **kwargs)

	ax.axvline(0.0, color='black', linestyle='-', lw=3)

	ax.set_ylabel('Year', **kwargs)
	ax.set_xlabel('Returns', **kwargs)
	ax.set_title("Annual returns", **kwargs)
	ax.legend(frameon=True, framealpha=0.5, **kwargs)
	return ax


def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
	"""
	Plots a heatmap of returns by month.

	Parameters
	----------
	returns : pd.Series
		Daily returns of the strategy, noncumulative.
		 - See full explanation in tears.create_full_tear_sheet.
	ax : matplotlib.Axes, optional
		Axes upon which to plot.
	**kwargs, optional
		Passed to seaborn plotting function.

	Returns
	-------
	ax : matplotlib.Axes
		The axes that were plotted on.
	"""

	if ax is None:
		ax = plt.gca()

	annot_kws = kwargs.get("annot_kws") if 'annot_kws' in kwargs.keys() else 12
	font_size = kwargs.get("fontsize") if 'fontsize' in kwargs.keys() else 12
	title_str = kwargs.get("titleStr") if 'titleStr' in kwargs.keys() else ''

	monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
	monthly_ret_table = monthly_ret_table.unstack().round(3)

	sns.heatmap(monthly_ret_table.fillna(0) * 100.0,
				annot=True,
				annot_kws={"size": annot_kws},
				alpha=1.0,
				center=0.0,
				cbar=False,
				cmap=matplotlib.cm.RdYlGn,
				ax=ax)
	ax.set_ylabel('Year', fontsize=font_size)
	ax.set_xlabel('Month', fontsize=font_size)
	ax.tick_params(labelsize=font_size)
	ax.set_yticklabels(ax.get_ymajorticklabels(),
					   va='center',
					   fontsize=font_size - 2)
	ax.set_title(title_str + " monthly returns (%)", fontsize=font_size)
	return ax


def plot_monthly_returns_dist(returns, ax=None, **kwargs):
	"""
	Plots a distribution of monthly returns.

	Parameters
	----------
	returns : pd.Series
		Daily returns of the strategy, noncumulative.
		 - See full explanation in tears.create_full_tear_sheet.
	ax : matplotlib.Axes, optional
		Axes upon which to plot.
	**kwargs, optional
		Passed to plotting function.

	Returns
	-------
	ax : matplotlib.Axes
		The axes that were plotted on.
	"""

	if ax is None:
		ax = plt.gca()

	font_size = kwargs.get("fontsize") if 'fontsize' in kwargs.keys() else 12
	title_str = kwargs.get("titleStr") if 'titleStr' in kwargs.keys() else ''

	x_axis_formatter = FuncFormatter(pf.utils.percentage)
	ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
	ax.tick_params(axis='x', which='major')

	monthly_ret_table = ep.aggregate_returns(returns, 'monthly')

	ax.hist(100 * monthly_ret_table, color='orangered', alpha=0.80, bins=20)

	ax.axvline(100 * monthly_ret_table.mean(),
			   color='gold',
			   linestyle='--',
			   lw=4,
			   alpha=1.0)

	ax.axvline(0.0, color='black', linestyle='-', lw=3, alpha=0.75)
	ax.legend([title_str + ' Mean'],
			  frameon=True,
			  framealpha=0.5,
			  fontsize=font_size)
	ax.set_ylabel('Number of months', fontsize=font_size)
	ax.set_xlabel('Returns', fontsize=font_size)
	ax.tick_params(labelsize=font_size)
	ax.set_title("Distribution of " + title_str + " monthly returns",
				 fontsize=font_size)
	return ax


def plot_drawdown_periods(returns, top=10, ax=None, legendStr=None, **kwargs):
	"""
	Plots cumulative returns highlighting top drawdown periods.

	Parameters
	----------
	returns : pd.Series
		Daily returns of the strategy, noncumulative.
		 - See full explanation in tears.create_full_tear_sheet.
	top : int, optional
		Amount of top drawdowns periods to plot (default 10).
	ax : matplotlib.Axes, optional
		Axes upon which to plot.
	**kwargs, optional
		Passed to plotting function.

	Returns
	-------
	ax : matplotlib.Axes
		The axes that were plotted on.
	"""
	if ax is None:
		ax = plt.gca()
	if legendStr is None:
		legendStr = 'Portfolio'

	font_size = kwargs.get("fontsize") if 'fontsize' in kwargs.keys() else 12
	title_str = kwargs.get("titleStr") if 'titleStr' in kwargs.keys() else ''
	line_color = kwargs.get(
		"lineColor") if 'lineColor' in kwargs.keys() else 'b'
	line_width = kwargs.get("lineWidth") if 'lineWidth' in kwargs.keys() else 5

	ax.margins(x=0)

	y_axis_formatter = FuncFormatter(pf.utils.two_dec_places)
	ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

	df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
	df_drawdowns = pf.timeseries.gen_drawdown_table(returns, top=top)

	df_cum_rets.plot(ax=ax, linewidth=line_width, color=line_color)

	lim = ax.get_ylim()
	colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
	for i, (peak, recovery) in df_drawdowns[['Peak date',
											 'Recovery date']].iterrows():
		if pd.isnull(recovery):
			recovery = returns.index[-1]
		ax.fill_between((peak, recovery),
						lim[0],
						lim[1],
						alpha=.4,
						color=colors[i])
	ax.set_ylim(lim)
	ax.set_title('Top %i drawdown periods' % top, fontsize=font_size)
	ax.set_ylabel('Cumulative returns', fontsize=font_size)
	ax.legend([legendStr],
			  loc='upper left',
			  frameon=True,
			  framealpha=0.5,
			  fontsize=font_size)
	ax.set_xlabel('')
	return ax


def plot_drawdown_underwater(returns, ax=None, legendStr=None, **kwargs):
	"""
	Plots how far underwaterr returns are over time, or plots current
	drawdown vs. date.

	Parameters
	----------
	returns : pd.Series
		Daily returns of the strategy, noncumulative.
		 - See full explanation in tears.create_full_tear_sheet.
	ax : matplotlib.Axes, optional
		Axes upon which to plot.
	**kwargs, optional
		Passed to plotting function.

	Returns
	-------
	ax : matplotlib.Axes
		The axes that were plotted on.
	"""

	if ax is None:
		ax = plt.gca()
	if legendStr is None:
		legendStr = 'Portfolio'

	font_size = kwargs.get("fontsize") if 'fontsize' in kwargs.keys() else 12
	title_str = kwargs.get("titleStr") if 'titleStr' in kwargs.keys() else ''
	line_color = kwargs.get(
		"lineColor") if 'lineColor' in kwargs.keys() else 'b'

	ax.margins(x=0)

	y_axis_formatter = FuncFormatter(pf.utils.percentage)
	ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

	df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
	running_max = np.maximum.accumulate(df_cum_rets)
	underwater = -100 * ((running_max - df_cum_rets) / running_max)
	(underwater).plot(ax=ax, kind='area', alpha=0.7, color=line_color)
	ax.set_ylabel('Drawdown', fontsize=font_size)
	ax.set_title('Underwater plot', fontsize=font_size)
	ax.legend([legendStr],
			  loc='upper left',
			  frameon=True,
			  framealpha=0.5,
			  fontsize=font_size)
	ax.set_xlabel('')
	return ax


def adjust_yaxis(ax, ydif, v):
	"""shift axis ax by ydiff, maintaining point v at the same location"""
	inv = ax.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
	miny, maxy = ax.get_ylim()
	miny, maxy = miny - v, maxy - v
	if -miny > maxy or (-miny == maxy and dy > 0):
		nminy = miny
		nmaxy = miny * (maxy + dy) / (miny + dy)
	else:
		nmaxy = maxy
		nminy = maxy * (miny + dy) / (maxy + dy)
	ax.set_ylim(nminy + v, nmaxy + v)
	return (nminy + v, nmaxy + v)


def yaxisScale(ax1, ax2, y1c, y2c, y1Base=None, y2Base=None):
	y1u = max(ax1.get_ylim())
	y1b = min(ax1.get_ylim())
	y2u = max(ax2.get_ylim())
	y2b = min(ax2.get_ylim())

	y1B = np.power(10, np.floor(np.log10(y1u - y1c))) if not y1Base else y1Base
	y2B = np.power(10, np.floor(np.log10(y2u - y2c))) if not y2Base else y2Base
	#logging.info(str(y1u)+', ' + str(y1c) + ', ' + str(y1b) + ', ' + str(y1B))
	#logging.info(str(y2u)+', ' + str(y2c) + ', ' + str(y2b) + ', ' + str(y2B))
	yU = max(np.ceil(abs(y1u - y1c) / y1B), np.ceil(abs(y2u - y2c) / y2B))
	yB = max(np.ceil(abs(y1b - y1c) / y1B), np.ceil(abs(y2b - y2c) / y2B))
	ax1.set_ylim(y1c - yB * y1B, y1c + yU * y1B)
	ax2.set_ylim(y2c - yB * y2B, y2c + yU * y2B)
	ax1.tick_params(axis='both', which='major', labelsize=26)
	ax2.tick_params(axis='both', which='major', labelsize=26)


def adjustXticks(ax, freq='Y', dataFreq='M'):
	adjXlabels = []  # = ticklabels = ['']*len(xlabels)
	newMajorLoc = []
	for xInd, xL in enumerate(ax.get_xticklabels()):
		if (pd.to_datetime(xL.get_text()).month == 1) and freq == 'Y':
			if str(pd.to_datetime(xL.get_text()).year) not in adjXlabels:
				adjXlabels.append(str(pd.to_datetime(xL.get_text()).year))
				newMajorLoc.append(xInd)
		elif (pd.to_datetime(xL.get_text()).month in [1, 4, 7, 10
													  ]) and freq == 'Q':
			if (str(pd.to_datetime(xL.get_text()).year) + '-' + str(
					pd.to_datetime(xL.get_text()).month)) not in adjXlabels:
				adjXlabels.append(
					str(pd.to_datetime(xL.get_text()).year) + '-' +
					str(pd.to_datetime(xL.get_text()).month))
				newMajorLoc.append(xInd)
		elif (pd.to_datetime(xL.get_text()).day == 1) and freq == 'M':
			if str(pd.to_datetime(xL.get_text()).date()) not in adjXlabels:
				adjXlabels.append(str(pd.to_datetime(xL.get_text()).date()))
				newMajorLoc.append(xInd)
	ax.set_xticks(newMajorLoc)
	return adjXlabels


def plotLineAndBar(lineDf,
				   barDf,
				   figsize=[36, 10],
				   ax1c=0,
				   ax2c=100,
				   xfreq='Y',
				   stacked=True,
				   baseline=False,
				   ax=None):
	if ax:
		ax1 = ax
	ax1 = barDf.plot.bar(figsize=figsize, stacked=stacked, rot=0, legend=False)
	ax1.set_xticklabels(adjustXticks(ax1, xfreq))
	ax1.margins(x=0)
	ax2 = ax1.twinx()
	lineDf = pd.DataFrame(lineDf) if not isinstance(lineDf,
													pd.DataFrame) else lineDf
	for c in lineDf.columns:
		ax2.plot(lineDf[c].values, label=c, linewidth=5)

	ax1.grid(True)
	ax2.grid(False)

	#if :
	#yaxisScale(ax1, ax2, ax1c, ax2c, 0.1, 10)
	#else:
	yaxisScale(ax1, ax2, ax1c, ax2c, y2Base=10)

	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	handles = h1 + h2
	labels = l1 + l2
	if len(labels) <= 8:
		ax1.legend(handles,
				   labels,
				   bbox_to_anchor=(.5, 1.05),
				   loc='center',
				   ncol=len(labels),
				   fontsize=26,
				   framealpha=1,
				   facecolor='white')
	else:
		ax1.legend(h2,
				   l2,
				   bbox_to_anchor=(.5, 1.05),
				   loc='center',
				   ncol=len(labels),
				   fontsize=26,
				   framealpha=1,
				   facecolor='white')

	if baseline:
		ax1.axhline(y=ax1c, linestyle='dashed', color='r', linewidth=2.5)
	if xfreq != 'Y':
		plt.gcf().autofmt_xdate()
	plt.tight_layout()
	#plt.show()
	#return plt.gca()


def combineImages(imgNames, orientation='vertical'):
	images = [Image.open(x) for x in imgNames]
	#widths, heights = zip(*(i.size for i in images))

	imgs_comb = np.vstack(
		(np.asarray(i)
		 for i in images)) if orientation == 'vertical' else np.hstack(
			 (np.asarray(i) for i in images))
	new_im = Image.fromarray(imgs_comb)
	"""
	total_width = sum(widths)
	max_height = max(heights)

	new_im = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	for im in images:
	  new_im.paste(im, (x_offset,0))
	  x_offset += im.size[0

	"""
	return new_im


def visualize_results(tradingPrice, tradingSignal, returns, params):
	if params['trading']['strategy'] == 'lo':
		figCM = 'Blues'
	elif params['trading']['strategy'] == 'ls':
		figCM = 'coolwarm_r'

	allNames = []

	# plot portfolio return
	plotLineAndBar(returns['cumSummary'][['Portfolio-bench', 'Portfolio-strategy']], \
		  returns['summary'].loc[:, (returns['summary'].columns.str.contains('-active'))&(~returns['summary'].columns.str.contains('Portfolio-active'))])
	plt.gcf().suptitle(
		params['performance']['expCode'], fontsize=30,
		y=1)  #(str(fDir).split('\\')[-2] + str(fDir).split('\\')[-1])
	allFn = Path(
		params['performance']['dir']) / ('portfolio-cumulativeReturn.png')
	plt.savefig(allFn)
	allNames.append(allFn)
	plt.close()

		
	# plot signal heatmap
	plt.figure(figsize=(36, 18))
	tradeSigVis = tradingSignal.T
	tradeSigVis.columns = [str(c.date()) for c in tradeSigVis.columns]
	sns.set(font_scale=2.5)
	snsFig = sns.heatmap(tradeSigVis * 100.0,
						 annot=False,
						 annot_kws={"size": 8},
						 alpha=1.0,
						 center=0.0,
						 cbar=False,
						 cmap=figCM)
	signalFn = Path(params['performance']['dir']) / (
		params['performance']['expCode'].replace('/', '_') + '_signal.png')
	plt.savefig(signalFn)
	allNames.append(signalFn)
	plt.close()

	from plotly.offline import plot
	from plotly.subplots import make_subplots

	figures = [
				px.line(tradingPrice[tradingSignal.columns]),
				px.imshow(tradeSigVis * 100.0)
		]

	fig = make_subplots(rows=len(figures), cols=1) 

	for i, figure in enumerate(figures):
		for trace in range(len(figure["data"])):
			fig.append_trace(figure["data"][trace], row=i+1, col=1)
	fig.write_html(Path(params['performance']['dir']) / (params['performance']['expCode'].replace('/', '_') + '_signal.html'))
	print(Path(params['performance']['dir']) / (params['performance']['expCode'].replace('/', '_') + '_signal.html'))
	
	# plot drawdown figure
	fig = plt.figure(figsize=(36, 2*12))
	gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.2)
	ax_benchDd = plt.subplot(gs[0,:])
	ax_portDd = plt.subplot(gs[1,:])
	plot_drawdown_periods(returns['return']['benchmark']['Portfolio'], top=5, ax=ax_benchDd, legendStr='Benchmark',  fontsize=26)
	plot_drawdown_periods(returns['return']['strategy']['Portfolio'], top=5, ax=ax_portDd, legendStr='Strategy',  fontsize=26, lineColor='coral')
	fig.suptitle(params['performance']['expCode'], fontsize=30, y=0.92)
	drawDownFn = Path(params['performance']['dir'])/('max_drawdown_periods.png')
	plt.savefig(drawDownFn)
	allNames.append(drawDownFn)
	plt.close()

	# plot underwarter figure
	fig = plt.figure(figsize=(36, 2*12))
	gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.2)
	ax_benchUw = plt.subplot(gs[0,:])
	ax_portUw = plt.subplot(gs[1,:])
	plot_drawdown_underwater(returns['return']['benchmark']['Portfolio'], ax=ax_benchUw, legendStr='Benchmark', fontsize=26)
	plot_drawdown_underwater(returns['return']['strategy']['Portfolio'], ax=ax_portUw, legendStr='Strategy', fontsize=26, lineColor='coral')
	fig.suptitle(params['performance']['expCode'], fontsize=30, y=0.92)
	underwaterFn = Path(params['performance']['dir'])/('drawdown_underwater.png')
	plt.savefig(underwaterFn)
	allNames.append(underwaterFn)
	plt.close()
	
	# plot monthly return heatmap
	fig = plt.figure(figsize=(36, 18))
	gs = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.2)
	ax_benchMonthlyRet = plt.subplot(gs[:, 0])
	ax_portMonthlyRet = plt.subplot(gs[:, 1])
	plot_monthly_returns_heatmap(returns['summary']['Portfolio-bench'],
								 ax=ax_benchMonthlyRet,
								 fontsize=26,
								 annot_kws=26,
								 titleStr='Becnmark')
	plot_monthly_returns_heatmap(returns['summary']['Portfolio-strategy'],
								 ax=ax_portMonthlyRet,
								 fontsize=26,
								 annot_kws=26,
								 titleStr='Strategy')
	fig.suptitle(params['performance']['expCode'], fontsize=30, y=0.92)
	monthlyRetFn = Path(
		params['performance']['dir']) / ('portfolio_monthly_return.png')
	plt.savefig(monthlyRetFn)
	allNames.append(monthlyRetFn)
	plt.close()

	# plot stats figure
	fig = plt.figure(figsize=(36, 18))
	gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)
	ax_monthly_heatmap = plt.subplot(gs[:, 0])
	ax_annual_returns = plt.subplot(gs[:, 1])
	ax_bench_monthly_dist = plt.subplot(gs[0, 2])
	ax_strat_monthly_dist = plt.subplot(gs[1, 2])
	plot_monthly_returns_heatmap(returns['summary']['Portfolio-strategy'],
								 ax=ax_monthly_heatmap,
								 fontsize=26,
								 annot_kws=26,
								 titleStr='Strategy')
	plot_annual_returns(returns['summary']['Portfolio-strategy'],
						returns['summary']['Portfolio-bench'],
						ax=ax_annual_returns,
						fontsize=26)
	plot_monthly_returns_dist(returns['summary']['Portfolio-bench'],
							  ax=ax_bench_monthly_dist,
							  fontsize=26,
							  titleStr='Benchmark')
	plot_monthly_returns_dist(returns['summary']['Portfolio-strategy'],
							  ax=ax_strat_monthly_dist,
							  fontsize=26,
							  titleStr='Strategy')
	fig.suptitle(params['performance']['expCode'], fontsize=30, y=0.93)
	returnStatsFn = Path(
		params['performance']['dir']) / ('portfolio_return_stats.png')
	plt.savefig(returnStatsFn)
	allNames.append(returnStatsFn)
	plt.close()


	# plot stats of active returns figure
	fig = plt.figure(figsize=(36, 18))
	gs = gridspec.GridSpec(1, 3, wspace=0.2, hspace=0.2)
	ax_monthly_heatmap = plt.subplot(gs[0, 0])
	ax_annual_returns = plt.subplot(gs[0, 1])
	ax_bench_monthly_dist = plt.subplot(gs[0, 2])
	plot_monthly_returns_heatmap(returns['summary']['Portfolio-active'],
								 ax=ax_monthly_heatmap,
								 fontsize=26,
								 annot_kws=26,
								 titleStr='Active Returns')
	plot_annual_returns(returns['summary']['Portfolio-active'],
						ax=ax_annual_returns,
						fontsize=26)
	plot_monthly_returns_dist(returns['summary']['Portfolio-active'],
							  ax=ax_bench_monthly_dist,
							  fontsize=26,
							  titleStr='Active Returns')
	fig.suptitle(params['performance']['expCode'], fontsize=30, y=0.93)
	actReturnStatsFn = Path(
		params['performance']['dir']) / ('portfolio_active_return_stats.png')
	plt.savefig(actReturnStatsFn)
	allNames.append(actReturnStatsFn)
	plt.close()

	if params['performance'].get('allFig', False):
		for cntInd, cnt in enumerate(tradingSignal.columns):
			#print(cnt)
			fig = plt.figure()
			plotLineAndBar(
				returns['cumSummary'][[cnt + '-bench', cnt + '-strategy']],
				returns['summary'].loc[:, cnt + '-active'],
				baseline=True)
			plt.gca().pcolorfast(
				fig.gca().get_xlim(),
				fig.gca().get_ylim(),
				tradingSignal[cnt].loc[
					tradingSignal[cnt].index <= returns['cumSummary'][
						[cnt + '-bench']].index.max()].values[np.newaxis],
				cmap=figCM,
				alpha=0.3)
			#plt.setp(ax1.get_xticklabels(), visible=True)
			#extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
			cntFn = Path(params['performance']['dir']) / (
				cnt.replace('/', ' ') + '_return_analysis.png')
			#print(cntFn)
			plt.savefig(cntFn)  #, bbox_inches=extent.expanded(1.1,1.3))
			allNames.append(cntFn)
			plt.close()

		new_im = combineImages(allNames)
		allFn = Path(params['performance']['dir']) / (
			params['performance']['expCode'] + '_return_analysis.png')
		new_im.save(allFn)

	return allNames


def visualizeTradingSignal(priceData, dataDf, params):
	signalData = prep.preprocess_data(dataDf, params)
	tradingDate = signal.get_trading_date(signalData, params)
	tradingSignal = signalData.loc[signalData.index.intersection(
		tradingDate.index)]
	tradingPrice = priceData.loc[priceData.index.intersection(
		tradingSignal.index)]

	dataIO.export_to_excel(
		{
			'data': tradingSignal,
			'price': tradingPrice
		},
		Path(params['performance']['dir']) /
		(params['performance']['expCode'].replace('/', '_') +
		 '_signalData.xlsx'))
	allNames = []
	plt.figure(figsize=(36, 18))
	tradeSigVis = signalData.T
	tradeSigVis.columns = [str(c.date()) for c in tradeSigVis.columns]
	sns.set(font_scale=2.5)
	snsFig = sns.heatmap(tradeSigVis,
						 annot=False,
						 annot_kws={"size": 12},
						 alpha=1.0,
						 center=0.0,
						 cbar=False,
						 cmap=matplotlib.cm.RdYlGn)
	signalFn = Path(params['performance']['dir']) / (
		params['performance']['expCode'].replace('/', '_') + '_signalData.png')
	plt.savefig(signalFn)
	allNames.append(signalFn)
	plt.close()

	for ind in signalData.columns:
		fig = plt.figure()
		tradingPrice[ind].plot(figsize=[24, 12], legend=True)
		signalData[ind].plot(secondary_y=True, legend=True)
		cntFn = Path(params['performance']['dir']) / (ind.replace('/', ' ') +
													  '_return_analysis.png')
		plt.savefig(cntFn)
		allNames.append(cntFn)
		plt.close()
	#new_im = combineImages(allNames)
	#allFn = Path(params['performance']['dir'])/(params['performance']['expCode'] + '_return_analysis.png')
	#new_im.save(allFn)
	return allNames


def get_mom_dashboard(priceData: pd.DataFrame,
					  signalData: pd.DataFrame,
					  params: dict = None):
	# configure the parameters
	if not params:
		params = ({
			'portfolio': {
				'assets':
				list(
					set(list(priceData.columns)).intersection(
						set(list(signalData.columns))))
			}
		})
	elif 'portfolio' not in params.keys():
		params.update({
			'portfolio': {
				'assets':
				list(
					set(list(priceData.columns)).intersection(
						set(list(signalData.columns))))
			}
		})
	elif 'assets' not in params['portfolio'].keys():
		params['portfolio'].update({
			'assets':
			list(
				set(list(priceData.columns)).intersection(
					set(list(signalData.columns))))
		})
	params = test.configure_params(params)
	# get trading signal
	if params['data']['signal'] != 'ranking':
		tradingSignal = test.get_trading_signal(signalData, params=params)
	else:
		tradingSignal = prep.preprocess_data(signalData, params).dropna(
			how='all', axis=0).rank(axis=1, ascending=False)
	#tradingSignalTrend = prep.getEMDTrend(signalData)
	#tradingSignalTrend = ((1+tradingSignalTrend.pct_change(0)).cumprod())*100
	# get portfolio data based on the trading date
	#tradingPrice = priceData.loc[priceData.index.intersection(tradingSignal.index), params['portfolio']['assets']]

	#signals = {'signal':tradingSignal, 'price':tradingPrice, 'trend':tradingSignalTrend,}
	return tradingSignal


###### ROUNDING ########
def one_dec_places(x, pos):
    """
    Adds 1/10th decimal to plot ticks.
    """
    return '%.1f' % x


def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """
    return '%.2f' % x

def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x