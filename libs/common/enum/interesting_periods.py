'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 00:51:42
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from typing import OrderedDict
import pandas as pd

PERIODS = OrderedDict()
# 70s

# 80s

# 90s

# Dotcom bubble
PERIODS['DotcomBubble'] = (pd.Timestamp('20000310'), pd.Timestamp('20000910'))

# Lehmann Brothers
PERIODS['Lehman'] = (pd.Timestamp('20080801'), pd.Timestamp('20081001'))

# 9/11
PERIODS['9/11'] = (pd.Timestamp('20010911'), pd.Timestamp('20011011'))

# 05/08/11  US down grade and European Debt Crisis 2011
PERIODS['US downgrade/European Debt Crisis'] = (pd.Timestamp('20110805'),
                                                pd.Timestamp('20110905'))

# 16/03/11  Fukushima melt down 2011
PERIODS['Fukushima'] = (pd.Timestamp('20110316'), pd.Timestamp('20110416'))

# 01/08/03  US Housing Bubble 2003
PERIODS['US Housing'] = (pd.Timestamp('20030108'), pd.Timestamp('20030208'))

# 06/09/12  EZB IR Event 2012
PERIODS['EZB IR Event'] = (pd.Timestamp('20120910'), pd.Timestamp('20121010'))

# August 2007, March and September of 2008, Q1 & Q2 2009,
PERIODS['Aug07'] = (pd.Timestamp('20070801'), pd.Timestamp('20070901'))
PERIODS['Mar08'] = (pd.Timestamp('20080301'), pd.Timestamp('20080401'))
PERIODS['Sept08'] = (pd.Timestamp('20080901'), pd.Timestamp('20081001'))
PERIODS['2009Q1'] = (pd.Timestamp('20090101'), pd.Timestamp('20090301'))
PERIODS['2009Q2'] = (pd.Timestamp('20090301'), pd.Timestamp('20090601'))

# Flash Crash (May 6, 2010 + 1 week post),
PERIODS['Flash Crash'] = (pd.Timestamp('20100505'), pd.Timestamp('20100510'))

# April and October 2014).
PERIODS['Apr14'] = (pd.Timestamp('20140401'), pd.Timestamp('20140501'))
PERIODS['Oct14'] = (pd.Timestamp('20141001'), pd.Timestamp('20141101'))

# Market down-turn in August/Sept 2015
PERIODS['Fall2015'] = (pd.Timestamp('20150815'), pd.Timestamp('20150930'))

# Market regimes
PERIODS['Low Volatility Bull Market'] = (pd.Timestamp('20050101'),
                                         pd.Timestamp('20070801'))

PERIODS['GFC Crash'] = (pd.Timestamp('20070801'), pd.Timestamp('20090401'))

PERIODS['Recovery'] = (pd.Timestamp('20090401'), pd.Timestamp('20130101'))

# COVID
PERIODS['Covid'] = (pd.Timestamp('20200301'), pd.Timestamp('20200331'))