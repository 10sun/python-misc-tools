'''
Author: J , jwsun1987@gmail.com
Date: 2024-04-23 22:18:50
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd
import numpy as np
import datetime

# 1. generate data for different time window
def get_data(raw_data, frequency:str='D'):
    ds = ds.copy()
    # Daily sentiment
    ds['TIMESTAMP_UTC'] = pd.to_datetime(ds['TIMESTAMP_UTC'])
    # Convert From UTC to Americas
    ds['TIMESTAMP_TZ'] =  ds['TIMESTAMP_UTC'].apply(lambda x: convert_datetime_timezone(x, tz1 = "UTC", tz2 = "UTC"))
    # Advance time 
    ds['TIMESTAMP_TZ'] = pd.to_datetime(ds['TIMESTAMP_TZ'])
    ds['INFOSETDAY'] = ds['TIMESTAMP_TZ'] + timedelta(hours=addh) 
    ds['DATE'] = pd.to_datetime(ds.INFOSETDAY).dt.date  
    ds['DATE'] = pd.to_datetime(ds.DATE)
            
    # Take sentiment per news article and then for a given day the average sentiment across all news articles
    sent = ds.groupby(['RP_DOCUMENT_ID', 'DATE'])['COMPOSITE_SENTIMENT_SCORE'].mean().reset_index()
    sent = sent.groupby(['DATE'])['COMPOSITE_SENTIMENT_SCORE'].mean().reset_index()
    sent.sort_values(by = "DATE")
    
    # Add missing dates and fill zeros where no sentiment 
    dates = pd.DataFrame(data = pd.date_range(start = pd.to_datetime(from_date[0:10]), 
                                           end = pd.to_datetime(to_date[0:10])), columns = ['DATE'])

    sent = sent.merge(dates, on = 'DATE', how = 'right')
    sent['CSS'] = sent['COMPOSITE_SENTIMENT_SCORE'].replace(np.nan, 0)
    
    # rolling weekly, monthly scores
    sent['CSS_W'] = (sent['CSS']).rolling(7).mean()
    sent['CSS_M'] = (sent['CSS']).rolling(30).mean()
    sent['CSS_E30'] = sent['CSS'].ewm(span = 30, adjust = False, min_periods = 30).mean()           
    return sent



# 2. run the signal with portfolio at different trading frequency

# 3. analyse the results

