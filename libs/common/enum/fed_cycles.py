'''
Author: J , jwsun1987@gmail.com
Date: 2023-12-08 00:28:20
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''
# %% TODO: change dates from str to datetime

FED_HIKES_CUTS = {
    'Cuts':[ # in the next 6M post last hike
        ['1963-07-17', '1966-11-02', '1966-12-01'], # 1M
        ['1972-03-21', '1974-07-16', '1974-11-19'], # 4M # '1973-11-01', #1972-02, 1974-07 # 1973-10-01
        ['1976-04-21', '1980-03-18', '1980-05-15'], # 2M '1980-01-31', # 1977-01, 1980-04
        ['1980-08-07', '1981-05-18', '1981-11-17'], # '1981-07-31', 2M #1980-07, 1981-06 # 1981-05
        ['1983-03-31', '1984-08-09', '1984-10-02'], # 2M
        ['1994-02-04', '1995-02-01', '1995-07-06'], # 5M
    ],
    'Flat':[ # no cuts in the next 6M post last hike
        ['1967-11-21', '1969-09-16', '1970-05-01'], # '1969-12-01', 3M       
        ['1988-03-29', '1989-05-16', '1989-12-19'], # '1990-07-31', 13M
        ['1999-06-30', '2000-05-16', '2001-01-03'], # '2001-03'
        ['2004-06-30', '2006-06-29', '2007-09-18'], # '2007-12-01', 18M
        ['2015-12-16', '2018-12-19', '2019-08-01'], # '2020-02-29',
    ],
}

FED_HIKES_RECESSION = {
    'Soft-landing':[
        ['1963-07-17', '1966-11-02', '1966-12-01', None],
        ['1983-03-31', '1984-08-09', '1984-10-02', None],
        ['1994-02-04', '1995-02-01', '1995-07-06', None],
    ],
    'Recession':[
        ['1967-11-21', '1969-09-16', '1970-05-01', '1969-12-01'], # '1969-12-01', 3M       
        ['1972-03-21', '1974-07-16', '1974-11-19', '1973-11-01'], # '1973-11-01', #1972-02, 1974-07 # 1973-10-01
        ['1976-04-21', '1980-03-18', '1980-05-15', '1980-01-01'], # '1980-01-31', # 1977-01, 1980-04
        ['1980-08-07', '1981-05-18', '1981-11-17', '1981-07-01'], # '1981-07-31', 2M #1980-07, 1981-06 # 1981-05
        ['1999-06-30', '2000-05-16', '2001-01-03', '2001-03-01'], # '2001-03'
    ],
    'Delayed Recession':[
        ['1988-03-29', '1989-05-16', '1989-12-19', '1990-07-01'], # '1990-07-31', 13M
        ['2004-06-30', '2006-06-29', '2007-09-18', '2007-12-01'], # '2007-12-01', 18M
        ['2015-12-16', '2018-12-19', '2019-08-01', '2020-02-01'], # '2020-02-29',
    ],
}