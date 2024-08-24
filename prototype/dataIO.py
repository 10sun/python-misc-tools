'''
Author: J , jwsun1987@gmail.com
Date: 2022-02-17 22:23:34
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from typing import Union
import pandas as pd
import os
import logging
from datetime import datetime as dt
from pathlib import Path


def config_path(params: dict) -> dict:
    """configure the path

    Args:
        params (dict): a dictionary with parameters for different paths
                        e.g. params = {'data_path': 'the path for the data on the server',
                                        'result_path': 'the path for the results on the server',
                                        'data_backup_path': 'the path to backup the data on local disks, if needed',
                                        'result_backup_path': 'the path to backup the results on local disks, if needed'}

    Returns:
        dict: the configured dictionary
    """
    if not params:
        logging.error('missing parameters...')
        return None

    path_dict = {}
    try:
        data_path = Path(
            params.get('data_path', ValueError('No data path provided...')))
        result_path = Path(
            params.get('result_path',
                       ValueError('No result path provided...')))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        path_dict.update({'data_path': data_path, 'result_path': result_path})

        if params.get('backup', False):
            data_backup_path = Path(
                params.get('data_backup_path', params.get('data_path'))) / (
                    str(params.get('latest_date', dt.today()).year) +
                    str(params.get('latest_date', dt.today()).month).zfill(2)
                ) / str(params.get('latest_date', dt.today()).date()).replace(
                    '-', '')
            result_backup_path = Path(
                params.get('result_backup_path', params.get('result_path'))
            ) / (str(params.get('latest_date', dt.today()).year) + str(
                params.get('latest_date', dt.today()).month).zfill(2)) / str(
                    params.get('latest_date', dt.today()).date()).replace(
                        '-', '')
            if not os.path.exists(data_backup_path):
                os.makedirs(data_backup_path)
            if not os.path.exists(result_backup_path):
                os.makedirs(result_backup_path)
            path_dict.update({
                'data_backup_path': data_backup_path,
                'result_backup_path': result_backup_path
            })
    except Exception as e:
        logging.error(e)
        return None
    return path_dict


def export_to_excel(data: dict,
                    filepath: Union[str, Path],
                    params: dict = None):
    """ export a dict of data to an excel file

    Args:
        data (dict): a dict of data to export, each (key, value) pair will be a sheet of the exported excel file
        filepath (Union[str, Path]): the file path for the exported excel file
        params (dict, optional): params for the excel formating
    """
    if not params:
        params = {
            'date_format': 'yyyy.mm.dd',
            'datetime_format': 'yyyy.mm.dd',
            'overwrite': True,
            'multiIndex': False,
            'index_name': 'date'
        }

    if os.path.isfile(filepath):
        if params.get('overwrite', True):
            data_file = pd.ExcelWriter(
                filepath,
                engine='openpyxl',
                date_format=params.get('date_format', 'yyyy.mm.dd'),
                datetime_format=params.get('datetime_format', 'yyyy.mm.dd'),
                options={'strings_to_urls': False})
            for k, v in data.items():
                if not isinstance(v.index, pd.MultiIndex) and not params.get(
                        'multiIndex', False) and params.get(
                            'index_name', None) is not None:
                    v.index = v.index.set_names(
                        [params.get('index_name', None)])
                v.to_excel(data_file, sheet_name=k, index_label=False)
        else:
            curt_book = pd.read_excel(filepath, sheet_name=None)
            data_file = pd.ExcelWriter(
                filepath,
                engine='openpyxl',
                date_format=params.get('date_format', 'yyyy.mm.dd'),
                datetime_format=params.get('datetime_format', 'yyyy.mm.dd'),
                options={'strings_to_urls': False})
            for k, v in data.items():
                v = pd.DataFrame(v)
                if k in curt_book.keys():
                    if not params.get('overwrite_sheet', False):
                        dateIndex = True
                        curt_data = curt_book[k]
                        if isinstance(curt_data.columns[1], dt):
                            curt_data = curt_data.T
                            curt_data.columns = curt_data.iloc[0]
                            curt_data = curt_data.drop(curt_data.index[0])
                            curt_data.index = pd.to_datetime(curt_data.index)
                            curt_data.reset_index(inplace=True)
                            date_col = curt_data.select_dtypes(['datetime64'
                                                                ]).columns[0]
                            curt_data = curt_data.rename(
                                {date_col: params.get('index_name', 'date')},
                                axis=1)
                            curt_data = curt_data.set_index(
                                params.get('index_name', 'date'))
                            dateIndex = False
                        elif params.get('index_name',
                                        'date') in curt_data.columns:
                            curt_data = curt_data.set_index(
                                curt_data.columns[0]) if params.get(
                                    'index_name',
                                    None) is None else curt_data.set_index(
                                        params.get('index_name', None))
                        if isinstance(v.columns[0], dt):
                            v = v.T
                        new_cols = v[v.columns.difference(curt_data.columns)]
                        if not new_cols.empty:
                            new_data = pd.concat([curt_data, new_cols], axis=1)
                        new_data = pd.concat(
                            [curt_data[~curt_data.index.isin(v.index)], v],
                            axis=0).sort_index()
                        if params.get('index_name', None) is not None:
                            new_data.index = new_data.index.set_names(
                                [params.get('index_name', None)])
                        if dateIndex:
                            new_data.to_excel(data_file, k, index_label=False)
                        else:
                            new_data.T.to_excel(data_file,
                                                k,
                                                index_label=False)
                    else:
                        logging.info('overwriting sheet ' + k)
                        if not isinstance(
                                v.index, pd.MultiIndex) and not params.get(
                                    'multiIndex', False) and params.get(
                                        'index_name', None) is not None:
                            v.index = v.index.set_names(
                                [params.get('index_name', None)])
                        v.to_excel(data_file, sheet_name=k, index_label=False)
                else:
                    if not isinstance(v.index,
                                      pd.MultiIndex) and not params.get(
                                          'multiIndex', False) and params.get(
                                              'index_name', None) is not None:
                        v.index = v.index.set_names(
                            [params.get('index_name', None)])
                    v.to_excel(data_file, sheet_name=k, index_label=False)
    else:
        data_file = pd.ExcelWriter(
            filepath,
            engine='openpyxl',
            date_format=params.get('dateFormat', 'yyyy.mm.dd'),
            datetime_format=params.get('datetimeFormat', 'yyyy.mm.dd'),
            options={'strings_to_urls': False})
        for k, v in data.items():
            if not isinstance(v.index, pd.MultiIndex) and not params.get(
                    'multiIndex', False) and params.get('index_name',
                                                        None) is not None:
                v.index = v.index.set_names([params.get('index_name', None)])
            v.to_excel(data_file, sheet_name=k, index_label=False)
    data_file.close()
    logging.info(filepath)


def df_to_tableau(dataDf: pd.DataFrame,
                  kpi_str: str,
                  index_name: str = 'Date',
                  col_name: str = 'Index',
                  kpi_name: str = 'KPI',
                  value_name: str = 'Value',
                  additional_cols: dict = None) -> pd.DataFrame:
    df_list = []
    for c in dataDf.columns:
        tmpDf = pd.DataFrame(dataDf.loc[:, c].values,
                             index=dataDf.index,
                             columns=[value_name])
        tmpDf.insert(loc=0, column=kpi_name, value=[kpi_str] * tmpDf.shape[0])
        tmpDf.insert(loc=0, column=col_name, value=[c] * tmpDf.shape[0])
        tmpDf.index.names = [index_name]
        df_list.append(tmpDf)
    tableauDf = pd.concat(df_list, axis=0)
    if additional_cols is not None:
        for col, value in additional_cols.items():
            tableauDf[col] = value
    return tableauDf
