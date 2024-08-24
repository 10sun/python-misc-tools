'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 03:22:56
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


def fill_in_dict(dict_to_fill: dict, data_dict: dict, overwrite: bool = False):
    try:
        for k, v in data_dict.items():
            if k not in dict_to_fill.keys():
                dict_to_fill.update({k: v})
            else:
                if overwrite:
                    dict_to_fill[k] = v
                else:
                    if isinstance(dict_to_fill[k], pd.DataFrame) or isinstance(
                        dict_to_fill[k], pd.Series
                    ):
                        if isinstance(
                            dict_to_fill[k].index, datetime.datetime
                        ) or isinstance(dict_to_fill[k].index, pd.DatetimeIndex):
                            dict_to_fill[k] = pd.concat([dict_to_fill[k], v], axis=1)
                        else:
                            dict_to_fill[k] = pd.concat([dict_to_fill[k], v], axis=0)
                    elif isinstance(dict_to_fill[k], list):
                        dict_to_fill[k] = dict_to_fill[k] + v
                    else:
                        raise ValueError(
                            "%s type not supported as a value type..." % type(k)
                        )
    except Exception as e:
        logging.error(e)
        return
    return dict_to_fill


def add_params(params_to_use: dict, source_params: dict):
    if not params_to_use:
        return source_params
    for param, param_v in source_params.items():
        if param not in params_to_use:
            params_to_use.update({param: param_v})
        else:
            if isinstance(param_v, dict):
                for k, v in param_v.items():
                    if k not in params_to_use[param]:
                        params_to_use[param].update({k: v})
                    else:
                        params_to_use[param][k] = v
            else:
                params_to_use[param] = param_v
    return params_to_use
