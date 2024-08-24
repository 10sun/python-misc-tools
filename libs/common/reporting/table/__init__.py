'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:10:24
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


import pandas as pd
from typing import Union, Dict, List, Optional
from IPython.display import display, HTML


def print_table(
    df: Union[pd.DataFrame, pd.Series],
    name: Optional[str],
    float_format=None,
    formatters=None,
    header_rows: Dict = None,
):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    if name is not None:
        df.columns.name = name

    html = df.to_html(float_format=float_format, formatters=formatters)

    if header_rows is not None:
        n_cols = html.split("<thead>")[1].split("</thead>")[0].count("<th>")

        rows = ""
        for name, value in header_rows.items():
            rows += (
                '\n <tr style="text-align: right;"><th>%s</th>'
                + "<td colspan=%d>%s</td></tr>"
            ) % (name, n_cols, value)

        html = html.replace("<thead>", "<thead>" + rows)
    display(HTML(html))

    return html
