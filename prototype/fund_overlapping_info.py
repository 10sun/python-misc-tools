'''
Author: J , jwsun1987@gmail.com
Date: 2022-11-11 01:47:58
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

import pandas as pd

# get the file path of the funding info file
file_path = (
    r"\\biz.lodh.com\UsersLODH$\GVA\Data\SunJ$\Desktop\Funds_Holdings and Overlap.xlsx"
)

# read occurence data from the excel sheet
fund_holding_data = pd.read_excel(
    file_path,
    sheet_name="Occurence sheet",
)

funds = fund_holding_data[
    [
        "Direct Lines",
        "LO - Climate Transition ",
        "LO - Natural Capital",
        "LO - New Food System",
        "BNP Smart Food",
        "BNP Climate",
        "BNP Acqua",
        "BGF Sustainable Energy",
    ]
]

funds = funds.set_index("Direct Lines")

# compute fund overlapping info
fund_overlapped_info = []

for fund in funds.columns:
    print(fund)
    other_funds = [f for f in funds.columns if f != fund]
    fund_direct_lines = funds[fund].dropna()
    overlapped_weights = {}
    for other_fund in other_funds:
        overlapped_weights.update(
            {
                other_fund: fund_direct_lines[
                    funds[other_fund]
                    .dropna()
                    .index.intersection(fund_direct_lines.index)
                ].sum()
            }
        )
    overlapped_info = (
        pd.DataFrame.from_dict(overlapped_weights, orient="index")
        .rename(columns={0: fund})
        .T
    )
    overlapped_info[fund] = 1 - fund_direct_lines.sum()
    fund_overlapped_info.append(overlapped_info.T)

fund_overlapped_info = pd.concat(fund_overlapped_info, axis=1)
fund_overlapped_info.to_excel("Funds_Holdings_overlapped.xlsx")
print(fund_overlapped_info)
