'''
Author: J , jwsun1987@gmail.com
Date: 2023-05-22 22:28:12
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='locp_quantlib',
    version='0.0.1',
    keywords=('Quantitative Framework', 'Quantitative Trading', 'Backtest'),
    description='LOCP Quantlib: Quantitative Research Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='JXW',
    install_requires=['sqlalchemy',
                      'pandas',
                      'numpy',
                      'pytz',
                      'clickhouse-driver',
                      'matplotlib',
                      'plotly',
                      'python-telegram-bot',
                      'dash',
                      'pycountry'],
    author='Jianwen Sun',
    author_email='j.sun@lombardodier.com',
    include_package_data=True,
    packages=find_packages(),
    # package_data={"": [
    # "*.ico",
    # "*.ini",
    # "*.dll",
    # "*.so",
    # "*.pyd",
    # ]},
    platforms='any',
    url='',
    entry_points={
        'console_scripts': [
            'example=examples.demo_strategy:run'
        ]
    },
)
