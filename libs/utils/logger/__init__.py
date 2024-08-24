'''
Author: J , jwsun1987@gmail.com
Date: 2023-11-21 20:07:04
LastEditors: J , jwsun1987@gmail.com
Description: 
Copyright: Copyright (c) 2024 by jwsun1987@gmail.com. All Rights Reserved.
'''


from asyncio.log import logger
import logging
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Union
from xmlrpc.client import Boolean


def config_logger(
    name: str = "LO",
    level=logging.INFO,
    console_logging: Boolean = True,
    log_dir: Union[str, Path] = None,
    log_file_base_name: str = "",
):
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s - %(levelname)s] %(message)s",  # .%(msecs)d
        datefmt="%Y-%m-%d %H:%M%S",
    )
    lo_logger = logging.getLogger(name)
    lo_logger.setLevel(level)

    if console_logging:
        add_log_stream_handler(logger, level, formatter)

    if log_dir:
        add_log_file_handler(lo_logger, level, formatter, log_dir, log_file_base_name)
    return lo_logger


def add_log_stream_handler(logger: logging.Logger, level, formatter):
    if not any(isinstance(handle, logging.StreamHandler) for handle in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def add_log_file_handler(
    logger: logging.Logger,
    level,
    formatter,
    log_dir: Union[str, Path],
    log_file_base_name: str = "",
):
    ## TODOS: get the absolute path of a given dir
    abs_log_path = Path("") / log_dir
    abs_log_path.mkdir(parents=True, exist_ok=True)
    log_filename = log_file_base_name + "_" + str(dt.now()) + ".txt"

    if not any(isinstance(handle, logging.FileHandler) for handle in logger.handlers):
        file_logger = logging.FileHandler(abs_log_path / log_filename)
        file_logger.setFormatter(formatter)
        file_logger.setLevel(level)
        logger.addHandler(file_logger)


def rethrow(exception, additional_message):
    """add message to exception warnings

    Args:
        exception (_type_): _description_
        additional_message (_type_): _description_

    Raises:
        e: _description_
    """
    e = exception
    message = additional_message
    if not e.args:
        e.args = (message,)
    else:
        e.args = (e.args[0] + message,) + e.args[1:]
    raise e
