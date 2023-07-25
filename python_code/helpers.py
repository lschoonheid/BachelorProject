import logging
import sys

from data_exploration.helpers import datetime_str


def setup_custom_logger(name=__name__, dir="", tag="") -> logging.Logger:
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(dir + f"log_{tag}_{datetime_str()}.txt", mode="w")
    print(f"Saving log to `{dir}log_{tag}_{datetime_str()}.txt`")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
