import os
import pathlib
import pickle
import time
import hashlib
from typing import Callable
from functools import wraps
from pandas import DataFrame
from trackml.dataset import load_event
from _constants import CACHE_LOC


def get_event_names(dir: str) -> list[str]:
    """Get unique list of event names in directory `dir`"""
    file_list = os.listdir(dir)
    event_names = set()

    # Go over all files in directory
    for filename in file_list:
        # Check if item is event file
        isCSV = filename[-4:] in [".csv", ".CSV"]
        if not isCSV:
            continue

        event_name = filename.split("-")[0]
        event_names.add(event_name)

    event_names = list(event_names)
    event_names.sort()

    return event_names


def get_event_names_str(table: DataFrame):
    event_names = list(table["event"].unique())
    event_names_str = ", ".join(event_names)
    return event_names_str


def prepare_path(path: str):
    """If a directory for `path` doesn't exist, make it."""

    # Filter directory from path
    directory = "/".join(path.split("/")[:-1])
    if directory == "":
        return

    isExist = os.path.exists(directory)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(directory)


def dump_pickle(data, output: str):
    """Dump pickle file."""
    prepare_path(output)
    with open(output, "wb") as handle:
        pickle.dump(data, handle)
    return data


def dump_result(data, directory: str):
    time_string = time.strftime("%Y%m%d-%H%M%S")
    path = directory + time_string + ".pyc"
    dump_pickle(data, path)
    return path


def load_pickle(location: str):
    """Load pickle file."""
    with open(location, "rb") as handle:
        data = pickle.load(handle)
    return data


def hashargs(*args, **kwds):
    """Takes `args` and `kwds` as arguments and hashes its information to a string."""
    args_identifier = hashlib.md5(str((args, kwds)).encode()).hexdigest()
    return args_identifier


def pickle_cache(func: Callable, verbose: bool = False):
    """Decorator function for caching function output to PYC files."""

    @wraps(func)
    def wrapper(*args, **kwds):
        args_identifier = hashargs(*args, **kwds)
        output = CACHE_LOC + args_identifier + ".pyc"

        try:
            data = load_pickle(output)
            if verbose:
                print("Found cached data. Loading from cache instead.")
        except FileNotFoundError:
            data = dump_pickle(func(*args, **kwds), output)
        return data

    return wrapper


@pickle_cache
def load_event_cached(
    prefix: str | pathlib.Path, parts: list[str] = ["hits", "cells", "particles", "truth"]
) -> tuple[DataFrame, ...]:
    """Load an event and cache it."""
    return load_event(prefix, parts)
