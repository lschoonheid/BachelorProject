import os
from os.path import isfile, join
import datetime
import pathlib
import pickle
import time
import hashlib
from typing import Callable, Any
from functools import wraps
from pandas import DataFrame, read_csv
from trackml.dataset import load_event
from .constants import CACHE_LOC
from numpy import load as load_numpy


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


def is_filetype(filename: str, extensions: list[str]) -> bool:
    file_extension = filename.split(".")[-1]
    return extensions.count(file_extension) > 0


def _is_pickle(filename: str) -> bool:
    return is_filetype(filename, ["pyc", "pkl"])


def _is_csv(filename: str) -> bool:
    return is_filetype(filename, ["csv"])


def _is_numpy(filename: str) -> bool:
    return is_filetype(filename, ["npy"])


def datetime_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def save(return_object, name: str | None = None, tag: str | None = None, prefix: str = "", save=True, extension="pkl"):
    """Wrapping function for dumping `return_object` to a pickle file with name `name` and optional tag `tag`."""
    object_type = type(return_object).__name__
    if name is None:
        try:
            name = return_object.__name__
        except AttributeError:
            name = object_type

    # Prepare file string
    if not save:
        return return_object

    if tag is not None:
        tag = f"_{tag}"
    folder_message = f" in folder `{prefix}`" if prefix else " in current folder."

    time = datetime_str()
    if _is_pickle(extension):
        dump_pickle(return_object, f"{prefix}{name}{tag}_{time}.{extension}")
    elif _is_csv(extension):
        return_object.to_csv(f"{prefix}{name}{tag}_{time}.{extension}")
    else:
        raise ValueError(f"File extension {extension} not supported.")
    print(f"Saved `{name} ({object_type})` as `{name}{tag}_{time}.{extension}`{folder_message}")

    return return_object


def find_file(
    name: str, dir: str = CACHE_LOC, extension="pkl", fallback_func: Callable | None = None, force_fallback=False
):
    """Load a file in `dir` that is similar to `name`."""
    if force_fallback and fallback_func is not None:
        return fallback_func()

    # Check supported extension
    is_pickle = _is_pickle(extension)
    is_csv = _is_csv(extension)
    is_numpy = _is_numpy(extension)
    is_supported = any([is_pickle, is_csv, is_numpy])
    if not is_supported:
        raise ValueError(f"File extension {extension} not supported.")

    try:
        # Look for files in ouput directory
        onlyfiles = [f for f in os.listdir(dir) if isfile(join(dir, f))]
        # See if there is a prediction matrix for this event
        for file in onlyfiles:
            # Check if file name and extension match
            file_extension = file.split(".")[-1]
            if name in file and file_extension == extension:
                print(f"Found `{file}` in {dir}")

                # Load file
                if _is_pickle(file_extension):
                    return load_pickle(dir + file)
                elif _is_csv(file_extension):
                    return read_csv(dir + file)
                elif _is_numpy(file_extension):
                    return load_numpy(dir + file, allow_pickle=True)
        raise FileNotFoundError
    except FileNotFoundError:
        print(f"Could not find {extension} file `{name}` in {dir}")
        if fallback_func is not None:
            return fallback_func()
        return None
