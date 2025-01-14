import os
from os.path import isfile, join
import datetime
import pathlib
import pickle
import time
import hashlib
from typing import Callable, Any
from functools import wraps
from pandas import DataFrame, read_csv  # type: ignore
import numpy as np
from numpy import load as load_numpy

from trackml.dataset import load_event  # type: ignore
from .constants import CACHE_LOC
from .dirs import LOG_DIR  # type: ignore

import logging
import sys


def setup_custom_logger(name=None, dir=LOG_DIR, tag="", level=logging.DEBUG) -> logging.Logger:
    """Setup a custom logger with the given name."""
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    prepare_path(dir)

    tag_str = f"_{tag}" if tag else ""

    _datetime_str = datetime_str()

    handler = logging.FileHandler(dir + f"log{tag_str}_{_datetime_str}.txt", mode="w")
    print(f"Saving log to `{dir}log{tag_str}_{_datetime_str}.txt`")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


def get_logger(name=None, **kwargs) -> logging.Logger:
    """Gets a logger with the given name, or creates one if it does not exist."""
    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        tag = kwargs.pop("tag", None)
        if tag is None:
            import __main__

            tag = __main__.__file__.split("/")[-1]
        return setup_custom_logger(name, tag=tag, **kwargs)
    return logger


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
    event_names = list(table["event"].unique())  # type: ignore
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

    funcname = func.__qualname__

    @wraps(func)
    def wrapper(*args, **kwds):
        args_identifier = funcname + "_" + hashargs(*args, **kwds)
        output = CACHE_LOC + args_identifier + ".pyc"

        try:
            data = load_pickle(output)
            get_logger().debug(f"Loaded cached output for `{funcname}` at `{output}`")
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


def save(return_object, name: str | None = None, tag: str | None = None, dir: str = "", save=True, extension="pkl"):
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
    else:
        tag = ""
    folder_message = f" in folder `{dir}`" if dir else " in current folder."

    time = datetime_str()
    if _is_pickle(extension):
        dump_pickle(return_object, f"{dir}{name}{tag}_{time}.{extension}")
    elif _is_csv(extension):
        return_object.to_csv(f"{dir}{name}{tag}_{time}.{extension}")
    else:
        raise ValueError(f"File extension {extension} not supported.")
    get_logger().info(f"Saved `{name}` ({object_type}) as `{name}{tag}_{time}.{extension}`{folder_message}")

    return return_object


def find_filenames(name: str, dir: str = CACHE_LOC, extension="pkl") -> list[str]:
    """Return all filenames that match with `name` and `extension` in `dir`."""
    onlyfiles = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    matched_files = []
    for file in onlyfiles:
        file_extension = file.split(".")[-1]
        if name in file and file_extension == extension:
            matched_files.append(file)
    return matched_files


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
        onlyfiles = find_filenames(name, dir=dir, extension=extension)
        # See if there is a prediction matrix for this event
        for file in onlyfiles:
            # Check if file name and extension match
            file_extension = file.split(".")[-1]
            if name in file and file_extension == extension:
                get_logger().debug(f"Found `{file}` in `{dir}`")

                # Load file
                if _is_pickle(file_extension):
                    return load_pickle(dir + file)
                elif _is_csv(file_extension):
                    return read_csv(dir + file)
                elif _is_numpy(file_extension):
                    return load_numpy(dir + file, allow_pickle=True)
        raise FileNotFoundError
    except FileNotFoundError:
        get_logger().debug(f"Could not find {extension} file `{name}` in {dir}")
        if fallback_func is not None:
            return fallback_func()
        return None


def find_files(name: str, dir: str = CACHE_LOC, extension="pkl"):
    """Load all files in `dir` that are similar to `name`."""

    # Check supported extension
    is_pickle = _is_pickle(extension)
    is_csv = _is_csv(extension)
    is_numpy = _is_numpy(extension)
    is_supported = any([is_pickle, is_csv, is_numpy])
    if not is_supported:
        raise ValueError(f"File extension {extension} not supported.")

    files = []
    # Look for files in ouput directory
    onlyfiles = find_filenames(name, dir=dir, extension=extension)
    # See if there is a prediction matrix for this event
    for file in onlyfiles:
        # Check if file name and extension match
        file_extension = file.split(".")[-1]
        if name in file and file_extension == extension:
            print(f"Found `{file}` in {dir}")

            # Load file
            if _is_pickle(file_extension):
                files.append(load_pickle(dir + file))
            elif _is_csv(file_extension):
                files.append(read_csv(dir + file))
            elif _is_numpy(file_extension):
                files.append(load_numpy(dir + file, allow_pickle=True))
    return files


def cached(
    name: str,
    dir: str = CACHE_LOC,
    extension="pkl",
    fallback_func: Callable | None = None,
    force_fallback=False,
    do_save=True,
):
    """Similar to `@pickle_cache` but able to configure name and directory for each individual call and looser file-function call matching which disregards function inputs."""
    return find_file(
        name=name,
        dir=dir,
        extension=extension,
        fallback_func=lambda: save(fallback_func(), name=name, dir=dir, save=do_save),  # type: ignore
        force_fallback=force_fallback,
    )


def retry(
    func: Callable,
    n_retries: int = 3,
    sleep_time: int = 1,
    exception: Exception | type[Exception] | None = None,
    fallback_func: Callable | None = None,
):
    """Retry a function `n_retries` times with `sleep_time` seconds in between."""
    if exception is None:
        exception = Exception

    for i in range(n_retries):
        try:
            return func()
        except exception as exc:  # type: ignore
            get_logger().debug(
                f"Failed to run function `{func.__qualname__}` ({exc}). Retrying in {sleep_time} seconds ({i+1}/{n_retries})..."
            )

            if fallback_func is not None:
                get_logger().debug(f"Running fallback function `{fallback_func.__qualname__}`")
                fallback_func()

            time.sleep(sleep_time)
    raise exception


def add_r(combined: DataFrame, mode="truth"):
    if mode == "truth":
        labels = ["tx", "ty", "tz"]
    else:
        labels = ["x", "y", "z"]
    r = np.sqrt(np.sum(combined[labels].values ** 2, axis=1))
    copy = combined.copy()
    copy.insert(4, "r", r)
    return copy


def select_r_0(combined: DataFrame):
    r_sorted = add_r(combined).sort_values("r", ascending=True)
    r_mask = r_sorted[r_sorted.duplicated(subset="particle_id", keep="first")]
    r_0 = r_sorted[~r_sorted.index.isin(r_mask.index)]  # .rename(columns={"r": "r_0"})
    return r_0


def select_r_less(hits: DataFrame, thr: float = 300):
    """Return indices of hits with r < `thr`."""
    inner_idx = np.where(add_r(hits, mode="hits")["r"] < thr)[0]
    return inner_idx


def extend_features(r_0: DataFrame):
    assert "r" in r_0, "r not in DataFrame"

    r_0.rename(
        columns={
            "hit_id": "hit_id_0",
            "r": "r_0",
            "tx": "x_0",
            "ty": "y_0",
            "tz": "z_0",
            "tpx": "px_0",
            "tpy": "py_0",
            "tpz": "pz_0",
            "weight": "weight_0",
        },
        inplace=True,
    )
    r_0["p_0"] = np.sqrt(r_0["px_0"] ** 2 + r_0["py_0"] ** 2 + r_0["pz_0"] ** 2)
    r_0["p_t_0"] = np.sqrt(r_0["px_0"] ** 2 + r_0["py_0"] ** 2)
    r_0["log_10_p_t_0"] = np.log10(r_0["p_t_0"])
    r_0["phi_0"] = np.arctan2(r_0["y_0"], r_0["x_0"])
    r_0["theta_0"] = np.arccos(r_0["z_0"] / r_0["r_0"])
    r_0["pseudo_rapidity_0"] = -np.log(np.tan(r_0["theta_0"] / 2))
    return r_0


def binning_analysis(samples):
    """Perform a binning analysis over samples and return
    errors: an array of the error estimate at each binning level,
    tau: the estimated integrated autocorrelation time,
    converged: a flag indicating if the binning has converged, and
    bins: the last bin values"""
    minbins = 2**6  # minimum number of bins
    maxlevel = int(np.log2(len(samples) / minbins))  # number of binning steps
    maxsamples = minbins * 2 ** (maxlevel)  # the maximal number of samples considered
    bins = np.array(samples[-maxsamples:])
    errors = np.zeros(maxlevel + 1)
    for k in range(maxlevel):
        errors[k] = np.std(bins) / np.sqrt(len(bins) - 1.0)
        bins = np.array((bins[::2] + bins[1::2]) / 2.0)

    errors[maxlevel] = np.std(bins) / np.sqrt(len(bins) - 1.0)
    tau = 0.5 * ((errors[-1] / errors[0]) ** 2 - 1.0)
    relchange = (errors[1:] - errors[:-1]) / errors[1:]
    meanlastchanges = np.mean(relchange[-3:])  # get the average over last changes
    converged = 1
    if meanlastchanges > 0.05:
        print("warning: binning maybe not converged, meanlastchanges:", meanlastchanges)
        converged = 0
    return (errors, tau, converged, bins)
