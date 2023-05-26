import argparse
from os import listdir
from trackml.dataset import load_event
from random import choice as random_choice
from pandas import DataFrame, Series
from visualize import *


DATA_ROOT = "/data/atlas/users/lschoonh/BachelorProject/data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"


def get_event_names(dir: str) -> set[str]:
    """Get unique list of event names in directory `dir`"""
    list = listdir(dir)
    event_names = set()

    # Go over all files in directory
    for filename in list:
        # Check if item is event file
        isCSV = filename[-4:] in [".csv", ".CSV"]
        if not isCSV:
            continue

        event_name = filename.split("-")[0]
        event_names.add(event_name)

    return event_names


def sample_event(random=False) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load a single event and return some info"""
    event_names = get_event_names(DATA_SAMPLE)

    if random:
        event_path = DATA_SAMPLE + random_choice(list(event_names))
    else:
        event_path = DATA_SAMPLE + next(event_names.__iter__())

    loaded_event = load_event(event_path)

    return loaded_event


def render_samples(random=False, **kwargs):
    visualize(sample_event(random), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("--random", dest="random", action="store_true", help="Load a random event")
    parser.add_argument("--dont-plot-hits", dest="plot_hits", action="store_false", help="Plot the hits")

    args = parser.parse_args()
    kwargs = vars(args)

    render_samples(**kwargs)
