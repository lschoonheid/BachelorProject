import argparse
from os import listdir
from trackml.dataset import load_event
from random import choice as random_choice
from pandas import DataFrame, Series
from visualize import *
from constants import DATA_SAMPLE


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


def sample_event(random=False, dir=DATA_SAMPLE) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load a single event and return some info"""
    event_names = get_event_names(DATA_SAMPLE)

    if random:
        event_path = dir + random_choice(list(event_names))
    else:
        event_path = dir + next(event_names.__iter__())

    loaded_event = load_event(event_path)

    return loaded_event


def render_sample(random=False, **kwargs):
    visualize_event(sample_event(random), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("-d", dest="dir", type=str, default=DATA_SAMPLE, help="Directory to load events from")
    parser.add_argument("--random", dest="random", action="store_true", help="Load a random event")
    parser.add_argument("--table", dest="do_table", action="store_true", help="Run without printing data rows")
    parser.add_argument("--hits", dest="do_plot_hits", action="store_true", help="Plot the hits")
    parser.add_argument("--hist", dest="do_plot_histogram", action="store_true", help="Plot histograms")

    args = parser.parse_args()
    kwargs = vars(args)

    render_sample(**kwargs)
