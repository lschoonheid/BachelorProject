import argparse
from helpers import get_event_names
from trackml.dataset import load_event
from random import choice as random_choice
from pandas import DataFrame
from visualize import *
from _constants import DATA_SAMPLE


def sample_event(random=False, dir=DATA_SAMPLE) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """Load a single event and return some info"""
    event_names = get_event_names(dir)

    # Choose event
    if random:
        event_name = random_choice(list(event_names))
    else:
        event_name = next(event_names.__iter__())
    event_path = dir + event_name

    # Load event
    loaded_event = load_event(event_path)

    # Add event name to each table for easier identification
    for item in loaded_event:
        item.insert(2, "event", event_name)

    return loaded_event


def render_sample(random=False, **kwargs):
    visualize_event(sample_event(random), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("-d", dest="dir", type=str, default=DATA_SAMPLE, help="Directory to load events from")
    parser.add_argument("-N", dest="N", type=int, default=None, help="Execute choice with parameter N")
    parser.add_argument("--random", dest="random", action="store_true", help="Execute choice with random picking")
    parser.add_argument("--table", dest="do_table", action="store_true", help="Run with printing data rows")
    parser.add_argument("--hits", dest="do_plot_hits", action="store_true", help="Plot the hits")
    # parser.add_argument("--hist", dest="do_plot_histogram", action="store_true", help="Plot histograms")
    parser.add_argument("--tracks", dest="do_plot_tracks", action="store_true", help="Plot tracks")

    args = parser.parse_args()
    kwargs = vars(args)

    render_sample(**kwargs)
