import argparse
from os import listdir
from trackml.dataset import load_event
from pandas import DataFrame
from random import choice as random_choice
from matplotlib import pyplot as plt


DATA_ROOT = "/data/atlas/users/lschoonh/BachelorProject/data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"

HITS_SAMPLES = None


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


def scatter_3D(hits: DataFrame):
    """Plot a 3D scatter plot of the hits"""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(hits.x, hits.y, hits.z, s=0.1)
    return fig


def scatter_2D(hits: DataFrame):
    """Plot a 2D scatter plot of the hits"""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(hits.x, hits.y, s=0.1)
    return fig


def visualize(random: bool = False, plot_hits: bool = True, **kwargs):
    # Load a single event
    loaded_event = sample_event()
    hits, cells, particles, truth = loaded_event

    # Take a subset of the hits, if size is specified
    if type(HITS_SAMPLES) == int:
        hits_subset = hits.head(HITS_SAMPLES)
    else:
        hits_subset = hits

    # Print a few datatable samples
    for table, name in zip([hits, cells, particles, truth], ["hits", "cells", "particles", "truth"]):
        print("Table: " + name + ": \n")
        print(table.head())
        print("\n \n")

    if not plot_hits:
        return

    # Plot hits 3D
    fig_3D = scatter_3D(hits_subset)
    fig_3D.savefig("event_3D_scatter_sample.png")

    # Plot hits 2D
    fig_2D = scatter_2D(hits_subset)
    fig_2D.savefig("event_2D_scatter_sample.png")

    # Plot hits 2D with color
    # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("--random", dest="random", action="store_true", help="Load a random event")
    parser.add_argument("--dont-plot-hits", dest="plot_hits", action="store_false", help="Plot the hits")

    args = parser.parse_args()
    kwargs = vars(args)

    visualize(**kwargs)
