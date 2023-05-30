import argparse
from matplotlib import pyplot as plt

from pandas import DataFrame, Series

from _constants import DATA_SAMPLE, TABLE_INDEX, FIG_X, FIG_Y, FIG_DPI
from helpers import get_event_names
from tqdm import tqdm
from helpers import load_event_cached

# from visualize import histogram


def run(dir=DATA_SAMPLE, N=None):
    event_names = get_event_names(dir)

    # events = [load_event(dir + name) for name in tqdm(event_names, desc="Loading events")]

    event_names = list(event_names)[:N] if N else list(event_names)

    n_events: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]] = [
        load_event_cached(dir + name) for name in tqdm(event_names, desc="Loading events")
    ]

    bins_df = DataFrame()
    unique_values = set()

    #  Charge distribution
    for index, event in enumerate(n_events):
        particles = event[TABLE_INDEX["particles"]]
        # hist = particles["q"].hist()

        unique_values = unique_values.union(set(particles["q"].unique()))
        bins = particles["q"].value_counts().sort_index()
        bins.name = "q" + str(index)
        bins_df = bins_df.merge(bins, how="outer", left_index=True, right_index=True)

    print(bins_df)

    hist_x = list(bins_df.index)
    hist_y = list(bins_df.mean(axis=1))
    hist_y_err = bins_df.std(axis=1)

    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax = fig.add_subplot()
    ax.bar(hist_x, hist_y, width=0.1)
    ax.errorbar(hist_x, hist_y, yerr=hist_y_err, fmt="o", ecolor="black", capsize=2)
    fig.savefig(f"charge_distribution.png", dpi=FIG_DPI)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("-d", dest="dir", type=str, default=DATA_SAMPLE, help="Directory to load events from")
    # parser.add_argument("--random", dest="random", action="store_true", help="Load a random event")
    # parser.add_argument("--table", dest="do_table", action="store_true", help="Run without printing data rows")
    # parser.add_argument("--hits", dest="do_plot_hits", action="store_true", help="Plot the hits")
    # parser.add_argument("--hist", dest="do_plot_histogram", action="store_true", help="Plot histograms")

    args = parser.parse_args()
    kwargs = vars(args)

    run(**kwargs)
