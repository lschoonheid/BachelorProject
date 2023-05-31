import argparse
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm

# Local imports
from _constants import DATA_SAMPLE, FIG_DPI, FIG_EXTENSION
from helpers import get_event_names, load_event_cached
from visualize import parameter_distribution


def run(dir=DATA_SAMPLE, N=None):
    # Load events
    event_names = get_event_names(dir)
    event_names = list(event_names)[:N] if N else list(event_names)
    n_events: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]] = [
        load_event_cached(dir + name) for name in tqdm(event_names, desc="Loading events")
    ]

    # Charge distribution
    fig = parameter_distribution(n_events, "particles", "q")
    fig.savefig(f"charge_distribution{FIG_EXTENSION}", dpi=FIG_DPI)
    plt.close()

    # Hits per particle
    fig = parameter_distribution(n_events, "particles", "nhits")
    fig.savefig(f"hits_per_particle_distribution{FIG_EXTENSION}", dpi=FIG_DPI)
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
