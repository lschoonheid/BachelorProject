# Interface for running visualizations on the data averages

import argparse
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm

# Local imports
from data_exploration.helpers import extend_features
from .constants import DATA_SAMPLE, FIG_DPI, FIG_EXTENSION, TABLE_INDEX
from .helpers import get_event_names, load_event_cached, select_r_0
from .visualize import parameter_distribution, histogram, versus_scatter
from .event import Event


def run(
    dir=DATA_SAMPLE,
    N=None,
    do_n_particles=False,
    do_cell_value=False,
    do_charge=False,
    do_hits_n=False,
    do_hits_ax=False,
    do_momentum_norm=False,
    do_momentum_ax=False,
    do_weight=False,
    weight_method=None,
):
    # Load events
    event_names = get_event_names(dir)
    event_names = list(event_names)[:N] if N else list(event_names)
    n_events: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]] = [
        load_event_cached(dir + name) for name in tqdm(event_names, desc="Loading events")
    ]

    weight_tag = f"{weight_method}" if weight_method else ""

    do_p_vs_r0 = True
    if do_p_vs_r0:
        for event in n_events:
            event = Event(loaded_event=event)
            event.__dict__["r_0"] = extend_features(select_r_0(event.truth))
            x_str = "r_0"
            # y_str = "tz"
            y_str = "p_0"
            fig = versus_scatter(event.__dict__, "r_0", x_str, "r_0", y_str, ylim=(0, 25), xlim=(500, 1000))
            fig.savefig(f"{'truth'}_{x_str}_versus_{'truth'}_{y_str}_scatter_event{FIG_EXTENSION}", dpi=FIG_DPI)
            plt.close()

    # Number of particles per event
    if do_n_particles:
        n_particles = Series([len(event[TABLE_INDEX["particles"]]) for event in n_events])
        fig = histogram(n_particles, bins=50)
        fig.axes[0].set_xlabel("Number of particles")
        fig.axes[0].set_ylabel("Number of events")
        fig.axes[0].set_title("Number of particles per event")

        fig.savefig(f"n_particles_histogram_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Cell value distribution
    if do_cell_value:
        # TODO: group cells by hit_id
        fig = parameter_distribution(n_events, "cells", "value", weight_method=weight_method, n_bins=100)
        fig.savefig(f"cell_value_distribution_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Charge distribution
    if do_charge:
        fig = parameter_distribution(n_events, "particles", "q", weight_method=weight_method)
        fig.savefig(f"charge_distribution_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Hits per particle
    if do_hits_n:
        fig = parameter_distribution(n_events, "particles", "nhits", weight_method=weight_method)
        fig.savefig(f"hits_per_particle_distribution_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Weight
    if do_weight:
        fig = parameter_distribution(n_events, "truth", "weight", n_bins=100, _min=0, _max=0.00005)
        fig.savefig(f"weight_per_hit_distribution_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Hits over axes distribution
    if do_hits_ax:
        axes = ["tx", "ty", "tz"]
        for ax in axes:
            fig = parameter_distribution(n_events, "truth", ax, weight_method=weight_method, n_bins=100)
            fig.savefig(f"hits_{ax}_distribution_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
            plt.close()

    if do_momentum_norm:
        ax_name = "tp"
        # Insert momentum norm
        for event in n_events:
            truth = event[TABLE_INDEX["truth"]]
            truth.insert(5, ax_name, np.linalg.norm(truth[["tpx", "tpy", "tpz"]].values, axis=1))

        # Full distribution
        fig = parameter_distribution(n_events, "truth", ax_name, weight_method=weight_method, n_bins=100)
        fig.savefig(f"hits_{ax_name}_distribution_full_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

        # Zoomed in distribution
        fig = parameter_distribution(
            n_events, "truth", ax_name, weight_method=weight_method, n_bins=100, _min=0, _max=20
        )
        fig.savefig(f"hits_{ax_name}_distribution_zoom_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

        # Zoomed twice in distribution
        fig = parameter_distribution(
            n_events, "truth", ax_name, weight_method=weight_method, n_bins=100, _min=0, _max=5
        )
        fig.savefig(f"hits_{ax_name}_distribution_zoom_x2_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

        # Zoomed shifted in distribution
        fig = parameter_distribution(
            n_events, "truth", ax_name, weight_method=weight_method, n_bins=100, _min=10**6 - 1, _max=10**6 + 1
        )
        fig.savefig(f"hits_{ax_name}_distribution_zoom_shifted_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()

    # Momentum over axes distribution
    if do_momentum_ax:
        p_axes = ["tpx", "tpy", "tpz"]
        for ax in p_axes:
            # Full distribution
            fig = parameter_distribution(n_events, "truth", ax, weight_method=weight_method, n_bins=100)
            fig.savefig(f"hits_{ax}_distribution_full_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
            plt.close()

            # Zoomed in distribution
            fig = parameter_distribution(
                n_events, "truth", ax, weight_method=weight_method, n_bins=100, _min=-10, _max=10
            )
            fig.savefig(f"hits_{ax}_distribution_zoom_{weight_tag}{FIG_EXTENSION}", dpi=FIG_DPI)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")
    parser.add_argument("-d", dest="dir", type=str, default=DATA_SAMPLE, help="Directory to load events from")
    parser.add_argument(
        "-w",
        "--weight_method",
        dest="weight_method",
        type=str,
        default=None,
        choices=[None, "group", "importance", "value"],
        help="Plot n_particles distribution",
    )
    parser.add_argument("--n_p", dest="do_n_particles", action="store_true", help="Plot n_particles distribution")
    parser.add_argument("--cell", dest="do_cell_value", action="store_true", help="Plot cell value distribution")
    parser.add_argument("--weight", dest="do_weight", action="store_true", help="Plot weight distribution")
    parser.add_argument("--hits_n", dest="do_hits_n", action="store_true", help="Plot nhits distribution")
    parser.add_argument("--hits_ax", dest="do_hits_ax", action="store_true", help="Plot hits over ax distribution")
    parser.add_argument("--p_ax", dest="do_momentum_ax", action="store_true", help="Plot momentum over ax distribution")
    parser.add_argument("--p_norm", dest="do_momentum_norm", action="store_true", help="Plot hits over ax distribution")
    parser.add_argument("-q", "--charge", dest="do_charge", action="store_true", help="Plot charge distribution")
    parser.add_argument("--all", dest="do_all", action="store_true", help="Run all visualizations")

    args = parser.parse_args()
    kwargs = vars(args)

    if kwargs.pop("do_all"):
        kwargs["do_n_particles"] = True
        kwargs["do_cell_value"] = True
        kwargs["do_weight"] = True
        kwargs["do_hits_n"] = True
        kwargs["do_hits_ax"] = True
        kwargs["do_momentum_ax"] = True
        kwargs["do_momentum_norm"] = True
        kwargs["do_charge"] = True

    run(**kwargs)
