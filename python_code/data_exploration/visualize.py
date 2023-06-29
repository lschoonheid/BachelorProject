from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, Series
from matplotlib import pyplot as plt
from random import sample as random_sample
from tqdm import tqdm

from _constants import FIG_X, FIG_Y, FIG_DPI, DETECTOR_KEYS, HITS_SAMPLES, TABLE_INDEX, PLOT_FONTSIZE, FIG_EXTENSION
from helpers import get_event_names_str

# TODO:
# [x] Charge distribution
# [x] Number of hits per particle
# [x] px py pz distribution, vs hits
#   [ ]look into 0 hits and 10 direction distribution
# [x] distribution of weight per hit

# [ ] Number of hits per detector
# [ ] Number of hits per layer
# [ ] Number of hits per module
# [ ] Number of hits per particle
# [ ] Number of hits per particle per detector
# [ ] Number of hits per particle per layer
# [ ] Number of hits per particle per module

# [x] Heatmap of hits in xyz
# [ ] Heatmap of hits in rphi
# [ ] Heatmap of hits in rz
# [ ] Heatmap of hits in r
# [ ] Heatmap of hits in phi

# [ ] Plot some high energy tracks
# [ ] Plot some low energy tracks
# [x] Total momentum vs #hits


def get_colors(data: DataFrame, mode: str = "volume_layer"):
    """Map data to colors"""
    match mode:
        # case "volume_layer":
        #     TODO
        #     test = data.apply(lambda x: str(int(x["layer_id"] / 2)) + str(int(x["volume_id"])), axis=1)
        #     # test = data.apply(lambda x: "".join([str(int(x[key])) for key in ["volume_id", "layer_id"]]), axis=1)
        #     _list = list(test)
        #     # unique = combined.unique()
        #     return data.volume_id
        case "volume_id":
            return data.volume_id
        case "layer_id":
            return data.layer_id
        case "module_id":
            return data.module_id
        case "particle_id":
            return data.particle_id
        case _:
            raise ValueError("Mode not recognized")


def scatter(data: DataFrame, *ax_keys: str, color_mode: str = "volume_id"):
    """Plot a scatter plot of the hits in either 2D or 3D"""
    # Create figure
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax_title_str = ", ".join(ax_keys)

    # Choose columns
    columns = [data[key] for key in ax_keys]

    # Choose axes and projection
    if len(ax_keys) == 2:
        ax = fig.add_subplot()
    elif len(ax_keys) == 3:
        ax = fig.add_subplot(projection="3d")
        ax.set_zlabel(ax_keys[2])
    else:
        raise ValueError("Axes not recognized")
    ax.set_xlabel(ax_keys[0])
    ax.set_ylabel(ax_keys[1])

    scatter = ax.scatter(*columns, s=0.1, c=get_colors(data, mode=color_mode))

    ax.set_title(f"Scatter plot of { ax_title_str } coordinates")
    ax.legend(*scatter.legend_elements(), title=color_mode)

    return fig


def versus_scatter(
    event_kv: dict[str, DataFrame], table_0: str, ax_0: str, table_1: str, ax_1: str, join_on: str = "hit_id"
):
    """Plot a scatter of two tables versus each other"""
    # Create figure
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax_title_str = f"{table_0} {ax_0} vs {table_1} {ax_1}"

    # Choose tables
    df_0 = event_kv[table_0]
    df_1 = event_kv[table_1]

    # Merge columns
    merged = df_0.merge(df_1, on=join_on)[[ax_0, ax_1]]

    # Choose axes
    ax = fig.add_subplot()
    ax.set_xlabel(ax_0)
    ax.set_ylabel(ax_1)

    scatter = ax.scatter(merged[ax_0], merged[ax_1], s=0.1)

    ax.set_title(f"Scatter plot of { ax_title_str }")

    return fig


def print_heads(event_kv: dict[str, DataFrame]):
    """Print a few datatable samples"""
    for name, table in event_kv.items():
        print("Table: " + name + ":")
        print(table.head())
        print("\n  \n")


def plot_hits(event_kv: dict[str, DataFrame], unique: bool = False, **kwargs):
    """Plot the hits"""

    # TODO cartesian product of all combinations of ids
    color_modes = DETECTOR_KEYS
    # Plot hits for all data types
    for table in ["truth", "hits"]:
        # Define axis keys
        match table:
            case "hits":
                ax1 = "x"
                ax2 = "y"
                ax3 = "z"
            case "truth":
                ax1 = "tx"
                ax2 = "ty"
                ax3 = "tz"

            case _:
                raise ValueError("Data type not recognized")
        selected_data = event_kv[table]

        # Take a subset of the hits, if size is specified
        if type(HITS_SAMPLES) == int:
            data_subset = selected_data.head(HITS_SAMPLES)
        else:
            data_subset = selected_data

        event_names_str = get_event_names_str(data_subset)

        # Plot all combinations of axes
        for ax_keys in [[ax1, ax2, ax3], [ax1, ax2], [ax1, ax3], [ax2, ax3]]:
            print("Plotting: " + str(ax_keys))
            ax_keys_str = "".join(ax_keys)

            # Plot hits for all color modes
            for color_mode in color_modes:
                pass
                # Plot hits
                fig = scatter(data_subset, *ax_keys, color_mode=color_mode)
                fig.savefig(
                    f"{event_names_str}_{table}_{ax_keys_str}_{color_mode}_scatter_sample{FIG_EXTENSION}",
                    dpi=FIG_DPI,
                )
                plt.close()

            # Skip intensive part of loop if `unique` is False
            if not unique:
                continue

            # Draw seperate plots for each unique value of `color_mode`
            for color_mode in color_modes:
                # Skip too many/uninteresting unique values
                if color_mode in ["particle_id", "module_id"]:
                    continue

                unique_values = selected_data[color_mode].unique()
                for unique_value in tqdm(unique_values, desc=f"{color_mode}"):
                    isolated_detector_data = data_subset.loc[data_subset[color_mode] == unique_value]
                    # Since `color_mode` data is isolated, choose other color mode to distinguish data
                    anti_color_mode = color_modes[1 - color_modes.index(color_mode)]

                    fig = scatter(isolated_detector_data, *ax_keys, color_mode=anti_color_mode)
                    fig.savefig(
                        f"{event_names_str}_{table}_{ax_keys_str}_{color_mode}_{unique_value}_vs_{anti_color_mode}_scatter{FIG_EXTENSION}",
                        dpi=FIG_DPI,
                    )
                    plt.close()


def generate_track_fig(fig_x=FIG_X, fig_y=FIG_Y) -> tuple[Figure, Axes, Axes, Axes, Axes]:
    """Get figure for plotting tracks"""
    # Create figure
    fig = plt.figure(figsize=(2 * fig_x, 2 * fig_y))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axyz = fig.add_subplot(2, 2, 4)

    # Set titles
    ax3d.set_title("3D tracks")
    axxy.set_title("xy tracks")
    axxz.set_title("xz tracks")
    axyz.set_title("yz tracks")

    # Set axis labels
    # 3D
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    # X,Y
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    # X,Z
    axxz.set_xlabel("x")
    axxz.set_ylabel("z")
    # Y,Z
    axyz.set_xlabel("y")
    axyz.set_ylabel("z")
    return fig, ax3d, axxy, axxz, axyz


def add_track_to_fig(
    track_points: DataFrame,
    ax3d: Axes | None = None,
    axxy: Axes | None = None,
    axxz: Axes | None = None,
    axyz: Axes | None = None,
    particle_id: int | str = "particle",
):
    # Sort hits by distance to origin
    track_points.insert(2, "r", track_points.apply(lambda x: (x.tx**2 + x.ty**2) ** 0.5, axis=1), True)
    # Z-axis is timewise
    track_points_sorted: DataFrame = track_points.sort_values(by=["tz", "r"])

    # Plot track
    marker = "."
    markersize = 3
    if ax3d:
        ax3d.plot(
            track_points_sorted.tx,
            track_points_sorted.ty,
            track_points_sorted.tz,
            label=particle_id,
            marker=marker,
            markersize=markersize,
        )

    for ax in filter(lambda x: x is not None, [axxy, axxz, axyz]):
        ax.plot(track_points_sorted.tx, track_points_sorted.ty, label=particle_id, marker=marker, markersize=markersize)  # type: ignore


def plot_event_tracks(
    event_kv: dict[str, DataFrame], event_id: str | None = None, Nt: int | None = 10, random=False, **kwargs
):
    """Plot the tracks"""
    # Note: only works with truth data so far

    # Create figure
    fig, ax3d, axxy, axxz, axyz = generate_track_fig(FIG_X, FIG_Y)

    # Select a subset of the particles
    particle_ids = event_kv["particles"].particle_id.unique()
    if Nt is not None:
        Nt = min(Nt, len(particle_ids))
        particle_ids = random_sample(list(particle_ids), Nt) if random else particle_ids[:Nt]
    else:
        Nt = len(particle_ids)

    # Select true tracks
    truth = event_kv["truth"]
    # Plot tracks
    for particle_id in tqdm(particle_ids, desc="plotting tracks"):
        # Skip ghost particles/hits
        if particle_id == 0:
            continue

        # Select all hits for this particular particle
        track_points: DataFrame = truth.loc[truth.particle_id == particle_id]

        # Add track to plot
        add_track_to_fig(track_points, ax3d, axxy, axxz, axyz, particle_id=particle_id)

    # Add legend if N is small enough

    opacity = 0.2
    if Nt <= 20:
        for ax in [ax3d, axxy, axxz, axyz]:
            ax.legend(fontsize=PLOT_FONTSIZE, framealpha=opacity)

    # Save figure
    event_names_str = get_event_names_str(truth)
    fig.savefig(f"{event_names_str}_tracks_n{Nt}{FIG_EXTENSION}", dpi=FIG_DPI)


def histogram(
    x: Series,
    bins: int = 100,
):
    """Plot a histogram of `x` and `y` on `ax`"""
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax = fig.add_subplot()
    ax.hist(x, bins=bins)
    return fig


def plot_histograms(
    event_kv: dict[str, DataFrame], errors: dict[str, DataFrame] | None = None, prefix: str | None = None, **kwargs
):
    """Plot histograms of the hits"""

    # Single event
    for table, parameter in [["particles", "q"]]:
        fig = histogram(event_kv[table][parameter])
        fig.savefig(f"{prefix}_{table}_{parameter}_histogram{FIG_EXTENSION}", dpi=FIG_DPI)
        plt.close()


def parameter_distribution(
    events: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]],
    table_type: str,
    parameter: str,
    weight_method: str | None = None,
    n_bins: int | None = None,
    _min=None,
    _max=None,
):
    """Plot the distribution of a parameter over all `events`"""
    bins_df = DataFrame()

    # get min and max value to use as bins
    if _min is None or _max is None:
        for event in events:
            min_current = event[TABLE_INDEX[table_type]][parameter].min()
            max_current = event[TABLE_INDEX[table_type]][parameter].max()
            if _min is None or min_current < _min:
                _min = min_current
            if _max is None or max_current > _max:
                _max = max_current
    if n_bins is not None:
        bins_index = pd.cut(Series([_min, _max]), bins=n_bins, retbins=True)[1]

    # Loop over events
    for index, event in tqdm(enumerate(events), total=len(events), desc="Binning events"):
        row = event[TABLE_INDEX[table_type]]

        if weight_method == "value":
            row = row.copy(deep=True)
            row[parameter] = row[parameter] * row["weight"]

        # Put into bins
        if n_bins is None:
            # Use unique values as bins
            groups = row.groupby(parameter)
        else:
            # Split in `n_bins` bins
            row["bin"] = pd.cut(row[parameter], bins=bins_index)  # type: ignore
            groups = row.groupby("bin")

        bin_sizes = groups.size()

        if weight_method in ["group", "importance"]:
            bin_weights = groups["weight"].sum()
            bins = bin_weights if weight_method == "importance" else bin_weights * bin_sizes
        else:
            bins = bin_sizes

        # Rename for readability
        bins.name = parameter + str(index)
        # Join series to dataframe
        bins_df = bins_df.merge(bins, how="outer", left_index=True, right_index=True)

    if n_bins is not None:
        idx: CategoricalIndex | Index = bins_df.index
        bins_df.index = idx.map(lambda x: x.mid)

    print(f"Bins of {parameter}:")
    print(bins_df)

    # Calculate mean and standard deviation
    hist_x = list(bins_df.index)
    hist_y = list(bins_df.mean(axis=1))
    hist_y_err = bins_df.std(axis=1)

    # Plot charge distribution
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax = fig.add_subplot()

    weight_method_tag = f" (weighted: {weight_method})" if weight_method else ""
    ax.set_title(f"Distribution of  { parameter } per {table_type} over { len(events) } events {weight_method_tag}")

    ax.set_xlabel(parameter)
    ax.set_ylabel("count")
    width = 0.4 * (hist_x[1] - hist_x[0])
    ax.bar(hist_x, hist_y, width=width)
    ax.errorbar(hist_x, hist_y, yerr=hist_y_err, fmt="o", markersize=1, ecolor="black", capsize=2)
    return fig


def do_cell_vs_tp(event_kv: dict[str, DataFrame]):
    """Plot cell value versus true momentum norm"""
    ax_name = "tp"
    truth = event_kv["truth"]
    truth.insert(5, ax_name, np.linalg.norm(truth[["tpx", "tpy", "tpz"]].values, axis=1))
    # truth.drop(truth.loc[truth["tp"] < 10**5].index, inplace=True)

    cells = event_kv["cells"]
    summed_value = (cells.groupby("hit_id", as_index=False).sum())[["hit_id", "value"]]
    event_kv["cells"] = summed_value

    fig = versus_scatter(event_kv, "truth", ax_name, "cells", "value")
    fig.savefig(f"{'truth'}_{'tp'}_versus_{'cells'}_{'value'}_scatter{FIG_EXTENSION}", dpi=FIG_DPI)
    plt.close()


def visualize_event(
    event_kv: dict[str, DataFrame],
    do_table: bool = True,
    do_plot_hits: bool = True,
    do_plot_histogram: bool = True,
    do_plot_event_tracks: bool = True,
    do_versus_scatter: bool = True,
    **kwargs,
):
    """Pipe for visualizing a single event"""

    if do_table:
        print_heads(event_kv)

    if not any([do_plot_hits, do_plot_histogram, do_plot_event_tracks]):
        return

    if do_plot_hits:
        plot_hits(event_kv, **kwargs)

    # Deprecated
    # if do_plot_histogram:
    #     plot_histograms(event_kv, **kwargs)

    if do_plot_event_tracks:
        plot_event_tracks(event_kv, **kwargs)

    if do_versus_scatter:
        do_cell_vs_tp(event_kv)
