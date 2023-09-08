from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas import CategoricalIndex, DataFrame, Index, Series
from matplotlib import pyplot as plt
from random import sample as random_sample
from tqdm import tqdm

from .constants import FIG_X, FIG_Y, FIG_DPI, DETECTOR_KEYS, HITS_SAMPLES, TABLE_INDEX, PLOT_FONTSIZE, FIG_EXTENSION
from .helpers import get_event_names_str, prepare_path

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


def get_crop(x, y, crop: float, square=False) -> tuple[tuple[float, float], tuple[float, float]]:
    x_range = min(x), max(x)
    x_middle = (x_range[0] + x_range[1]) / 2
    x_span: float = x_range[1] - x_range[0]

    y_range = min(y), max(y)
    y_middle = (y_range[0] + y_range[1]) / 2
    y_span: float = y_range[1] - y_range[0]

    max_span = max(x_span, y_span)
    if square:
        x_span = max_span
        y_span = max_span

    # lim = middle - crop * span / 2, middle + crop * span /2
    xlim: tuple[float, float] = x_middle - crop * x_span / 2, x_middle + crop * x_span / 2
    ylim: tuple[float, float] = y_middle - crop * y_span / 2, y_middle + crop * y_span / 2

    return xlim, ylim


def make_compact(fig: Figure, new_size: tuple[float, float] = (5, 3)):
    """Make a figure compact"""
    fig.set_size_inches(*new_size)
    fig.gca().set_title("")
    fig.tight_layout()
    return fig


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
        ax.set_zlabel(ax_keys[2] + " [mm]")
    else:
        raise ValueError("Axes not recognized")
    ax.set_xlabel(ax_keys[0] + " [mm]")
    ax.set_ylabel(ax_keys[1] + " [mm]")

    scatter = ax.scatter(*columns, s=0.1, c=get_colors(data, mode=color_mode))

    # ax.set_title(f"Hits { ax_title_str } coordinates")
    ax.legend(*scatter.legend_elements(), title=color_mode)

    fig.tight_layout()
    return fig


def versus_scatter(
    event: dict[str, DataFrame],
    table_0: str,
    ax_0: str,
    table_1: str,
    ax_1: str,
    join_on: str = "hit_id",
    xlim=None,
    ylim=None,
    **kwargs,
):
    """Plot a scatter of two tables versus each other"""
    # Create figure
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax_title_str = f"{table_0} {ax_0} vs {table_1} {ax_1}"

    # Merge columns
    if table_0 != table_1:
        # Choose tables
        df_0 = event[table_0]
        df_1 = event[table_1]
        merged = df_0.merge(df_1, on=join_on)[[ax_0, ax_1]]
    else:
        merged = event[table_0][[ax_0, ax_1]]

    # Choose axes
    ax = fig.add_subplot()
    ax.set_xlabel(ax_0)
    ax.set_ylabel(ax_1)

    scatter = ax.scatter(merged[ax_0], merged[ax_1], s=0.1, **kwargs)

    ax.set_title(f"Scatter plot of { ax_title_str }")

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    fig.tight_layout()
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
    fig.tight_layout()
    return fig, ax3d, axxy, axxz, axyz


def add_track_to_fig(
    track_points: DataFrame,
    ax3d: Axes | None = None,
    axxy: Axes | None = None,
    axxz: Axes | None = None,
    axyz: Axes | None = None,
    type: str = "truth",
    particle_id: int | str = "particle",
):
    assert type in ["truth", "hits"]

    # Choose axis labels
    if type == "truth":
        x_str, y_str, z_str = "tx", "ty", "tz"
    else:
        x_str, y_str, z_str = (
            "x",
            "y",
            "z",
        )

    # Sort hits by distance to origin
    track_points_copy = track_points.copy(deep=True)
    track_points_copy.insert(
        2, "r", track_points_copy.apply(lambda x: (x[x_str] ** 2 + x[y_str] ** 2 + x[z_str] ** 2) ** 0.5, axis=1), True
    )
    # Z-axis is timelike
    track_points_sorted: DataFrame = track_points_copy.sort_values(by=[z_str, "r"])

    # Plot track
    marker = "."
    markersize = 3
    if ax3d:
        ax3d.plot(
            track_points_sorted[x_str],
            track_points_sorted[y_str],
            track_points_sorted[z_str],
            label=particle_id,
            marker=marker,
            markersize=markersize,
        )

    axxy.plot(track_points_sorted[x_str], track_points_sorted[y_str], label=particle_id, marker=marker, markersize=markersize)  # type: ignore
    axxz.plot(track_points_sorted[x_str], track_points_sorted[z_str], label=particle_id, marker=marker, markersize=markersize)  # type: ignore
    axyz.plot(track_points_sorted[y_str], track_points_sorted[z_str], label=particle_id, marker=marker, markersize=markersize)  # type: ignore


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
    fig.tight_layout()
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
    fig.tight_layout()
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


def plot_prediction(
    truth: DataFrame,
    reconstructed: DataFrame,
    label: int,
    label_type="Seed",
    data_table="truth",
    tag: str | None = None,
):
    """Plot the prediction of a single seed"""
    # Show tracks

    plot_targets = generate_track_fig()
    fig = plot_targets[0]
    axes: tuple[Axes, Axes, Axes, Axes] = plot_targets[1:]  # type: ignore

    # TODO plot adjacent hits, too?

    add_track_to_fig(
        truth,
        *axes,
        particle_id=f"{label_type} {label} truth",
        type=data_table,
    )
    add_track_to_fig(
        reconstructed,
        *axes,
        particle_id=f"{label_type} {label} reconstructed",
        type=data_table,
    )

    if label_type.lower() == "seed":
        # Plot seed hit
        t_r = reconstructed
        s_x, s_y, s_z = t_r[t_r["hit_id"] == label][["tx", "ty", "tz"]].values[0]
        axes[0].plot([s_x], [s_y], [s_z], marker="o", color="red", markersize=15, label=label_type, zorder=0, alpha=0.2)
        axes[1].plot([s_x], [s_y], marker="o", color="red", markersize=15, label=label_type, zorder=0, alpha=0.2)
        axes[2].plot([s_x], [s_z], marker="o", color="red", markersize=15, label=label_type, zorder=0, alpha=0.2)
        axes[3].plot([s_y], [s_z], marker="o", color="red", markersize=15, label=label_type, zorder=0, alpha=0.2)

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    return fig


def get_range(
    data: list[DataFrame],
    variable: str,
    min: float | None = None,  # type: ignore
    max: float | None = None,  # type: ignore
):
    # Find range
    if min is None:
        mins: list[float] = [df[variable].min() for df in data]
        min: float = np.min(mins)
    if max is None:
        maxs: list[float] = [df[variable].max() for df in data]
        max: float = np.max(maxs)
    range: tuple[float, float] = (min, max)  # type: ignore
    return range


def compare_histograms(
    truth,
    test,
    variable: str,
    bins=100,
    x_min: float | None = None,  # type: ignore
    x_max: float | None = None,  # type: ignore
    y_min: float | None = None,  # type: ignore
    y_max: float | None = None,  # type: ignore
    density=False,
    title="",
    xlabel="x",
    ylabel="y",
    figsize=(10, 6),
    base: Figure | None = None,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=figsize) if base is None else (base, base.gca())

    range = get_range([truth, test], variable, x_min, x_max)
    ax.hist(x=[truth[variable], test[variable]], bins=bins, range=range, density=density, alpha=0.75, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)

    plt.legend()
    fig.tight_layout()
    return fig


def fraction_histogram(
    data: list[DataFrame],
    variable: str,
    bins=100,
    min: float | None = None,  # type: ignore
    max: float | None = None,  # type: ignore
    density=False,
    labels: list[str] | None = None,
    title="",
    xlabel="x",
    ylabel="y",
    figsize=(10, 6),
    type="fill",
    bar_pos=None,
    colors=None,
    base: Figure | None = None,
    **kwargs,
):
    # Find labels
    if labels is not None:
        assert len(labels) == len(data), "Number of labels must match number of dataframes"
    else:
        labels = [f"df_{i}" for i in np.arange(len(data))]

    if bar_pos is not None:
        assert len(bar_pos) == len(data), "Number of bar positions must match number of dataframes"

    # Find range
    range = get_range(data, variable, min, max)
    (min, max) = range

    # Make histograms
    hists = [np.histogram(df[variable], bins=bins, range=range, density=density) for df in data]
    tots = sum([hist[0] for hist in hists])

    # Plot
    fig, ax = plt.subplots(figsize=figsize) if base is None else (base, base.gca())

    x = np.arange(min, max, (max - min) / bins)
    fractions = np.nan_to_num([hist[0] / tots for hist in hists])
    cumsum = np.cumsum(fractions, axis=0)

    for i, (cum, label) in enumerate(zip(cumsum, labels)):
        if colors is not None:
            kwargs["color"] = colors[i]

        floor: float | np.ndarray = cumsum[i - 1] if i > 0 else np.zeros(len(cum))
        ceiling = cum
        bpos = bar_pos[i] if bar_pos is not None else x
        # ax.plot(x, cum, **kwargs)
        if type == "fill":
            ax.fill_between(x, floor, ceiling, label=label, **kwargs)  # type: ignore
        elif type == "bar":
            ax.bar(bpos, cum, label=label, zorder=-i, **kwargs)
        elif type == "barh":
            ax.barh(bpos, cum, label=label, zorder=-i, **kwargs)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Efficiency by {xlabel}" if title is None else title)

    ax.legend()

    ax.autoscale(enable=True, axis="both", tight=True)
    fig.tight_layout()
    return fig


def horizontal_fractions(
    data_arr: list[list[DataFrame]],
    variable: str,
    datalabels: list[str],
    min: float | None = None,  # type: ignore
    max: float | None = None,  # type: ignore
    labels: list[str] | None = None,
    title="",
    xlabel="x",
    ylabel="y",
    colors=None,
    **kwargs,
):
    fig: Figure = None  # type: ignore
    yticks = []
    ylabels = []

    for matches_arr in data_arr:
        if datalabels is None:
            datalabels = [""] * len(matches_arr)
        else:
            assert len(datalabels) == len(matches_arr), "Number of data labels must match number of dataframes"

    for i, tag in enumerate(datalabels):
        # for i, data in enumerate(zip(data_arr, datalabels)):
        match_types = [arr[i] for arr in data_arr]
        fig = fraction_histogram(
            match_types,
            variable=variable,
            labels=labels,
            bins=1,
            min=min,
            max=max,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            type="barh",
            bar_pos=len(match_types) * [i],
            base=fig,
            colors=colors,
            **kwargs,
        )
        yticks.append(i)
        ylabels.append(tag)
    fig.gca().set_yticks(yticks, ylabels)
    fig.gca().invert_yaxis()

    # Reset legend
    handles, _labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(_labels, handles))
    fig.gca().legend(by_label.values(), by_label.keys())

    fig.tight_layout()

    return fig


def plot_efficiency(
    truth: pd.DataFrame,
    test: pd.DataFrame,
    variable: str,
    bins: int = 100,
    x_min=None,
    x_max=None,
    title=None,
    xlabel="x",
    ylabel="Efficiency",
    datalabel: str | None = None,
    figsize=(10, 6),
    base: Figure | None = None,
    **kwargs,
):
    min_bin = truth[variable].min() if x_min is None else x_min
    max_bin = truth[variable].max() if x_max is None else x_max
    bins_index = pd.cut(pd.Series([min_bin, max_bin]), bins=bins, retbins=True)[1]

    cut_truth = pd.cut(truth[variable], bins=bins_index)  # type: ignore
    groups_truth = truth.groupby(cut_truth)
    binned_truth_count = groups_truth.count()[variable]

    cut_test = pd.cut(test[variable], bins=bins_index)  # type: ignore
    groups_test = test.groupby(cut_test)
    binned_test_count = groups_test.count()[variable]

    efficiency = binned_test_count / binned_truth_count

    fig, ax = plt.subplots(figsize=figsize) if base is None else (base, base.gca())

    x = np.arange(min_bin, max_bin, (max_bin - min_bin) / bins)
    yvariance = ((binned_test_count + 1) * (binned_test_count + 2)) / (
        (binned_truth_count + 2) * (binned_truth_count + 3)
    ) - ((binned_test_count + 1) ** 2) / ((binned_truth_count + 2) ** 2)
    yerr = np.sqrt(yvariance)
    # yerr = groups_test.std()[variable] / (np.sqrt(binned_test_count))
    # yerr = np.sqrt(binned_test_count) / binned_truth_count

    datalabel_str = f"$\\epsilon$ ({datalabel})" if datalabel else "$\\epsilon$"
    ax.plot(x, efficiency.values, label=datalabel_str, **kwargs)
    ax.fill_between(x, efficiency - yerr, efficiency + yerr, alpha=0.5, label=f"$\\Delta${datalabel_str}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Efficiency by {xlabel}" if title is None else title)
    ax.grid(True)
    ax.autoscale(enable=True, axis="both", tight=True)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_mean(
    truth: pd.DataFrame,
    test: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    bins: int = 100,
    x_min=None,
    x_max=None,
    title=None,
    xlabel="x",
    ylabel="Value",
    datalabel: str | None = None,
    figsize=(10, 6),
    base: Figure | None = None,
    **kwargs,
):
    min_bin = truth[x_variable].min() if x_min is None else x_min
    max_bin = truth[x_variable].max() if x_max is None else x_max
    bins_index = pd.cut(pd.Series([min_bin, max_bin]), bins=bins, retbins=True)[1]

    cut_test = pd.cut(test[x_variable], bins=bins_index)  # type: ignore
    groups_test = test.groupby(cut_test)
    binned_test_count = groups_test.count()[x_variable]

    yvalue = groups_test.mean()[y_variable]
    yerr = groups_test.std()[y_variable] / np.sqrt(binned_test_count)

    fig, ax = plt.subplots(figsize=figsize) if base is None else (base, base.gca())

    x = np.arange(min_bin, max_bin, (max_bin - min_bin) / bins)

    datalabel_str = datalabel if datalabel else f"<{y_variable}>"
    ax.plot(x, yvalue.values, label=datalabel_str, **kwargs)
    ax.fill_between(x, yvalue - yerr, yvalue + yerr, alpha=0.5, label=f"$\\Delta${datalabel_str}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Purity by {xlabel}" if title is None else title)
    ax.grid(True)
    ax.autoscale(enable=True, axis="both", tight=True)
    # ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    return fig


def _save_extra_figs(
    fig: Figure,
    dir: str,
    match_type_str: str,
    plot_type: str,
    variable: str,
    tag_str: str,
    y_min: float,
    y_max: float,
    extra_tag: str | None = None,
):
    extra_tag_str = "_" + extra_tag if extra_tag is not None else ""

    fig.savefig(dir + f"{match_type_str}_{variable}{extra_tag_str}_{plot_type}{tag_str}.jpg", dpi=600)

    fig.gca().set_ylim(y_min, y_max)
    fig.savefig(dir + f"{match_type_str}_{variable}{extra_tag_str}_{plot_type}{tag_str}_y_{y_min}_{y_max}.jpg", dpi=600)

    fig.gca().set_ylim(0, 1)
    make_compact(fig).savefig(dir + f"{match_type_str}_{variable}{extra_tag_str}_{plot_type}{tag_str}_big.jpg", dpi=600)
    fig.gca().set_ylim(y_min, y_max)
    fig.savefig(
        dir + f"{match_type_str}_{variable}{extra_tag_str}_{plot_type}{tag_str}_y_{y_min}_{y_max}_big.jpg", dpi=600
    )
    plt.close()


def evaluate_submission(
    particles_arr: list[DataFrame],
    pairs_arr: list[DataFrame],
    n_events_arr: list[int],
    tags_arr: list[str] | None = None,
    thr=0.5,
    dir="",
    bins=100,
):
    """Evaluate the submission by plotting histograms and efficiencies."""
    assert len(particles_arr) == len(pairs_arr), "Number of particles and pairs must match"
    if tags_arr is None:
        tags_arr = [""] * len(particles_arr)
    else:
        assert len(particles_arr) == len(tags_arr), "Number of particle lists and number of tags must match"

    prepare_path(dir)

    tag_str = "_" + "_".join(f"{n}_events" for n in n_events_arr)

    # Define types of matches
    good_arr = []
    split_arr = []
    multiple_arr = []
    bad_arr = []
    for particles, pairs in zip(particles_arr, pairs_arr):
        good_arr.append(pairs[(pairs["particle_purity"] >= thr) & (pairs["track_purity"] >= thr)])
        split_arr.append(pairs[(pairs["particle_purity"] < thr) & (pairs["track_purity"] >= thr)])
        multiple_arr.append(pairs[(pairs["particle_purity"] >= thr) & (pairs["track_purity"] < thr)])
        bad_arr.append(pairs[(pairs["particle_purity"] < thr) & (pairs["track_purity"] < thr)])

    match_types_arrs: list[list[pd.DataFrame]] = [good_arr, split_arr, multiple_arr, bad_arr]
    match_types_str: list[str] = ["good", "split", "multiple", "bad"]

    variables_str = ["r_0", "z_0", "p_0", "p_t_0", "log_10_p_t_0", "phi_0", "theta_0", "pseudo_rapidity_0"]
    var_labels = [
        "Detection point $r_0$ [mm]",
        "Detection point $z_0$ [mm]",
        "$p$ [GeV/c]",
        "$P_{T}$ [GeV/c]",
        "$log_{10}$ $p_{T}$",
        "$\\phi_0$",
        "$\\theta_0$",
        "$\\eta_0$",
    ]

    mean_variables = ["track_purity", "particle_purity"]
    mean_variables_str = ["track purity", "particle purity"]

    x_mins = [0, -15, 0, 0, None, -np.pi, 0, -np.pi]
    x_maxs = [600, 15, 25, 5, 1.3, np.pi, np.pi, np.pi]

    # For efficiencies only
    y_mins = [0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    y_maxs = [1, 1, 1, 1, 1, 1, 1, 1]

    # Plot serperately
    for match_type_str, matches_arr in zip(match_types_str, match_types_arrs):
        for variable, label, x_min, x_max in zip(variables_str, var_labels, x_mins, x_maxs):
            fig: Figure = None  # type: ignore
            for particles, matches, tag in zip(particles_arr, matches_arr, tags_arr):
                fig = compare_histograms(
                    particles,
                    matches,
                    variable=variable,
                    bins=bins,
                    x_min=x_min,
                    x_max=x_max,
                    title=f"Track {label}",
                    xlabel=label,
                    ylabel="Frequence",
                    label=[f"Truth {tag}", f"Reconstructed {tag}"],
                    histtype="step",
                    linewidth=1,
                    color=["blue", "orange"],
                    base=fig,
                )
            fig.savefig(dir + f"{match_type_str}_{variable}_histogram{tag_str}.jpg", dpi=600)
            make_compact(fig).savefig(dir + f"{match_type_str}_{variable}_histogram{tag_str}_big.jpg", dpi=600)
            plt.close()

        # Plot efficiency over variables
        for variable, label, x_min, x_max, y_min, y_max in zip(
            variables_str, var_labels, x_mins, x_maxs, y_mins, y_maxs
        ):
            fig: Figure = None  # type: ignore
            for particles, matches, tag in zip(particles_arr, matches_arr, tags_arr):
                fig = plot_efficiency(
                    particles,
                    matches,
                    variable=variable,
                    bins=bins,
                    x_min=x_min,
                    x_max=x_max,
                    xlabel=label,
                    datalabel=tag,
                    base=fig,
                )
            _save_extra_figs(fig, dir, match_type_str, variable, "efficiency", tag_str, y_min, y_max)

            # fig: Figure = None  # type: ignore
            # for particles, matches, tag in zip(particles_arr, matches_arr, tags_arr):
            #     for mean_var, mean_var_str in zip(mean_variables, mean_variables_str):
            #         fig = plot_mean(
            #             particles,
            #             matches,
            #             x_variable=variable,
            #             y_variable=mean_var,
            #             ylabel="Purity",
            #             x_max=600,
            #             xlabel=label,
            #             datalabel=mean_var_str,
            #             base=fig,
            #         )
            # _save_extra_figs(
            #     fig, dir, match_type_str, variable, f"means_{'_'.join(mean_variables)}", tag_str, y_min, y_max
            # )

        # Plot zoom of r_0
        fig: Figure = None  # type: ignore
        for particles, matches, tag in zip(particles_arr, matches_arr, tags_arr):
            fig = plot_efficiency(
                particles,
                matches,
                variable="r_0",
                bins=bins,
                x_min=30,
                x_max=100,
                xlabel="zoom vertex $r_0$ [mm]",
                datalabel=tag,
                base=fig,
            )
        _save_extra_figs(fig, dir, match_type_str, "r_0", "efficiency", tag_str, 0.6, 1, "zoom")

    # Plot stacked
    for variable, label, x_min, x_max, y_min, y_max in zip(variables_str, var_labels, x_mins, x_maxs, y_mins, y_maxs):
        for i, tag in enumerate(tags_arr):
            match_types = [arr[i] for arr in match_types_arrs]
            fig = fraction_histogram(
                match_types,
                variable=variable,
                labels=match_types_str,
                bins=bins,
                min=x_min,
                max=x_max,
                title=f"Track {label}",
                xlabel=label,
                ylabel="Fraction",
            )
            _save_extra_figs(fig, dir, "fractions", variable, "histogram", f"{tag_str}_{tag}", y_min, y_max)

    # Plot zoom of r_0
    for i, tag in enumerate(tags_arr):
        match_types = [arr[i] for arr in match_types_arrs]
        fig = fig = fraction_histogram(
            match_types,
            variable="r_0",
            labels=match_types_str,
            bins=bins,
            min=30,
            max=100,
            title=f"Track $r_0$",
            xlabel="zoom vertex $r_0$ [mm]",
            ylabel="Fraction",
        )
        _save_extra_figs(fig, dir, "fractions", "r_0", "histogram", f"{tag_str}_{tag}", 0.6, 1, "zoom")

    # Plot general purity distribution
    for i, tag in enumerate(tags_arr):
        match_types = [arr[i] for arr in match_types_arrs]
        [df.insert(0, "track_count", 1) for df in match_types]
        fig = fraction_histogram(
            match_types,
            variable="track_count",
            labels=match_types_str,
            bins=1,
            min=0,
            max=1,
            title="Track purity distribution",
            xlabel="",
            ylabel="Fraction",
            type="barh",
        )
        # remove irrelevant xticks
        fig.gca().set_xticks([])
        _save_extra_figs(fig, dir, "fractions", "track_count", "histogram", f"{tag_str}_{tag}", 0.6, 1, "zoom")

    fig = horizontal_fractions(
        match_types_arrs,
        datalabels=tags_arr,
        variable="track_count",
        labels=match_types_str,
        min=0,
        max=1,
        title="Track purity distribution",
        xlabel="Fraction",
        ylabel="",
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        height=1,
    )
    fig.savefig(dir + f"fractions_track_count_horizontal{tag_str}.jpg", dpi=600)


def plot_fit(X, Y, Z, x_new, y_new, z_new, vaxis: str, crop=2, **kwargs):
    assert vaxis.lower() in ["x", "y"], "vaxis must be either 'x' or 'y'"
    v_plot = Y if vaxis == "y" else X
    v_new_plot = y_new if vaxis == "y" else x_new

    zlim, vlim = get_crop(Z, v_plot, crop=crop)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(Z, v_plot, "o", **kwargs)
    ax.plot(z_new, v_new_plot, "r--", label="poly fit")
    ax.set_xlim(zlim)
    ax.set_ylim(*vlim)
    ax.set_xlabel("z")
    ax.set_ylabel(vaxis)
    ax.legend()
    fig.tight_layout()
    return fig, ax
