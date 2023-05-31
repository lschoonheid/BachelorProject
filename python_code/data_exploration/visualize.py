from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from _constants import FIG_X, FIG_Y, FIG_DPI, DETECTOR_KEYS, HITS_SAMPLES, TABLE_INDEX


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


def print_heads(event_kv: dict[str, DataFrame]):
    """Print a few datatable samples"""
    for name, table in event_kv.items():
        print("Table: " + name + ":")
        print(table.head())
        print("\n  \n")


def plot_hits(event_kv: dict[str, DataFrame], unique: bool = False):
    """Plot the hits"""

    # TODO cartesian product of all combinations of ids
    color_modes = DETECTOR_KEYS
    # Plot hits for all data types
    for table in ["hits", "truth"]:
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

        # Plot all combinations of axes
        for ax_keys in [[ax1, ax2, ax3], [ax1, ax2], [ax1, ax3], [ax2, ax3]]:
            ax_keys_str = "".join(ax_keys)

            # Plot hits for all color modes
            for color_mode in color_modes:
                pass
                # Plot hits
                fig = scatter(data_subset, *ax_keys, color_mode=color_mode)
                fig.savefig(f"{table}_{ax_keys_str}_{color_mode}_scatter_sample.png", dpi=FIG_DPI)
                plt.close()

            # Skip intensive part of loop if `unique` is False
            if not unique:
                continue

            # Draw seperate plots for each unique value of `color_mode`
            for color_mode in color_modes:
                unique_values = selected_data[color_mode].unique()
                for unique_value in unique_values:
                    isolated_detector_data = data_subset.loc[data_subset[color_mode] == unique_value]
                    # Since `color_mode` data is isolated, choose other color mode to distinguish data
                    anti_color_mode = color_modes[1 - color_modes.index(color_mode)]

                    fig = scatter(isolated_detector_data, *ax_keys, color_mode=anti_color_mode)
                    fig.savefig(
                        f"{table}_{ax_keys_str}_{color_mode}_{unique_value}_vs_{anti_color_mode}_scatter.png",
                        dpi=FIG_DPI,
                    )
                    plt.close()


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
    event_kv: dict[str, DataFrame], errors: dict[str, DataFrame] | None = None, prefix: str | None = None
):
    """Plot histograms of the hits"""

    # TODO:
    # [x] Charge distribution
    # [ ] Number of hits per particle
    # px py pz distribution, vs hits
    # ^-> look into 0 hits and 10 direction distribution

    # [ ] Number of hits per detector
    # [ ] Number of hits per layer
    # [ ] Number of hits per module
    # [ ] Number of hits per particle
    # [ ] Number of hits per particle per detector
    # [ ] Number of hits per particle per layer
    # [ ] Number of hits per particle per module
    # [ ] Heatmap of hits in xyz
    # [ ] Heatmap of hits in rphi
    # [ ] Heatmap of hits in rz
    # [ ] Heatmap of hits in r
    # [ ] Heatmap of hits in phi
    # momentum vs #hits

    # Single event
    for table, parameter in [["particles", "q"]]:
        fig = histogram(event_kv[table][parameter])
        fig.savefig(f"{prefix}_{table}_{parameter}_histogram.png", dpi=FIG_DPI)
        plt.close()


def parameter_distribution(
    events: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame]], table_type: str, parameter: str
):
    """Plot the distribution of a parameter over all `events`"""
    bins_df = DataFrame()
    unique_values = set()

    # Loop over events
    for index, event in enumerate(events):
        row = event[TABLE_INDEX[table_type]]

        unique_values = unique_values.union(set(row[parameter].unique()))
        bins = row[parameter].value_counts().sort_index()
        # Rename for readability
        bins.name = parameter + str(index)
        # Join series to dataframe
        bins_df = bins_df.merge(bins, how="outer", left_index=True, right_index=True)
    print(bins_df)

    hist_x = list(bins_df.index)
    hist_y = list(bins_df.mean(axis=1))
    hist_y_err = bins_df.std(axis=1)

    # Plot charge distribution
    fig = plt.figure(figsize=(FIG_X, FIG_Y))
    ax = fig.add_subplot()
    ax.set_title(f"Distribution of  { parameter } per {table_type} over { len(events) } events")
    ax.bar(hist_x, hist_y, width=0.1)
    ax.errorbar(hist_x, hist_y, yerr=hist_y_err, fmt="o", ecolor="black", capsize=2)
    return fig


def visualize_event(
    loaded_event: tuple[DataFrame, DataFrame, DataFrame, DataFrame],
    do_table: bool = True,
    do_plot_hits: bool = True,
    do_plot_histogram: bool = True,
    unique: bool = False,
    **kwargs,
):
    """Pipe for visualizing a single event"""
    # Load a single event
    hits, cells, particles, truth = loaded_event
    event_kv = {"hits": hits, "cells": cells, "particles": particles, "truth": truth}

    if do_table:
        print_heads(event_kv)

    if not any([do_plot_hits, do_plot_histogram]):
        return

    # Add detector identifiers to true particle data
    event_kv["truth"] = truth.merge(hits[["hit_id", *DETECTOR_KEYS]], on="hit_id")

    if do_plot_hits:
        plot_hits(event_kv, unique=unique)

    if do_plot_histogram:
        plot_histograms(event_kv)
