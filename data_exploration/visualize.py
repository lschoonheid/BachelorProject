from pandas import DataFrame, Series
from matplotlib import pyplot as plt

DETECTOR_KEYS = ["volume_id", "layer_id", "module_id"]
HITS_SAMPLES = None


def get_colors(data: DataFrame, mode: str = "volume_layer"):
    """Map data to colors"""
    match mode:
        case "volume_layer":
            # TODO
            # test = data.apply(lambda x: str(int(x["layer_id"] / 2)) + str(int(x["volume_id"])), axis=1)
            # # test = data.apply(lambda x: "".join([str(int(x[key])) for key in ["volume_id", "layer_id"]]), axis=1)
            # _list = list(test)
            # # unique = combined.unique()
            return data.volume_id
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
    fig = plt.figure(figsize=(10, 10))
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


def visualize(
    loaded_event: tuple[DataFrame, DataFrame, DataFrame, DataFrame],
    plot_event: bool = True,
    unique: bool = False,
    **kwargs,
):
    """Visualize the data"""
    # Load a single event
    hits, cells, particles, truth = loaded_event

    # Print a few datatable samples
    event_kv = {"hits": hits, "cells": cells, "particles": particles, "truth": truth}
    for name, table in event_kv.items():
        print("Table: " + name + ":")
        print(table.head())
        print("\n \n")

    if not plot_event:
        return

    # TODO
    color_modes = [*DETECTOR_KEYS]
    # color_modes = ["volume_layer", *DETECTOR_KEYS]

    # Add detector identifiers to true particle data
    event_kv["truth"] = truth.merge(hits[["hit_id", *DETECTOR_KEYS]], on="hit_id")

    # Plot hits for all data types
    for data_type in ["hits", "truth"]:
        # Define axis keys
        match data_type:
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
        selected_data = event_kv[data_type]

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
                # Plot hits
                fig = scatter(data_subset, *ax_keys, color_mode=color_mode)
                fig.savefig(f"{data_type}_{ax_keys_str}_{color_mode}_scatter_sample.png", dpi=600)
                plt.close()

            # Skip intensive part of loop if `unique` is False
            if not unique:
                continue

            # Draw seperate plots for each unique value of `color_mode`
            for color_mode in color_modes[:2]:
                unique_values = selected_data[color_mode].unique()
                for unique_value in unique_values:
                    isolated_detector_data = data_subset.loc[data_subset[color_mode] == unique_value]
                    # Since `color_mode` data is isolated, choose other color mode to distinguish data
                    anti_color_mode = color_modes[1 - color_modes.index(color_mode)]

                    fig = scatter(isolated_detector_data, *ax_keys, color_mode=anti_color_mode)
                    fig.savefig(
                        f"{data_type}_{ax_keys_str}_{color_mode}_{unique_value}_vs_{anti_color_mode}_scatter.png",
                        dpi=600,
                    )
                    plt.close()
