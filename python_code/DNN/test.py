from typing import Any
import random
import numpy as np
import numpy.typing as npt
import pandas as pd


import matplotlib.pyplot as plt

from data_exploration.visualize import plot_prediction
from data_exploration.event import Event
from features import get_module_id
from produce import get_path


TEST_THRESHOLD = 0.95


def test(
    event: Event,
    seed_0: int = 1,
    n_test: int = 1,
    test_thr: float = TEST_THRESHOLD,
    module_id: npt.NDArray | None = None,
    verbose: bool = True,
):
    """Test the model on a single event"""
    # Load event
    hits, cells, particles, truth = event.all
    features = event.features

    if module_id is None:
        module_id = get_module_id(hits)

    tracks = []

    # select one hit to construct a track
    for hit_id in range(seed_0, seed_0 + n_test):
        # Predict corresponding hits that belong to the same track
        reconstructed_ids = get_path(
            hit_id,
            thr=test_thr,
            mask=np.ones(len(hits)),
            module_id=module_id,
            features=features,
            hits=hits,
        )

        # Select data corresponding to reconstructed track
        hits_reconstructed = hits[hits.hit_id.isin(reconstructed_ids)]  # hits data of reconstructed track
        truth_reconstructed = truth[truth.hit_id.isin(reconstructed_ids)]  # truth data of reconstructed track

        # Select particle id of hit by index (hit_id - 1)
        particle_id = truth.particle_id[hit_id - 1]

        # Select data corresponding to true track
        truth_truth = truth[truth.particle_id == particle_id]  # truth data of true track
        truth_ids = truth_truth.hit_id.values
        hits_truth = hits[hits.hit_id.isin(truth_ids)]  # hits data of true track

        tracks.append(
            {
                "seed_id": hit_id,
                "seed_particle_id": particle_id,
                "truth_ids": truth_ids,
                "reconstructed_ids": reconstructed_ids,
                "hits_truth": hits_truth,
                "truth_truth": truth_truth,
                "hits_reconstructed": hits_reconstructed,
                "truth_reconstructed": truth_reconstructed,
            }
        )

        if verbose:
            print("hit_id = ", hit_id)
            print("reconstruct :", reconstructed_ids)
            print("ground truth:", truth_ids.tolist())
            print(truth_truth)
            print(truth_reconstructed)

    return tracks


def show_test(
    event: Event,
    module_id,
    repeats: int = 1,
    n_test: int = 1,
    pick_random: bool = True,
    animate: bool = False,
):
    # Generate some tracks and compare with truth
    for i in range(1, 1 + repeats):
        # set seed
        n_max = 10000  # TODO get length of hits table
        seed = random.randrange(1, n_max) if pick_random else i

        # Generate some track(s)
        generated_tracks = test(event, module_id=module_id, seed_0=seed, n_test=n_test)

        for track in generated_tracks:
            # Get truth and reconstructed hits
            truth: pd.DataFrame = track["truth_truth"]
            reconstructed: pd.DataFrame = track["truth_reconstructed"]

            # Plot track
            if animate:
                # Order by the order in which the hits were added to the track.
                reconstruction_order_index = track["reconstructed_ids"] - 1
                reconstructed = reconstructed.reindex(reconstruction_order_index)

                # Divide the track into frames
                frames_total = len(reconstructed)
                for f in range(frames_total):
                    tag = f"_f{f+1}:{frames_total}"
                    fig = plot_prediction(truth, reconstructed[: f + 1], seed, tag=tag)
                    fig.savefig(f"{n_test}_generated_tracks_seed_{seed}{tag}.png", dpi=300)
                    plt.close()

            else:
                fig = plot_prediction(truth, reconstructed, seed)
                fig.savefig(f"{n_test}_generated_tracks_seed_{seed}.png", dpi=300)
                plt.close()
