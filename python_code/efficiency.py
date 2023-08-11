import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from trackml.dataset import load_event_truth

from data_exploration.constants import DIRECTORY, DATA_SAMPLE
from data_exploration.helpers import find_file, find_files, prepare_path, add_r, select_r_0, extend_features, get_logger
from data_exploration.visualize import compare_histograms, evaluate_submission
from dirs import OUTPUT_DIR, PLOT_PATH


def get_truth(submission: DataFrame, truth_dir: str = DATA_SAMPLE, output_dir: str = OUTPUT_DIR):
    """Link truth data with submission data."""
    # Assert one event
    event_id = submission["event_id"].unique()
    assert len(event_id) == 1, "Submission must contain one event"
    # Reconstruct event name
    event_name = "event000000000"
    event_name = event_name[: -len(str(event_id[0]))] + str(event_id[0])

    # Load truth data
    # First try to load from output_dir, where reduced truth data is stored
    truth = find_file(f"truth_{event_name}", dir=output_dir, extension="csv")
    if truth is None:
        truth = load_event_truth(DATA_SAMPLE + event_name)

    # # Link together
    # combined: DataFrame = submission.merge(truth, how="right", on="hit_id")
    # # combined_relevant = combined[combined["particle_id"] > 0]
    return truth


def get_pairs(submission: DataFrame, truth: DataFrame):
    combined: pd.DataFrame = truth.merge(submission, how="left", on="hit_id")  # type: ignore
    return combined[
        [
            "hit_id",
            "particle_id",
            "track_id",
            "event_id",
        ]
    ]


def count_both(pairs: DataFrame):
    """Add column with number of hits overlapping between track and particle."""
    # Group by unique combinations of track_id (our) and particle_id (truth); count number of hits overlapping
    grouped: DataFrame = (
        pairs.groupby(
            [
                "event_id",
                "track_id",
                "particle_id",
            ]
        )
        .hit_id.count()
        .to_frame("count_both")
        .reset_index()
    )
    # grouped_relevant = grouped[(grouped["track_id"] > 0) & (grouped["particle_id"] > 0)]
    return grouped


def count_reconstructed(pairs: DataFrame):
    reco_count = pairs[pairs["track_id"] != 0].value_counts("track_id").to_frame("count_reco").sort_index()
    return reco_count


def count_truth(combined: DataFrame):
    truth_count = (
        combined[combined["particle_id"] != 0].value_counts("particle_id").to_frame("count_truth").sort_index()
    )
    return truth_count


def select_particles(truth: DataFrame, min_hits: int = 4):
    p_count = (
        truth[truth["particle_id"] != 0]
        .value_counts("particle_id")
        .sort_values(ascending=False)
        .to_frame("count_truth")
    )
    selected_particles = p_count[p_count["count_truth"] >= min_hits]
    return selected_particles


def add_purities(
    all_pairs: DataFrame, considered_particles: DataFrame | None = None, min_hits: int = 4, thr=0.5, verbose=False
):
    """Select pairs of tracks and particles that are considered good matches.
    Only tracks and particles with at least `min_hits` are considered.
    Both track purity and particle purity have to be above `thr`.
    """
    # Count hits overlapping between track and particle
    n_both = count_both(all_pairs)

    # Count reconstructed tracks hits
    n_reco = count_reconstructed(all_pairs)
    reco_considered = n_reco[n_reco["count_reco"] >= min_hits]

    # Filter discarded tracks
    valid_pairs = n_both[
        (n_both["particle_id"] != 0) & (n_both["track_id"] != 0) & (n_both["count_both"] > 1)
    ].reset_index(drop=True)

    # Select particles with enough hits
    particles_considered = (
        select_particles(all_pairs, min_hits=min_hits) if considered_particles is None else considered_particles
    )

    # Combine counts on considered truth and reco tracks
    considered_pairs = (
        valid_pairs[valid_pairs["particle_id"].isin(particles_considered.index)][
            valid_pairs["track_id"].isin(reco_considered.index)
        ]
        .sort_values(["count_both", "track_id"], ascending=[False, True])
        .reset_index(drop=True)
        .merge(particles_considered, on="particle_id")
        .merge(reco_considered, on="track_id")
    )

    # Find particles linked to too many reco tracks
    duplicates_mask = considered_pairs[considered_pairs.duplicated(subset="particle_id", keep="first")].sort_index()
    # Select majority track for reco's
    primary_matched = considered_pairs[~considered_pairs.index.isin(duplicates_mask.index)]
    # Define 'good' tracks
    track_purity = primary_matched["count_both"] / primary_matched["count_reco"]
    particle_purity = primary_matched["count_both"] / primary_matched["count_truth"]
    primary_matched.insert(5, "particle_purity", particle_purity)
    primary_matched.insert(7, "track_purity", track_purity)

    good_pairs = primary_matched[(primary_matched["particle_purity"] >= thr) & (primary_matched["track_purity"] >= thr)]

    if verbose:
        # Confirm too many matches for amount of particles, should consider only primary particles
        print("Number of matches: ", len(valid_pairs))
        print("Number of particles", len(particles_considered))
        efficiency = len(good_pairs) / len(particles_considered)
        print(f"Efficiency: {100*efficiency:.2f}%")
    return good_pairs


def extract(dir=OUTPUT_DIR, event_name=None, min_hits=4, verbose=False):
    min_hits = 4

    if event_name is None:
        submissions: list[pd.DataFrame] = find_files("submission_tracks", dir=dir, extension="pkl")
    else:
        # Inspect specific event
        submission = find_file(f"submission_tracks_{event_name}", dir=dir, extension="pkl")
        if submission is None:
            raise FileNotFoundError(f"Event {event_name} not found")
        submissions: list[pd.DataFrame] = [submission]

    assert len(submissions) != 0, "No submissions found"

    particles_list = []
    considered_pairs_list = []
    for submission in submissions:
        assert "event_id" in submission, "Submission does not contain event_id"

        truth = get_truth(submission, output_dir=dir)
        pairs = get_pairs(submission, truth)

        considered_particles = select_particles(pairs, min_hits=min_hits)
        considered_pairs = add_purities(
            all_pairs=pairs, considered_particles=considered_particles, min_hits=min_hits, thr=0
        )

        r_0 = extend_features(select_r_0(truth))
        considered_particles_extended = considered_particles.merge(r_0, on=["particle_id"])

        considered_pairs_extended = considered_pairs.merge(r_0, on=["particle_id"])
        particles_list.append(considered_particles_extended)
        considered_pairs_list.append(considered_pairs_extended)

        if verbose:
            get_logger().debug(considered_pairs)
            get_logger().debug(considered_particles_extended)
            get_logger().debug(considered_pairs_extended)

    # Combine all events
    all_considered_particles_extended = pd.concat(particles_list)
    all_considered_pairs_extended = pd.concat(considered_pairs_list)
    n_events = len(submissions)

    return all_considered_particles_extended, all_considered_pairs_extended, n_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")

    parser.add_argument("-d", dest="dir", type=str, default=OUTPUT_DIR, help="Choose directory")
    parser.add_argument("-e", dest="event_name", type=str, default=None, help="Choose event name")
    parser.add_argument("-o", dest="output_dir", type=str, default=PLOT_PATH, help="Choose event name")

    args = parser.parse_args()
    kwargs = vars(args)

    all_considered_particles_extended, all_considered_pairs_extended, n_events = extract(
        dir=kwargs.pop("dir"), event_name=kwargs.pop("event_name")
    )

    # Plot evaluations
    evaluate_submission(
        all_considered_particles_extended,
        all_considered_pairs_extended,
        tag=f"{n_events}_events",
        dir=kwargs.pop("output_dir"),
        bins=100,
    )
