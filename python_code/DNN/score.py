import sys
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from data_exploration.helpers import get_logger  # type: ignore


# Checked
def get_track_scores(
    tracks_all: list[npt.NDArray], factor: int = 8, limit: int | None = None, index_shift=1
) -> npt.NDArray:
    """Generate confidence score for each track."""
    scores = np.zeros(len(tracks_all))

    if limit is not None:
        track_selection = tracks_all[:limit]
    else:
        track_selection = tracks_all

    for seed_index, path_ids in tqdm(
        enumerate(track_selection), total=len(tracks_all), desc="Generating track scores", file=sys.stdout
    ):
        n_hits = len(path_ids)

        # Skip paths with only one hit
        if n_hits > 1:
            tp = 0  # number of estimated true positives
            fp = 0  # number of estimated false positives
            for hit_id in path_ids:
                # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
                hit_index = hit_id - index_shift
                seed_referenced_path = tracks_all[hit_index]
                tp = tp + np.sum(np.isin(seed_referenced_path, path_ids, assume_unique=True))
                fp = fp + np.sum(np.isin(seed_referenced_path, path_ids, assume_unique=True, invert=True))

            # Calculate track score
            # TODO: understand this metric
            # tp = estimated true positives
            # fp = estimated false positives
            # (balance) factor = 8 (why?); punishing false positives more than true positives
            # n_hits = number of hits in path
            scores[seed_index] = (tp - fp * factor - n_hits) / n_hits / (n_hits - 1)
        else:
            # Useless path, set score to -inf
            scores[seed_index] = -np.inf
    return scores


# TODO: add comments
def score_event_fast(submission, truth: pd.DataFrame):
    """Calculate score of a single event based on `truth` information."""
    combined = truth[["hit_id", "particle_id", "weight"]].merge(submission, how="left", on="hit_id")
    grouped = combined.groupby(["track_id", "particle_id"]).hit_id.count().to_frame("count_both").reset_index()
    print(grouped[grouped["track_id"] == 1])
    combined = combined.merge(grouped, how="left", on=["track_id", "particle_id"])

    df1 = grouped.groupby(["particle_id"]).count_both.sum().to_frame("count_particle").reset_index()
    combined = combined.merge(df1, how="left", on="particle_id")
    df1 = grouped.groupby(["track_id"]).count_both.sum().to_frame("count_track").reset_index()
    combined = combined.merge(df1, how="left", on="track_id")
    combined.count_both *= 2
    score = combined[
        (combined.count_both > combined.count_particle) & (combined.count_both > combined.count_track)
    ].weight.sum()
    particles = combined[
        (combined.count_both > combined.count_particle) & (combined.count_both > combined.count_track)
    ].particle_id.unique()

    return (
        score,
        combined[combined.particle_id.isin(particles)].weight.sum(),
        1 - combined[combined.track_id > 0].weight.sum(),
    )


# TODO: add comments
def evaluate_tracks(tracks: npt.NDArray, truth: pd.DataFrame):
    """Evaluate tracks by comparing them to the ground truth."""
    submission = pd.DataFrame({"hit_id": truth.hit_id, "track_id": tracks})
    score = score_event_fast(submission, truth)[0]
    tracks_count = tracks.max()
    get_logger().info(
        "event score: %.4f | hits per track: %2.2f | #tracks: %4d | #noise %5d | weight missed %.4f | weight of unassigned %.4f"
        % (
            score,
            np.sum(tracks > 0) / tracks_count,
            tracks_count,
            np.sum(tracks == 0),
            1 - score - np.sum(truth.weight.values[tracks == 0]),
            np.sum(truth.weight.values[tracks == 0]),
        )
    )
