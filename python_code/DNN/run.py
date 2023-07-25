import argparse
import logging
import os
import numpy.typing as npt
import pandas as pd

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
import matplotlib.pyplot as plt

from model import get_model
from test import show_test  # type: ignore
from data_exploration.helpers import get_logger, datetime_str, find_file, save  # type: ignore
from data_exploration.visualize import plot_prediction
from trackml.score import score_event
from features import get_featured_event, get_module_id
from predict import make_predict_matrix
from produce import get_all_paths, run_merging
from score import get_track_scores, score_event_fast

from dirs import LOG_DIR, OUTPUT_DIR


def run(
    event_name: str = "event000001001",
    new_model=False,
    preload=True,
    do_export=True,
    batch_size=20000,
    do_test: bool = False,
    repeats=20,
    n_test=1,
    pick_random=False,
    animate=False,
    dir=OUTPUT_DIR,
    verbose=True,
    **kwargs,
):
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = get_logger(tag=event_name, dir=LOG_DIR, level=log_level)
    logger.debug("Start")
    event_id = int(event_name.split("event")[-1])
    logger.debug(f"Vars: { locals()}")

    logger.info(f"Num GPUs Available: { len(tf.config.list_physical_devices('GPU'))}")
    logger.debug(tf.config.list_physical_devices("GPU"))

    # Load event and extract required data for prediction
    event = get_featured_event(event_name)
    hits = event.hits
    module_id = get_module_id(hits)

    model = get_model(preload=not new_model, save=do_export)
    logger.info("Model loaded")

    if do_test:
        # Test model, output some visualized tracks
        show_test(event, module_id, repeats, n_test, pick_random, animate)

    # Make prediction matrix for all hits in the event
    # Look for prediction matrices already existing:
    _make_predict = lambda: save(
        make_predict_matrix(model, event.features, batch_size=batch_size),
        name="preds",
        tag=event_name,
        prefix=OUTPUT_DIR,
        save=do_export,
    )
    preds: list[npt.NDArray] = find_file(f"preds_{event_name}", dir=OUTPUT_DIR, fallback_func=_make_predict, force_fallback=not preload)  # type: ignore
    logger.info("Predictions loaded")

    # Generate tracks for each hit as seed
    thr: float = 0.85

    _make_tracks = lambda: save(
        get_all_paths(hits, thr, module_id, preds, do_redraw=True),
        name="tracks_all",
        tag=event_name,
        prefix=OUTPUT_DIR,
        save=do_export,
    )
    tracks_all: list[npt.NDArray] = find_file(f"tracks_all_{event_name}", dir=OUTPUT_DIR, fallback_func=_make_tracks, force_fallback=not preload)  # type: ignore
    logger.info("Tracks loaded")

    # calculate track's confidence
    _make_scores = lambda: save(
        get_track_scores(tracks_all), name="scores", tag=event_name, prefix=OUTPUT_DIR, save=do_export
    )
    scores: npt.NDArray = find_file(f"scores_{event_name}", dir=OUTPUT_DIR, fallback_func=_make_scores, force_fallback=not preload)  # type: ignore
    logger.info("Scores loaded")

    # Merge tracks
    _make_merged_tracks = lambda: save(
        run_merging(
            tracks_all, scores, preds, multi_stage=True, module_id=module_id, log_evaluations=True, truth=event.truth
        ),
        name="merged_tracks",
        tag=event_name,
        prefix=OUTPUT_DIR,
        save=do_export,
    )  # type: ignore
    merged_tracks: npt.NDArray = find_file(
        f"merged_tracks_{event_name}", dir=OUTPUT_DIR, fallback_func=_make_merged_tracks, force_fallback=not preload
    )  # type: ignore
    logger.info("Merged tracks loaded")

    # Save submission
    _make_submission = lambda: save(
        pd.DataFrame({"event_id": event_id, "hit_id": hits.hit_id, "track_id": merged_tracks}),
        name="submission",
        tag=event_name,
        prefix=OUTPUT_DIR,
        save=do_export,
    )
    submission: pd.DataFrame = find_file(
        f"submission_{event_name}", dir=OUTPUT_DIR, fallback_func=_make_submission, force_fallback=not preload
    )  # type: ignore
    logger.info("Submission loaded")

    # Evaluate submission
    score = score_event(event.truth, submission)
    logger.info(f"TrackML Score:{ score}")
    logger.info(f"Fast score: {score_event_fast(submission, event.truth)}")

    if do_test:
        # Add our track_id to truth
        combined: pd.DataFrame = event.truth[["hit_id", "particle_id", "weight", "tx", "ty", "tz"]].merge(
            submission, how="left", on="hit_id"
        )
        # Group by unique combinations of track_id (our) and particle_id (truth); count number of hits overlapping
        grouped: pd.DataFrame = (
            combined.groupby(["track_id", "particle_id"]).hit_id.count().to_frame("count_both").reset_index()
        )

        # Show some tracks
        n_per_cycle = 10
        for n_start in [0, 1000, 2000, 3000, 4000, 5000]:
            for i in range(n_start, n_start + n_per_cycle):
                # Show tracks
                # Tracks are already ordered by score
                track_id = i
                possible_particle_ids: pd.DataFrame = grouped[grouped["track_id"] == track_id].sort_values(
                    "count_both", ascending=False
                )
                most_likely_particle_id = int(possible_particle_ids.iloc[0]["particle_id"])

                # Select related truth and reconstructed hits
                reconstructed_track = combined[combined["track_id"] == track_id]
                truth_track = combined[combined["particle_id"] == most_likely_particle_id]

                track_ids = reconstructed_track["hit_id"].values
                logger.info(f"Selected track ids: \n {track_ids }")
                logger.info(reconstructed_track)

                # Do some weight analysis
                reconstructed_weight_total = reconstructed_track["weight"].sum()
                reconstructed_weight_overlap = reconstructed_track[
                    reconstructed_track["particle_id"] == most_likely_particle_id
                ]["weight"].sum()
                truth_weight = truth_track["weight"].sum()
                ratio = reconstructed_weight_overlap / truth_weight
                logger.info(
                    f"Track {track_id} has total weight {reconstructed_weight_total}, vs {truth_weight} from particle {most_likely_particle_id}, ratio: {ratio:.4f}"
                )

                # Make figure
                fig = plot_prediction(
                    truth_track, reconstructed_track, most_likely_particle_id, label_type="particle_id"
                )
                fig.suptitle(
                    f"Track {track_id} with particle id {most_likely_particle_id} \n\
                    weight ratio: {ratio:.2f}\
                    ",
                    fontsize=20,
                )
                fig.savefig(f"reconstructed_track_{track_id}_{event_name}.png", dpi=300)
                plt.close()
    logger.info(f"End: { datetime_str()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")

    parser.add_argument("-e", dest="event_name", type=str, default="event000001001", help="Choose event name")
    parser.add_argument("-bs", dest="batch_size", type=int, default=20000, help="Choose batch_size")

    args = parser.parse_args()
    kwargs = vars(args)

    run(**kwargs)
