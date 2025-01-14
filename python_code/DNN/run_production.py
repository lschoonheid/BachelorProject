import argparse
import logging
import os
import numpy.typing as npt
import pandas as pd


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
import matplotlib.pyplot as plt

from trackml.score import score_event

from data_exploration.helpers import get_logger, datetime_str, find_file, save, cached, select_r_less  # type: ignore
from data_exploration.visualize import plot_prediction
from data_exploration.event import Event
from helpers.model import get_model
from helpers.test import show_test  # type: ignore
from helpers.features import get_featured_event, get_module_id
from helpers.predict import make_predict_matrix
from helpers.produce import get_all_paths, run_merging
from helpers.score import get_track_scores, score_event_fast

from helpers.dirs import LOG_DIR, MODELS_ROOT, OUTPUT_DIR, DATA_SAMPLE, DATA_1


def run(
    event_name: str = "event000001001",
    new_model=False,
    mode="tracks",
    reduce=0.05,
    preload=True,
    do_export=True,
    batch_size=20000,
    do_test: bool = False,
    repeats=20,
    n_test=1,
    pick_random=True,
    do_fit=False,
    seed=0,
    animate=False,
    dir=OUTPUT_DIR,
    verbose=True,
    abs_tolerance_line=10,
    abs_tolerance_r=10,
    abs_tolerance_trig=10,
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
    event: Event = get_featured_event(
        event=Event(dir=DATA_1, event_name=event_name).reduce(fraction=reduce, random=pick_random, seed=seed)
    )

    if reduce is not None:
        save(event.truth, f"truth_{event_name}_reduced_{reduce}", dir=dir, save=do_export, extension="csv")

    hits = event.hits
    module_id = get_module_id(hits)

    # Choose hit subjects
    if mode == "tracks":
        # Select all hits as subjects
        subject_idx = None
        stages = [
            {"thr": 6},  # 0: merge
            {"thr": 0.6},  # 1: extend
            {"thr": 3, "thr_extend_0": 3, "thr_extend_1": 0.6},  # 2: merge + extend
            {"thr": 0.5},  # 3: extend
            {"thr": 2, "thr_extend_0": 1, "thr_extend_1": 0.5},  # 4: merge + extend
            {"thr": 0.5},  # 5: extend
        ]
    elif mode == "tracklets":
        subject_idx = select_r_less(hits, thr=300)
        # stages = [
        #     {"thr": 2},  # 0: merge
        #     {"thr": 0.6},  # 1: extend
        #     {"thr": 1, "thr_extend_0": 3, "thr_extend_1": 0.6},  # 2: merge + extend
        #     {"thr": 0.5},  # 3: extend
        #     {"thr": 1, "thr_extend_0": 1, "thr_extend_1": 0.5},  # 4: merge + extend
        #     {"thr": 0.5},  # 5: extend
        # ]
        # stages = [
        #     {"thr": 3},  # 0: merge
        #     {"thr": 0.6},  # 1: extend
        #     {"thr": 2, "thr_extend_0": 3, "thr_extend_1": 0.6},  # 2: merge + extend
        #     {"thr": 0.5},  # 3: extend
        #     {"thr": 1, "thr_extend_0": 1, "thr_extend_1": 0.5},  # 4: merge + extend
        #     {"thr": 0.5},  # 5: extend
        # ]
        stages = [
            {"thr": 6},  # 0: merge
            {"thr": 0.6},  # 1: extend
            {"thr": 3, "thr_extend_0": 3, "thr_extend_1": 0.6},  # 2: merge + extend
            {"thr": 0.5},  # 3: extend
            {"thr": 2, "thr_extend_0": 1, "thr_extend_1": 0.5},  # 4: merge + extend
            {"thr": 0.5},  # 5: extend
        ]
    else:
        raise ValueError(f"Mode {mode} not supported.")

    model = get_model(preload=not new_model, save=do_export, dir=MODELS_ROOT)
    logger.info("Model loaded")

    if do_test:
        # Test model, output some visualized tracks
        show_test(event, module_id, repeats, n_test, pick_random, animate)

    preds: list[npt.NDArray] = cached(
        f"preds_{event_name}",
        dir=dir,
        fallback_func=lambda: make_predict_matrix(model, event.features, batch_size=batch_size),
        force_fallback=not preload,
        do_save=do_export,
    )  # type: ignore
    logger.info("Predictions loaded")

    # Generate tracks for each hit as seed
    thr: float = 0.85

    tracks_all: list[npt.NDArray] = cached(
        f"{mode}_all_{event_name}",
        dir=dir,
        fallback_func=lambda: get_all_paths(
            hits,
            thr,
            module_id,
            preds,
            do_redraw=True,
            subject_idx=subject_idx,
            fit=do_fit,
            abs_tolerance_line=abs_tolerance_line,
            abs_tolerance_r=abs_tolerance_r,
            abs_tolerance_trig=abs_tolerance_trig,
        ),
        force_fallback=not preload,
        do_save=do_export,
    )  # type: ignore
    logger.info(f"{mode} loaded")

    # calculate track's confidence
    scores: npt.NDArray = cached(f"scores_{mode}_{event_name}", dir=dir, fallback_func=lambda: get_track_scores(tracks_all, subject_idx=subject_idx), force_fallback=not preload, do_save=do_export)  # type: ignore
    logger.info("Scores loaded")

    # Merge tracks
    merged_tracks: npt.NDArray = cached(
        f"merged_{mode}_{event_name}",
        dir=dir,
        fallback_func=lambda: run_merging(
            tracks_all,
            scores,
            preds,
            multi_stage=True,
            module_id=module_id,
            log_evaluations=True,
            truth=event.truth,
            subject_idx=subject_idx,
            fit=do_fit,
            hits=hits,
            stages=stages,
            abs_tolerance_line=abs_tolerance_line,
            abs_tolerance_r=abs_tolerance_r,
            abs_tolerance_trig=abs_tolerance_trig,
        ),
        force_fallback=not preload,
        do_save=do_export,
    )  # type: ignore
    logger.info("Merged tracks loaded")

    # Save submission
    submission: pd.DataFrame = cached(
        f"submission_{mode}_{event_name}",
        dir=dir,
        fallback_func=lambda: pd.DataFrame({"event_id": event_id, "hit_id": hits.hit_id, "track_id": merged_tracks}),
        force_fallback=not preload,
        do_save=do_export,
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
        # for n_start in [0, 1000, 2000, 3000, 4000, 5000]:
        for n_start in [0, 50, 100, 150, 300]:
            for i in range(n_start, n_start + n_per_cycle):
                # Show tracks
                # Tracks are already ordered by score
                track_id = i
                possible_particle_ids: pd.DataFrame = grouped[grouped["track_id"] == track_id].sort_values(
                    "count_both", ascending=False
                )
                logger.debug(f"Possible particle ids: \n {possible_particle_ids}")
                if len(possible_particle_ids) == 0:
                    # No match
                    continue
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
                fig.savefig(f"reconstructed_track_{track_id}_{event_name}_{most_likely_particle_id}.png", dpi=300)
                plt.close()
    logger.info(f"End: { datetime_str()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="data_exploration", description="Explore the data")

    parser.add_argument("-e", dest="event_name", type=str, default="event000001001", help="Choose event name")
    parser.add_argument("-bs", dest="batch_size", type=int, default=20000, help="Choose batch_size")
    parser.add_argument("-r", dest="reduce", type=float, default=0.05, help="Choose batch_size")
    parser.add_argument("-d", dest="dir", type=str, default=OUTPUT_DIR, help="Choose batch_size")
    parser.add_argument("-f", dest="do_fit", action="store_true", help="Run with fitting during track construction")
    parser.add_argument("-t", dest="tol_h", type=float, default=10, help="Choose helix tolerance")

    args = parser.parse_args()
    kwargs = vars(args)

    tol_h = kwargs.pop("tol_h")
    kwargs["abs_tolerance_line"] = tol_h
    kwargs["abs_tolerance_r"] = tol_h
    kwargs["abs_tolerance_trig"] = tol_h

    run(**kwargs)
