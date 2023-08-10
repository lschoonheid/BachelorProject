import logging
import os
import sys
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd

from data_exploration.helpers import get_logger


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from keras.models import Model
from tqdm import tqdm

from .features import mask_same_module
from .predict import retrieve_predict, make_predict
from .fit import get_validator
from .score import evaluate_tracks

from .dirs import *


def validator_from_path_idx(
    path_indices: list[int],
    hits: pd.DataFrame,
    abs_tolerance_line: float,
    abs_tolerance_r: float,
    abs_tolerance_trig: float,
    **kwargs,
) -> Callable[[float, float, float], dict]:
    if len(path_indices) > 3:
        path_indices_selection = path_indices[:3]
    else:
        path_indices_selection = path_indices

    x_2, y_2, z_2 = hits.iloc[path_indices_selection][["x", "y", "z"]].values.T  # type: ignore
    # Add origin
    x_0, y_0, z_0 = 0, 0, 0
    x_3 = np.append(x_2, x_0)
    y_3 = np.append(y_2, y_0)
    z_3 = np.append(z_2, z_0)

    # Get validator
    validator = get_validator(
        x_3,  # type: ignore
        y_3,  # type: ignore
        z_3,  # type: ignore
        abs_tolerance_line=abs_tolerance_line,
        abs_tolerance_r=abs_tolerance_r,
        abs_tolerance_trig=abs_tolerance_trig,
        verbose=False,
    )

    return validator


def path_add_one(
    path_indices: list[int],
    mask: npt.NDArray,
    a: float | npt.NDArray,
    validator: Callable[[float, float, float], dict] | None,
    thr: float,
    module_id: npt.NDArray,
    skip_same_module: bool = True,
    preds: list[npt.NDArray] | None = None,
    features: npt.NDArray | None = None,
    hits: pd.DataFrame | None = None,
    model: Model | None = None,
    last=False,
    fit: bool = False,
    abs_tolerance_line: float = 10,
    abs_tolerance_r: float = 10,
    abs_tolerance_trig: float = 10,
    **kwargs,
):
    if fit and hits is None:
        raise ValueError("hits must be provided when fitting")

    # Predict probability of each pair of hits with the last hit in the path
    hit_id_last = path_indices[-1] + 1

    if preds is not None:
        p = retrieve_predict(hit_id_last, preds)
    else:
        p = make_predict(model=model, features=features, hits=hits, hit_id=hit_id_last, thr=thr)  # type: ignore

    if not last:
        # Generate mask of hits that have a probability above the threshold
        mask = (p > thr) * mask
    # Mask last added hit
    mask[path_indices[-1]] = 0

    if skip_same_module:
        path_ids = np.array(path_indices) + 1
        mask = mask_same_module(mask, path_ids, p, thr, module_id)

    # `a` is the culuminative probability between each hit in the path
    # At each step we look at the best candidate for the whole (previously geberate) track
    a = (p + a) * mask

    # When two points are added, we can build a helix and use it to validate the next point
    if fit and validator is None and len(path_indices) >= 2:
        validator = validator_from_path_idx(**locals())  # type: ignore

    # TODO
    # p_ = p + diffs(p) * weight

    # TODO
    # if validator(x,y)
    if validator:
        for i in range(len(a)):
            cand_idx = a.argmax()
            x_t, y_t, z_t = hits.iloc[cand_idx][["x", "y", "z"]].values  # type: ignore
            validation = validator(x_t, y_t, z_t)
            validated = validation["validated"]
            if not validated and a.max() > thr * len(path_indices):
                a[cand_idx] = 0
                continue
            else:
                break

    # Add index of hit with highest probability to path, proceed with this hit as the seed for the next iteration
    if a.max() > thr * len(path_indices) and not last:
        path_indices.append(a.argmax())  # type: ignore
        do_break = False
    # Breaking condition: if best average probability is below threshold, end path
    else:
        do_break = True

    return path_indices, mask, a, validator, do_break


def get_path(
    hit_id: int,
    thr: float,
    mask: npt.NDArray,
    module_id: npt.NDArray,
    skip_same_module: bool = True,
    preds: list[npt.NDArray] | None = None,
    features: npt.NDArray | None = None,
    hits: pd.DataFrame | None = None,
    model: Model | None = None,
    fit=False,
    abs_tolerance_line=10,
    abs_tolerance_r=10,
    abs_tolerance_trig=10,
):
    """Predict set of hits that belong to the same track as hit_id.
    Returns list[hit_id].
    """
    # Verify correct input
    if preds is None and (features is None or hits is None or model is None):
        raise ValueError("Either preds or features and truth must be provided")
    if fit and hits is None:
        raise ValueError("hits must be provided if fit=True")

    # Convert to index
    hit_index = hit_id - 1
    path_indices = [hit_index]
    a = 0

    validator = None
    while True:
        # Keep adding hits until breaking condition is met
        path_indices, mask, a, validator, do_break = path_add_one(**locals())
        if do_break:
            break
    # Convert indices back to hit_ids by adding 1
    return np.array(path_indices) + 1


def redraw(
    path_ids: npt.NDArray,
    hit_id: int,
    thr: float,
    mask: npt.NDArray,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    fit: bool,
    hits: pd.DataFrame | None = None,
):
    """Try redrawing path with one hit removed for improved confidence."""
    # Try redrawing path with one hit removed
    if len(path_ids) > 1:
        # Remove first added hit from path and re-predict
        # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
        second_hit_index = path_ids[1] - 1
        mask[second_hit_index] = 0
        # Redraw; try second best hit as second hit
        path2 = get_path(hit_id, thr, mask, module_id, preds=preds, fit=fit, hits=hits)

        # Check for improvement
        if len(path2) > len(path_ids):
            path_ids = path2
            # Remove first added hit from path and re-predict
            # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
            second_hit_index = path_ids[1] - 1
            mask[second_hit_index] = 0

            # Redraw again; try third best hit as second hit
            path2 = get_path(hit_id, thr, mask, module_id, preds=preds, fit=fit, hits=hits)

            # Check for improvement
            if len(path2) > len(path_ids):
                path_ids = path2

        # No improvement yet. Try redrawing path with second hit of redrawn path removed
        elif len(path2) > 1:
            # Add first hit of redrawn path back to mask
            # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
            mask[path_ids[1] - 1] = 1
            # Remove second hit of redrawn path from mask
            mask[path2[1] - 1] = 0

            # Redraw
            path2 = get_path(hit_id, thr, mask, module_id, preds=preds, fit=fit, hits=hits)

            # Check for improvement
            if len(path2) > len(path_ids):
                path_ids = path2
    return path_ids


# Checked
# Ready for tracklets
def get_all_paths(
    hits: pd.DataFrame,
    thr: float,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    do_redraw: bool = True,
    debug_limit: int | None = None,
    subject_idx: npt.NDArray | None = None,
    fit: bool = False,
    abs_tolerance_line=10,
    abs_tolerance_r=10,
    abs_tolerance_trig=10,
) -> list[npt.NDArray]:
    """Generate all paths for all hits in the event as seeds. Returns list of hit_ids per seed."""

    tracks_all = []
    N = len(preds)
    # If subjects are specified, only generate paths for those hits
    skip_mask = np.zeros(N)
    skip_mask[subject_idx] = 1  # subject_idx == None -> skip_mask = np.ones(N)

    # Generate paths for all subjects
    for index, mask in zip(tqdm(range(N), desc="Generating all paths", file=sys.stdout), skip_mask):
        # Skip hits that are not in `only_idx`
        if mask == 0:
            # This hit not accepted by mask
            hit_id = index + 1
            tracks_all.append([hit_id])
            continue

        # Limit number of paths for debugging time saving
        if debug_limit and index > debug_limit:
            continue

        # Shift hit_id -> index + 1 because hit_id starts at 1 and index starts at 0
        hit_id = index + 1

        mask = np.zeros(len(hits))
        mask[subject_idx] = 1  # subject_idx == None -> skip_mask = np.ones(N)

        path = get_path(
            hit_id,
            thr,
            mask,
            module_id,
            preds=preds,
            fit=fit,
            hits=hits,
            abs_tolerance_line=abs_tolerance_line,
            abs_tolerance_r=abs_tolerance_r,
            abs_tolerance_trig=abs_tolerance_trig,
        )

        if do_redraw:
            path = redraw(path, hit_id, thr, mask, module_id, preds, fit=fit, hits=hits)
        tracks_all.append(path)
    return tracks_all


def extend_path(
    path_ids: npt.NDArray,
    thr: float,
    mask: npt.NDArray,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    skip_same_module=True,
    last=False,
    fit=False,
    hits=None,
):
    """Extend path by adding hits with a probability above the threshold."""
    a = 0  # Cumulative probability

    # Generate sum of prediction probabilities for all hits in path except the last
    for hit_id in path_ids[:-1]:
        p = retrieve_predict(hit_id, preds)
        if last == False:
            mask = (p > thr) * mask
        # Occlude current hit
        # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
        hit_index = hit_id - 1
        mask[hit_index] = 0

        if skip_same_module:
            mask = mask_same_module(mask, path_ids, p, thr, module_id)

        a = (p + a) * mask

    path_indices = (path_ids - 1).tolist()

    # Add hits until no hits with a probability above the threshold are found
    validator = None
    while True:
        path_indices, mask, a, validator, do_break = path_add_one(**locals())
        if do_break:
            break

    return np.array(path_indices) + 1


def get_leftovers(hit_index: int, tracks_all: list[npt.NDArray], merged_tracks: npt.NDArray) -> npt.NDArray:
    """Get path from `hit_index` seed and select hits from that path that have not been assigned to a (merged) track yet."""
    path = np.array(tracks_all[hit_index])
    path_indices = path - 1  # type: ignore
    # Select seeds in `path` that have not been assigned to a (merged) track yet
    empties = np.where(merged_tracks[path_indices] == 0)[0]
    path = path[empties]
    return path


# Should be ready for tracklets
def merge_tracks(
    tracks_all: list[npt.NDArray],
    thr: int,
    ordered_by_score: npt.NDArray | None = None,
    scores: npt.NDArray | None = None,
    merged_tracks=None,
    max_track_id=0,
    do_extend=False,
    thr_extend_0: int | None = None,
    thr_extend_1: float | None = None,
    module_id=None,
    preds=None,
    subject_idx: npt.NDArray | None = None,
    fit: bool = False,
    hits: pd.DataFrame | None = None,
    verbose: bool = True,
    debug=False,
):
    # Check input variables
    if do_extend and (thr_extend_0 is None or thr_extend_1 is None or module_id is None or preds is None):
        raise ValueError("thr_extend, module_id, preds must be provided if do_extend is True")

    if ordered_by_score is None:
        assert scores, "Either scores or score_order must be provided"
        ordered_by_score = np.argsort(scores)[::-1]

    if merged_tracks is None:
        merged_tracks = np.zeros(len(ordered_by_score))

    # When debugging, start with seed index 0 for easier evaluation of path
    if debug:
        ordered_by_score = np.array([i for i in range(len(merged_tracks[subject_idx]))])  # type: ignore

    subject_mask = np.zeros(len(merged_tracks))
    subject_mask[subject_idx] = 1

    # Merge tracks by confidence
    for hit_index in tqdm(ordered_by_score, desc="Assigning track id's", file=sys.stdout):
        # Skip not selected subjects
        if subject_mask[hit_index] != 1:
            continue

        # Get path from `hit_index` seed, filtered on hits that have not been assigned to a (merged) track yet
        leftovers_ids = get_leftovers(hit_index, tracks_all, merged_tracks)

        if do_extend and len(leftovers_ids) > thr_extend_0:  # type: ignore
            mask = 1 * (merged_tracks == 0) * subject_mask
            leftovers_ids = extend_path(leftovers_ids, thr=thr_extend_1, mask=mask, module_id=module_id, preds=preds, fit=fit, hits=hits)  # type: ignore

        if subject_idx is not None:
            # Filter on subject hits
            leftovers_ids = leftovers_ids[subject_mask[leftovers_ids] == 1]

        # If leftover track is long enough, assign track id
        if len(leftovers_ids) > thr:
            # New track defined, increase highest track id
            max_track_id += 1
            path_indices = leftovers_ids - 1
            # Assign current track id to leftover hits in path
            merged_tracks[path_indices] = max_track_id

    # Print number of tracks
    if verbose:
        get_logger().info(f"Number of tracks: { max_track_id}")

    return merged_tracks, max_track_id


# TODO: add comments
# Should be ready for tracklets
def extend_tracks(
    merged_tracks,
    thr,
    module_id,
    preds,
    check_modulus=False,
    last=False,
    subject_idx: npt.NDArray | None = None,
    fit: bool = False,
    hits: pd.DataFrame | None = None,
):
    subject_mask = np.zeros(len(merged_tracks))
    subject_mask[subject_idx] = 1
    # Go over all previously assigned tracks
    for track_id in tqdm(range(1, int(merged_tracks.max()) + 1), "Extending tracks", file=sys.stdout):
        # Select hits that belong to current track id
        # Add 1 because track_id starts at 1 and index starts at 0
        path_ids = np.where(merged_tracks == track_id)[0] + 1

        if len(path_ids) == 0:
            get_logger().info(f"Track {track_id} has no hits")
            continue

        if check_modulus and len(path_ids) % 2 != 0:
            continue

        mask = 1 * (merged_tracks == 0) * subject_mask
        path_ids = extend_path(
            path_ids=path_ids, thr=thr, mask=mask, module_id=module_id, preds=preds, last=last, fit=fit, hits=hits
        )
        path_indices = path_ids - 1
        merged_tracks[path_indices] = track_id
    return merged_tracks


# TODO: add comments
def run_merging(
    tracks_all: list[npt.NDArray],
    scores: npt.NDArray,
    preds: list[npt.NDArray],
    multi_stage=True,
    module_id: npt.NDArray | None = None,
    log_evaluations=True,
    truth: pd.DataFrame | None = None,
    subject_idx: npt.NDArray | None = None,
    fit=False,
    hits: pd.DataFrame | None = None,
    stages=[
        {"thr": 6},  # 0: merge
        {"thr": 0.6},  # 1: extend
        {"thr": 3, "thr_extend_0": 3, "thr_extend_1": 0.6},  # 2: merge + extend
        {"thr": 0.5},  # 3: extend
        {"thr": 2, "thr_extend_0": 1, "thr_extend_1": 0.5},  # 4: merge + extend
        {"thr": 0.5},  # 5: extend
    ],
):
    # merge tracks by confidence and get score
    if log_evaluations and truth is None:
        raise ValueError("`truth` must be provided if `log_evaluations` is True")

    # Order hits by score
    ordered_by_score = np.argsort(scores)[::-1]

    if not multi_stage:
        merged_tracks, _ = merge_tracks(
            thr=3, tracks_all=tracks_all, ordered_by_score=ordered_by_score, subject_idx=subject_idx, fit=fit, hits=hits
        )
        evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore
        return merged_tracks

    # multistage
    # Stage 0: merge
    max_track_id = 0
    merged_tracks, max_track_id = merge_tracks(
        **stages[0],
        tracks_all=tracks_all,
        ordered_by_score=ordered_by_score,
        max_track_id=max_track_id,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    # Stage 1: extend
    merged_tracks = extend_tracks(
        **stages[1],
        merged_tracks=merged_tracks,
        module_id=module_id,
        preds=preds,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    # Stage 2: merge + extend
    merged_tracks, max_track_id = merge_tracks(
        **stages[2],
        tracks_all=tracks_all,
        ordered_by_score=ordered_by_score,
        merged_tracks=merged_tracks,
        max_track_id=max_track_id,
        do_extend=True,
        module_id=module_id,
        preds=preds,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    # Stage 3: extend
    merged_tracks = extend_tracks(
        **stages[3],
        merged_tracks=merged_tracks,
        module_id=module_id,
        preds=preds,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    # Stage 4: merge + extend
    merged_tracks, max_track_id = merge_tracks(
        **stages[4],
        tracks_all=tracks_all,
        ordered_by_score=ordered_by_score,
        merged_tracks=merged_tracks,
        max_track_id=max_track_id,
        do_extend=True,
        module_id=module_id,
        preds=preds,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    # Stage 5: extend
    merged_tracks = extend_tracks(
        **stages[5],
        merged_tracks=merged_tracks,
        module_id=module_id,
        preds=preds,
        check_modulus=True,
        last=True,
        subject_idx=subject_idx,
        fit=fit,
        hits=hits,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    return merged_tracks
