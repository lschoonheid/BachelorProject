from typing import Any
import random
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import pandas as pd

from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_exploration.visualize import generate_track_fig, add_track_to_fig
from classes.event import Event
from data_exploration.helpers import datetime_str, find_file, save, pickle_cache
from data_exploration.visualize import plot_prediction
from trackml.score import score_event

# print(os.listdir("../input"))
# print(os.listdir("../input/trackml/"))
# prefix='../input/trackml-particle-identification/'


DIRECTORY = "/data/atlas/users/lschoonh/BachelorProject/"
DATA_ROOT = DIRECTORY + "data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"
MODELS_ROOT = DIRECTORY + "trained_models/2nd_place/"
SOLUTION_DIR = MODELS_ROOT + "original_model/"
prefix = DATA_SAMPLE


# Hyperparameters
N_EVENTS = 10
EVENT_OFFSET = 10
EVENT_RANGE = range(EVENT_OFFSET, EVENT_OFFSET + N_EVENTS)

# First training
VALIDATION_SPLIT = 0.05
BATCH_SIZE = 8000
LOSS_FUNCTION = "binary_crossentropy"

# Learning rates
LEARNING_RATES: list[float] = [-5, -4, -5]
EPOCHS = [1, 20, 3]

# Hard negative training
LR_HARD: list[float] = [-4, -5, -6]
EPOCHS_HARD = [30, 10, 2]

TEST_THRESHOLD = 0.95


def get_event(event_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get event data."""
    # zf = zipfile.ZipFile(DATA_SAMPLE)
    hits = pd.read_csv(f"{DATA_SAMPLE}{event_name}-hits.csv")
    cells = pd.read_csv((f"{DATA_SAMPLE}{event_name}-cells.csv"))
    truth = pd.read_csv((f"{DATA_SAMPLE}{event_name}-truth.csv"))
    particles = pd.read_csv((f"{DATA_SAMPLE}{event_name}-particles.csv"))
    return hits, cells, truth, particles


def get_particle_ids(truth: pd.DataFrame) -> npt.NDArray:
    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids != 0)[0]]
    return particle_ids


def get_features(event: Event) -> npt.NDArray:
    """Extract the following features per hit:
    - x, y, z: coordinates in 3D space
    - TODO volume_id, layer_id, module_id: detector ID
    - cell count: number of cells that have fired in this hit
    - cell value sum: sum of cell values for this hit


    """
    # Take #cells hit per hit_id
    hit_cells = event.cells.groupby(["hit_id"]).value.count().values
    # Take cell value sum per hit_id
    hit_value = event.cells.groupby(["hit_id"]).value.sum().values
    # hstack hit features per hit_id
    features = np.hstack(
        (
            event.hits[["x", "y", "z"]] / 1000,
            hit_cells.reshape(len(hit_cells), 1) / 10,  # type: ignore
            hit_value.reshape(len(hit_cells), 1),  # type: ignore
        )
    )
    return features


@pickle_cache
def get_featured_event(event_name: str) -> Event:
    event = Event(DATA_SAMPLE, event_name, feature_generator=get_features)
    # Call features, so that they are cached
    f_cache = event.features
    return event


def get_train_0(size: int, n: int, truth: pd.DataFrame, features: npt.NDArray) -> npt.NDArray:
    i = np.random.randint(n, size=size)
    j = np.random.randint(n, size=size)
    p_id = truth.particle_id.values
    pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
    pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

    Train0 = np.hstack((features[pair[:, 0]], features[pair[:, 1]], np.zeros((len(pair), 1))))
    return Train0


def get_train(event_range: range = EVENT_RANGE) -> npt.NDArray:
    Train = np.array([])
    for i in tqdm(event_range):
        event_name = "event0000010%02d" % i
        event = get_featured_event(event_name)
        hits, truth = event.hits, event.truth
        features = event.features

        particle_ids = get_particle_ids(truth)

        # Take all pairs of hits that belong to the same track id (cartesian product)
        pair1 = []
        for particle_id in particle_ids:
            hit_ids = truth[truth.particle_id == particle_id].hit_id.values - 1
            for i in hit_ids:
                for j in hit_ids:
                    if i != j:
                        pair1.append([i, j])
        pair1 = np.array(pair1)
        fleft = features[pair1[:, 0]]
        fright = features[pair1[:, 1]]
        ones = np.ones((len(pair1), 1))
        # hstack hit features of pairs that belong to the same track with boolean for belonging to same track: x1,y1,z1..,x2,y2,z2..,1
        Train1 = np.hstack((fleft, fright, ones))
        # Train1 = np.hstack((features[pair1[:, 0]], features[pair1[:, 1]], np.ones((len(pair1), 1))))

        # Extend input training set with current event in loop
        if len(Train) == 0:
            Train = Train1
        else:
            Train = np.vstack((Train, Train1))

        # Take all pairs of hits that do not belong to the same track id
        # TODO: this can be done more efficiently
        n = len(hits)
        size = len(Train1) * 3
        Train0 = get_train_0(size, n, truth, features)

        print(event_name, Train1.shape)

        Train = np.vstack((Train, Train0))
    del Train0, Train1  # type: ignore

    np.random.shuffle(Train)
    print(Train.shape)  # type: ignore
    return Train


def init_model(fs: int = 10) -> Model:
    model = Sequential()
    model.add(Dense(800, activation="selu", input_shape=(fs,)))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(200, activation="selu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def do_train(
    model: Model,
    Train: npt.NDArray,
    lr: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    loss_function: str,
    callbacks: list[Any] = [],
    epochs_passed: int = 0,
):
    model.compile(loss=[loss_function], optimizer=Adam(learning_rate=10 ** (lr)), metrics=["accuracy"])
    History = model.fit(
        x=Train[:, :-1],  # type: ignore
        y=Train[:, -1],  # type: ignore
        batch_size=batch_size,
        initial_epoch=epochs_passed,
        epochs=epochs + epochs_passed,
        verbose=2,  # type: ignore
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
    )
    return History


def get_hard_negatives(model: Model, event_range: range = EVENT_RANGE):
    """Get hard negative feature set for training."""
    Train_hard = []
    for i in tqdm(event_range):
        # Load event
        event_name = "event0000010%02d" % i
        event = get_featured_event(event_name)
        hits, cells, particles, truth = event.all
        features = event.features

        # Take all pairs of hits that do not belong to the same track id
        size = 30000000
        n = len(truth)
        Train0 = get_train_0(size, n, truth, features)

        pred = model.predict(Train0[:, :-1], batch_size=20000)
        s = np.where(pred > 0.5)[0]

        print(event, len(Train0), len(s))

        if len(Train_hard) == 0:
            Train_hard = Train0[s]
        else:
            Train_hard = np.vstack((Train_hard, Train0[s]))
    del Train0  # type: ignore
    print(Train_hard.shape)  # type: ignore
    return Train_hard


def _get_log_dir() -> str:
    return DIRECTORY + "training_logs/2nd_place_example/fit/" + datetime_str()


def _get_tensorboard_callback():
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_get_log_dir(), histogram_freq=1)
    return tensorboard_callback


def run_training(
    event_range: range = EVENT_RANGE,
    learning_rates: list[float] = LEARNING_RATES,
    epochs: list[int] = EPOCHS,
    learning_rates_hard: list[float] = LR_HARD,
    epochs_hard: list[int] = EPOCHS_HARD,
    batch_size: int = BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
) -> Model:
    """Get trained model."""
    # Prepare training set
    Train = get_train(event_range)
    # Init model
    model = init_model()

    epochs_passed = 0

    # Train model

    # Prerpare tensorboard
    tensorboard_callback = _get_tensorboard_callback()

    # Train regular
    for lr, n_epoch in zip(learning_rates, epochs):
        # Execute training
        do_train(
            model,
            Train,
            lr,
            n_epoch,
            batch_size,
            validation_split,
            LOSS_FUNCTION,
            epochs_passed=epochs_passed,
            callbacks=[tensorboard_callback],
        )
        epochs_passed += n_epoch

    # Add hard negatives to training set
    Train_hard = get_hard_negatives(model, event_range)
    Train = np.vstack((Train, Train_hard))
    np.random.shuffle(Train)
    print(Train.shape)

    # Train hard
    for lr, n_epoch in zip(learning_rates_hard, epochs_hard):
        # Execute training
        do_train(
            model,
            Train,
            lr,
            n_epoch,
            batch_size,
            validation_split,
            LOSS_FUNCTION,
            epochs_passed=epochs_passed,
            callbacks=[tensorboard_callback],
        )
        epochs_passed += n_epoch

    return model


def make_predict(
    features: npt.NDArray, hits: pd.DataFrame, hit_id: int, thr=0.5, batch_size: int | None = None
) -> npt.NDArray:
    """Predict probability of each pair of hits with the last hit in the path. Generates a prediction array of length len(truth) with the probability of each hit belonging to the same track as hit_id."""
    Tx = np.zeros((len(hits), 10))
    # Set first five columns of Tx to be the features of the hit with hit_id
    # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
    hit_index = hit_id - 1
    Tx[:, :5] = np.tile(features[hit_index], (len(Tx), 1))
    # Set last five columns of Tx to be the features of all hits
    Tx[:, 5:] = features

    # Make prediction
    batch_size = batch_size or round(len(Tx) / 5)
    pred = model.predict(Tx, batch_size=batch_size)[:, 0]  # type: ignore

    # TTA (test time augmentation)
    """ TTA takes a similar concept but applies it during the testing or inference phase. Instead of making predictions on the original test samples alone, TTA generates multiple augmented versions of the test samples by applying various transformations or augmentations. The model then makes predictions on each augmented version, and the final prediction is obtained by aggregating the predictions from all the augmented samples. Common aggregation techniques include taking the average or the maximum probability across the augmented predictions. """

    # Take indices of prediction that have a prediction above the threshold
    idx = np.where(pred > thr)[0]

    # Filter Tx on predictions above threshold and swap first and last five columns
    # TTA
    Tx2 = np.zeros((len(idx), 10))
    Tx2[:, 5:] = Tx[idx, :5]
    Tx2[:, :5] = Tx[idx, 5:]

    # Predict again with swapped columns
    pred1 = model.predict(Tx2, batch_size=batch_size)[:, 0]  # type: ignore

    # Take average of predictions and swapped predictions
    pred[idx] = (pred[idx] + pred1) / 2

    return pred


# TODO: add comments
def make_predict_matrix(
    model: Model,
    features: npt.NDArray,
    thr_0=0.2,
    thr_1=0.5,
    verbosity: str = "0",
) -> list[npt.NDArray]:
    TestX = np.zeros((len(features), 10))
    TestX[:, 5:] = features

    # for TTA
    TestX1 = np.zeros((len(features), 10))
    TestX1[:, :5] = features

    preds = []

    for index in tqdm(range(len(features) - 1), desc="Generating prediction matrix"):
        TestX[index + 1 :, :5] = np.tile(features[index], (len(TestX) - index - 1, 1))

        pred = model.predict(TestX[index + 1 :], batch_size=20000, verbose=verbosity)[:, 0]
        # Filter predictions above threshold
        idx = np.where(pred > thr_0)[0]

        if len(idx) > 0:
            TestX1[idx + index + 1, 5:] = TestX[idx + index + 1, :5]
            pred1 = model.predict(TestX1[idx + index + 1], batch_size=20000, verbose=verbosity)[:, 0]
            pred[idx] = (pred[idx] + pred1) / 2

        idx = np.where(pred > thr_1)[0]

        preds.append([idx + index + 1, pred[idx]])

        # if i==0: print(preds[-1])

    preds.append([np.array([], dtype="int64"), np.array([], dtype="float32")])

    # rebuild to NxN
    for index in range(len(preds)):
        ii = len(preds) - index - 1
        for j in range(len(preds[ii][0])):
            jj = preds[ii][0][j]
            preds[jj][0] = np.insert(preds[jj][0], 0, ii)
            preds[jj][1] = np.insert(preds[jj][1], 0, preds[ii][1][j])

    return preds


# Checked
def retrieve_predict(hit_id: int, preds: list[npt.NDArray]) -> npt.NDArray:
    """Generate prediction array of length len(truth) with the probability of each hit belonging to the same track as `hit_id`, by taking the prediction from the prediction matrix `preds`."""
    p = np.zeros(len(preds))
    # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
    hit_index = hit_id - 1
    hit_preds = preds[hit_index]

    candidate_indices = hit_preds[0]
    candidate_probabilities = hit_preds[1]

    p[candidate_indices] = candidate_probabilities
    return p


# Checked
def get_module_id(hits: pd.DataFrame) -> npt.NDArray:
    """Generate list of `module_id` for each hit with hit index as index."""
    # Group by volume_id, layer_id, module_id and count number of hits
    count = hits.groupby(["volume_id", "layer_id", "module_id"])["hit_id"].count().values

    # Assign module_id with hit_id as index
    module_id = np.zeros(len(hits), dtype="int32")

    # Loop over unique (volume_id, layer_id, module_id) tuples
    # TODO no idea why this is done in such a convoluted way
    for i in range(len(count)):
        # Take sum of counts of previous tuples
        si = np.sum(count[:i])
        # Assign module_id to hit_ids
        # module_id[hit_id] = module_id
        module_id[si : si + count[i]] = i

    return module_id


# Checked
def mask_same_module(
    mask: npt.NDArray, path_ids: npt.NDArray | list[int], p: npt.NDArray, thr: float, module_id: npt.NDArray
) -> npt.NDArray:
    """Skip hits that are in the same module as any hit in the path, because the best hit is already found for this module."""
    cand_indices = np.where(p > thr)[0]  # indices of candidate hits
    path_indices = np.array(path_ids) - 1
    if len(cand_indices) > 0:
        cand_module_ids = module_id[cand_indices]  # module ids of candidate hits
        path_module_ids = module_id[path_indices]  # module ids of hits in path
        overlap = np.isin(cand_module_ids, path_module_ids)
        # Mask out hits that are in the same module as any hit in the path
        mask[cand_indices[overlap]] = 0
    return mask


def get_path(
    hit_id: int,
    thr: float,
    mask: npt.NDArray,
    module_id: npt.NDArray,
    skip_same_module: bool = True,
    preds: list[npt.NDArray] | None = None,
    features: npt.NDArray | None = None,
    hits: pd.DataFrame | None = None,
):
    """Predict set of hits that belong to the same track as hit_id.
    Returns list[hit_id].
    """
    # Verify correct input
    if preds is None:
        assert features is not None and hits is not None, "Either preds or features and truth must be provided"

    # Convert to index
    hit_index = hit_id - 1
    path_indices = [hit_index]
    a = 0
    while True:
        # Predict probability of each pair of hits with the last hit in the path
        hit_id_last = path_indices[-1] + 1
        if preds is not None:
            p = retrieve_predict(hit_id_last, preds)
        else:
            if features is None or hits is None:
                raise ValueError("Either preds or features and truth must be provided")

            p = make_predict(features, hits, hit_id_last, thr / 2)

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
        # a += p
        # a *= mask

        # a[0] at step 3 = p01 + p02
        # a[1] at step 3 = p01 + p12

        # a[0] at step 4 = p01 + p02 + p03
        # a[1] at step 4 = p01 + p12 + p13
        # a[2] at step 4 = p02 + p12 + p23

        # a[0] at step 5 = p01 + p02 + p03 + p04
        # a[1] at step 5 = p01 + p12 + p13 + p14
        # a[2] at step 5 = p02 + p12 + p23 + p24
        # a[3] at step 5 = p03 + p13 + p23 + p34

        # a[0] at step 6 = p01 + p02 + p03 + p04 + p05
        # a[1] at step 6 = p01 + p12 + p13 + p14 + p15
        # a[2] at step 6 = p02 + p12 + p23 + p24 + p25

        # a[n] = sum(p(n belonging to path))

        # Breaking condition: if best average probability is below threshold, end path
        if a.max() < thr * len(path_indices):
            break
        # Add index of hit with highest probability to path, proceed with this hit as the seed for the next iteration
        path_indices.append(a.argmax())  # type: ignore
    # Convert indices back to hit_ids by adding 1
    return np.array(path_indices) + 1


def redraw(
    path_ids: npt.NDArray, hit_id: int, thr: float, mask: npt.NDArray, module_id: npt.NDArray, preds: list[npt.NDArray]
):
    """Try redrawing path with one hit removed for improved confidence."""
    # Try redrawing path with one hit removed
    if len(path_ids) > 1:
        # Remove first added hit from path and re-predict
        # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
        second_hit_index = path_ids[1] - 1
        mask[second_hit_index] = 0
        # Redraw; try second best hit as second hit
        path2 = get_path(hit_id, thr, mask, module_id, preds=preds)

        # Check for improvement
        if len(path2) > len(path_ids):
            path_ids = path2
            # Remove first added hit from path and re-predict
            # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
            second_hit_index = path_ids[1] - 1
            mask[second_hit_index] = 0

            # Redraw again; try third best hit as second hit
            path2 = get_path(hit_id, thr, mask, module_id, preds=preds)

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
            path2 = get_path(hit_id, thr, mask, module_id, preds=preds)

            # Check for improvement
            if len(path2) > len(path_ids):
                path_ids = path2
    return path_ids


# Checked
def get_all_paths(
    hits: pd.DataFrame,
    thr: float,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    do_redraw: bool = True,
    debug_limit: None | int = None,
) -> list[npt.NDArray]:
    """Generate all paths for all hits in the event as seeds. Returns list of hit_ids per seed."""
    tracks_all = []
    N = len(preds)
    for index in tqdm(range(N), desc="Generating all paths"):
        # Limit number of paths for debugging time saving
        if debug_limit and index > debug_limit:
            continue

        # Shift hit_id -> index + 1 because hit_id starts at 1 and index starts at 0
        hit_id = index + 1
        mask = np.ones(len(hits))
        path = get_path(hit_id, thr, mask, module_id, preds=preds)

        if do_redraw:
            path = redraw(path, hit_id, thr, mask, module_id, preds)
        tracks_all.append(path)
    return tracks_all


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

    for seed_index, path_ids in tqdm(enumerate(track_selection), total=len(tracks_all), desc="Generating track scores"):
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
    print(
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


def extend_path(
    path_ids: npt.NDArray,
    thr: float,
    mask: npt.NDArray,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    skip_same_module=True,
    last=False,
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

    # Add hits until no hits with a probability above the threshold are found
    while True:
        p = retrieve_predict(path_ids[-1], preds)

        if last == False:
            mask = (p > thr) * mask

        # Shift hit_id -> hit_id - 1 because hit_id starts at 1 and index starts at 0
        # Occlude last added hit
        mask[path_ids[-1] - 1] = 0

        if skip_same_module:
            mask = mask_same_module(mask, path_ids, p, thr, module_id)

        a = (p + a) * mask

        if a.max() < thr * len(path_ids):
            break

        # Add hit with highest probability to path
        best_hit_id = a.argmax() + 1
        path_ids = np.append(path_ids, best_hit_id)
        if last:
            break

    return path_ids


def get_leftovers(hit_index: int, tracks_all: list[npt.NDArray], merged_tracks: npt.NDArray) -> npt.NDArray:
    """Get path from `hit_index` seed and select hits from that path that have not been assigned to a (merged) track yet."""
    path = np.array(tracks_all[hit_index])
    path_indices = path - 1
    # Select seeds in `path` that have not been assigned to a (merged) track yet
    empties = np.where(merged_tracks[path_indices] == 0)[0]
    path = path[empties]
    return path


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
        ordered_by_score = [i for i in range(len(merged_tracks))]  # type: ignore

    # Merge tracks by confidence
    for hit_index in tqdm(ordered_by_score, desc="Assigning track id's"):
        # Get path from `hit_index` seed, filtered on hits that have not been assigned to a (merged) track yet
        leftovers_ids = get_leftovers(hit_index, tracks_all, merged_tracks)

        if do_extend and len(leftovers_ids) > thr_extend_0:  # type: ignore
            leftovers_ids = extend_path(leftovers_ids, thr=thr_extend_1, mask=1 * (merged_tracks == 0), module_id=module_id, preds=preds)  # type: ignore

        # If leftover track is long enough, assign track id
        if len(leftovers_ids) > thr:
            # New track defined, increase highest track id
            max_track_id += 1
            path_indices = leftovers_ids - 1
            # Assign current track id to leftover hits in path
            merged_tracks[path_indices] = max_track_id

    # Print number of tracks
    if verbose:
        print("Number of tracks:", max_track_id)

    return merged_tracks, max_track_id


# TODO: add comments
def extend_tracks(merged_tracks, thr, module_id, preds, check_modulus=False, last=False):
    # Go over all previously assigned tracks
    for track_id in tqdm(range(1, int(merged_tracks.max()) + 1), "Extending tracks"):
        # Select hits that belong to current track id
        # Add 1 because track_id starts at 1 and index starts at 0
        path_ids = np.where(merged_tracks == track_id)[0] + 1

        if len(path_ids) == 0:
            print("Track", track_id, "has no hits")
            continue

        if check_modulus and len(path_ids) % 2 != 0:
            continue

        path_ids = extend_path(
            path_ids=path_ids, thr=thr, mask=1 * (merged_tracks == 0), module_id=module_id, preds=preds, last=last
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
    module_id: npt.NDArray = None,
    log_evaluations=True,
    truth: pd.DataFrame | None = None,
):
    # merge tracks by confidence and get score
    if log_evaluations and truth is None:
        raise ValueError("`truth` must be provided if `log_evaluations` is True")

    # Order hits by score
    ordered_by_score = np.argsort(scores)[::-1]

    if not multi_stage:
        merged_tracks, _ = merge_tracks(tracks_all=tracks_all, thr=3, ordered_by_score=ordered_by_score)
        evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore
        return merged_tracks

    # multistage
    max_track_id = 0
    merged_tracks, max_track_id = merge_tracks(
        tracks_all=tracks_all, thr=6, ordered_by_score=ordered_by_score, max_track_id=max_track_id
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    merged_tracks = extend_tracks(merged_tracks, thr=0.6, module_id=module_id, preds=preds)
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    merged_tracks, max_track_id = merge_tracks(
        tracks_all=tracks_all,
        thr=3,
        ordered_by_score=ordered_by_score,
        merged_tracks=merged_tracks,
        max_track_id=max_track_id,
        do_extend=True,
        thr_extend_0=3,
        thr_extend_1=0.6,
        module_id=module_id,
        preds=preds,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    merged_tracks = extend_tracks(merged_tracks, 0.5, module_id, preds)
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    merged_tracks, max_track_id = merge_tracks(
        tracks_all=tracks_all,
        thr=2,
        ordered_by_score=ordered_by_score,
        merged_tracks=merged_tracks,
        max_track_id=max_track_id,
        do_extend=True,
        thr_extend_0=1,
        thr_extend_1=0.5,
        module_id=module_id,
        preds=preds,
    )
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    merged_tracks = extend_tracks(merged_tracks, 0.5, module_id, preds, check_modulus=True, last=True)
    evaluate_tracks(merged_tracks, truth) if log_evaluations else None  # type: ignore

    return merged_tracks


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


if __name__ == "__main__":
    new_model = False
    do_export = True
    preload = True
    do_test: bool = False
    repeats = 20
    n_test = 1
    pick_random = False
    animate = False
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print(tf.config.list_physical_devices("GPU"))

    # Get model
    if new_model:
        model = run_training()
        if do_export:
            model.save(MODELS_ROOT + f"/new_{datetime_str()}.h5")
    else:
        model = load_model(MODELS_ROOT + "original_model/my_model.h5")

    # Load event and extract required data for prediction
    event_name: str = "event000001001"
    event = get_featured_event(event_name)
    hits = event.hits
    module_id = get_module_id(hits)

    if do_test:
        # Test model, output some visualized tracks
        show_test(event, module_id, repeats, n_test, pick_random, animate)

    # Make prediction matrix for all hits in the event
    # Look for prediction matrices already existing:
    _make_predict = lambda: save(
        make_predict_matrix(model, event.features), name="preds", tag=event_name, prefix=DIRECTORY, save=do_export
    )
    preds: list[npt.NDArray] = find_file(f"preds_{event_name}", dir=DIRECTORY, fallback_func=_make_predict, force_fallback=not preload)  # type: ignore

    # Generate tracks for each hit as seed
    thr: float = 0.85

    _make_tracks = lambda: save(
        get_all_paths(hits, thr, module_id, preds, do_redraw=True),
        name="tracks_all",
        tag=event_name,
        prefix=DIRECTORY,
        save=do_export,
    )
    tracks_all: list[npt.NDArray] = find_file(f"tracks_all_{event_name}", dir=DIRECTORY, fallback_func=_make_tracks, force_fallback=not preload)  # type: ignore

    # calculate track's confidence
    _make_scores = lambda: save(
        get_track_scores(tracks_all), name="scores", tag=event_name, prefix=DIRECTORY, save=do_export
    )
    scores: npt.NDArray = find_file(f"scores_{event_name}", dir=DIRECTORY, fallback_func=_make_scores, force_fallback=not preload)  # type: ignore

    # Merge tracks
    _make_merged_tracks = lambda: save(
        run_merging(tracks_all, scores, preds, multi_stage=True, log_evaluations=True, truth=event.truth),
        name="merged_tracks",
        tag=event_name,
        prefix=DIRECTORY,
        save=do_export,
    )  # type: ignore
    merged_tracks: npt.NDArray = find_file(
        f"merged_tracks_{event_name}", dir=DIRECTORY, fallback_func=_make_merged_tracks, force_fallback=not preload
    )  # type: ignore

    # Save submission
    _make_submission = lambda: save(
        pd.DataFrame({"hit_id": hits.hit_id, "track_id": merged_tracks}),
        name="submission",
        tag=event_name,
        prefix=DIRECTORY,
        save=do_export,
    )
    submission: pd.DataFrame = find_file(
        f"submission_{event_name}", dir=DIRECTORY, fallback_func=_make_submission, force_fallback=not preload
    )  # type: ignore

    # Evaluate submission
    score = score_event(event.truth, submission)
    print("TrackML Score:", score)
    print("Fast score: ", score_event_fast(submission, event.truth))

    # Add our track_id to truth
    combined: pd.DataFrame = event.truth[["hit_id", "particle_id", "weight", "tx", "ty", "tz"]].merge(
        submission, how="left", on="hit_id"
    )
    # Group by unique combinations of track_id (our) and particle_id (truth); count number of hits overlapping
    grouped: pd.DataFrame = (
        combined.groupby(["track_id", "particle_id"]).hit_id.count().to_frame("count_both").reset_index()
    )

    # Show some tracks
    n_start = 1000
    n_total = 10
    for i in range(n_start, n_start + n_total):
        # Show tracks
        # Tracks are already ordered by score
        track_id = i
        possible_particle_ids: pd.DataFrame = grouped[grouped["track_id"] == track_id].sort_values(
            "count_both", ascending=False
        )
        most_likely_particle_id = int(possible_particle_ids.iloc[0]["particle_id"])

        best_reconstructed_track_indices = np.where(merged_tracks == track_id)[0]
        best_reconstructed_track_ids = best_reconstructed_track_indices + 1
        seed = best_reconstructed_track_ids[0]
        reconstructed = combined[combined["track_id"] == track_id]
        truth = combined[combined["particle_id"] == most_likely_particle_id]
        print("Selected track ids: \n", best_reconstructed_track_ids)
        print(reconstructed)
        fig = plot_prediction(truth, reconstructed, most_likely_particle_id, label_type="particle_id")
        fig.suptitle(f"Track {track_id} with particle id {most_likely_particle_id}")

        fig.savefig(f"reconstructed_track_{track_id}_{event_name}.png", dpi=300)
        plt.close()
