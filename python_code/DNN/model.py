import argparse
import os
import sys
from typing import Any
import numpy as np
import numpy.typing as npt
import pandas as pd

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm


import numpy.typing as npt
import pandas as pd
from dirs import MODELS_ROOT, LOG_DIR, OUTPUT_DIR
from data_exploration.helpers import datetime_str, get_logger, pickle_cache
from features import get_particle_ids, get_featured_event

# Hyperparameters
N_EVENTS = 100
EVENT_OFFSET = 0
EVENT_RANGE = range(EVENT_OFFSET, EVENT_OFFSET + N_EVENTS)

# First training
VALIDATION_SPLIT = 0.05
# BATCH_SIZE = 8000
BATCH_SIZE = 8000
LOSS_FUNCTION = "binary_crossentropy"

# Learning rates
# LEARNING_RATES: list[float] = [-5, -4, -5]
LEARNING_RATES: list[float] = [-3, -3, -4, -4, -5, -5]
# EPOCHS = [1, 20, 3]
# EPOCHS = [5, 50,50, 30]
EPOCHS = [5, 5, 50, 50, 15, 15]

# Hard negative training
# LR_HARD: list[float] = [-4, -5, -6]
LR_HARD: list[float] = [-4, -4, -5, -5, -6, -6]
# EPOCHS_HARD = [30, 10, 2]
EPOCHS_HARD = [75, 75, 25, 25, 10, 10]


def get_train_0(size: int, n: int, truth: pd.DataFrame, features: npt.NDArray) -> npt.NDArray:
    i = np.random.randint(n, size=size)
    j = np.random.randint(n, size=size)
    p_id = truth.particle_id.values
    pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
    pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

    Train0 = np.hstack((features[pair[:, 0]], features[pair[:, 1]], np.zeros((len(pair), 1))))
    return Train0


@pickle_cache
def get_train(event_range: range = EVENT_RANGE) -> npt.NDArray:
    """Get training feature set."""
    Train = np.array([])
    for i in tqdm(event_range, desc="Loading events", file=sys.stdout):
        # TODO improve name
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

        Train = np.vstack((Train, Train0))
        get_logger().debug(f"Added `{event_name}` with shape `{Train1.shape}` to feature stack")
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


@pickle_cache
def get_hard_negatives(model: Model, event_range: range = EVENT_RANGE):
    """Get hard negative feature set for training."""
    Train_hard = []
    for i in tqdm(event_range, file=sys.stdout):
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

        if len(Train_hard) == 0:
            Train_hard = Train0[s]
        else:
            Train_hard = np.vstack((Train_hard, Train0[s]))
        get_logger().info(f"Added `{event}` to hard negative feature stack, {len(Train0)}, {len(s)}")
    del Train0  # type: ignore
    print(Train_hard.shape)  # type: ignore
    return Train_hard


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


# TODO: rename
def _get_log_dir() -> str:
    return LOG_DIR + "tensorboard_logs/fit/" + datetime_str()


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
    event_batch_size: int = 10,
    validation_split=VALIDATION_SPLIT,
    save_loc: str | None = None,
) -> Model:
    """Get trained model."""
    get_logger().debug(f"Vars `run_training`: { locals()}")

    n_events = len(event_range)
    do_event_batch = event_batch_size < n_events
    ranges = [range(i, min(i + event_batch_size, n_events)) for i in range(0, n_events, event_batch_size)]

    # Prepare training set
    Train_batches = []
    if do_event_batch:
        for curr_range in tqdm(ranges, desc="Getting event batches", file=sys.stdout):
            Train_batches.append(get_train(curr_range))
    else:
        Train_batches = [get_train(event_range)]

    # Init model
    model = init_model()

    epochs_passed = 0

    # Train model

    # Prerpare tensorboard
    tensorboard_callback = _get_tensorboard_callback()

    # Train regular
    get_logger().info(f"Training model with {len(Train_batches)} event batches")
    for lr, n_epoch in zip(learning_rates, epochs):
        for Train in Train_batches:
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

            if save_loc is not None:
                model.save(save_loc + f"/model_epoch{epochs_passed}_{datetime_str()}.h5")
                get_logger().debug(f"Saved model to `{save_loc}/model_epoch{epochs_passed}_{datetime_str()}.h5`")
    get_logger().info(f"Trained model with {epochs_passed} epochs in {len(Train_batches)} event batches")

    Train_hard_batches = []
    for Train, curr_range in zip(Train_batches, ranges):
        # Add hard negatives to training set
        Train_hard = get_hard_negatives(model, curr_range)
        Train_new = np.vstack((Train, Train_hard))
        np.random.shuffle(Train_new)
        get_logger().debug(f"Train HN shape: {Train_new.shape}")
        Train_hard_batches.append(Train_new)

    # Train hard
    get_logger().info(f"Training HN model with {len(Train_batches)} event batches")
    for lr, n_epoch in zip(learning_rates_hard, epochs_hard):
        for Train in Train_hard_batches:
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

            if save_loc is not None:
                model.save(save_loc + f"/model_epoch_hard{epochs_passed}_{datetime_str()}.h5")
                get_logger().debug(f"Saved model to `{save_loc}/model_epoch{epochs_passed}_{datetime_str()}.h5`")

    return model


def get_model(
    preload=False,
    save=True,
    dir=OUTPUT_DIR,
    inname="original_model/my_model_h.h5",
    outname: str | None = None,
    **kwargs,
) -> Model:
    # Get model
    if not preload:
        model = run_training(save_loc=dir + "/between/", **kwargs)
        if save:
            outname = f"/new_{datetime_str()}" if outname is None else outname
            model.save(dir + outname + ".h5")
            get_logger().info(f"Saved model to `{dir} / {outname}.h5`")
    else:
        model = load_model(dir + inname)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model.")
    parser.add_argument("-nt", dest="preload", action="store_true", help="Run without retraining")
    parser.add_argument("-ns", dest="save", action="store_false", help="Do not save model")
    parser.add_argument("-d", dest="dir", type=str, default=OUTPUT_DIR, help="Directory of model")
    parser.add_argument("-out", dest="outname", type=str, default=None, help="Name of output model")
    parser.add_argument(
        "-r", dest="event_range", type=tuple[int, int], default=EVENT_RANGE, help="Range of events for training"  # type: ignore
    )
    parser.add_argument("-e", dest="event_range", type=tuple[int, int, int], default=EPOCHS, help="Epochs for training")  # type: ignore
    parser.add_argument(
        "-eh",
        dest="event_range",
        type=tuple[int, int, int],  # type: ignore
        default=EPOCHS_HARD,
        help="Epochs for hard negative training",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    if isinstance(kwargs["event_range"], tuple):
        kwargs["event_range"] = range(*kwargs["event_range"])

    get_logger(tag="train_model").debug(f"Vars: { kwargs}")

    model = get_model(**kwargs)
