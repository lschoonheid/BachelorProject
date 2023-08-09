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
from keras import backend as kb
import tensorflow as tf
from tqdm import tqdm


import numpy.typing as npt
import pandas as pd
from dirs import MODELS_ROOT, LOG_DIR, OUTPUT_DIR
from data_exploration.helpers import datetime_str, get_logger, pickle_cache, find_filenames, retry

if __name__ == "__main__":
    from features import get_particle_ids, get_featured_event  # type: ignore
else:
    from .features import get_particle_ids, get_featured_event

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
LEARNING_RATE_EXPS: list[float] = [-3, -4, -5]
# EPOCHS = [1, 20, 3]
# EPOCHS = [5, 50,50, 30]
EPOCHS = [10, 100, 30]

# Hard negative training
# LR_HARD: list[float] = [-4, -5, -6]
LR_EXPS_HARD: list[float] = [-4, -5, -6]
# EPOCHS_HARD = [30, 10, 2]
EPOCHS_HARD = [150, 50, 20]


# TODO: rename
def _get_log_dir() -> str:
    return LOG_DIR + "tensorboard_logs/fit/" + datetime_str()


def _get_tensorboard_callback():
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_get_log_dir(), histogram_freq=1)
    return tensorboard_callback


def _get_last_model_name(dir: str = OUTPUT_DIR) -> None | str:
    """Get name of last saved model in directory `dir`."""
    candidates = find_filenames(name="model_epoch", dir=dir, extension="h5")
    if len(candidates) == 0:
        get_logger().warning(f"No model found in `{dir}` when trying to continue training.")
        return None

    recency = [os.path.getmtime(dir + name) for name in candidates]
    last_idx = np.argmax(recency)

    last_name = candidates[last_idx]

    return last_name


def _batchify(start: int, stop: int, batch_size: int):
    """Get ranges of batches."""
    ranges = [range(i, min(i + batch_size, stop)) for i in range(start, stop, batch_size)]
    return ranges


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
    lr_exp: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
    loss_function: str,
    callbacks: list[Any] = [],
    epochs_passed: int = 0,
):
    """Execute training of model."""
    # Configure TensorFlow to allow memory growth
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # Compile model with learning rate
    model.compile(loss=[loss_function], optimizer=Adam(learning_rate=10**lr_exp), metrics=["accuracy"])

    # Train model
    # Retry in case of memory error (leave some time for memory to be freed)
    History = retry(
        func=lambda: model.fit(
            x=Train[:, :-1],  # type: ignore
            y=Train[:, -1],  # type: ignore
            batch_size=batch_size,
            initial_epoch=epochs_passed,
            epochs=epochs + epochs_passed,
            verbose=2,  # type: ignore
            validation_split=validation_split,
            shuffle=True,
            callbacks=callbacks,
        ),
        fallback_func=lambda: tf.compat.v1.keras.backend.clear_session(),
        n_retries=5,
        sleep_time=300,
    )
    tf.compat.v1.keras.backend.clear_session()
    kb.clear_session()
    return History


def cycle_train(
    model: Model,
    learning_rates: list[float],
    epochs: list[int],
    Train_batches: list[npt.NDArray],
    batch_size: int,
    epochs_passed: int = 0,
    skip_epochs: int = 0,
    validation_split=VALIDATION_SPLIT,
    tensorboard_callback=None,
    save_loc: str | None = None,
    save_interval: int = 1,
    **kwargs,
) -> int:
    """Train model in cycles. Returns `int` of total number of epochs passed."""
    get_logger().info(f"Training model with {len(Train_batches)} event batches")
    # Cycle through learning rates and epochs
    for lr, n_epoch in zip(learning_rates, epochs):
        # Cycle through event batches
        for Train in tqdm(Train_batches, desc="Training event batches", file=sys.stdout):
            n_epoch_batched = _batchify(epochs_passed, epochs_passed + n_epoch, save_interval)
            # Do epochs in batches
            for n_epoch_curr in [len(epoch_range) for epoch_range in n_epoch_batched]:
                # Continue from checkpoint of last model, skip until epochs_passed matches that of last model
                if epochs_passed + n_epoch_curr <= skip_epochs:
                    epochs_passed += n_epoch_curr
                    get_logger().debug(f"Skipping until {epochs_passed} epochs")
                    continue

                # Execute training
                do_train(
                    model,
                    Train,
                    lr,
                    n_epoch_curr,
                    batch_size,
                    validation_split,
                    LOSS_FUNCTION,
                    epochs_passed=epochs_passed,
                    callbacks=[tensorboard_callback],
                )
                epochs_passed += n_epoch_curr

                if save_loc is not None:
                    model.save(save_loc + f"/model_epoch{epochs_passed}_{datetime_str()}.h5")
                    get_logger().debug(f"Saved model to `{save_loc}/model_epoch{epochs_passed}_{datetime_str()}.h5`")
    return epochs_passed


def run_training(
    event_range: range = EVENT_RANGE,
    learning_rates_pre: list[float] = LEARNING_RATE_EXPS,
    epochs_pre: list[int] = EPOCHS,
    learning_rates_hard: list[float] = LR_EXPS_HARD,
    epochs_hard: list[int] = EPOCHS_HARD,
    batch_size: int = BATCH_SIZE,
    event_batch_size: int = 50,
    validation_split=VALIDATION_SPLIT,
    save_loc: str | None = None,
    save_interval: int = 5,
    continue_train=False,
) -> Model:
    """Get fully trained model."""
    get_logger().debug(f"Vars `run_training`: { locals()}")

    skip_epochs = 0
    # # Init model
    # # Create a MirroredStrategy. For GPU training, this will use all GPUs available to the process.
    # # strategy = tf.distribute.MirroredStrategy()
    # # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    # get_logger().debug("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # # Open a strategy scope.
    # with strategy.scope():
    if continue_train:
        assert save_loc, "No save location specified"
        continue_name = _get_last_model_name(dir=save_loc)
        assert continue_name, "No model found to continue training from"
        model = load_model(save_loc + continue_name)
        # Get number of epochs passed from name
        skip_epochs = int(continue_name.split("_")[1].split(".")[0].replace("epoch", ""))
        get_logger().debug(f"Continuing training from `{continue_name}` with {skip_epochs} epochs passed")
    else:
        model = init_model()

    # Prepare training set
    n_events = len(event_range)
    do_event_batch = event_batch_size < n_events
    event_start = event_range[0]
    event_ranges = _batchify(event_start, n_events, event_batch_size)
    Train_batches_pre = []
    if do_event_batch:
        for curr_range in tqdm(event_ranges, desc="Getting event batches", file=sys.stdout):
            Train_batches_pre.append(get_train(curr_range))
    else:
        Train_batches_pre = [get_train(event_range)]

    # Prerpare tensorboard
    tensorboard_callback = _get_tensorboard_callback()

    # Train model

    # Train regular
    get_logger().info(f"Start regular training of model.")
    epochs_passed = cycle_train(
        learning_rates=learning_rates_pre,
        epochs=epochs_pre,
        Train_batches=Train_batches_pre,
        epochs_passed=0,
        **locals(),
    )  # type: ignore
    get_logger().info(f"Trained model with {epochs_passed} epochs in {len(Train_batches_pre)} event batches")

    # Prepare hard negative training set
    Train_batches_hard = []
    for Train, curr_range in zip(Train_batches_pre, event_ranges):
        # Add hard negatives to training set
        Train_hard = get_hard_negatives(model, curr_range)
        Train_new = np.vstack((Train, Train_hard))
        np.random.shuffle(Train_new)
        get_logger().debug(f"Train HN shape: {Train_new.shape}")
        Train_batches_hard.append(Train_new)

    # Train hard
    get_logger().info(f"Start hard negative training of model.")
    epochs_passed = cycle_train(
        learning_rates=learning_rates_hard,
        epochs=epochs_hard,
        Train_batches=Train_batches_hard,
        **locals(),
    )  # type: ignore
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
        model = run_training(save_loc=dir + "between/", **kwargs)
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
    parser.add_argument(
        "-continue", dest="continue_train", action="store_true", help="Continue training from last model"
    )

    args = parser.parse_args()
    kwargs = vars(args)

    if isinstance(kwargs["event_range"], tuple):
        kwargs["event_range"] = range(*kwargs["event_range"])

    get_logger(tag="train_model").debug(f"Vars: { kwargs}")

    model = get_model(**kwargs)
