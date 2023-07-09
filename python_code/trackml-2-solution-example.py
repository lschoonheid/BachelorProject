import random
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tqdm import tqdm_notebook
import datetime
import tensorflow as tf
from tqdm import tqdm

from data_exploration.visualize import generate_track_fig, add_track_to_fig
from classes.event import Event
from data_exploration.helpers import pickle_cache

# print(os.listdir("../input"))
# print(os.listdir("../input/trackml/"))
# prefix='../input/trackml-particle-identification/'


DIRECTORY = "/data/atlas/users/lschoonh/BachelorProject/"
DATA_ROOT = DIRECTORY + "data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"
MODELS_ROOT = DIRECTORY + "trained_models/2nd_place/"
prefix = DATA_SAMPLE


# Hyperparameters
N_EVENTS = 10
EVENT_OFFSET = 10
EVENT_RANGE = range(EVENT_OFFSET, EVENT_OFFSET + N_EVENTS)

# Learning rates
LR_0, LR_1, LR_2 = -5, -4, -5
LEARNING_RATES = [10 ** (lr) for lr in [LR_0, LR_1, LR_2]]
BATCH_SIZE = 8000

# First training
VALIDATION_SPLIT = 0.05
E_0, E_1, E_2 = 1, 20, 3
EPOCHS = [E_0, E_1, E_2]
LOSS_FUNCTION = "binary_crossentropy"

# Hard negative training
LR_HARD = [-4, -5, -6]
EPOCHS_HARD = 30, 10, 2

TEST_THRESHOLD = 0.95


def get_event(event_name: str):
    """Get event data."""
    # zf = zipfile.ZipFile(DATA_SAMPLE)
    hits = pd.read_csv(f"{DATA_SAMPLE}{event_name}-hits.csv")
    cells = pd.read_csv((f"{DATA_SAMPLE}{event_name}-cells.csv"))
    truth = pd.read_csv((f"{DATA_SAMPLE}{event_name}-truth.csv"))
    particles = pd.read_csv((f"{DATA_SAMPLE}{event_name}-particles.csv"))
    return hits, cells, truth, particles


def get_particle_ids(truth):
    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids != 0)[0]]
    return particle_ids


def get_features(event: Event):
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
def get_featured_event(event_name: str):
    event = Event(DATA_SAMPLE, event_name, feature_generator=get_features)
    # Call features, so that they are cached
    f_cache = event.features
    return event


def get_train_0(size, n, truth, features):
    i = np.random.randint(n, size=size)
    j = np.random.randint(n, size=size)
    p_id = truth.particle_id.values
    pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
    pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

    Train0 = np.hstack((features[pair[:, 0]], features[pair[:, 1]], np.zeros((len(pair), 1))))
    return Train0


def get_train(event_range=EVENT_RANGE, features=None):
    Train = []
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


def init_model(fs=10):
    model = Sequential()
    model.add(Dense(800, activation="selu", input_shape=(fs,)))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(400, activation="selu"))
    model.add(Dense(200, activation="selu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def do_train(model, Train, lr, epochs, batch_size, validation_split, loss_function, callbacks=[], epochs_passed=0):
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


def get_hard_negatives(Train, model, event_range=EVENT_RANGE):
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


def _datetime_str():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _get_log_dir():
    return DIRECTORY + "training_logs/2nd_place_example/fit/" + _datetime_str()


def _get_tensorboard_callback():
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_get_log_dir(), histogram_freq=1)
    return tensorboard_callback


def run_training(
    learning_rates=LEARNING_RATES,
    epochs=EPOCHS,
    learning_rates_hard=LR_HARD,
    epochs_hard=EPOCHS_HARD,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
):
    # Prepare training set
    Train = get_train()
    # Init model
    model = init_model()

    epochs_passed = 0

    # Train model

    # Prerpare tensorboard
    tensorboard_callback = _get_tensorboard_callback()

    # Train regular
    for lr, epochs in zip(learning_rates, epochs):
        # Execute training
        do_train(
            model,
            Train,
            lr,
            epochs,
            batch_size,
            validation_split,
            LOSS_FUNCTION,
            epochs_passed=epochs_passed,
            callbacks=[tensorboard_callback],
        )
        epochs_passed += epochs

    # Add hard negatives to training set
    Train_hard = get_hard_negatives(Train, model)
    Train = np.vstack((Train, Train_hard))
    np.random.shuffle(Train)
    print(Train.shape)

    # Train hard
    for lr, epochs in zip(learning_rates_hard, epochs_hard):
        # Execute training
        do_train(
            model,
            Train,
            lr,
            epochs,
            batch_size,
            validation_split,
            LOSS_FUNCTION,
            epochs_passed=epochs_passed,
            callbacks=[tensorboard_callback],
        )
        epochs_passed += epochs

    return model


def get_predict(features, truth, hit_id, thr=0.5, batch_size=None):
    # TODO why is len of truth taken and not hits? They should be equal
    Tx = np.zeros((len(truth), 10))
    # Set first five columns of Tx to be the features of the hit with hit_id
    Tx[:, :5] = np.tile(features[hit_id], (len(Tx), 1))
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
    # TODO: why is this done? Is this TTA?
    Tx2 = np.zeros((len(idx), 10))
    Tx2[:, 5:] = Tx[idx, :5]
    Tx2[:, :5] = Tx[idx, 5:]

    # Predict again with swapped columns
    pred1 = model.predict(Tx2, batch_size=batch_size)[:, 0]  # type: ignore

    # Take average of predictions and swapped predictions
    pred[idx] = (pred[idx] + pred1) / 2

    return pred


def get_path(features, module_id, hit_id, truth, mask, thr, skip_same_module=True):
    """Predict set of hits that belong to the same track as hit_id"""
    # Convert to index
    i_0 = hit_id - 1
    path = [i_0]
    a = 0
    while True:
        # Predict probability of each pair of hits with the last hit in the path
        p = get_predict(features, truth, path[-1], thr / 2)
        # Generate mask of hits that have a probability above the threshold
        mask = (p > thr) * mask
        # Mask last added hit
        mask[path[-1]] = 0

        if skip_same_module:
            # Skip hits that are in the same module as any hit in the path, because the best hit is already found for this module
            cand = np.where(p > thr)[0]  # indices of candidate hits
            if len(cand) > 0:
                cand_module_ids = module_id[cand]  # module ids of candidate hits
                path_module_ids = module_id[path]  # module ids of hits in path
                overlap = np.isin(cand_module_ids, path_module_ids)
                # Mask out hits that are in the same module as any hit in the path
                mask[cand[overlap]] = 0

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
        if a.max() < thr * len(path):
            break
        # Add index of hit with highest probability to path, proceed with this hit as the seed for the next iteration
        path.append(a.argmax())
    # Convert indices back to hit_ids by adding 1
    return np.array(path) + 1


def test(event_name="event000001001", seed_0=1, n_test=1, test_thr=TEST_THRESHOLD, verbose=True):
    """Test the model on a single event"""
    # Load event
    event = get_featured_event(event_name)
    hits, cells, particles, truth = event.all
    features = event.features

    # Group by volume_id, layer_id, module_id and count number of hits
    count = hits.groupby(["volume_id", "layer_id", "module_id"])["hit_id"].count().values
    # print(hits.groupby(["volume_id", "layer_id", "module_id"])["hit_id"].count().head(20))

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

    tracks = []

    # select one hit to construct a track
    for hit_id in range(seed_0, seed_0 + n_test):
        # Predict corresponding hits that belong to the same track
        reconstructed_ids = get_path(features, module_id, hit_id, truth, np.ones(len(truth)), test_thr)

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


def plot_prediction(truth, reconstructed, seed: int, tag: str | None = None):
    # Show tracks

    plot_targets = generate_track_fig()
    fig = plot_targets[0]
    axes = plot_targets[1:]

    # TODO plot adjacent hits, too?

    add_track_to_fig(
        truth,
        *axes,
        particle_id=f"Seed {seed} truth",
    )
    add_track_to_fig(
        reconstructed,
        *axes,
        particle_id=f"Seed {seed} reconstructed",
    )

    # Plot seed hit
    t_r = reconstructed
    s_x, s_y, s_z = t_r[t_r["hit_id"] == seed][["tx", "ty", "tz"]].values[0]
    axes[0].plot([s_x], [s_y], [s_z], marker="o", color="red", markersize=15, label="Seed", zorder=0, alpha=0.2)
    axes[1].plot([s_x], [s_y], marker="o", color="red", markersize=15, label="Seed", zorder=0, alpha=0.2)
    axes[2].plot([s_x], [s_z], marker="o", color="red", markersize=15, label="Seed", zorder=0, alpha=0.2)
    axes[3].plot([s_y], [s_z], marker="o", color="red", markersize=15, label="Seed", zorder=0, alpha=0.2)

    for ax in axes:
        ax.legend()

    tag = "" if tag is None else f"_{tag}"
    fig.savefig(f"{n_test}_generated_tracks_seed_{seed}{tag}.png", dpi=300)


if __name__ == "__main__":
    new_model = False
    export = True
    repeats = 20
    n_test = 1
    pick_random = False
    animate = True
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print(tf.config.list_physical_devices("GPU"))

    if new_model:
        model = run_training()
        if export:
            model.save(MODELS_ROOT + f"/new_{_datetime_str()}.h5")

    else:
        model = load_model(MODELS_ROOT + "original_model/my_model.h5")

    # Generate some tracks and compare with truth
    for i in range(1, 1 + repeats):
        # set seed
        n_max = 10000  # TODO get length of hits table
        # seed = random.randrange(1, n_max) if pick_random else i
        seed = 9606

        # Generate some track(s)
        generated_tracks = test(seed_0=seed, n_test=n_test)

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
                    plot_prediction(truth, reconstructed[: f + 1], seed, tag=f"f{f+1}:{frames_total}")
            else:
                plot_prediction(truth, reconstructed, seed)
