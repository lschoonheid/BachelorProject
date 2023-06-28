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

# print(os.listdir("../input"))
# print(os.listdir("../input/trackml/"))
# prefix='../input/trackml-particle-identification/'

# keras.tf.config.list_physical_devices("GPU")

DIRECTORY = "/data/atlas/users/lschoonh/BachelorProject/"
DATA_ROOT = DIRECTORY + "data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"
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


def get_event(event):
    # zf = zipfile.ZipFile(DATA_SAMPLE)
    hits = pd.read_csv(f"{DATA_SAMPLE}{event}-hits.csv")
    cells = pd.read_csv((f"{DATA_SAMPLE}{event}-cells.csv"))
    truth = pd.read_csv((f"{DATA_SAMPLE}{event}-truth.csv"))
    particles = pd.read_csv((f"{DATA_SAMPLE}{event}-particles.csv"))
    return hits, cells, truth, particles


def get_particle_ids(truth):
    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids != 0)[0]]
    return particle_ids


def get_features(hits, cells):
    """Extract the following features per hit:
    - x, y, z: coordinates in 3D space
    - TODO volume_id, layer_id, module_id: detector ID
    - cell count: number of cells that have fired in this hit
    - cell value sum: sum of cell values for this hit


    """
    # Take #cells hit per hit_id
    hit_cells = cells.groupby(["hit_id"]).value.count().values
    # Take cell value sum per hit_id
    hit_value = cells.groupby(["hit_id"]).value.sum().values
    # hstack hit features per hit_id
    features = np.hstack(
        (
            hits[["x", "y", "z"]] / 1000,
            hit_cells.reshape(len(hit_cells), 1) / 10,  # type: ignore
            hit_value.reshape(len(hit_cells), 1),  # type: ignore
        )
    )
    return features


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
        hits, cells, truth, particles = get_event(event_name)
        features = get_features(hits, cells)
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
        event = "event0000010%02d" % i
        hits, cells, truth, particles = get_event(event)
        features = get_features(hits, cells)

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


def get_predict(hit, truth, features, thr=0.5):
    Tx = np.zeros((len(truth), 10))
    Tx[:, 5:] = features
    Tx[:, :5] = np.tile(features[hit], (len(Tx), 1))
    pred = model.predict(Tx, batch_size=len(Tx))[:, 0] # type: ignore
    # TTA
    idx = np.where(pred > thr)[0]
    Tx2 = np.zeros((len(idx), 10))
    Tx2[:, 5:] = Tx[idx, :5]
    Tx2[:, :5] = Tx[idx, 5:]
    pred1 = model.predict(Tx2, batch_size=len(idx))[:, 0] # type: ignore
    pred[idx] = (pred[idx] + pred1) / 2
    return pred


def test():
    # Load event
    event = "event000001001"
    hits, cells, truth, particles = get_event(event)

    count = hits.groupby(["volume_id", "layer_id", "module_id"])["hit_id"].count().values
    module_id = np.zeros(len(hits), dtype="int32")

    for i in range(len(count)):
        si = np.sum(count[:i])
        module_id[si : si + count[i]] = i

    # Define test input
    features = get_features(hits, cells)


def _get_log_dir():
    return DIRECTORY + "training_logs/2nd_place_example/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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


if __name__ == "__main__":
    new_model = True

    if new_model:
        model = run_training()
    else:
        model = load_model("../input/trackml/my_model.h5")

    test()
