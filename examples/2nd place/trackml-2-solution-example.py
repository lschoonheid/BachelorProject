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

DIRECTORY = "/data/atlas/users/lschoonh/BachelorProject"
DATA_ROOT = DIRECTORY + "BachelorProject/data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"
prefix = DATA_SAMPLE
import zipfile

# Hyperparameters
N_EVENTS = 2
# Learning rates
LR_0, LR_1, LR_2 = -5, -4, -5
LEARNING_RATES = [10 ** (lr) for lr in [LR_0, LR_1, LR_2]]
BATCH_SIZE = 8000
CROSS_VALIDATION = 0.05
E_0, E_1, E_2 = 1, 20, 3
EPOCHS = [E_0, E_1, E_2]
LOSS_FUNCTION = "binary_crossentropy"


def get_event(event):
    # zf = zipfile.ZipFile(DATA_SAMPLE)
    hits = pd.read_csv(f"{DATA_SAMPLE}{event}-hits.csv")
    cells = pd.read_csv((f"{DATA_SAMPLE}{event}-cells.csv"))
    truth = pd.read_csv((f"{DATA_SAMPLE}{event}-truth.csv"))
    particles = pd.read_csv((f"{DATA_SAMPLE}{event}-particles.csv"))
    return hits, cells, truth, particles


def get_train():
    Train = []
    offset = 10

    for i in tqdm(range(offset, offset + N_EVENTS)):
        # Load event
        event = "event0000010%02d" % i
        hits, cells, truth, particles = get_event(event)

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
        # Take all valid track ids
        particle_ids = truth.particle_id.unique()
        particle_ids = particle_ids[np.where(particle_ids != 0)[0]]

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
        p_id = truth.particle_id.values
        i = np.random.randint(n, size=size)
        j = np.random.randint(n, size=size)
        pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
        pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

        Train0 = np.hstack((features[pair[:, 0]], features[pair[:, 1]], np.zeros((len(pair), 1))))

        print(event, Train1.shape)

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


def do_train(model, Train, lr, epochs, batch_size, cross_validation, loss_function, epochs_passed=0):
    model.compile(loss=[loss_function], optimizer=Adam(learning_rate=10 ** (lr)), metrics=["accuracy"])
    History = model.fit(
        x=Train[:, :-1],  # type: ignore
        y=Train[:, -1],  # type: ignore
        batch_size=batch_size,
        initial_epoch=epochs_passed,
        epochs=epochs + epochs_passed,
        verbose=2,  # type: ignore
        validation_split=cross_validation,
        shuffle=True,
    )
    return History


if __name__ == "__main__":
    # Prepare training set
    Train = get_train()
    # Init model
    model = init_model()

    epochs_passed = 0

    # Train model
    # Train regular
    log_dir = DATA_ROOT + "/training_logs/2nd_place_example/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    for lr, epochs in zip(LEARNING_RATES, EPOCHS):
        # Prerpare tensorboard
        # Execute training
        do_train(model, Train, lr, epochs, BATCH_SIZE, CROSS_VALIDATION, LOSS_FUNCTION, epochs_passed=epochs_passed)
        epochs_passed += epochs

    # for i in tqdm(range(10, 20)):
    #     event = "event0000010%02d" % i
    #     hits, cells, truth, particles = get_event(event)
    #     hit_cells = cells.groupby(["hit_id"]).value.count().values
    #     hit_value = cells.groupby(["hit_id"]).value.sum().values
    #     features = np.hstack(
    #         (
    #             hits[["x", "y", "z"]] / 1000,
    #             hit_cells.reshape(len(hit_cells), 1) / 10,
    #             hit_value.reshape(len(hit_cells), 1),
    #         )
    #     )

    #     size = 30000000
    #     n = len(truth)
    #     i = np.random.randint(n, size=size)
    #     j = np.random.randint(n, size=size)
    #     p_id = truth.particle_id.values
    #     pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
    #     pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

    #     Train0 = np.hstack((features[pair[:, 0]], features[pair[:, 1]], np.zeros((len(pair), 1))))

    #     pred = model.predict(Train0[:, :-1], batch_size=20000)
    #     s = np.where(pred > 0.5)[0]

    #     print(event, len(Train0), len(s))

    #     if len(Train_hard) == 0:
    #         Train_hard = Train0[s]
    #     else:
    #         Train_hard = np.vstack((Train_hard, Train0[s]))
    # del Train0
    # print(Train_hard.shape)

    # model.compile(loss=[LOSS_FUNCTION], optimizer=Adam(learning_rate=10 ** (lr)), metrics=["accuracy"])
    # History = model.fit(
    #     x=Train[:, :-1],
    #     y=Train[:, -1],
    #     batch_size=BATCH_SIZE,
    #     epochs=1,
    #     verbose=2,
    #     validation_split=CROSS_VALIDATION,
    #     shuffle=True,
    # )  # type: ignore

    # lr = LR_1
    # model.compile(loss=[LOSS_FUNCTION], optimizer=Adam(learning_rate=10 ** (lr)), metrics=["accuracy"])
    # History = model.fit(
    #     x=Train[:, :-1],
    #     y=Train[:, -1],
    #     batch_size=BATCH_SIZE,
    #     epochs=20,
    #     verbose=2,
    #     validation_split=CROSS_VALIDATION,
    #     shuffle=True,
    # )  # type: ignore

    # lr = LR_2
    # model.compile(loss=[LOSS_FUNCTION], optimizer=Adam(learning_rate=10 ** (lr)), metrics=["accuracy"])
    # History = model.fit(
    #     x=Train[:, :-1],
    #     y=Train[:, -1],
    #     batch_size=BATCH_SIZE,
    #     epochs=3,
    #     verbose=2,
    #     validation_split=CROSS_VALIDATION,
    #     shuffle=True,
    # )  # type: ignore
