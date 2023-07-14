import numpy as np
import pandas as pd
import os
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from tqdm import tqdm
from data_exploration.helpers import pickle_cache
import zipfile

DATA_ROOT = "/data/atlas/users/lschoonh/BachelorProject/data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"
prefix = DATA_SAMPLE

SOLUTION_DIR = "/data/atlas/users/lschoonh/BachelorProject/trained_models/2nd_place/original_model/"

EVENT_NAME = "event000001001"


def get_event(event):
    # zf = zipfile.ZipFile(DATA_SAMPLE)
    hits = pd.read_csv(f"{DATA_SAMPLE}{event}-hits.csv")
    cells = pd.read_csv((f"{DATA_SAMPLE}{event}-cells.csv"))
    truth = pd.read_csv((f"{DATA_SAMPLE}{event}-truth.csv"))
    particles = pd.read_csv((f"{DATA_SAMPLE}{event}-particles.csv"))
    return hits, cells, truth, particles


def get_path(hit, mask, thr):
    path = [hit]
    a = 0
    while True:
        c = get_predict(path[-1], thr / 2)
        mask = (c > thr) * mask
        mask[path[-1]] = 0

        if 1:
            cand = np.where(c > thr)[0]
            if len(cand) > 0:
                mask[cand[np.isin(module_id[cand], module_id[path])]] = 0

        a = (c + a) * mask
        if a.max() < thr * len(path):
            break
        path.append(a.argmax())
    return path


def get_predict(hit, thr=0.5):
    Tx = np.zeros((len(truth), 10))
    Tx[:, 5:] = features
    Tx[:, :5] = np.tile(features[hit], (len(Tx), 1))
    pred = model.predict(Tx, batch_size=len(Tx))[:, 0]  # type: ignore
    # TTA
    idx = np.where(pred > thr)[0]
    Tx2 = np.zeros((len(idx), 10))
    Tx2[:, 5:] = Tx[idx, :5]
    Tx2[:, :5] = Tx[idx, 5:]
    pred1 = model.predict(Tx2, batch_size=len(idx))[:, 0]  # type: ignore
    pred[idx] = (pred[idx] + pred1) / 2
    return pred


def get_path2(hit, mask, thr):
    path = [hit]
    a = 0
    while True:
        c = get_predict2(path[-1])
        mask = (c > thr) * mask
        mask[path[-1]] = 0

        if 1:
            cand = np.where(c > thr)[0]
            if len(cand) > 0:
                mask[cand[np.isin(module_id[cand], module_id[path])]] = 0

        a = (c + a) * mask
        if a.max() < thr * len(path):
            break
        path.append(a.argmax())
    return path


def get_predict2(p):
    c = np.zeros(len(preds))
    c[preds[p, 0]] = preds[p, 1]  # type: ignore
    return c


# @pickle_cache
def get_track_scores(tracks_all, n=4, limit: int | None = None):
    scores = np.zeros(len(tracks_all))

    if limit is not None:
        track_selection = tracks_all[:limit]
    else:
        track_selection = tracks_all

    for i, path in tqdm(enumerate(track_selection), total=len(tracks_all), desc="getting tracks scores"):
        count = len(path)
        print((path))

        if count > 1:
            tp = 0
            fp = 0
            for p in path:
                tp = tp + np.sum(np.isin(tracks_all[p], path, assume_unique=True))
                fp = fp + np.sum(np.isin(tracks_all[p], path, assume_unique=True, invert=True))
            scores[i] = (tp - fp * n - count) / count / (count - 1)
        else:
            scores[i] = -np.inf
    return scores


def score_event_fast(truth, submission):
    truth = truth[["hit_id", "particle_id", "weight"]].merge(submission, how="left", on="hit_id")
    df = truth.groupby(["track_id", "particle_id"]).hit_id.count().to_frame("count_both").reset_index()
    truth = truth.merge(df, how="left", on=["track_id", "particle_id"])

    df1 = df.groupby(["particle_id"]).count_both.sum().to_frame("count_particle").reset_index()
    truth = truth.merge(df1, how="left", on="particle_id")
    df1 = df.groupby(["track_id"]).count_both.sum().to_frame("count_track").reset_index()
    truth = truth.merge(df1, how="left", on="track_id")
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    particles = truth[
        (truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)
    ].particle_id.unique()

    return score, truth[truth.particle_id.isin(particles)].weight.sum(), 1 - truth[truth.track_id > 0].weight.sum()


def evaluate_tracks(tracks, truth):
    submission = pd.DataFrame({"hit_id": truth.hit_id, "track_id": tracks})
    score = score_event_fast(truth, submission)[0]
    track_id = tracks.max()
    print(
        "%.4f %2.2f %4d %5d %.4f %.4f"
        % (
            score,
            np.sum(tracks > 0) / track_id,
            track_id,
            np.sum(tracks == 0),
            1 - score - np.sum(truth.weight.values[tracks == 0]),
            np.sum(truth.weight.values[tracks == 0]),
        )
    )


def extend_path(path, mask, thr, last=False):
    a = 0
    for p in path[:-1]:
        c = get_predict2(p)
        if last == False:
            mask = (c > thr) * mask
        mask[p] = 0
        cand = np.where(c > thr)[0]
        mask[cand[np.isin(module_id[cand], module_id[path])]] = 0
        a = (c + a) * mask

    while True:
        c = get_predict2(path[-1])
        if last == False:
            mask = (c > thr) * mask
        mask[path[-1]] = 0
        cand = np.where(c > thr)[0]
        mask[cand[np.isin(module_id[cand], module_id[path])]] = 0
        a = (c + a) * mask

        if a.max() < thr * len(path):
            break

        path.append(a.argmax())
        if last:
            break

    return path


if __name__ == "__main__":
    hits, cells, truth, particles = get_event(EVENT_NAME)
    hit_cells = cells.groupby(["hit_id"]).value.count().values
    hit_value = cells.groupby(["hit_id"]).value.sum().values
    features = np.hstack(
        (hits[["x", "y", "z"]] / 1000, hit_cells.reshape(len(hit_cells), 1) / 10, hit_value.reshape(len(hit_cells), 1))  # type: ignore
    )
    count = hits.groupby(["volume_id", "layer_id", "module_id"])["hit_id"].count().values
    module_id = np.zeros(len(hits), dtype="int32")

    for i in range(len(count)):
        si = np.sum(count[:i])
        module_id[si : si + count[i]] = i

    # Predict all pairs for reconstruct by all hits. (takes 2.5hr but can skip)
    skip_predict = True

    if skip_predict == False:
        TestX = np.zeros((len(features), 10))
        TestX[:, 5:] = features

        # for TTA
        TestX1 = np.zeros((len(features), 10))
        TestX1[:, :5] = features

        preds = []

        for i in tqdm_notebook(range(len(features) - 1)):
            TestX[i + 1 :, :5] = np.tile(features[i], (len(TestX) - i - 1, 1))

            pred = model.predict(TestX[i + 1 :], batch_size=20000)[:, 0]  # type: ignore
            idx = np.where(pred > 0.2)[0]

            if len(idx) > 0:
                TestX1[idx + i + 1, 5:] = TestX[idx + i + 1, :5]
                pred1 = model.predict(TestX1[idx + i + 1], batch_size=20000)[:, 0]  # type: ignore
                pred[idx] = (pred[idx] + pred1) / 2

            idx = np.where(pred > 0.5)[0]

            preds.append([idx + i + 1, pred[idx]])

            # if i==0: print(preds[-1])

        preds.append([np.array([], dtype="int64"), np.array([], dtype="float32")])

        # rebuild to NxN
        for i in range(len(preds)):
            ii = len(preds) - i - 1
            for j in range(len(preds[ii][0])):
                jj = preds[ii][0][j]
                preds[jj][0] = np.insert(preds[jj][0], 0, ii)
                preds[jj][1] = np.insert(preds[jj][1], 0, preds[ii][1][j])

        # np.save('my_%s.npy'%event, preds)
    else:
        print("load predicts")
        preds = np.load(SOLUTION_DIR + "my_%s.npy" % EVENT_NAME, allow_pickle=True)

    # reconstruct by all hits. (takes 0.6hr but can skip)
    skip_reconstruct = True

    if skip_reconstruct == False:
        tracks_all = []
        thr = 0.85
        x4 = True
        for hit in tqdm_notebook(range(len(preds))):
            m = np.ones(len(truth))
            path = get_path2(hit, m, thr)
            if x4 and len(path) > 1:
                m[path[1]] = 0
                path2 = get_path2(hit, m, thr)
                if len(path) < len(path2):
                    path = path2
                    m[path[1]] = 0
                    path2 = get_path2(hit, m, thr)
                    if len(path) < len(path2):
                        path = path2
                elif len(path2) > 1:
                    m[path[1]] = 1
                    m[path2[1]] = 0
                    path2 = get_path2(hit, m, thr)
                    if len(path) < len(path2):
                        path = path2
            tracks_all.append(path)
        # np.save('my_tracks_all', tracks_all)
    else:
        print("load tracks")
        tracks_all = np.load(SOLUTION_DIR + "my_tracks_all.npy", allow_pickle=True)

    # calculate track's confidence (about 2 mins)
    scores = get_track_scores(tracks_all, 8)

    # multistage
    idx = np.argsort(scores)[::-1]
    tracks = np.zeros(len(hits))
    track_id = 0

    for hit in tqdm(idx, desc="merging"):
        path = np.array(tracks_all[hit])
        path = path[np.where(tracks[path] == 0)[0]]

        if len(path) > 6:
            track_id = track_id + 1
            tracks[path] = track_id

    evaluate_tracks(tracks, truth)

    for track_id in tqdm(range(1, int(tracks.max()) + 1)):
        path = np.where(tracks == track_id)[0]
        path = extend_path(path.tolist(), 1 * (tracks == 0), 0.6)
        tracks[path] = track_id

    evaluate_tracks(tracks, truth)

    for hit in tqdm(idx):
        path = np.array(tracks_all[hit])
        path = path[np.where(tracks[path] == 0)[0]]

        if len(path) > 3:
            path = extend_path(path.tolist(), 1 * (tracks == 0), 0.6)
            track_id = track_id + 1
            tracks[path] = track_id

    evaluate_tracks(tracks, truth)

    for track_id in tqdm(range(1, int(tracks.max()) + 1)):
        path = np.where(tracks == track_id)[0]
        path = extend_path(path.tolist(), 1 * (tracks == 0), 0.5)
        tracks[path] = track_id

    evaluate_tracks(tracks, truth)

    for hit in tqdm(idx):
        path = np.array(tracks_all[hit])
        path = path[np.where(tracks[path] == 0)[0]]

        if len(path) > 1:
            path = extend_path(path.tolist(), 1 * (tracks == 0), 0.5)
        if len(path) > 2:
            track_id = track_id + 1
            tracks[path] = track_id

    evaluate_tracks(tracks, truth)

    for track_id in tqdm(range(1, int(tracks.max()) + 1)):
        path = np.where(tracks == track_id)[0]
        if len(path) % 2 == 0:
            path = extend_path(path.tolist(), 1 * (tracks == 0), 0.5, True)
            tracks[path] = track_id

    evaluate_tracks(tracks, truth)
