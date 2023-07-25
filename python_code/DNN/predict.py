import os
import sys
import numpy as np
import numpy.typing as npt
import pandas as pd

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from keras.models import Model
from tqdm import tqdm

import numpy.typing as npt
import pandas as pd


def make_predict(
    model: Model, features: npt.NDArray, hits: pd.DataFrame, hit_id: int, thr=0.85, batch_size: int | None = None
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
    batch_size=20000,
    verbosity: str = "0",
    debug_limit: None | int = None,
) -> list[npt.NDArray]:
    TestX = np.zeros((len(features), 10))
    TestX[:, 5:] = features

    # for TTA
    TestX1 = np.zeros((len(features), 10))
    TestX1[:, :5] = features

    preds = []

    for index in tqdm(range(len(features) - 1), desc="Generating prediction matrix", file=sys.stdout):
        # Limit number of predicts for debugging time saving
        if debug_limit and index > debug_limit:
            continue

        TestX[index + 1 :, :5] = np.tile(features[index], (len(TestX) - index - 1, 1))

        pred = model.predict(TestX[index + 1 :], batch_size=batch_size, verbose=verbosity)[:, 0]
        # Filter predictions above threshold
        idx = np.where(pred > thr_0)[0]

        if len(idx) > 0:
            TestX1[idx + index + 1, 5:] = TestX[idx + index + 1, :5]
            pred1 = model.predict(TestX1[idx + index + 1], batch_size=batch_size, verbose=verbosity)[:, 0]
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
