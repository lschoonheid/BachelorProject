import sys
import numpy as np
import numpy.typing as npt  # type: ignore
import pandas as pd
from tqdm import tqdm
from produce import get_path, redraw
from data_exploration.helpers import select_r_less


# Checked
def get_all_tracklets(
    hits: pd.DataFrame,
    thr: float,
    module_id: npt.NDArray,
    preds: list[npt.NDArray],
    r_max: float = 300,
    do_redraw: bool = True,
    debug_limit: None | int = None,
) -> list[npt.NDArray]:
    """Generate all paths for all hits in the event as seeds. Returns list of hit_ids per seed."""
    seed_idx = select_r_less(hits, thr=r_max)

    tracklets_all = []
    N = len(hits)
    seed_mask = np.zeros(N)
    seed_mask[seed_idx] = 1
    for index, mask in zip(tqdm(range(N), desc="Generating all paths", file=sys.stdout), seed_mask):
        if mask == 0:
            # Not a seed hit
            hit_id = index + 1
            tracklets_all.append([hit_id])
            continue

        # Shift hit_id -> index + 1 because hit_id starts at 1 and index starts at 0
        hit_id = index + 1
        # Mask everything except inner shell
        mask = np.zeros(len(hits))
        mask[seed_idx] = 1

        path = get_path(hit_id, thr, mask, module_id, preds=preds)

        if do_redraw:
            path = redraw(path, hit_id, thr, mask, module_id, preds)
        tracklets_all.append(path)
    return tracklets_all
