import numpy as np
import numpy.typing as npt
import pandas as pd

from data_exploration.helpers import pickle_cache
from data_exploration.event import Event
from dirs import DATA_SAMPLE


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
def get_featured_event(event_name: str | None = None, event: Event | None = None) -> Event:
    if event is None:
        event = Event(DATA_SAMPLE, event_name, feature_generator=get_features)
    else:
        event.feature_generator = get_features

    # Call features, so that they are cached
    f_cache = event.features
    return event


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
