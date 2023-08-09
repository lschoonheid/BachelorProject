from functools import cached_property
from pathlib import Path
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Index
from trackml.dataset import load_event
from .helpers import get_logger, load_event_cached
from .constants import DETECTOR_KEYS

# from python_code.constants import DETECTOR_KEYS  # type: ignore


class Event:
    """Load a single event and return some info

    Args:
        dir (str | None, optional): _description_. Defaults to None.
        event_name (str | None, optional): _description_. Defaults to None.
        prefix (str | None, optional): _description_. Defaults to None.
        _cached (bool, optional): _description_. Defaults to True.
    """

    def __init__(
        self,
        dir: str | None = None,
        event_name: str | None = None,
        prefix: str | None = None,
        loaded_event: tuple[DataFrame, DataFrame, DataFrame, DataFrame] | None = None,
        feature_generator: Callable[..., NDArray] | None = None,
        _cached=True,
    ):
        self.hits: DataFrame
        self.cells: DataFrame
        self.particles: DataFrame
        self.truth: DataFrame

        # Set event name
        self.event_name = event_name if event_name else prefix.split("/")[-1] if prefix else None
        # Set prefix
        prefix = dir + str(event_name) if dir and event_name else prefix
        assert prefix or loaded_event, "Either dir and event_name, prefix, or loaded event must be given"

        if loaded_event is None:
            # Load event
            self.all: tuple[DataFrame, DataFrame, DataFrame, DataFrame] = (
                load_event_cached(prefix) if _cached else load_event(prefix)
            )
        else:
            self.all = loaded_event
        self.hits, self.cells, self.particles, self.truth = self.all

        self.is_reduced = False

        # Keep original data
        self._hits_original, self._cells_original, self._particles_original, self._truth_original = self.all
        self._all_original = self.all

        self.feature_generator = feature_generator

    def link(self):
        """Link the tables together"""

        # TODO: complete
        #         # get the particles data
        # particles = load_event('path/to/event000000123', parts=['particles'])
        # # decode particle id into vertex id, generation, etc.
        # particles = decode_particle_id(particles)
        # # add vertex rho, phi, r
        # particles = add_position_quantities(particles, prefix='v')
        # # add momentum eta, p, pt
        # particles = add_momentum_quantities(particles)
        # Add detector identifiers to true particle data

        self.truth = self.truth.merge(self.hits[["hit_id", *DETECTOR_KEYS]], on="hit_id")
        # Add particle_ids to hits
        self.hits = self.hits.merge(self.truth[["hit_id", "particle_id"]], on="hit_id")

    def get_particles(self, particle_ids: list[int]) -> DataFrame:
        """Get the particles with the given particle_ids"""
        return self.particles[self.particles["particle_id"].isin(particle_ids)]

    def set_name(self, name: str):
        """Set the name of the event on the DataFrames"""
        for table in self.all:
            table.insert(2, "event", name)

    # def filter(self, filter_func: Callable[[DataFrame], Index]):
    #     """Filter the event with the given filter function"""
    #     for table in self.all:
    #         table = table[filter_func(table)]

    @cached_property
    def n_particles(self):
        """Return the number of particles in the event"""
        return len(self.particles)

    @cached_property
    def features(self):
        """Return the features of the event"""
        assert self.feature_generator, "No feature generator specified"
        return self.feature_generator(self)

    def reduce(self, fraction: None | float = 1, random=False, seed=None):
        """Reduce the event to a subset of the data."""
        if fraction == 1 or self.is_reduced or fraction is None:
            return self

        if seed is not None:
            np.random.seed(seed)

        # Make a copy of the original data
        self._hits_reduced = self.hits.copy(True)
        self._cells_reduced = self.cells.copy(True)
        self._particles_reduced = self.particles.copy(True)
        self._truth_reduced = self.truth.copy(True)

        # Select `fraction` of particle id's
        particle_ids = self._particles_reduced["particle_id"].unique()
        p_selection_size = round(fraction * len(particle_ids))
        # Particle selection never includes noise particle
        if random:
            particle_selection = set(np.random.choice(particle_ids, p_selection_size))
        else:
            particle_selection = set(particle_ids[:p_selection_size])

        # Remove all particles not in selection
        self._particles_reduced = self._particles_reduced[
            self._particles_reduced["particle_id"].isin(particle_selection)
        ]
        # Select all hits and cells of selected particles (and noise particle)
        hit_id_selection = set(
            self._truth_reduced[self._truth_reduced["particle_id"].isin(particle_selection)]["hit_id"].unique()
        )
        noise_hits = set(self._truth_reduced[self._truth_reduced["particle_id"] == 0]["hit_id"].unique())
        n_selection_size = round(fraction * len(noise_hits))

        # Select `fraction` of discarded hits
        if random:
            noise_hits_selection = set(np.random.choice(list(noise_hits), n_selection_size))
        else:
            # Since `noise_hits` is a set, the selection is semi random but deterministic
            noise_hits_selection = set(list(noise_hits)[:n_selection_size])

        hit_id_selection = hit_id_selection.union(noise_hits_selection)

        # Apply selection
        self._truth_reduced = self._truth_reduced[self._truth_reduced["hit_id"].isin(hit_id_selection)]
        self._hits_reduced = self._hits_reduced[self._hits_reduced["hit_id"].isin(hit_id_selection)]
        self._cells_reduced = self._cells_reduced[self._cells_reduced["hit_id"].isin(hit_id_selection)]

        # Update weights to sum to original weight
        weight_ratio = self._truth_original["weight"].sum() / self._truth_reduced["weight"].sum()
        self._truth_reduced["weight"] *= weight_ratio

        # Update all
        self._all_reduced = self._hits_reduced, self._cells_reduced, self._particles_reduced, self._truth_reduced
        self.all = self._all_reduced
        self.hits, self.cells, self.particles, self.truth = self.all

        self.is_reduced = True

        get_logger().debug(f"Reduced event {self.event_name} to {fraction * 100}%")
        get_logger().debug(
            f"Reduced particles to {len(self._particles_reduced)/len(self._particles_original)*100:.4f}%"
        )
        get_logger().debug(f"Reduced hits to {len(self._hits_reduced)/len(self._hits_original)*100:.4f}%")
        get_logger().debug(f"Reduced hells to {len(self._cells_reduced)/len(self._cells_original)*100:.4f}%")
        get_logger().debug(f"Reduced truth to {len(self._truth_reduced)/len(self._truth_original)*100:.4f}%")
        return self

    @cached_property
    def expand(self):
        self.all = self._all_original
        self.hits, self.cells, self.particles, self.truth = self.all
        self.is_reduced = False
        return None

    def __str__(self):
        """Return a string representation of the event"""
        return f"Event {self.event_name}" + (f" (reduced)" if self.is_reduced else "")
