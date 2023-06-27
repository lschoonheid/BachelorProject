from functools import cached_property
from pathlib import Path
from typing import Callable
from pandas import DataFrame, Index
from helpers import load_event_cached
from trackml.dataset import load_event

from python_code.constants import DETECTOR_KEYS


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
        assert prefix, "Either dir and event_name or prefix must be specified"
        # Load event
        self.all: tuple[DataFrame, ...] = load_event_cached(prefix) if _cached else load_event(prefix)
        self.hits, self.cells, self.particles, self.truth = self.all

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

    def __str__(self):
        """Return a string representation of the event"""
        return f"Event {self.event_name}"
