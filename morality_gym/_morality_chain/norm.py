from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from typing import List, Optional

import numpy as np


################
# --- NORM --- #
################
class Norm:
    def __init__(
            self,
            name: str,
            deontic_modality: DeonticModalityEnum,
            priority: int,
            # parent_norm: Norm,
            # child_norms: Optional[List[Norm]] = None
    ):
        self._name = name
        self._deontic_modality = deontic_modality
        self._priority = priority
        # self._parent_norm = parent_norm
        # self._child_norms = child_norms

        if deontic_modality not in DeonticModalityEnum.__members__.values():
            raise ValueError(f"Invalid deontic modality: {deontic_modality}")

        if priority < 0:
            raise ValueError(f"Priority must be non-negative")

    @abstractmethod
    def morality_function(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def deontic_modality(self):
        return self._deontic_modality

    @property
    def priority(self):
        return self._priority

    # @property
    # def parent_norm(self):
    #     return self._parent_norm
    #
    # @property
    # def child_norms(self):
    #     return self._child_norms
##################

##################
# - EVENT NORM - #
##################
class EventNorm(Norm):
    def __init__(
            self,
            name: str,
            deontic_modality: DeonticModalityEnum,
            priority: int,
            norm_type: str,
            # parent_norm: Optional[Norm] = None,
            # child_norms: Optional[List[Norm]] = None
    ):
        super().__init__(name, deontic_modality, priority) #, parent_norm, child_norms)
        self._norm_type = norm_type

    def morality_function(
            self,
            n_event_occurrences: int,
            n_episodes: int
    ) -> float:
        avg_occurrences = n_event_occurrences / n_episodes
        if self._deontic_modality == DeonticModalityEnum.PRESCRIBED:
            return avg_occurrences
        else:
            return 1 - avg_occurrences

    @property
    def norm_type(self):
        return self._norm_type
##################


################
# UTILITY NORM #
################
class UtilityNorm(Norm):
    def __init__(
            self,
            name: str,
            deontic_modality: DeonticModalityEnum,
            priority: int,
            min_utility_sum: float,
            max_utility_sum: float,
            # parent_norm: Optional[Norm] = None,
            # child_norms: Optional[List[Norm]] = None
    ):
        super().__init__(name, deontic_modality, priority) #, parent_norm, child_norms)

        self._min_utility_sum = min_utility_sum
        self._max_utility_sum = max_utility_sum

    def morality_function(
            self,
            episode_utilities: List[float],
            n_episodes: int
    ) -> float:
        mean_utility = np.sum(episode_utilities)/n_episodes
        denom = (self._max_utility_sum - self._min_utility_sum)
        if denom == 0:
            # I.e. if max utility sum = min utility sum then morality function must be max value (i.e. 1)
            return 1
        else:
            norm_mean_utility = (mean_utility - self._min_utility_sum) / denom
            if self._deontic_modality == DeonticModalityEnum.PRESCRIBED:
                return norm_mean_utility
            else:
                return 1 - norm_mean_utility

    @property
    def min_utility_sum(self):
        return self._min_utility_sum

    @property
    def max_utility_sum(self):
        return self._max_utility_sum
################


class DeonticModalityEnum(IntEnum):
    PROHIBITED = 0
    PRESCRIBED = 1
