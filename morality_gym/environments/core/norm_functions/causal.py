from typing import Callable, Optional, Dict, Set

from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.event import Event

######################
# -     CAUSAL     - #
# - NORM FUNCTIONS - #
######################
def causal_fn(
        event: Event,
        filter_fn: Callable[[Event, BaseEntity], bool],
        name_fn: Callable[[Event, BaseEntity], str]
) -> Set[str]:
    norms = set()
    # if len()

    for entity in event.affected_entities:
        if not filter_fn(event, entity):
            curr_norm = name_fn(event, entity)
            norms.add(curr_norm)
    return norms

