from abc import abstractmethod
from typing import List, Callable, Dict, Optional

from morality_gym._environments.core.action import ActionEnum
from morality_gym._environments.core.custom_types import PosType
from morality_gym._environments.core.entity.base import BaseEntity
from morality_gym._environments.core.entity.group import EntityGroup
from morality_gym._environments.core.state import WorldState

#################
# BASE DYNAMICS #
#################
class BaseDynamics:
    def __init__(
            self,
            world_state: WorldState,
    ):
        self._world_state = world_state
        # self._rng = self._world_state.rng

        # if rng is None:
        #     rng = np.random.default_rng()
        # self._rng = rng


    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def get_nearby_entities(
            self,
            pos: PosType,
            is_cardinal: bool = False
    ) -> List[BaseEntity]:
        # Note: Includes entities at pos
        y, x = pos
        pos_to_entities = self._world_state.pos_to_entities
        nearby_entities = set()

        if is_cardinal:
            # UP, DOWN, LEFT, RIGHT
            positions = {
                (y-1, x), (y+1, x),
                (y, x-1), (y, x+1),
            }
            for ny, nx in positions:
                if (ny, nx) in pos_to_entities:
                    nearby_entities = nearby_entities.union(pos_to_entities[ny, nx])
        else:
            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if (ny, nx) in pos_to_entities:
                        nearby_entities = nearby_entities.union(pos_to_entities[ny, nx])

        nearby_entities = [self._world_state.entities[name] for name in nearby_entities]
        return nearby_entities

#################