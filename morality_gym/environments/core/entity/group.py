from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Optional, Set, Dict, List, Callable, Union, Any

import numpy as np
from overrides import overrides

from morality_gym.environments.core.action import ActionEnum
from morality_gym.environments.core.custom_types import PosType
from morality_gym.environments.core.entity.base import BaseEntity


################
# ENTITY GROUP #
################
class EntityGroup(BaseEntity):
    def __init__(
            self,
            name: str,
            group: str,
            pos: PosType,
            traversability_group: int = 0,
            is_scripted: bool = False,
            **kwargs
    ):
        super().__init__(
            name=name,
            group=group,
            is_collidable=False,
            is_movable=False,
            pos=pos,
            traversability_group=traversability_group,
            is_scripted=is_scripted,
            **kwargs
        )

        self._entities: Dict[str, BaseEntity] = {}
        self._entity_groups: Dict[str, Dict[str, BaseEntity]] = {}

        self._movable_entities: Dict[str, BaseEntity] = {}
        self._collidable_entities: Dict[str, BaseEntity] = {}
        self._actable_entities: Dict[str, BaseEntity] = {}

        # self._

        # Required?
        # self._entity_scripts: Dict[str, Callable] = {}  # ?

    # # Compute actions of individual entities in EntityGroup based only on info within EntityGroup
    # def compute_actions(self):
    #     raise NotImplementedError

    @property
    def entities(self) -> Dict[str, BaseEntity]:
        return self._entities

    @property
    def entity_groups(self) -> Dict[str, Dict[str, BaseEntity]]:
        return self._entity_groups

    @property
    def movable_entities(self) -> Dict[str, BaseEntity]:
        return self._movable_entities

    @property
    def collidable_entities(self) -> Dict[str, BaseEntity]:
        return self._collidable_entities

    @property
    def actable_entities(self) -> Dict[str, BaseEntity]:
        return self._actable_entities

    # noinspection PyMethodOverriding
    def set_state(
            self
    ):
        # TODO: Fix stuff with different args
        raise NotImplementedError

    def reset_to_init(
            self,
            excludes: Optional[Set[str]] = None
    ):
        raise NotImplementedError

    def script(self):
        raise NotImplementedError
################