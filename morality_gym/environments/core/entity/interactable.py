from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Optional, Set, Dict, List, Callable, Union, Any

import numpy as np
from overrides import overrides

from morality_gym.environments.core.action import ActionEnum, SubActionEnum
from morality_gym.environments.core.custom_types import PosType, StateChangeType
from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.event import Event


#################
# LINKED ENTITY #
#################
class LinkedEntity(BaseEntity):
    def __init__(
            self,
            # linked_entities: Dict[str, BaseEntity],
            interact_entities: Union[Dict[str, BaseEntity], List[BaseEntity]],
            *args,
            entity_kwargs_fns: Optional[Dict[str, Callable[[BaseEntity], Dict[str, Any]]]] = None,

            entity_interact_fns: Optional[Dict[str,
                                               Callable[[BaseEntity, BaseEntity, Optional[np.random.Generator], ...],
                                                         Tuple[StateChangeType, StateChangeType, bool]]]] = None,

            # # Function that returns true if interaction should be stopped - i.e. if true do not call entity_interact_fns
            # term_interact_fn: Optional[Callable[[BaseEntity], bool]] = None,

            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self.linked_entities: Dict[str, BaseEntity] = linked_entities

        # INTERACT ENTITIES #
        if isinstance(interact_entities, list):
            interact_entities = {entity.name: entity for entity in interact_entities}
        self.interact_entities: Dict[str, BaseEntity] = interact_entities

        # ENTITY KWARGS FNS #
        if entity_kwargs_fns is None:
            entity_kwargs_fns = {}
        for name in self.interact_entities:
            if name not in entity_kwargs_fns:
                entity_kwargs_fns[name] = lambda x: {}
        self.entity_kwargs_fns = entity_kwargs_fns

        # ENTITY INTERACT FNS #
        if entity_interact_fns is None:
            entity_interact_fns = {}
        for name, entity in self.interact_entities.items():
            if name not in entity_interact_fns:
                entity_interact_fns[name] = entity.interact_fn

        self.entity_interact_fns = entity_interact_fns

        # # TERM INTERACT FN #
        # if term_interact_fn is None:
        #     term_interact_fn = lambda x: False
        # self.term_interact_fn = term_interact_fn

#################

########################
# LINKED NEARBY ENTITY #
########################
class LinkedNearbyEntity(LinkedEntity):
    def __init__(
            self,
            *args,
            entity_kwargs_fn: Optional[Callable[[BaseEntity], Dict[str, Any]]] = None,

            entity_interact_fn: Optional[Callable[[BaseEntity, BaseEntity, Optional[np.random.Generator], ...],
                                                  Tuple[StateChangeType, StateChangeType, bool]]] = None,
            **kwargs
    ):

        super().__init__(
            *args,
            interact_entities=[],
            # interact_fn=_interact_fn,
            has_post_step=True,
            **kwargs
        )
        self.entity_kwargs_fn = entity_kwargs_fn
        self.entity_interact_fn = entity_interact_fn

        self.entity_kwargs_fns = None
        self.entity_interact_fns = None
        self.interact_entities = None

    def set_nearby_entities(
            self,
            entities: List[BaseEntity]
    ):
        self.interact_entities: Dict[str, BaseEntity] = {entity.name: entity for entity in entities}

        # ENTITY KWARGS FNS #
        self.entity_kwargs_fns = {}
        for name in self.interact_entities:
            if self.entity_kwargs_fn is not None:
                self.entity_kwargs_fns[name] = self.entity_kwargs_fn
            else:
                self.entity_kwargs_fns[name] = lambda x: {}

        # ENTITY INTERACT FNS #
        self.entity_interact_fns = {}
        for name, entity in self.interact_entities.items():
            if self.entity_interact_fn is not None:
                self.entity_interact_fns[name] = self.entity_interact_fn
            else:
                self.entity_interact_fns[name] = entity.interact_fn

    def post_step(self):
        self.entity_kwargs_fns = None
        self.entity_interact_fns = None
        self.interact_entities = None

########################

###############
# -- LEVER -- #
###############
class LeverEntity(LinkedEntity):
    def __init__(
            self,
            name: str,
            group: str,
            pos: PosType,
            # linked_entities: Dict[str, BaseEntity],
            interact_entities: Union[Dict[str, BaseEntity], List[BaseEntity]],
            entity_kwargs_fns: Optional[Dict[str, Callable[[BaseEntity], Dict[str, Any]]]] = None,

            # entity_interact_fns: Optional[Dict[str, Callable[[BaseEntity], Tuple[StateChangeType, StateChangeType]]]] = None,
            # interact_fn: Optional[Callable[[BaseEntity], Tuple[StateChangeType, StateChangeType]]] = None,
            # is_collidable: bool = False,
            vis_layer: int = 0,
            n_states: int = 2,  # States of lever - is 2 or 3
            is_looped: bool = False,  # If state will loop back
            **kwargs
    ):
        def to_asset_fn(entity: BaseEntity) -> str:
            return f"lever_{self.curr_state}"
        def to_rot_fn(entity: BaseEntity) -> int:
            return 0
        def to_alpha_fn(entity: BaseEntity) -> int:
            return 255

        if entity_kwargs_fns is None:
            entity_kwargs_fns = {}
        if isinstance(interact_entities, list):
            interact_names = interact_entities
        elif isinstance(interact_entities, dict):
            interact_names = list(interact_entities.keys())
        else:
            raise ValueError(f"interact_entities must be a list or dict. interact_entities={interact_entities}")
        for interact_name in interact_names:
            if interact_name not in entity_kwargs_fns:
                entity_kwargs_fns[interact_name] = lambda _entity: {"state": _entity.curr_state}

        if "interact_fn" in kwargs:
            raise ValueError("interact_fn is not overridable for LeverEntity.")

        if "term_interact_fn" in kwargs:
            raise ValueError("term_interact_fn is not overridable for LeverEntity.")

        def _interact_fn(
                initiating_entity: BaseEntity,
                affected_entity: LeverEntity,
                rng: Optional[np.random.Generator] = None,
                # **_kwargs
        ):
            prev_state = self.curr_state
            curr_state = self.curr_state + 1
            if affected_entity.is_looped:
                curr_state = curr_state % affected_entity.n_states
            else:
                curr_state = min(curr_state, affected_entity.n_states - 1)


            affected_entity.curr_state = curr_state
            state_change_init = {}
            state_change_affect = {
                "curr_state": (prev_state, curr_state),
            }

            is_term_interact = prev_state == curr_state

            return state_change_init, state_change_affect, is_term_interact

        super().__init__(

            # interact_entities,
            name=name,
            group=group,
            pos=pos,

            interact_entities=interact_entities,
            entity_kwargs_fns=entity_kwargs_fns,
            interact_fn=_interact_fn,
            # entity_interact_fns=entity_interact_fns,

            is_collidable=False,
            is_movable=False, is_actable=False,
            is_interactable=True, is_agent_interactable=True,
            interact_type=SubActionEnum.DEFAULT,
            traversability_group=0,
            to_asset_fn=to_asset_fn, to_rot_fn=to_rot_fn, to_alpha_fn=to_alpha_fn,
            vis_layer=vis_layer,
            **kwargs

        )
        if n_states not in {2, 3}:
            raise ValueError(f"n_states must be 2 or 3. n_states={n_states}")

        self.n_states = n_states
        self.curr_state = 0
        self.is_looped = is_looped

    # def _interact_default(
    #         self,
    #         val: Optional[Any] = None,
    #         interact_entity: Optional[BaseEntity] = None,
    #         prev_event: Optional[Event] = None
    # ):
    #     curr_state = self.curr_state + 1
    #     if self.is_looped:
    #         curr_state = curr_state % self.n_states
    #     else:
    #         curr_state = min(curr_state, self.n_states - 1)
    #         if self.curr_state == curr_state:
    #             # If state did not change return out of function
    #             return
    #     self.curr_state = curr_state
    #
    #     # self.curr_state = (self.curr_state + 1) % self.n_states
    #     for entity in self.linked_entities.values():
    #         if entity.is_interactable:
    #             entity.interact(val=self.curr_state, interact_entity=self, interact_type=SubActionEnum.DEFAULT)

    def set_state(
            self,
            **states
    ):
        if "curr_state" in states:
            self.curr_state = states["curr_state"]

        super().set_state(**states)

    def calc_obs(
            self,
            is_normalise: bool = False
    ) -> Dict[str, Any]:
        onehot = np.zeros(self.n_states)
        onehot[self.curr_state] = 1

        obs = {
            "curr_state": onehot
        }


        return obs
###############

################
# LIGHT ENTITY #
################
# Mostly Debugging/Visualisation Purposes
class LightEntity(BaseEntity):
    pass
################