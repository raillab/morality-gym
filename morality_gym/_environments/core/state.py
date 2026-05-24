from typing import List, Optional, Dict, Any, Set

import numpy as np

from morality_gym._environments.core.custom_types import PosType
from morality_gym._environments.core.entity.base import BaseEntity
from morality_gym._environments.core.entity.landmark import LandmarkEntity
from morality_gym._environments.core.entity.player import PlayerEntity
from morality_gym._environments.core.entity.group import EntityGroup
from morality_gym._environments.core.event import Event


# @dataclass
# class BaseState:
#     pos: PosType


class WorldState:
    def __init__(
            self,
            grid_width: int,
            grid_height: int,
            player: PlayerEntity,
            entities: Dict[str, BaseEntity],
            grouped_entities: Dict[str, List[BaseEntity]],
            entity_start_states: Dict[str, Dict[str, List[Any]]],
            traversability_grids: Dict[int, np.ndarray],
            entity_groups: Optional[Dict[str, EntityGroup]] = None,
            landmark: Optional[LandmarkEntity] = None,
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None,
            max_timesteps: int = np.inf
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.player = player
        self.landmark = landmark
        self.is_landmark_found = False

        self.entities = entities
        self.grouped_entities = grouped_entities
        self.entity_groups = entity_groups

        self.entity_start_states = entity_start_states

        self.traversability_grids = traversability_grids

        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng
        self.seed = seed

        self.temp_data: Dict[str, Any] = {}

        self.entity_names: List[str] = list(entities.keys())
        self.movable_entities: Dict[str, BaseEntity] = {}
        self.collidable_entities: Dict[str, BaseEntity] = {}
        self.actable_entities: Dict[str, BaseEntity] = {}
        self.terminatable_entities: Dict[str, BaseEntity] = {}

        self.timestep = 0
        self.max_timesteps = max_timesteps

        self.is_terminated = False
        self.is_truncated = False  # TODO: Implement
        self.has_reset = False

        # Hacky way to observe when landmark is found
        # TODO: Rework
        self.landmark_just_found = False

        self.trolley_start_mode = None

        # # Which entities moved this step. Entities represented as string
        # self.moved_entities: Set[str] = set()
        # Which entities terminated this step
        self.just_terminated: Set[str] = set()  # TODO: Remove and rework

        ################
        # -- EVENTS -- #
        ################
        self.events: List[Event] = []
        self.recent_events: List[Event] = []

        self.action_events: List[Event] = []
        self.recent_action_events: List[Event] = []

        self.player_action_events: List[Event] = []
        self.recent_player_action_events: List[Event] = []

        self.outcome_events: List[Event] = []
        self.recent_outcome_events: List[Event] = []

        self.causal_events: List[Event] = []
        self.recent_causal_events: List[Event] = []

        self.proc_causal_events: List[Event] = []
        self.recent_proc_causal_events: List[Event] = []
        ################

        # Entity represented as string
        self.pos_to_entities: Dict[PosType, Set[str]] = {}
        # Add buffers for computational efficiency later
        for y in range(-1, grid_height + 1):
            for x in range(-1, grid_width + 1):
                self.pos_to_entities[(y, x)] = set()

    def add_event(
            self,
            event: Event,
            event_type: str,
            is_player: bool = False,
            is_proc_causal: bool = False,
    ):
        if not is_proc_causal:
            self.events.append(event)
            self.recent_events.append(event)

        if event_type == "action":
            self.action_events.append(event)
            self.recent_action_events.append(event)

            if is_player:
                self.player_action_events.append(event)
                self.recent_player_action_events.append(event)
        elif event_type == "outcome":
            self.outcome_events.append(event)
            self.recent_outcome_events.append(event)
        elif event_type == "causal":
            if is_proc_causal:
                self.proc_causal_events.append(event)
                self.recent_proc_causal_events.append(event)
            else:
                self.causal_events.append(event)
                self.recent_causal_events.append(event)
        else:
            raise ValueError(f"event_type must be one of ['action', 'outcome', 'causal']. Got {event_type}.")

# @dataclass
# class WorldStateDC:
#     grid_width: int
#     grid_height: int
#
#     player: AgentEntity
#
#     entities: Dict[str, BaseEntity] = field(default_factory=dict)
#     grouped_entities: Dict[str, List[BaseEntity]] = field(default_factory=dict)
#
#     # TODO: Check
#     entity_start_states: Dict[str, Dict[Union[str, Tuple[str, ...]], List[Any]]] = field(default_factory=dict)
#
#     # entity_groups: List[EntityGroup] = field(default_factory=list)
#     # grouped_entity_groups: Dict[str, EntityGroup] = field(default_factory=dict)
#
#     traversability_grids: Dict[str, np.ndarray] = field(default_factory=dict)
#
#     rng: Optional[np.random.Generator] = None
#     seed: Optional[int] = None
#
#     temp_data: Dict[str, Any] = field(default_factory=dict, init=False)
#     # TODO: Add events
#
#     entity_names: List[str] = field(default_factory=list, init=False)
#     movable_entities: Dict[str, BaseEntity] = field(default_factory=dict, init=False)
#     collidable_entities: Dict[str, BaseEntity] = field(default_factory=dict, init=False)
#     actable_entities: Dict[str, AgentEntity] = field(default_factory=dict, init=False)
#
#
#     def __post_init__(self):
#         self.entity_names = list(self.entities.keys())
#         if self.rng is None:
#             self.rng = np.random.default_rng(self.seed)
#
#         # TODO: Update these values when required
#         # self.comp_movable_entities()
#         # self.comp_collidable_entities()
#         # self.comp_actable_entities()
#
#         # self.movable_entities: Dict[str, BaseEntity] = {entity.name: entity for entity in self.entities.values()
#         #                                                  if entity.is_movable}
#         # self.collidable_entities: Dict[str, BaseEntity] = {entity.name: entity for entity in self.entities.values()
#         #                                                    if entity.is_collidable}
#         # self.actable_entities: Dict[str, AgentEntity] = {entity.name: entity for entity in self.entities.values()
#         #                                                  if (isinstance(entity, AgentEntity) and entity.is_movable)}
#
#         self._validate_init()
#
#     # def comp_movable_entities(self):
#     #     self.movable_entities: Dict[str, BaseEntity] = {entity.name: entity for entity in self.entities.values()
#     #                                                      if entity.is_movable}
#     #
#     # def comp_collidable_entities(self):
#     #     self.collidable_entities: Dict[str, BaseEntity] = {entity.name: entity for entity in self.entities.values()
#     #                                                        if entity.is_collidable}
#     #
#     # def comp_actable_entities(self):
#     #     self.actable_entities: Dict[str, AgentEntity] = {entity.name: entity for entity in self.entities.values()
#     #                                                      if (isinstance(entity, AgentEntity) and entity.is_movable)}
#
#     def _validate_init(self):
#         # TODO
#         pass
#
#     # def reset(self):
#     #     # TODO: Finish
#     #     self.temp_data = {}


def main():
    ws = WorldState(10, 10)
    print(ws)


if __name__ == "__main__":
    main()