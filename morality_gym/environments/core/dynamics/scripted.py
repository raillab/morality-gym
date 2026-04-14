from typing import Optional, Dict, Callable

from morality_gym.environments.core.action import ActionEnum
from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.entity.group import EntityGroup
from morality_gym.environments.core.state import WorldState


#############################
# SCRIPTED ACTIONS DYNAMICS #
#############################
class ScriptedDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState,
            entity_scripts: Optional[Dict[str, Callable[[WorldState], ActionEnum]]] = None,
            entity_group_scripts: Optional[Dict[str, Callable[[WorldState], ActionEnum]]] = None
    ):
        super().__init__(world_state)

        self.entity_scripts = entity_scripts
        self.entity_group_scripts = entity_group_scripts

        # Entities with is_scripted=True
        self.scripted_entity_groups: Dict[str, EntityGroup] = {entity_group.name: entity_group
                                                               for entity_group in self._world_state.entity_groups.values()
                                                               if entity_group.is_scripted}

        self.scripted_entities: Dict[str, BaseEntity] = {name: entity for (name, entity) in
                                                         self._world_state.entities.items() if entity.is_scripted}

    def call_scripted_entity_groups(self):
        for entity_group in self.scripted_entity_groups.values():
            entity_group.script()

    def call_scripted_entities(self):
        for entity in self.scripted_entities.values():
            entity.script()

    def __call__(self, *args, **kwargs):
        self.call_scripted_entity_groups()
        # TODO

#############################