import numpy as np

from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.event import Event


#####################
# PRE STEP DYNAMICS #
#####################
# Called prior to step
class PreStepDynamics(BaseDynamics):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._pre_step_entities = {name: entity for (name, entity) in self._world_state.entities.items()
                                   if entity.has_pre_step}
        self._pre_step_entity_groups = {name: entity_group for (name, entity_group) in self._world_state.entity_groups.items()
                                        if entity_group.has_pre_step}

    def __call__(self):
        self._pre_step_world()
        self._apply_entities_pre_step()
        self._apply_entity_groups_pre_step()
        self._process_player_action_event()

    def _pre_step_world(self):
        self._world_state.movable_entities = {name: entity for name, entity in self._world_state.entities.items()
                                              if entity.is_movable}
        self._world_state.collidable_entities = {name: entity for name, entity in self._world_state.entities.items()
                                                 if entity.is_collidable}
        self._world_state.actable_entities = {name: entity for name, entity in self._world_state.entities.items()
                                              if entity.is_actable}

        self._world_state.temp_data = {}
        self._world_state.landmark_just_found = False

        # Events
        self._world_state.recent_events = []
        self._world_state.recent_action_events = []
        self._world_state.recent_player_action_events = []
        self._world_state.recent_outcome_events = []
        self._world_state.recent_causal_events = []
        self._world_state.recent_proc_causal_events = []

    def _apply_entities_pre_step(self):
        for entity in self._pre_step_entities.values():
            entity.pre_step()

    def _apply_entity_groups_pre_step(self):
        for entity_group in self._pre_step_entity_groups.values():
            entity_group.pre_step()

    def _process_player_action_event(self):
        ws = self._world_state
        player = ws.player

        event = Event(
            timestep=ws.timestep, initiated_entities=None, affected_entities=[player],
            prev_events=None, next_events=None,
            is_action=True, action_descr=None, action=player.action, sub_action=None,
            is_outcome=False, outcome_descr=None,
            is_causal=False,
            state_change_affect_entities={
                player.name : {
                    "action": (None, player.action),
                    "action_taken": (None, player.action),
                }
            }
        )
        ws.add_event(event, event_type="action", is_player=True)
        player.events["action"] = [event]
        player.events["action_taken"] = [event]

#####################

######################
# POST STEP DYNAMICS #
######################
# Called after step
class PostStepDynamics(BaseDynamics):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._post_step_entities = {name: entity for (name, entity) in self._world_state.entities.items() if entity.has_post_step}
        self._post_step_entity_groups = {name: entity_group for (name, entity_group) in
                                        self._world_state.entity_groups.items()
                                        if entity_group.has_post_step}


    # Reset relevant fields at end of each step
    def _reset_entities(self):
        for entity in self._world_state.entities.values():
            entity.next_pos = None
            entity.action = None

    def _handle_moved(self):
        world_state = self._world_state
        pos_to_entities = world_state.pos_to_entities
        # world_state.movable_entities - movable as per start of step
        moved_entities = [entity.name for entity in world_state.movable_entities.values()
                          if entity.just_moved]

        for name in moved_entities: # world_state.moved_entities:
            entity = world_state.entities[name]
            pos_to_entities[entity.pre_step_pos].remove(name)
            pos_to_entities[entity.pos].add(name)

        # world_state.moved_entities = set()

    def _apply_entities_post_step(self):
        for entity in self._post_step_entities.values():
            entity.post_step()

    def _apply_entity_groups_post_step(self):
        for entity_group in self._post_step_entity_groups.values():
            entity_group.post_step()

    def _check_world_terminated(self):
        terminated = [entity.is_terminated for entity in self._world_state.terminatable_entities.values()]
        if len(terminated) > 0:
            self._world_state.is_terminated = np.all(terminated)
        else:
            self._world_state.is_terminated = False
        pass

    def _check_world_truncated(self):
        if self._world_state.timestep > self._world_state.max_timesteps:
            self._world_state.is_truncated = True
        else:
            self._world_state.is_truncated = False

    def _process_harmed(self):
        pass

    def __call__(self):
        self._handle_moved()
        self._reset_entities()
        self._apply_entities_post_step()
        self._apply_entity_groups_post_step()
        self._check_world_terminated()
        self._world_state.timestep += 1
        self._check_world_truncated()

######################