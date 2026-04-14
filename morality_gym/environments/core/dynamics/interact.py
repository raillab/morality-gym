from typing import Callable, Optional, Dict, Any, Tuple

import numpy as np

from morality_gym.environments.core.action import ActionEnum, SubActionEnum
from morality_gym.environments.core.custom_types import StateChangeType
from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.entity.interactable import LinkedEntity, LinkedNearbyEntity
from morality_gym.environments.core.event import Event
from morality_gym.environments.core.state import WorldState


def push_fn(init_entity: BaseEntity, affect_entity: BaseEntity, rng: np.random.Generator, **kwargs) \
        -> Tuple[StateChangeType, StateChangeType, bool]:
    # state_change_init, state_change_affect, is_term_interact = interact_fn(init_entity, affect_entity, self._rng,
    #                                                                                **interact_kwargs)
    orig_action_taken = affect_entity.action_taken
    is_term_interact = False

    y1, x1 = init_entity.pos
    y2, x2 = affect_entity.pos

    if y1 == y2:
        if x1 < x2:
            # Push Right
            affect_entity.action_taken = ActionEnum.RIGHT
        else:
            # Push LEFT
            affect_entity.action_taken = ActionEnum.LEFT
    elif x1 == x2:
        if y1 < y2:
            # Push DOWN
            affect_entity.action_taken = ActionEnum.DOWN
        else:
            # Push UP
            affect_entity.action_taken = ActionEnum.UP
    else:
        raise ValueError(f"Invalid positions for pusher and pushed entity: {init_entity.pos}, {affect_entity.pos}")

    state_change_init = {}
    state_change_affect = {
        "action_taken": (orig_action_taken, affect_entity.action_taken)
    }
    return state_change_init, state_change_affect, is_term_interact


class InteractDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState,
            # is_nearby_cardinal: bool = True
    ):
        super().__init__(world_state)
        # self._is_nearby_cardinal = is_nearby_cardinal

    # @staticmethod
    # def _push_entity(
    #         pusher_entity: BaseEntity,
    #         pushed_entity: BaseEntity
    # ):
    #     y1, x1 = pusher_entity.pos
    #     y2, x2 = pushed_entity.pos
    #
    #     if y1 == y2:
    #         if x1 < x2:
    #             # Push Right
    #             pushed_entity.action_taken = ActionEnum.RIGHT
    #         else:
    #             # Push LEFT
    #             pushed_entity.action_taken = ActionEnum.LEFT
    #     elif x1 == x2:
    #         if y1 < y2:
    #             # Push DOWN
    #             pushed_entity.action_taken = ActionEnum.DOWN
    #         else:
    #             # Push UP
    #             pushed_entity.action_taken = ActionEnum.UP
    #     else:
    #         raise ValueError(f"Invalid positions for pusher and pushed entity: {pusher_entity.pos}, {pushed_entity.pos}")

        # print(f"{entity.name} pushed!")
    # entity.pos = entity.next_pos

    def recursive_interact(
            self,
            init_event: Event,
            init_entity: BaseEntity,
            affect_entity: BaseEntity,
            interact_fn: Callable,
            interact_kwargs: Optional[Dict[str, Any]]
    ):
        # ????
        if interact_kwargs is None:
            interact_kwargs = {}
        state_change_init, state_change_affect, is_term_interact = interact_fn(
            init_entity, affect_entity, self._world_state.rng, **interact_kwargs)

        event = Event(
            timestep=self._world_state.timestep,
            initiated_entities=[init_entity], affected_entities=[affect_entity],
            prev_events=[init_event], next_events=None, is_causal=True,
            state_change_init_entities={
                init_entity.name: state_change_init
            },
            state_change_affect_entities={
                affect_entity.name: state_change_affect
            }
        )
        init_event.next_events.append(event)
        for state_name, (old_val, new_val) in state_change_init.items():
            if old_val != new_val:
                init_entity.events[state_name] = [event]

        for state_name, (old_val, new_val) in state_change_affect.items():
            if old_val != new_val:
                affect_entity.events[state_name] = [event]

        self._world_state.add_event(event, "causal")

        if not is_term_interact and isinstance(affect_entity, LinkedEntity):
            for next_entity in affect_entity.interact_entities.values():
                curr_interact_kwargs = affect_entity.entity_kwargs_fns[next_entity.name](affect_entity)
                curr_interact_fn = affect_entity.entity_interact_fns[next_entity.name]

                self.recursive_interact(event, affect_entity, next_entity, curr_interact_fn, curr_interact_kwargs)

    def __call__(self):
        #######
        interact_nearby_entities = list(filter(lambda _entity: isinstance(_entity, LinkedNearbyEntity),
                                               list(self._world_state.entities.values())))
        for entity in interact_nearby_entities:
            nearby = self.get_nearby_entities(entity.pos, True)
            interactable_nearby = [nearby_entity for nearby_entity
                                   in nearby if nearby_entity.is_agent_interactable]
            entity.set_nearby_entities(interactable_nearby)
        #######

        for act_entity in self._world_state.actable_entities.values():
            if act_entity.action == ActionEnum.INTERACT:
                nearby_entities = self.get_nearby_entities(act_entity.pos, True)
                interactable_nearby = [nearby_entity for nearby_entity
                                       in nearby_entities if nearby_entity.is_agent_interactable]

                ##################
                # INTERACT EVENT #
                ##################
                # Create event for curr act_entity
                if "action_taken" in act_entity.events:
                    prev_events = act_entity.events["action_taken"]
                else:
                    prev_events = None

                # action_type = None
                # event = Event(act_entity, prev_events, action=ActionEnum.INTERACT)
                ##################

                for interact_entity in interactable_nearby:
                    curr_sub_action = interact_entity.interact_type

                    #######################
                    # INTERACT SUB ACTION #
                    #       EVENT         #
                    #######################
                    event = Event(
                        timestep=self._world_state.timestep,
                        initiated_entities=[act_entity],
                        affected_entities=[act_entity],  # This event does not yet affect another entity
                        prev_events=prev_events, next_events=None,
                        action_descr=interact_entity.interact_descr,
                        is_action=True,
                        action=ActionEnum.INTERACT, sub_action=curr_sub_action,
                        is_outcome=False, outcome_descr=None, is_causal=False,
                        # state_change_init_entities={
                        #     act_entity.name: {""}
                        # }
                        # Does this need state change info?
                    )
                    self._world_state.add_event(event, "action", is_player=act_entity.is_player)
                    if prev_events is not None:
                        for prev_event in prev_events:
                            prev_event.next_events.append(event)
                    #######################
                    # sub_event = Event(host_entity=act_entity, prev_events=[event], action=ActionEnum.INTERACT,
                    #                   sub_action=interact_entity.interact_type)
                    # event.next_events.append(sub_event)
                    # def recursive_interact(curr_entity):
                    #     # ?
                    #     pass

                    if isinstance(interact_entity, LinkedEntity):
                        if curr_sub_action == SubActionEnum.DEFAULT:
                            # interact_entity.interact_fn(act_entity, interact_entity)
                            # state_change_init, state_change_affect, is_term_interact = \
                            #     interact_entity.interact_fn(act_entity, interact_entity, self._rng)
                            #
                            # sub_event = Event(
                            #     timestep=self._world_state.timestep,
                            #     initiated_entities=[act_entity], affected_entities=[interact_entity],
                            #     prev_events=[event], next_events=None, is_causal=True,
                            #     state_change_init_entities={
                            #         act_entity.name: state_change_init
                            #     },
                            #     state_change_affect_entities={
                            #         interact_entity.name: state_change_affect
                            #     }
                            # )
                            # event.next_events.append(sub_event)
                            # for state_name, (old_val, new_val) in state_change_init.items():
                            #     if old_val != new_val:
                            #         act_entity.events[state_name] = [sub_event]
                            #
                            # for state_name, (old_val, new_val) in state_change_affect.items():
                            #     if old_val != new_val:
                            #         interact_entity.events[state_name] = [sub_event]
                            #
                            # self._world_state.add_event(sub_event, "causal")

                            self.recursive_interact(init_event=event, init_entity=act_entity,
                                                    affect_entity=interact_entity,
                                                    interact_fn=interact_entity.interact_fn, interact_kwargs=None)

                        else:
                            raise ValueError(f"sub_action = {curr_sub_action} not supported for LinkedEntity")

                        # raise NotImplementedError
                    else:
                        if curr_sub_action == SubActionEnum.DEFAULT:
                            state_change_init, state_change_affect, _ = \
                                interact_entity.interact_fn(act_entity, interact_entity, self._world_state.rng)
                        else:
                            state_change_init, state_change_affect, _ = \
                                interact_entity.sub_interact_fns[curr_sub_action](
                                    act_entity, interact_entity, self._world_state.rng)

                        sub_event = Event(
                            timestep=self._world_state.timestep,
                            initiated_entities=[act_entity], affected_entities=[interact_entity],
                            prev_events=[event], next_events=None, is_causal=True,
                            state_change_init_entities={
                                act_entity.name: state_change_init
                            },
                            state_change_affect_entities={
                                interact_entity.name: state_change_affect
                            }
                        )
                        event.next_events.append(sub_event)
                        for state_name, (old_val, new_val) in state_change_init.items():
                            if old_val != new_val:
                                act_entity.events[state_name] = [sub_event]

                        for state_name, (old_val, new_val) in state_change_affect.items():
                            if old_val != new_val:
                                interact_entity.events[state_name] = [sub_event]

                        self._world_state.add_event(sub_event, "causal")
                    # interact_entity.interact(interact_entity=act_entity, interact_type=interact_entity.interact_type,
                    #                          prev_event=sub_event)
