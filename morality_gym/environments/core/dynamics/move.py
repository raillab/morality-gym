import copy
from typing import Tuple, List

from morality_gym.environments.core.action import ActionEnum
from morality_gym.environments.core.custom_types import PosType
from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.entity.base import HarmEnum
from morality_gym.environments.core.event import Event


#################
# MOVE DYNAMICS #
#################
class MoveDynamics(BaseDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_map = {
            ActionEnum.UP: (-1, 0),
            ActionEnum.DOWN: (1, 0),
            ActionEnum.LEFT: (0, -1),
            ActionEnum.RIGHT: (0, 1),
        }

        self.move_actions = {
            ActionEnum.MOVE_TO_POS,
            ActionEnum.UP, ActionEnum.DOWN, ActionEnum.LEFT, ActionEnum.RIGHT
        }

        # self.collided_entities = {}
        self.collisions = []

    ##############
    # -- CALL -- #
    ##############
    def __call__(self):
        self._take_move_actions()
        self._handle_entity_collisions()
        self._update_positions()
        self._compute_events()
        self._compute_harms()  # Done after update positions since is_movable may change

    def _in_bounds(self, pos: PosType) -> bool:
        y, x = pos
        y_valid = 0 <= y < self._world_state.grid_height
        x_valid = 0 <= x < self._world_state.grid_width
        return y_valid and x_valid
    ##############

    def _take_move_actions(self):
        for entity in self._world_state.actable_entities.values():
            action = entity.action_taken
            if action in self.move_actions:
                if entity.is_movable:
                    # If is movable and move action taken
                    if action == ActionEnum.MOVE_TO_POS:
                        next_pos = entity.next_pos
                    else:
                        modifier = self.action_map[action]
                        next_pos = (entity.pos[0] + modifier[0], entity.pos[1] + modifier[1])

                    if self._in_bounds(next_pos):
                        trav_group = entity.traversability_group
                        is_traversable = not self._world_state.traversability_grids[trav_group][next_pos[0], next_pos[1]]
                        if not is_traversable:
                            next_pos = entity.pos
                            entity.action_taken = ActionEnum.NOOP
                    else:
                        next_pos = entity.pos
                        entity.action_taken = ActionEnum.NOOP

                    entity.next_pos = next_pos
                else:
                    entity.action_taken = ActionEnum.NOOP
                    entity.next_pos = entity.pos
            else:
                entity.next_pos = entity.pos

    def _passthrough_collision(self, entity_1, entity_2):
        if entity_1.action_taken not in self.move_actions or entity_2.action_taken not in self.move_actions:
            return False

        if entity_1.next_pos is None or entity_2.next_pos is None:
            print(f"WARNING: {entity_1.name} or {entity_2.name} has next_pos = None in _passthrough_collision. "
                  f"entity_1.next_pos = {entity_1.next_pos}, entity_2.next_pos = {entity_2.next_pos}.")
            return False

        if entity_1.pos == entity_2.next_pos and entity_1.next_pos == entity_2.pos:
            return True


    def _handle_entity_collisions(self):
        # This method only considers collidable entities
        # Assume: Cannot be intersectable = False if not collidable
        collidable_entities = self._world_state.collidable_entities

        movable_entities = {name: entity for name, entity in collidable_entities.items() if entity.is_movable}
        immovable_entities = {name: entity for name, entity in collidable_entities.items() if not entity.is_movable}

        self.collisions = []
        self.collisions_set = set()

        ####################
        # ---- STEP 1 ---- #
        ####################
        # STEP 1: Check collisions between movable & immovable entities
        for movable_entity in movable_entities.values():
            for immovable_entity in immovable_entities.values():
                # Note: next_pos will be None for immovable_entity so use curr pos
                if movable_entity.next_pos == immovable_entity.pos:
                    movable_entity.just_collided = True
                    immovable_entity.just_collided = True
                    self.collisions.append((movable_entity, immovable_entity))
                    self.collisions_set.add((movable_entity.name, immovable_entity.name))

                    if not (movable_entity.is_intersectable and immovable_entity.is_intersectable):
                        # If it is not the case that both entities are intersectable then moveable entity cannot move
                        # to occupy space
                        movable_entity.next_pos = movable_entity.pos
                        movable_entity.action_taken = ActionEnum.NOOP

        ####################
        # ---- STEP 2 ---- #
        ####################
        # STEP 2: Check collisions between movable & movable entities
        # Note: Simple implementation, but can be very inefficient and may raise error
        # TODO: Optimise
        is_collisions = True # If this loop had collisions
        curr_iter = 0
        max_iters = 10

        while curr_iter < max_iters and is_collisions:
            is_collisions = False
            for entity_1 in movable_entities.values():
                for entity_2 in movable_entities.values():
                    if entity_1 == entity_2:
                        # Do not check collision between entity and itself
                        continue

                    if ((entity_1.name, entity_2.name) in self.collisions_set
                            or (entity_2.name, entity_1.name) in self.collisions_set):
                        # Skip if has already checked collision between entities
                        continue

                    if entity_1.next_pos == entity_2.next_pos or self._passthrough_collision(entity_1, entity_2):
                        is_collisions = True
                        entity_1.just_collided = True
                        entity_2.just_collided = True
                        self.collisions.append((entity_1, entity_2))
                        self.collisions_set.add((entity_1.name, entity_2.name))

                        if not (entity_1.is_intersectable and entity_2.is_intersectable):
                            entity_1.next_pos = entity_1.pos
                            entity_1.action_taken = ActionEnum.NOOP

                            entity_2.next_pos = entity_2.pos
                            entity_2.action_taken = ActionEnum.NOOP

            curr_iter += 1

        if curr_iter >= max_iters:
            raise ValueError("Maximum number of iterations reached.")


    def _compute_harms(self):
        # TODO: Add event handling
        for (entity1, entity2) in self.collisions:
            if self._world_state.rng.random() <= entity1.prob_harm_collide:
                prev_harm = entity1.curr_harm
                entity1.set_harm(HarmEnum.MAJOR)

                ################
                # CREATE EVENT #
                ################
                if "collided" not in entity1.events:
                    raise ValueError(f"Entity {entity1.name} does not have 'collided' event.")
                prev_events = entity1.events["collided"]

                event = Event(
                    timestep=self._world_state.timestep,
                    initiated_entities=[entity1, entity2], affected_entities=[entity1],
                    prev_events=entity1.events["collided"], next_events=None,
                    is_outcome=True,
                    outcome_descr='harm',
                    state_change_init_entities={entity1.name: {"curr_harm": (prev_harm, entity1.curr_harm)}},
                    state_change_affect_entities={entity1.name: {"curr_harm": (prev_harm, entity1.curr_harm)}}
                )
                entity1.events["curr_harm"] = [event]
                for prev_event in prev_events:
                    prev_event.next_events.append(event)
                self._world_state.add_event(event, "outcome")
                ################


                # UPDATE MOVABLE & ACTABLE ENTITIES #
                if entity1.name in self._world_state.movable_entities:
                    del self._world_state.movable_entities[entity1.name]
                if entity1.name in self._world_state.actable_entities:
                    del self._world_state.actable_entities[entity1.name]


            if self._world_state.rng.random() <= entity2.prob_harm_collide:
                prev_harm = entity2.curr_harm
                entity2.set_harm(HarmEnum.MAJOR)

                ################
                # CREATE EVENT #
                ################
                if "collided" not in entity2.events:
                    raise ValueError(f"Entity {entity2.name} does not have 'collided' event.")
                prev_events = entity2.events["collided"]

                event = Event(
                    timestep=self._world_state.timestep,
                    initiated_entities=[entity2, entity2], affected_entities=[entity2],
                    prev_events=entity2.events["collided"], next_events=None,
                    is_outcome=True,
                    outcome_descr='harm',
                    state_change_init_entities={entity2.name: {"curr_harm": (prev_harm, entity2.curr_harm)}},
                    state_change_affect_entities={entity2.name: {"curr_harm": (prev_harm, entity2.curr_harm)}}
                )
                for prev_event in prev_events:
                    prev_event.next_events.append(event)
                entity2.events["curr_harm"] = [event]
                self._world_state.add_event(event, "outcome")
                ################

                if entity2.name in self._world_state.movable_entities:
                    del self._world_state.movable_entities[entity2.name]
                if entity2.name in self._world_state.actable_entities:
                    del self._world_state.actable_entities[entity2.name]


    def _update_positions(self):
        for entity in self._world_state.movable_entities.values():
            if entity.next_pos is not None:  # Hacky fix - TODO: Fix root cause
                if entity.pos != entity.next_pos:
                    entity.just_moved = True  # Hacky?

                    # CHANGE IN POS EVENT #
                    state_change = {
                        entity.name: {"pos": (entity.pos, entity.next_pos)}
                    }

                    if "action_taken" in entity.events:
                        event = Event(
                            timestep=self._world_state.timestep,
                            initiated_entities=[entity], affected_entities=[entity],
                            prev_events=entity.events["action_taken"],
                            is_causal=True,
                            state_change_init_entities=state_change, state_change_affect_entities=state_change
                        )
                        for prev_event in entity.events["action_taken"]:
                            prev_event.next_events.append(event)
                        entity.events["pos"] = [event]

                        self._world_state.add_event(event, "causal")

                entity.pos = entity.next_pos

            entity.next_pos = None

    def _compute_events(self):
        # Compute collision outcome events
        for (entity_1, entity_2) in self.collisions:
            prev_events = []
            if "pos" in entity_1.events:
                prev_events += entity_1.events["pos"]
            if "pos" in entity_2.events:
                prev_events += entity_2.events["pos"]

            event = Event(
                timestep=self._world_state.timestep,
                initiated_entities=[entity_1, entity_2],
                affected_entities=[entity_1, entity_2],
                prev_events=prev_events,
                is_outcome=True, outcome_descr='collision',
                # state_change_init_entities={
                #     entity_1.name: {"just_collided": (False, True)},
                # }
            )

            for prev_event in prev_events:
                prev_event.next_events.append(event)

            entity_1.events["collided"] = [event]
            entity_2.events["collided"] = [event]
            # entity.events["pos"] = [event]

            self._world_state.add_event(event, "outcome")
#################

