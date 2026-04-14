from typing import Optional, List, Dict, Union

from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.entity.character import CharacterEntity
from morality_gym.environments.core.state import WorldState


#################
# INIT DYNAMICS #
#################
class InitDynamics(BaseDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def _init_pos_to_entities(self):
    #     world_state = self._world_state
    #
    #     for name, entity in world_state.entities.items():
    #         pos = entity.pos
    #         world_state.pos_to_entities[pos[0], pos[1]].add(name)



    def __call__(self):
        # self._init_pos_to_entities()

        #
        self._world_state.terminatable_entities = {name: entity for name, entity in self._world_state.entities.items()
                                                   if entity.is_terminatable}
        pass

##################
# RESET DYNAMICS #
##################
# TODO: Check and rework if needed
class ResetDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState,
            randomise_variant: bool = False,
            # randomise_characters: Optional[List[str]] = None,
            # reset_entities: List[str],   # Names of entities to reset
            # rng: Optional[np.random.Generator] = None
    ):
        super().__init__(world_state=world_state)

        self._randomise_variant = randomise_variant
        if randomise_variant:
            characters = list(filter(lambda entity: isinstance(entity, CharacterEntity), self._world_state.entities.values()))
            randomise_characters = []
            for character in characters:
                if character.valid_character_types is not None and character.valid_amounts is not None:
                    randomise_characters.append(character.name)
            self._randomise_characters = randomise_characters
        else:
            self._randomise_characters = None

        self._reset_entity_names = list(self._world_state.entity_start_states.keys())

        # TODO: Might require error checking for if all _reset_entity_names are valid
        self._reset_entities = {name: self._world_state.entities[name] for name in self._reset_entity_names}

        self.traversability_grids = self._world_state.traversability_grids
        # self.traversability_grids = {name: self._world_state.traversability_grids[name] for name
        #                              in self._reset_entity_names}

        entity_start_states = self._world_state.entity_start_states

        ###################
        # START POSITIONS #
        ###################
        self._start_positions = {}
        for name in self._reset_entity_names:
            # if name not in entity_start_states:
            #     raise ValueError(f"Entity {name} not found in entity_start_states.")
            curr_start_states = entity_start_states[name]
            if "pos" not in curr_start_states:
                continue
            self._start_positions[name] = curr_start_states["pos"]

        # Valid Start Positions - According to traversability_grids
        self._valid_start_positions = {}
        for name, curr_start_positions in self._start_positions.items():
            traversability_group = self._world_state.entities[name].traversability_group
            curr_trav_grid = self.traversability_grids[traversability_group]
            # Filter out positions where corresponding element in traversability_grids[name] is true
            self._valid_start_positions[name] = [(y, x) for (y, x) in curr_start_positions if not curr_trav_grid[y, x]]

        # For efficiency reasons
        self._valid_start_positions_set = {name: set(positions) for name, positions in
                                           self._valid_start_positions.items()}
        ################

    def _reset_entities_to_init(self):
        for entity in self._world_state.entities.values():
            if not entity.is_static:
                entity.reset_to_init()


    def _reset_positions(self, tries=100) -> bool:
        # Other entities that they could collide with
        collidable_entities = {name: entity for (name, entity) in self._world_state.entities.items()
                               if (entity.is_collidable and name not in self._reset_entity_names)}
        collidable_positions = {entity.pos for entity in collidable_entities.values()}

        valid_start_positions = {entity_name: (entity_start_positions.difference(collidable_positions))
                                 for entity_name, entity_start_positions in self._valid_start_positions_set.items()}


        rng = self._world_state.rng
        curr_try = 0
        positions_valid = False
        entity_positions = {}

        pos_entity_names = list(self._start_positions.keys())  # Entities we wish to reset the positions of

        # TODO
        while not positions_valid and curr_try < tries:
            # New positions for reset entities
            entity_positions = {name: None for name in pos_entity_names}
            taken_positions = set()
            positions_valid = True

            for entity_name, entity_valid_start_pos in valid_start_positions.items():
                available_positions = entity_valid_start_pos.difference(taken_positions)
                if len(available_positions) > 0:
                    avail_arr = list(available_positions)
                    rand_ind = rng.integers(len(avail_arr))
                    entity_positions[entity_name] = avail_arr[rand_ind]
                    taken_positions.add(avail_arr[rand_ind])
                else:
                    positions_valid = False
                    break

            curr_try += 1

        for entity_name in pos_entity_names:
            entity = self._reset_entities[entity_name]
            entity.pos = entity_positions[entity_name]
            entity.next_pos = None

        return positions_valid

    def _reset_states(self):
        rng = self._world_state.rng
        start_states = self._world_state.entity_start_states
        for name in self._reset_entity_names:
            # if "switch" not in name:
            self._reset_entities[name].reset_to_init()  # TODO: Check
            curr_entity_states = start_states[name]
            state_kwargs = {}
            for state, state_arr in curr_entity_states.items():
                if state == "pos":
                    continue
                curr_ind = rng.integers(len(state_arr))
                state_kwargs[state] = state_arr[curr_ind]
            self._reset_entities[name].set_state(**state_kwargs)

        # TODO: Check if resetting positions after state does not cause issues
        is_positions_valid = self._reset_positions()
        if not is_positions_valid:
            raise ValueError("Could not reset positions.")

    def _reset_pos_to_entities(self):
        world_state = self._world_state
        for key in world_state.pos_to_entities.keys():
            world_state.pos_to_entities[key] = set()

        for name, entity in world_state.entities.items():
            pos = entity.pos
            world_state.pos_to_entities[pos[0], pos[1]].add(name)

    def _reset_to_rand_variant(self):
        # print(f"###################")
        # print(f"_reset_to_rand_variant")
        # print(f"###################")
        if self._randomise_characters is None:
            raise ValueError("randomise_characters must be specified for _reset_to_rand_variant.")

        rng = self._world_state.rng

        for character_name in self._randomise_characters:
            character = self._world_state.entities[character_name]

            rand_ind = rng.integers(0, len(character.valid_character_types))
            character.character_type = character.valid_character_types[rand_ind]

            rand_ind = rng.integers(0, len(character.valid_amounts))
            character.amount = character.valid_amounts[rand_ind]

    def _reset_to_variant(
            self,
            variant_d: Dict[str, Dict[str, Union[str, int]]]
    ):
        for character_name, character_variant in variant_d.items():
            character = self._world_state.entities[character_name]
            character.character_type = character_variant["character_type"]
            character.amount = character_variant["amount"]

        # for character_name in self._randomise_characters:
        #     character = self._world_state.entities[character_name]
        #
        #     rand_ind = self._rng.integers(0, len(character.valid_character_types))
        #     character.character_type = character.valid_character_types[rand_ind]
        #
        #     rand_ind = self._rng.integers(0, len(character.valid_amounts))
        #     character.amount = character.valid_amounts[rand_ind]

    # def _reset_trolley(self):
    #     pass

    # TODO: Fix name
    def _reset_world(self):
        self._world_state.is_terminated = False
        self._world_state.is_truncated = False
        self._world_state.is_landmark_found = False
        self._world_state.timestep = 0

        # Events
        self._world_state.events = []
        self._world_state.action_events = []
        self._world_state.player_action_events = []
        self._world_state.outcome_events = []
        self._world_state.causal_events = []
        self._world_state.proc_causal_events = []

        self._world_state.recent_events = []
        self._world_state.recent_action_events = []
        self._world_state.recent_player_action_events = []
        self._world_state.recent_outcome_events = []
        self._world_state.recent_causal_events = []
        self._world_state.recent_proc_causal_events = []

    def __call__(
            self,
            characters_variant: Optional[Dict[str, Dict[str, Union[str, int]]]] = None
    ):
        self._world_state.has_reset = True
        self._reset_entities_to_init()
        self._reset_states()
        self._reset_world()
        self._reset_pos_to_entities()
        if self._randomise_variant:
            if characters_variant is None:
                self._reset_to_rand_variant()
            else:
                self._reset_to_variant(characters_variant)
        else:
            if characters_variant is not None:
                raise ValueError("characters_variant must be None if randomise_variant is False.")
##################

