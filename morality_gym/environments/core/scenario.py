from typing import Optional, Union, Dict, Any, List, Callable, Set, SupportsFloat, Tuple

import numpy as np

from morality_gym.environments.core.custom_types import PosType, StateChangeType
from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.dynamics.interact import InteractDynamics
from morality_gym.environments.core.dynamics.landmark import LandmarkDynamics
from morality_gym.environments.core.dynamics.move import MoveDynamics
from morality_gym.environments.core.dynamics.pre_post_step import PreStepDynamics, PostStepDynamics
from morality_gym.environments.core.dynamics.proc_events import ProcessEventsDynamics
from morality_gym.environments.core.dynamics.scripted import ScriptedDynamics
from morality_gym.environments.core.dynamics.state_setup import InitDynamics, ResetDynamics
from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.entity.character import CharacterEntity
from morality_gym.environments.core.entity.group import EntityGroup
from morality_gym.environments.core.entity.landmark import LandmarkEntity
from morality_gym.environments.core.entity.player import PlayerEntity
from morality_gym.environments.core.event import Event
from morality_gym.environments.core.state import WorldState
from morality_gym.environments.core.utils import compute_kwargs
from morality_gym.environments.core.vis_fns import make_player_fns
from morality_gym.environments.core.world import World


# from morality_gym.environment.dynamics.dynamics import BaseDynamics, ResetDynamics, MoveDynamics, PreStepDynamics, \
#     PostStepDynamics, InitDynamics, InteractDynamics, ScriptedDynamics
# from morality_gym.environment.entity.entity import PlayerEntity, BaseEntity, EntityGroup
# from morality_gym.environment.scenario.vis_fns import make_player_fns
# from morality_gym.environment.state import WorldState
# from morality_gym.environment.custom_types import PosType
# from morality_gym.environment.utils import compute_kwargs



class BaseScenario:
    DEFAULT_PLAYER_KWARGS = {
        "is_collidable": True,
        "is_movable": True,
        "is_actable": True,
        "is_harmable": True,
        "is_terminatable": True,
        "is_intersectable": True,
        "vis_layer": 2,
    }

    def __init__(
            self,
            grid_width: int,
            grid_height: int,
            traversability_grids: Optional[Dict[int, np.ndarray]] = None,
            # entity_start_states: Optional[Dict[str, Dict[Union[str, Tuple[str, ...]], List[Any]]]] = None,
            entity_start_states: Optional[Dict[str, Dict[str, List[Any]]]] = None,
            # Player
            player_kwargs: Optional[Dict[str, Any]] = None,
            # Landmark
            landmark_pos: Optional[PosType] = None,
            # RNG
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None,

            event_to_outcome_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_action_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_causal_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_utility_fns: Optional[Dict[str, Callable[[Event], Dict[str, SupportsFloat]]]] = None,

            utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,
            global_utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,

            salient_norms: Optional[List[str]] = None,

            landmark_reached_mode: str = "interact",
            randomise_variant: bool = False,  # I.e. if to randomise variant on reset

            max_timesteps: int = np.inf
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height

        # self.entity_start_states = entity_start_states
        self.entity_start_states = entity_start_states

        self.player_kwargs = player_kwargs

        self.rng = rng
        self.seed = seed

        self._is_built = False

        self.pos_bounds = ((0, 0), (grid_height, grid_width))

        self.salient_norms = salient_norms

        self.max_timesteps = max_timesteps

        ###################
        # INIT NONE DICTS #
        ###################
        if self.player_kwargs is None:
            self.player_kwargs = {}
        ###################

        # ##############
        # # RNG & SEED #
        # ##############
        # if self.rng is None:
        #     self.rng = np.random.default_rng(self.seed)
        # ##############

        self.entities: Dict[str, BaseEntity] = {}
        self.grouped_entities: Dict[str, List[BaseEntity]] = {}
        self.entity_groups: Dict[str, EntityGroup] = {}

        ##############
        # - PLAYER - #
        ##############
        player_name = "player"
        p_to_asset_fn, p_to_rot_fn, p_to_alpha_fn = make_player_fns("robot")

        full_player_kwargs = compute_kwargs(
            default_kwargs=self.DEFAULT_PLAYER_KWARGS,
            kwargs=self.player_kwargs,
            overrides={
                "name": player_name,
                "pos": (0, 0),
                "group": "player",
                "to_asset_fn": p_to_asset_fn,
                "to_rot_fn": p_to_rot_fn,
                "to_alpha_fn": p_to_alpha_fn,
                "pos_bounds": self.pos_bounds,
            }
        )
        self.player = PlayerEntity(**full_player_kwargs)

        # self.player = AgentEntity(
        #     name="player", group="player",
        #     is_collidable=self._is_player_collidable, is_movable=True, is_actable=True,
        #     is_player=True
        # )
        self.entities[player_name] = self.player
        self.grouped_entities["player"] = [self.player]
        ##############

        ##################
        # -- LANDMARK -- #
        ##################
        self.landmark_reached_mode = landmark_reached_mode
        if landmark_reached_mode not in {"interact", "enter"}:
            raise ValueError(f"Invalid landmark reached mode = {landmark_reached_mode}. Valid values = 'interact' or 'enter'")
        if landmark_pos is not None:
            self.landmark = LandmarkEntity("landmark", "landmark", landmark_pos, vis_layer=0)
            self.entities["landmark"] = self.landmark
            self.grouped_entities["landmark"] = [self.landmark]
        else:
            self.landmark = None
        ##################

        ########################
        # TRAVERSABILITY GRIDS #
        ########################
        if traversability_grids is None:
            traversability_grids = {}

        if not set(traversability_grids.keys()).issubset(set(range(3))):
            raise ValueError(f"traversability_grids must be a dict with keys in {set(range(3))}. "
                             f"traversability_grids.keys() = {list(traversability_grids.keys())}")

        for i in range(3):
            if i in traversability_grids:
                if traversability_grids[i].shape != (grid_height, grid_width):
                    raise ValueError(f"traversability_grids[{i}] has invalid shape = {traversability_grids[i].shape}.")
                continue
            traversability_grids[i] = np.zeros((grid_height, grid_width), dtype=bool)

        self.traversability_grids = traversability_grids
        ########################

        self.world_state: Optional[WorldState] = None
        self.world: Optional[World] = None

        ##################
        # -- DYNAMICS -- #
        ##################
        self.dynamics: Optional[List[BaseDynamics]] = None
        self.init_dynamics: Optional[InitDynamics] = None
        self.pre_step_dynamics: Optional[PreStepDynamics] = None
        self.post_step_dynamics: Optional[PostStepDynamics] = None
        self.reset_dynamics: Optional[ResetDynamics] = None
        ##################

        ##################
        # NORM EVENT FNS #
        ##################
        self.event_to_outcome_fns = event_to_outcome_fns
        self.event_to_action_fns = event_to_action_fns
        self.event_to_causal_fns = event_to_causal_fns
        self.event_to_utility_fns = event_to_utility_fns
        ##################

        ##################
        # UTILITY BOUNDS #
        ##################
        self.utility_bounds = utility_bounds
        self.global_utility_bounds = global_utility_bounds
        ##################

        self.randomise_variant = randomise_variant
        self.randomise_characters = None

    ###############
    # -- BUILD -- #
    ###############
    def _build_dynamics(self):
        if self.world_state is None:
            raise ValueError("world_state must be built before calling _build_dynamics.")

        scripted_dynamics = ScriptedDynamics(self.world_state)
        interact_dynamics = InteractDynamics(self.world_state)
        move_dynamics = MoveDynamics(self.world_state)
        self.dynamics: List[BaseDynamics] = [scripted_dynamics, interact_dynamics, move_dynamics]
        if self.landmark is not None:
            landmark_dynamics = LandmarkDynamics(self.world_state, landmark_reached_mode=self.landmark_reached_mode)
            self.dynamics.append(landmark_dynamics)

        process_events_dynamics = ProcessEventsDynamics(self.world_state)
        self.dynamics.append(process_events_dynamics)

        # INIT DYNAMICS #
        self.init_dynamics: InitDynamics = InitDynamics(self.world_state)

        # PRE & POST STEP DYNAMICS #
        self.pre_step_dynamics: PreStepDynamics = PreStepDynamics(self.world_state)
        self.post_step_dynamics: PostStepDynamics = PostStepDynamics(self.world_state)

        # RESET DYNAMICS #
        self.reset_dynamics: ResetDynamics = ResetDynamics(
            self.world_state,
            self.randomise_variant,
            # self.randomise_characters
        )


    def _build_world(self):
        if not self._is_built:
            raise ValueError("Scenario must be built before calling _build_world.")
        self.world = World(
            world_state=self.world_state,
            dynamics=self.dynamics,
            init_dynamics=self.init_dynamics,
            pre_step_dynamics=self.pre_step_dynamics,
            post_step_dynamics=self.post_step_dynamics,
            reset_dynamics=self.reset_dynamics,
            event_to_outcome_fns=self.event_to_outcome_fns,
            event_to_action_fns=self.event_to_action_fns,
            event_to_causal_fns=self.event_to_causal_fns,
            event_to_utility_fns=self.event_to_utility_fns,
            utility_bounds=self.utility_bounds,
            salient_norms=self.salient_norms
        )

    def _build_env(self):
        pass

    def build(self):
        self._is_built = True
        self.world_state = self._create_world_state()
        self._build_dynamics()
        self._build_world()

        # return self.world_state
    ###############

    ##################
    # ADD ENTITIES & #
    # ENTITY GROUPS  #
    ##################
    def add_entities(self, entities: Union[BaseEntity, List[BaseEntity]]):
        if isinstance(entities, BaseEntity):
            entities = [entities]
        elif not isinstance(entities, list):
            raise ValueError(f"entities must be a list of BaseEntity or a single BaseEntity. "
                             f"type(entities) = {type(entities)}")

        for entity in entities:
            self.entities[entity.name] = entity
            if entity.group not in self.grouped_entities:
                self.grouped_entities[entity.group] = [entity]
            else:
                self.grouped_entities[entity.group].append(entity)


    def add_entity_groups(self, entity_groups: Union[EntityGroup, List[EntityGroup]]):
        entities = []

        for entity_group in entity_groups:
            self.entity_groups[entity_group.name] = entity_group
            entities = entities + list(entity_group.entities.values())

        self.add_entities(entities)
    ##################

    def _load_from_config(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        # TODO
        raise NotImplementedError

    def _validate_init(self):
        # TODO
        raise NotImplementedError

    # # noinspection PyMethodMayBeStatic
    # def _init_dynamics(self) -> List[BaseDynamics]:
    #     dynamics = []
    #     return dynamics

    # # noinspection PyMethodMayBeStatic
    # def _init_reset_dynamics(self) -> ResetDynamics:
    #     reset_dynamics = ResetDynamics(self.world_state, reset_entities=["player"])
    #     return reset_dynamics


    def _create_world_state(self) -> WorldState:
        world_state = WorldState(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            player=self.player,
            entities=self.entities,
            grouped_entities=self.grouped_entities,
            entity_start_states=self.entity_start_states,
            traversability_grids=self.traversability_grids,
            entity_groups=self.entity_groups,
            landmark=self.landmark,
            rng=self.rng,
            seed=self.seed,
            max_timesteps=self.max_timesteps
        )
        return world_state

    ##############
    # PROPERTIES #
    ##############
    @property
    def is_built(self) -> bool:
        return self._is_built


def main():
    grid_width = 10
    grid_height = 10
    traversability_grids = {
        0: np.zeros((grid_height, grid_width), dtype=bool),
    }
    traversability_grids[0][1,1] = True

    entity_start_states = {
        "player": {
            "pos": [(0,0)]
        }
    }
    # player_start_pos = [(0, 0)]
    seed = 42

    scenario = BaseScenario(
        grid_width=grid_width,
        grid_height=grid_height,
        traversability_grids=traversability_grids,
        entity_start_states=entity_start_states,
        seed=seed
    )

    # world_state = scenario.create_world_state()
    # print(world_state)


if __name__ == "__main__":
    main()