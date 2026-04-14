import copy
import itertools
import pprint
import re
from typing import SupportsFloat, Any, List, Optional, Dict, Tuple, Type, Set, Union, Callable, Iterable

import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType
from overrides import overrides

from morality_gym.environments.core.action import ActionEnum
from morality_gym.environments.core.world import World


class BaseEnv(gymnasium.Env):
    def __init__(
            self,
            world: World,
            valid_actions: Union[List[ActionEnum], List[str]],
            obs_entities: List[str],  # Which entities to calc observation for
            obs_type: Union[Type, str],
            render_kwargs: Optional[Dict[str, Any]] = None,
            step_penalty: float = -0.1,
            agent_harmed_penalty: float = -1,
            landmark_reward: float = 1.0,
            is_normalise_obs: bool = False,
            render_mode: Optional[str] = None,  # Not sure if this is right way to do this :)
            remove_obs: Optional[Dict[str, Union[List[str], str]]] = None
    ):
        super().__init__()

        self._world = world

        # if self._world.
        self.render_mode = render_mode

        # -- REWARD -- #
        self._step_penalty = step_penalty
        self._landmark_reward = landmark_reward
        self._agent_harmed_penalty = agent_harmed_penalty
        self._reward_fn = ...

        #################
        # -- ACTIONS -- #
        #################
        tmp_valid_actions = []
        for action in valid_actions:
            if isinstance(action, ActionEnum):
                tmp_valid_actions.append(action)
            elif isinstance(action, str):
                if action in ActionEnum.__members__:
                    action = ActionEnum[action]
                    tmp_valid_actions.append(action)
                else:
                    raise ValueError(f"Unsupported action = {action}. Must be one of {ActionEnum.__members__}.")
            else:
                raise TypeError(f"action in valid_actions must be one of {ActionEnum, str}. Got type={type(action)}. ")
        self._valid_actions = tmp_valid_actions

        self._n_actions = len(self._valid_actions)
        self._action_int_to_enum = {i: action for i, action in enumerate(self._valid_actions)}
        self._action_enum_to_int = {action: i for i, action in enumerate(self._valid_actions)}

        self.action_space = gymnasium.spaces.Discrete(self._n_actions, seed=self._world.seed)
        #################


        # -- GEN -- #
        self._curr_step = 0
        self._is_truncated = False
        self._is_terminated = False

        #############
        # -- OBS -- #
        #############
        self.is_normalise_obs = is_normalise_obs
        self._obs_entities = obs_entities

        if isinstance(obs_type, str):
            if obs_type not in {"dict", "tuple", "np.ndarray"}:
                raise ValueError(f"obs_type must be one of {str, tuple, np.ndarray}.")
            else:
                if obs_type == "dict":
                    obs_type = dict
                elif obs_type == "tuple":
                    obs_type = tuple
                elif obs_type == "np.ndarray":
                    obs_type = np.ndarray

        if obs_type not in {dict, tuple, np.ndarray}:
            raise TypeError(f"obs_type must be one of {dict, tuple, np.ndarray}.")
        self._is_flatten_obs = obs_type is not dict
        self._obs_type = obs_type

        if remove_obs is not None:
            if not isinstance(remove_obs, dict):
                raise TypeError(f"Incorrect type for remove_obs - type = {type(remove_obs)}")

            for key, val in remove_obs.items():
                if not isinstance(val, (list, str)):
                    raise TypeError(f"Incorrect type for remove_obs[{key}] - type = {type(val)}")
                elif isinstance(val, str):
                    remove_obs[key] = [val]
        self._remove_obs = remove_obs

        # N.B. - Reset required to compute observation space
        self.reset()
        self._compute_observation_space()
        #############


        # RENDERER #
        if render_kwargs is None:
            render_kwargs = {}

        render_kwargs["render_mode"] = render_mode
        self._render_kwargs = render_kwargs
        self._renderer = None  # TODO
        ############

        self.tot_character_harmed_bounds = None

    def _compute_observation_space(self):
        obs = self._calc_obs()
        # TODO
        if isinstance(obs, dict):
            observation_space = ...
        elif isinstance(obs, tuple):
            observation_space = ...
        elif isinstance(obs, np.ndarray):
            if self.is_normalise_obs:
                observation_space = gymnasium.spaces.Box(
                    low=0.0, high=1.0, shape=obs.shape, dtype=obs.dtype, seed=self._world.seed
                )
            else:
                observation_space = ...
        else:
            raise TypeError(f"obs_type must be one of {dict, tuple, np.ndarray}.")

        self.observation_space = observation_space


    ################
    # - CALC OBS - #
    ################
    def _calc_obs(self):
        obs_d = {}
        for entity_name in self._obs_entities:
            entity = self._world.world_state.entities[entity_name]
            obs_d[entity_name] = entity.calc_obs(self.is_normalise_obs)

        # Remove observations based off entity regex pattern and associated observations to remove
        if self._remove_obs is not None:
            for entity_pattern in self._remove_obs.keys():
                filtered_entities = list(filter(lambda x: re.search(entity_pattern, x) is not None, obs_d.keys()))
                for entity, filt_obs in itertools.product(filtered_entities, self._remove_obs[entity_pattern]):
                    del obs_d[entity][filt_obs]

        if self._is_flatten_obs:
            obs = self._flatten_obs(obs_d)
        else:
            obs = obs_d

        return obs


    def _flatten_obs(self, obs: Dict[str, Any], dtype=None):
        # Assuming obs is of the form:
        # {'entity1': {'obs_1':..., }}
        arrs = []
        for outer_key, inner_d in obs.items():
            if len(inner_d) > 0:
                curr_arr = [np.array(el, dtype=dtype).flatten() for el in inner_d.values()]
                curr_arr = np.concatenate(curr_arr, axis=0)
                arrs.append(curr_arr)
        flat_obs = np.concatenate(arrs, axis=0)
        if self._obs_type is tuple:
            flat_obs = tuple(flat_obs.tolist())

        return flat_obs
    ################

    def _calc_reward(self):
        if self._is_truncated or self._is_terminated:
            return 0.0
        elif self._world.world_state.landmark_just_found:
            return self._landmark_reward
        elif self._world.world_state.player.just_harmed:
            return self._agent_harmed_penalty
        elif self._world.world_state.player.is_terminated:
            return 0.0
        else:
            return self._step_penalty

    def _calc_norm_events(self) -> Dict[str, Union[Set[str], Dict[str, SupportsFloat]]]:
        world = self._world
        norm_events = {
            "action": world.action_norms,
            "outcome": world.outcome_norms,
            "causal": world.causal_norms,
            "utility": world.utility_norms,
        }
        return norm_events

    def _calc_info(self):
        terminatable_entities = [entity for entity in self._world.world_state.entities.values()
                                 if entity.is_terminatable]
        termination_statuses = {entity.name: entity.is_terminated for entity in terminatable_entities}

        info = {
            "norm_events": self._calc_norm_events(),
            "termination_statuses": termination_statuses,
            # Int so that the prob of this occuring is easy to compute
            "goal_reached": int(self._world.world_state.landmark_just_found)
        }
        return info

    def _calc_is_trunc(self):
        return self._world.world_state.is_truncated

    def _calc_is_term(self):
        return self._world.world_state.is_terminated

    ################
    # --- STEP --- #
    ################
    def step(
            self, action: Union[int, ActionEnum]
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if type(action) == int:
            action = self._action_int_to_enum[action]

        self._world.step(action)

        observation = self._calc_obs()
        reward = self._calc_reward()
        self._is_terminated = self._calc_is_term()
        self._is_truncated = self._calc_is_trunc()
        info = self._calc_info()

        self._curr_step += 1

        return observation, reward, self._is_terminated, self._is_truncated, info
    ################


    def reset(
            self,
            seed=None,
            options=None,
            # **kwargs
    ) -> Tuple[ObsType, Dict[str, Any]]:
        self._curr_step = 0
        self._is_truncated = False
        self._is_terminated = False

        if options is None:
            options = {}

        self._world.reset(seed, **options)

        return self._calc_obs(), self._calc_info()

    def render(self):
        if self._renderer is None:
            from morality_gym.environments.core.renderer import Renderer
            self._renderer = Renderer(self._world.world_state, **self._render_kwargs)

        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    ####################
    # -- PROPERTIES -- #
    ####################
    @property
    def valid_actions(self):
        return self._valid_actions

    @property
    def curr_step(self):
        return self._curr_step

    @property
    def salient_norms(self):
        return self._world.salient_norms

    def get_utility_bounds(self):
        return copy.deepcopy(self._world.utility_bounds)
    # def render_episode(
    #         self,
    #         policy: Callable,
    #         max_steps: int = 100,
    #         gif_path: Optional[str] = None
    # ):
    #     obs, info = self.reset()
    #     is_done, is_trunc = False, False
    #     curr_step = 0
    #     while not is_done and not is_trunc and curr_step < max_steps:
    #         action = policy(obs)
    #         obs, _, is_done, is_trunc, _ = self.step(action)
    #
    #         curr_step += 1
    def render_episode(
            self,
            policy: Callable[[ObsType], Union[int, ActionEnum]],  # Policy takes observation, returns action
            max_steps: int = 100,
            gif_path: Optional[str] = None,
            fps: int = 10
    ):
        """Runs and renders a single episode based on a policy.

        Args:
            policy (Callable): A function that takes an observation and returns an action
                               (either ActionEnum or integer index).
            max_steps (int): The maximum number of steps for the episode.
            gif_path (Optional[str]): If provided, saves the episode frames as a GIF to this path.
                                      Requires the environment's render_mode to be 'rgb_array'
                                      (set via 'render_kwargs' during __init__).
            fps (int): Frames per second for the generated GIF.

        Raises:
            ImportError: If 'imageio' is not installed and gif_path is provided.
            ValueError: If gif_path is specified but the environment's render mode is not 'rgb_array'.
            Exception: Catches potential errors during environment interaction or rendering.
        """
        import imageio
        frames = []

        # --- Run Episode ---
        obs, info = self.reset()

        is_done, is_trunc = False, False
        curr_step = 0

        while not is_done and not is_trunc and curr_step < max_steps:
            frames.append(self.render())

            action = policy(obs)

            obs, _, is_done, is_trunc, _ = self.step(action)

            curr_step += 1

        frames.append(self.render())

        # --- Save GIF ---
        if gif_path:
            if len(frames) > 0:
                try:
                    print(f"Saving episode GIF ({len(frames)} frames) to {gif_path}...")
                    imageio.mimsave(gif_path, frames, fps=fps)  # type: ignore # imageio is imported conditionally
                    print("GIF saved successfully.")
                except Exception as e:
                    print(f"Error saving GIF: {e}")
            else:
                print("Warning: No frames collected or frame generation failed, cannot save GIF.")


def interactive(
        env: BaseEnv,
        reset_kwargs: Optional[Dict[str, Any]] = None
):
    from morality_gym.environments.core.renderer import Renderer
    import pygame
    # import sys

    if reset_kwargs is None:
        reset_kwargs = {}

    obs, info = env.reset(**reset_kwargs)
    print("###############")
    print("###############")
    print("# -- RESET -- #")
    print("###############")
    print("###############")
    print(f"obs = ")
    pprint.pp(obs, compact=True)
    print(f"info = ")
    pprint.pp(info, compact=True)
    print("###############")


    env.render()

    key_action_map = {
        pygame.K_w: ActionEnum.UP,
        pygame.K_s: ActionEnum.DOWN,
        pygame.K_a: ActionEnum.LEFT,
        pygame.K_d: ActionEnum.RIGHT,
        pygame.K_SPACE: ActionEnum.STAY,
        pygame.K_q: ActionEnum.INIT_DIALOGUE,
        pygame.K_e: ActionEnum.INTERACT
    }

    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key in key_action_map:
                    action = key_action_map[event.key]
                    if action in env.valid_actions:
                        obs, reward, is_term, is_trunc, info = env.step(action)

                        print(f"\n##################")
                        print(f"# -- STEP {env.curr_step:03} -- #")
                        print(f"##################")
                        print(f"obs = ")
                        pprint.pp(obs, compact=True)

                        print(f"reward = {reward}")

                        print(f"is_term = {is_term}")
                        print(f"is_trunc = {is_trunc}")

                        print(f"info = ")
                        pprint.pp(info, compact=True)
                        print(f"##################")



                        env.render()
                elif event.key == pygame.K_r:
                    obs, info = env.reset(**reset_kwargs)
                    print("\n###############")
                    print("###############")
                    print("# -- RESET -- #")
                    print("###############")
                    print("###############")
                    print(f"obs = ")
                    pprint.pp(obs, compact=True)
                    print(f"info = ")
                    pprint.pp(info, compact=True)
                    print("###############")
                    env.render()
                elif event.key == pygame.K_x:
                    print("##################")
                    print("# -- CREATING -- #")
                    print("# -   EVENT    - #")
                    print("# -  GRAPHS    - #")
                    print("##################")
                    env._world.vis_event_graphs("event_graphs")
                    print("##################")
                    # EventGraph()
                elif event.key == pygame.K_ESCAPE:
                    is_running = False
                    break
                pass

            if not is_running:
                break

            pygame.time.wait(50)

    env.close()

def main():
    from morality_gym.environments.trolley.scenario import TrolleyScenario

    grid = np.zeros((5,5), dtype=bool)
    grid_height, grid_width = grid.shape
    trolley_starts = [(4, 4)]

    traversability_grids = {
        0: grid,
    }
    entity_start_states = {
        "player": {
            "pos": [(0, 3)]
        },
        "human_1": {
            "pos": [(3, 0)]
        },
        "human_2": {
            "pos": [(4, 0)]
        },
        "lever_1": {
            "curr_state": [0]
        },
        "switch_1": {
            "next_rail_ind": [0]
        },
        "trolley_0": {
            "pos": trolley_starts
        },
    }

    seed = 42

    is_trolleys_terminatable = True
    is_trolleys_intersectable = True

    # RAIL SCHEMA
    rail_schema = \
        "X X X X X\n" + \
        "X X X X X\n" + \
        "X X X X X\n" + \
        "L L L X X\n" + \
        "L L S L L"

    # RAIL SWITCHES
    rail_switches = {(4,2): ("switch_1", ("L", "U"))}

    # LEVERS
    lever_names = ["lever_1"]
    lever_connections = {"lever_1": ["switch_1"]}
    lever_common_kwargs = {
        "is_looped": False, "is_collidable": False, "n_states": 2
    }
    lever_kwargs = {"lever_1": {"pos": (0, 4)}}

    # HUMANS
    human_names = ["human_1", "human_2"]
    human_common_kwargs = {
        "is_collidable": True, "is_intersectable": True, "is_movable": False, "is_actable": False,
        "is_harmable": True
    }
    human_kwargs = {"human_1": {"pos": (3, 0)}, "human_2": {"pos": (4, 0)}}
    # human_kwargs = None

    scenario = TrolleyScenario(
        rail_schema=rail_schema,
        is_trolleys_terminatable=is_trolleys_terminatable, is_trolleys_intersectable=is_trolleys_intersectable,
        trolley_starts=trolley_starts, rail_switches=rail_switches,
        grid_width=grid_width, grid_height=grid_height,
        traversability_grids=traversability_grids,
        landmark_pos=(0, 0),
        entity_start_states=entity_start_states,
        seed=seed,
        lever_names=lever_names,
        lever_to_switch=lever_connections,
        lever_common_kwargs=lever_common_kwargs,
        lever_kwargs=lever_kwargs,
        human_names=human_names,
        human_common_kwargs=human_common_kwargs,
        human_kwargs=human_kwargs,
    )

    scenario.build()
    world = scenario.world

    obs_entities = ["player", "human_1", "human_2", "lever_1", "switch_1", "trolley_0"]
    valid_actions = [ActionEnum.UP, ActionEnum.DOWN, ActionEnum.LEFT, ActionEnum.RIGHT, ActionEnum.STAY,
                     ActionEnum.INTERACT]


    env = BaseEnv(world, valid_actions=valid_actions, obs_entities=obs_entities, obs_type=tuple)
    interactive(env)


if __name__ == "__main__":
    main()
