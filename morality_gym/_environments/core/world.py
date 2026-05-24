import os
from typing import List, Optional, Tuple, Dict, Callable, Set

import numpy as np
from typing_extensions import SupportsFloat

from morality_gym._environments.core.action import ActionEnum
from morality_gym._environments.core.custom_types import PosType, StateChangeType
from morality_gym._environments.core.dynamics.base import BaseDynamics
from morality_gym._environments.core.dynamics.pre_post_step import PreStepDynamics, PostStepDynamics
from morality_gym._environments.core.dynamics.state_setup import ResetDynamics, InitDynamics
from morality_gym._environments.core.entity.base import BaseEntity
from morality_gym._environments.core.event import EventGraph, Event
# from morality_gym._environments.core.scenario import BaseScenario
from morality_gym._environments.core.state import WorldState
from morality_gym._utils.common import join_paths


class World:
    def __init__(
            self,
            world_state: WorldState,
            dynamics: List[BaseDynamics],
            init_dynamics: InitDynamics,
            pre_step_dynamics: PreStepDynamics,
            post_step_dynamics: PostStepDynamics,
            reset_dynamics: ResetDynamics,
            # event_to_outcome_fns: Optional[str, Callable[[List[BaseEntity], Dict[str, StateChangeType]], str]] = None,
            event_to_outcome_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_action_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_causal_fns: Optional[Dict[str, Callable[[Event], Set[str]]]] = None,
            event_to_utility_fns: Optional[Dict[str, Callable[[Event], Dict[str, SupportsFloat]]]] = None,
            utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,
            salient_norms: Optional[List[str]] = None,
    ):
        self.world_state = world_state

        self._dynamics = dynamics
        self._init_dynamics = init_dynamics
        self._pre_step_dynamics = pre_step_dynamics
        self._post_step_dynamics = post_step_dynamics
        self._reset_dynamics = reset_dynamics

        self._init_dynamics()

        self._action_norms: Set[str] = set()
        self._outcome_norms: Set[str] = set()
        self._causal_norms: Set[str] = set()
        self._utility_norms: Dict[str, SupportsFloat] = {}

        self._event_to_outcome_fns = event_to_outcome_fns
        self._event_to_action_fns = event_to_action_fns
        self._event_to_causal_fns = event_to_causal_fns
        self._event_to_utility_fns = event_to_utility_fns

        self.utility_bounds = utility_bounds
        self.salient_norms = salient_norms
        # self.entity_state_to_asset_fns = scenario.entity_state_to_asset_fns
        # self.entity_state_to_rot_fns = scenario.entity_state_to_rot_fns
        # self.entity_layers = scenario.entity_layers

    def _call_dynamics(self):
        self._pre_step_dynamics()
        for dyn in self._dynamics:
            dyn()
        self._post_step_dynamics()

    ###############
    # NORM EVENTS #
    ###############
    def _reset_norms(self):
        self._action_norms = set()
        self._outcome_norms = set()
        self._causal_norms = set()
        self._utility_norms = {}

    def _proc_outcome_norms(self):
        self._outcome_norms = set()
        if self._event_to_outcome_fns is None:
            return
        for event in self.world_state.recent_outcome_events:
            outcome_descr = event.outcome_descr
            if outcome_descr in self._event_to_outcome_fns:
                if event.affected_entities is None:
                    raise ValueError("Event has no affected entities.")

                curr_norms = self._event_to_outcome_fns[outcome_descr](event)

                if curr_norms is not None:
                    self._outcome_norms =self._outcome_norms.union(curr_norms)

    def _proc_action_norms(self):
        self._action_norms = set()
        if self._event_to_action_fns is None:
            return
        for event in self.world_state.recent_action_events:
            action_descr = event.action_descr
            if action_descr in self._event_to_action_fns:
                if event.affected_entities is None:
                    raise ValueError("Event has no affected entities.")

                curr_norms = self._event_to_action_fns[action_descr](event)

                if curr_norms is not None:
                    self._action_norms =self._action_norms.union(curr_norms)
        # raise NotImplementedError(f"This method has not been checked")

    def _proc_causal_norms(self):
        self._causal_norms = set()
        if self._event_to_causal_fns is None:
            return
        for event in self.world_state.recent_proc_causal_events:
            causal_descr = event.causal_descr
            if causal_descr in self._event_to_causal_fns:
                if event.affected_entities is None:
                    raise ValueError("Event has no affected entities.")

                curr_norms = self._event_to_causal_fns[causal_descr](event)

                if curr_norms is not None:
                    self._causal_norms =self._causal_norms.union(curr_norms)
        # raise NotImplementedError(f"This method has not been checked")

    def _proc_utility_norms(self):
        # This may or may not work - idk
        self._utility_norms = {}
        # Utilities are defined off outcomes
        if self._event_to_utility_fns is None:
            return

        for event in self.world_state.recent_outcome_events:
            outcome_descr = event.outcome_descr
            if outcome_descr in self._event_to_utility_fns:
                if event.affected_entities is None:
                    raise ValueError("Event has no affected entities.")
                curr_utilities = self._event_to_utility_fns[outcome_descr](event)
                for key, val in curr_utilities.items():
                    if key in self._utility_norms:
                        self._utility_norms[key] += val
                    else:
                        self._utility_norms[key] = val

    def _proc_norms(self):
        self._reset_norms()

        self._proc_outcome_norms()
        self._proc_action_norms()
        self._proc_causal_norms()
        self._proc_utility_norms()

    @property
    def action_norms(self) -> Set[str]:
        return self._action_norms

    @property
    def outcome_norms(self) -> Set[str]:
        return self._outcome_norms

    @property
    def causal_norms(self) -> Set[str]:
        return self._causal_norms

    @property
    def utility_norms(self) -> Dict[str, SupportsFloat]:
        return self._utility_norms
    ###############

    def step(
            self,
            action: ActionEnum,
            _next_pos: Optional[PosType] = None
    ):
        if not self.world_state.has_reset:
            raise ValueError("World has not been reset.")

        self.world_state.player.set_action(action, _next_pos)
        if not self.world_state.is_terminated and not self.world_state.is_truncated:
            self._call_dynamics()
            self._proc_norms()
        else:
            self._reset_norms()
            print("World has terminated or truncated. Cannot process dynamics.")

    def reset(
            self,
            seed: Optional[int] = None,
            **kwargs
    ):
        if seed is not None:
            del self.world_state.rng  # To ensure old rng is not accidentally used
            self.world_state.rng = np.random.default_rng(seed)  # Reseed if seed given
        self._reset_dynamics(**kwargs)
        self._reset_norms()

    def vis_event_graphs(self, folder_path):
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        ws = self.world_state
        common_kwargs = {"font_size": 12, "node_size": 5000, "k_layout":1.25}

        # Events
        EventGraph(ws.events).vis_graph(join_paths([folder_path, "events.png"]), **common_kwargs)
        # Recent Events
        EventGraph(ws.recent_events).vis_graph(join_paths([folder_path, "recent_events.png"]), **common_kwargs)

        # Processed Causal Events
        EventGraph(ws.proc_causal_events, include_connections=True).vis_graph(
            join_paths([folder_path, "proc_causal_events.png"]), **common_kwargs)
        EventGraph(ws.recent_proc_causal_events, include_connections=True).vis_graph(
            join_paths([folder_path, "recent_proc_causal_events.png"]), **common_kwargs)

    # OTHER PROPERTIES #
    @property
    def seed(self):
        return self.world_state.seed


def interactive(world: World):

    from morality_gym._environments.core.renderer import Renderer
    import pygame
    # import sys

    renderer = Renderer(
        world.world_state,
        # world.entity_state_to_asset_fns, world.entity_state_to_rot_fns, world.entity_layers,
        render_mode="human"
    )
    renderer.render()

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
                    world.step(action)
                    renderer.render()
                elif event.key == pygame.K_r:
                    world.reset()
                    renderer.render()
                elif event.key == pygame.K_ESCAPE:
                    is_running = False
                    break
                pass

            if not is_running:
                break

            pygame.time.wait(50)

    renderer.close()

def main():
    pass


if __name__ == "__main__":
    main()
