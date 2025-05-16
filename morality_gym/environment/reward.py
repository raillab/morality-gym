from abc import abstractmethod
from typing import List, Dict, Any

import numpy as np

from morality_gym.environment.state import DangerState
from morality_gym.environment.world import World


# BASE REWARD #
class BaseReward:
    def __init__(
            self,
            world: World
    ):
        self._world = world

    def __call__(self):
        return self.calc_reward()

    @abstractmethod
    def calc_reward(self):
        raise NotImplementedError


class MultiSumReward(BaseReward):
    def __init__(
            self,
            world: World,
            reward_fns: List[str],
            reward_fn_kwargs: List[Dict[str, Any]]
    ):
        super().__init__(world)

        if len(reward_fns) == 0:
            raise ValueError("No reward_fns provided")

        self._reward_fns = []
        for curr_fn, fn_kwargs in zip(reward_fns, reward_fn_kwargs):
            if curr_fn == "harm":
                self._reward_fns.append(HarmReward(world, **fn_kwargs))
            elif curr_fn == "dist_to_landmark":
                self._reward_fns.append(DistToLandmarkReward(world, **fn_kwargs))
            else:
                raise ValueError(f"Invalid reward_fn = {curr_fn}")

    def calc_reward(self):
        return np.sum([reward_fn() for reward_fn in self._reward_fns])


# HARM REWARD #
class HarmReward(BaseReward):
    def __init__(
            self,
            world: World,
            # Robot
            r_harm: float = 0.0,
            r_min_harm: float = 0.0,
            r_maj_harm: float = 0.0,

            # Human
            h_harm: float = 0.0,
            h_min_harm: float = 0.0,
            h_maj_harm: float = 0.0,
            h_min_harm_by_robot: float = 0.0,
            h_min_harm_by_env: float = 0.0,
            h_maj_harm_by_robot: float = 0.0,
            h_maj_harm_by_env: float = 0.0,

            # Decoration
            d_harm: float = 0.0,
            d_min_harm: float = 0.0,
            d_maj_harm: float = 0.0,
    ):

        super().__init__(world)

        self._danger_state_rewards = {
            DangerState.RobotHarm: r_harm,
            DangerState.MajorRobotHarm: r_maj_harm,
            DangerState.MinorRobotHarm: r_min_harm,

            DangerState.HumanHarm: h_harm,
            DangerState.MajorHumanHarm: h_maj_harm,
            DangerState.MinorHumanHarm: h_min_harm,
            DangerState.MinorHumanHarmByRobot: h_min_harm_by_robot,
            DangerState.MinorHumanHarmByEnv: h_min_harm_by_env,
            DangerState.MajorHumanHarmByRobot: h_maj_harm_by_robot,
            DangerState.MajorHumanHarmByEnv: h_maj_harm_by_env,

            DangerState.PropertyHarm: d_harm,
            DangerState.MajorPropertyHarm: d_maj_harm,
            DangerState.MinorPropertyHarm: d_min_harm,
        }

    def calc_reward(self):
        danger_states = self._world.danger_states
        # all_danger_states = danger_states["option"] + danger_states["outcome"] + danger_states["option-outcome"]
        reward = np.sum([self._danger_state_rewards[danger_state] for danger_state in danger_states])
        # if len(danger_states) == 0:
        #     reward = 0.0
        # else:
        #     reward = np.sum([self._danger_state_rewards[danger_state] for danger_state in danger_states])
        return reward

# DIST TO LANDMARK REWARD #
class DistToLandmarkReward(BaseReward):
    def __init__(
            self,
            world: World,
            # dist_type: str = 'euclidean', # 'euclidean' or 'manhattan'
            a: float = 1.0,
            lm_reached_reward: float = 10.0
    ):
        # r = -1 * a * dist
        super().__init__(world)
        self._a = a
        self._lm_reached_reward = lm_reached_reward

        # def man_dist(pos1, pos2):
        #     return np.sum(pos1 - pos2)
        #
        # def eucl_dist(pos1, pos2):
        #     return np.linalg.norm(pos1 - pos2)
        #
        # if dist_type == 'euclidean':
        #     self._dist_func = eucl_dist
        # elif dist_type == 'manhattan':
        #     self._dist_func = man_dist
        # else:
        #     raise ValueError('Invalid dist_type')

    def calc_reward(self):
        world = self._world
        if self._world.landmark_just_found:
            return self._lm_reached_reward
        else:
            lm_pos = world.landmark.pos
            robot_pos = world.robot.pos
            dist = np.linalg.norm(robot_pos - lm_pos)
            exp = np.exp(-dist)
            reward = self._a * (-1/exp + 1)
            return reward

        # # Distance from outside of entities - distance should not be negative (i.e. if entities overlapping)
        # dist = max(self._dist_func(lm_pos, robot_pos) - world.robot.size - world.landmark.size, 0)
        # reward = -1 * self._a * dist
        # if self._world.landmark_just_found:
        #    reward += self._lm_reached_reward
        #
        # return reward
