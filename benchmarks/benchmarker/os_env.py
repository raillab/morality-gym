# from morality_gym.environment.utils import OmniSafeMoralityEnv
# from omnisafe.envs.core import env_register
# import omnisafe
#
# from typing import Optional
#
# import torch
# import numpy as np
# import gymnasium as gym
# from gymnasium.spaces import Box, Dict
# from omnisafe.envs.core import CMDP
#
# from morality_gym.environment.cost import MoralityCost
# from morality_gym.environment.env import env_from_config
# from morality_gym.morality_tree.morality_tree import MoralityTree
#
#
# @env_register
# class OmniSafeMoralityEnv(CMDP):
#     _support_envs = ['Morality-v0']  # Add more envs if needed
#     need_auto_reset_wrapper = True
#     need_time_limit_wrapper = True
#     need_evaluation = True
#
#     def __init__(
#             self,
#             env_id: str,  # This is required by omnisafe to create it
#             env_config: str,
#             morality_tree_config: str,
#             env_overrides: Optional[Dict] = None,
#             num_envs: int = 1,
#             max_episode_steps: int = 1024,
#             time_limit: int = 1024,
#             device: str = "cpu",
#     ) -> None:
#         self._env_id = env_id
#         self._num_envs = num_envs
#         self._max_episode_steps = max_episode_steps
#         self._time_limit = time_limit
#         self._device = device
#         # self._env_config = env_config
#
#         self._env = env_from_config(
#             env_config,
#             overrides=env_overrides
#         )
#
#         self._morality_tree = MoralityTree(morality_tree_config)
#         self._morality_cost = MoralityCost(self._morality_tree)
#
#         self._observation_space = self._env.observation_space
#         self._action_space = self._env.action_space
#
#     def set_seed(self, seed: int) -> None:
#         self._env._rng = np.random.default_rng(seed)
#
#     def reset(self, seed: int = None, options: dict = None):
#         if seed is not None:
#             self.set_seed(seed)
#         obs, info = self._env.reset(seed=seed, options=options)
#         return self._process_obs(obs), info
#
#     def step(self, action: torch.Tensor):
#         if isinstance(self._env.action_space, gym.spaces.Discrete):
#             action_np = action.item()
#         else:
#             action_np = action.detach().numpy()
#
#         obs, reward, terminated, truncated, info = self._env.step(action_np)
#
#         cost = self._morality_cost.calc(info)
#
#         return (
#             self._process_obs(obs),
#             torch.tensor(reward, dtype=torch.float32),
#             torch.tensor(cost, dtype=torch.float32),
#             torch.tensor(terminated, dtype=torch.bool),
#             torch.tensor(truncated, dtype=torch.bool),
#             info
#         )
#
#     def _process_obs(self, obs):
#         """Flatten dictionary observations into single vector."""
#         if isinstance(obs, dict):
#             return torch.tensor(
#                 np.concatenate([v.flatten() for v in obs.values()]),
#                 dtype=torch.float32
#             )
#         return torch.tensor(obs, dtype=torch.float32)
#
#     @property
#     def max_episode_steps(self) -> int:
#         return self._max_episode_steps
#
#     def render(self):
#         return self._env.render()
#
#     def close(self):
#         self._env.close()
#
#
# # @env_register
# # class CustomMoralityEnv(OmniSafeMoralityEnv):
# #
# #     def __init__(
# #             self,
# #             *args,
# #             **kwargs
# #     ) -> None:
# #         super(CustomMoralityEnv, self).__init__(
# #             *args, **kwargs
# #         )