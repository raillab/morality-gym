from typing import Any, ClassVar, Union, Optional, Dict, Tuple, SupportsFloat

import gymnasium
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

import omnisafe
from baselines.wrappers.sb3 import SB3Wrapper
from morality_gym.morality_chain.morality_chain import MoralityChain
from omnisafe.envs import CMDP, env_register


@env_register
class CMDPEnv(CMDP):
    _support_envs: ClassVar[list[str]] = ["CMDPEnvDummyID"]
    need_action_scale_wrapper = False
    need_obs_normalize_wrapper = False
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    def __init__(
            self,
            env,
            morality_chain: MoralityChain,
            # env_id: str,
            norm_reward: bool = False,
            cost_function_kwargs: Dict[str, Any] = None,
            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
            # CMDP stuff
            num_envs: int = 1,
            device: str = 'cpu',
            seed: int = None,
            max_episode_steps: int = np.inf,
    ):
        super().__init__(env_id="CMDPEnvDummyID", num_envs=num_envs)

        def _make_env_fn():
            return SB3Wrapper(
                env=env,
                morality_chain=morality_chain,
                norm_reward=norm_reward,
                shape_reward=False,
                cost_function_kwargs=cost_function_kwargs,
                return_bounds=return_bounds,
            )

        self._base_env = _make_env_fn()

        # Nothing is done with this
        self._num_envs = num_envs
        self._seed = seed

        # Needed for tensor conversion
        self._device = device

        # --- SPACES --- #
        if not isinstance(self._base_env.action_space, gymnasium.spaces.Discrete):
            raise ValueError("Action space must be Discrete.")
        self._action_space = self._base_env.action_space

        if not isinstance(self._base_env.observation_space, gymnasium.spaces.Box):
            raise ValueError("Observation space must be Box.")
        self._observation_space = self._base_env.observation_space
        # -------------- #

        self.max_episode_steps = max_episode_steps

        #######################
        # -- VEC ENV STUFF -- #
        #######################
        self._is_vec = num_envs > 1
        if self._is_vec:
            self._vec_env = AsyncVectorEnv(
                env_fns = [lambda: _make_env_fn() for _ in range(num_envs)]
                # autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
            )
        else:
            self._vec_env = None

        #######################

    ################
    # --- STEP --- #
    ################
    def step(self, action: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        if self._is_vec:
            return self._step_vec(action)
        else:
            return self._step_non_vec(action)

    # Non Vec #
    def _step_non_vec(self, action: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        action = action.item()
        obs, reward, is_term, is_trunc, info = self._base_env.step(action)

        if "cost" not in info:
            raise ValueError("Cost not found in info.")

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        cost_tensor = torch.as_tensor(info["cost"], dtype=torch.float32, device=self._device)
        is_term_tensor = torch.as_tensor(is_term, dtype=torch.bool, device=self._device)
        is_trunc_tensor = torch.as_tensor(is_trunc, dtype=torch.bool, device=self._device)

        if is_term or is_trunc:
            new_obs, new_info = self.reset()
            # assert (
            #     'final_observation' not in new_info
            # ), 'info dict cannot contain key "final_observation" '
            # assert 'final_info' not in new_info, 'info dict cannot contain key "final_info" '

            new_info['final_observation'] = obs
            new_info['final_info'] = info

            obs_tensor = new_obs
            info = new_info

        # if is_term or is_trunc:
        #     info["final_observation"] = obs_tensor

        return obs_tensor, reward_tensor, cost_tensor, is_term_tensor, is_trunc_tensor, info

    # Vec #
    def _step_vec(self, action: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        # N.B. AsyncVectorEnv auto resets environments

        action = torch.flatten(action).tolist()
        obs, reward, is_term, is_trunc, info = self._vec_env.step(action)

        if "cost" not in info:
            info["cost"] = [None for _ in range(self._num_envs)]

        if "final_observation" in info:
            final_obs = [
                torch.as_tensor(_curr_obs, dtype=torch.float32, device=self._device)
                if _curr_obs is not None else None
                for _curr_obs
                in info["final_observation"]
            ]
            info["final_observation"] = final_obs

            # For envs that were reset add cost back to info
            if "_cost" not in info:
                info["_cost"] = [False for _ in range(self._num_envs)]
            for i in range(self._num_envs):
                if not info["_cost"][i]:
                    info["cost"][i] = info["final_info"][i]["cost"]


        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        cost_tensor = torch.as_tensor(info["cost"], dtype=torch.float32, device=self._device)
        is_term_tensor = torch.as_tensor(is_term, dtype=torch.bool, device=self._device)
        is_trunc_tensor = torch.as_tensor(is_trunc, dtype=torch.bool, device=self._device)

        return obs_tensor, reward_tensor, cost_tensor, is_term_tensor, is_trunc_tensor, info

    ################

    #################
    # --- RESET --- #
    #################
    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._is_vec:
            return self._reset_vec(seed=seed, options=options)
        else:
            return self._reset_non_vec(seed=seed, options=options)

    # Non Vec #
    def _reset_non_vec(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        obs, info = self._base_env.reset(seed=seed, options=options)
        if not isinstance(obs, np.ndarray):
            raise TypeError(f"Expected numpy array for obs, got {type(obs)}")
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        # obs = torch.from_numpy(obs).to(self._device)

        return obs, info

    def _reset_vec(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        obs, info = self._vec_env.reset(seed=seed, options=options)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)

        return obs, info
    #################

    def set_seed(self, seed: int) -> None:
        # Does this need to do anything?
        pass

    def sample_action(self) -> torch.Tensor:
        action = self.action_space.sample()
        action = torch.tensor(action, device=self._device)
        return action

    def render(self) -> Any:
        self._base_env.render()

    def close(self) -> None:
        self._base_env.close()
        self._vec_env.close()


def main():
    from morality_gym.setup.setup import make  # as env_mt_make
    env_kwargs = {
        "scenario_overrides": {
            "seed": 42,
            "max_timesteps": 200
        },
        "env_overrides": {
            "obs_type": np.ndarray,
            "is_normalise_obs": True,
            "render_mode": "human",
            # "step_penalty": 0
        }
    }


    _env_id = 'SwitchStandard-Human-v1'

    return_bounds = (-300, 100)
    # env = SB3Wrapper(
    #     env, mt, shape_reward=True,
    #     cost_function_kwargs=cost_function_kwargs
    # )
    # cmdp_env = CMDPEnv(
    #     env=env, morality_chain=mc,
    #     return_bounds=return_bounds,
    #     norm_reward=True,
    #     cost_function_kwargs=cost_function_kwargs,
    #     device="cpu",
    # )

    def make_env_fn(
            env_id,
            num_envs,
            device,
            seed
    ):
        # NOTE: Nothing is done currently with _env_id, _num_envs & _seed
        # _seed is specified in env
        env, mc = make(
            # env_id='PushSelfSacrifice-All-v1',
            env_id=_env_id,
            # env_id='MoralityGym/Trolley-PushStandard-0-v0',
            # morality_chain_id='Trolley-Common-Utilitarian-UtilityHarm-v0',
            morality_chain_id='Utility',
            env_kwargs=env_kwargs
        )
        cost_function_kwargs = {
            "scale_fact": 10.0,
        }

        return CMDPEnv(
            env=env, morality_chain=mc,
            return_bounds=return_bounds,
            norm_reward=True,
            cost_function_kwargs=cost_function_kwargs,
            device=device,
        )

    algo_name = "PPOLag"
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 500000,
            'vector_env_nums': 1,
            'parallel': 1,
        },
        "model_cfgs": {
            "actor_type": "discrete"
        },
        'algo_cfgs': {
            'steps_per_epoch': 5000,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
        },
        "lagrange_cfgs": {
            "cost_limit": 5.0
        }
    }

    agent = omnisafe.Agent(algo=algo_name, env_id="Dummy", custom_cfgs=custom_cfgs, make_env_fn=make_env_fn)
    agent.learn()
    # print(cmdp_env.reset())



if __name__ == "__main__":
    main()
