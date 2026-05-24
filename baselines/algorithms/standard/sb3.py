import os
from typing import Any, Dict, Optional, List, SupportsFloat, Tuple

import numpy as np
import torch

from morality_gym._setup.setup import make as env_mc_make

from baselines.algorithms.learner import BaseLearner
from baselines.logger import Logger
from baselines.wrappers.sb3 import SB3Wrapper

from stable_baselines3.common.env_util import make_vec_env


class SB3Learner(BaseLearner):
    def __init__(
            self,
            env_id: str,
            mc_id: str,
            logger: Logger,
            seed: int,

            learn_mm_kwargs: Dict[str, Any],
            final_mm_kwargs: Dict[str, Any],

            env_overrides: Optional[Dict[str, Any]] = None,
            mc_overrides: Optional[Dict[str, Any]] = None,

            # alg_kwargs: Optional[Dict[str, Any]] = None,  # Remove?
            eval_metrics: Optional[List[str]] = None,

            vis_episode_kwargs: Optional[Dict[str, Any]] = None,

            deterministic_eval: bool = False,

            # Env Wrapper
            shape_reward: bool = False,
            norm_reward: bool = False,
            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
            cost_function_kwargs: Dict[str, Any] = None,
            # salient_norms: Optional[List] = None,
            n_torch_threads: int = 4,

            # -- SB3 KWARGS -- #
            n_vec_envs: int = 1,
            device: str = "cpu",
            policy: str = "MlpPolicy",
            model_type: str = "PPO",
            sb3_verbosity: int = 0,
            log_tensorboard: bool = False,
            sb3_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ###############
        # --- ENV --- #
        ###############
        self.shape_reward = shape_reward
        self.norm_reward = norm_reward

        def env_wrapper_fn(env):
            return SB3Wrapper(
                env=env, morality_chain=self.morality_chain, cost_function_kwargs=cost_function_kwargs,
                shape_reward=self.shape_reward, norm_reward=self.norm_reward, return_bounds=return_bounds
            )
        self.env_wrapper_fn = env_wrapper_fn

        def make_env_fn():
            env, _ = env_mc_make(env_id, mc_id, self.env_overrides, self.mc_overrides)
            # return env
            return env_wrapper_fn(env)

        # WARNING: env_wrapper_fn & make_env_fn can only be called after super().__init__(...) since
        # they use properties set in that init

        self.make_env_fn = make_env_fn

        self._n_vec_envs = n_vec_envs
        ###############

        ######################
        # --- SUPER INIT --- #
        ######################
        super().__init__(
            env_id=env_id, mc_id=mc_id,
            logger=logger,
            seed=seed,
            eval_env_wrapper_fn=self.env_wrapper_fn,  # New
            learn_mm_kwargs=learn_mm_kwargs, final_mm_kwargs=final_mm_kwargs,
            env_overrides=env_overrides, mc_overrides=mc_overrides,
            # alg_kwargs=alg_kwargs,
            eval_metrics=eval_metrics,
            vis_episode_kwargs=vis_episode_kwargs,
            return_bounds=return_bounds
        )
        ######################

        ###########################
        # --- CREATE VEC ENVS --- #
        ###########################
        # Note: This must be called after super().__init__(...) because env_wrapper_fn and make_env_fn can only be called
        # after super().__init__(...). Kinda janky but it works
        self._vec_env = make_vec_env(env_id=make_env_fn, n_envs=n_vec_envs, seed=seed)
        ###########################

        ######################
        # GENERAL PROPERTIES #
        ######################
        self.deterministic_eval = deterministic_eval
        self.learn_step = 0
        ######################

        #################
        # -- LOGGING -- #
        #################
        if log_tensorboard:
            self.logger.create_sub_folder("tensorboard")
            self.tensorboard_path = self.logger.sub_folder_path("tensorboard")
        else:
            self.tensorboard_path = None
        #################

        ###################
        # -- SB3 MODEL -- #
        ###################
        self._model_type = model_type
        if self._model_type == "PPO":
            from stable_baselines3 import PPO
            self._model_class = PPO
            if device != "cpu":
                print(f"WARNING: It is advised to run PPO on CPU and not GPU.")
        elif self._model_type == "DQN":
            from stable_baselines3 import DQN
            self._model_class = DQN
        else:
            raise ValueError(f"Invalid model type: {self._model_type}")

        if sb3_kwargs is None:
            sb3_kwargs = {}

        self.sb3_model = self._model_class(
            policy=policy,
            env=self._vec_env,
            verbose=sb3_verbosity,
            seed=seed,
            device=device,
            tensorboard_log=self.tensorboard_path,
            **sb3_kwargs
        )
        ###################
        # Specify no threads
        os.environ["OMP_NUM_THREADS"] = str(n_torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_torch_threads)
        torch.set_num_threads(n_torch_threads)

    def learn(
            self,
            n_timesteps: int,
            eval_freq: int,
            vis_eval_episodes: bool = False,
            save_models: bool = False,
            prog_bar: bool = True,
    ):
        def _callback(sb3_locals, sb3_globals):
            if self.learn_step % eval_freq == 0:
                self.evaluate("learn", log_metrics=True, vis_episode=vis_eval_episodes, save_model=save_models)
            self.learn_step += self._n_vec_envs
            return True

        self.sb3_model.learn(total_timesteps=n_timesteps, callback=_callback, progress_bar=prog_bar)

        self.evaluate("final", log_metrics=True, vis_episode=vis_eval_episodes, save_model=save_models)


    # def eval_policy(self, obs: Any) -> Any:
    #     pass

    def eval_policy(self, obs: np.ndarray) -> int:
        obs = obs.reshape(1, -1)  # Needs extra dimension due to vectorised envs
        with torch.no_grad():
            action = self.sb3_model.policy.predict(obs, deterministic=self.deterministic_eval)
        return int(action[0][0])

    def save_model(self):
        raise NotImplementedError

    ####################
    # -- PROPERTIES -- #
    ####################
    @property
    def curr_step(self):
        return self.learn_step

    @property
    def curr_episode(self):
        return 0
    ####################

    def finish(self):
        self._vec_env.close()
        self.eval_env.close()
