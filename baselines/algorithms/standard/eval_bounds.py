from typing import Any, Dict, Optional, List, SupportsFloat, Tuple

import numpy as np
from tqdm import tqdm

from baselines.algorithms.learner import BaseLearner
from baselines.logger import Logger
from baselines.wrappers.sb3 import SB3Wrapper

from morality_gym.setup.setup import make as env_mc_make


# Kinda hacky way to evaluate bounds without modifying too much code - but works
class EvalBoundsLearner(BaseLearner):
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
            eval_metrics=eval_metrics,
            vis_episode_kwargs=vis_episode_kwargs,
            return_bounds=return_bounds
        )
        ######################


        self._curr_step = 0
        self._curr_episode = 0

        #######################
        # CREATE AGR EVAL CSV #
        #######################
        eval_metrics = self.logger.metrics["eval"]
        eval_metrics.append("agr")
        self.logger.define_metrics(eval_metrics, "agr_eval")
        #######################


    def eval_policy(self, obs: Any) -> Any:
        return self.eval_env.action_space.sample()

    #########################
    # -- EVALUATE BOUNDS -- #
    #########################
    def eval_bounds(
            self,
            n_repeats: int
    ):
        self._curr_step = 0
        self._curr_episode = 0

        min_mm = np.inf
        max_mm = -np.inf

        # morality functions corresponding to min/max morality metrics
        min_mfs = None
        max_mfs = None

        # info corresponding to min/max morality metrics
        min_info = None
        max_info = None

        for i in tqdm(range(n_repeats), desc="Evaluating..."):
            morality_metric, morality_functions, info = self.evaluate("learn")
            if morality_metric is not None and morality_metric < min_mm:
                min_mm = morality_metric
                min_mfs = morality_functions
                min_info = info

            if morality_metric is not None and morality_metric > max_mm:
                max_mm = morality_metric
                max_mfs = morality_functions
                max_info = info

            self._curr_step += 1

        ############################
        # --- AGR EVAL METRICS --- #
        ############################
        # MIN MORALITY METRIC #
        log_d = {
            "curr_step": 0,
            "curr_episode": 0,
            "morality_metric": min_mm,
            "avg_return": min_info["avg_return"],
            "avg_steps": min_info["avg_steps"],
            "avg_orig_return": min_info["orig_reward"],
            "avg_cost": min_info["cost"],
            "agr": "min"
        }
        for mf_name, mf_val in min_mfs.items():
            log_d[f"mf_{mf_name}"] = mf_val

        self.logger.log(log_d, "agr_eval", _check_keys=True)

        # MAX MORALITY METRIC #
        log_d = {
            "curr_step": 0,
            "curr_episode": 0,
            "morality_metric": max_mm,
            "avg_return": max_info["avg_return"],
            "avg_steps": max_info["avg_steps"],
            "avg_orig_return": max_info["orig_reward"],
            "avg_cost": max_info["cost"],
            "agr": "max"
        }
        for mf_name, mf_val in max_mfs.items():
            log_d[f"mf_{mf_name}"] = mf_val

        self.logger.log(log_d, "agr_eval", _check_keys=True)
        ############################

    # Hacky way to conform to existing interface
    def learn(
            self,
            n_timesteps: int,
            eval_freq: int,
            vis_eval_episodes: bool = False,
            save_models: bool = False
    ):
        self.eval_bounds(n_repeats=n_timesteps)
    #########################

    def save_model(self):
        pass

    @property
    def curr_step(self):
        return self._curr_step

    @property
    def curr_episode(self):
        return self._curr_episode