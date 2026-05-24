import os
from typing import Any, Dict, Optional, List, SupportsFloat, Tuple

import numpy as np

import omnisafe
import torch

from baselines.wrappers.cmdp import CMDPEnv
from morality_gym._setup.setup import make

from baselines.algorithms.learner import BaseLearner
from baselines.logger import Logger
from baselines.wrappers.sb3 import SB3Wrapper

from morality_gym._utils.common import copy_to_dict_recr


class OSLearner(BaseLearner):
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
            norm_reward: bool = False,
            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
            cost_function_kwargs: Dict[str, Any] = None,

            # -- OS KWARGS -- #
            algo_name: str = "PPOLag",
            os_custom_cfgs: Optional[Dict[str, Any]] = None,
            device: str = "cpu",
            # n_parallel: int = 8,
            n_vec_env: int = 8,
            n_torch_threads: int = 8,

            n_timesteps: int = 100000,
            log_tensorboard: bool = False,
            save_model_freq: int = 1000000,

            # is_lagrange: bool = True,

            # cost_bounds: Optional[Tuple[float, float]] = None,
            # cost_limit: Optional[float] = None,
            # cost_limit_perc: Optional[float] = None,

            # n_vec_envs: int = 1,
            # device: str = "cpu",
            # policy: str = "MlpPolicy",
            # model_type: str = "PPO",
            # sb3_verbosity: int = 0,
            # log_tensorboard: bool = False,
            # sb3_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ##################
        # -- EVAL ENV -- #
        ##################
        self.norm_reward = norm_reward
        def env_wrapper_fn(env):
            return SB3Wrapper(
                env=env, morality_chain=self.morality_chain, cost_function_kwargs=cost_function_kwargs,
                shape_reward=False, norm_reward=self.norm_reward, return_bounds=return_bounds
            )

        self.eval_env_wrapper_fn = env_wrapper_fn

        ###############

        ######################
        # --- SUPER INIT --- #
        ######################
        super().__init__(
            env_id=env_id, mc_id=mc_id,
            logger=logger,
            seed=seed,
            eval_env_wrapper_fn=self.eval_env_wrapper_fn,  # New
            learn_mm_kwargs=learn_mm_kwargs, final_mm_kwargs=final_mm_kwargs,
            env_overrides=env_overrides, mc_overrides=mc_overrides,
            # alg_kwargs=alg_kwargs,
            eval_metrics=eval_metrics,
            vis_episode_kwargs=vis_episode_kwargs,
            return_bounds=return_bounds
        )
        ######################

        ############################
        # --- CREATE CMDP FUNC --- #
        ############################
        def make_env_fn(is_eval = False):
            # NOTE: Nothing is done currently with _env_id, _num_envs & _seed
            # _seed is specified in env
            _env, _mc = make(
                env_id=env_id,
                morality_chain_id=mc_id,
                env_kwargs=env_overrides,
                morality_chain_kwargs=mc_overrides,
            )

            if is_eval:
                _curr_n_vec_env = 1
            else:
                if algo_name == "PPOShaped":
                    _env = SB3Wrapper(
                        env=_env, morality_chain=self.morality_chain,
                        cost_function_kwargs=cost_function_kwargs,
                        shape_reward=True,
                        norm_reward=self.norm_reward,
                        return_bounds=return_bounds
                    )

                _curr_n_vec_env = n_vec_env
            return CMDPEnv(
                env=_env, morality_chain=_mc,
                return_bounds=return_bounds,
                norm_reward=True,
                cost_function_kwargs=cost_function_kwargs,
                device=device,
                num_envs = _curr_n_vec_env
            )

        self._n_vec_env = n_vec_env
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
        self.logger.create_sub_folder("os")
        #################

        ######################
        # ------ EVAL ------ #
        ######################
        self.learn_step = 0
        # _eval_freq & _vis_eval_episode will be set during learn
        self._eval_freq = None
        self._vis_eval_episode = False

        def _step_callback():
            if self._eval_freq is not None and self.learn_step % self._eval_freq == 0:
                print(f"\n>>> ------------------ <<<")
                print(f">>> STARTED EVALUATION <<<")
                print(f">>> ------------------ <<<")
                print(f"\t Step = {self.learn_step}")
                morality_metric, morality_functions, info = self.evaluate(
                    mode="learn",
                    log_metrics=True,
                    vis_episode=self._vis_eval_episode,
                    save_model=False
                )
                print(f"\t Morality Metric = {np.round(morality_metric, 3)}")
                print(f"\t Morality Functions:")
                for key, val in morality_functions.items():
                    print(f"\t\t {key} = {np.round(val, 3)}")
                print(f">>> ------------------- <<<")
                print(f">>> FINISHED EVALUATION <<<")
                print(f">>> ------------------- <<<\n")
            self.learn_step += self._n_vec_env
        ######################

        ######################
        # ---- OS AGENT ---- #
        ######################
        self._n_timesteps = n_timesteps
        if seed in os_custom_cfgs:
            raise ValueError("seed must not be in os_custom_cfgs.")
        if "train_cfgs" in os_custom_cfgs:
            raise ValueError("train_cfgs must not be in os_custom_cfgs.")
        if "logger_cfgs" in os_custom_cfgs:
            raise ValueError("logger_cfgs must not be in os_custom_cfgs.")

        self._n_torch_threads = n_torch_threads

        # ---- Set Cfgs ---- #
        overrides_cfg = {
            "seed": seed,
            "train_cfgs": {
                "device": device,
                "torch_threads": n_torch_threads,
                "vector_env_nums": n_vec_env,
                "parallel": 1,
                "total_steps": n_timesteps
            },
            "algo_cfgs": {
                "obs_normalize": False
            },
            "model_cfgs": {
                "actor_type": "discrete"
            },
            "logger_cfgs": {
                "use_wandb": False,
                "use_tensorboard": log_tensorboard,
                "save_model_freq": save_model_freq,
                "log_dir": self.logger.sub_folder_path("os")
            },
            "step_callback": _step_callback,
            "make_env_fn": make_env_fn
        }
        # if is_lagrange:
        #     if cost_limit is None:
        #         raise NotImplementedError
        #     overrides_cfg["lagrange_cfgs"] = {
        #         "cost_limit": cost_limit,
        #     }

        copy_to_dict_recr(from_dict=overrides_cfg, to_dict=os_custom_cfgs)

        # Reward shaping is handled by SB3Wrapper defined above
        if algo_name == "PPOShaped":
            os_algo_name = "PPO"
        else:
            os_algo_name = algo_name

        # -- Create Agent -- #
        self._os_agent = omnisafe.Agent(
            algo=os_algo_name,
            env_id=f"{env_id}_{mc_id}",
            custom_cfgs=os_custom_cfgs,
            # make_env_fn=make_env_fn
        )

        os.environ["OMP_NUM_THREADS"] = str(n_vec_env)
        os.environ["MKL_NUM_THREADS"] = str(n_vec_env)
        torch.set_num_threads(n_vec_env)
        ######################


    def learn(
            self,
            n_timesteps: int,
            eval_freq: int,
            vis_eval_episodes: bool = False,
            save_models: bool = False,
            prog_bar: bool = True,
    ):
        if n_timesteps != self._n_timesteps:
            raise ValueError(f"n_timesteps must be {self._n_timesteps} but is {n_timesteps}. n_timesteps is set at init for OSLearner.")

        self._eval_freq = eval_freq
        self._vis_eval_episode = vis_eval_episodes

        torch.set_num_threads(self._n_torch_threads)  # Probably not needed but omnisafe is being weird with no of threads used
        self._os_agent.learn()

        # self.sb3_model.learn(total_timesteps=n_timesteps, callback=_callback, progress_bar=prog_bar)

        self.evaluate("final", log_metrics=True, vis_episode=vis_eval_episodes, save_model=save_models)


    # def eval_policy(self, obs: Any) -> Any:
    #     pass

    def eval_policy(self, obs: np.ndarray) -> int:
        # obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_tensor = self._os_agent.agent._actor_critic.actor.predict(obs_tensor, deterministic=False)

        return int(action_tensor.item())
        # obs = obs.reshape(1, -1)  # Needs extra dimension due to vectorised envs
        # with torch.no_grad():
        #     action = self.sb3_model.policy.predict(obs, deterministic=self.deterministic_eval)
        # return int(action[0][0])

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
        # This is essential to avoid ghost threads & deadlocks
        env_adapter = self._os_agent.agent._env
        env_adapter._env.close()

        self.eval_env.close()
        self._os_agent.agent.logger.close()

        del env_adapter._env  # Extreme measures to stop whatever keeps making the threads hang

        # env = agent._env.close()
        # pass


def main():
    # remove_obs = None
    # remove_obs = {
    #     "character_*": ["pos", "amount", "character_type", "is_harmed"],
    #     "trolley_*": ["is_harmed", "is_actable"],
    #     "player": ["is_harmed", "is_terminated"]
    # }
    remove_obs = {
        "character_*": ["amount", "character_type"],
        "trolley_*": ["is_harmed", "is_actable"],
        "player": ["is_harmed"]
    }

    env = "SwitchStandard-Human-v1"
    mc = "Utility"

    logger = Logger(
        name=f"{env}_{mc}",
        log_dir=f"{env}_{mc}",
        config={"name": "test"},
        project="test",
        group="test",
        job_type="test",
        tags=["test"],
        mode="local",
        reinit=True
    )

    env_overrides = {
        "env_overrides": {
            "render_mode": "human",
            "remove_obs": remove_obs
        },
    }
    vis_episode_kwargs = {
        "max_steps": 50
    }

    mc_overrides = {
        "beta": 0.1
    }

    cost_function_kwargs = {
        "scale_fact": 10.0,
    }

    os_custom_cfgs = {

    }

    n_timesteps = 50000

    learner = OSLearner(
        env_id=env, mc_id=mc,
        logger=logger,
        seed=42,
        learn_mm_kwargs={},
        final_mm_kwargs={},
        env_overrides=env_overrides,
        mc_overrides=mc_overrides,
        norm_reward=True,
        return_bounds=(-300, 100),
        cost_function_kwargs=cost_function_kwargs,
        n_timesteps=n_timesteps,
        log_tensorboard=False,
        algo_name="PPOLag",
        os_custom_cfgs=os_custom_cfgs

    )
    learner.learn(n_timesteps, eval_freq=5000, vis_eval_episodes=False, save_models=False)


    logger.finish()


if __name__ == "__main__":
    main()
