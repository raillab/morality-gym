import copy
from abc import abstractmethod
from typing import Optional, Dict, Any, Tuple, List, SupportsFloat, Callable
from baselines.logger import Logger
from morality_gym.utils.common import copy_to_dict, join_paths

from morality_gym.setup.setup import make as env_mc_make


class BaseLearner:
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

            eval_env_wrapper_fn: Callable = None,
            eval_metrics: Optional[List[str]] = None,

            vis_episode_kwargs: Optional[Dict[str, Any]] = None,

            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
    ):
        self.logger = logger
        self.env_id = env_id
        self.mc_id = mc_id
        self.env_overrides = copy.deepcopy(env_overrides)  # Copy since env_overrides is later modified
        self.mc_overrides = copy.deepcopy(mc_overrides)
        self.seed = seed

        if vis_episode_kwargs is None:
            vis_episode_kwargs = {}
        self.vis_episode_kwargs = vis_episode_kwargs

        ##############################
        # --- SEED ENV OVERRIDES --- #
        #       Override seed
        ##############################
        # TODO: Check this is correct :)
        if self.env_overrides is None:
            self.env_overrides = {}
        if "scenario_overrides" not in self.env_overrides:
            self.env_overrides["scenario_overrides"] = {}
        self.env_overrides["scenario_overrides"]["seed"] = seed

        ###################################
        # -- EVAL ENV & MORALITY CHAIN -- #
        ###################################
        self.eval_env, self.morality_chain = env_mc_make(env_id, mc_id, self.env_overrides, mc_overrides)
        if eval_env_wrapper_fn is not None:
            self.eval_env = eval_env_wrapper_fn(self.eval_env)
        ###################################

        ##############################
        # -- MORALITY METRIC (MM) -- #
        ##############################
        self.learn_mm_kwargs = learn_mm_kwargs
        self.final_mm_kwargs = final_mm_kwargs

        if eval_metrics is None:
            eval_metrics = []

        default_eval_metrics = [
            "curr_step", "curr_episode", "morality_metric", "avg_return", "avg_steps", "avg_orig_return", "avg_cost",
            "avg_goal_reached"
        ]
        default_eval_metrics = default_eval_metrics + [f"mf_{norm}" for norm in self.morality_chain.norm_names]
        eval_metrics = set(eval_metrics).union(set(default_eval_metrics))
        eval_metrics = list(eval_metrics)

        self.logger.define_metrics(eval_metrics, "eval")
        ##############################

    ##################
    # -- EVALUATE -- #
    ##################
    def evaluate(
            self,
            mode: str,
            log_metrics: bool = True,
            vis_episode: bool = False,  # If true will also save video
            # vis_kwargs: Optional[Dict[str, Any]] = None,
            save_model: bool = False,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        if mode == "learn":
            mm_kwargs = self.learn_mm_kwargs
        elif mode == "final":
            mm_kwargs = self.final_mm_kwargs
        else:
            raise ValueError(f"Invalid mode: {mode}")

        agr_info_keys = ["orig_reward", "cost", "goal_reached"]

        morality_metric, morality_functions, info = self.morality_chain.eval_morality_metric(
            policy=self.eval_policy, env=self.eval_env, agr_info_keys=agr_info_keys, **mm_kwargs
        )
        for k in agr_info_keys:
            if k not in info:
                info[k] = None

        if log_metrics:
            log_d = {
                "curr_step": self.curr_step,
                "curr_episode": self.curr_episode,
                "morality_metric": morality_metric,
                "avg_return": info["avg_return"],
                "avg_steps": info["avg_steps"],
                "avg_orig_return": info["orig_reward"],
                "avg_cost": info["cost"],
                "avg_goal_reached": info["goal_reached"],
            }
            for mf_name, mf_val in morality_functions.items():
                log_d[f"mf_{mf_name}"] = mf_val

            self.logger.log(log_d, "eval", _check_keys=True)

        if vis_episode:
            # if vis_kwargs is None:
            #     vis_kwargs = {}
            vis_path = join_paths([self.logger.log_path, "videos", f"step_{self.curr_step:06d}.gif"])
            self.vis_episode(vis_path, **self.vis_episode_kwargs)

        if save_model:
            raise NotImplementedError("Saving model not yet implemented")

        return morality_metric, morality_functions, info

    @abstractmethod
    def eval_policy(self, obs: Any) -> Any:
        raise NotImplementedError("Learner must implement eval_policy method")
    ##################

    #################
    # --- LEARN --- #
    #################
    @abstractmethod
    def learn(
            self,
            n_timesteps: int,
            eval_freq: int,
            vis_eval_episodes: bool = False,
            save_models: bool = False,
    ):
        raise NotImplementedError("Learner must implement learn method")
    #################

    ###########################
    # -- VISUALISE EPISODE -- #
    ###########################
    def vis_episode(
            self,
            save_path: str,
            reset_seed: Optional[int] = None,
            reset_options: Optional[Dict[str, Any]] = None,
            max_steps: int = 100,
            fps: int = 30,
    ):
        try:
            import imageio
        except ImportError:
            raise ImportError("Cannot visualise episode, imageio not installed")

        from PIL import GifImagePlugin
        GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_ALWAYS

        env = self.eval_env
        policy = self.eval_policy

        # if env.render_mode != "rgb_array":
        #     raise ValueError(f"Cannot visualise episode, env render mode must be 'rgb_array', not {self.eval_env.render_mode}")

        is_done = False
        curr_step = 0
        frames_arr = []

        obs, _ = env.reset(reset_seed, reset_options)
        frames_arr.append(env.render())
        while not is_done and curr_step < max_steps:
            curr_step += 1
            action = policy(obs)
            obs, _, is_term, is_trunc, _ = env.step(action)
            frames_arr.append(env.render())
            is_done = is_term or is_trunc

        # imageio.mimsave(save_path, frames_arr, fps=fps)
        imageio.mimwrite(save_path, frames_arr, format="GIF", duration=1/fps, loop=0, version="GIF89a")
    ###########################

    ################
    # --- MISC --- #
    ################
    @abstractmethod
    def save_model(self):
        raise NotImplementedError("Learner must implement save_model method")
    ################

    ##################
    # -- PROPERTY -- #
    ##################
    # Note: Not sure if property and abstractmethod decorators can be combined
    @property
    @abstractmethod
    def curr_step(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def curr_episode(self):
        raise NotImplementedError
    ##################

    ##################
    # --- FINISH --- #
    ##################
    def finish(self):
        raise NotImplementedError("Learner must implement finish method")
    ##################