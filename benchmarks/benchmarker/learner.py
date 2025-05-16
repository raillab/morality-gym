import os
import sys
from typing import Callable, Dict, Any, Optional

# from torch import cuda
import torch

sys.path.append('..')

import omnisafe

from morality_gym.environment.env.cmdp_env import OmniSafeMoralityEnv
from morality_gym.environment.env.morality_env import env_from_config
from morality_gym.morality_tree.morality_metric import evaluate_morality_metric
from morality_gym.morality_tree.morality_tree import MoralityTree
from morality_gym.utils.common import copy_to_dict, join_paths, create_or_clear_folder
from omnisafe.common import Logger
from omnisafe.envs import env_register


# class Learner:
#     def __init__(
#             self,
#             env_fn: Callable,
#             eval_env_fn: Callable,
#
#     ):
#         pass

@env_register
class CustomMoralityEnv(OmniSafeMoralityEnv):

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        super(CustomMoralityEnv, self).__init__(
            *args, **kwargs
        )


class SafeLearner:
    VALID_ALGO_NAMES = ["PPOLag", "RCPO","CPO", "TRPO", "PPOSaute", "PPO"]  # TODO: Add in others
    def __init__(
            self,
            seed: int,
            algo_name: str,
            algo_kwargs: Dict[str, Any],
            model_kwargs: Dict[str, Any],
            train_kwargs: Dict[str, Any],
            logger_kwargs: Dict[str, Any],
            env_config: str,
            morality_tree_config: str,
            # Eval
            n_eval_episodes: int = 20,
            max_eval_steps: int = 1000,
            vis_eval_episode: bool = False,
            vis_eval_spf: float = 0.1,
            vis_eval_max_steps: int = 100,
            # Env
            env_overrides: Optional[Dict[str, Any]] = None,
            cost_kwargs: Optional[Dict[str, Any]] = None,
            env_time_limit: int = 1024,
            env_max_episode_steps: int = 1024,

            lagrange_kwargs: Optional[Dict[str, Any]] = None,
            other_os_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ############
        # - ALGO - #
        ############
        # Algo name
        if algo_name not in self.VALID_ALGO_NAMES:
            raise ValueError(f"Invalid algo_name={algo_name}")
        ############

        ###########
        # - ENV - #
        ###########
        env_id = "Morality-v0"
        if env_overrides is None:
            env_overrides = {"seed": seed}
        else:
            env_overrides["seed"] = seed

        env_cfgs = {
            "env_config": env_config,
            "env_overrides": env_overrides,
            "cost_kwargs": cost_kwargs,
            "morality_tree_config": morality_tree_config,
            "time_limit": env_time_limit,
            "max_episode_steps": env_max_episode_steps,
        }

        self.env_config = env_config
        self.env_overrides = env_overrides


        if vis_eval_episode:
            from morality_gym.environment.env.morality_env import env_from_config
            env_overrides["render_mode"] = "rgb_array"
            self._vis_env = env_from_config(env_config, env_overrides)
        else:
            self._vis_env = None
        ###########

        #############
        # - TRAIN - #
        #############
        self.vis_eval_episode = vis_eval_episode
        self.vis_dir = ""  # Will be set later

        # Device
        if "device" in train_kwargs:
            device = train_kwargs["device"]
        else:
            device = "auto"

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif "cuda" in device:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available.")
        elif device == "cpu":
            pass
        else:
            raise ValueError(f"Invalid device={device}")

        train_kwargs["device"] = device

        morality_tree = MoralityTree(morality_tree_config)

        def eval_callback(actor, eval_env, logger: Logger, curr_epoch, curr_step):
            def policy(obs):
                with torch.no_grad():
                    pred = actor.predict(obs, deterministic=True)
                return pred

            morality_metric, stats = evaluate_morality_metric(policy, morality_tree, eval_env,
                                                              n_eval_episodes, max_eval_steps)

            log_d = {
                'epoch': curr_epoch,
                'curr_step': curr_step,
                'timeouts': stats['tot_timeouts'],
                'successes': stats['tot_successes'],
                'tot_term': stats['tot_term'],
                'tot_trunc': stats['tot_trunc'],
                'avg_return': stats['avg_return'],
                'avg_steps': stats['avg_steps'],
                'avg_cost': stats['avg_cost'],
                'morality_metric': morality_metric,
            }

            for ds in morality_tree.ordered_danger_states:
                log_d[f'{ds.name}/prob'] = stats["prob_ds_occurrences"][ds]
                log_d[f'{ds.name}/occur'] = stats["tot_ds_occurrences"][ds]

            self.log_eval(log_d)
            # logger.dump_tabular()

            if self.vis_eval_episode:
                print(f"Creating 'vis_{self._curr_step}.gif'...")
                def vis_policy(obs):
                    with torch.no_grad():
                        pred = actor.predict(obs, deterministic=True)
                    return pred

                curr_vis_path = join_paths([self.vis_dir, f"vis_{self._curr_step}.gif"])
                self.vis_episode(vis_policy, save_to_path=curr_vis_path,
                                 max_steps=vis_eval_max_steps, seconds_per_frame=vis_eval_spf)


        train_kwargs["eval_callback"] = eval_callback
        #############

        custom_cfgs = {
            "seed": seed,
            "train_cfgs": train_kwargs,
            "algo_cfgs": algo_kwargs,
            "model_cfgs": model_kwargs,  # Not sure if this is valid :)
            "env_cfgs": env_cfgs,
            "logger_cfgs": logger_kwargs
        }
        if lagrange_kwargs is not None:
            # Probs do validation
            custom_cfgs["lagrange_cfgs"] = lagrange_kwargs

        if other_os_kwargs is not None:
            copy_to_dict(other_os_kwargs, custom_cfgs)

        self._os_agent = omnisafe.Agent(algo_name, env_id, custom_cfgs=custom_cfgs)

        self.logger = self._os_agent.agent.logger

        ######################
        # -- EVAL LOGGING -- #
        ######################
        self._eval_path = join_paths([self.logger.log_dir, "eval.csv"])
        self._eval_file = open(self._eval_path, "w")
        self._curr_step = None
        self._eval_cols = [
            'epoch', 'curr_step',
            'avg_steps', 'avg_return', 'avg_cost',
            'timeouts', 'successes', 'tot_term', 'tot_trunc',
            'morality_metric'
        ]
        for ds in morality_tree.ordered_danger_states:
            self._eval_cols.append(f"{ds.name}/prob")
            self._eval_cols.append(f"{ds.name}/occur")

        col_header = ",".join(self._eval_cols) + "\n"
        self._eval_file.write(col_header)
        ######################

        ########################
        # Register Logger Keys #
        ########################
        # _logger = self._os_agent.agent.logger
        # _logger.register_key('Eval/epoch')
        # _logger.register_key('Eval/curr_step')
        #
        # _logger.register_key('Eval/avg_steps')
        # _logger.register_key('Eval/avg_return')
        # _logger.register_key('Eval/avg_cost')
        #
        # _logger.register_key('Eval/timeouts')
        # _logger.register_key('Eval/successes')
        # _logger.register_key('Eval/tot_term')
        # _logger.register_key('Eval/tot_trunc')
        # _logger.register_key('Eval/morality_metric')

        # for ds in morality_tree.ordered_danger_states:
        #     _logger.register_key(f'EvalDangerStates/{ds.name}/prob')
        #     _logger.register_key(f'EvalDangerStates/{ds.name}/occur')
        ########################

        ##############
        # - LOGGER - #
        ##############
        if "log_dir" not in logger_kwargs:
            raise ValueError("log_dir must be specified in logger_kwargs")
        self.base_logger_dir = logger_kwargs["log_dir"]
        self.logger_dir = os.path.normpath(self.logger.log_dir)

        # Vis
        if self.vis_eval_episode:
            self.vis_dir = str(join_paths([self.logger_dir, "vis"]))
            os.mkdir(self.vis_dir)
        ##############

    def create_base_env(self):
        env = env_from_config(self.env_config, self.env_overrides)
        return env


    def learn(self):
        self._os_agent.learn()

    def vis_episode(
            self,
            policy,
            save_to_path: str,
            max_steps = 100,
            seconds_per_frame = 0.1,
    ):
        import imageio

        img_arrs = []

        vis_env = self._os_agent.agent._env._eval_env
        obs, info = vis_env.reset()
        is_done = False
        curr_step = 0
        while not is_done and curr_step < max_steps:
            action = policy(obs)
            obs, rew, cost, is_term, is_trunc, info = vis_env.step(action)

            curr_arr = vis_env.render()
            img_arrs.append(curr_arr)

            curr_step += 1
            is_done = is_term or is_trunc

        with imageio.get_writer(save_to_path, mode='I', duration=seconds_per_frame) as writer:
            for rgb_arr in img_arrs:
                writer.append_data(rgb_arr)

    def log_eval(self, log_d):
        keys_s = set(log_d.keys())
        eval_cols_s = set(self._eval_cols)
        if not keys_s.issubset(eval_cols_s):
            raise ValueError(f"Invalid keys in log_d = {keys_s - eval_cols_s}")

        if "curr_step" in log_d:
            self._curr_step = log_d["curr_step"]

        vals_arr = []
        for col in self._eval_cols:
            if col in log_d:
                vals_arr.append(str(log_d[col]))
            else:
                vals_arr.append("")
        vals_str = ",".join(vals_arr) + "\n"
        self._eval_file.write(vals_str)

    def finish(self):
        self.logger.close()
        self._eval_file.close()

    def move_logs(self, new_dir):
        import shutil
        create_or_clear_folder(new_dir)

        # Vis
        if os.path.isdir(self.vis_dir):
            shutil.copytree(src=self.vis_dir, dst=join_paths([new_dir, "vis"]))

        # Checkpoints
        ckpt_dir = join_paths([self.logger_dir, "torch_save"])
        if os.path.isdir(ckpt_dir):
            shutil.copytree(src=ckpt_dir, dst=join_paths([new_dir, "checkpoints"]))

        # WANDB
        wandb_dir = join_paths([self.logger_dir, "wandb"])
        if os.path.isdir(wandb_dir):
            shutil.copytree(src=wandb_dir, dst=join_paths([new_dir, "wandb"]))

        # run_config.json
        shutil.copy(src=join_paths([self.logger_dir, "run_config.json"]), dst=join_paths([new_dir, "run_config.json"]))

        # progress.csv
        shutil.copy(src=join_paths([self.logger_dir, "progress.csv"]),
                    dst=join_paths([new_dir, "progress.csv"]))

        # eval.csv
        shutil.copy(src=join_paths([self.logger_dir, "eval.csv"]),
                    dst=join_paths([new_dir, "eval.csv"]))

        # config.json
        shutil.copy(src=join_paths([self.logger_dir, "config.json"]),
                    dst=join_paths([new_dir, "os_config.json"]))

        # node.txt
        shutil.copy(src=join_paths([self.logger_dir, "node.txt"]),
                    dst=join_paths([new_dir, "node.txt"]))

        # del_path = comp_base_path(learner.logger_dir, level=1)
        # try:
        #     shutil.rmtree(del_path)
        # except PermissionError as e:
        #     print("Cannot fully delete del_path due to permission error:", e)


def main():
    seed = 42
    algo_name= "PPOLag"
    train_kwargs = {
        "total_steps": 50000,
        "eval_freq": 1,
        "vector_env_nums": 1,
        'parallel': 1
    }
    algo_kwargs = {
        "steps_per_epoch": 2048,
        "update_iters": 1
    }
    model_kwargs = {}
    logger_kwargs = {
        "log_dir": "./logs",
        "wandb_project": "safebench",
        "use_wandb": False
    }
    n_eval_episodes = 20
    max_eval_steps = 1000

    env_config = "sml_static_2l_1h.json"
    env_overrides = {
        "flatten_observation": True,
        "action_discrete": False
    }
    morality_tree_config = "asimov_3_laws.json"

    learner = SafeLearner(
        seed=seed,
        algo_name=algo_name, algo_kwargs=algo_kwargs,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
        logger_kwargs=logger_kwargs,
        env_config=env_config, morality_tree_config=morality_tree_config, env_overrides=env_overrides,
        n_eval_episodes=n_eval_episodes, max_eval_steps=max_eval_steps
    )
    print(learner.logger_dir)

    # learner.learn()


if __name__ == "__main__":
    main()
