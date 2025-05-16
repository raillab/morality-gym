from typing import Optional, Dict

import gymnasium
import hydra
import torch
from omegaconf import DictConfig

from morality_gym.environment.utils_old import OmniSafeMoralityEnv
from morality_gym.morality_tree.morality_metric import evaluate_morality_metric

# from omnisafe.algorithms import BaseAlgo
from omnisafe.envs.core import env_register
import omnisafe


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


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    from morality_gym.morality_tree.morality_tree import MoralityTree
    from omnisafe.common import Logger

    # custom_cfgs = {
    #     "train_cfgs": dict(cfg.train),
    #     'algo_cfgs': dict(cfg.algo),
    #     'logger_cfgs': dict(cfg.logger),
    # }

    morality_tree = MoralityTree(cfg.env.morality_tree_config)
    n_eval_episodes = 20
    max_eval_steps = 1000

    def eval_callback(actor, eval_env, logger: Logger, curr_epoch, curr_step):
        # print(f"methods of eval_env: {dir(eval_env)}")

        def policy(obs):
            # TODO: Check
            with torch.no_grad():
                pred = actor.predict(obs, deterministic=True)
            return pred

        morality_metric, stats = evaluate_morality_metric(policy, morality_tree, eval_env, n_eval_episodes, max_eval_steps)
        log_d = {
            'Eval/epoch': curr_epoch,
            'Eval/curr_step': curr_step,
            'Eval/timeouts': stats['tot_timeouts'],
            'Eval/successes': stats['tot_successes'],
            'Eval/avg_return': stats['avg_return'],
            'Eval/avg_steps': stats['avg_steps'],
            'Eval/avg_cost': stats['avg_cost'],
            'Eval/morality_metric': morality_metric,
        }

        for ds in morality_tree.ordered_danger_states:
            log_d[f'EvalDangerStates/{ds.name}/prob'] = stats["prob_ds_occurrences"][ds]
            log_d[f'EvalDangerStates/{ds.name}/occur'] = stats["tot_ds_occurrences"][ds]

        logger.store(log_d)

    custom_cfgs = {
        'train_cfgs': {
            'total_steps': cfg.train.total_timesteps,
            'vector_env_nums': cfg.train.vector_env_nums,
            'parallel': cfg.train.parallel,
            'eval_callback': eval_callback,
            'eval_freq': cfg.train.eval_freq
        },
        'algo_cfgs': {
            'steps_per_epoch': cfg.train.steps_per_epoch,
            'update_iters': cfg.train.update_iters,
        },
        # 'logger_cfgs': {
        #     'use_wandb': cfg.logger.use_wandb,
        #     'project': cfg.logger.project,
        # },
        'logger_cfgs': dict(cfg.logger)
    }

    env_kwargs = {
        "env_config": cfg.env.env_config,
        "env_overrides": dict(cfg.env.env_overrides),  # Causes error without type conversion4
        "morality_tree_config": cfg.env.morality_tree_config,
        'time_limit': cfg.env.time_limit,
        "max_episode_steps": cfg.env.max_episode_steps,
        # "num_envs": cfg.env.num_envs,
    }
    # custom_cfgs.update({'env_cfgs': {'time_limit': cfg.env.time_limit,
    #                                  "max_episode_steps": cfg.env.max_episode_steps,
    #                                  "env_config": cfg.env.config_dir,
    #                                  "overrides": dict(cfg.env.overrides)  # Causes error without type conversion
    #                                  }})
    custom_cfgs.update({'env_cfgs': env_kwargs})

    agent = omnisafe.Agent(cfg.model.algorithm, cfg.env.env_id, custom_cfgs=custom_cfgs)

    # Register logger keys
    _logger = agent.agent.logger  # I don't know why the logger is private
    _logger.register_key('Eval/epoch')
    _logger.register_key('Eval/curr_step')

    _logger.register_key('Eval/avg_steps')
    _logger.register_key('Eval/avg_return')
    _logger.register_key('Eval/avg_cost')

    _logger.register_key('Eval/timeouts')
    _logger.register_key('Eval/successes')
    _logger.register_key('Eval/morality_metric')

    for ds in morality_tree.ordered_danger_states:
        _logger.register_key(f'EvalDangerStates/{ds.name}/prob')
        _logger.register_key(f'EvalDangerStates/{ds.name}/occur')


    agent.learn()
    # agent.learn()
    # agent.plot(smooth=1)

    if cfg.evaluation.evaluate:
        agent.render(num_episodes=cfg.evaluation.eval_iters, render_mode='rgb_array', width=256, height=256)
    
if __name__ == "__main__":
    main()