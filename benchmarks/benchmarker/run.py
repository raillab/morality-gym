import argparse
import json
import os
import sys
from typing import Dict, Any, Optional, List
import shutil

import wandb

sys.path.append('..')

from morality_gym.utils.common import join_paths, create_or_clear_folder, comp_base_path


def exec_run(
        config_path: str,  # relative to benchmarks/benchmarker/runs if is_rel_path = True
        learner_overrides: Optional[Dict[str, Any]] = None,
        is_interactive_env: bool = False,
        log_path: Optional[str] = None,
        log_temp_path: Optional[str] = None,
        node: Optional[str] = None,
        is_config_rel_path: bool = True
):
    from benchmarks.benchmarker.learner import SafeLearner

    is_tmp_log = log_temp_path is not None

    base_path = comp_base_path(os.path.abspath(__file__), level=1)
    if is_config_rel_path:
        config_path = join_paths([base_path, "runs", config_path])


    with open(config_path, "r") as f:
        config_d = json.load(f)

    learner_kwargs = config_d["learner_kwargs"]
    if learner_overrides is not None:
        # TODO
        raise NotImplementedError

    if is_interactive_env:
        from morality_gym.environment.env.morality_env import interactive, env_from_config
        tmp_env_overrides = learner_kwargs["env_overrides"]
        tmp_env_overrides["action_discrete"] = True

        inter_env = env_from_config(learner_kwargs["env_config"], tmp_env_overrides)
        interactive(inter_env)

    if log_temp_path is not None:
        learner_kwargs["logger_kwargs"]["log_dir"] = log_temp_path
    elif log_path is not None:
        learner_kwargs["logger_kwargs"]["log_dir"] = log_path
    else:
        pass

    learner = SafeLearner(**learner_kwargs)

    # save config in log folder
    # new_path = join_paths([learner.logger_dir, "run_config.json"])
    # shutil.copyfile(src=config_path, dst=new_path)

    if config_d["run_type"] == "learn":
        learner.learn()
    else:
        raise ValueError(f"Invalid run type: {config_d['run_type']}. Valid types are: ['learn']")

    wandb.finish()
    learner.finish()

    # run_config.json
    shutil.copy(src=config_path, dst=join_paths([learner.logger_dir, "run_config.json"]))
    node_file = join_paths([learner.logger_dir, "node.txt"])
    with open(node_file, "w") as f:
        f.write(str(node))

    if is_tmp_log:
        learner.move_logs(log_path)

        try:
            shutil.rmtree(log_temp_path)
        except PermissionError as e:
            print("Cannot fully delete del_path due to permission error:", e)

    # # Vis
    # if os.path.isdir(learner.vis_dir):
    #     shutil.copytree(src=learner.vis_dir, dst=join_paths([learner.base_logger_dir, "vis"]))
    #
    # # Checkpoints
    # ckpt_dir = join_paths([learner.logger_dir, "torch_save"])
    # if os.path.isdir(ckpt_dir):
    #     shutil.copytree(src=ckpt_dir, dst=join_paths([learner.base_logger_dir, "checkpoints"]))
    #
    # # WANDB
    # wandb_dir = join_paths([learner.logger_dir, "wandb"])
    # if os.path.isdir(wandb_dir):
    #     shutil.copytree(src=wandb_dir, dst=join_paths([learner.base_logger_dir, "wandb"]))
    #
    # # run_config.json
    # shutil.copy(src=config_path, dst=join_paths([learner.base_logger_dir, "run_config.json"]))
    # # progress.csv
    # shutil.copy(src=join_paths([learner.logger_dir, "progress.csv"]),
    #             dst=join_paths([learner.base_logger_dir, "progress.csv"]))
    # # config.json
    # shutil.copy(src=join_paths([learner.logger_dir, "config.json"]),
    #             dst=join_paths([learner.base_logger_dir, "os_config.json"]))
    #
    # node_file = join_paths([learner.base_logger_dir, "node.txt"])
    # with open(node_file, "w") as f:
    #     f.write(str(node))


    # data_dir = join_paths([learner.base_logger_dir, "data"])
    # create_or_clear_folder(data_dir)
    #
    # ckpt_dir = join_paths([learner.logger_dir, "torch_save"])
    # if os.path.isdir(learner.vis_dir):
    #     shutil.copytree(src=ckpt_dir, dst=join_paths([data_dir, "checkpoints"]))
    #
    # vis_dir = join_paths([learner.logger_dir, "vis"])
    # new_vis_dir = join_paths([data_dir, "vis"])
    # if os.path.isdir(learner.vis_dir):
    #     shutil.copytree(src=vis_dir, dst=new_vis_dir)
    #
    # shutil.copy(src=config_path, dst=join_paths([data_dir, "run_config.json"]))
    # shutil.copy(src=join_paths([learner.logger_dir, "progress.csv"]), dst=join_paths([data_dir, "progress.csv"]))


    # Move log files to base dir
    # file_dir_names = os.listdir(learner.logger_dir)
    # for file_dir_name in file_dir_names:
    #     shutil.move(join_paths([learner.logger_dir, file_dir_name]), join_paths([learner.base_logger_dir, file_dir_name]))
        # shutil.move(
        #     join_paths([learner.logger_dir, file_dir_name]),
        #     join_paths([learner.logger_dir, "..", file_dir_name])
        # )

def exec_multi_run(config_paths: List[str], log_base_path: str, log_rel_paths: List[str],
                   log_temp_path: Optional[str] = None,
                   node: Optional[str] = None, is_config_rel_path: bool = True):

    if not os.path.isdir(log_base_path):
        os.mkdir(log_base_path)
    for i, curr_config in enumerate(config_paths):
        print(f"\n#############################")
        print(f"Running config {curr_config} on node={node}...")
        print(f"#############################")
        log_temp_path = join_paths([log_temp_path, log_rel_paths[i]])
        create_or_clear_folder(log_temp_path)

        log_path = join_paths([log_base_path, log_rel_paths[i]])
        create_or_clear_folder(log_path)
        # curr_path = join_paths([log_base_path, log_rel_paths[i]])
        # create_or_clear_folder(curr_path)
        exec_run(curr_config, is_interactive_env=False, log_path=log_path, node=node,
                 is_config_rel_path=is_config_rel_path, log_temp_path=log_temp_path)
        print(f"#############################")


def exec_all_runs(
        config_base_path: str, # relative to benchmarks/benchmarker/runs
        log_base_path: str  # relative benchmarks/benchmarker/logs
):

    base_path = comp_base_path(os.path.abspath(__file__), level=1)
    config_base_path = join_paths([base_path, "runs", config_base_path])

    log_base_path = join_paths([base_path, "logs", log_base_path])
    create_or_clear_folder(log_base_path, confirm=True)


    print(f"Running all configs in {config_base_path} and logging in {log_base_path}...\n")
    all_runs = os.listdir(config_base_path)

    def extract_file_name(file_path):
        base_name = os.path.basename(os.path.normpath(file_path))
        file_name, ext = os.path.splitext(base_name)
        return file_name

    run_names = [extract_file_name(run_path) for run_path in all_runs]
    run_tups = [(name, path) for name, path in zip(run_names, all_runs)]
    run_tups.sort(key=lambda x: int(x[0].split("_")[1]))

    for i, (run_name, run_path) in enumerate(run_tups):
        run_abs_path = join_paths([config_base_path, run_path])
        log_path = join_paths([log_base_path, run_name])
        print(f"\n#############################")
        print(f"[Config {i+1}/{len(run_tups)}] - Running config {run_path} and logging to {log_path}...")
        # print(f"")
        # print(f"Logging to {log_path}...")
        print(f"#############################")
        # curr_config = join_paths([config_base_path, run_path])

        # create_or_clear_folder(log_path)
        exec_run(run_abs_path, is_interactive_env=False, log_path=log_path, is_rel_path=False)
        print(f"#############################")



def debug():
    exec_all_runs("test", "test_alt")
    # config_path = [f"test/run_{i}.json" for i in range(2)]
    # log_base_path = "logs/test"
    # log_rel_paths = [f"run_{i}" for i in range(2)]
    # exec_multi_run(config_paths=config_path, log_base_path=log_base_path,log_rel_paths=log_rel_paths)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cp", "--config-path", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("-ie", "--interactive-env", action="store_true",
                        help="Whether to run the environment in interactive mode")
    parser.add_argument("-lp", "--log-path", type=str,)
    args = parser.parse_args()

    exec_run(args.config_path, is_interactive_env=args.interactive_env, log_path=args.log_path)


if __name__ == "__main__":
    debug()
