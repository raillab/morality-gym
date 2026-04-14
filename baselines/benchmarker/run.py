import json
from typing import Dict, Any, Optional, Union

from baselines.benchmarker.utils import load_base_paths
from baselines.logger import Logger
from morality_gym.utils.common import join_paths


def _get_runs_path():
    base_paths_d = load_base_paths()
    return base_paths_d["runs"]

# I don't like that this runs on import, but I also don't want to constantly reload json :/
RUNS_PATH = _get_runs_path()

OS_LEARNERS = {
    "PPO", "PPOShaped",
    "PPOLag", "CPO",
    "CPPOPID", "TRPOPID",
}

def _create_learner(
        config: Dict[str, Any],
        logger: Logger,
        is_interactive: bool = False
):
    name = config["name"]
    init_kwargs = config["init_kwargs"]

    if name == "random":
        from baselines.algorithms.standard.random_learner import RandomLearner
        learner = RandomLearner(logger=logger, **init_kwargs)
    elif name in {"ppo", "ppo_shaped"}:
        # raise ValueError("SB3 algorithms are deprecated. Use omnisafe versions instead with PPO or PPOShaped")
        is_shaped = config["init_kwargs"]["shape_reward"]
        if name == "ppo_shaped":
            if not is_shaped:
                raise ValueError("Cannot use ppo_shaped if shape_reward is False.")
        else:
            if is_shaped:
                raise ValueError("Cannot use ppo if shape_reward is True.")

        from baselines.algorithms.standard.sb3 import SB3Learner
        learner = SB3Learner(logger=logger, **init_kwargs)

        if is_interactive:  # For debugging environments
            from baselines.wrappers.sb3 import interactive
            interactive(learner.eval_env)
            exit()
    elif name in OS_LEARNERS:
        from baselines.algorithms.safe.os_learner import OSLearner
        learner = OSLearner(logger=logger, algo_name=name, **init_kwargs)
    else:
        raise ValueError(f"Invalid name: {name}.")

    return learner


def run(
        config: Union[Dict[str, Any], str],
        node: Optional[str] = None,
        log_dir: Optional[str] = None,
        is_interactive: bool = False
):
    if isinstance(config, str):
        full_path = join_paths([RUNS_PATH, config])
        with open(full_path, "r") as f:
            config = json.load(f)

    config["node"] = node
    if log_dir is not None:
        config["logger"]["log_dir"] = log_dir

    exp_type = config["exp"]["type"]
    if exp_type == "learn_and_eval":
        _learn_and_eval(config, is_interactive=is_interactive)
    elif exp_type == "eval_bounds":
        _eval_bounds(config)
    else:
        raise ValueError(f"Invalid exp type: {exp_type}.")


############################
# ---- LEARN AND EVAL ---- #
############################
def _learn_and_eval(
        config: Dict[str, Any],
        is_interactive: bool = False,
):
    logger = Logger(config=config, **config["logger"])
    learner = _create_learner(config["learner"], logger, is_interactive)
    learner.learn(**config["learner"]["learn_kwargs"])
    logger.finish()
    learner.finish()
    del learner  # Something makes the threads hang, so maybe this will help :P


############################


######################################################
# Evaluate Bounds of Morality Metric & Cost Function #
######################################################
def _eval_bounds(
        config: Dict[str, Any],
):
    from baselines.algorithms.standard.eval_bounds import EvalBoundsLearner

    logger = Logger(config=config, **config["logger"])
    learner_config = config["learner"]
    eval_learner = EvalBoundsLearner(logger=logger, **learner_config["init_kwargs"])
    eval_learner.eval_bounds(learner_config["learn_kwargs"]["n_repeats"])
    logger.finish()

######################################################

def main():
    run("d0/run_0.json", "local")


if __name__ == "__main__":
    main()