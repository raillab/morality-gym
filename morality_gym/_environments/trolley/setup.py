import json
import os
from typing import Optional, Any, Dict, Tuple, List

from morality_gym._environments.core.env import BaseEnv
from morality_gym._environments.trolley.scenario import easy_create_from_config
from morality_gym._utils.common import join_paths, comp_base_path, copy_to_dict

CONFIGS_PATH = join_paths([comp_base_path(os.path.abspath(__file__)), "configs"])

def _compute_registry():
    configs_path = join_paths([CONFIGS_PATH, "environment"])
    supported_envs = os.listdir(configs_path)
    supported_envs = [".".join(curr_env_id.split(".")[:-1]) for curr_env_id in supported_envs]
    supported_envs = list(filter(lambda x: x!="", supported_envs))
    return supported_envs

ENV_REGISTRY = _compute_registry()


def make(
        env_id: str,
        scenario_overrides: Optional[Dict[str, Any]] = None,
        env_overrides: Optional[Dict[str, Any]] = None,
):
    # Example env_id: "SwitchStandard-0-v0"
    # Format: [base scenario name]-[variant no]-v[version no]

    info = {}

    tmp_arr = env_id.split("-")
    base_env_id = tmp_arr[:-2] + tmp_arr[-1:]
    variant = tmp_arr[-2]
    if variant.lower() == "all":
        # Hacky way to ensure consistent formatting :)
        variant = "all"

    if variant != "all":
        try:
            conv_variant = int(variant)
            variant = conv_variant
        except ValueError:
            pass
    base_env_id = "-".join(base_env_id)

    configs_path = join_paths([CONFIGS_PATH, "environment"])
    # supported_envs = os.listdir(configs_path)
    # supported_envs = [".".join(curr_env_id.split(".")[:-1]) for curr_env_id in supported_envs]
    if base_env_id not in ENV_REGISTRY:
        raise ValueError(f"Environment {base_env_id} not supported. Supported environments: {ENV_REGISTRY}")

    config_path = join_paths([configs_path, f"{base_env_id}.json"])
    with open(config_path, "r") as f:
        config_d = json.load(f)

    seed = config_d["seed"]

    ################
    # - SCENARIO - #
    ################
    scenario_d = config_d["scenario"]
    if "overrides" not in scenario_d:
        scenario_d["overrides"] = {}
    scenario_d["overrides"]["seed"] = seed
    if scenario_overrides is not None:
        copy_to_dict(from_dict=scenario_overrides, to_dict=scenario_d["overrides"])
    scenario = easy_create_from_config(variant, **scenario_d)
    scenario.build()

    info["utility_bounds"] = scenario.utility_bounds
    info["global_utility_bounds"] = scenario.global_utility_bounds
    ################

    ###############
    # --  ENV  -- #
    ###############
    env_kwargs = config_d["env"]
    obs_entities_copy = env_kwargs["obs_entities"].copy()
    for obs_entity in obs_entities_copy:
        if obs_entity not in scenario.entities:
            env_kwargs["obs_entities"].remove(obs_entity)

    if env_overrides is not None:
        copy_to_dict(from_dict=env_overrides, to_dict=env_kwargs)
    # env_kwargs["seed"] = seed

    world = scenario.world

    env = BaseEnv(world, **env_kwargs)

    # env.tot_character_harmed_bounds = scenario.utility_bounds
    ###############

    return env, info
    # # NOTE: Temp janky fix to allow for functionality - will do properly
    # if env_id=="PushStandard-v0":
    #     return make_push_env(**env_kwargs)
    # elif env_id=="SwitchStandard-v0":
    #     return make_switch_env(**env_kwargs)
    # else:
    #     raise NotImplementedError

