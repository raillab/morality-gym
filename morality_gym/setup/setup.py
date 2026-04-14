from typing import Optional, Dict, Any, Union, List


def make(
        env_id,
        morality_chain_id,
        env_kwargs: Optional[Dict[str, Any]] = None,
        morality_chain_kwargs: Optional[Dict[str, Any]] = None,
):
    from morality_gym.morality_chain.setup import make as morality_chain_make

    # env_id examples:
    #   - "MoralityGym/Trolley-SwitchStandard-v0"
    #   - "MoralityGym/Trolley-PushStandard-v0"
    #   - "MoralityGym/Trolley-SelfSacrificeStandard-v0"

    ##############
    # CREATE ENV #
    ##############
    if env_kwargs is None:
        env_kwargs = {}

    if env_id.startswith('MoralityGym/'):
        env_id = env_id.replace('MoralityGym/', '')
        # 'MoralityGym/' no longer required at start of ID

    split_arr = env_id.split("-")
    if len(split_arr) == 4:
        env_category = env_id.split('-')[0]
        env_id = "-".join(env_id.split('-')[1:])
    elif len(split_arr) == 3:
        # Only Trolley category is supported so Trolley may be omitted from env_id
        env_category = "Trolley"
    else:
        raise ValueError(f"Invalid format for env_id = {env_id}.")

    # env_category = env_id.split('-')[0]
    if env_category == "Trolley":
        # TROLLEY #
        from morality_gym.environments.trolley.setup import make as trolley_make
        # env_sub_id = "-".join(env_id.split('-')[1:])
        env, env_info = trolley_make(env_id, **env_kwargs)
    else:
        raise NotImplementedError(f"env_category {env_category} not implemented")
    ##############

    #################
    # MORALITY TREE #
    #################
    if morality_chain_kwargs is None:
        morality_chain_kwargs = {}

    # utility_bounds = env.get_utility_bounds()
    utility_bounds = env_info["utility_bounds"]
    morality_chain_kwargs["utility_bounds"] = utility_bounds

    global_utility_bounds = env_info["global_utility_bounds"]
    morality_chain_kwargs["global_utility_bounds"] = global_utility_bounds

    # utility_bounds = env.get_utility_bounds()
    # if utility_bounds is not None:
    #     morality_chain_kwargs["utility_bounds"] = utility_bounds

    if env.salient_norms is not None:
        morality_chain_kwargs["salient_norms"] = env.salient_norms

    if "beta" not in morality_chain_kwargs:
        max_bound = 0
        for norm in ["n_animal_harm", "n_human_harm", "n_robot_harm"]:
            if norm in utility_bounds:
                max_bound = max(max_bound, utility_bounds[norm][1])
        if max_bound > 0:
            beta = 1 / max_bound
        else:
            beta = 1.0

        morality_chain_kwargs["beta"] = beta
        # raise NotImplementedError("beta=None for morality chain is not supported yet.")

    # # Kinda hacky way to set utility bounds:
    # if "Utilitarian" in morality_chain_id:
    #     if env.tot_character_harmed_bounds is None:
    #         raise ValueError(f"tot_character_harmed_bounds not set for env {env_id}")
    #     morality_chain_kwargs["human_utility_range"] = env.tot_character_harmed_bounds["human"]
    #     morality_chain_kwargs["animal_utility_range"] = env.tot_character_harmed_bounds["animal"]
    #     morality_chain_kwargs["robot_utility_range"] = env.tot_character_harmed_bounds["robot"]


    mt = morality_chain_make(morality_chain_id, **morality_chain_kwargs)
    # mt.comp_weights()
    #################

    return env, mt

def get_supported(
        scenario_id: Optional[str] = None
) -> Union[Dict[str, List[str]], List[str]]:
    from morality_gym.environments.trolley.supported_envs import SUPPORTED_ENVS
    if scenario_id is None:
        return SUPPORTED_ENVS
    else:
        if scenario_id in SUPPORTED_ENVS.keys():
            return SUPPORTED_ENVS[scenario_id]
        else:
            raise ValueError(f"scenario_id = {scenario_id} not supported")

def main():
    import pprint
    supported_envs = get_supported()
    n_envs = sum([len(el) for el in supported_envs.values()])
    print(f"n_envs = {n_envs}\n\n")
    pprint.pp(supported_envs)

if __name__ == "__main__":
    main()
