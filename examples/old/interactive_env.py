import numpy as np

from morality_gym.environments.core.env import interactive
from morality_gym.setup.setup import make #as env_mt_make

def main():
    remove_obs = {
        "character_*": ["amount", "character_type", "pos", "is_harmed"],
        "trolley_*": ["is_actable"],
        "player": ["is_harmed"]
    }

    # env_id: str = "MoralityGym/Trolley-SwitchStandard-10-v0"
    #     morality_chain_id: str = "Trolley-Common-StandardUtilitarian-v0"
    env_kwargs = {
        "scenario_overrides": {
            "seed": 42,
            "max_timesteps": 50,
        },
        "env_overrides": {
            "obs_type": "dict",
            "is_normalise_obs": True,
            "render_mode": "human",
            "remove_obs": remove_obs, #remove_obs
            # "step_penalty": 0
        }
    }

    # from experiments.baselines.common.utils import create_rand_variants
    # variants = create_rand_variants("SwitchStandard-V1")


    reset_kwargs = {
    }
    env, mt = make(
        env_id='SwitchStandard-HumanA-v1',
        morality_chain_id='Utility',
        env_kwargs=env_kwargs,
    )
    # print(mt.compute_weights({"n_human_harm"}))

    interactive(env, reset_kwargs=reset_kwargs)

if __name__ == "__main__":
    main()