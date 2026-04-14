import json
import os

from PIL import Image

from morality_gym.setup.setup import make as env_mt_make
from morality_gym.utils.common import comp_base_path, join_paths, create_or_clear_folder

PROJECT_PATH = comp_base_path(os.path.abspath(__file__), level=2)

def main():
    env_kwargs = {
        "scenario_overrides": {"seed": 42},
        "env_overrides": {
            "obs_type": dict,
            "is_normalise_obs": True,
            "render_mode": "human"
        }
    }

    # create_or_clear_folder("figs")

    # all_envs = [
    #     "Push2OrSwitch", "PushOrSwitch", "Switch2Trolley4Track", "Switch5", "Switch7", "SwitchStandard",
    #     "PushStandard", "PushOrSwitchSelfSacrifice", "SwitchSelfSacrifice",
    #     "Push3SelfSacrifice", "PushOrSwitchSelfSacrifice", "PushSelfSacrifice",
    #     "Switch18", "Switch4Trolley"
    # ]

    all_envs = [
        "SwitchStandard", "PushStandard",
        "PushOrSwitch", "SwitchSelfSacrifice", "PushSelfSacrifice",
        "Switch5", "Switch2Trolley4Track", "Push2OrSwitch", "Push3SelfSacrifice",
        "PushOrSwitchSelfSacrifice",
        "Switch7"
    ]



    env_config_path = join_paths([PROJECT_PATH, "baselines", "benchmarker", "configs", "environment.json"])
    with open(env_config_path, "r") as f:
        env_config = json.load(f)

    # groups = {"x_hard"}

    version = env_config["version"]

    for env_name in all_envs:
        env_d = env_config[env_name]
    # for env_name, env_d in env_config.items():
    #     if env_name in {"version", "sanity_check", "x_easy", "easy", "med", "hard", "x_hard"}:
    #         continue

        # if env_name not in {"Switch18"}:
        #     continue
        #
        difficulty = env_d["default"]
        # if difficulty not in groups:
        #     continue

        if "excluded" in env_d and env_d["excluded"]:
            continue

        if env_d["variants"] is not None:
            for variant in env_d["variants"].keys():
                fig_name = f"{difficulty}-{env_name}-{variant}"
                env_id = f"{env_name}-{variant}-v{version}"
                mc_id = env_d["variants"][variant][0]
                print(f"\n>>> Creating image for {env_name} with id = {env_id} <<<")
                env, mt = env_mt_make(
                    env_id=env_id,
                    # env_id='MoralityGym/Trolley-PushStandard-0-v0',
                    morality_chain_id=mc_id,
                    env_kwargs=env_kwargs
                )
                # img_path = f"figs/{fig_name}.png"

                img_path = f"figs/{env_name}.png"

                img_arr = env.render()
                img = Image.fromarray(img_arr)
                img.save(img_path, "PNG")
                env.close()
                break

    # for env_name in all_trolley_envs:
    #     env_name = env_name.split("-")[0]
    #     env_id = f'MoralityGym/Trolley-{env_name}-0-v1'
    #     print(f"\n>>> Creating image for {env_name} with id = {env_id} <<<")
    #     env, mt = env_mt_make(
    #         env_id=env_id,
    #         # env_id='MoralityGym/Trolley-PushStandard-0-v0',
    #         morality_chain_id='Trolley-Common-Utilitarian-UtilityHarm-v0',
    #         env_kwargs=env_kwargs
    #     )
    #     img_path = f"figs/{env_name}.png"
    #
    #     img_arr = env.render()
    #     img = Image.fromarray(img_arr)
    #     img.save(img_path, "PNG")
    #     env.close()


    # print(all_trolley_envs)

if __name__ == "__main__":
    main()
