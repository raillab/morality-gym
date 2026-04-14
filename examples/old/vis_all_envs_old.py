from PIL import Image

from morality_gym.setup.setup import make as env_mt_make

def main():
    from morality_gym.environments.trolley.setup import ENV_REGISTRY
    all_trolley_envs = ENV_REGISTRY
    env_kwargs = {
        "scenario_overrides": {"seed": 42},
        "env_overrides": {
            "obs_type": dict,
            "is_normalise_obs": True,
            # "step_penalty": 0
        }
    }

    for env_name in all_trolley_envs:
        env_name = env_name.split("-")[0]
        env_id = f'MoralityGym/Trolley-{env_name}-0-v1'
        print(f"\n>>> Creating image for {env_name} with id = {env_id} <<<")
        env, mt = env_mt_make(
            env_id=env_id,
            # env_id='MoralityGym/Trolley-PushStandard-0-v0',
            morality_chain_id='Trolley-Common-Utilitarian-UtilityHarm-v0',
            env_kwargs=env_kwargs
        )
        img_path = f"figs/{env_name}.png"

        img_arr = env.render()
        img = Image.fromarray(img_arr)
        img.save(img_path, "PNG")
        env.close()


    print(all_trolley_envs)

if __name__ == "__main__":
    main()
