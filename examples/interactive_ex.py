from morality_gym import interactive
from morality_gym import make

def main():
    scenario = "Switch5"
    variant = "Human"
    env_kwargs = {
        "env_overrides": {
            "obs_type": "dict",
            "render_mode": "human"
        }
    }
    env, mc = make(
        env_id=f'{scenario}-{variant}-v1',
        morality_chain_id='Utility',
        env_kwargs=env_kwargs
    )
    interactive(env)

if __name__ == "__main__":
    main()
