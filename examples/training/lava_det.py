# Interactive environment containing two
# Dynamics are deterministic
# Static environment: static start (reset) location for robot. Static locations for landmark and lavas

from morality_gym.environment.env.morality_env import env_from_config, interactive, MoralityEnv


def main():
    env: MoralityEnv = env_from_config(
        "sml_static_2l.json", # config file
        overrides={"flatten_observation": True}  # override certain kwargs from config file when creating env
    )

    # For testing environment. Comment out next line
    interactive(env)

    # Do training


if __name__ == "__main__":
    main()