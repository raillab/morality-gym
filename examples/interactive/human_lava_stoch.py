# Interactive environment containing a human and two lava
# Dynamics are stochastic due to noise on control (action) signal
# Static environment: static start (reset) location for robot. Static locations for landmark, lavas and human

from morality_gym.environment.env.morality_env import env_from_config, interactive


def main():
    env = env_from_config(
        "sml_static_2l_1h_stoch.json", # config file
        overrides={"control_noise": 1.0}  # override certain kwargs from config file when creating env
    )

    # or similarly:
    # env_kwargs = load_config("sml_static_2l_1h_stoch.json")
    # env_kwargs["control_noise"] = 1.0
    # env = MoralityEnv(**env_kwargs)

    interactive(env)


if __name__ == "__main__":
    main()