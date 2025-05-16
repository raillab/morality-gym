from morality_gym.environment.env.morality_env import env_from_config, interactive


def interactive_simple():
    env = env_from_config("sml_static_2l.json")
    interactive(env)

def main():
    interactive_simple()


if __name__ == "__main__":
    main()