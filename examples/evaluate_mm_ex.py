import pprint


def main():
    from morality_gym import make
    env, mc = make(env_id="SwitchStandard-HumanA-v1", morality_chain_id="Utility")

    def random_policy(obs):
        return env.action_space.sample()

    morality_metric, morality_functions, info = \
        mc.evaluate_morality_metric(policy=random_policy, env=env, is_prog_bar=True)

    print("Morality metric: ", morality_metric)
    print("Morality functions: ")
    pprint.pp(morality_functions)
    print("Info: ")
    pprint.pprint(info)

if __name__ == "__main__":
    main()