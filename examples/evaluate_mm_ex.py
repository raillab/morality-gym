def main():
    from morality_gym.setup.setup import make
    env_id = "SwitchStandard-HumanA-v1"
    env, mc = make(env_id=env_id, morality_chain_id="Utility")

    def random_policy(obs):
        return env.action_space.sample()

    # Override norms to evaluate using
    # Only humans can be harmed in SwitchStandard-HumanA so only n_humans_harmed norm is applicable
    norms = []

    morality_metric, morality_functions, avg_return, info = \
        mc.evaluate_morality_metric(policy=random_policy, env=env, progress_bar=True, norms=norms)

if __name__ == "__main__":
    main()