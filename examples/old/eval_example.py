import pprint

import numpy as np


def main():
    from morality_gym.setup.setup import make as env_mt_make
    env_kwargs = {
        # "scenario_overrides": {
        #     "character_amounts": {
        #         "character_1": 99, "character_2": 1
        #     },
        #     "character_types": {
        #         "character_1": "human", "character_2": "animal"
        #     },
        # }
    }
    #1727
    env, mt = env_mt_make(
        env_id='MoralityGym/Trolley-Push2Character-all-v1',
        morality_chain_id='Trolley-Common-DualProcess-Complex-v0',
        env_kwargs=env_kwargs
    )

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Interactive Mode Keys:
    # Movement: W, A, S, D
    # Stay (No movement): SPACE
    # Interact (push): E

    # Uncomment if you wish to interact with environment manually
    from morality_gym.environments.core.env import interactive
    interactive(env)

    def random_policy(_):
        return env.action_space.sample()

    morality_metric, morality_functions, avg_return, info = \
        mt.evaluate_morality_metric(policy=random_policy, env=env, max_episode_steps=1000, n_repeats=100,
                                    handle_trunc='include', is_prog_bar=True)

    print(f"Morality metric: {morality_metric}")
    print(f"Morality functions:")
    pprint.pprint(morality_functions)
    print(f"Avg. return: {avg_return}")
    print("Info:")
    pprint.pprint(info)



if __name__ == '__main__':
    main()
