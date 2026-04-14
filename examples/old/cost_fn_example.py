import pprint

import numpy as np

from experiments.cost_function import Cost
from morality_gym.environments.core.env import interactive


def main():
    from morality_gym.setup.setup import make as env_mt_make
    # env, mt = env_mt_make(
    #     env_id='MoralityGym/Trolley-PushStandard-v0',
    #     env_kwargs={"seed": 42, "obs_type": np.ndarray, "is_normalise_obs": True},
    #     morality_chain_id='Trolley-PushStandard-Utilitarian-v0'
    # )
    # scenario_overrides: Optional[Dict[str, Any]] = None,
    #         env_overrides: Optional[Dict[str, Any]] = None,

    env_kwargs = {
        "scenario_overrides": {"seed": 42},
        "env_overrides": {"obs_type": np.ndarray, "is_normalise_obs": True, "step_penalty": -0.01, "landmark_reward": 10}
    }

    env, mt = env_mt_make(
        env_id='MoralityGym/Trolley-Push2Character-all-v1',
        env_kwargs=env_kwargs,
        morality_chain_id='Trolley-Common-DualProcess-Complex-v0'
    )

    max_episode_steps = 100
    n_episodes = 10
    interactive(env)

    cost_obj = Cost(mt)

    def random_policy(_):
        return env.action_space.sample()

    for curr_episode in range(n_episodes):

        print(f"\n################")
        print(f"# EPISODE = {curr_episode:02} #")
        print(f"################")
        obs, info = env.reset()
        cost_obj.reset()
        is_done = False
        curr_step = 0


        while not is_done and curr_step < max_episode_steps:
            action = random_policy(obs)
            obs, reward, is_term, is_trunc, info = env.step(action)
            is_done = is_term or is_trunc

            cost = cost_obj(info, is_term)
            print(f"Step {curr_step}: cost={cost}")

            curr_step += 1

    # morality_metric, morality_functions, avg_return, info = \
    #     mt.evaluate_morality_metric(policy=random_policy, env=env, max_episode_steps=1000, n_repeats=100,
    #                                 handle_trunc='include', is_prog_bar=True)
    #
    # print(f"Morality metric: {morality_metric}")
    # print(f"Morality functions:")
    # pprint.pprint(morality_functions)
    # print(f"Avg. return: {avg_return}")
    # print("Info:")
    # pprint.pprint(info)



if __name__ == '__main__':
    main()
