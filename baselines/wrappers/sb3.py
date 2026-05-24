import copy
from typing import Dict, Any, Optional, List, Set, Tuple, SupportsFloat

import gymnasium

from baselines.cost import Cost
from morality_gym._morality_chain.morality_chain import MoralityChain


class SB3Wrapper(gymnasium.Wrapper):
    def __init__(
            self,
            env,
            morality_chain: MoralityChain,
            # salient_norms: Optional[Set[str]] = None,
            norm_reward: bool = False,
            shape_reward: bool = False,  # Whether to augment env reward with -cost
            cost_function_kwargs: Dict[str, Any] = None,
            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
    ):
        super().__init__(env)
        self._morality_chain = morality_chain

        self._shape_reward = shape_reward
        self._norm_reward = norm_reward
        self._return_bounds = return_bounds

        if self._norm_reward and self._return_bounds is None:
            raise ValueError("Return bounds must be provided if norm_reward is True.")

        if norm_reward:
            self._norm_fact = max(abs(self._return_bounds[0]), abs(self._return_bounds[1]))
        else:
            self._norm_fact = 1.0

        # if shape_reward:
        #     if _morality_chain is None:
        #         raise ValueError("Morality chain must be provided to shape reward.")
        #
        #     if cost_function_kwargs is None:
        #         cost_function_kwargs = {}
        #
        #     self._cost = Cost(self._morality_chain, **cost_function_kwargs)
        # else:
        #     self._cost = None

        if cost_function_kwargs is None:
            cost_function_kwargs = {}
        self._cost = Cost(self._morality_chain, **cost_function_kwargs)

        # Get Inner variables if they exist #
        if isinstance(env, gymnasium.Wrapper):
            unwrapped_env = env.unwrapped
        else:
            unwrapped_env = env

        if hasattr(unwrapped_env, "valid_actions"):
            self.valid_actions = unwrapped_env.valid_actions
        else:
            self.valid_actions = None

        if hasattr(unwrapped_env, "curr_step"):
            self.curr_step = unwrapped_env.curr_step
        else:
            self.curr_step = None

        # Handle stuff :)
        self._last_obs = None
        self._info = None
        self._was_term = False


    def step(self, action):
        obs, orig_reward, is_term, is_trunc, info = self.env.step(action)

        reward = copy.copy(orig_reward)
        if self._norm_reward:
            reward = reward / self._norm_fact

        cost_val = self._cost(info)
        info["cost"] = cost_val

        if self._shape_reward:
            info["orig_reward"] = reward

            reward -= cost_val
            if self._norm_reward:
                # Re-Normalise so reward is still approximately between -1 and 1
                reward = reward/(1 + self._cost.scale_fact)

        info["orig_reward"] = orig_reward

        return obs, reward, is_term, is_trunc, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self._shape_reward:
            self._cost.reset()

        self._last_obs = obs
        self._info = info

        return obs, info


#######################
# --- INTERACTIVE --- #
#######################
def interactive(
        env: gymnasium.Wrapper,
        reset_kwargs: Optional[Dict[str, Any]] = None
):
    import pygame
    import pprint
    from morality_gym._environments.core.action import ActionEnum

    if reset_kwargs is None:
        reset_kwargs = {}

    obs, info = env.reset(**reset_kwargs)
    print("###############")
    print("###############")
    print("# -- RESET -- #")
    print("###############")
    print("###############")
    print(f"obs = ")
    pprint.pp(obs, compact=True)
    print(f"info = ")
    pprint.pp(info, compact=True)
    print("###############")


    env.render()

    key_action_map = {
        pygame.K_w: ActionEnum.UP,
        pygame.K_s: ActionEnum.DOWN,
        pygame.K_a: ActionEnum.LEFT,
        pygame.K_d: ActionEnum.RIGHT,
        pygame.K_SPACE: ActionEnum.STAY,
        pygame.K_q: ActionEnum.INIT_DIALOGUE,
        pygame.K_e: ActionEnum.INTERACT
    }

    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key in key_action_map:
                    action = key_action_map[event.key]
                    if action in env.valid_actions:
                        obs, reward, is_term, is_trunc, info = env.step(action)

                        print(f"\n##################")
                        print(f"# -- STEP {env.curr_step:03} -- #")
                        print(f"##################")
                        print(f"obs = ")
                        pprint.pp(obs, compact=True)

                        print(f"reward = {reward}")

                        print(f"is_term = {is_term}")
                        print(f"is_trunc = {is_trunc}")

                        print(f"info = ")
                        pprint.pp(info, compact=True)
                        print(f"##################")



                        env.render()
                elif event.key == pygame.K_r:
                    obs, info = env.reset(**reset_kwargs)
                    print("\n###############")
                    print("###############")
                    print("# -- RESET -- #")
                    print("###############")
                    print("###############")
                    print(f"obs = ")
                    pprint.pp(obs, compact=True)
                    print(f"info = ")
                    pprint.pp(info, compact=True)
                    print("###############")
                    env.render()
                elif event.key == pygame.K_ESCAPE:
                    is_running = False
                    break
                pass

            if not is_running:
                break

            pygame.time.wait(50)

    env.close()
#######################


def main():
    from morality_gym._setup.setup import make  # as env_mt_make
    env_kwargs = {
        "scenario_overrides": {"seed": 42},
        "env_overrides": {
            "obs_type": dict,
            "is_normalise_obs": True,
            "render_mode": "human",
            # "step_penalty": 0
        }
    }

    env, mt = make(
        # env_id='PushSelfSacrifice-All-v1',
        env_id='SwitchStandard-Standard-v1',
        # env_id='MoralityGym/Trolley-PushStandard-0-v0',
        # morality_chain_id='Trolley-Common-Utilitarian-UtilityHarm-v0',
        morality_chain_id='Utility',
        env_kwargs=env_kwargs
    )
    cost_function_kwargs = {
        "scale_fact": 100.0,
        "salient_norms": {"n_human_harm"}
    }
    env = SB3Wrapper(
        env, mt, shape_reward=True,
        cost_function_kwargs=cost_function_kwargs
    )
    interactive(env)

if __name__ == "__main__":
    main()
