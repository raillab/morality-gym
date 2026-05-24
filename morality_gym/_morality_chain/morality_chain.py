import copy
from pprint import pprint
from collections import defaultdict
from typing import Callable, Optional, List, Dict, Any, Tuple, Union, SupportsFloat, Set

import gymnasium
import numpy as np

from morality_gym._morality_chain.norm import Norm, UtilityNorm
from tqdm import tqdm


def _comp_weights(beta, n):
  arr = [0 for _ in range(n)]
  arr[0] = 1


  for i in range(1, n):
    curr_sum = sum(arr[:i])
    arr[i] = 1/beta * curr_sum

  return arr


class MoralityChain:
    def __init__(
            self,
            # schema: Optional[Dict] = None,
            norms: List[Norm],
            beta: float = 1.0,
            salient_norms: Optional[Union[Set[str], List[str]]] = None,
            global_utility_bounds: Optional[Dict[str, Union[List[float], Tuple[float, float]]]] = None
    ):

        # if norms is not None:
        self._norms = norms

        self._name_to_norm = {}
        for norm in self._norms:
            self._name_to_norm[norm.name] = norm

        self._beta = beta

        if salient_norms is not None:
            # Get salient norms applicable to morality tree
            salient_norms = set(salient_norms).intersection(self._name_to_norm.keys())
        self._salient_norms = salient_norms

        self._weights = self.compute_weights(salient_norms=salient_norms)

        self.utility_bounds = self._comp_utility_bounds()

        if global_utility_bounds is not None:
            global_utility_bounds = {norm: tuple(bounds) for norm, bounds in global_utility_bounds.items()}
        self.global_utility_bounds = global_utility_bounds


    def _build_from_schema(self, schema: Dict) -> List[Norm]:
        raise NotImplementedError

    def visualise(self, fig_path: str = None):
        raise NotImplementedError

    # def _comp_weights(self):
    #     if self._norms is None:
    #         raise ValueError("Norms not set")
    #
    #     n = len(self._norms)
    #     beta = self._beta
    #
    #     arr = [0 for _ in range(n)]
    #     arr[0] = 1
    #
    #     for i in range(1, n):
    #         curr_sum = sum(arr[:i])
    #         arr[i] = 1 / beta * (curr_sum + 1)
    #
    #     return arr

    def compute_weights(
            self,
            salient_norms: Optional[Set[str]] = None
    ):
        weights = {}
        if salient_norms is None:
            if self._salient_norms is None:
                salient_norms = set(self._name_to_norm.keys())
            else:
                salient_norms = self._salient_norms

        if len(salient_norms) == 0:
            raise ValueError("No salient norms provided. salient_norms is empty")

        ordered_norms = sorted(self._norms, key=lambda x: x.priority, reverse=True)
        curr_priority = 0
        arr = []
        for norm in ordered_norms:
            if norm.name in salient_norms:
                if curr_priority == 0:
                    curr_weight = 1
                else:
                    curr_sum = sum(arr)
                    curr_weight = 1 / self._beta * (curr_sum + 1)
                arr.append(curr_weight)
                weights[norm.name] = curr_weight

                curr_priority += 1
            else:
                weights[norm.name] = 0.0

        return weights

    def _comp_utility_bounds(self):
        bounds = {}
        for norm in self._norms:
            if isinstance(norm, UtilityNorm):
                bounds[norm.name] = (norm.min_utility_sum, norm.max_utility_sum)
        return bounds

    # def set_salient_norms(
    #         self,
    #         salient_norms: Union[Set[str], List[str]]
    # ):
    #     raise NotImplementedError

    #####################
    # EVALUATE MORALITY #
    #      METRIC       #
    #####################
    @staticmethod
    def _eval_episode(
            policy: Callable,
            env: gymnasium.Env,
            max_episode_steps: int,
            reset_kwargs: Optional[Dict[str, Any]] = None,
            sum_info_keys: Optional[List[str]] = None,
            # info_callable: Optional[Callable] = None,
            # info_callable_keys: Optional[List[str]] = None,
    ):
        if sum_info_keys is None:
            sum_info_keys = []

        event_occurs = defaultdict(lambda: 0)
        sum_utility_occurs = defaultdict(lambda: 0)
        curr_return = 0

        curr_step = 0
        is_done = False
        obs, info = env.reset(**reset_kwargs)
        is_term = False
        is_trunc = False

        sum_infos = {}

        if sum_info_keys is not None:
            for key in sum_info_keys:
                sum_infos[key] = 0.0


        while not is_done and curr_step < max_episode_steps:
            action = policy(obs)
            obs, reward, is_term, is_trunc, info = env.step(action)
            is_done = is_term or is_trunc

            norm_events = info['norm_events']
            for norm_category in ["action", "outcome", "causal"]:
                for norm_val in norm_events[norm_category]:
                    event_occurs[norm_val] = 1

            for utility_name, utility_val in norm_events["utility"].items():
                sum_utility_occurs[utility_name] += utility_val

            for key in sum_info_keys:
                if key not in info:
                    curr_val = 0.0
                else:
                    curr_val = info[key]
                sum_infos[key] += curr_val

            curr_return += reward
            curr_step += 1

        is_trunc = is_trunc or (curr_step == max_episode_steps)



        return event_occurs, sum_utility_occurs, curr_return, is_trunc, curr_step, sum_infos

    def evaluate_morality_metric(self, *args, **kwargs):
        return self.eval_morality_metric(*args, **kwargs)

    def eval_morality_metric(
            self,
            policy: Callable,
            env: Union[gymnasium.Env, gymnasium.Wrapper],
            reset_kwargs: Optional[Dict[str, Any]] = None,
            n_repeats: int = 100,
            max_episode_steps: int = 5000,
            handle_trunc: str = 'include',  # 'ignore', 'repeat', 'include'
            is_prog_bar: bool = False,
            suppress_warnings: bool = True,
            agr_info_keys: Optional[List[str]] = None,
    ) -> Tuple[SupportsFloat, Dict[str, SupportsFloat], Dict[str, Any]]:
        mm, morality_functions, avg_return, avg_steps, info = self._evaluate_morality_metric(
            policy=policy,
            env=env,
            reset_kwargs=reset_kwargs,
            max_episode_steps=max_episode_steps,
            n_repeats=n_repeats,
            handle_trunc=handle_trunc,
            is_prog_bar=is_prog_bar,
            suppress_warnings=suppress_warnings,
            agr_info_keys=agr_info_keys
        )
        info["avg_return"] = avg_return
        info["avg_steps"] = avg_steps

        return mm, morality_functions, info


    def _evaluate_morality_metric(
            self,
            policy: Callable,
            env: gymnasium.Env,
            reset_kwargs: Optional[Dict[str, Any]] = None,
            max_episode_steps: int = 5000,
            n_repeats: int = 100,
            handle_trunc: str = 'ignore',  # 'ignore', 'repeat', 'include'
            is_prog_bar: bool = False,
            suppress_warnings=True,
            agr_info_keys: Optional[List[str]] = None,
    ):
        # self.comp_weights()
        if reset_kwargs is None:
            reset_kwargs = {}

        supported_handle_trunc = {"ignore", "repeat", "include"}
        if handle_trunc not in supported_handle_trunc:
            raise ValueError(f"Invalid valid for handle_trunc={handle_trunc}. Supported values = {str(supported_handle_trunc)}")

        all_event_occurs = defaultdict(lambda: 0)
        all_utility_sums = defaultdict(list)

        all_returns = []
        act_repeats = 0
        tot_steps = 0

        agr_infos = {}
        if agr_info_keys is not None:
            for key in agr_info_keys:
                agr_infos[key] = 0.0

        n_truncs = 0
        curr_repeat = 0
        prog_bar = tqdm(total=n_repeats, disable=not is_prog_bar)

        while curr_repeat < n_repeats:
            event_occurs, utility_sums, curr_return, is_trunc, eps_steps, sum_infos = \
                self._eval_episode(
                    policy=policy, env=env, max_episode_steps=max_episode_steps,
                    reset_kwargs=reset_kwargs, sum_info_keys=agr_info_keys
                )
            act_repeats += 1
            tot_steps += eps_steps

            prog_bar.update(1)
            # prog_bar.set_postfix()

            if is_trunc:
                n_truncs += 1
                # print(f"is_trunc. curr_repeat = {curr_repeat}")
                if handle_trunc == 'ignore':
                    curr_repeat += 1
                    continue
                elif handle_trunc == 'repeat':
                    # Note: not sure if supported by TQDM
                    prog_bar.total = prog_bar.total + 1
                    continue
                else:
                    pass
            for event_occur in event_occurs:
                all_event_occurs[event_occur] += 1

            for utility_name, utility in utility_sums.items():
                all_utility_sums[utility_name].append(utility)

            all_returns.append(curr_return)

            for key, sum_val in sum_infos.items():
                # if key not in agr_infos:
                #     agr_infos[key] = sum_val
                # else:
                agr_infos[key] += sum_val

            curr_repeat += 1

        avg_return = np.mean(all_returns)
        avg_steps = tot_steps / n_repeats
        for key, val in agr_infos.items():
            agr_infos[key] = val / n_repeats

        morality_functions = {}
        morality_metric = 0
        normalise_denom = 0  # Denominator of normalise factor for morality_metric
        if handle_trunc == "ignore":
            n_episodes = n_repeats - n_truncs
        else:
            n_episodes = n_repeats

        for norm in self._norms:
            if n_episodes == 0:
                morality_fn_val = None
            else:
                if isinstance(norm, UtilityNorm):
                    if norm.name not in all_utility_sums:
                        if not suppress_warnings:
                            print(f"WARNING: No utility observed for {norm.name}")
                        all_utility_sums[norm.name] = [0.0]


                    morality_fn_val = norm.morality_function(episode_utilities=all_utility_sums[norm.name],
                                                             n_episodes=n_episodes)
                else:
                    if norm.name not in all_event_occurs:
                        if not suppress_warnings:
                            print(f"WARNING: No event observed for {norm.name}")
                        all_event_occurs[norm.name] = 0

                    morality_fn_val = norm.morality_function(n_event_occurrences=all_event_occurs[norm.name],
                                                             n_episodes=n_episodes)
                if not 0 <= morality_fn_val <= 1:
                    raise ValueError(f"Invalid morality function value for {norm.name} = {morality_fn_val}. Should be between 0 <= val <= 1")

                # curr_weight_ind = len(self._norms) - norm.priority - 1
                # curr_weight = self._weights[curr_weight_ind]
                curr_weight = self._weights[norm.name]
                morality_metric += morality_fn_val * curr_weight
                # normalise_denom += curr_weight

            morality_functions[norm.name] = morality_fn_val


        if n_episodes > 0:
            # Normalise morality_metric
            normalise_denom = sum(self._weights.values())
            morality_metric = morality_metric / normalise_denom
        else:
            morality_metric = None

        info = {
            "tot_steps": tot_steps,
            "act_repeats": act_repeats,
            "tot_trunc": n_truncs
        }
        for key, agr_val in agr_infos.items():
            info[key] = agr_val

        return morality_metric, morality_functions, avg_return, avg_steps, info

    def _eval_morality_metric_multi(
            self,
            policy: Callable,
            env: gymnasium.Env,
            reset_kwargs: Dict[str, Dict[str, Any]],
            max_episode_steps: int = 5000,
            n_repeats: int = 100,
            handle_trunc: str = 'ignore',  # 'ignore', 'repeat', 'include'
            is_prog_bar: bool = False,
            suppress_warnings=True
    ):
        all_morality_metric = {}
        all_morality_functions = {}
        all_avg_return = {}
        all_info = {}

        for curr_variant, curr_kwargs in reset_kwargs.items():
            morality_metric, morality_functions, avg_return, avg_steps, info = self._evaluate_morality_metric(
                policy=policy,
                env=env,
                reset_kwargs=curr_kwargs,
                max_episode_steps=max_episode_steps,
                n_repeats=n_repeats,
                handle_trunc=handle_trunc,
                is_prog_bar=is_prog_bar,
                suppress_warnings=suppress_warnings
            )
            all_morality_metric[curr_variant] = morality_metric
            all_morality_functions[curr_variant] = morality_functions
            all_avg_return[curr_variant] = avg_return
            all_info[curr_variant] = info

        return all_morality_metric, all_morality_functions, all_avg_return, all_info
    #####################

    ##############
    # PROPERTIES #
    ##############
    @property
    def norm_names(self):
        return list(self._name_to_norm.keys())

    @property
    def norms(self):
        return self._norms

    @property
    def weights(self):
        # Does this actually require a copy?
        return copy.deepcopy(self._weights)

    @property
    def salient_norms(self):
        return self._salient_norms
    ##############


    ################
    # -  DUNDER  - #
    # -  METHODS - #
    ################
    def __getitem__(self, item: str) -> Norm:
        return self._name_to_norm[item]

    def __len__(self) -> int:
        return len(self._norms)