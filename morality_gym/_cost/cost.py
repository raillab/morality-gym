from collections import defaultdict
from typing import Optional, List, Set, Union, Tuple

from typing_extensions import SupportsFloat

from morality_gym._morality_chain.morality_chain import MoralityChain
from morality_gym._morality_chain.norm import UtilityNorm


class Cost:
    def __init__(
            self,
            morality_chain: MoralityChain,
            scale_fact: float = 1.0,
            is_normalise_utility: bool = True,
            information_mode: str = "minimal",  # How much information is available for cost function to use
            scalarisation: str = "expert"  # ["expert", "linear"]  - expert corresponds to
    ):
        self._morality_chain = morality_chain
        self._scalarisation = scalarisation

        self._is_normalise_utility = is_normalise_utility

        if information_mode == "full":
            # Salient norms known
            # Norm specific utility bounds known
            if morality_chain.salient_norms is None:
                raise ValueError("_morality_chain.salient_norms cannot be None when information_mode='full'")
            self._salient_norms = morality_chain.salient_norms
            self._utility_bounds = self._morality_chain.utility_bounds

        elif information_mode == "partial":
            # Salient norms not known - uses all norms
            # Norm specific utility bounds known
            self._salient_norms = set(self._morality_chain.norm_names)
            self._utility_bounds = self._morality_chain.utility_bounds
        elif information_mode == "minimal":
            # Salient norms not known - uses all norms
            # Norm specific utility bounds not known - uses provided global bounds
            if self._morality_chain.global_utility_bounds is None:
                raise ValueError("_morality_chain.global_utility_bounds cannot be None when information_mode='minimal'")

            self._salient_norms = set(self._morality_chain.norm_names)
            self._utility_bounds = self._morality_chain.global_utility_bounds
        else:
            raise ValueError(f"information_mode must be one of 'full', 'partial' or 'minimal'. Got {information_mode}")

        self._weights = None
        self._set_salient_norms(self._salient_norms)

        self._norm_event_occurred = defaultdict(lambda: False)
        self._scale_fact = scale_fact

    def _set_salient_norms(self, salient_norms: Set[str]):
        self._salient_norms = salient_norms
        if self._scalarisation == "expert":
            self._weights = self._morality_chain.compute_weights(self._salient_norms)
        elif self._scalarisation == "linear":
            n_norms = len(self._morality_chain.norms)
            self._weights = {
                norm_name: (n_norms - self._morality_chain[norm_name].priority)
                for norm_name in self._salient_norms
            }
        else:
            raise ValueError(f"scalarisation must be one of 'expert' or 'linear'. Got {self._scalarisation}")

    def _normalise_utility(
            self,
            norm: UtilityNorm,
            utility_val: float
    ):
        min_val = self._utility_bounds[norm.name][0]
        max_val = self._utility_bounds[norm.name][1]
        if min_val == max_val:
            norm_val = 0
        else:
            norm_val = (utility_val - min_val)/(max_val - min_val)
        return norm_val

    def __call__(self, info, is_term=False):
        if "norm_events" not in info:
            raise ValueError("norm_events not found in info")

        norm_events = info["norm_events"]

        # if salient norms specified in info field & they changed
        if "salient_norms" in info and info["salient_norms"] != self._salient_norms:
            self._set_salient_norms(info["salient_norms"])

        cost_val = 0
        for norm_category in ["action", "outcome", "causal"]:
            for norm_name in norm_events[norm_category]:
                if norm_name in self._salient_norms:
                    if not self._norm_event_occurred[norm_name]:
                        cost_val += self._weights[norm_name]
                        self._norm_event_occurred[norm_name] = True

        for norm_name, norm_val in norm_events["utility"].items():
            if norm_name in self._salient_norms:
                if self._is_normalise_utility:
                    norm_val = self._normalise_utility(self._morality_chain[norm_name], norm_val)
                cost_val += norm_val * self._weights[norm_name]

        cost_val = cost_val/sum(self._weights.values())
        return cost_val * self._scale_fact

    @property
    def scale_fact(self):
        return self._scale_fact

    def reset(self):
        self._norm_event_occurred = defaultdict(lambda: False)