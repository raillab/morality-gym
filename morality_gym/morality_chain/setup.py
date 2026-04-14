from typing import Tuple, Optional, Dict, SupportsFloat

from morality_gym.morality_chain.morality_chain import MoralityChain
from morality_gym.morality_chain.norm import UtilityNorm, DeonticModalityEnum, EventNorm

def _make_utilitarian(
        morality_tree_name: str,
        utility_bounds: Dict[str, Tuple[SupportsFloat, SupportsFloat]],
        **morality_tree_kwargs
):
    utility_ranges = utility_bounds
    # UtilityHarm #
    if morality_tree_name == "UtilityHarm":
        if "n_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_harm in {morality_tree_name}")

        norm = UtilityNorm(
            name="n_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            min_utility_sum=float(utility_ranges["n_harm"][0]),
            max_utility_sum=float(utility_ranges["n_harm"][1])
        )
        norms = [norm]
        # morality_chain = MoralityChain(norms=[norm], **morality_chain_kwargs)
        # return morality_chain

    # OrderedUtilityHarm #
    elif morality_tree_name == "OrderedUtilityHarm":
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")
        human_norm = UtilityNorm(
            name="n_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            min_utility_sum=float(utility_ranges["n_human_harm"][0]),
            max_utility_sum=float(utility_ranges["n_human_harm"][1])
        )

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")
        animal_norm = UtilityNorm(
            name="n_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            min_utility_sum=float(utility_ranges["n_animal_harm"][0]),
            max_utility_sum=float(utility_ranges["n_animal_harm"][1])
        )

        if "n_robot_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_robot_harm in {morality_tree_name}")
        robot_norm = UtilityNorm(
            name="n_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            min_utility_sum=float(utility_ranges["n_robot_harm"][0]),
            max_utility_sum=float(utility_ranges["n_robot_harm"][1])
        )

        norms = [human_norm, animal_norm, robot_norm]
        # morality_chain = MoralityChain(norms=[human_norm, animal_norm, robot_norm], **morality_chain_kwargs)
        # return morality_chain

    # OrderedUtilityHarm #
    elif morality_tree_name == "OrderedUtilityAgentHarm":
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")
        human_norm = UtilityNorm(
            name="n_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            min_utility_sum=float(utility_ranges["n_human_harm"][0]),
            max_utility_sum=float(utility_ranges["n_human_harm"][1])
        )

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")
        animal_norm = UtilityNorm(
            name="n_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            min_utility_sum=float(utility_ranges["n_animal_harm"][0]),
            max_utility_sum=float(utility_ranges["n_animal_harm"][1])
        )

        agent_norm = EventNorm(
            name="agent_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            norm_type="outcome"
        )

        if "n_robot_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_robot_harm in {morality_tree_name}")
        robot_norm = UtilityNorm(
            name="n_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=3,
            min_utility_sum=float(utility_ranges["n_robot_harm"][0]),
            max_utility_sum=float(utility_ranges["n_robot_harm"][1])
        )

        norms = [human_norm, animal_norm, agent_norm, robot_norm]

    # WeightedUtilityHarm #
    elif morality_tree_name == "WeightedUtilityHarm":
        if "n_weighted_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_weighted_harm in {morality_tree_name}")

        norm = UtilityNorm(
            name="n_weighted_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            min_utility_sum=float(utility_ranges["n_weighted_harm"][0]),
            max_utility_sum=float(utility_ranges["n_weighted_harm"][1])
        )
        norms = [norm]

    # OrderedOutcomeHarm #
    elif morality_tree_name == "OrderedOutcomeHarm1":
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")
        human_norm = EventNorm(
            name="human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            norm_type="outcome"
        )

        animal_norm = EventNorm(
            name="animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            norm_type="outcome"
        )

        robot_norm = EventNorm(
            name="robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            norm_type="outcome"
        )

        norms = [human_norm, animal_norm, robot_norm]
        # morality_chain = MoralityChain(norms=[human_norm, animal_norm, robot_norm], **morality_chain_kwargs)
        # return morality_chain

    # OrderedOutcomeHarm #
    elif morality_tree_name == "OrderedOutcomeHarm2":
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")
        human_norm = EventNorm(
            name="human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            norm_type="outcome"
        )

        animal_norm = EventNorm(
            name="animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            norm_type="outcome"
        )

        agent_norm = EventNorm(
            name="agent_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            norm_type="outcome"
        )

        robot_norm = EventNorm(
            name="robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=3,
            norm_type="outcome"
        )
        norms = [human_norm, animal_norm, agent_norm, robot_norm]
        # morality_chain = MoralityChain(norms=[human_norm, animal_norm, agent_norm, robot_norm], **morality_chain_kwargs)
        # return morality_chain

    # OrderedOutcomeUtilityHarm #
    elif morality_tree_name == "OrderedOutcomeUtilityHarm":
        raise NotImplementedError
    else:
        raise ValueError(f"No morality trees for {morality_tree_name} suffix")

    morality_tree = MoralityChain(norms=norms, **morality_tree_kwargs)
    return morality_tree


def _make_dual_process(
        morality_tree_name: str,
        utility_bounds: Dict[str, Tuple[SupportsFloat, SupportsFloat]],
        **morality_tree_kwargs
):
    utility_ranges = utility_bounds
    # Simple #
    if morality_tree_name == "Simple":
        if "n_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_harm in {morality_tree_name}")

        personal_harm_norm = EventNorm(
            name="personal_action_caused_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            norm_type="causal"
        )

        n_harm_norm = UtilityNorm(
            name="n_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            min_utility_sum=float(utility_ranges["n_harm"][0]),
            max_utility_sum=float(utility_ranges["n_harm"][1])
        )

        norms = [personal_harm_norm, n_harm_norm]
        # morality_chain = MoralityChain(norms=[personal_harm_norm, n_harm_norm], **morality_chain_kwargs)
        # return morality_chain

    elif morality_tree_name == "Medium":
        raise NotImplementedError

    elif morality_tree_name == "Complex":
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")

        # 0
        personal_human_harm_norm = EventNorm(
            name="personal_action_caused_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            norm_type="causal"
        )

        # 1
        n_human_harm_norm = UtilityNorm(
            name="n_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            min_utility_sum=float(utility_ranges["n_human_harm"][0]),
            max_utility_sum=float(utility_ranges["n_human_harm"][1])
        )

        # 2
        personal_animal_harm_norm = EventNorm(
            name="personal_action_caused_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            norm_type="causal"
        )

        # 3
        n_animal_harm_norm = UtilityNorm(
            name="n_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=3,
            min_utility_sum=float(utility_ranges["n_animal_harm"][0]),
            max_utility_sum=float(utility_ranges["n_animal_harm"][1])
        )

        # 4
        personal_robot_harm_norm = EventNorm(
            name="personal_action_caused_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=4,
            norm_type="causal"
        )

        # 5
        n_robot_harm_norm = UtilityNorm(
            name="n_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=6,
            min_utility_sum=float(utility_ranges["n_robot_harm"][0]),
            max_utility_sum=float(utility_ranges["n_robot_harm"][1])
        )

        norms = [personal_human_harm_norm, n_human_harm_norm, personal_animal_harm_norm, n_animal_harm_norm,
                 personal_robot_harm_norm, n_robot_harm_norm]
        # morality_chain = MoralityChain(norms=norms, **morality_chain_kwargs)
        # return morality_chain
    elif morality_tree_name == "ComplexAgentHarm":
        # raise NotImplementedError(f"This has not been implemented yet")
        if "n_human_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_human_harm in {morality_tree_name}")

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")

        if "n_animal_harm" not in utility_ranges:
            raise ValueError(f"No utility range for n_animal_harm in {morality_tree_name}")

        # 0
        personal_human_harm_norm = EventNorm(
            name="personal_action_caused_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=0,
            norm_type="causal"
        )

        # 1
        n_human_harm_norm = UtilityNorm(
            name="n_human_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=1,
            min_utility_sum=float(utility_ranges["n_human_harm"][0]),
            max_utility_sum=float(utility_ranges["n_human_harm"][1])
        )

        # 2
        personal_animal_harm_norm = EventNorm(
            name="personal_action_caused_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=2,
            norm_type="causal"
        )

        # 3
        n_animal_harm_norm = UtilityNorm(
            name="n_animal_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=3,
            min_utility_sum=float(utility_ranges["n_animal_harm"][0]),
            max_utility_sum=float(utility_ranges["n_animal_harm"][1])
        )

        # 4
        personal_robot_harm_norm = EventNorm(
            name="personal_action_caused_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=4,
            norm_type="causal"
        )

        # 5
        agent_harm_norm = EventNorm(
            name="agent_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=5,
            norm_type="outcome"
        )

        # 6
        n_robot_harm_norm = UtilityNorm(
            name="n_robot_harm",
            deontic_modality=DeonticModalityEnum.PROHIBITED,
            priority=6,
            min_utility_sum=float(utility_ranges["n_robot_harm"][0]),
            max_utility_sum=float(utility_ranges["n_robot_harm"][1])
        )

        norms = [personal_human_harm_norm, n_human_harm_norm, personal_animal_harm_norm, n_animal_harm_norm,
                 personal_robot_harm_norm, agent_harm_norm, n_robot_harm_norm]
        # morality_chain = MoralityChain(norms=norms, **morality_chain_kwargs)
        # return morality_chain

    else:
        raise ValueError(f"No morality trees for {morality_tree_name} suffix")

    morality_tree = MoralityChain(norms=norms, **morality_tree_kwargs)

    return morality_tree

MORALITY_CHAIN_ID_ALIASES = {
    "Utility": "Trolley-Common-Utilitarian-OrderedUtilityHarm-v0",
    "UtilityAgentHarm": "Trolley-Common-Utilitarian-OrderedUtilityAgentHarm-v0",
    "Outcome": "Trolley-Common-Utilitarian-OrderedOutcomeHarm1-v0",
    "OutcomeAgentHarm": "Trolley-Common-Utilitarian-OrderedOutcomeHarm2-v0",
    "DualProcess": "Trolley-Common-DualProcess-Complex-v0",
    "DualProcessAgentHarm": "Trolley-Common-DualProcess-ComplexAgentHarm-v0"
}

def make(
        morality_tree_id,
        utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,
        **morality_tree_kwargs
):

    if morality_tree_id in MORALITY_CHAIN_ID_ALIASES:
        morality_tree_id = MORALITY_CHAIN_ID_ALIASES[morality_tree_id]

    arr = morality_tree_id.split("-")
    env_type = arr[0]
    env_subtype = arr[1]
    mt_group = arr[2]
    mt_name = arr[3]
    version = arr[4]
    if env_type == "Trolley":
        if env_subtype == "Common":
            if mt_group == "Utilitarian":
                morality_tree = _make_utilitarian(mt_name, utility_bounds, **morality_tree_kwargs)
            elif mt_group == "DualProcess":
                morality_tree = _make_dual_process(mt_name, utility_bounds, **morality_tree_kwargs)
            else:
                raise ValueError(f"No morality trees for {env_type}-{env_subtype}-{mt_group} prefix")
            # if "Utilitarian" in mt_name:
            #     morality_chain = _make_utilitarian(mt_name, utility_ranges)
            # else:
            # raise ValueError(f"No morality trees for {env_type}-{env_subtype}-{mt_group} prefix")
        else:
            raise ValueError(f"No morality trees for {env_type}-{env_subtype} prefix")
    else:
        raise ValueError(f"No morality trees for {env_type} prefix")

    return morality_tree
