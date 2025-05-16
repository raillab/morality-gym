import json
import os
from enum import IntEnum
from typing import Optional, Callable, Union

import matplotlib.pyplot as plt
import networkx as nx

from morality_gym.environment.state import NormSet
from morality_gym.utils.common import join_paths

BASE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_PATH = join_paths([BASE_PATH, "configs"])

class MoralityTree:
    def __init__(
            self,
            config_name: str,
    ):
        self._T = nx.DiGraph()
        self._n_nodes = 0
        self.norm_set_to_node = {}

        self.load_from_config(config_name)

    def load_from_config(self, name):
        config_path = join_paths([CONFIGS_PATH, name])
        if not os.path.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist.")
        with open(config_path, "r") as f:
            json_d = json.load(f)
            tree_arr = json_d["head"]

        self._T.add_node("head")
        # head_node = self._T.nodes["head"]
        self._n_nodes = 0

        def add_children(node, child_arr):
             # curr_order
            for child in child_arr:
                self._n_nodes += 1

                norm_type = child[0]
                order = self._n_nodes
                if norm_type in ["outcome", "option", "option-outcome"]:
                    norm_set = child[1]
                    deontic_modality = child[2]
                    norm = Norm(norm_type, order, norm_set, deontic_modality)
                    self.norm_set_to_node[norm.norm_set] = norm

                elif norm_type == "utility":
                    raise NotImplementedError("Utility norms are not yet implemented.")
                else:
                    raise ValueError(f"Invalid norm type: {norm_type}. Valid types are: [outcome, option, "
                                     f"option-outcome, utility]")

                self._T.add_node(norm)
                self._T.add_edge(node, norm)

                self.norm_set_to_node[norm.norm_set] = norm

                add_children(
                    norm,
                    child[4]
                )

        add_children("head", tree_arr)

    def draw(self):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        nx.draw_networkx(
            self._T,
            pos=nx.spring_layout(self._T),
            with_labels=True,
            ax=ax
        )
        return fig, ax

    def contains_norm_set(self, norm_set):
        return norm_set in self.norm_set_to_node

    @property
    def n_nodes(self):
        return self._n_nodes


class DeonticModality(IntEnum):
    Prohibit = 0
    Prescribe = 1


# ...
class Norm:
    def __init__(
            self,
            norm_type: str,
            order: int,
            norm_set: Optional[Union[str, NormSet]] = None,
            deontic_modality: Optional[Union[DeonticModality, str]] = None,
            utility_fn: Optional[Callable] = None
    ):
        self.norm_type = norm_type
        self.utility_fn = utility_fn

        if order <= 0:
            raise ValueError("Order must be positive and non-zero.")
        self.order = order

        if isinstance(deontic_modality, str):
            valid_enum_names = [x.name for x in DeonticModality]
            if deontic_modality not in valid_enum_names:
                raise ValueError(f"Invalid deontic modality: {deontic_modality}. "
                                 f"Valid deontic modalities are: {valid_enum_names}")
            deontic_modality = DeonticModality[deontic_modality]
        elif deontic_modality is None:
            pass
        else:
            raise TypeError(f"deontic_modality must of type None, str, or DeonticModality. "
                            f"type(deontic_modality) = {type(deontic_modality)}")

        self.deontic_modality = deontic_modality

        if isinstance(norm_set, str):
            valid_enum_names = [x.name for x in NormSet]
            if norm_set not in valid_enum_names:
                raise ValueError(f"Invalid norm set: {norm_set}. Valid sets are: {valid_enum_names}")
            norm_set = NormSet[norm_set]
        elif norm_set is None:
            pass
        else:
            raise TypeError(f"norm_set must of type None, str, or NormSet. type(norm_set) = {type(norm_set)}")
        self.norm_set = norm_set

        if norm_type in ["outcome", "option", "option-outcome"]:
            if deontic_modality is None:
                raise ValueError("Deontic modality must be specified for outcome, option, and option-outcome norms.")
        elif norm_type == "utility":
            if utility_fn is None:
                raise ValueError("Utility norm requires a utility function to be specified.")
        else:
            raise ValueError(f"Invalid norm type: {norm_type}. Valid types are: [outcome, option, option-outcome, utility]")

    def __hash__(self):
        return self.order

    def __repr__(self):
        norm_set = self.norm_set.name if self.norm_set is not None else None
        deontic_modality = self.deontic_modality.name if self.deontic_modality is not None else None
        return f"Norm - {self.norm_type}, {self.order}, {norm_set}, {deontic_modality}, {self.utility_fn}"

    def __str__(self):
        return self.__repr__()

def main():
    # moral_tree = MoralityTree("asimov_3_laws.json")
    moral_tree = MoralityTree("morality_med.json")
    fig, ax = moral_tree.draw()
    fig.savefig("tree.png")

if __name__ == "__main__":
    main()