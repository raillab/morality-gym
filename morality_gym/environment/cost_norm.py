from morality_gym.morality_tree.morality_tree import MoralityTree


class MoralityCost:
    def __init__(self, morality_tree: MoralityTree):
        self.morality_tree = morality_tree

    def calc(self, info):
        mt = self.morality_tree
        norm_sets = info["norm_sets"]
        cost = 0.0

        # Outcome, option and option-outcome norms
        for norm_set in (norm_sets["outcome"] + norm_sets["option"] + norm_sets["option-outcome"]):
            if mt.contains_norm_set(norm_set):
                norm = mt.norm_set_to_node[norm_set]
                cost += mt.n_nodes - norm.order + 1

        # TODO: Utility
        return cost
