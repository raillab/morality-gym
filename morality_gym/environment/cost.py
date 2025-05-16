from morality_gym.morality_tree.morality_tree import MoralityTree


class MoralityCost:
    def __init__(
            self,
            morality_tree: MoralityTree,
            scale: float = 1.0
    ):
        self.morality_tree = morality_tree
        self._scale = scale

    # TODO: Change to only use obs
    def calc(self, info):
        mt = self.morality_tree
        danger_states = info["danger_states"]
        cost = 0.0

        # Convert to set to remove repeats
        for danger_state in set(danger_states):
            if mt.contains(danger_state):
                order = mt.get_order(danger_state)
                cost += mt.n_nodes - order + 1

        cost *= self._scale

        return cost
