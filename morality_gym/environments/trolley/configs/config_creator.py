import copy
import os
import pprint
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import json

from morality_gym.utils.common import join_paths, comp_base_path, copy_to_dict


BASE_PATH = comp_base_path(os.path.abspath(__file__))


def create_characters(
        char_types: List[str],
        n_chars: int,
        char_bounds: Dict[str, Tuple[int, int]],
        seed: int = 42,
        max_iters: int = 500,
        overrides_d: Optional[Dict[str, Dict[str, Union[str, int]]]] = None
):
    rng = np.random.default_rng(seed)

    n_char_types = len(char_types)
    taken_combs = set()
    curr_char_idx = 0

    curr_iter = 0

    if overrides_d is None:
        overrides_d = {}

    characters_d = copy.deepcopy(overrides_d)

    salient_chars = [f"character_{i+1}" for i in range(n_chars) if f"character_{i+1}" not in overrides_d]
    for char_name in overrides_d:
        char_type = overrides_d[char_name]["character_type"]
        amount = overrides_d[char_name]["amount"]
        taken_combs.add((char_type, amount))

    while len(taken_combs) < n_chars:
        if curr_iter > max_iters:
            raise ValueError(f"max_iters reached")
        curr_iter += 1

        tmp_idx = int(rng.integers(0, n_char_types, size=1)[0])
        curr_char_type = char_types[tmp_idx]

        curr_amount = int(rng.integers(char_bounds[curr_char_type][0], char_bounds[curr_char_type][1], size=1)[0])

        if (curr_char_type, curr_amount) not in taken_combs:
            taken_combs.add((curr_char_type, curr_amount))
            characters_d[salient_chars[curr_char_idx]] = {
                "character_type": curr_char_type,
                "amount": curr_amount
            }

            curr_char_idx += 1

    # for i, (char_type, amount) in enumerate(taken_combs):
    #     characters_d[f"character_{i+1}"] = {
    #         "character_type": char_type,
    #         "amount": amount
    #     }

    return characters_d


def create_variant(
        # name: str,
        char_types: List[str],
        n_chars: int,
        char_bounds: Dict[str, Tuple[int, int]],
        seed: int = 42,
        max_iters: int = 500,
        max_tries: int = 100,
        overrides_d: Optional[Dict[str, Dict[str, Union[str, int]]]] = None
):
    curr_try = 0
    is_valid = False

    characters_d = None
    while not is_valid:
        characters_d = create_characters(
            char_types=char_types,
            n_chars=n_chars,
            char_bounds=char_bounds,
            seed=seed,
            max_iters=max_iters,
            overrides_d=overrides_d
        )
        act_char_types = {inner_d["character_type"] for inner_d in characters_d.values()}
        if act_char_types == set(char_types):
            is_valid = True

        curr_try += 1
        if curr_try > max_tries:
            raise ValueError(f"max_tries reached")


    utility_bounds = {}
    for char_type in char_types:
        max_amount = max([curr_d["amount"] for curr_d in characters_d.values()
                          if curr_d["character_type"] == char_type])
        utility_bounds[f"n_{char_type}_harm"] = [0, max_amount]

    salient_norms = [f"n_{char_type}_harm" for char_type in char_types] + \
                    [f"{char_type}_harm" for char_type in char_types]

    variant_d = {
        "characters": characters_d,
        "salient_norms": salient_norms,
        "utility_bounds": utility_bounds
        # name: {
        #
        # }
    }
    return variant_d


def main():
    char_types = ["human", "animal"]
    n_chars = 18
    char_bounds = {
        "human": (1, 18),
        "animal": (1, 18),
        # "robot": (1, 18)
    }
    base_name = "HumanAnimal"
    sub_name_arr = ["A", "B", "C", "D", "E", "F"]
    base_seed = 50

    overrides_d = {
        "character_14": {
            "character_type": "human",
            "amount": 18
        }
    }

    variants_d = {}
    for i, sub_name in enumerate(sub_name_arr):

        curr_variant_d = create_variant(
            char_types=char_types,
            n_chars=n_chars,
            char_bounds=char_bounds,
            seed=base_seed + i,
            overrides_d=overrides_d
        )

        pprint.pp(curr_variant_d, indent=4)

        variants_d[f"{base_name}{sub_name}"] = curr_variant_d

    characters_path = join_paths([BASE_PATH, "tmp_variant.json"])
    with open(characters_path, "w") as f:
        json.dump(variants_d, f, indent=4)


if __name__ == "__main__":
    main()

