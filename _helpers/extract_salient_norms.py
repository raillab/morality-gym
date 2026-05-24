import json
import os

import pandas as pd

from morality_gym._utils.common import join_paths

NORM_MAP = {
    "n_human_harm": "MHH", # Min Human Harm
    # "human_harm": "AHH", # Avoid Human Harm
    "personal_action_caused_human_harm": "APHH", # Avoid Personal Human Harm
    "n_animal_harm": "MAH",
    # "animal_harm": "AAH",
    "personal_action_caused_animal_harm": "APAH",
    "n_robot_harm": "MRH",
    # "robot_harm": "ARH",
    "personal_action_caused_robot_harm": "APRH",
    "agent_harm": "AAgH",
}

def extract_salient_norms():
    dir_path = r"C:\Users\User\Documents\Code\morality-gym-tabular\morality_gym\_environments\trolley\configs\variants"
    salient_norms = {}
    salient_norms_flat = {
        "scenario": [],
        "variant": [],
        "salient_norms": [],
    }

    for f_name in os.listdir(dir_path):
        f_path = join_paths([dir_path, f_name])
        scenario = f_name.replace("-v1.json", "")

        curr_salient_norms = {}

        with open(f_path, "r") as f:
            d = json.load(f)

        for variant, inner_d in d.items():
            norm_arr = []
            for norm in inner_d["salient_norms"]:
                if norm in NORM_MAP:
                    norm_arr.append(NORM_MAP[norm])
            curr_salient_norms[variant] = norm_arr

            salient_norms_flat["scenario"].append(scenario)
            salient_norms_flat["variant"].append(variant)
            salient_norms_flat["salient_norms"].append(";".join(curr_salient_norms[variant]))

        salient_norms[scenario] = curr_salient_norms

    df = pd.DataFrame.from_dict(salient_norms_flat)
    df.to_csv("salient_norms.csv", index=False)

if __name__ == '__main__':
    extract_salient_norms()