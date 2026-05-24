import json
import os

import pandas as pd

from morality_gym._utils.common import join_paths

# NORM_MAP = {
#     "n_human_harm": "MHH", # Min Human Harm
#     # "human_harm": "AHH", # Avoid Human Harm
#     "personal_action_caused_human_harm": "APHH", # Avoid Personal Human Harm
#     "n_animal_harm": "MAH",
#     # "animal_harm": "AAH",
#     "personal_action_caused_animal_harm": "APAH",
#     "n_robot_harm": "MRH",
#     # "robot_harm": "ARH",
#     "personal_action_caused_robot_harm": "APRH",
#     "agent_harm": "AAgH",
# }

def extract_sv_info():
    dir_path = r"C:\Users\User\Documents\Code\morality-gym-tabular\morality_gym\_environments\trolley\configs\variants"
    sv_info = {}

    for f_name in os.listdir(dir_path):
        f_path = join_paths([dir_path, f_name])
        scenario = f_name.replace("-v1.json", "")

        # Info for curr scenario - i.e. for all variants
        curr_s_info= {}

        with open(f_path, "r") as f:
            d = json.load(f)

        for variant, inner_d in d.items():
            curr_v_info = {
                "salient_norms": inner_d["salient_norms"],
                "utility_bounds": inner_d["utility_bounds"],
            }

            curr_s_info[variant] = curr_v_info

        sv_info[scenario] = curr_s_info

    with open("sv_info.json", "w") as f:
        json.dump(sv_info, f, indent=1)

if __name__ == '__main__':
    extract_sv_info()