import json
import os

from morality_gym._utils.common import comp_base_path, join_paths

BASE_PATH = comp_base_path(os.path.abspath(__file__), level=1)


def load_base_paths():
    paths_file_path = join_paths([BASE_PATH, "paths.json"])
    if not os.path.exists(paths_file_path):
        paths_d = {}

        print(" >>> Set Base Paths <<< ")
        # Setup
        paths_d["runs"] = input("Enter runs base path: ")
        paths_d["slurms"] = input("Enter slurms base path: ")
        paths_d["logs"] = input("Enter logs base path: ")
        paths_d["zipped_logs"] = input("Enter zipped logs base path: ")
        paths_d["run_logs"] = input("Enter run logs base path: ")
        paths_d["slurm_logs"] = input("Enter slurm logs base path: ")

        with open(paths_file_path, "w") as f:
            json.dump(paths_d, f, indent=4)
        print(" >>> Set Base Paths Completed <<< ")

    else:
        with open(paths_file_path, "r") as f:
            paths_d = json.load(f)

    return paths_d