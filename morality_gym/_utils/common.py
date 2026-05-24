import copy
import os
from typing import Dict, Any, Optional, List


def copy_to_dict(
        from_dict: Dict,
        to_dict: Dict,
        deepcopy_vals: bool = False
) -> Dict:
    if deepcopy_vals:
        for key in from_dict.keys():
            to_dict[key] = copy.deepcopy(from_dict[key])
    else:
        for key in from_dict.keys():
            to_dict[key] = from_dict[key]
    return to_dict

# Recursively copy sub dictionaries
def copy_to_dict_recr(
        from_dict: Dict,
        to_dict: Dict
):
    for key, val in from_dict.items():
        if isinstance(val, dict):
            if key not in to_dict.keys():
                to_dict[key] = {}
            copy_to_dict_recr(from_dict=val, to_dict=to_dict[key])
        else:
            to_dict[key] = copy.deepcopy(val)

def join_paths(paths: List[str]):
    return os.path.normpath(os.path.join(*paths))


def comp_base_path(path, level=1):
    curr_path = path
    for i in range(level):
        curr_path = os.path.normpath(os.path.dirname(curr_path))

    curr_path = os.path.normpath(curr_path)

    return curr_path


def dot_d_to_nested(dot_d: Dict[str, Any]): #, nested_d: Dict[str, Any]):
    nested_d = {}
    for key, val in dot_d.items():
        key_arr = key.split(".")
        curr_d = nested_d
        for inner_key in key_arr[:-1]:
            if inner_key not in curr_d:
                curr_d[inner_key] = {}
            curr_d = curr_d[inner_key]
        curr_d[key_arr[-1]] = val
    return nested_d


def create_or_clear_folder(folder_path, confirm=False):
    import shutil

    if os.path.exists(folder_path):
        if confirm:
            answer = input(f"folder_path={folder_path} exists. Confirm delete? Y/N\n")
            if answer.upper() != "Y":
                print(f"Aborted and Terminating. folder_path={folder_path} not deleted.")
                exit()

        shutil.rmtree(folder_path)

    os.makedirs(folder_path)


def main():
    import pprint
    dot_d = {
        "sb3.a": 1,
        "sb3.b": 1,
    }
    d = dot_d_to_nested(dot_d)
    pprint.pprint(d)


if __name__ == '__main__':
    main()