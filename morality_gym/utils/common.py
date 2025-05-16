import os
import glob
import pickle
from typing import Optional, Dict, List, Any
import shutil


# def create_or_clear_folder(folder_path):
#     if os.path.exists(folder_path):
#         shutil.rmtree(folder_path)
#
#     os.mkdir(folder_path)

def create_or_clear_folder(folder_path, confirm=False):
    import shutil

    if os.path.exists(folder_path):
        if confirm:
            input(f"Warning: {folder_path} will be cleared. Press 'enter' to proceed.")

        shutil.rmtree(folder_path)

    os.makedirs(folder_path)


def clear_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)


def save_file(obj: Any, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_file(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def copy_to_dict(from_dict: Optional[Dict], to_dict: Dict) -> Dict:
    assert from_dict is not None
    assert to_dict is not None

    for key in from_dict.keys():
        to_dict[key] = from_dict[key]
    return to_dict


def join_paths(paths: List[str]):
    return os.path.normpath(os.path.join(*paths))


def comp_base_path(path, level=1):
    curr_path = path
    for i in range(level):
        curr_path = os.path.normpath(os.path.dirname(curr_path))

    curr_path = os.path.normpath(curr_path)

    return curr_path
