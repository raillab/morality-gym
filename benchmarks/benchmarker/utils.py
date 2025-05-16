import os
from typing import Optional

from morality_gym.utils.common import comp_base_path, join_paths


# def move_and_zip_logs(old_path, new_path):
#     from morality_gym.utils.common import join_paths, create_or_clear_folder
#     import os
#     create_or_clear_folder(new_path)
#     create_or_clear_folder(join_paths([new_path, "tmp"]))
#
#     log_dirs = os.listdir(old_path)
#     for curr_dir in log_dirs:
#         pass


def load_all_data(log_base_path, csv_name: str, cols: Optional[str] = None, cols_reg: Optional[str] = None):
    import json
    import pandas as pd

    if cols is not None and cols_reg is not None:
        raise ValueError("Only one of cols or cols_reg can be specified")

    # base_path = comp_base_path(os.path.abspath(__file__), level=1)
    # log_base_path = join_paths([base_path, log_base_path])
    run_folders = os.listdir(log_base_path)

    if cols_reg is not None:
        import re
        tmp_csv_path = join_paths([log_base_path, run_folders[0], csv_name])
        tmp_df = pd.read_csv(tmp_csv_path, nrows=1)
        all_cols = tmp_df.columns.tolist()
        match_cols = list(filter(lambda x: re.match(cols_reg, x) is not None, all_cols))
        del tmp_df
        cols = match_cols

    # print(run_folders)
    runs_d = {}
    for curr_dir in run_folders:
        curr_csv_path = join_paths([log_base_path, curr_dir, csv_name])
        df = pd.read_csv(curr_csv_path, usecols=cols)

        curr_config_path = join_paths([log_base_path, curr_dir, "run_config.json"])
        with open(curr_config_path, "r") as f:
            curr_config = json.load(f)
        runs_d[curr_dir] = {
            "df": df,
            "config": curr_config
        }

    return runs_d


def main():
    base_path = comp_base_path(os.path.abspath(__file__), level=1)
    print(base_path)
    log_base_path = join_paths([base_path, "logs", "debug"])
    runs_d = load_all_data(log_base_path)
    print(runs_d)

if __name__ == '__main__':
    main()