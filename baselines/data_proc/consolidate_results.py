import os.path
from typing import Dict

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
# import tensorflow as tf

from morality_gym._utils.common import join_paths


def tb_events_to_dataframe_dict(event_file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parses a TensorBoard event file and returns a dictionary of wide-format
    pandas DataFrames, with one entry for each metric group.

    Args:
        event_file_path (str): The full path to the TensorBoard event file.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are group names
                                 (e.g., 'rollout', 'train') and values are
                                 the corresponding DataFrames.
    """
    # 1. Load the event file and extract all scalar data
    accumulator = event_accumulator.EventAccumulator(
        event_file_path,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()
    tags = accumulator.Tags()['scalars']

    data = []
    for tag in tags:
        for event in accumulator.Scalars(tag):
            data.append({'step': event.step, 'tag': tag, 'value': event.value})

    long_df = pd.DataFrame(data)

    if long_df.empty:
        print(f"Warning: No scalar data found in {event_file_path}")
        return {}

    # 2. Split the 'tag' into 'group' and 'metric'
    tag_split = long_df['tag'].str.split('/', n=1, expand=True)
    long_df['group'] = tag_split[0]
    long_df['metric'] = tag_split[1]
    long_df['metric'].fillna(long_df['group'], inplace=True)

    # 3. Create the dictionary to hold the final DataFrames
    dataframe_dict = {}

    # 4. Iterate over each group, create a DataFrame, and add it to the dictionary
    for group_name, group_df in long_df.groupby('group'):
        wide_df = group_df.pivot_table(index='step', columns='metric', values='value')
        wide_df = wide_df.sort_index().ffill()
        dataframe_dict[group_name] = wide_df

    return dataframe_dict


def consolidate(logs_path, cons_logs_path):
    params_path = join_paths([logs_path, "params.csv"])
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params.csv found at {params_path}")

    all_dirs = os.listdir(logs_path)
    all_dirs.remove("params.csv")

    df = pd.read_csv(params_path)
    if df.shape[0] != len(all_dirs):
        raise ValueError(f"Number of rows in params.csv ({df.shape[0]}) does not "
                         f"match number of directories ({len(all_dirs)})")

    df_arrs = {}

    for i in range(df.shape[0]):
        curr_row = df.iloc[i]
        details = dict(
            alg = curr_row["algorithm"],
            env_group = curr_row["environment"],
            seed = curr_row["seed"],
            base_env = curr_row["env.base"],
            variant = curr_row["env.variant"],
            mc = curr_row["_morality_chain"]
        )
        log_name = curr_row["logger.name"]

        curr_path = join_paths([logs_path, log_name, "metrics"])
        if not os.path.exists(curr_path):
            raise FileNotFoundError(f"No metrics directory found at {curr_path}")

        for curr_file in os.listdir(curr_path):
            if curr_file.endswith(".csv"):
                curr_df = pd.read_csv(join_paths([curr_path, curr_file]))
                curr_df = curr_df.assign(**details)

                metric_name = curr_file.split(".")[0]
                if metric_name not in df_arrs:
                    df_arrs[metric_name] = []

                df_arrs[metric_name].append(curr_df)

    comb_dfs = {}
    for metric_name, df_arr in df_arrs.items():
        comb_dfs[metric_name] = pd.concat(df_arr, ignore_index=True)  # Check

    if not os.path.exists(cons_logs_path):
        os.makedirs(cons_logs_path)

    for metric_name, df in comb_dfs.items():
        df.to_csv(join_paths([cons_logs_path, f"{metric_name}.csv"]), index=False)



def process_tb_events(log_dir):
    # Assumes all dirs in log_dir have valid format
    all_dirs = os.listdir(log_dir)
    for curr_dir in all_dirs:
        curr_path = join_paths([log_dir, curr_dir])
        if os.path.isdir(curr_path):
            print(f"Processing {curr_path}...")
            tb_path = join_paths([curr_path, "tensorboard"])
            tb_path = join_paths([tb_path, os.listdir(tb_path)[0]])
            tb_path = join_paths([tb_path, os.listdir(tb_path)[0]])
            dfs_d = tb_events_to_dataframe_dict(tb_path)
            for group, df in dfs_d.items():
                df.to_csv(join_paths([curr_path, "metrics", f"tb_{group}.csv"]))



def debug():
    base_path = "logs/exp/x_easy/standard/ppo_shaped_SwitchStandard_HumanAnimal_Outcome_40"
    tb_base_path = base_path + "/tensorboard"
    tb_base_path = join_paths([tb_base_path, os.listdir(tb_base_path)[0]])
    tb_base_path = join_paths([tb_base_path, os.listdir(tb_base_path)[0]])
    dfs_d = tb_events_to_dataframe_dict(tb_base_path)
    for group, df in dfs_d.items():
        df.to_csv(join_paths([base_path, "metrics", f"tb_{group}.csv"]))



def main():
    # process_tb_events("logs/exp/x_easy/standard")
    consolidate(logs_path="logs/exp/x_easy/standard", cons_logs_path="cons_logs/exp/x_easy/standard")
    # debug()

if __name__ == "__main__":
    main()