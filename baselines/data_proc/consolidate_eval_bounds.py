import json
import os
import sys
from typing import Optional, List, Dict

import pandas as pd
from tqdm import tqdm

from morality_gym.utils.common import comp_base_path, join_paths

BASE_PATH = comp_base_path(os.path.abspath(__file__), level=2)
DATA_PATH = join_paths([BASE_PATH, "data"])

def get_dot_val(dot_key, d):
    keys = dot_key.split(".")
    val = d
    for key in keys:
        val = val[key]
    return val


def consolidate(
        exp_id: str,
        config_keys: Optional[Dict[str, str]] = None
):
    print(f"\n>>> Consolidating eval bounds data for {exp_id}")
    raw_data_path = join_paths([DATA_PATH, "raw", exp_id, "raw"])
    all_runs = list(filter(lambda x: "run" in x, os.listdir(raw_data_path)))
    # all_eval_dfs = {}
    # all_configs = {}
    all_dfs = []
    bounds_d = {}

    for i in tqdm(range(len(all_runs)), desc="Consolidating eval dataframes"):
        run = all_runs[i]
        run_path = join_paths([raw_data_path, run, run])
        eval_csv_path = join_paths([run_path, "metrics", "agr_eval.csv"])
        tmp_eval_df = pd.read_csv(eval_csv_path)
        mf_cols = [col for col in tmp_eval_df.columns if "mf" in col]
        curr_eval_df = tmp_eval_df[["morality_metric", "avg_cost", "agr"] + mf_cols].copy()
        with open(join_paths([run_path, "config.json"]), "r") as f:
            curr_config = json.load(f)

        # curr_eval_df["seed"] = curr_config["learner"]["init_kwargs"]["seed"]
        # curr_eval_df["learner"] = curr_config["learner"]["name"]

        env_id = curr_config["learner"]["init_kwargs"]["env_id"]
        base_env, variant, _ = env_id.split("-")
        curr_eval_df["base_env"] = base_env
        curr_eval_df["variant"] = variant
        curr_eval_df["mc"] = curr_config["learner"]["init_kwargs"]["mc_id"]
        curr_eval_df["env"] = env_id
        curr_eval_df["run"] = run

        if config_keys is not None:
            for key, dot_key in config_keys.items():
                curr_eval_df[key] = get_dot_val(dot_key, curr_config)

        agr_df = curr_eval_df.iloc[:1].drop(["morality_metric", "avg_cost", "agr"] + mf_cols, axis=1)
        min_row = curr_eval_df[curr_eval_df["agr"] == "min"].iloc[0]
        max_row = curr_eval_df[curr_eval_df["agr"] == "max"].iloc[0]

        agr_df["min_mm"] = min_row["morality_metric"]
        agr_df["max_mm"] = max_row["morality_metric"]

        agr_df["min_cost"] = min_row["avg_cost"]
        agr_df["max_cost"] = max_row["avg_cost"]

        min_mf_srs = curr_eval_df[mf_cols].min()
        max_mf_srs = curr_eval_df[mf_cols].max()

        for col in mf_cols:
            agr_df[f"min_{col}"] = min_mf_srs[col]
            agr_df[f"max_{col}"] = max_mf_srs[col]

        all_dfs.append(agr_df)

    cons_df = pd.concat(all_dfs, ignore_index=True, axis=0)
    proc_path = join_paths([DATA_PATH, "proc", exp_id])
    if not os.path.exists(proc_path):
        os.mkdir(proc_path)
    cons_df.to_csv(join_paths([proc_path, "agr_eval.csv"]), index=False)
    # cons_df.to_csv(join_paths([DATA_PATH, "consolidated", f"{exp_id}_consolidated.csv"]))

def main():
    # config_keys = {
    #     "mc_beta": "learner.init_kwargs.mc_overrides.beta"
    # }
    # for i in range(4):
    #     exp_id = f"s1edmc{i}"
    #     # consolidate(exp_id=sys.argv[1])
    #     consolidate(exp_id=exp_id, config_keys=config_keys)

    config_keys = {
        # "cost_fact": "learner.init_kwargs.cost_function_kwargs.scale_fact"
    }
    exp_id = f"eb5e"
    # consolidate(exp_id=sys.argv[1])
    consolidate(exp_id=exp_id, config_keys=config_keys)


if __name__ == "__main__":
    main()