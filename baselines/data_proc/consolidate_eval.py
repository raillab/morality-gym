import json
import os
import sys
from typing import Optional, List, Dict, Union

import pandas as pd
from tqdm import tqdm

from morality_gym._utils.common import comp_base_path, join_paths

PROJECT_BASE_PATH = comp_base_path(os.path.abspath(__file__), level=3)
ENV_VARIANTS_BASE_PATH = join_paths(
    [PROJECT_BASE_PATH, "morality_gym", "_environments", "trolley", "configs", "variants"]
)

BASE_PATH = comp_base_path(os.path.abspath(__file__), level=2)
DATA_PATH = join_paths([BASE_PATH, "data"])

def get_dot_val(dot_key, d):
    keys = dot_key.split(".")
    val = d
    for key in keys:
        if key not in val:
            return None
        val = val[key]
    return val


def consolidate(
        exp_id: str,
        config_keys: Optional[Dict[str, Union[List[str], str]]] = None
):
    print(f"\n>>> Consolidating eval data for {exp_id}")
    raw_data_path = join_paths([DATA_PATH, "raw", exp_id, "raw"])
    all_runs = list(filter(lambda x: "run" in x, os.listdir(raw_data_path)))
    # all_eval_dfs = {}
    # all_configs = {}
    all_learn_dfs = []
    all_final_dfs = []
    all_mf_dfs = []

    for i in tqdm(range(len(all_runs)), desc="Consolidating eval dataframes"):
        try:

            run = all_runs[i]
            run_path = join_paths([raw_data_path, run, run])
            eval_csv_path = join_paths([run_path, "metrics", "eval.csv"])
            curr_eval_df = pd.read_csv(eval_csv_path)
            with open(join_paths([run_path, "config.json"]), "r") as f:
                curr_config = json.load(f)

            curr_eval_df["seed"] = curr_config["learner"]["init_kwargs"]["seed"]
            curr_eval_df["learner"] = curr_config["learner"]["name"]

            env_id = curr_config["learner"]["init_kwargs"]["env_id"]
            base_env, variant, _ = env_id.split("-")
            curr_eval_df["base_env"] = base_env
            curr_eval_df["variant"] = variant
            curr_eval_df["mc"] = curr_config["learner"]["init_kwargs"]["mc_id"]
            curr_eval_df["env"] = env_id
            curr_eval_df["run"] = run

            if config_keys is not None:
                for key, dot_key_arr in config_keys.items():
                    if isinstance(dot_key_arr, str):
                        dot_key_arr = [dot_key_arr]

                    for dot_key in dot_key_arr:
                        curr_val = get_dot_val(dot_key, curr_config)
                        if curr_val is not None:
                            curr_eval_df[key] = curr_val
                            continue

            all_learn_dfs.append(curr_eval_df.iloc[:-1])  # Exclude final

            # Final
            curr_final_df = curr_eval_df.iloc[-1:].copy()
            all_final_dfs.append(curr_final_df)

            # mfs df
            with open(join_paths([ENV_VARIANTS_BASE_PATH, f"{base_env}-v1.json"]), "r") as f:
                variants_d = json.load(f)

            salient_norms = variants_d[variant]["salient_norms"]
            all_mfs = [col for col in curr_final_df.columns if "mf" in col]
            curr_mfs_arr = []
            for sn in salient_norms:
                if f"mf_{sn}" in curr_final_df.columns:
                    tmp_df = curr_final_df.drop(all_mfs, axis=1)
                    tmp_df["mf_val"] = curr_final_df[f"mf_{sn}"].iloc[0]
                    tmp_df["mf_name"] = sn

                    curr_mfs_arr.append(tmp_df)

            curr_mfs_df = pd.concat(curr_mfs_arr, axis=0, ignore_index=True)

            all_mf_dfs.append(curr_mfs_df)
        except Exception as e:
            print(f"ERROR: {e} for run {all_runs[i]} in exp {exp_id}. Skipping...")

    proc_path = join_paths([DATA_PATH, "proc", exp_id])
    if not os.path.exists(proc_path):
        os.mkdir(proc_path)

    # Learn Evals
    cons_df = pd.concat(all_learn_dfs, ignore_index=True, axis=0)
    cons_df.to_csv(join_paths([proc_path, "learn_eval.csv"]), index=False)

    # Final Evals
    fin_cons_df = pd.concat(all_final_dfs, ignore_index=True, axis=0)
    fin_cons_df.to_csv(join_paths([proc_path, "final_eval.csv"]), index=False)

    # Final MFs
    mf_cons_df = pd.concat(all_mf_dfs, ignore_index=True, axis=0)
    mf_cons_df.to_csv(join_paths([proc_path, "final_mf_eval.csv"]), index=False)


def main():
    # config_keys = {
    #     "mc_beta": "learner.init_kwargs.mc_overrides.beta"
    # }
    # for i in range(4):
    #     exp_id = f"s1edmc{i}"
    #     # consolidate(exp_id=sys.argv[1])
    #     consolidate(exp_id=exp_id, config_keys=config_keys)

    # config_keys = {
    #     # "cost_fact": "learner.init_kwargs.cost_function_kwargs.scale_fact"
    # }

    # for exp_id in ["r3e", "r3m", "r3xe"]:
    #     consolidate(exp_id=exp_id, config_keys={})
    config_keys = {}
    config_keys = {
        # "cost_limit": "learner.init_kwargs.os_custom_cfgs.lagrange_cfgs.cost_limit",
        # "pid_kp": "learner.init_kwargs.os_custom_cfgs.lagrange_cfgs.pid_kp",
        # "pid_ki": "learner.init_kwargs.os_custom_cfgs.lagrange_cfgs.pid_ki",
        # "cost_fact": "learner.init_kwargs.cost_function_kwargs.scale_fact",
        "info_mode": "learner.init_kwargs.cost_function_kwargs.information_mode",
        "cost_limit": [
            "learner.init_kwargs.os_custom_cfgs.algo_cfgs.cost_limit",
            "learner.init_kwargs.os_custom_cfgs.lagrange_cfgs.cost_limit"
        ],
        "scalar": "learner.init_kwargs.cost_function_kwargs.scalarisation"
        # "entropy": "learner.init_kwargs.os_custom_cfgs.algo_cfgs.entropy_coef"
    }

    id_arr = [
        # "pl12xh1",  "r12xh0", "orl12xh1", "c12xh1",

        # "r9xe", "r9e", "r9m", "r9h", "r9xh",
        # "orl12xe", "orl12e", "orl12m", "orl12h", "orl12xh",
        # # "p9xe", "p9e", "p9m", "p9h",  # "p10xh",
        # # "ps9xe", "ps9e", "ps9m", "ps9h",  # "ps10xh",
        # "pl9xe", "pl9e", "pl9m", "pl9h", "pl11xh0",
        "c9xe", "c9e", "c9m", "c9h", "c11xh0",
    ]
    # id_arr = ["r10xh0", "orl10xh0", "r10xh0", "c10xh0", "pl10xh0"]
    # id_arr = ["e_c13a", "e_pl13a"]
    # id_arr = ["ps9xe", "ps9e", "ps9m", "ps9h"]
    for exp_id in id_arr:
        # exp_id = f"r9xe"
        consolidate(exp_id=exp_id, config_keys=config_keys)
    # for exp_id in ["ps3ecf", "ps3xecf"]:
    #     consolidate(exp_id=exp_id, config_keys=config_keys)


if __name__ == "__main__":
    main()