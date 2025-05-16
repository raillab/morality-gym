import copy
import itertools
import json
import math
import os
import subprocess
import sys
from typing import Dict, Tuple, List, Any

import numpy as np
from tqdm import tqdm

sys.path.append('..')

from morality_gym.utils.common import copy_to_dict, join_paths, comp_base_path, create_or_clear_folder

# BENCHMARKER:
# 1. Init env
# 2. Init learner
# 3. Do learning
# 4. Do evaluation
# 5. Handle logging

BASE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_PATH = join_paths([BASE_PATH, "configs"])
RUNS_PATH = join_paths([BASE_PATH, "runs"])
SLURMS_PATH = join_paths([BASE_PATH, "slurms"])

NODE_TMP_FOLDER = "/tmp/myrun"

ABS_MAX_NODES = 20

# PATH_SEP = os.path.sep
# BASE_PATH_ARR = os.path.abspath(__file__).split(PATH_SEP)[:-1]

# RUNS_PATH = BASE_PATH_ARR.copy()
# RUNS_PATH.append("runs")
# RUNS_PATH = PATH_SEP.join(RUNS_PATH)

# CONFIGS_PATH = BASE_PATH_ARR.copy()
# CONFIGS_PATH.append("exp_configs")
# CONFIGS_PATH = PATH_SEP.join(CONFIGS_PATH)

# BASE_PATH = os.path.normpath(PATH_SEP.join(BASE_PATH_ARR))

# RUNS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "runs"))
# MODELS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "models"))
# SLURMS_FOLDER = os.path.normpath(os.path.join(BASE_PATH, "slurms"))
# LOCAL_DATA_LOGS_PATH = os.path.normpath(os.path.join(BASE_PATH, "data_logs"))

# MSL #
# CLUSTER_CODE_PATH = "/home-mscluster/srosen/code/golem-morality-gym"
# CLUSTER_PARTITION = "datasets"
# CLUSTER_LOGS_PATH = f"/{CLUSTER_PARTITION}/srosen/logs/mor_gym/exp"
# CLUSTER_CODE_LOGS_PATH = f"/{CLUSTER_PARTITION}/srosen/logs/mor_gym/code"
# CLUSTER_JOB_LOGS_PATH = f"/{CLUSTER_PARTITION}/srosen/logs/mor_gym/cluster"
# CLUSTER_DATA_LOGS_PATH = f"/{CLUSTER_PARTITION}/srosen/logs/mor_gym/data"




# def create_or_clear_folder(folder_path, confirm=False):
#     import shutil
#
#     if os.path.exists(folder_path):
#         if confirm:
#             input(f"Warning: {folder_path} will be cleared. Press 'enter' to proceed.")
#
#         shutil.rmtree(folder_path)
#
#     os.mkdir(folder_path)


# Get all combination of multi_kwargs and add static_kwargs to each of these
def get_all_combs(static_kwargs: Dict[str, Any], multi_kwargs: Dict[str, List]) -> List[Dict[str, Any]]:
    kwargs_arr = [list(itertools.product([key], val)) for key, val in multi_kwargs.items()]
    kwargs_arr = list(itertools.product(*kwargs_arr))

    kwargs_arr = [{inner_tup[0]: inner_tup[1] for inner_tup in outer_tup} for outer_tup in kwargs_arr]

    for curr_kwargs in kwargs_arr:
        copy_to_dict(static_kwargs, curr_kwargs)

    return kwargs_arr

class Benchmarker:
    def __init__(self,
                 exp_config_path="",
                 is_local=False
                 ):

        self._is_local = is_local

        exp_config_path = join_paths([CONFIGS_PATH, exp_config_path])
        with open(exp_config_path, "r") as f:
            self._exp_config = json.load(f)

        self._exp_type = self._exp_config["exp_type"]
        self._seed = self._exp_config["seed"]

        self._learner_config = self._exp_config["learner"]

        self._logger_config = self._exp_config["logger"]
        self._base_log_dir = os.path.normpath(self._logger_config["log_dir"])

        # CLUSTER #
        self._cluster_config = self._exp_config["cluster"]

        self._exp_name = self._cluster_config["exp_name"]
        self._partition = self._cluster_config["partition"]
        self._max_nodes = self._cluster_config["max_nodes"]
        self._par_runs_per_node = self._cluster_config["runs_per_node"]
        self._seq_repeats = self._cluster_config["seq_repeats_per_run"]
        self._par_repeats = self._cluster_config["par_repeats_per_run"]
        self._tot_repeats = self._seq_repeats * self._par_repeats
        self._conda_env = self._cluster_config["conda_env"]

        if self._max_nodes > ABS_MAX_NODES:
            raise ValueError(f"max_nodes={self._max_nodes} is greater than ABS_MAX_NODES={ABS_MAX_NODES}")


        # PATHS & DIRECTORIES #
        self._runs_dir = join_paths([BASE_PATH, "runs", self._exp_name])
        self._slurms_dir = join_paths([BASE_PATH, "slurms", self._exp_name])

        self._abs_code_path = comp_base_path(os.path.abspath(__file__), level=3)
        # Get path for logs
        self._base_logs_path = None
        ini_path = join_paths([BASE_PATH, "logs_path.ini"])
        if os.path.exists(ini_path):
            with open(ini_path, "r") as f:
                line = f.readline().replace("\n", "")
                self._base_logs_path = line.split("=")[1]
        else:
            self._base_logs_path = input("Specify the value of base_logs_path: ")
            with open(ini_path, "w") as f:
                f.write(f"base_logs_path={self._base_logs_path}")

        self._logs_path = join_paths([self._base_logs_path, self._exp_name])
        self._run_logs_path = join_paths([self._logs_path, "runs"])
        self._slurm_logs_path = join_paths([self._logs_path, "slurms"])
        self._data_logs_path = join_paths([self._logs_path, "data"])

        # self._setup_dirs()

        self._n_runs = 0
        self._run_configs = None
        # cluster_config = self.exp_config["cluster"]

        self._n_nodes = None



        # self.job_name_prefix = cluster_config["job_name"]
        # self.run_logs_folder = cluster_config["logs_folder"]
        #
        # self.conda_env = cluster_config["conda_env"]
        # if self.conda_env not in ["func-approx", "marl"]:
        #     raise ValueError(f"Invalid value for conda_env={self.conda_env}")
        #
        # self.log_mode = self.exp_config["logger"]["log_mode"]
        #
        # if self.cluster_name == "msl":
        #     self.slurm_logs_folder = cluster_config["logs_folder"]
        # elif self.cluster_name == "chpc":
        #     self.qsubs_logs_folder = cluster_config["logs_folder"]
        #
        # self.runs_per_node = cluster_config["runs_per_node"]
        # self.seq_repeats = cluster_config["seq_repeats_per_run"]
        # self.par_repeats = cluster_config["par_repeats_per_run"]
        #
        # # self.runs_per_node = runs_per_node
        # self.no_runs = 0
        # self.n_nodes = 0
        # # self.log_folder = log_folder
        # self.slurms_folder = None
        # self.qsubs_folder = None
        #
        # self.run_configs = None
        #
        # #######################
        # # Local Logging Stuff #
        # #######################
        # if is_local:
        #     self.local_data_logs_path = LOCAL_DATA_LOGS_PATH
        # else:
        #     self.local_data_logs_path = CLUSTER_DATA_LOGS_PATH
        #
        # self.zip_group_folder_path = None
        #
        # if self.log_mode == "local":
        #     if not os.path.isdir(self.local_data_logs_path):
        #         raise ValueError(f"self.local_data_logs_path={self.local_data_logs_path} is not a valid path to a directory")
        #
        #     zip_folder_path = join_paths([self.local_data_logs_path, "zipped"])
        #     if not os.path.isdir(zip_folder_path):
        #         os.mkdir(zip_folder_path)
        #
        #     # Create group folder or clear existing folder
        #     group_name = self.exp_config["logger"]["local_kwargs"]["group"]
        #     group_folder_path = join_paths([self.local_data_logs_path, group_name])
        #     self.zip_group_folder_path = join_paths([zip_folder_path, group_name])
        #
        #     # if os.path.isdir(group_folder_path):
        #     #     print(f"WARNING: group_folder_path={group_folder_path} existed and was cleared")
        #
        #     create_or_clear_folder(group_folder_path, confirm=True)
        #     create_or_clear_folder(self.zip_group_folder_path)
        #
        # ########################
        #
        # if self.cluster_name == "msl":
        #     self.slurms_folder = os.path.join(SLURMS_FOLDER, cluster_config["job_name"])
        #     self.qsubs_logs_folder = None
        #     self.slurm_logs_folder = f"{CLUSTER_LOGS_PATH}/slurms/{self.slurm_logs_folder}"
        #     self.wandb_base_folder = None
        #     self.run_logs_folder = f"{CLUSTER_LOGS_PATH}/runs/{self.run_logs_folder}"
        #
        #     if not is_local:
        #         self.local_data_logs_path = CLUSTER_DATA_LOGS_PATH
        # elif self.cluster_name == "chpc":
        #     self.qsubs_folder = os.path.join(QSUBS_FOLDER, cluster_config["job_name"])
        #     self.qsubs_logs_folder = f"{CHPC_LOGS_PATH}/qsubs/{self.qsubs_logs_folder}"
        #     self.slurm_logs_folder = None
        #     self.run_logs_folder = f"{CHPC_LOGS_PATH}/runs/{self.run_logs_folder}"
        #     self.wandb_base_folder = f"{CHPC_WANDB_PATH}/{cluster_config['job_name']}"
        #
        #     if not is_local:
        #         raise NotImplementedError("CHPC local_data_logs_path not set")
        #         # self.local_data_logs_path = CLUSTER_DATA_LOGS_PATH
        #
        #     raise NotImplementedError("CHPC functionality not fully implemented")
        #
        # self.runs_folder = os.path.join(RUNS_FOLDER, cluster_config["job_name"])
        #
        # self.is_local = is_local
        #
        # if not is_local:
        #     self.make_log_folders()
        #     if self.cluster_name == "chpc":
        #         create_or_clear_folder(self.wandb_base_folder)

    def _setup_dirs(self):
        pass
        # create_or_clear_folder(self._runs_dir, confirm=False)
        # create_or_clear_folder(self._slurms_dir, confirm=False)
        #
        # if not self._is_local:
        #     create_or_clear_folder(self._logs_path, confirm=True)
        #     os.mkdir(self._data_logs_path)
        #     os.mkdir(self._run_logs_path)
        #     os.mkdir(self._slurm_logs_path)

    def make_log_folders(self):
        if not self._is_local:
            create_or_clear_folder(self._logs_path, confirm=True)
            os.mkdir(self._data_logs_path)
            os.mkdir(self._run_logs_path)
            os.mkdir(self._slurm_logs_path)
        else:
            raise ValueError(f"make_log_folders not supported for self._is_local={self._is_local}")

    def create(self):
        if not self._is_local:
            self.make_log_folders()

        self.create_run_configs()
        self.create_slurms()

    def create_run_configs(self):
        create_or_clear_folder(self._runs_dir, confirm=False)

        # Each of this is a dict
        d_groups = {"algo_kwargs", "model_kwargs", "train_kwargs", "env_overrides", "lagrange_kwargs", "cost_kwargs"}

        learner_config = self._learner_config
        l_kwargs_config = learner_config["kwargs"]

        algos = learner_config["algos"].copy()

        #################
        # STATIC KWARGS #
        #################
        common_static_kwargs = copy.deepcopy(l_kwargs_config["common"]["static"])
        alg_static_kwargs = {}
        for alg in algos:
            curr_kwargs = copy.deepcopy(common_static_kwargs)
            for key, val in l_kwargs_config[alg]["static"].items():
                if key in d_groups:
                    tmp_d = {}
                    if key in curr_kwargs:
                        if not isinstance(curr_kwargs[key], dict):
                            raise TypeError(f"curr_kwargs[{key}] should be of type dict. "
                                            f"Got type {type(curr_kwargs[key])}")
                        tmp_d = curr_kwargs[key]
                    copy_to_dict(val, tmp_d)
                    curr_kwargs[key] = tmp_d
                else:
                    curr_kwargs[key] = copy.deepcopy(val)
            alg_static_kwargs[alg] = curr_kwargs
        # print("common_static_kwargs")
        # print(common_static_kwargs)
        #
        # print("alg_static_kwargs")
        # print(alg_static_kwargs)
        #################

        ###############
        # -- MULTI -- #
        ###############
        # Common
        common_multi_kwargs = copy.deepcopy(l_kwargs_config["common"]["multi"])
        flat_c_m_kwargs = {}
        for outer_key, outer_val in common_multi_kwargs.items():
            if outer_key in d_groups:
                for inner_key, inner_val in outer_val.items():
                    flat_c_m_kwargs[(outer_key, inner_key)] = inner_val
            else:
                flat_c_m_kwargs[(None, outer_key)] = outer_val
        # for d_group in d_groups:
        #     if d_group in common_multi_kwargs:
        #         for key, val in common_multi_kwargs[d_group].items():
        #             flat_c_m_kwargs[(d_group, key)] = val

        # print("flat_c_m_kwargs:\n", flat_c_m_kwargs)

        # alg_multi_kwargs = copy.deepcopy(l_kwargs_config["common"]["multi"])
        flat_a_m_kwargs = {}
        for algo in algos:
            if algo in l_kwargs_config.keys():
                curr_multi_kwargs = copy.deepcopy(l_kwargs_config[algo]["multi"])
                curr_c_m_kwargs = {}
                for outer_key, outer_val in curr_multi_kwargs.items():
                    if outer_key in d_groups:
                        for inner_key, inner_val in outer_val.items():
                            curr_c_m_kwargs[(outer_key, inner_key)] = inner_val
                    else:
                        curr_c_m_kwargs[(None, outer_key)] = outer_val
                flat_a_m_kwargs[algo] = curr_c_m_kwargs

        # print("flat_a_m_kwargs:\n", flat_a_m_kwargs)

        # GET COMBINATIONS #
        algo_kwarg_combs = {}
        for algo in algos:
            if algo in flat_a_m_kwargs:
                curr_multi_kwargs = copy.deepcopy(flat_c_m_kwargs)
                copy_to_dict(flat_a_m_kwargs[algo], curr_multi_kwargs)
                # tmp_arrs = [() for key, arr in curr_multi_kwargs.items()]
                curr_multi_kwargs_arr = []
                ind_to_key = {}
                vals_arr = []
                for i, (key, arr) in enumerate(curr_multi_kwargs.items()):
                    ind_to_key[i] = key
                    vals_arr.append(arr)

                curr_combs_tup_arr = list(itertools.product(*vals_arr))
                curr_combs_arr = []
                for tup in curr_combs_tup_arr:
                    curr_d = {}
                    for i, val in enumerate(tup):
                        curr_d[ind_to_key[i]] = val
                    curr_combs_arr.append(curr_d)

                algo_kwarg_combs[algo] = curr_combs_arr
            else:
                algo_kwarg_combs[algo] = []
        ###############

        curr_run = 0
        run_configs = []
        for algo in algos:
            curr_algo_combs = algo_kwarg_combs[algo]
            algo_kwargs_arr = []
            if len(curr_algo_combs) == 0:  # curr_algo_combs empty
                curr_algo_kwargs = copy.deepcopy(alg_static_kwargs[algo])
                algo_kwargs_arr.append(curr_algo_kwargs)
                # TODO
            else:
                for comb_d in curr_algo_combs:
                    curr_algo_kwargs = copy.deepcopy(alg_static_kwargs[algo])
                    for (outer_key, inner_key), val in comb_d.items():
                        if outer_key is None:
                            curr_algo_kwargs[inner_key] = val
                        else:
                            if outer_key in curr_algo_kwargs:
                                curr_algo_kwargs[outer_key][inner_key] = val
                            else:
                                curr_algo_kwargs[outer_key] = {inner_key: val}
                    algo_kwargs_arr.append(curr_algo_kwargs)

            for curr_algo_kwargs in algo_kwargs_arr:
                rng = np.random.default_rng(seed=self._seed)
                for repeat_no in range(self._tot_repeats):
                    run_name = f"run_{curr_run}"
                    repeat_seed = int(rng.integers(low=0, high=2**31-1, size=1)[0])
                    curr_logger_kwargs = copy.deepcopy(self._logger_config)
                    curr_logger_kwargs["log_dir"] = join_paths([self._base_log_dir, run_name])

                    run_config = {
                        "name": run_name,
                        "run_type": self._exp_type,
                        "learner_kwargs": {
                            "seed": repeat_seed,
                            "algo_name": algo,
                            "logger_kwargs": curr_logger_kwargs
                        }
                    }
                    copy_to_dict(curr_algo_kwargs, run_config["learner_kwargs"])
                    # Override env seed with curr repeat seed
                    if "env_overrides" in run_config["learner_kwargs"]:
                        run_config["learner_kwargs"]["env_overrides"]["seed"] = repeat_seed
                    else:
                        run_config["learner_kwargs"]["env_overrides"] = {
                            "seed": repeat_seed
                        }

                    run_configs.append(run_config)
                    curr_run += 1
        # base_learner_kwargs = copy.deepcopy(l_kwargs_config["common"]["static"])
        # static_learner_kwargs = {}
        # for algo in algos:
        #     tmp_kwargs = l_kwargs_config
        print(f"Creating run configs in {self._runs_dir}...")

        for i in tqdm(range(len(run_configs))):
            config_path = join_paths([self._runs_dir, f"run_{i}.json"])
            with open(config_path, "w") as f:
                json.dump(run_configs[i], f, indent=4)

        self._n_runs = len(run_configs)
        self._run_configs = run_configs

    def load_slurm_template(self):
        template_path = os.path.join(BASE_PATH, "template.slurm")
        with open(template_path, "r") as f:
            file_contents = "".join(f.readlines())

        file_contents = file_contents.replace("{partition}", self._partition)
        file_contents = file_contents.replace("{conda_env}", self._conda_env)
        file_contents = file_contents.replace("{slurm_logs_folder}", self._slurm_logs_path)
        # file_contents = file_contents.replace("{cluster_job_logs_path}", self.slurm_logs_folder)
        return file_contents

    def create_slurms(self):
        create_or_clear_folder(self._slurms_dir, confirm=False)

        tot_parallel_runs = self._n_runs // self._seq_repeats
        self._n_nodes = math.ceil(tot_parallel_runs/self._par_runs_per_node)

        if self._n_nodes > self._max_nodes:
            raise ValueError(f"no of nodes requested is greater than maximum allowed. n_nodes={self._n_nodes} > "
                             f"max_no_nodes={self._max_nodes}")

        slurm_template = self.load_slurm_template()

        # if self.no_runs % self.runs_per_node != 0:
        #     no_slurms += 1

        # no_slurms = self.no_runs//self.runs_per_node
        # if self.no_runs % self.runs_per_node != 0:
        #     no_slurms += 1

        # slurm_folder = f"{CLUSTER_CODE_PATH}slurms/"
        # slurm_folder = f"{BASE_PATH}{PATH_SEP}slurms{PATH_SEP}"


        ##################
        # Setup & Finish #
        ##################
        logs_path = f"{NODE_TMP_FOLDER}/logs"
        logs_temp_path = f"{NODE_TMP_FOLDER}/temp"

        setup_commands = f"rm -rf {NODE_TMP_FOLDER}\n"
        # setup_commands += f"mkdir {NODE_TMP_FOLDER}\n"
        setup_commands += f"mkdir {logs_path} -p\n"
        setup_commands += f"mkdir {logs_temp_path} -p\n"


        finish_commands = ""

        # finish_commands += f"mv {raw_logs_path}/"
        # Zip data
        finish_commands += f"zip -r {logs_path}/runs_{{run_i}}-{{run_j}}.zip {logs_path}/* \n"
        finish_commands += f"mv {logs_path}/runs_{{run_i}}-{{run_j}}.zip {self._data_logs_path}/runs_{{run_i}}-{{run_j}}.zip \n"
        finish_commands += f"rm -rf {NODE_TMP_FOLDER}\n"
        # finish_commands += f"zip -r {NODE_TMP_FOLDER}/runs_{{run_i}}-{{run_j}}.zip {tmp_local_logger_dir}/*"

        # tmp_local_logger_dir = ""

        # if self.log_mode == "wandb":
        #     # SET UP
        #     setup_commands += "\n" + f"mkdir {NODE_TMP_FOLDER} -p"
        #     setup_commands += "\n" + f"export WANDB_DIR={NODE_TMP_FOLDER}"
        #     # Finish
        #     finish_commands += "echo WANDB sync started\n"
        #     finish_commands += f"cd {NODE_TMP_FOLDER}\n"
        #     finish_commands += "wandb sync --sync-all --include-offline\n"
        #     finish_commands += "echo WANDB sync finished\n"
        #     finish_commands += "cd $SLURM_SUBMIT_DIR \n"
        #
        # elif self.log_mode == "local":
        #     # SET UP
        #     log_group = self.exp_config["logger"]["local_kwargs"]["group"]
        #     tmp_local_logger_dir = join_paths([NODE_TMP_FOLDER, log_group])
        #     setup_commands += "\n" + f"mkdir {tmp_local_logger_dir} -p"
        #     # Finish - TODO
        #     # self.local_data_logs_path
        #     # Move data
        #     logs_path = join_paths([self.local_data_logs_path, log_group])
        #
        #     # Zip data
        #     # TODO: Uncomment and implement this
        #     finish_commands += f"zip -r {NODE_TMP_FOLDER}/runs_{{run_i}}-{{run_j}}.zip {tmp_local_logger_dir}/*"
        #     finish_commands += "\n" + f"mv {NODE_TMP_FOLDER}/runs_{{run_i}}-{{run_j}}.zip {self.zip_group_folder_path}/runs_{{run_i}}-{{run_j}}.zip"
        #
        #     finish_commands += "\n" + f"mv {tmp_local_logger_dir}/* {logs_path}"
        # else:
        #     raise ValueError

        # finish_commands += "\n" + f"rm -rf {NODE_TMP_FOLDER}"

        slurm_template = slurm_template.replace("{setup_commands}", setup_commands)
        slurm_template = slurm_template.replace("{finish_commands}", finish_commands)

        ##################
        # logs_path = f"{CLUSTER_CODE_LOGS_PATH}{self.log_folder}/"
        # logs_folder = f"{BASE_PATH}{PATH_SEP}logs{PATH_SEP}"

        # clear_folder(slurm_folder)

        tot_runs_per_node = self._par_runs_per_node * self._seq_repeats
        # tot_processes = self._par_runs_per_node

        for slurm_id in tqdm(range(self._n_nodes)):
            start_id = slurm_id * tot_runs_per_node
            end_id = min(start_id + tot_runs_per_node, self._n_runs)
            job_name = f"{self._exp_name}_{slurm_id}"

            slurm_str = slurm_template.replace("{job_name}", job_name)
            slurm_str = slurm_str.replace("{run_i}", str(start_id))
            slurm_str = slurm_str.replace("{run_j}", str(end_id - 1))

            run_commands = ""
            for i in range(start_id, end_id, self._seq_repeats):
                run_commands += f'echo "Attempting to run run{i}.pkl - run{i + self._seq_repeats - 1}.pkl"\n'
                # run_commands += f"python3 {CLUSTER_CODE_PATH}benchmarker.py --run {self.runs_folder}/run{i}.pkl " \
                #                 f"--node $SLURM_JOB_NODELIST " \
                #                 f"-rr {n_seq_repeats}" \
                #                 f" > {logs_path}/run{i}.out 2>&1 & \n"
                runs = [f"run_{j}.json" for j in range(i, i + self._seq_repeats)]
                run_paths = [join_paths([self._runs_dir, curr_run]) for curr_run in runs]
                runs_str = " ".join(run_paths)

                # run_path = join_paths([self.runs_folder, f"run_{i}.pkl"])
                script_path = join_paths([self._abs_code_path, "benchmarks", "cli.py"])
                # run_commands += f"python3 {script_path} --run {run_path} " \
                #                 f"--node $SLURM_JOB_NODELIST "
                run_commands += f"python3 {script_path} --exec-multi-run {runs_str} " \
                                f"--node $SLURM_JOB_NODELIST "

                run_commands += f"--log-base-path {logs_path} "
                run_commands += f"--log-temp-path {logs_temp_path} "
                rel_log_paths = [f"run_{j}" for j in range(i, i + self._seq_repeats)]
                run_commands += f"--log-multi-path {' '.join(rel_log_paths)} "

                run_commands += f" > {self._run_logs_path}/run_{i}-{i + self._seq_repeats - 1}.out 2>&1 & \n"

            slurm_str = slurm_str.replace("{run_commands}", run_commands)
            # slurm_str = slurm_str.replace("{start_id}", str(start_id))
            # slurm_str = slurm_str.replace("{end_id}", str(end_id))
            slurm_path = os.path.join(self._slurms_dir, f"job_{slurm_id}.slurm")
            # f"{slurm_folder}job_{slurm_id}.slurm"
            with open(slurm_path, "w") as f:
                f.write(slurm_str)

        # for slurm_id in tqdm(range(no_slurms)):
            # curr_n_runs = self.runs_per_node * self.seq_repeats
            # start_id = slurm_id * curr_n_runs
            # end_id = min(start_id + curr_n_runs, self.no_runs)
            # job_name = f"{self.job_name_prefix}_{slurm_id}"
            #
            # slurm_str = slurm_template.replace("{job_name}", job_name)
            # slurm_str = slurm_str.replace("{run_i}", str(start_id))
            # slurm_str = slurm_str.replace("{run_j}", str(end_id-1))

            # curr_finish_commands = slurm_str.replace("{run_i}", str(start_id))
            # curr_finish_commands = curr_finish_commands.replace("{run_j}", str(end_id))
            # slurm_str = slurm_str.replace("{finish_commands}", curr_finish_commands)

            # run_commands = ""
            # for i in range(start_id, end_id, self.seq_repeats):
            #     run_commands += f'echo "Attempting to run run{i}.pkl - run{i+self.seq_repeats-1}.pkl"\n'
            #     # run_commands += f"python3 {CLUSTER_CODE_PATH}benchmarker.py --run {self.runs_folder}/run{i}.pkl " \
            #     #                 f"--node $SLURM_JOB_NODELIST " \
            #     #                 f"-rr {n_seq_repeats}" \
            #     #                 f" > {logs_path}/run{i}.out 2>&1 & \n"
            #     runs = [f"run_{j}.pkl" for j in range(i, i+self.seq_repeats)]
            #     run_paths = [join_paths([self.runs_folder, curr_run]) for curr_run in runs]
            #     runs_str = " ".join(run_paths)
            #     # run_path = join_paths([self.runs_folder, f"run_{i}.pkl"])
            #     script_path = join_paths([CLUSTER_CODE_PATH, "api.py"])
            #     # run_commands += f"python3 {script_path} --run {run_path} " \
            #     #                 f"--node $SLURM_JOB_NODELIST "
            #     run_commands += f"python3 {script_path} --run {runs_str} " \
            #                     f"--node $SLURM_JOB_NODELIST "
            #     if self.log_mode == "local":
            #         run_commands += f"--local-logger-dir {tmp_local_logger_dir}"
            #     run_commands += f" > {self.run_logs_folder}/run_{i}-{i+self.seq_repeats-1}.out 2>&1 & \n"
            #
            # slurm_str = slurm_str.replace("{run_commands}", run_commands)
            # # slurm_str = slurm_str.replace("{start_id}", str(start_id))
            # # slurm_str = slurm_str.replace("{end_id}", str(end_id))
            # slurm_path = os.path.join(self.slurms_folder, f"job_{slurm_id}.slurm")
            # # f"{slurm_folder}job_{slurm_id}.slurm"
            # with open(slurm_path, "w") as f:
            #     f.write(slurm_str)

    def exec_slurms(self):
        slurm_files = os.listdir(self._slurms_dir)
        assert len(slurm_files) <= self._max_nodes, "Too many slurm files requested"

        for file in slurm_files:
            curr_file_path = join_paths([self._slurms_dir, file])
            subprocess.run(["sbatch", curr_file_path])



def debug():
    benchmarker = Benchmarker(exp_config_path="test_config.json", is_local=True)
    benchmarker.create_run_configs()
    benchmarker.create_slurms()
    # print(benchmarker.exp_config)
    # benchmarker.create_run_configs()


if __name__ == "__main__":
    debug()