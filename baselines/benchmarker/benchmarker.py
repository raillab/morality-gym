import copy
import itertools
import json
import os
import pprint
from typing import Optional, List, Union, Dict, Tuple, Any

from tqdm import tqdm

from baselines.benchmarker.utils import load_base_paths
from morality_gym._utils.common import comp_base_path, create_or_clear_folder, join_paths, copy_to_dict_recr, \
    dot_d_to_nested

BASE_PATH = comp_base_path(os.path.abspath(__file__), level=1)
NODE_TMP_PATH = "/tmp/myrun"


class Benchmarker:
    def __init__(
            self,
            config_path: Optional[str] = None,  # Must be relative to baselines/benchmarker/configs/experiment
            nuke_partition: Optional[str] = None,
            confirm_dir_delete: bool = True
    ):
        ###################
        # LOAD EXP CONFIG #
        ###################
        if config_path is not None:
            abs_config_path = os.path.join(BASE_PATH, "configs", "experiment", config_path)
            if not os.path.exists(abs_config_path):
                raise ValueError(f"Config path {config_path} does not exist at absolute path = {abs_config_path}.")

            with open(abs_config_path, "r") as f:
                self._exp_config = json.load(f)

            self._exp_id = self._exp_config["exp"]["id"]
            self._is_nuke = False
        elif nuke_partition is not None:
            self._exp_id = nuke_partition
            self._is_nuke = True

        ###################

        #########################
        # ----- SET PATHS ----- #
        #########################
        base_paths_d = load_base_paths()
        self._runs_path = join_paths([base_paths_d["runs"], self._exp_id])
        if "slurms" not in base_paths_d:
            base_paths_d["slurms"] = base_paths_d["jobs"]
        self._slurms_path = join_paths([base_paths_d["slurms"], self._exp_id])
        self._logs_path = join_paths([base_paths_d["logs"], self._exp_id])
        self._zipped_logs_path = join_paths([base_paths_d["zipped_logs"], self._exp_id])
        self._run_logs_path = join_paths([base_paths_d["run_logs"], self._exp_id])
        if "slurm_logs" not in base_paths_d:
            base_paths_d["slurm_logs"] = base_paths_d["job_logs"]
        self._slurm_logs_path = join_paths([base_paths_d["slurm_logs"], self._exp_id])

        #########################

        self._run_configs = None
        self._run_configs_metadata = None

        self._confirm_dir_delete = confirm_dir_delete


    #################
    # --- SETUP --- #
    #################
    def _setup_dirs(self, confirm_delete=True):
        if confirm_delete and (os.path.exists(self._runs_path) or
                               os.path.exists(self._slurms_path) or os.path.exists(self._logs_path)):
            print(
                f"One or multiple of {[self._runs_path, self._slurms_path, self._logs_path]} already exists and will be deleted.")
            ans = input("Continue? [Y/N] ")
            if ans.upper() != "Y":
                exit(1)

        create_or_clear_folder(self._runs_path, confirm=False)
        create_or_clear_folder(self._logs_path, confirm=False)
        create_or_clear_folder(self._slurms_path, confirm=False)
        create_or_clear_folder(self._run_logs_path, confirm=False)
        create_or_clear_folder(self._slurm_logs_path, confirm=False)
        create_or_clear_folder(self._zipped_logs_path, confirm=False)
    #################

    ##################
    # --- CREATE --- #
    ##################
    def create(self):
        self._setup_dirs(confirm_delete=self._confirm_dir_delete)
        self._create_runs()
        self._create_slurms()

    #  CREATE RUNS   #
    def _create_runs(self):
        # exp_name = self._exp_config["exp"]["name"]
        # exp_type = self._exp_config["exp"]["type"]
        # exp_id = self._exp_id
        seeds = self._exp_config["exp"]["seeds"]

        #####################
        # ------ ENV ------ #
        #####################
        raw_env_config = self._exp_config["env"]
        if raw_env_config["base_envs"] is not None and raw_env_config["env_mcs"] is not None:
            raise ValueError(f"Cannot specify both base_envs and env_mcs in exp config for env.")

        if raw_env_config["base_envs"] is not None or raw_env_config["env_mcs"] is not None:
            if "env_mc_overrides" in raw_env_config:
                env_mc_overrides = raw_env_config["env_mc_overrides"]
            else:
                env_mc_overrides = None
            env_mc_configs = self._load_env_mc_configs(
                base_envs=raw_env_config["base_envs"],
                env_mcs=raw_env_config["env_mcs"],
                env_mc_overrides=env_mc_overrides
            )
        else:
            raise ValueError(f"Must specify either base_envs or env_mcs in exp config for env.")
        #####################

        ######################
        # ---- LEARNERS ---- #
        ######################
        learner_configs = self._load_learner_configs()
        ######################

        ###################################
        # --- CREATE AND SAVE CONFIGS --- #
        ###################################
        all_configs = []
        idx = 0
        for seed, (env_id, mc_id), learner in itertools.product(seeds, env_mc_configs.keys(), learner_configs.keys()):
            if isinstance(learner_configs[learner], list):
                raise NotImplementedError("Multi kwarg functionality for learners not currently supported")

            static_init_kwargs = learner_configs[learner]["init"]["static"]
            multi_init_kwargs = learner_configs[learner]["init"]["multi"]

            base_init_kwargs = copy.deepcopy(env_mc_configs[(env_id, mc_id)])
            copy_to_dict_recr(static_init_kwargs, base_init_kwargs)
            base_init_kwargs["seed"] = seed
            base_init_kwargs["env_id"] = env_id
            base_init_kwargs["mc_id"] = mc_id

            if len(multi_init_kwargs) == 0:
                # Do something
                all_init_kwargs = [copy.deepcopy(base_init_kwargs)]
            else:
                all_init_kwargs = []

                # Pretty sure .values() and .keys() would give same order, but doing it this was just in case :)
                items = list(multi_init_kwargs.items())
                # {"a": [1, 2, 3], "b": [4, 5]} -> []
                # zipped_arr =
                keys = [item[0] for item in items]
                arrs = [item[1] for item in items]
                arr_prod = list(itertools.product(*arrs))
                d_prod = [{keys[i]: val for i, val in enumerate(el)} for el in arr_prod]
                all_multi = [dot_d_to_nested(dot_d) for dot_d in d_prod]

                for d in all_multi:
                    curr_init_kwargs = copy.deepcopy(base_init_kwargs)
                    copy_to_dict_recr(from_dict=d, to_dict=curr_init_kwargs)
                    all_init_kwargs.append(curr_init_kwargs)

            for init_kwargs in all_init_kwargs:
                curr_learner_config = {
                    "name": learner,
                    "init_kwargs": init_kwargs,
                    "learn_kwargs": learner_configs[learner]["learn"]
                }

                logger_config = copy.deepcopy(self._exp_config["logger"])
                logger_config["name"] = f"run_{idx}"
                logger_config["log_dir"] = self._logs_path

                curr_config = {
                    "learner": curr_learner_config,
                    "exp": {
                        "name": self._exp_config["exp"]["name"],
                        "id": self._exp_config["exp"]["id"],
                        "type": self._exp_config["exp"]["type"],
                        "details": self._exp_config["exp"]["details"],
                    },
                    "logger": logger_config,
                }

                all_configs.append(curr_config)

                idx += 1

        self._run_configs = all_configs
        # print(f">>> Saving {len(all_configs)} run configs to {self._runs_path} ")
        for i in tqdm(range(len(all_configs)), desc="Saving run configs: "):
            curr_path = join_paths([str(self._runs_path), f"run_{i}.json"])
            with open(curr_path, "w") as f:
                json.dump(self._run_configs[i], f, indent=4)
    ###################################

    def _load_env_mc_configs(
            self,
            base_envs: Optional[List[str]] = None,
            env_mcs: Optional[List[List[str]]] = None,
            env_mc_overrides: Optional[Dict[str, Any]] = None
    ):
        if base_envs is not None and env_mcs is not None:
            raise ValueError(f"Cannot specify both base_envs and env_mcs.")

        # (env_id, mc_id) -> (env_overrides, mc_overrides)
        env_mc_configs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        with open(join_paths([BASE_PATH, "configs", "environment.json"]), "r") as f:
            all_env_config = json.load(f)
        env_version = all_env_config["version"]

        if base_envs is not None:
            for base_env in base_envs:
                if base_env not in all_env_config:
                    raise ValueError(f"Base env {base_env} not found in environment.json.")
                base_env_config = all_env_config[base_env]
                if "kwargs" not in base_env_config:
                    base_env_config["kwargs"] = {}

                # Based off default kwargs as per arg in base_env_config
                base_env_kwargs = all_env_config[base_env_config["default"]]

                for variant in base_env_config["variants"]:
                    env_id = f"{base_env}-{variant}-v{env_version}"
                    env_kwargs = copy.deepcopy(base_env_kwargs)
                    copy_to_dict_recr(from_dict=base_env_config["kwargs"], to_dict=env_kwargs)
                    if env_mc_overrides is not None:
                        copy_to_dict_recr(from_dict=env_mc_overrides, to_dict=env_kwargs)

                    for mc_id in base_env_config["variants"][variant]:
                        env_mc_configs[(env_id, mc_id)] = copy.deepcopy(env_kwargs)


        elif env_mcs is not None:
            for env_mc_arr in env_mcs:
                env_id, mc_id = tuple(env_mc_arr)
                base_env, variant, version = env_id.split("-")

                if base_env not in all_env_config:
                    raise ValueError(f"Base env {base_env} not found in environment.json.")
                base_env_config = all_env_config[base_env]
                if "kwargs" not in base_env_config:
                    base_env_config["kwargs"] = {}

                if variant not in base_env_config["variants"]:
                    raise ValueError(f"env_id = {env_id} invalid. Variant {variant} not found in environment.json "
                                     f"for base env {base_env}.")

                base_env_kwargs = all_env_config[base_env_config["default"]]
                env_kwargs = copy.deepcopy(base_env_kwargs)
                copy_to_dict_recr(from_dict=base_env_config["kwargs"], to_dict=env_kwargs)
                if env_mc_overrides is not None:
                    copy_to_dict_recr(from_dict=env_mc_overrides, to_dict=env_kwargs)

                env_mc_configs[(env_id, mc_id)] = copy.deepcopy(env_kwargs)

            # raise NotImplementedError(f"env_mcs arg not yet supported.")
        else:
            raise ValueError(f"Must specify either base_envs or env_mcs.")

        return env_mc_configs
    
    
    def _load_learner_configs(self):
        static_configs = {}
        raw_config = self._exp_config["learner"]
        names = raw_config["names"]

        #######################
        # --- INIT KWARGS --- #
        #######################
        raw_init_config = raw_config["init"]

        # Static Kwargs #
        # 1 get common init static kwargs
        # 1.a load common kwargs via default config
        common_init_config_path = join_paths([BASE_PATH, "configs", "learner", raw_init_config["default"]["common"]])
        with open(common_init_config_path, "r") as f:
            common_init_kwargs = json.load(f)["init"]
        # 1.b override/set specified common static kwargs
        copy_to_dict_recr(
            from_dict=raw_init_config["static_kwargs"]["common"],
            to_dict=common_init_kwargs
        )  # Copy common static kwargs specified in exp config

        for name in names:
            # 2.a use common static init kwargs as starting point for current learner
            curr_init_kwargs = copy.deepcopy(common_init_kwargs)

            # 2.b load default static init kwargs via specified config file in init.default.{name}
            tmp_path = join_paths([BASE_PATH, "configs", "learner", raw_init_config["default"][name]])
            with open(tmp_path, "r") as f:
                curr_def_kwargs = json.load(f)["init"]
            copy_to_dict_recr(from_dict=curr_def_kwargs, to_dict=curr_init_kwargs)

            # 2.c override static init kwargs based off those specified in exp config
            copy_to_dict_recr(
                from_dict=raw_init_config["static_kwargs"][name],
                to_dict=curr_init_kwargs
            )

            static_configs[name] = {
                "init": curr_init_kwargs,
            }

        # Multi Kwargs #
        if raw_config["init"]["multi_kwargs"] is not None:
            common_multi_kwargs = raw_init_config["multi_kwargs"]["common"]
            multi_kwargs = {}
            for name in names:
                multi_kwargs[name] = copy.deepcopy(common_multi_kwargs)
                copy_to_dict_recr(
                    from_dict=raw_init_config["multi_kwargs"][name],
                    to_dict=multi_kwargs[name]
                )
        else:
            multi_kwargs = {name: {} for name in names}

        # configs = copy.deepcopy(static_configs)
        #
        # for name in names:
        #     # current init kwargs
        #     curr_init = []
        #     # curr_multi - current multi kwargs
        #     for curr_multi in

        configs = copy.deepcopy(static_configs)
        for name in names:
            configs[name]["init"] = {
                "static": static_configs[name]["init"],
                "multi": multi_kwargs[name],
            }

        #######################

        #######################
        # --- LEARN KWARGS --- #
        #######################
        raw_learn_config = raw_config["learn"]
        # 1 get common learn static kwargs
        # 1.a load common kwargs via default config
        common_learn_config_path = join_paths([BASE_PATH, "configs", "learner", raw_learn_config["default"]["common"]])
        with open(common_learn_config_path, "r") as f:
            common_learn_kwargs = json.load(f)["learn"]
        # 1.b override/set specified common static kwargs
        copy_to_dict_recr(
            from_dict=raw_learn_config["static_kwargs"]["common"],
            to_dict=common_learn_kwargs
        )  # Copy common static kwargs specified in exp config

        for name in names:
            # 2.a use common static learn kwargs as starting point for current learner
            curr_learn_kwargs = copy.deepcopy(common_learn_kwargs)

            # 2.b load default static learn kwargs via specified config file in learn.default.{name}
            tmp_path = join_paths([BASE_PATH, "configs", "learner", raw_learn_config["default"][name]])
            with open(tmp_path, "r") as f:
                curr_def_kwargs = json.load(f)["learn"]
            copy_to_dict_recr(from_dict=curr_def_kwargs, to_dict=curr_learn_kwargs)

            # 2.c override static learn kwargs based off those specified in exp config
            copy_to_dict_recr(
                from_dict=raw_learn_config["static_kwargs"][name],
                to_dict=curr_learn_kwargs
            )

            configs[name]["learn"] = curr_learn_kwargs
        #######################

        return configs

    #######################
    # -- CREATE SLURMS -- #
    #######################
    @staticmethod
    def _replace_templated(txt, d):
        for key, val in d.items():
            txt = txt.replace("{" + key + "}", val)
        return txt

    def _load_slurm_template(self):
        template_path = join_paths([BASE_PATH, "template.slurm"])
        with open(template_path, "r") as f:
            template_txt = f.read()

        cluster_config = self._exp_config["cluster"]

        replace_d = {
            "conda_env": cluster_config["conda_env"],
            "slurm_logs_folder": self._slurm_logs_path,
            "partition": cluster_config["partition"]
        }
        template_txt = self._replace_templated(template_txt, replace_d)
        return template_txt

    def _create_slurms(self):
        cluster_config = self._exp_config["cluster"]
        exp_id = self._exp_config["exp"]["id"]
        n_runs = len(self._run_configs)

        template_txt = self._load_slurm_template()

        n_seq_procs = cluster_config["n_seq_procs"]
        n_parallel_procs = cluster_config["n_par_node_procs"]

        runs_per_node = n_seq_procs * n_parallel_procs
        n_nodes = n_runs // runs_per_node + (1 if n_runs % runs_per_node > 0 else 0)
        if n_nodes > cluster_config["max_nodes"]:
            raise ValueError(
                f"Number of nodes ({n_nodes}) required is greater than maximum number of nodes ({cluster_config['max_nodes']})")

        ########################
        # -- Setup & Finish -- #
        ########################
        node_logs_path = f"{NODE_TMP_PATH}/logs"
        node_zipped_logs_path = f"{NODE_TMP_PATH}/zipped_logs"

        # node_models_path = f"{NODE_TMP_PATH}/logs"
        # node_zipped_models_path = f"{NODE_TMP_PATH}/zipped_logs"

        # SETUP #
        setup_commands = f"rm -rf {NODE_TMP_PATH}\n"
        setup_commands += f"mkdir {node_logs_path} -p\n"
        setup_commands += f"mkdir {node_zipped_logs_path} -p\n"

        # setup_commands += f"mkdir {node_models_path} -p\n"
        # setup_commands += f"mkdir {node_zipped_models_path} -p\n"
        template_txt = template_txt.replace("{setup_commands}", setup_commands)

        # FINISH #
        finish_commands = ""
        finish_commands += f"zip -r {node_zipped_logs_path}/runs_{{run_i}}-{{run_j}}.zip {node_logs_path}/* \n"
        finish_commands += f"mv {node_zipped_logs_path}/runs_{{run_i}}-{{run_j}}.zip {self._zipped_logs_path}/runs_{{run_i}}-{{run_j}}.zip \n"

        # finish_commands += f"zip -r {node_zipped_models_path}/runs_{{run_i}}-{{run_j}}.zip {node_models_path}/* \n"
        # finish_commands += f"mv {node_zipped_models_path}/runs_{{run_i}}-{{run_j}}.zip {self._zipped_models_path}/runs_{{run_i}}-{{run_j}}.zip \n"
        finish_commands += f"rm -rf {NODE_TMP_PATH}\n"

        template_txt = template_txt.replace("{finish_commands}", finish_commands)
        ########################

        template_txt = template_txt.replace("{node_excludes}", cluster_config["node_excludes"])


        cli_path = join_paths([comp_base_path(BASE_PATH, level=1), "cli.py"])

        for slurm_id in tqdm(range(n_nodes), desc="Creating slurms: "):
            # start_id for run config
            start_id = slurm_id * runs_per_node
            # end_id for run config
            end_id = min(start_id + runs_per_node, n_runs)
            job_name = f"{exp_id}_{slurm_id}"

            slurm_str = template_txt.replace("{job_name}", job_name)
            slurm_str = slurm_str.replace("{run_i}", str(start_id))
            slurm_str = slurm_str.replace("{run_j}", str(end_id - 1))

            run_commands = ""
            for i in range(start_id, end_id, n_seq_procs):
                run_commands += f'echo "Attempting to run run{i}.pkl - run{i + n_seq_procs - 1}.pkl"\n'
                # run_commands += f"python3 {CLUSTER_CODE_PATH}benchmarker.py --run {self.runs_folder}/run{i}.pkl " \
                #                 f"--node $SLURM_JOB_NODELIST " \
                #                 f"-rr {n_seq_repeats}" \
                #                 f" > {logs_path}/run{i}.out 2>&1 & \n"
                runs = [f"run_{j}.json" for j in range(i, i + n_seq_procs)]
                run_paths = [join_paths([self._runs_path, curr_run]) for curr_run in runs]
                runs_str = " ".join(run_paths)

                run_names = [f"run_{j}" for j in range(i, i + n_seq_procs)]

                curr_logs_arr = [join_paths([node_logs_path, run_name]) for run_name in run_names]
                log_multi_path = " ".join(curr_logs_arr)

                # curr_models_arr = [join_paths([node_models_path, run_name]) for run_name in run_names]
                # model_multi_path = " ".join(curr_models_arr)

                run_commands += f"python3 {cli_path} --exec-multi-run {runs_str} " \
                                + f"--node $SLURM_JOB_NODELIST " \
                                + f"--log-multi-path {log_multi_path} " \
                                # + f"--model-multi-path {model_multi_path} "

                run_commands += f" > {self._run_logs_path}/run_{i}-{i + n_seq_procs - 1}.out 2>&1 & \n"

            slurm_str = slurm_str.replace("{run_commands}", run_commands)

            slurm_path = join_paths([self._slurms_path, f"job_{slurm_id}.slurm"])
            with open(slurm_path, "w") as f:
                f.write(slurm_str)

    ##################

    def exec_slurms(self):
        import subprocess
        slurm_files = os.listdir(self._slurms_path)
        max_nodes = self._exp_config["cluster"]["max_nodes"]
        assert len(slurm_files) <= max_nodes, "Too many slurm files requested"

        for file in slurm_files:
            curr_file_path = join_paths([self._slurms_path, file])
            subprocess.run(["sbatch", curr_file_path])

def main():
    benchmarker = Benchmarker("exp_d0.json", confirm_dir_delete=False)
    benchmarker.create()

if __name__ == "__main__":
    main()