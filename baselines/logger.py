import datetime
import json
import os
import shutil
from typing import Dict, Any, Optional, List
import copy

from morality_gym._utils.common import create_or_clear_folder, join_paths


class Logger:
    def __init__(
            self,
            name: str,
            log_dir: str,
            config: Optional[Dict[str, Any]] = None,
            project: Optional[str] = None,
            group: Optional[str] = None,
            job_type: Optional[str] = None,
            tags: Optional[List[Any]] = None,
            mode: str = "local",
            reinit: bool = True,
    ):
        self._config = config
        self._name = name
        self._log_dir = log_dir
        self._project = project
        self._group = group
        self._job_type = job_type
        self._tags = tags
        self._mode = mode
        self._reinit = reinit
        if not reinit:
            raise NotImplementedError("reinit == False not currently supported")

        self._is_finished = False
        self._log_path = join_paths([self._log_dir, self._name])
        self._is_disabled = mode == "disabled"

        self._run_start = datetime.datetime.now()
        self._run_end = None
        self._time_elapsed = None
        self._system_specs = self._comp_system_specs()

        # Metadata
        self._metadata = {
            "name": name,
            "project": project,
            "group": group,
            "job_type": job_type,
            "tags": tags,
            "system_specs": self._system_specs,
            "mode": mode,
            "reinit": reinit,
            "finished": False,
            "start_time": self._run_start.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": None,
            "duration": None,
        }

        # Paths
        self._metrics_base_path = join_paths([self._log_path, "metrics"])
        self._files_base_path = join_paths([self._log_path, "files"])

        self._videos_path = join_paths([self._log_path, "videos"])
        self._models_path = join_paths([self._log_path, "models"])

        self._file_groups = []

        self._sub_folders = {}

        self._log_files = {}

        # self._metric_groups = []
        # self._file_groups = []  # Required?

        self._metrics = {}  # Grouped by metric_group

        self._setup()

    #################
    # -- HELPERS -- #
    #################
    @staticmethod
    def _comp_system_specs():
        # TODO
        system_specs = {}
        return system_specs

    ###############
    # -- SETUP -- #
    ###############
    def _setup(self):
        if self._is_disabled:
            return

        if self._mode == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unknown mode {self._mode}")

    def _setup_local(self):
        # TODO:
        # 1. Set up folders
        # 2. Save config.json & metadata.json

        #################
        # SETUP FOLDERS #
        #################
        if self._reinit:
            create_or_clear_folder(self._log_path, confirm=False)
        else:
            raise NotImplementedError("reinit == False not currently supported")

        os.mkdir(join_paths([self._log_path, "metrics"]))
        os.mkdir(join_paths([self._log_path, "files"]))
        os.mkdir(self._models_path)
        os.mkdir(self._videos_path)
        #################

        # SAVE CONFIG & METADATA #
        with open(join_paths([self._log_path, "config.json"]), "w") as f:
            json.dump(self._config, f, indent=1)
        with open(join_paths([self._log_path, "metadata.json"]), "w") as f:
            json.dump(self._metadata, f, indent=1)

    ###############

    def create_sub_folder(self, name: str):
        # raise NotImplementedError
        if self._is_disabled:
            return

        if name in self._sub_folders:
            print(f"WARNING: Sub-folder already exists, skipping create_sub_folder for {name}.")
        else:
            curr_path = join_paths([self._log_path, name])
            if os.path.exists(curr_path):
                raise FileExistsError(f"Invalid name={name}. Sub-folder already exists at {curr_path}.")
            else:
                os.mkdir(curr_path)
                self._sub_folders[name] = curr_path

    def sub_folder_path(self, name: str):
        return self._sub_folders[name]

    #################
    # -- METRICS -- #
    #################
    def define_metrics(
            self,
            metrics: List[str],
            metric_group: Optional[str] = None,
            step_metric: Optional[str] = None,
    ):
        if self._is_disabled:
            return

        if self._mode == "local":
            if metric_group is None:
                raise ValueError("Must provide metric group when in local mode")
            self._define_metrics_local(metrics, metric_group)
        else:
            raise ValueError(f"Unknown mode {self._mode}")

    def _define_metrics_local(
            self,
            metrics: List[str],
            metric_group: str
    ):
        if self._is_disabled:
            return

        if metric_group in self._metrics:
            raise NotImplementedError(f"Defining metrics for existing group is not currently supported")
        else:
            # self._metric_groups.append(metric_group)
            self._metrics[metric_group] = metrics
            self._log_files[metric_group] = open(join_paths([self._metrics_base_path, f"{metric_group}.csv"]), "w")
            cols_str = ",".join(metrics)
            self._log_files[metric_group].write(f"{cols_str}")
    #################

    ###############
    # --- LOG --- #
    ###############
    def log(
            self,
            d: Dict[str, Any],
            metric_group: Optional[str] = None,
            _check_keys: bool = False,
    ):
        if self._is_disabled:
            return

        if self._is_finished:
            raise ValueError("Cannot log in finished logger")

        if self._mode == "local":
            if metric_group is None:
                raise ValueError("Must provide group when logging in local mode")
            self._log_local(d, metric_group, _check_keys=_check_keys)
        else:
            raise ValueError(f"Unknown mode {self._mode}")

    def _log_local(
            self,
            d: Dict[str, Any],
            metric_group: str,
            _check_keys: bool = False,
    ):
        if _check_keys:
            if not set(d.keys()).issubset(set(self._metrics[metric_group])):
                # TODO: Check this is correct
                raise ValueError(f"Invalid keys in d: ", set(d.keys()).difference(set(self._metrics[metric_group])))

        row_arr = [str(d[col]) if (col in d and d[col] is not None) else "" for col in self._metrics[metric_group]]
        row_str = ",".join(row_arr)
        self._log_files[metric_group].write(f"\n{row_str}")

    def log_file(
            self,
            file_path: str,
            group: str,
            mode: str = "copy",
    ):
        if self._is_disabled:
            return

        if self._is_finished:
            raise ValueError("Cannot log file in finished logger")

        if self._mode == "local":
            self._log_file_local(file_path, group, mode)
        else:
            raise ValueError(f"Unknown mode {self._mode}")


    def _log_file_local(
            self,
            file_path: str,
            group: str,
            mode: str = "copy",
    ):
        dst_bast_path = join_paths([self._files_base_path, group])
        if group not in self._file_groups:
            os.mkdir(dst_bast_path)

        file_name = os.path.basename(file_path)
        dst_path = join_paths([dst_bast_path, file_name])

        if mode == "copy":
            shutil.copy(file_path, dst_path)
        elif mode == "move":
            shutil.move(file_path, dst_path)
        else:
            raise ValueError(f"Unknown mode {mode}")
    ##############

    def move(self, dst: str):
        if self._is_disabled:
            raise ValueError("Cannot move logger in disabled mode")
        pass

    def zip(
            self,
            dst: str,
            remove: bool=False
    ):
        # TODO: Check
        if self._is_disabled:
            raise ValueError("Cannot zip logger in disabled mode")
        if not self._is_finished:
            raise ValueError("Cannot zip logger that has not finished")

        shutil.make_archive(dst, "zip", self._log_path)

        if remove:
            shutil.rmtree(self._log_path)

    def finish(self):
        if self._is_disabled:
            return

        for f in self._log_files.values():
            f.close()

        ############
        # METADATA #
        ############
        self._metadata["finished"] = True
        self._run_end = datetime.datetime.now()
        self._time_elapsed = (self._run_end - self._run_start).total_seconds()
        self._metadata["duration"] = self._time_elapsed
        self._metadata["end_time"] = self._run_end.strftime('%Y-%m-%d %H:%M:%S')

        with open(join_paths([self._log_path, "metadata.json"]), "w") as f:
            json.dump(self._metadata, f, indent=1)
        ############
        self._is_finished = True

    ####################
    # -- PROPERTIES -- #
    ####################
    @property
    def is_finished(self):
        return self._is_finished

    @property
    def name(self):
        return self._name

    @property
    def log_path(self):
        return self._log_path

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)
    ####################


def main():
    logger = Logger(
        name="test_log",
        log_dir="tmp_logs",
        config={"name": "test"},
        project="test",
        group="test",
        job_type="test",
        tags=["test"],
        mode="local",
        reinit=True
    )
    logger.define_metrics(["tot_steps", "tot_episodes", "avg_return"], "eval")
    logger.log({"tot_steps": 10.0, "tot_episodes": 100, "avg_return": 100}, metric_group="eval")
    with open("tmp.txt", "w") as f:
        f.write("test")
    logger.log_file("tmp.txt", "test", "move")

    logger.finish()
    # logger.zip(logger.name, remove=True)

if __name__ == "__main__":
    main()
