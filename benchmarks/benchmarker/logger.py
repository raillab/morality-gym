import os
import typing
from typing import Optional, List, Any, Dict, Union, Tuple

import wandb

from morality_gym.utils.common import join_paths, create_or_clear_folder

METRIC_TUP = Tuple[str, str, str]  # name, step_metric, summary


class Logger:
    ##################
    # ---- INIT ---- #
    ##################
    def __init__(
            self,
            log_mode: str = 'wandb', # 'wandb', 'local' or 'disabled'

            project: Optional[str] = None,
            base_dir: Optional[str] = None,
            name: Optional[str] = None,
            notes: Optional[str] = None,
            tags: Optional[List[str]] = None,
            run_group: Optional[str] = None,
            job_type: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,

            metrics: Optional[Dict[str, List[METRIC_TUP]]] = None,

            # wandb
            wandb_entity: str = None,
            wandb_mode: str = 'online',
    ):
        # Log mode & enabled
        self._log_mode = log_mode
        self._enabled = log_mode != 'disabled'

        # General
        self._project = project
        self._name = name
        self._notes = notes
        self._tags = tags
        self._run_group = run_group
        self._job_type = job_type
        self._config = config

        # Metrics
        self._metrics = metrics

        # Dirs
        self._base_dir = base_dir
        self._log_dir = None
        self._files_dir = None
        self._artifacts_dir = None
        self._metrics_dir = None
        self._temp_dir = None

        # Wandb
        self._wandb_run: Optional[wandb.run] = None
        self._wandb_entity = wandb_entity
        self._wandb_mode = wandb_mode

        # Local
        # ...

        # log_mode specific init
        if self._enabled:
            if self._log_mode == 'wandb':
                self._init_wandb()
            elif self._log_mode == 'local':
                self._init_local()
            else:
                raise ValueError(f"Invalid log mode: {self._log_mode}. "
                                 f"Valid modes are: ['wandb', 'local', 'disabled']")

    def _init_local(self):
        raise NotImplementedError

    def _init_wandb(self):
        self._wandb_run = wandb.init(
            entity=self._wandb_entity, project=self._project, name=self._name, notes=self._notes, tags=self._tags,
            dir=self._base_dir, mode=self._wandb_mode, reinit=True, config=self._config
        )

        self._log_dir = os.path.normpath(os.path.dirname(self._wandb_run.dir))

        self._files_dir = join_paths([self._log_dir, "files"])
        self._artifacts_dir = join_paths([self._log_dir, "artifacts"])
        self._metrics_dir = join_paths([self._log_dir, "metrics"])

        self._temp_dir = join_paths([self._log_dir, "temp"])
        create_or_clear_folder(self._temp_dir)

        # Metrics
        for metric_group, metric_arr in self._metrics.items():
            for metric, step_metric, summary in metric_arr:
                self._wandb_run.define_metric(metric, step_metric=step_metric, summary=summary)

    ##################

    ##################
    # PUBLIC METHODS #
    ##################
    def update_config(
            self,
            config: Dict[str, Any]
    ):
        if self._enabled:
            if self._log_mode == 'wandb':
                self._wandb_update_config(config)
            else:
                self._local_update_config(config)

    def update_metrics(
            self,
            metrics: Optional[Dict[str, List[METRIC_TUP]]]
    ):
        if self._enabled:
            if self._log_mode == 'wandb':
                self._wandb_update_metrics(metrics)
            else:
                self._local_update_metrics(metrics)

    def log(
            self,
            data: Dict[str, Any],
            group: Optional[str] = None
    ):
        if self._enabled:
            if self._log_mode == 'wandb':
                self._wandb_log(data)
            else:
                self._local_log(data, group)

    def log_file(
            self,
            file: Union[str, typing.IO]
    ):
        if self._enabled:
            if self._log_mode == 'wandb':
                self._wandb_log_file(file)
            else:
                self._local_log_file(file)

    def log_artifact(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ) -> Optional[wandb.Artifact]:
        if self._enabled:
            if self._log_mode == 'wandb':
                return self._wandb_log_artifact(file_path, name, artifact_type, aliases)
            else:
                return self._local_log_artifact(file_path, name, artifact_type, aliases)

    def finish(self):
        if self._enabled:
            if self._log_mode == 'wandb':
                self._wandb_finish()
            else:
                self._local_finish()

    ##################

    #################
    # --- WANDB --- #
    #################
    def _wandb_update_config(
            self,
            config: Dict[str, Any]
    ):
        raise NotImplementedError

    def _wandb_update_metrics(
            self,
            metrics: Optional[Dict[str, List[METRIC_TUP]]]
    ):
        for metric_group, metric_arr in metrics.items():
            if metric_group is self._metrics:
                self._metrics[metric_group].extend(metric_arr)
                # TODO: Check below statement
                self._metrics[metric_group] = list(set(self._metrics[metric_group]))  # Remove repeats
            else:
                self._metrics[metric_group] = metric_arr

            for metric, step_metric, summary in metric_arr:
                self._wandb_run.define_metric(metric, step_metric=step_metric, summary=summary)

    def _wandb_log(
            self,
            data: Dict[str, Any]
    ):
        self._wandb_run.log(data, commit=True)

    def _wandb_log_file(
            self,
            file: Union[str, typing.IO]
    ):
        raise NotImplementedError

    def _wandb_log_artifact(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ) -> wandb.Artifact:
        raise NotImplementedError

    def _wandb_finish(self):
        self._wandb_run.finish()
    #################

    #################
    # --- LOCAL --- #
    #################
    def _local_update_config(
            self,
            config: Dict[str, Any]
    ):
        raise NotImplementedError

    def _local_update_metrics(
            self,
            metrics: Optional[Dict[str, List[METRIC_TUP]]]
    ):
        raise NotImplementedError

    def _local_log(
            self,
            data: Dict[str, Any],
            group: Optional[str] = None
    ):
        raise NotImplementedError

    def _local_log_file(
            self,
            file: Union[str, typing.IO]
    ):
        raise NotImplementedError

    def _local_log_artifact(
            self,
            file_path: str,
            name: Optional[str] = None,
            artifact_type: Optional[str] = None,
            aliases: Optional[List[str]] = None
    ):
        raise NotImplementedError

    def _local_finish(self):
        raise NotImplementedError
    #################


def test_wandb():
    metrics = {
        "learn": [("curr_step", None, "max"), ("curr_episode", None, "max"), ("avg_return", "curr_step", "max")]
    }
    config = {
        "test_arg": "Hello"
    }
    logger = Logger(
        log_mode='wandb',
        project='test', wandb_entity='simonrosen42', base_dir=r'.\wandb_logs',
        metrics=metrics, config=config
    )
    logger.log({"curr_step": 0, "curr_episode": 0, "avg_return": 2.5}, group="learn")
    logger.log({"curr_step": 1, "curr_episode": 2, "avg_return": 5}, group="learn")
    logger.finish()


def test_local():
    metrics = {
        "learn": [("curr_step", None, "max"), ("curr_episode", None, "max"), ("avg_return", "curr_step", "max")]
    }
    config = {
        "test_arg": "Hello"
    }
    logger = Logger(
        log_mode='local',
        project='test', base_dir=r'logs/debug_logger',
        metrics=metrics, config=config
    )
    logger.log({"curr_step": 0, "curr_episode": 0, "avg_return": 2.5}, group="learn")
    logger.log({"curr_step": 1, "curr_episode": 2, "avg_return": 5}, group="learn")
    logger.finish()


if __name__ == "__main__":
    test_local()
