from typing import Callable, Dict, Any

import omnisafe


class Learner:
    def __init__(
            self,
            env_fn: Callable,
            eval_env_fn: Callable,

    ):
        pass


class SafeLearner:
    def __init__(
            self,
            env_fn: Callable,
            eval_env_fn: Callable,
            algo_name: str,
            train_kwargs: Dict[str, Any],
            algo_kwargs: Dict[str, Any],
            model_kwargs: Dict[str, Any],
            logger_kwargs: Dict[str, Any]
    ):
        env_id = "..."  # TODO
        custom_cfgs = {
            "train_cfgs": train_kwargs,
            "algo_cfgs": algo_kwargs,
            "model_cfgs": model_kwargs,  # Not sure if this is valid :)
            "logger_cfgs": logger_kwargs
        }

        self._os_agent = omnisafe.Agent(algo_name, env_id, custom_cfgs=custom_cfgs)

    def evaluate(self):
        pass

    def learn_and_evaluate(self):
        pass
