from typing import Any, Dict, Optional, List, Tuple, SupportsFloat

from tqdm import tqdm

from baselines.algorithms.learner import BaseLearner
from baselines.logger import Logger


# N.B.: This is for evaluating a random policy - does not actually learn anything
class RandomLearner(BaseLearner):
    def __init__(
            self,
            env_id: str,
            mc_id: str,
            logger: Logger,
            seed: int,

            learn_mm_kwargs: Dict[str, Any],
            final_mm_kwargs: Dict[str, Any],

            env_overrides: Optional[Dict[str, Any]] = None,
            mc_overrides: Optional[Dict[str, Any]] = None,

            eval_metrics: Optional[List[str]] = None,

            vis_episode_kwargs: Optional[Dict[str, Any]] = None,

            return_bounds: Optional[Tuple[SupportsFloat, SupportsFloat]] = None,
    ):
        super().__init__(
            env_id=env_id, mc_id=mc_id,
            logger=logger, seed=seed,
            learn_mm_kwargs=learn_mm_kwargs, final_mm_kwargs=final_mm_kwargs,
            env_overrides=env_overrides, mc_overrides=mc_overrides,
            eval_metrics=eval_metrics,
            vis_episode_kwargs=vis_episode_kwargs,
            return_bounds=return_bounds
        )
        self._curr_step = 0
        self._curr_episode = 0

        self.eval_env.action_space.seed(seed)

    def eval_policy(self, obs: Any) -> Any:
        return self.eval_env.action_space.sample()

    def learn(
            self,
            n_timesteps: int,
            eval_freq: int,
            vis_eval_episodes: bool = False,
            save_models: bool = False,
    ):
        # TODO: Progress bar
        n_iters = n_timesteps // eval_freq

        with tqdm(total=n_iters+1, desc="Evals...") as prog_bar:
            for curr_iter in range(n_iters):
                self._curr_step = curr_iter * eval_freq
                self.evaluate("learn", log_metrics=True, vis_episode=vis_eval_episodes, save_model=save_models)
                prog_bar.update()

            self._curr_step += 1

            self.evaluate("final", log_metrics=True, vis_episode=vis_eval_episodes, save_model=save_models)
            prog_bar.update()


    def save_model(self):
        print("WARNING: No model to save for random learner, thus skipping.")

    @property
    def curr_step(self):
        return self._curr_step

    @property
    def curr_episode(self):
        return self._curr_episode

    def finish(self):
        # Nothing required here :)
        pass


def main():
    logger = Logger(
        name="debug",
        log_dir="debug",
        config={"name": "test"},
        project="test",
        group="test",
        job_type="test",
        tags=["test"],
        mode="local",
        reinit=True
    )

    env_overrides = {
        "env_overrides": {
            "render_mode": "rgb_array"
        },
    }
    vis_episode_kwargs = {
        "max_steps": 50
    }

    learner = RandomLearner(
        env_id="SwitchStandard-Standard-v1",
        mc_id="Utility",
        logger=logger,
        env_overrides=env_overrides,
        learn_mm_kwargs={"n_repeats": 10},
        final_mm_kwargs={"n_repeats": 10},
        vis_episode_kwargs=vis_episode_kwargs
    )
    learner.learn(
        n_timesteps=100,
        eval_freq=10,
        vis_eval_episodes=True,
    )
    logger.finish()


if __name__ == "__main__":
    main()
