# Run
Use `exec_run` from benchmarker/run to create a training run using the specified kwargs in the specified config file.

`exec_run(config_path, learner_overrides, is_interactive_env)`

 - `config_path`: path relative to benchmarker/configs to json config file for run
 - `learner_overrides`: ignore this
 - `is_interactive_env`: set to true if you wish to test env in interactive mode. For debugging not running experiments.



### Run Configs
 - Everything from `learner_kwargs` gets passed as kwargs to `SafeLearner` object. 
   - `algo`, `train_kwargs`, `algo_kwargs`, `model_kwargs`, `logger_kwargs` are used for omnisafe agent creation
 - Note: only the following algo values are supported currently: `CPO`, `PPOLag`, `TRPO`, `PPO`

#### Examples
Some example run configs are in `example`.
 - `example/run_2l.json` - 2 lavas. Top left and bottom right corners.  (easiest)
 - `example/run_1l.json` - 1 lava in middle (easy) 
 - `example/run_2l_1h.json` - 2 lavas. 1 human. (harder)
These examples all use CPO as alg.
 
#### Template
 - Note: Comments added are not valid JSON. You must remove these
 
````
{
    {
      "name": "",  # This isn't used for anything
      "run_type":  "learn",  # Do not change
      "learner_kwargs": { 
        "seed": 42,
        "algo_name": "PPOLag",
        "train_kwargs": {
          "total_steps": 50000,
          "eval_freq": 1,  # eval_freq is multiple of algo_kwargs['steps_per_epoch'] and denotes freq of eval function being called
          "vector_env_nums": 1,
          "parallel": 1
        },
        "algo_kwargs": {
            "steps_per_epoch": 2048,
            "update_iters": 1
        },
        "model_kwargs": {},
        "logger_kwargs": {
          "log_dir": "./logs",  # Path relative to where you are calling code from
          "wandb_project": "safebench",
          "use_wandb": true
        },
        "env_config": "scenarios/static/static_1l_mid.json",  # Env config path relative to morality_gym/environment/configs 
        "env_overrides": {
          "flatten_observation": true,
          "action_discrete": false,
          "harm_obs_verb": 1
        },
        "cost_kwargs": {
          "scale": 1.0
        },
        "env_time_limit": 1024,
        "env_max_episode_steps": 1024,
        "morality_tree_config": "asimov_3_laws.json",
        "n_eval_episodes": 20,
        "max_eval_steps": 1000,
        "vis_eval_episode": true  # Whether to create gifs of visualised eval trajectories. Set this to false if using cluster since it will give issues.
      }
    }
}
````