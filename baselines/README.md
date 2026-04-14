# Baselines
## Benchmarker
Sets of experiments/runs are created through benchmarker code and config system.

Configs are stored in `benchmarker/configs/experiment`.

Example config structure (`exp_c3xe.json`). Note: Comments are added and NOT valid JSON.
````
{
    "exp": {
        "name": "cpo_3_extra_easy",
        # id is used for names of log directories & cluster stuff
        "id": "c3xe",  
        "type": "learn_and_eval",
        "details": "",
        "seeds": [41, 42, 43]
    },
    "env": {
        "base_envs": [
            "SwitchStandard", "PushStandard"
        ],
        "env_mcs": null,
        "env_mc_overrides": {
            "mc_overrides": {
                "beta": 0.1
            }
        }
    },
    "learner": {
        "names": ["CPO"],
        "init": {
            "default": {
                "common": "x_easy_common.json",
                "CPO": "cpo_default.json"
            },
            "static_kwargs": {
                "common": {},
                "CPO": {
                    "n_timesteps": 500000,
                    "cost_function_kwargs": {
                        "scale_fact": 1.0
                    },
                    "os_custom_cfgs": {
                        "algo_cfgs": {
                            "steps_per_epoch": 10000,
                            "update_iters": 20,
                            "batch_size": 64,
                            "cost_limit": 0.1
                        },
                        "model_cfgs": {
                            "actor_type": "discrete"
                        }
                    }
                }
            },
            "multi_kwargs": null
        },
        "learn": {
            "default": {
                "common": "x_easy_common.json",
                "CPO": "cpo_default.json"
            },
            "static_kwargs": {
                "common": {},
                "CPO": {
                    "n_timesteps": 500000
                }
            }
        }
    },
    "logger": {
        "mode": "local"
    },
    "cluster": {
        "conda_env": "asimov-new",
        "partition": "stampede",
        "n_seq_procs": 4,
        "n_par_node_procs": 4,
        "max_nodes": 20
    }
}
````

## CLI Usage
`cli.py` is used as the CLI to the benchmarking functionality.

To generate a set of experiments use:
`python cli.py --create {rel_config_path}`
where `{rel_config_path}` is path to config file relative to `benchmarker/configs/experiment`.

To run a particular experiment use 
`python cli.py --exec-run {exp_id}/run_{run_no}.json` where `{run_no}` is index of that run.

### Examples
Create all runs associated with `exp_c3xe.json` and run first run of that set of experiments.
```
python cli.py --create exp_c3xe.json
python cli.py --exec-run c3xe/run_0.json
```

Note these docs assume file directory structure of Windows which uses `/` so amend `c3xe/run_0.json` to `c3xe/run_0.json` if required.

### Misc
