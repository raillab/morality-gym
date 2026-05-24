# Baselines

This directory contains the experiment stack used to run baseline agents on Morality Gym environments. It is separate from the lightweight `morality_gym` environment package.

## Layout

| Path | Purpose |
| --- | --- |
| `algorithms/` | Learner wrappers for random, SB3, and OmniSafe-style agents |
| `wrappers/` | Environment adapters that add costs, reward shaping, and CMDP interfaces |
| `benchmarker/` | Config-driven experiment generation, local execution, and SLURM generation |
| `data_proc/` | Research scripts for consolidating logs into CSV outputs |
| `cli.py` | Command-line entry point for creating and running benchmark jobs |

## Learners

The benchmarker resolves learner names in `baselines/benchmarker/run.py`.

| Config name | Implementation | Notes |
| --- | --- | --- |
| `random` | `RandomLearner` | Random-action baseline |
| `ppo` | `SB3Learner` | Stable-Baselines3 PPO without cost shaping |
| `ppo_shaped` | `SB3Learner` | Stable-Baselines3 PPO with reward shaped by cost |
| `PPO`, `PPOShaped`, `PPOLag`, `CPO`, `CPPOPID`, `TRPOPID` | `OSLearner` | OmniSafe-backed learners; `PPOShaped` maps to OmniSafe `PPO` with reward shaping |

## Running One Experiment Locally

Commands below assume you are at the repository root.

Create run configs and SLURM scripts from an experiment config:

```bash
python baselines/cli.py --create exp_p9xe.json
```

Run one generated config:

```bash
python baselines/cli.py --exec-run p9xe/run_0.json
```

The path passed to `--create` is relative to `baselines/benchmarker/configs/experiment/`. The path passed to `--exec-run` is relative to the `runs` base path configured in `baselines/benchmarker/paths.json`.

On first use, the benchmarker prompts for base paths and writes `baselines/benchmarker/paths.json`. The required keys are:

| Key | Meaning |
| --- | --- |
| `runs` | Generated per-run JSON configs |
| `slurms` | Generated SLURM job scripts |
| `logs` | Local run outputs |
| `zipped_logs` | Archived node logs copied back from SLURM jobs |
| `run_logs` | stdout/stderr from individual run batches |
| `slurm_logs` | scheduler-level SLURM logs |

Example local-only path choices:

```text
runs: /tmp/morality-gym/runs
slurms: /tmp/morality-gym/slurms
logs: /tmp/morality-gym/logs
zipped_logs: /tmp/morality-gym/zipped_logs
run_logs: /tmp/morality-gym/run_logs
slurm_logs: /tmp/morality-gym/slurm_logs
```

## Experiment Configs

Experiment configs live in `baselines/benchmarker/configs/experiment/`. A config has five main blocks:

| Block | Purpose |
| --- | --- |
| `exp` | Name, id, type, details, and seeds |
| `env` | Which environment and morality-chain pairs to run |
| `learner` | Learner names plus init and learn defaults/overrides |
| `logger` | Local logger mode and per-run log root |
| `cluster` | SLURM settings used when generating job files |

Minimal structure:

```json
{
  "exp": {
    "name": "ppo_lag_x_easy",
    "id": "pl9xe",
    "type": "learn_and_eval",
    "details": "",
    "seeds": [41, 42, 43]
  },
  "env": {
    "base_envs": ["SwitchStandard", "PushStandard"],
    "env_mcs": null,
    "env_mc_overrides": {
      "mc_overrides": {
        "beta": 0.1
      }
    }
  },
  "learner": {
    "names": ["PPOLag"],
    "init": {
      "default": {
        "common": "x_easy_common_v2.json",
        "PPOLag": "ppo_lag_default_v3.json"
      },
      "static_kwargs": {
        "common": {},
        "PPOLag": {}
      },
      "multi_kwargs": null
    },
    "learn": {
      "default": {
        "common": "x_easy_common_v2.json",
        "PPOLag": "ppo_lag_default_v3.json"
      },
      "static_kwargs": {
        "common": {},
        "PPOLag": {}
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
    "max_nodes": 20,
    "node_excludes": "8-9,11-13"
  }
}
```

## Logs

Each local run writes:

```text
{logs}/{exp_id}/run_{i}/
  config.json
  metadata.json
  metrics/eval.csv
  files/
  models/
  videos/
```

`metrics/eval.csv` is the main evaluation output. It includes average return, average cost, morality metric, goal reach rate, and per-norm morality function values where available.

## Result CSVs

The scripts in `baselines/data_proc/` are research utilities for consolidating logs. They currently assume particular raw-data locations under `baselines/data/` and may need path edits before reuse on a new machine.

Typical outputs are:

| File | Meaning |
| --- | --- |
| `learn_eval.csv` | Evaluation rows during training, excluding final row |
| `final_eval.csv` | Final evaluation row per run |
| `final_mf_eval.csv` | Final per-norm morality function rows |
| `agr_eval.csv` | Aggregated min/max evaluation bounds for bound experiments |

When final result CSVs are added to the repository, keep the raw generated logs separate from the publication-ready CSVs so the provenance is clear.
