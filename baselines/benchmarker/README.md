# Benchmarker

The benchmarker expands compact experiment configs into per-run JSON files, optionally generates SLURM jobs, and runs those per-run configs through `baselines.cli`.

Use this when reproducing the paper experiments from the full repository. For a small safe-RL training example, use `examples/train_safe_rl_agent.py` instead.

## Directory Structure

| Path | Purpose |
| --- | --- |
| `benchmarker.py` | Expands experiment configs and writes run configs plus SLURM scripts |
| `run.py` | Executes a single run config |
| `utils.py` | Loads local base paths from `paths.json` |
| `template.slurm` | Template used by generated SLURM jobs |
| `configs/environment.json` | Maps base environments to variants, morality chains, overrides, and return bounds |
| `configs/experiment/` | High-level experiment definitions |
| `configs/learner/` | Shared learner defaults used by experiment configs |

## First-Time Setup

Run any benchmarker command and it will ask for base paths if `paths.json` does not exist:

```bash
python baselines/cli.py --create exp_p9xe.json
```

The prompts create `baselines/benchmarker/paths.json`. Use absolute paths. For local testing, `/tmp/morality-gym/...` paths are usually enough.

Required keys:

```json
{
  "runs": "/tmp/morality-gym/runs",
  "slurms": "/tmp/morality-gym/slurms",
  "logs": "/tmp/morality-gym/logs",
  "zipped_logs": "/tmp/morality-gym/zipped_logs",
  "run_logs": "/tmp/morality-gym/run_logs",
  "slurm_logs": "/tmp/morality-gym/slurm_logs"
}
```

`paths.json` is machine-local configuration and should not be treated as a reproducibility artifact.

## Creating Runs

From the repository root:

```bash
python baselines/cli.py --create exp_p9xe.json
```

This reads:

```text
baselines/benchmarker/configs/experiment/exp_p9xe.json
```

and writes:

```text
{runs}/p9xe/run_0.json
{runs}/p9xe/run_1.json
...
{slurms}/p9xe/job_0.slurm
...
```

Existing generated directories for the same experiment id are deleted after confirmation.

## Running Locally

Run one generated config:

```bash
python baselines/cli.py --exec-run p9xe/run_0.json
```

Override the log directory for that run:

```bash
python baselines/cli.py \
  --exec-run p9xe/run_0.json \
  --log-path /tmp/morality-gym/debug_logs/run_0
```

Run several generated configs sequentially from one command:

```bash
python baselines/cli.py \
  --exec-multi-run /tmp/morality-gym/runs/p9xe/run_0.json /tmp/morality-gym/runs/p9xe/run_1.json \
  --log-multi-path /tmp/morality-gym/logs/p9xe/run_0 /tmp/morality-gym/logs/p9xe/run_1
```

`--exec-multi-run` requires one `--log-multi-path` entry per run config.

## Running on SLURM

After creating runs, submit generated SLURM jobs:

```bash
python baselines/cli.py --exec-slurms exp_p9xe.json
```

Cluster behavior is controlled by the experiment config:

| Key | Meaning |
| --- | --- |
| `conda_env` | Conda environment activated inside `template.slurm` |
| `partition` | SLURM partition |
| `n_seq_procs` | Number of run configs executed sequentially per process group |
| `n_par_node_procs` | Number of process groups launched on a node |
| `max_nodes` | Upper bound on generated jobs |
| `node_excludes` | Node exclusion string inserted into the template |

Generated jobs stage logs under `/tmp/myrun` on the compute node, zip them, and copy the archives to `{zipped_logs}/{exp_id}`.

## Config Expansion Rules

The experiment config chooses environment/morality-chain pairs in one of two ways:

```json
"env": {
  "base_envs": ["SwitchStandard", "PushStandard"],
  "env_mcs": null
}
```

or:

```json
"env": {
  "base_envs": null,
  "env_mcs": [
    ["SwitchStandard-HumanA-v1", "Utility"]
  ]
}
```

Do not set both `base_envs` and `env_mcs`.

Learner config expansion happens in three stages:

1. load the common default JSON from `configs/learner/`,
2. load the learner-specific default JSON,
3. apply `static_kwargs` and, if present, `multi_kwargs`.

`multi_kwargs` values use dotted keys and create one run per Cartesian-product combination. This path exists in the config loader, but many current configs keep `multi_kwargs` as `null`.

## Output Contract

Every run config is passed to `baselines.benchmarker.run.run`. For `learn_and_eval` experiments, the runner:

1. creates a local `Logger`,
2. instantiates the configured learner,
3. calls `learner.learn(...)`,
4. writes evaluation rows to `metrics/eval.csv`,
5. closes learner and logger resources.

The main file to inspect after a run is:

```text
{logs}/{exp_id}/run_{i}/metrics/eval.csv
```

The final row is the final evaluation. Earlier rows are periodic evaluations during learning.

## Common Failure Points

- `paths.json` points to directories that do not exist or are not writable.
- The experiment config references a learner default file that does not exist in `configs/learner/`.
- `base_envs` contains a name not present in `configs/environment.json`.
- OmniSafe learners require the OmniSafe, PyTorch, and related experiment dependencies to be installed.
- Rendering or GIF generation requires optional image and pygame dependencies.
- On restricted systems, Matplotlib may warn about unwritable font/cache directories. Set `MPLCONFIGDIR` to a writable directory before running if import is slow.
