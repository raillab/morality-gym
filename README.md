# MoralityGym

MoralityGym provides Gymnasium-compatible trolley-problem environments for studying how reinforcement-learning agents behave under moral constraints.

The repository has two layers:

- `morality_gym/`: the environment package, morality chains, cost function utilities, and lightweight wrappers.
- `baselines/` and `omnisafe/`: experiment code used for the paper benchmarks. These are intended for repository users, not as the minimal pip-facing API.

## Installation

For development from this repository:

```bash
git clone https://github.com/raillab/morality-gym.git
cd morality-gym
pip install -e .
```

The lightweight environment path depends on `numpy`, `matplotlib`, and `gymnasium`. The experiment stack also uses additional packages such as `torch`, `stable-baselines3`, `omnisafe`, `pandas`, `tqdm`, and cluster tooling where relevant.

## Basic Usage

```python
from morality_gym import make

env, morality_chain = make(
    env_id="SwitchStandard-HumanA-v1",
    morality_chain_id="Utility",
)

obs, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Environment IDs use:

```text
{scenario_id}-{variant_id}-v1
```

Examples include `SwitchStandard-HumanA-v1`, `PushStandard-HumanA-v1`, `Switch5-Human-v1`, and `PushOrSwitch-Human-v1`.

## Safe-RL Example

The standalone example in [examples/train_safe_rl_agent.py](examples/train_safe_rl_agent.py) shows how to:

1. create an environment and morality chain,
2. wrap the environment so `info["cost"]` is emitted at each step,
3. train a small constrained Q-learning agent with a Lagrangian cost penalty.

Run a short smoke test:

```bash
python examples/train_safe_rl_agent.py \
  --episodes 10 \
  --eval-episodes 3 \
  --max-episode-steps 25 \
  --log-every 5
```

Run a longer example:

```bash
python examples/train_safe_rl_agent.py --episodes 500 --eval-episodes 50
```

This script is a readable demonstration of the cost-wrapper pattern. It is not the paper benchmark implementation.

## Experiments

Paper-style experiments are configured and run through the benchmarker under [baselines/](baselines/README.md). In short:

```bash
python baselines/cli.py --create exp_p9xe.json
python baselines/cli.py --exec-run p9xe/run_0.json
```

See [baselines/README.md](baselines/README.md) and [baselines/benchmarker/README.md](baselines/benchmarker/README.md) for the experiment configuration format, local execution, SLURM execution, logs, and result consolidation notes.

## Documentation

| Resource | Description |
| --- | --- |
| [morality_gym/README.md](morality_gym/README.md) | Supported environment ID syntax and scenarios |
| [examples/](examples/) | Small scripts for interacting with and evaluating environments |
| [baselines/README.md](baselines/README.md) | Baseline learner and benchmarker overview |
| [baselines/benchmarker/README.md](baselines/benchmarker/README.md) | Experiment configuration and execution details |

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{rosen2026MoralityGym,
    author = {Rosen, Simon and Singh, Siddarth and Gelo, Ebenezer and Robertson, Helen Sarah and Suder, Ibrahim and Williams, Victoria and Rosman, Benjamin and Tasse, Geraud Nangue and James, Steven},
    title = {MoralityGym: A Benchmark for Evaluating Hierarchical Moral Alignment in Sequential Decision-Making Agents},
    year = {2026},
    isbn = {9798400723179},
    publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
    address = {Richland, SC},
    url = {https://doi.org/10.65109/SAKL6648},
    doi = {10.65109/SAKL6648},
    abstract = {Evaluating moral alignment in agents navigating conflicting, hierarchically structured human norms is a critical challenge at the intersection of AI safety, moral philosophy, and cognitive science. We introduce Morality Chains, a novel formalism for representing moral norms as ordered deontic constraints, and MoralityGym, a benchmark of 98 ethical-dilemma problems presented as trolley-dilemma-style Gymnasium environments. By decoupling task-solving from moral evaluation and introducing a novel morality metric, MoralityGym allows the integration of insights from psychology and philosophy into the evaluation of norm-sensitive reasoning. Baseline results with Safe RL methods reveal key limitations, underscoring the need for more principled approaches to ethical decision-making. This work provides a foundation for developing AI systems that behave more reliably, transparently, and ethically in complex real-world contexts.},
    booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems},
    pages = {1464–1472},
    numpages = {9},
    keywords = {reinforcement learning, safe rl, moral rl, safety, alignment, benchmark},
    location = {Paphos, Cyprus},
    series = {AAMAS '26}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgement
Sprite assets created by Alexandra van Meelis.

## License

This project is licensed under the terms of [LICENSE](LICENSE).
