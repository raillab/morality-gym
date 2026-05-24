# Morality Gym Package

This package contains the environment constructors, trolley-problem scenarios, morality chains, cost utilities, and lightweight wrappers.

## Creating an Environment

```python
from morality_gym import make

env, morality_chain = make(
    env_id="SwitchStandard-HumanA-v1",
    morality_chain_id="Utility",
)
```

`make` returns:

- `env`: a Gymnasium-compatible environment,
- `morality_chain`: the morality-chain object used for morality metrics and cost construction.

## Environment ID Syntax

```text
{scenario_id}-{variant_id}-v1
```

`MoralityGym/` and `Trolley-` prefixes are accepted by the constructor for compatibility, but the shorter form above is preferred in examples and configs.

## Supported Scenarios

The supported scenario ids are:

- `SwitchStandard`
- `PushStandard`
- `PushOrSwitch`
- `Push2OrSwitch`
- `SwitchSelfSacrifice`
- `PushSelfSacrifice`
- `Switch5`
- `Switch7`
- `Switch2Trolley4Track`
- `Push3SelfSacrifice`
- `PushOrSwitchSelfSacrifice`

The exact variant names are listed in `morality_gym/_environments/trolley/supported_envs.py`.

Common examples:

```text
SwitchStandard-HumanA-v1
SwitchStandard-HumanAnimalA-v1
PushStandard-HumanA-v1
PushOrSwitch-Human-v1
Switch5-Human-v1
SwitchSelfSacrifice-HumanA-v1
```

## Observation Format

Most training examples use flat NumPy observations:

```python
env, morality_chain = make(
    env_id="SwitchStandard-HumanA-v1",
    morality_chain_id="Utility",
    env_kwargs={
        "env_overrides": {
            "obs_type": "np.ndarray",
            "is_normalise_obs": True,
        }
    },
)
```

Dictionary observations are useful for debugging:

```python
env, morality_chain = make(
    env_id="SwitchStandard-HumanA-v1",
    morality_chain_id="Utility",
    env_kwargs={
        "env_overrides": {
            "obs_type": "dict",
        }
    },
)
```

## Costs

The cost function consumes the `info["norm_events"]` emitted by the environment and scalarises norm violations into a numeric cost. See `examples/train_safe_rl_agent.py` for a standalone wrapper that adds `info["cost"]` during training.
