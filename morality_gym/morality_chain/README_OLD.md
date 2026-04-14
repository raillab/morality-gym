# Morality Trees

The Morality Tree module provides a framework for evaluating agent behavior against different ethical theories and moral frameworks. This allows researchers to assess how agents perform not just in terms of reward maximization but also in terms of alignment with various moral perspectives.

## Overview

A morality tree is a structured representation of an ethical theory that can be used to evaluate agent behavior. Each tree:

1. Defines a set of moral norms or principles
2. Specifies how these norms are evaluated in the context of environment states and agent actions
3. Provides metrics for how well agent behavior aligns with the defined moral framework

## Supported Ethical Frameworks

The package currently implements several ethical frameworks, primarily for the trolley problem domain:

### Utilitarian Frameworks

Utilitarian ethics evaluate actions based on their consequences, particularly in terms of maximizing overall wellbeing.

#### Utility-Based

1. `Trolley-Common-Utilitarian-UtilityHarm-v0`: Basic utility calculation based on harm minimization
2. `Trolley-Common-Utilitarian-OrderedUtilityHarm-v0`: Utility calculation that considers the order of actions
3. `Trolley-Common-Utilitarian-WeightedUtilityHarm-v0`: Utility calculation with weighted harm values for different entities

#### Outcome-Based

4. `Trolley-Common-StandardUtilitarian-v0`: Standard utilitarian approach focusing on outcome optimization

#### Hybrid (Outcome & Utility)

5. `Trolley-Common-Utilitarian-OutcomeUtility-v0`: Combined framework considering both outcomes and utility calculations

### Deontological Frameworks

Deontological ethics focus on the inherent rightness or wrongness of actions themselves, rather than their consequences.

1. `Trolley-Common-Deontological-DoNoHarm-v0`: Focuses on the principle of avoiding direct harm
2. `Trolley-Common-Deontological-DoNotKill-v0`: Specifically evaluates avoidance of actions that directly cause death

### Virtue Ethics

Virtue ethics focus on the character and moral character of the agent making decisions.

1. `Trolley-Common-VirtueEthics-Courage-v0`: Evaluates decisions based on the virtue of courage
2. `Trolley-Common-VirtueEthics-Compassion-v0`: Evaluates decisions based on the virtue of compassion

## Usage

To use a morality tree in your experiments:

```python
from morality_gym.setup.setup import make

# Create environment with specific morality tree
env, mt = make(
    env_id="MoralityGym/Trolley-SwitchStandard-v0",
    morality_chain_id="Trolley-Common-Utilitarian-UtilityHarm-v0"
)

# Use morality tree for evaluation
moral_evaluation = mt.evaluate(env.get_state())
```

## Creating Custom Morality Trees

Researchers can define their own morality trees by:

1. Subclassing the `MoralityChain` class
2. Implementing the required evaluation methods
3. Registering the new tree with the framework

See `morality_tree.py` for implementation details and `setup.py` for registration examples.