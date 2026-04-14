# Morality Gym Tabular

A research framework for investigating AI agent behavior in moral dilemmas, providing tabular environments with the Gymnasium API for training and evaluating reinforcement learning agents on scenarios involving ethical decisions.

## Overview

AI systems are increasingly deployed in complex real-world scenarios where they face moral dilemmas. Morality Gym provides standardized environments for:

1. Testing how agents behave when facing moral decisions
2. Developing algorithms that align with human values and ethical principles 
3. Evaluating algorithmic solutions to moral dilemmas

## Key Features

- **Gymnasium-compatible environments:** Build on the familiar Gymnasium API for easy integration with existing RL libraries
- **Tabular moral dilemmas:** Classic trolley problem thought experiments implemented as reinforcement learning environments
- **Modular design:** Core simulation engine with specialized modules for different trolley problem variants
- **Evaluation framework:** Tools to assess agent behavior against different ethical frameworks
- **Research tools:** Comprehensive experiment tracking and analysis utilities

## Available Environments

The framework currently implements various formulations of the trolley problem:

- **Switch Variants:** Classic trolley problem scenarios where the agent can pull a lever to divert a trolley
- **Push Variants:** Scenarios where the agent can push a person to stop the trolley
- **Combination Variants:** Scenarios offering multiple intervention options with different moral implications

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/SimonRosen173/morality-gym-tabular.git
cd morality-gym-tabular

# Install the package
pip install -e .
```

### Basic Usage

```python
import gymnasium as gym
import morality_gym.setup.setup

# Create a morality environment
env = morality_gym.setup.setup.make("MoralityGym/Trolley-SwitchStandard-v0")

# Reset the environment
obs, info = env.reset()

# Interact with the environment
for _ in range(100):
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Environments](morality_gym/environments/README.md) | Learn about the available environments |
| [Experiments](experiments/README.md) | Understand how to run experiments with different agents |
| [Examples](examples/) | Explore example scripts demonstrating various features |

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{rosen2024moralitygym,
  author = {Rosen, Simon and Singh, Siddarth and Robertson, Helen Sarah and Gelo, Ebenezer and 
            Suder, Ibrahim and Nangue Tasse, Geraud and Williams, Victoria and 
            James, Steven and Rosman, Benjamin},
  title = {Morality Gym Tabular: A Framework for Moral Dilemmas in Reinforcement Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/SimonRosen173/morality-gym-tabular}}
}
```

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository. 

