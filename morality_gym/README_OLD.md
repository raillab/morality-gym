# Morality Gym Package
Note: Rough draft of readme and some readme components may be out of date

`morality-gym` is a Python package designed for research into agent behaviour in sequential moral dilemmas. It provides a suite of simulation environments compatible with the Gymnasium API.

## Goal

The primary goal of this project is to offer standardised environments for developing and evaluating agents, particularly reinforcement learning agents, on their ability to navigate complex scenarios involving moral choices and ethical considerations.

## Core Components

The package is structured into several key components:

### 1. Environments (`morality_gym.environments`)

This is the core simulation engine and collection of dilemma scenarios.

* **Core Engine (`environments.core`)**: Provides the fundamental building blocks for all environments. This includes base classes for entities (`BaseEntity`, `PlayerEntity`), world state representation (`WorldState`), simulation dynamics (`BaseDynamics`, `MoveDynamics`, `InteractDynamics`), scenario definition (`BaseScenario`), rendering (`Renderer`), and more.
* **Dilemma Implementations (e.g., `environments.trolley`)**: Contains specific implementations of moral dilemmas built upon the core engine. Currently, this includes variations of the trolley problem. Future plans include adding scenarios for healthcare, child/elder care, and AI alignment challenges.

See the `environments/README.md` for more details on the environment structure.

### 2. TODO

## Getting Started

(TODO: Add installation and basic usage instructions)