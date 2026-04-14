# MoralityGym

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
