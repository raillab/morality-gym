# Morality Gym Examples

This directory contains example scripts that demonstrate how to use the Morality Gym Tabular framework. These examples show different ways to interact with the environments, visualize them, and evaluate agent behavior.

## Example Scripts

### Interactive Environment (`interactive_env.py`)

This script allows you to interact with a moral dilemma environment manually, using keyboard inputs to control the agent. It's useful for understanding how the environments work and what different actions do.

```bash
python examples/interactive_env.py
```

The script creates a trolley problem environment and allows you to explore it interactively. You can modify the environment ID, morality tree ID, and other parameters in the script.

### Visualize All Environments (`vis_all_envs.py`)

This script demonstrates how to visualize all available environments in the framework. It's useful for getting an overview of the different moral dilemmas implemented.

```bash
python examples/vis_all_envs.py
```

### Cost Function Example (`cost_fn_example.py`)

This script demonstrates how to define and use custom cost functions for moral evaluations. Cost functions are used to represent different ethical perspectives or rules.

```bash
python examples/cost_fn_example.py
```

### Evaluation Example (`eval_example.py`)

This script shows how to evaluate agent performance using various metrics, including moral metrics based on different ethical frameworks.

```bash
python examples/eval_example.py
```

### Trolley Interactive (`trolley_interactive.py`)

This is a more detailed interactive script specifically designed for the trolley problem environments, with additional visualization and information display.

```bash
python examples/trolley_interactive.py
```

## Key Concepts Demonstrated

1. **Environment Creation**: How to create Morality Gym environments with different configurations
2. **Interactive Exploration**: How to manually explore environments to understand their dynamics
3. **Morality Trees**: How to use different ethical frameworks (morality trees) for evaluation
4. **Visualization**: How to visualize environments and their states
5. **Custom Parameters**: How to override default parameters for scenarios and environments

## Modifying Examples

These examples are designed to be starting points for your own experiments. You can modify:

- The environment ID to use different moral dilemmas
- The morality tree ID to use different ethical frameworks for evaluation
- The scenario and environment parameters to change the specific setup
- The agent behavior to test different strategies

When modifying examples, refer to the documentation in the `morality_gym/environments` directory for details on available environments and their parameters. 