# Morality Gym Core Components

This directory contains the core classes and modules that form the foundation of the `morality-gym` simulation engine. These components are designed to be reusable across different moral dilemma environments.

## Key Classes and Modules

* **`entity.py`**:
    * `BaseEntity`: The fundamental class for all objects within the simulation world. It defines common properties like position (`pos`), name, group, visual representation functions (`to_asset_fn`, `to_rot_fn`, `to_alpha_fn`), and physical attributes (collidable, movable, interactable).
    * `PlayerEntity`: A specialized `BaseEntity` representing the agent controlled by the learning algorithm or human player.
    * `EntityGroup`: A `BaseEntity` subclass that can contain and manage multiple other entities, useful for complex objects like the railway system. It can have its own scripted behaviour (`act_script`).
    * `LeverEntity`: A specific `LinkedEntity` example representing an interactable lever that can affect the state of other linked entities.

* **`state.py`**:
    * `WorldState`: Holds the complete snapshot of the simulation at any given time. This includes grid dimensions, references to all entities (player, other entities, groups), traversability grids, entity start states, and temporary step data.

* **`action.py`**:
    * `ActionEnum`: An enumeration defining all possible actions an agent can take (e.g., `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `INTERACT`, `MOVE_TO_POS`, `NOOP`).

* **`dynamics.py`**:
    * `BaseDynamics`: Abstract base class for all dynamics modules.
    * `PreStepDynamics` / `PostStepDynamics`: Handle setup and cleanup logic before and after the main dynamics processing within a simulation step. They manage entity state updates, reset temporary flags, and handle entity/group-specific pre/post step methods.
    * `ScriptedDynamics`: Manages entities or entity groups that follow predefined scripts (`act_script`) rather than agent actions.
    * `InteractDynamics`: Handles the `INTERACT` action, determining which nearby interactable entities are affected.
    * `MoveDynamics`: Processes all movement actions, checks for valid moves (bounds, traversability), and updates entity positions. Includes placeholders for collision handling.
    * `ResetDynamics`: Manages resetting the environment state, repositioning entities based on their defined start states and handling potential conflicts.

* **`world.py`**:
    * `World`: The main orchestrator of the simulation. It initializes the `WorldState` based on a `Scenario` and drives the simulation loop via its `step(action)` method, which calls the various dynamics modules in the correct order. Also includes a helper function `interactive` for running the simulation with keyboard controls.

* **`scenario.py`**:
    * `BaseScenario`: Responsible for defining a specific environment setup. It configures the grid, player properties, initial entity states, traversability maps, and the sequence of dynamics modules to be used by the `World`. Specific dilemma scenarios (like those for the trolley problem) inherit from this class.

* **`renderer.py`**:
    * `Renderer`: Handles visualizing the `WorldState` using Pygame. It uses entity properties (`to_asset_fn`, `to_rot_fn`, `to_alpha_fn`, `vis_layer`) and assets loaded from disk to draw the grid, background, and entities. Supports different rendering modes (`human`, `rgb_array`).
    * `EntitySprite`: A Pygame sprite representing a `BaseEntity` for rendering purposes.

* **Other Files**:
    * `custom_types.py`: Defines simple type aliases like `PosType` (Tuple[int, int]).
    * `event.py`: Base class for potential future event handling systems.
    * `env.py`: Placeholder for the Gymnasium `Env` wrapper class.
    * `utils.py`, `vis_fns.py`: Contain helper functions for configuration processing and visualization logic, respectively.
    * `assets/`: Contains default visual assets (images) used by the `Renderer`.

## Core Simulation Loop

The core simulation progresses in discrete steps, primarily managed by the `World.step(action)` method. Each step follows this sequence:

1.  **Set Player Action**: The action chosen by the agent (player) for the current step is recorded in the `PlayerEntity` object.
2.  **Execute Dynamics**: The `World` class calls a sequence of dynamics handlers, typically in this order:
    * **Pre-Step Dynamics (`PreStepDynamics`)**:
        * Resets step-specific data (e.g., which entities moved).
        * Updates lists of currently movable/collidable/actable entities.
        * Stores the position of entities before any changes (`pre_step_pos`).
        * Executes any custom `pre_step()` logic defined in entities or entity groups.
    * **Scripted Actions Dynamics (`ScriptedDynamics`)**:
        * Executes the `act_script()` method for any entities or entity groups marked as `is_scripted=True`. This allows for non-agent-controlled behaviours (like the trolley moving along the track).
    * **Interact Dynamics (`InteractDynamics`)**:
        * Processes `INTERACT` actions.
        * Identifies nearby interactable entities.
        * Calls the `interact()` method on the target entities, allowing for state changes (e.g., flipping a lever, which in turn interacts with a switch).
    * **Move Dynamics (`MoveDynamics`)**:
        * Calculates the intended `next_pos` for all entities based on their chosen actions (including player movement and scripted movements like the trolley's).
        * Validates the move against grid boundaries and traversability maps (`WorldState.traversability_grids`). Invalid moves are cancelled (`NOOP`).
        * *(Collision handling is planned but not yet fully implemented)*.
        * Updates the `pos` of entities that successfully moved.
    * **Post-Step Dynamics (`PostStepDynamics`)**:
        * Updates internal mappings based on entity movements (`WorldState.pos_to_entities`).
        * Resets transient entity states (like `action` and `next_pos`) in preparation for the next step.
        * Executes any custom `post_step()` logic defined in entities or entity groups (e.g., updating a trolley's reference to its current rail).

3.  **Return State**: After the dynamics are processed, the environment typically calculates and returns the next observation, reward, termination status, and info dictionary (this part would be handled by the Gymnasium `Env` wrapper, currently represented by `env.py`).

This cycle repeats for each step the agent takes in the environment.