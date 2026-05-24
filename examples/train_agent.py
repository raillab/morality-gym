"""Train a minimal constrained RL agent on Morality Gym.

This example is intentionally small and dependency-light. It shows the core
pattern needed by safe-RL algorithms:

1. create a Morality Gym environment and morality chain,
2. wrap the environment so each step exposes `info["cost"]`,
3. train a policy against reward while penalising expected cost.

The learner below is tabular Lagrangian Q-learning. It is useful as a readable
example, not as a benchmark-quality implementation.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import gymnasium as gym
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from morality_gym import make
from morality_gym._cost.cost import Cost


class CostInfoWrapper(gym.Wrapper):
    """Adds `cost` and `orig_reward` entries to the info dict."""

    def __init__(
        self,
        env: gym.Env,
        morality_chain,
        *,
        cost_scale: float = 1.0,
        information_mode: str = "minimal",
        scalarisation: str = "expert",
        shape_reward: bool = False,
    ) -> None:
        super().__init__(env)
        self.cost_fn = Cost(
            morality_chain,
            scale_fact=cost_scale,
            information_mode=information_mode,
            scalarisation=scalarisation,
        )
        self.shape_reward = shape_reward

    def reset(self, *args, **kwargs):
        self.cost_fn.reset()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = float(self.cost_fn(info))
        info["cost"] = cost
        info["orig_reward"] = reward
        if self.shape_reward:
            reward = reward - cost
        return obs, reward, terminated, truncated, info


@dataclass
class TrainStats:
    avg_return: float
    avg_cost: float
    avg_length: float
    penalty: float


class LagrangianQLearner:
    """Small constrained Q-learning agent for discrete actions."""

    def __init__(
        self,
        n_actions: int,
        *,
        alpha: float,
        gamma: float,
        lambda_lr: float,
        cost_limit: float,
        seed: int,
    ) -> None:
        self.q_values: DefaultDict[tuple[float, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float64)
        )
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_lr = lambda_lr
        self.cost_limit = cost_limit
        self.cost_penalty = 0.0
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def state_key(obs: np.ndarray, decimals: int = 3) -> tuple[float, ...]:
        return tuple(np.asarray(obs, dtype=np.float64).round(decimals).tolist())

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.q_values[self.state_key(obs)])))
        return int(np.argmax(self.q_values[self.state_key(obs)]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        cost: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        state = self.state_key(obs)
        next_state = self.state_key(next_obs)
        penalised_reward = reward - self.cost_penalty * cost
        bootstrap = 0.0 if done else self.gamma * np.max(self.q_values[next_state])
        target = penalised_reward + bootstrap
        self.q_values[state][action] += self.alpha * (target - self.q_values[state][action])

    def update_penalty(self, episode_cost: float) -> None:
        self.cost_penalty = max(
            0.0,
            self.cost_penalty + self.lambda_lr * (episode_cost - self.cost_limit),
        )


def build_env(args: argparse.Namespace) -> gym.Env:
    env, morality_chain = make(
        env_id=args.env_id,
        morality_chain_id=args.morality_chain,
        env_kwargs={
            "scenario_overrides": {
                "max_timesteps": args.max_episode_steps,
            },
            "env_overrides": {
                "obs_type": "np.ndarray",
                "render_mode": None,
            },
        },
        morality_chain_kwargs={
            "beta": args.beta,
        },
    )
    return CostInfoWrapper(
        env,
        morality_chain,
        cost_scale=args.cost_scale,
        information_mode=args.information_mode,
        scalarisation=args.scalarisation,
        shape_reward=False,
    )


def linear_epsilon(start: float, end: float, episode: int, total_episodes: int) -> float:
    frac = min(1.0, episode / max(1, total_episodes - 1))
    return start + frac * (end - start)


def train(env: gym.Env, agent: LagrangianQLearner, args: argparse.Namespace) -> TrainStats:
    returns: list[float] = []
    costs: list[float] = []
    lengths: list[int] = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        epsilon = linear_epsilon(args.epsilon_start, args.epsilon_end, episode, args.episodes)

        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cost = float(info["cost"])

            agent.update(obs, action, reward, cost, next_obs, done)

            obs = next_obs
            episode_return += float(reward)
            episode_cost += cost
            episode_length += 1

        agent.update_penalty(episode_cost)
        returns.append(episode_return)
        costs.append(episode_cost)
        lengths.append(episode_length)

        if (episode + 1) % args.log_every == 0:
            window = slice(max(0, episode + 1 - args.log_every), episode + 1)
            print(
                f"episode={episode + 1:04d} "
                f"avg_return={np.mean(returns[window]):.3f} "
                f"avg_cost={np.mean(costs[window]):.3f} "
                f"penalty={agent.cost_penalty:.3f} "
                f"epsilon={epsilon:.3f}"
            )

    return TrainStats(
        avg_return=float(np.mean(returns)),
        avg_cost=float(np.mean(costs)),
        avg_length=float(np.mean(lengths)),
        penalty=agent.cost_penalty,
    )


def evaluate(env: gym.Env, agent: LagrangianQLearner, args: argparse.Namespace) -> TrainStats:
    returns: list[float] = []
    costs: list[float] = []
    lengths: list[int] = []

    for episode in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + 10_000 + episode)
        done = False
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0

        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            episode_cost += float(info["cost"])
            episode_length += 1

        returns.append(episode_return)
        costs.append(episode_cost)
        lengths.append(episode_length)

    return TrainStats(
        avg_return=float(np.mean(returns)),
        avg_cost=float(np.mean(costs)),
        avg_length=float(np.mean(lengths)),
        penalty=agent.cost_penalty,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SwitchStandard-HumanA-v1")
    parser.add_argument("--morality-chain", default="Utility")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-lr", type=float, default=0.01)
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--cost-scale", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--information-mode", choices=["minimal", "partial", "full"], default="minimal")
    parser.add_argument("--scalarisation", choices=["expert", "linear"], default="expert")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_env(args)
    agent = LagrangianQLearner(
        env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        lambda_lr=args.lambda_lr,
        cost_limit=args.cost_limit,
        seed=args.seed,
    )

    train_stats = train(env, agent, args)
    eval_stats = evaluate(env, agent, args)
    env.close()

    print("\nTraining summary")
    print(f"avg_return={train_stats.avg_return:.3f}")
    print(f"avg_cost={train_stats.avg_cost:.3f}")
    print(f"avg_length={train_stats.avg_length:.3f}")
    print(f"penalty={train_stats.penalty:.3f}")

    print("\nEvaluation summary")
    print(f"avg_return={eval_stats.avg_return:.3f}")
    print(f"avg_cost={eval_stats.avg_cost:.3f}")
    print(f"avg_length={eval_stats.avg_length:.3f}")
    print(f"penalty={eval_stats.penalty:.3f}")


if __name__ == "__main__":
    main()
