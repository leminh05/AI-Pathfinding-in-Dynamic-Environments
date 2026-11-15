"""
Tabular Q-learning on the same 2D grid used by Dijkstra, Greedy and A*.

Grid format: 0 = free cell, 1 = obstacle, positions are (row, col).

The learning problem
--------------------
State   : the agent's current position (row, col)
Actions : up, down, left, right
Rewards : reach the goal       -> +100 (episode ends)
          hit an obstacle/edge -> -10  (agent stays where it is)
          any other move       -> -1   (pushes the agent to find SHORT paths)

How Q-learning works
--------------------
Unlike the search algorithms, the agent is NOT given the map. It learns by
trial and error. It keeps a table Q[state][action] = "how good is it to take
this action in this state", starting at 0 everywhere.

Each step of an episode:
1. Choose an action with the epsilon-greedy rule: with probability epsilon
   pick a random action (explore), otherwise the best one in the Q-table
   (exploit). Epsilon starts at 1 (fully random) and shrinks every episode.
2. Take the action, observe the reward and the next state.
3. Update the table with the Q-learning rule:

       Q[s, a] += learning_rate * (reward + discount * max_a' Q[s', a'] - Q[s, a])

   i.e. move Q[s, a] a little toward "reward now + best value from the next
   state". Over many episodes, the goal's +100 spreads backwards through the
   table, so every cell learns which direction leads to the goal fastest.

After training, the learned path is found by starting at the start and always
taking the action with the highest Q-value (no more randomness).
"""

import random

import numpy as np

from pathfinding.algorithms.dijkstra import FREE, check_start_and_goal

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]

GOAL_REWARD = 100
OBSTACLE_REWARD = -10
STEP_REWARD = -1


class QLearningAgent:
    def __init__(self, grid, start, goal, learning_rate=0.1, discount=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 max_steps_per_episode=None, seed=None):
        check_start_and_goal(grid, start, goal)
        self.grid = grid
        self.start = start
        self.goal = goal
        self.num_rows = len(grid)
        self.num_cols = len(grid[0])

        self.learning_rate = learning_rate  # alpha: how far each update moves Q
        self.discount = discount            # gamma: how much future reward counts
        self.epsilon = epsilon_start        # current exploration probability
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # Stop an episode that wanders too long, so training always finishes.
        self.max_steps_per_episode = max_steps_per_episode or 4 * self.num_rows * self.num_cols

        self.rng = random.Random(seed)
        # Q-table: one value per (row, col, action), all starting at 0.
        self.q_table = np.zeros((self.num_rows, self.num_cols, len(ACTIONS)))

    def step(self, state, action):
        """Apply one action. Returns (next_state, reward, episode_done)."""
        row_change, col_change = ACTIONS[action]
        next_row, next_col = state[0] + row_change, state[1] + col_change

        inside_grid = 0 <= next_row < self.num_rows and 0 <= next_col < self.num_cols
        if not inside_grid or self.grid[next_row][next_col] != FREE:
            return state, OBSTACLE_REWARD, False  # bumped: stay in place

        next_state = (next_row, next_col)
        if next_state == self.goal:
            return next_state, GOAL_REWARD, True
        return next_state, STEP_REWARD, False

    def best_action(self, state):
        """Action with the highest Q-value (random choice among ties)."""
        q_values = self.q_table[state]
        best_value = q_values.max()
        best_actions = [a for a in range(len(ACTIONS)) if q_values[a] == best_value]
        return self.rng.choice(best_actions)

    def choose_action(self, state):
        """Epsilon-greedy: explore with probability epsilon, else exploit."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        return self.best_action(state)

    def train(self, episodes):
        """
        Run `episodes` training episodes.

        Returns (episode_rewards, episode_successes):
            episode_rewards   : total reward collected in each episode
            episode_successes : True if that episode reached the goal
        """
        episode_rewards = []
        episode_successes = []

        for _ in range(episodes):
            state = self.start
            total_reward = 0
            reached_goal = False

            for _ in range(self.max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)

                # Q-learning update. At the goal there is no future, so the
                # target is just the reward.
                future_value = 0 if done else self.q_table[next_state].max()
                target = reward + self.discount * future_value
                self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

                state = next_state
                total_reward += reward
                if done:
                    reached_goal = True
                    break

            episode_rewards.append(total_reward)
            episode_successes.append(reached_goal)
            # Explore a bit less after every episode.
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return episode_rewards, episode_successes

    def learned_path(self):
        """
        Follow the best action from the start (no exploration).

        Returns (path, reached_goal). If the policy bumps into a wall or walks
        in a loop, it stops there and reached_goal is False.
        """
        path = [self.start]
        seen = {self.start}
        state = self.start
        while state != self.goal:
            state, _, _ = self.step(state, self.best_action(state))
            if state in seen:  # a bump (stayed put) or a loop: the policy is stuck
                return path, False
            path.append(state)
            seen.add(state)
        return path, True


def moving_average(values, window):
    """Average of each `window` consecutive values (smooths noisy curves)."""
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt

    from pathfinding.algorithms.astar import astar

    demo_grid = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    ]
    start, goal = (0, 0), (9, 11)
    episodes = 1500
    window = 50

    agent = QLearningAgent(demo_grid, start, goal, seed=0)
    rewards, successes = agent.train(episodes)
    path, reached_goal = agent.learned_path()
    _, _, optimal_cost = astar(demo_grid, start, goal)

    print(f"Trained for {episodes} episodes")
    print(f"Success rate, all episodes:     {np.mean(successes):.1%}")
    print(f"Success rate, last 100 episodes: {np.mean(successes[-100:]):.1%}")
    if reached_goal:
        print(f"Learned path: {len(path) - 1} moves (A* optimum: {optimal_cost})")
    else:
        print("Learned policy does not reach the goal yet - try more episodes")

    figure, (reward_axis, success_axis, path_axis) = plt.subplots(1, 3, figsize=(16, 4.5))

    episode_numbers = np.arange(1, episodes + 1)
    reward_axis.plot(episode_numbers, rewards, alpha=0.3, label="per episode")
    reward_axis.plot(episode_numbers[window - 1:], moving_average(rewards, window),
                     label=f"{window}-episode average")
    reward_axis.set(title="Reward per episode", xlabel="Episode", ylabel="Total reward")
    reward_axis.legend()

    success_axis.plot(episode_numbers[window - 1:], moving_average(successes, window) * 100)
    success_axis.set(title=f"Success rate ({window}-episode window)", xlabel="Episode",
                     ylabel="Episodes reaching goal (%)", ylim=(0, 105))

    path_axis.imshow(demo_grid, cmap="Greys")
    path_rows, path_cols = zip(*path)
    path_axis.plot(path_cols, path_rows, color="orange", linewidth=3, label="learned path")
    path_axis.plot(start[1], start[0], "o", color="green", markersize=12, label="start")
    path_axis.plot(goal[1], goal[0], "o", color="red", markersize=12, label="goal")
    result = f"{len(path) - 1} moves" if reached_goal else "goal not reached"
    path_axis.set(title=f"Learned path: {result} (optimal {optimal_cost})",
                  xticks=[], yticks=[])
    path_axis.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    os.makedirs("results", exist_ok=True)
    figure.savefig("results/qlearning_training.png", dpi=120)
    print("Saved results/qlearning_training.png")
    plt.show()
