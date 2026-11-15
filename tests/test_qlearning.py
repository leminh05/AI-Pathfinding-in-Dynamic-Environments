"""Tests for Q-learning. Run with `pytest` or `python -m tests.test_qlearning`."""

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.qlearning import (
    GOAL_REWARD,
    OBSTACLE_REWARD,
    STEP_REWARD,
    QLearningAgent,
)

TRAP_GRID = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
]
START, GOAL = (0, 0), (4, 6)


def test_step_rewards():
    agent = QLearningAgent(TRAP_GRID, START, GOAL, seed=0)
    up, down, left, right = range(4)
    assert agent.step((0, 0), up) == ((0, 0), OBSTACLE_REWARD, False)     # grid edge
    assert agent.step((1, 0), down) == ((1, 0), OBSTACLE_REWARD, False)   # obstacle
    assert agent.step((0, 0), right) == ((0, 1), STEP_REWARD, False)
    assert agent.step((4, 5), right) == (GOAL, GOAL_REWARD, True)


def test_learns_optimal_path():
    agent = QLearningAgent(TRAP_GRID, START, GOAL, seed=0)
    rewards, successes = agent.train(800)
    assert len(rewards) == len(successes) == 800
    assert sum(successes[-100:]) >= 90  # nearly every late episode reaches the goal

    path, reached_goal = agent.learned_path()
    _, _, optimal_cost = astar(TRAP_GRID, START, GOAL)
    assert reached_goal
    assert path[0] == START and path[-1] == GOAL
    assert len(path) - 1 == optimal_cost


def test_untrained_policy_reports_failure_instead_of_looping():
    path, reached_goal = QLearningAgent(TRAP_GRID, START, GOAL, seed=0).learned_path()
    assert not reached_goal and path[0] == START


if __name__ == "__main__":
    for test in (test_step_rewards, test_learns_optimal_path,
                 test_untrained_policy_reports_failure_instead_of_looping):
        test()
    print("all Q-learning tests passed")
