"""Tests for Q-learning with moving obstacles. Run with `pytest` or `python -m tests.test_qlearning_dynamic`."""

import random

from pathfinding.algorithms.qlearning import DynamicQLearningAgent
from pathfinding.env.obstacles import DynamicObstacle, create_obstacles
from pathfinding.sim.simulator import run_episode


class ScriptedAgent:
    """Always tries the same move; used to test the simulator's rules."""

    def __init__(self, row_change, col_change):
        self.move = (row_change, col_change)
        self.events = []

    def reset(self):
        pass

    def next_cell(self, grid, position, goal, occupied, result):
        return position[0] + self.move[0], position[1] + self.move[1]

    def after_step(self, position, occupied, event):
        self.events.append(event)


def test_danger_bits():
    # bits: 0 = own cell, 1 = up, 2 = down, 3 = left, 4 = right
    assert DynamicQLearningAgent.observe((2, 2), set()) == ((2, 2), 0)
    # obstacle two cells to the right: only the right-hand cell is dangerous
    assert DynamicQLearningAgent.observe((2, 2), {(2, 4)}) == ((2, 2), 0b10000)
    # obstacle directly above: own cell and the cell above are dangerous
    assert DynamicQLearningAgent.observe((2, 2), {(1, 2)}) == ((2, 2), 0b00011)


def test_bumping_into_a_wall_keeps_the_agent_in_place():
    agent = ScriptedAgent(-1, 0)  # always "up" from the top row
    result = run_episode([[0, 0], [0, 0]], (0, 0), (1, 1), [], agent, max_steps=3)
    assert result.bumps == result.steps == 3
    assert result.trajectory == [(0, 0)] * 4
    assert agent.events == ["bump"] * 3 and not result.success


def test_walking_into_an_obstacle_is_a_collision():
    standing = DynamicObstacle((0, 1), route=[(0, 1)])
    agent = ScriptedAgent(0, 1)  # always "right"
    result = run_episode([[0, 0, 0]], (0, 0), (0, 2), [standing], agent)
    assert result.collided and not result.success and result.steps == 1
    assert agent.events == ["collision"]


def test_learns_shortest_route_without_obstacles():
    open_grid = [[0] * 6 for _ in range(6)]
    agent = DynamicQLearningAgent(seed=0)
    agent.train(open_grid, (0, 0), (5, 5), lambda rng: [], episodes=300)
    result = run_episode(open_grid, (0, 0), (5, 5), [], agent)
    assert result.success and result.moves == result.steps == 10


def test_learns_to_get_past_a_standing_obstacle():
    open_grid = [[0] * 5 for _ in range(3)]
    start, goal = (1, 0), (1, 4)

    def make_obstacles(rng):
        return [DynamicObstacle((1, 2), route=[(1, 2)])]  # blocks the straight line

    agent = DynamicQLearningAgent(seed=0)
    agent.train(open_grid, start, goal, make_obstacles, episodes=500)
    result = run_episode(open_grid, start, goal, make_obstacles(None), agent)
    assert result.success and not result.collided
    assert (1, 2) not in result.trajectory


def test_training_reports_one_entry_per_episode():
    open_grid = [[0] * 6 for _ in range(6)]

    def make_obstacles(rng):
        return create_obstacles(open_grid, 2, rng, forbidden={(0, 0), (5, 5)})

    agent = DynamicQLearningAgent(seed=0)
    rewards, successes, collisions = agent.train(open_grid, (0, 0), (5, 5), make_obstacles, 50)
    assert len(rewards) == len(successes) == len(collisions) == 50
    assert not agent.training  # back to greedy mode after training
    assert not any(s and c for s, c in zip(successes, collisions))


if __name__ == "__main__":
    random.seed(0)
    for test in (test_danger_bits, test_bumping_into_a_wall_keeps_the_agent_in_place,
                 test_walking_into_an_obstacle_is_a_collision,
                 test_learns_shortest_route_without_obstacles,
                 test_learns_to_get_past_a_standing_obstacle,
                 test_training_reports_one_entry_per_episode):
        test()
    print("all dynamic Q-learning tests passed")
