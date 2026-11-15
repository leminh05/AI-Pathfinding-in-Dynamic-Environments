"""Tests for A*. Run with `pytest` or `python -m tests.test_astar`."""

import math
import random

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.dijkstra import dijkstra

TRAP_GRID = [  # GBFS returns cost 14 here, the optimum is 10
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
]


def assert_valid_path(grid, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] == 0


def test_optimal_where_greedy_fails():
    path, visited, cost = astar(TRAP_GRID, (0, 0), (4, 6))
    _, dijkstra_visited, _ = dijkstra(TRAP_GRID, (0, 0), (4, 6))
    assert_valid_path(TRAP_GRID, path, (0, 0), (4, 6))
    assert cost == len(path) - 1 == 10
    assert len(visited) < len(dijkstra_visited)


def test_same_cost_as_dijkstra_on_random_grids():
    rng = random.Random(0)
    for _ in range(300):
        grid = [[1 if rng.random() < 0.3 else 0 for _ in range(8)] for _ in range(8)]
        grid[0][0] = grid[7][7] = 0
        path, _, cost = astar(grid, (0, 0), (7, 7))
        _, _, dijkstra_cost = dijkstra(grid, (0, 0), (7, 7))
        assert cost == dijkstra_cost
        if path:
            assert_valid_path(grid, path, (0, 0), (7, 7))


def test_start_equals_goal():
    assert astar([[0]], (0, 0), (0, 0)) == ([(0, 0)], [(0, 0)], 0)


def test_unreachable_goal():
    walled = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    path, visited, cost = astar(walled, (0, 0), (2, 2))
    assert path == [] and cost == math.inf
    assert visited == [(0, 0)]


if __name__ == "__main__":
    for test in (test_optimal_where_greedy_fails, test_same_cost_as_dijkstra_on_random_grids,
                 test_start_equals_goal, test_unreachable_goal):
        test()
    print("all A* tests passed")
