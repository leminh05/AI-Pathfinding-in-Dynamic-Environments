"""Tests for Greedy Best-First Search. Run with `pytest` or `python -m tests.test_greedy_best_first`."""

import math

from pathfinding.algorithms.dijkstra import dijkstra
from pathfinding.algorithms.greedy_best_first import greedy_best_first


def assert_valid_path(grid, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] == 0


def test_open_grid_is_optimal_and_visits_fewer_cells():
    open_grid = [[0] * 10 for _ in range(10)]
    path, visited, cost = greedy_best_first(open_grid, (0, 0), (9, 9))
    _, dijkstra_visited, dijkstra_cost = dijkstra(open_grid, (0, 0), (9, 9))
    assert_valid_path(open_grid, path, (0, 0), (9, 9))
    assert cost == dijkstra_cost == 18
    assert len(visited) < len(dijkstra_visited)


def test_can_return_a_longer_path_than_dijkstra():
    trap_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ]
    path, _, cost = greedy_best_first(trap_grid, (0, 0), (4, 6))
    _, _, dijkstra_cost = dijkstra(trap_grid, (0, 0), (4, 6))
    assert_valid_path(trap_grid, path, (0, 0), (4, 6))
    assert cost == len(path) - 1 == 14
    assert dijkstra_cost == 10


def test_start_equals_goal():
    assert greedy_best_first([[0]], (0, 0), (0, 0)) == ([(0, 0)], [(0, 0)], 0)


def test_unreachable_goal():
    walled = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    path, visited, cost = greedy_best_first(walled, (0, 0), (2, 2))
    assert path == [] and cost == math.inf
    assert visited == [(0, 0)]


if __name__ == "__main__":
    for test in (test_open_grid_is_optimal_and_visits_fewer_cells,
                 test_can_return_a_longer_path_than_dijkstra,
                 test_start_equals_goal, test_unreachable_goal):
        test()
    print("all GBFS tests passed")
