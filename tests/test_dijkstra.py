"""Tests for Dijkstra. Run with `pytest` or `python -m tests.test_dijkstra`."""

import math

from pathfinding.algorithms.dijkstra import dijkstra

GRID = [
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
]


def test_finds_shortest_path():
    path, visited, cost = dijkstra(GRID, (0, 0), (4, 2))
    assert cost == 10  # only route: along the top, down the right, back left
    assert path[0] == (0, 0) and path[-1] == (4, 2)
    assert len(path) == cost + 1
    # every step moves exactly one cell and never onto an obstacle
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert GRID[r2][c2] == 0
    assert visited[0] == (0, 0) and visited[-1] == (4, 2)


def test_start_equals_goal():
    assert dijkstra(GRID, (0, 0), (0, 0)) == ([(0, 0)], [(0, 0)], 0)


def test_unreachable_goal():
    walled = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    path, visited, cost = dijkstra(walled, (0, 0), (2, 2))
    assert path == [] and cost == math.inf
    assert visited == [(0, 0)]


def test_rejects_start_on_obstacle():
    try:
        dijkstra(GRID, (1, 0), (0, 0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for test in (test_finds_shortest_path, test_start_equals_goal,
                 test_unreachable_goal, test_rejects_start_on_obstacle):
        test()
    print("all Dijkstra tests passed")
