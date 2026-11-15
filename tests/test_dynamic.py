"""Tests for moving obstacles and the simulator. Run with `pytest` or `python -m tests.test_dynamic`."""

import random

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.dijkstra import dijkstra
from pathfinding.algorithms.greedy_best_first import greedy_best_first
from pathfinding.env.maps import random_grid
from pathfinding.env.obstacles import DynamicObstacle, create_obstacles, move_obstacles
from pathfinding.sim.simulator import run_episode

START, GOAL = (0, 0), (14, 14)


def make_world(seed, count, mode="random_walk"):
    rng = random.Random(seed)
    grid = random_grid(15, 15, 0.2, rng, START, GOAL)
    return grid, create_obstacles(grid, count, rng, mode, forbidden={START, GOAL})


def test_no_obstacles_matches_static_search():
    grid, obstacles = make_world(seed=1, count=0)
    result = run_episode(grid, START, GOAL, obstacles, astar)
    _, _, optimal_cost = astar(grid, START, GOAL)
    assert result.success and not result.collided
    assert result.moves == result.steps == optimal_cost
    assert result.replans == 0 and result.waits == 0


def test_agent_plans_around_a_standing_obstacle():
    open_grid = [[0] * 5 for _ in range(5)]
    standing = DynamicObstacle((0, 2), route=[(0, 2)])
    result = run_episode(open_grid, (0, 0), (0, 4), [standing], astar)
    assert result.success
    # it keeps one cell away from the obstacle: (0,1), (0,3), (1,2) are avoided too
    assert not {(0, 1), (0, 2), (0, 3), (1, 2)} & set(result.trajectory)
    assert result.moves == 8  # down 2 rows, across, back up


def test_obstacles_follow_the_rules():
    for mode in ("random_walk", "patrol"):
        grid, obstacles = make_world(seed=2, count=25, mode=mode)
        rng = random.Random(0)
        for _ in range(300):
            move_obstacles(obstacles, grid, GOAL, rng)
            cells = [obstacle.position for obstacle in obstacles]
            assert len(set(cells)) == len(cells)          # never overlap
            assert GOAL not in cells                      # never enter the goal
            assert all(grid[r][c] == 0 for r, c in cells)  # never enter walls


def test_patrol_goes_back_and_forth():
    open_grid = [[0] * 5 for _ in range(5)]
    guard = DynamicObstacle((2, 0), route=[(2, 0), (2, 1), (2, 2)])
    rng = random.Random(0)
    positions = []
    for _ in range(6):
        move_obstacles([guard], open_grid, (4, 4), rng)
        positions.append(guard.position)
    assert positions == [(2, 1), (2, 2), (2, 1), (2, 0), (2, 1), (2, 2)]


def test_same_seed_gives_every_algorithm_the_same_obstacle_moves():
    grid, obstacles = make_world(seed=3, count=15)
    histories = [run_episode(grid, START, GOAL, obstacles, search, seed=7).obstacle_history
                 for search in (dijkstra, greedy_best_first, astar)]
    shortest = min(len(h) for h in histories)
    assert histories[0][:shortest] == histories[1][:shortest] == histories[2][:shortest]
    assert obstacles[0].position == histories[0][0][0]  # caller's obstacles were not moved


def test_episodes_are_consistent():
    for seed in range(40):
        grid, obstacles = make_world(seed, count=20, mode="random_walk" if seed % 2 else "patrol")
        for search in (dijkstra, greedy_best_first, astar):
            result = run_episode(grid, START, GOAL, obstacles, search, seed=seed)
            assert not (result.success and result.collided)
            assert result.bumps == 0  # planners never walk into walls
            assert result.steps == result.moves + result.waits == len(result.trajectory) - 1
            for (r1, c1), (r2, c2) in zip(result.trajectory, result.trajectory[1:]):
                assert abs(r1 - r2) + abs(c1 - c2) <= 1  # one move or a wait
                assert grid[r2][c2] == 0
            if result.success:
                assert result.trajectory[-1] == GOAL


if __name__ == "__main__":
    for test in (test_no_obstacles_matches_static_search,
                 test_agent_plans_around_a_standing_obstacle,
                 test_obstacles_follow_the_rules, test_patrol_goes_back_and_forth,
                 test_same_seed_gives_every_algorithm_the_same_obstacle_moves,
                 test_episodes_are_consistent):
        test()
    print("all dynamic-environment tests passed")
