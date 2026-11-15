"""Tests for the experiment runner. Run with `pytest` or `python -m tests.test_experiments`."""

import csv
import os
import tempfile

from pathfinding.experiments import run


def test_task_list():
    tasks = run.all_tasks(num_seeds=2)
    combos_per_map = 1 + (len(run.OBSTACLE_COUNTS) - 1) * len(run.MODES)  # 0 obstacles runs once
    assert len(tasks) == len(run.GRID_SIZES) * len(run.WALL_DENSITIES) * combos_per_map * 2
    assert len(set(tasks)) == len(tasks)
    assert all(mode == "none" for _, _, n, mode, _ in tasks if n == 0)


def test_run_task_rows():
    rows = run.run_task((8, 0.1, 3, "random_walk", 0), training_episodes=50)
    assert [row["algorithm"] for row in rows] == ["Dijkstra", "Greedy", "A*", "Q-learning"]
    for row in rows:
        assert set(row) == set(run.FIELDS)
        assert row["success"] + row["collided"] + row["timeout"] == 1
        assert row["optimal_length"] == rows[0]["optimal_length"]  # same map for everyone
    assert rows[-1]["train_episodes"] == 50 and rows[0]["train_time_s"] == ""


def test_same_map_for_every_obstacle_count():
    without = run.run_task((8, 0.1, 0, "none", 1), training_episodes=10)
    with_obstacles = run.run_task((8, 0.1, 5, "patrol", 1), training_episodes=10)
    assert without[0]["optimal_length"] == with_obstacles[0]["optimal_length"]
    assert without[2]["success"] == 1  # A* always succeeds with no moving obstacles


def test_completed_keys_skips_only_finished_tasks():
    task_done, task_partial = (8, 0.1, 0, "none", 0), (8, 0.1, 0, "none", 1)
    rows = run.run_task(task_done, 10) + run.run_task(task_partial, 10)[:2]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "results.csv")
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=run.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        assert run.completed_keys(path) == {task_done}
    assert run.completed_keys("does/not/exist.csv") == set()


if __name__ == "__main__":
    for test in (test_task_list, test_run_task_rows, test_same_map_for_every_obstacle_count,
                 test_completed_keys_skips_only_finished_tasks):
        test()
    print("all experiment-runner tests passed")
