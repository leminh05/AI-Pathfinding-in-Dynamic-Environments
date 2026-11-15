"""
Experiment runner: compares Dijkstra, Greedy, A* and Q-learning on many maps.

Run from the project root:
    python -m pathfinding.experiments.run                 # full experiment
    python -m pathfinding.experiments.run --seeds 2 --episodes 300   # quick test

Design (one "task" = one map):
    - Factors: grid size x wall density x number of moving obstacles x movement
      mode (see the constants below). With 0 moving obstacles the mode does not
      matter, so it is run once with mode "none".
    - Each factor combination is repeated for `--seeds` seeds.
    - The map depends only on (size, density, seed), so the same map is reused
      for every obstacle count and mode: comparisons are paired.
    - All four algorithms face the same map, the same obstacles and the same
      obstacle movement. Q-learning is first trained on that map with OTHER
      obstacle layouts and movements, then evaluated greedily (no exploration).

Output: one CSV row per (task, algorithm) in results/results.csv. Rows are
appended as soon as a task finishes, and tasks already in the file are skipped,
so an interrupted run can simply be started again to continue.
"""

import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.dijkstra import dijkstra
from pathfinding.algorithms.greedy_best_first import greedy_best_first
from pathfinding.algorithms.qlearning import DynamicQLearningAgent
from pathfinding.env.maps import random_grid
from pathfinding.env.obstacles import create_obstacles
from pathfinding.sim.simulator import run_episode

GRID_SIZES = [15, 20]
WALL_DENSITIES = [0.1, 0.2]
OBSTACLE_COUNTS = [0, 5, 10, 20]
MODES = ["random_walk", "patrol"]

PLANNERS = [("Dijkstra", dijkstra), ("Greedy", greedy_best_first), ("A*", astar)]

TASK_FIELDS = ["size", "density", "num_obstacles", "mode", "seed"]
FIELDS = TASK_FIELDS + [
    "algorithm", "success", "collided", "timeout", "steps", "moves", "waits", "bumps",
    "replans", "nodes_expanded", "planning_time_ms", "optimal_length", "path_ratio",
    "train_time_s", "train_episodes",
]


def all_tasks(num_seeds):
    """Every (size, density, num_obstacles, mode, seed) combination to run."""
    tasks = []
    for size in GRID_SIZES:
        for density in WALL_DENSITIES:
            for num_obstacles in OBSTACLE_COUNTS:
                for mode in (["none"] if num_obstacles == 0 else MODES):
                    for seed in range(num_seeds):
                        tasks.append((size, density, num_obstacles, mode, seed))
    return tasks


def task_key(row):
    """Identify a task from a CSV row (all values come back as strings)."""
    return (int(row["size"]), float(row["density"]), int(row["num_obstacles"]),
            row["mode"], int(row["seed"]))


def run_task(task, training_episodes):
    """Run all four algorithms on one map. Returns a list of CSV rows (dicts)."""
    size, density, num_obstacles, mode, seed = task
    start, goal = (0, 0), (size - 1, size - 1)

    # String seeds keep maps and obstacles independent of each other and stable
    # between runs.
    grid = random_grid(size, size, density, random.Random(f"map-{size}-{density}-{seed}"),
                       start, goal)
    _, _, optimal_length = astar(grid, start, goal)

    def make_obstacles(rng):
        if num_obstacles == 0:
            return []
        return create_obstacles(grid, num_obstacles, rng, mode, forbidden={start, goal})

    obstacles = make_obstacles(random.Random(f"obstacles-{task}"))

    rows = []

    def add_row(name, result, train_time_s="", train_episodes=""):
        timeout = not result.success and not result.collided
        rows.append({
            "size": size, "density": density, "num_obstacles": num_obstacles,
            "mode": mode, "seed": seed, "algorithm": name,
            "success": int(result.success), "collided": int(result.collided),
            "timeout": int(timeout), "steps": result.steps, "moves": result.moves,
            "waits": result.waits, "bumps": result.bumps, "replans": result.replans,
            "nodes_expanded": result.nodes_expanded,
            "planning_time_ms": round(result.planning_time_ms, 4),
            "optimal_length": optimal_length,
            "path_ratio": round(result.moves / optimal_length, 4) if result.success else "",
            "train_time_s": train_time_s, "train_episodes": train_episodes,
        })

    for name, search in PLANNERS:
        add_row(name, run_episode(grid, start, goal, obstacles, search, seed=seed))

    agent = DynamicQLearningAgent(seed=seed)
    training_start = time.perf_counter()
    # Training seeds start far above the evaluation seeds so they never overlap.
    agent.train(grid, start, goal, make_obstacles, training_episodes,
                first_seed=1_000_000 + seed * training_episodes)
    train_time_s = round(time.perf_counter() - training_start, 3)
    add_row("Q-learning", run_episode(grid, start, goal, obstacles, agent, seed=seed),
            train_time_s, training_episodes)
    return rows


def completed_keys(path):
    """Tasks that already have all four algorithm rows in the CSV file."""
    if not os.path.exists(path):
        return set()
    rows_per_task = {}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            key = task_key(row)
            rows_per_task[key] = rows_per_task.get(key, 0) + 1
    return {key for key, count in rows_per_task.items() if count == len(PLANNERS) + 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seeds", type=int, default=30, help="maps per factor combination")
    parser.add_argument("--episodes", type=int, default=3000, help="Q-learning training episodes per map")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()

    done = completed_keys(args.out)
    tasks = [task for task in all_tasks(args.seeds) if task not in done]
    print(f"{len(done)} tasks already done, {len(tasks)} to run with {args.workers} workers")
    if not tasks:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    started = time.perf_counter()
    with open(args.out, "a", newline="") as file, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        futures = {pool.submit(run_task, task, args.episodes): task for task in tasks}
        for finished, future in enumerate(as_completed(futures), start=1):
            writer.writerows(future.result())
            file.flush()  # keep finished work even if the run is interrupted
            elapsed = time.perf_counter() - started
            remaining = elapsed / finished * (len(tasks) - finished)
            print(f"[{finished}/{len(tasks)}] {futures[future]} "
                  f"elapsed {elapsed / 60:.1f} min, ~{remaining / 60:.1f} min left", flush=True)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
