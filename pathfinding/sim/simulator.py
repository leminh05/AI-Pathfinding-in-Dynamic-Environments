"""
Step-by-step simulation of an agent moving among dynamic obstacles.

The search algorithms (Dijkstra, Greedy, A*) only know how to plan on a fixed
grid. To use them in a changing world, the agent follows a "plan, move,
replan when blocked" loop:

Every time step:
1. The agent sees where every moving obstacle is RIGHT NOW (but not where
   they will go next). It marks a "danger zone": the obstacle cells plus
   every free cell next to them, because an obstacle could step there next.
2. If it has no plan, or the next cell of its plan is in the danger zone, it
   replans: it runs the search on a copy of the grid where the danger zone is
   treated as walls. If that finds no path, it tries again treating only
   the obstacle cells themselves as walls (a riskier path is better than none).
3. It moves one cell along the plan. If no path exists right now, it waits
   in place and tries again next step.
4. The obstacles move. If one steps onto the agent, it is a collision and
   the episode ends (failure).

The episode ends with success at the goal, failure on collision, or failure
when `max_steps` runs out (timeout).

Fair comparison: obstacles ignore the agent and use their own random
generator, so with the same obstacles and the same seed every algorithm
faces exactly the same obstacle movements.
"""

import copy
import random
import time
from dataclasses import dataclass, field

from pathfinding.algorithms.dijkstra import check_start_and_goal, get_neighbors
from pathfinding.env.obstacles import move_obstacles, obstacle_positions


@dataclass
class EpisodeResult:
    success: bool = False         # reached the goal
    collided: bool = False        # an obstacle stepped onto the agent
    steps: int = 0                # time steps used (moves + waits)
    moves: int = 0               # steps where the agent actually moved
    waits: int = 0               # steps where no path existed, so it waited
    plans: int = 0               # number of times the search was run
    nodes_expanded: int = 0      # total over all plans
    planning_time_ms: float = 0.0  # total time spent inside the search
    trajectory: list = field(default_factory=list)        # agent cell at each time step
    obstacle_history: list = field(default_factory=list)  # obstacle cells at each time step

    @property
    def replans(self):
        """Plans after the first one."""
        return max(0, self.plans - 1)


def grid_with_obstacles(grid, obstacle_cells):
    """Copy of the grid where the given cells are marked as walls."""
    planning_grid = [list(row) for row in grid]
    for row, col in obstacle_cells:
        planning_grid[row][col] = 1
    return planning_grid


def danger_zone(grid, obstacle_cells):
    """Obstacle cells plus every free cell an obstacle could move into next step."""
    cells = set(obstacle_cells)
    for cell in obstacle_cells:
        cells.update(get_neighbors(grid, cell))
    return cells


def run_episode(grid, start, goal, obstacles, search, seed=0, max_steps=None):
    """
    Simulate one episode.

    grid      : static map (0 = free, 1 = wall)
    obstacles : list of DynamicObstacle (copied, so the caller's list is untouched)
    search    : dijkstra, greedy_best_first or astar
    seed      : seed for the obstacles' random movement
    """
    check_start_and_goal(grid, start, goal)
    obstacles = copy.deepcopy(obstacles)
    rng = random.Random(seed)
    if max_steps is None:
        max_steps = 4 * len(grid) * len(grid[0])

    result = EpisodeResult(trajectory=[start], obstacle_history=[obstacle_positions(obstacles)])
    agent = start
    plan = []  # remaining planned cells; plan[0] is the agent's current cell

    while agent != goal and result.steps < max_steps:
        occupied = set(obstacle_positions(obstacles))
        danger = danger_zone(grid, occupied) - {agent, goal}  # goal is a safe zone

        def plan_avoiding(blocked_cells):
            start_time = time.perf_counter()
            new_plan, visited, _ = search(grid_with_obstacles(grid, blocked_cells), agent, goal)
            result.planning_time_ms += (time.perf_counter() - start_time) * 1000
            result.nodes_expanded += len(visited)
            result.plans += 1
            return new_plan

        # 2. Replan if there is no plan or the next cell might be hit next step.
        if len(plan) < 2 or plan[1] in danger:
            plan = plan_avoiding(danger)
            if len(plan) < 2:
                plan = plan_avoiding(occupied)  # no safe path: accept some risk

        # 3. Move along the plan, or wait if there is no path at the moment.
        if len(plan) >= 2:
            plan = plan[1:]
            agent = plan[0]
            result.moves += 1
        else:
            plan = []
            result.waits += 1
        result.steps += 1

        # 4. Obstacles move; collision if one lands on the agent.
        move_obstacles(obstacles, grid, goal, rng)
        result.trajectory.append(agent)
        result.obstacle_history.append(obstacle_positions(obstacles))
        if agent in result.obstacle_history[-1]:
            result.collided = True
            break

    result.success = agent == goal
    return result


if __name__ == "__main__":
    from pathfinding.algorithms.astar import astar
    from pathfinding.algorithms.dijkstra import dijkstra
    from pathfinding.algorithms.greedy_best_first import greedy_best_first
    from pathfinding.env.maps import random_grid
    from pathfinding.env.obstacles import create_obstacles

    algorithms = [("Dijkstra", dijkstra), ("Greedy", greedy_best_first), ("A*", astar)]
    size, density, num_seeds = 20, 0.2, 30
    start, goal = (0, 0), (size - 1, size - 1)

    for mode in ("random_walk", "patrol"):
        for num_obstacles in (0, 10, 30):
            print(f"\n{mode}, {num_obstacles} moving obstacles, {num_seeds} maps of {size}x{size}")
            print(f"{'algorithm':10}{'success':>9}{'collided':>10}{'moves':>8}"
                  f"{'replans':>9}{'nodes':>8}{'time ms':>9}")
            results = {name: [] for name, _ in algorithms}
            for seed in range(num_seeds):
                rng = random.Random(seed)
                grid = random_grid(size, size, density, rng, start, goal)
                obstacles = create_obstacles(grid, num_obstacles, rng, mode, forbidden={start, goal})
                for name, search in algorithms:
                    results[name].append(run_episode(grid, start, goal, obstacles, search, seed=seed))
            for name, episodes in results.items():
                successful = [e for e in episodes if e.success] or [EpisodeResult()]
                print(f"{name:10}"
                      f"{sum(e.success for e in episodes) / num_seeds:>9.0%}"
                      f"{sum(e.collided for e in episodes) / num_seeds:>10.0%}"
                      f"{sum(e.moves for e in successful) / len(successful):>8.1f}"
                      f"{sum(e.replans for e in episodes) / num_seeds:>9.1f}"
                      f"{sum(e.nodes_expanded for e in episodes) / num_seeds:>8.0f}"
                      f"{sum(e.planning_time_ms for e in episodes) / num_seeds:>9.2f}")
