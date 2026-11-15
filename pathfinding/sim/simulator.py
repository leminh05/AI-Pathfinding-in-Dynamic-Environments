"""
Step-by-step simulation of an agent moving among dynamic obstacles.

Any "agent" object can be simulated, as long as it has three methods:
    reset()                                           start of an episode
    next_cell(grid, position, goal, occupied, result) where it wants to go
    after_step(position, occupied, event)             what happened (for learning)

Two kinds of agent exist:
    PlannerAgent                 Dijkstra / Greedy / A* + the replanning rule below
    DynamicQLearningAgent        in pathfinding/algorithms/qlearning.py

For convenience you can also pass a plain search function (dijkstra,
greedy_best_first or astar) as the agent; it is wrapped in a PlannerAgent.

One time step:
1. The agent sees where every moving obstacle is RIGHT NOW (but not where
   they will go next) and chooses a cell: a neighbour, or its own cell (wait).
2. The agent moves. Trying to enter a wall or leave the grid is a "bump":
   it stays where it is. Walking into a moving obstacle is a collision.
3. The obstacles move. If one steps onto the agent, it is a collision.

The episode ends with success at the goal, failure on collision, or failure
when `max_steps` runs out (timeout).

Fair comparison: obstacles ignore the agent and use their own random
generator, so with the same obstacles and the same seed every agent faces
exactly the same obstacle movements.
"""

import copy
import random
import time
from dataclasses import dataclass, field

from pathfinding.algorithms.dijkstra import FREE, check_start_and_goal, get_neighbors
from pathfinding.env.obstacles import move_obstacles, obstacle_positions


@dataclass
class EpisodeResult:
    success: bool = False         # reached the goal
    collided: bool = False        # agent and an obstacle ended up in the same cell
    steps: int = 0                # time steps used (moves + waits + bumps)
    moves: int = 0                # steps where the agent actually moved
    waits: int = 0                # steps where the agent chose to stay
    bumps: int = 0                # steps where the agent walked into a wall / grid edge
    plans: int = 0                # number of times a search was run (planners only)
    nodes_expanded: int = 0       # total over all plans (planners only)
    planning_time_ms: float = 0.0  # total time spent choosing moves
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


def is_free(grid, cell):
    row, col = cell
    return 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == FREE


class PlannerAgent:
    """
    Turns a search function into an agent with a "plan, move, replan" rule:

    - It marks a "danger zone": the obstacle cells plus every free cell next
      to them, because an obstacle could step there next.
    - If it has no plan, or the next cell of its plan is in the danger zone,
      it replans on a copy of the grid where the danger zone is treated as
      walls. If that finds no path, it tries again treating only the obstacle
      cells themselves as walls (a riskier path is better than none).
    - If no path exists at all right now, it waits and tries again next step.
    """

    def __init__(self, search):
        self.search = search
        self.plan = []  # remaining planned cells; plan[0] is the current cell

    def reset(self):
        self.plan = []

    def next_cell(self, grid, position, goal, occupied, result):
        danger = danger_zone(grid, occupied) - {position, goal}  # goal is a safe zone
        if len(self.plan) < 2 or self.plan[1] in danger:
            self.plan = self.plan_avoiding(grid, danger, position, goal, result)
            if len(self.plan) < 2:
                self.plan = self.plan_avoiding(grid, occupied, position, goal, result)

        if len(self.plan) < 2:
            self.plan = []
            return position  # no path right now: wait
        self.plan = self.plan[1:]
        return self.plan[0]

    def plan_avoiding(self, grid, blocked_cells, position, goal, result):
        start_time = time.perf_counter()
        path, visited, _ = self.search(grid_with_obstacles(grid, blocked_cells), position, goal)
        result.planning_time_ms += (time.perf_counter() - start_time) * 1000
        result.nodes_expanded += len(visited)
        result.plans += 1
        return path

    def after_step(self, position, occupied, event):
        pass  # planners do not learn


def run_episode(grid, start, goal, obstacles, agent, seed=0, max_steps=None):
    """
    Simulate one episode.

    grid      : static map (0 = free, 1 = wall)
    obstacles : list of DynamicObstacle (copied, so the caller's list is untouched)
    agent     : an agent object, or a search function (dijkstra / greedy_best_first / astar)
    seed      : seed for the obstacles' random movement
    """
    check_start_and_goal(grid, start, goal)
    if callable(agent):
        agent = PlannerAgent(agent)
    obstacles = copy.deepcopy(obstacles)
    rng = random.Random(seed)
    if max_steps is None:
        max_steps = 4 * len(grid) * len(grid[0])

    result = EpisodeResult(trajectory=[start], obstacle_history=[obstacle_positions(obstacles)])
    position = start
    agent.reset()

    while position != goal and result.steps < max_steps:
        occupied = set(obstacle_positions(obstacles))

        # 1. The agent chooses where to go.
        start_time = time.perf_counter()
        target = agent.next_cell(grid, position, goal, occupied, result)
        if not isinstance(agent, PlannerAgent):  # planners time only their searches
            result.planning_time_ms += (time.perf_counter() - start_time) * 1000

        # 2. The agent moves (or waits, or bumps into a wall and stays).
        if target == position:
            event = "wait"
            result.waits += 1
        elif not is_free(grid, target):
            event = "bump"
            result.bumps += 1
        else:
            event = "move"
            position = target
            result.moves += 1
        result.steps += 1

        # 3. Obstacles move, unless the agent already walked into one.
        if position not in occupied:
            move_obstacles(obstacles, grid, goal, rng)
        occupied = set(obstacle_positions(obstacles))
        result.trajectory.append(position)
        result.obstacle_history.append(obstacle_positions(obstacles))

        if position in occupied:
            result.collided = True
            event = "collision"
        elif position == goal:
            event = "goal"
        agent.after_step(position, occupied, event)
        if result.collided:
            break

    result.success = position == goal
    return result


if __name__ == "__main__":
    from pathfinding.algorithms.astar import astar
    from pathfinding.algorithms.dijkstra import dijkstra
    from pathfinding.algorithms.greedy_best_first import greedy_best_first
    from pathfinding.algorithms.qlearning import DynamicQLearningAgent
    from pathfinding.env.maps import random_grid
    from pathfinding.env.obstacles import create_obstacles

    planners = [("Dijkstra", dijkstra), ("Greedy", greedy_best_first), ("A*", astar)]
    # Q-learning trains 3000 episodes on every map (about 10-20 s each), so the
    # demo uses few maps; the experiment runner will use more.
    size, density, num_maps, training_episodes = 20, 0.2, 10, 3000
    start, goal = (0, 0), (size - 1, size - 1)

    for mode in ("random_walk", "patrol"):
        for num_obstacles in (0, 10):
            print(f"\n{mode}, {num_obstacles} moving obstacles, {num_maps} maps of {size}x{size}"
                  f" (Q-learning trained {training_episodes} episodes per map)")
            print(f"{'algorithm':12}{'success':>9}{'collided':>10}{'moves':>8}"
                  f"{'replans':>9}{'nodes':>8}{'time ms':>9}{'train s':>9}")
            results = {name: [] for name, _ in planners + [("Q-learning", None)]}
            training_seconds = 0.0
            for seed in range(num_maps):
                rng = random.Random(seed)
                grid = random_grid(size, size, density, rng, start, goal)

                def make_obstacles(obstacle_rng):
                    return create_obstacles(grid, num_obstacles, obstacle_rng, mode,
                                            forbidden={start, goal})

                obstacles = make_obstacles(rng)
                for name, search in planners:
                    results[name].append(run_episode(grid, start, goal, obstacles, search, seed=seed))

                # Q-learning learns this map with other obstacle layouts, then is tested
                # on exactly the same obstacles and seed as the planners.
                q_agent = DynamicQLearningAgent(seed=seed)
                training_start = time.perf_counter()
                q_agent.train(grid, start, goal, make_obstacles, training_episodes,
                              first_seed=100_000 + seed * training_episodes)
                training_seconds += time.perf_counter() - training_start
                results["Q-learning"].append(run_episode(grid, start, goal, obstacles, q_agent, seed=seed))

            for name, episodes in results.items():
                successful = [e for e in episodes if e.success] or [EpisodeResult()]
                train_text = f"{training_seconds / num_maps:>9.2f}" if name == "Q-learning" else f"{'-':>9}"
                print(f"{name:12}"
                      f"{sum(e.success for e in episodes) / num_maps:>9.0%}"
                      f"{sum(e.collided for e in episodes) / num_maps:>10.0%}"
                      f"{sum(e.moves for e in successful) / len(successful):>8.1f}"
                      f"{sum(e.replans for e in episodes) / num_maps:>9.1f}"
                      f"{sum(e.nodes_expanded for e in episodes) / num_maps:>8.0f}"
                      f"{sum(e.planning_time_ms for e in episodes) / num_maps:>9.2f}"
                      f"{train_text}")
