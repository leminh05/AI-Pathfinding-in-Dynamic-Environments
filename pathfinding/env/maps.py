"""Random grid maps (0 = free, 1 = static obstacle)."""

from pathfinding.algorithms.astar import astar


def random_grid(num_rows, num_cols, obstacle_density, rng, start, goal):
    """
    Random map where each cell is an obstacle with probability `obstacle_density`.
    Start and goal are always free, and only maps where the goal is reachable
    are returned (unreachable ones are thrown away and regenerated).
    """
    while True:
        grid = [[1 if rng.random() < obstacle_density else 0 for _ in range(num_cols)]
                for _ in range(num_rows)]
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
        path, _, _ = astar(grid, start, goal)
        if path:
            return grid
