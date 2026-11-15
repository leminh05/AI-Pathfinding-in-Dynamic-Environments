"""
Dynamic (moving) obstacles for the grid world.

Two kinds of movement:
    random walk : every time step, stay or move to a random free neighbour
    patrol      : walk back and forth along a fixed straight route

Rules every obstacle follows:
    - never enters a static obstacle (1 in the grid) or leaves the grid
    - never enters the goal cell (the goal is a safe zone)
    - never enters a cell where another moving obstacle is
    - does NOT avoid the agent: if it steps onto the agent, that is a collision
"""

from pathfinding.algorithms.dijkstra import FREE, MOVES, get_neighbors


class DynamicObstacle:
    def __init__(self, position, route=None):
        self.position = position
        # route = None means random walk; otherwise a list of cells to patrol.
        self.route = route
        self.route_index = 0  # where on the route we are
        self.direction = 1    # +1 = forward along the route, -1 = backward

    def move(self, grid, blocked, rng):
        """Take one step. `blocked` = cells this obstacle may not enter right now."""
        if self.route is None:
            options = [self.position] + [cell for cell in get_neighbors(grid, self.position)
                                         if cell not in blocked]
            self.position = rng.choice(options)
            return

        if len(self.route) == 1:
            return  # a route of one cell is a standing obstacle
        next_index = self.route_index + self.direction
        if not 0 <= next_index < len(self.route):
            self.direction = -self.direction  # reached an end: turn around
            next_index = self.route_index + self.direction
        if self.route[next_index] in blocked:
            return  # something is in the way: wait this step
        self.route_index = next_index
        self.position = self.route[next_index]


def move_obstacles(obstacles, grid, goal, rng):
    """Move every obstacle one step (one at a time, so they never overlap)."""
    taken = {obstacle.position for obstacle in obstacles}
    for obstacle in obstacles:
        taken.discard(obstacle.position)
        obstacle.move(grid, taken | {goal}, rng)
        taken.add(obstacle.position)


def obstacle_positions(obstacles):
    return [obstacle.position for obstacle in obstacles]


def create_obstacles(grid, count, rng, mode="random_walk", forbidden=(), patrol_length=6):
    """
    Place `count` moving obstacles on random free cells (not in `forbidden`,
    which should contain at least the start and goal).

    mode = "random_walk" or "patrol". A patrol route goes in a random straight
    direction for up to `patrol_length` cells, stopping at walls.
    """
    if mode not in ("random_walk", "patrol"):
        raise ValueError(f"unknown obstacle mode: {mode}")
    forbidden = set(forbidden)

    def allowed(cell):
        row, col = cell
        return (0 <= row < len(grid) and 0 <= col < len(grid[0])
                and grid[row][col] == FREE and cell not in forbidden)

    free_cells = [(row, col) for row in range(len(grid)) for col in range(len(grid[0]))
                  if allowed((row, col))]
    if count > len(free_cells):
        raise ValueError(f"cannot place {count} obstacles on {len(free_cells)} free cells")

    obstacles = []
    for position in rng.sample(free_cells, count):
        if mode == "random_walk":
            obstacles.append(DynamicObstacle(position))
            continue
        row_change, col_change = rng.choice(MOVES)
        route = [position]
        while len(route) < patrol_length:
            next_cell = (route[-1][0] + row_change, route[-1][1] + col_change)
            if not allowed(next_cell):
                break
            route.append(next_cell)
        obstacles.append(DynamicObstacle(position, route))
    return obstacles
