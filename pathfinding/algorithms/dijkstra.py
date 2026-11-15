"""
Dijkstra's shortest-path algorithm on a 2D grid.

Grid format
-----------
The grid is a list of lists (or a 2D numpy array) of integers:
    0 = free cell (the agent can walk here)
    1 = obstacle  (the agent cannot enter)

Positions are (row, col) tuples. The agent moves in 4 directions
(up, down, left, right) and every move costs 1.

How Dijkstra works (short version)
----------------------------------
1. Every cell starts with distance = infinity, except the start (distance 0).
2. Keep a priority queue of cells ordered by their current best distance.
3. Repeatedly take the cell with the SMALLEST distance out of the queue.
   Its distance is now final: no shorter route to it can exist, because
   all other routes would go through cells that are already farther away.
4. For each neighbour of that cell, check if going through the current cell
   gives a shorter distance. If yes, update it and push it into the queue.
5. Stop when the goal is taken out of the queue (or the queue is empty,
   which means the goal is unreachable).
6. Rebuild the path by following "came_from" links back from the goal.
"""

import heapq
import math

# The four moves the agent can make: up, down, left, right.
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FREE = 0
MOVE_COST = 1


def get_neighbors(grid, cell):
    """Return the free cells next to `cell` that are inside the grid."""
    num_rows = len(grid)
    num_cols = len(grid[0])
    row, col = cell

    neighbors = []
    for row_change, col_change in MOVES:
        next_row = row + row_change
        next_col = col + col_change
        inside_grid = 0 <= next_row < num_rows and 0 <= next_col < num_cols
        if inside_grid and grid[next_row][next_col] == FREE:
            neighbors.append((next_row, next_col))
    return neighbors


def check_start_and_goal(grid, start, goal):
    """Raise ValueError if start or goal is outside the grid or on an obstacle."""
    for name, cell in (("start", start), ("goal", goal)):
        row, col = cell
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
            raise ValueError(f"{name} {cell} is outside the grid")
        if grid[row][col] != FREE:
            raise ValueError(f"{name} {cell} is on an obstacle")


def reconstruct_path(came_from, start, goal):
    """Follow the came_from links backwards from goal to start."""
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()  # we built it goal -> start, so flip it
    return path


def dijkstra(grid, start, goal):
    """
    Find the shortest path from `start` to `goal` on `grid`.

    Returns a tuple (path, visited, cost):
        path    : list of (row, col) from start to goal, or [] if no path exists
        visited : list of cells in the order Dijkstra expanded (finalized) them
        cost    : total cost of the path, or math.inf if no path exists
    """
    check_start_and_goal(grid, start, goal)

    # Best known distance from start to each cell. Missing cells = infinity.
    distance = {start: 0}

    # came_from[cell] = the cell we came from on the best known route.
    came_from = {}

    # Cells whose shortest distance is final (already expanded).
    finalized = set()
    visited = []  # same cells as `finalized`, but kept in expansion order

    # Priority queue of (distance, cell). heapq always pops the smallest.
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_cell = heapq.heappop(priority_queue)

        # The same cell can be in the queue several times with different
        # distances (we never delete old entries). Skip the outdated ones.
        if current_cell in finalized:
            continue

        finalized.add(current_cell)
        visited.append(current_cell)

        # The goal has the smallest distance in the queue, so it is final.
        if current_cell == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, visited, current_distance

        # "Relax" each neighbour: is going through current_cell shorter?
        for neighbor in get_neighbors(grid, current_cell):
            if neighbor in finalized:
                continue
            new_distance = current_distance + MOVE_COST
            if new_distance < distance.get(neighbor, math.inf):
                distance[neighbor] = new_distance
                came_from[neighbor] = current_cell
                heapq.heappush(priority_queue, (new_distance, neighbor))

    # Queue is empty and we never reached the goal: it is unreachable.
    return [], visited, math.inf


if __name__ == "__main__":
    # Small demo: S = start, G = goal, # = obstacle, * = path.
    demo_grid = [
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
    ]
    start, goal = (0, 0), (4, 2)
    path, visited, cost = dijkstra(demo_grid, start, goal)

    print(f"Cost: {cost}   Path length: {len(path)}   Cells visited: {len(visited)}")
    for row_index, row in enumerate(demo_grid):
        line = ""
        for col_index, value in enumerate(row):
            cell = (row_index, col_index)
            if cell == start:
                line += "S"
            elif cell == goal:
                line += "G"
            elif value != FREE:
                line += "#"
            elif cell in path:
                line += "*"
            else:
                line += "."
        print(line)
