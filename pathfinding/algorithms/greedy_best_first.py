"""
Greedy Best-First Search (GBFS) on a 2D grid.

Same grid format and same interface as dijkstra.py:
    0 = free cell, 1 = obstacle, positions are (row, col),
    4-directional moves, each move costs 1.

How GBFS works
--------------
GBFS looks a lot like Dijkstra, with one key difference in the priority queue:

    Dijkstra : priority = g(n) = distance travelled from the start so far
    GBFS     : priority = h(n) = ESTIMATED distance left to the goal

GBFS always expands the cell that *looks* closest to the goal, measured by a
heuristic (here: Manhattan distance). It ignores how far it has already
walked. That makes it fast (it heads straight for the goal and usually
expands far fewer cells), but the path it returns is NOT guaranteed to be
the shortest: it can walk into a dead end or around an obstacle the long way.

Steps:
1. Put the start in the priority queue with priority h(start).
2. Pop the cell with the smallest h. If it is the goal, stop.
3. Add each neighbour we have never seen before to the queue, remembering
   which cell we came from. (Unlike Dijkstra, we never "improve" a route
   later: the first route that reaches a cell is the one we keep.)
4. If the queue runs empty, the goal is unreachable.
"""

import heapq
import math

from pathfinding.algorithms.dijkstra import (
    MOVE_COST,
    check_start_and_goal,
    get_neighbors,
    reconstruct_path,
)


def manhattan_distance(cell_a, cell_b):
    """Number of 4-directional moves between two cells if there were no walls."""
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])


def greedy_best_first(grid, start, goal):
    """
    Find a path from `start` to `goal` on `grid` using Greedy Best-First Search.

    Returns a tuple (path, visited, cost), exactly like dijkstra():
        path    : list of (row, col) from start to goal, or [] if no path exists
        visited : list of cells in the order GBFS expanded them
        cost    : total cost of the returned path (may be longer than optimal),
                  or math.inf if no path exists
    """
    check_start_and_goal(grid, start, goal)

    # came_from[cell] = the cell we came from when we first discovered it.
    # The start has no parent, but we add it so we know it was "seen".
    came_from = {start: None}

    visited = []  # cells in the order they were expanded

    # Priority queue of (heuristic, cell): the cell that looks closest to the
    # goal comes out first.
    priority_queue = [(manhattan_distance(start, goal), start)]

    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)
        visited.append(current_cell)

        if current_cell == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = (len(path) - 1) * MOVE_COST
            return path, visited, cost

        for neighbor in get_neighbors(grid, current_cell):
            # Each cell is added to the queue only once, so no cell is
            # expanded twice and we never need to update a route.
            if neighbor in came_from:
                continue
            came_from[neighbor] = current_cell
            heapq.heappush(priority_queue, (manhattan_distance(neighbor, goal), neighbor))

    # Queue is empty and we never reached the goal: it is unreachable.
    return [], visited, math.inf


if __name__ == "__main__":
    from pathfinding.algorithms.dijkstra import dijkstra

    # A map where greedy gets fooled: it heads toward the goal, runs into the
    # walls near it and takes a detour. Dijkstra finds the shortest route.
    demo_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ]
    start, goal = (0, 0), (4, 6)

    for name, search in (("Dijkstra", dijkstra), ("GBFS", greedy_best_first)):
        path, visited, cost = search(demo_grid, start, goal)
        print(f"{name:8}  cost={cost}  cells visited={len(visited)}")
