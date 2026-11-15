"""
A* search on a 2D grid.

Same grid format and same interface as dijkstra.py and greedy_best_first.py:
    0 = free cell, 1 = obstacle, positions are (row, col),
    4-directional moves, each move costs 1.

How A* works
------------
A* combines the two ideas you have already seen:

    Dijkstra : priority = g(n)          (distance travelled from the start)
    GBFS     : priority = h(n)          (estimated distance left to the goal)
    A*       : priority = f(n) = g(n) + h(n)

f(n) is an estimate of the TOTAL length of the best path that goes through n.
So A* prefers cells that are both cheap to reach AND look close to the goal.

The heuristic here is Manhattan distance. It never overestimates the real
remaining distance (walls can only make the trip longer), which is called
an "admissible" heuristic. With an admissible heuristic A* is guaranteed to
return the SHORTEST path, like Dijkstra, while usually expanding far fewer
cells because it does not waste time exploring away from the goal.

The code is Dijkstra's code with one change: the queue is ordered by g + h
instead of g.
"""

import heapq
import math

from pathfinding.algorithms.dijkstra import (
    MOVE_COST,
    check_start_and_goal,
    get_neighbors,
    reconstruct_path,
)
from pathfinding.algorithms.greedy_best_first import manhattan_distance


def astar(grid, start, goal):
    """
    Find the shortest path from `start` to `goal` on `grid` using A*.

    Returns a tuple (path, visited, cost), exactly like dijkstra():
        path    : list of (row, col) from start to goal, or [] if no path exists
        visited : list of cells in the order A* expanded them
        cost    : total cost of the path, or math.inf if no path exists
    """
    check_start_and_goal(grid, start, goal)

    # g_score[cell] = best known distance from start to cell (like Dijkstra).
    g_score = {start: 0}
    came_from = {}
    finalized = set()
    visited = []

    # Queue entries are (f, h, cell). Sorting by f first gives A*.
    # If two cells have the same f, the one with smaller h (closer to the
    # goal) comes out first. This tie-break does not change the path cost,
    # it just avoids expanding many equally good cells.
    start_h = manhattan_distance(start, goal)
    priority_queue = [(start_h, start_h, start)]

    while priority_queue:
        _, _, current_cell = heapq.heappop(priority_queue)

        # Skip outdated queue entries (same trick as in Dijkstra).
        if current_cell in finalized:
            continue

        finalized.add(current_cell)
        visited.append(current_cell)

        if current_cell == goal:
            path = reconstruct_path(came_from, start, goal)
            return path, visited, g_score[goal]

        for neighbor in get_neighbors(grid, current_cell):
            if neighbor in finalized:
                continue
            new_g = g_score[current_cell] + MOVE_COST
            if new_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = new_g
                came_from[neighbor] = current_cell
                h = manhattan_distance(neighbor, goal)
                heapq.heappush(priority_queue, (new_g + h, h, neighbor))

    # Queue is empty and we never reached the goal: it is unreachable.
    return [], visited, math.inf


if __name__ == "__main__":
    from pathfinding.algorithms.dijkstra import dijkstra
    from pathfinding.algorithms.greedy_best_first import greedy_best_first

    # Same map as the GBFS demo: greedy takes a detour here, A* does not.
    demo_grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ]
    start, goal = (0, 0), (4, 6)

    for name, search in (("Dijkstra", dijkstra), ("GBFS", greedy_best_first), ("A*", astar)):
        path, visited, cost = search(demo_grid, start, goal)
        print(f"{name:8}  cost={cost}  cells visited={len(visited)}")
