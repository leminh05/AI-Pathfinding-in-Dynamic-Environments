"""
Pygame visualizer for Dijkstra, Greedy Best-First Search and A*.

Run from the project root:
    python -m pathfinding.viz.pygame_viz

Controls:
    1 / 2 / 3 or click a button : choose Dijkstra / Greedy / A*
    Space                       : replay the animation
    R                           : generate a new random map
    Esc or close the window     : quit

How it works: when you pick an algorithm, the search runs once (and is timed)
before any drawing. The animation then replays the result: first the cells
in the order the algorithm visited them, then the final path.
"""

import random
import time

import pygame

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.dijkstra import dijkstra
from pathfinding.algorithms.greedy_best_first import greedy_best_first

ALGORITHMS = [("Dijkstra", dijkstra), ("Greedy", greedy_best_first), ("A*", astar)]

# Map and window settings
ROWS, COLS = 25, 35
CELL_SIZE = 24
PANEL_WIDTH = 300
OBSTACLE_DENSITY = 0.28
START, GOAL = (0, 0), (ROWS - 1, COLS - 1)

# Animation speed
FPS = 60
VISITED_PER_FRAME = 3  # visited cells revealed each frame
PATH_PER_FRAME = 1     # path cells revealed each frame

# Colors (R, G, B)
BACKGROUND = (30, 30, 36)
FREE_COLOR = (240, 240, 240)
OBSTACLE_COLOR = (55, 60, 75)
VISITED_COLOR = (150, 200, 240)
PATH_COLOR = (250, 200, 50)
START_COLOR = (60, 180, 90)
GOAL_COLOR = (220, 70, 70)
TEXT_COLOR = (230, 230, 230)
BUTTON_COLOR = (70, 75, 95)
SELECTED_BUTTON_COLOR = (90, 140, 220)

PANEL_LEFT = COLS * CELL_SIZE
BUTTON_RECTS = [pygame.Rect(PANEL_LEFT + 20, 20 + i * 55, PANEL_WIDTH - 40, 42)
                for i in range(len(ALGORITHMS))]


def random_grid(rng):
    """Random map with obstacles. Only maps where the goal is reachable are kept."""
    while True:
        grid = [[1 if rng.random() < OBSTACLE_DENSITY else 0 for _ in range(COLS)]
                for _ in range(ROWS)]
        grid[START[0]][START[1]] = 0
        grid[GOAL[0]][GOAL[1]] = 0
        path, _, _ = astar(grid, START, GOAL)
        if path:
            return grid


class Visualizer:
    """Holds the map, the latest search result and the animation progress."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.grid = random_grid(self.rng)
        self.algorithm_index = 2  # start with A*
        self.run_search()

    def run_search(self):
        """Run the selected algorithm once, time it, and restart the animation."""
        _, search = ALGORITHMS[self.algorithm_index]
        start_time = time.perf_counter()
        self.path, self.visited, self.cost = search(self.grid, START, GOAL)
        self.elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.restart_animation()

    def restart_animation(self):
        self.visited_shown = 0  # how many visited cells are drawn so far
        self.path_shown = 0     # how many path cells are drawn so far

    def select_algorithm(self, index):
        self.algorithm_index = index
        self.run_search()

    def handle_event(self, event):
        """React to a key press or mouse click. Returns False to quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                self.select_algorithm(event.key - pygame.K_1)
            elif event.key == pygame.K_SPACE:
                self.restart_animation()
            elif event.key == pygame.K_r:
                self.grid = random_grid(self.rng)
                self.run_search()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in enumerate(BUTTON_RECTS):
                if rect.collidepoint(event.pos):
                    self.select_algorithm(index)
        return True

    def update(self):
        """Advance the animation by one frame: visited cells first, then the path."""
        if self.visited_shown < len(self.visited):
            self.visited_shown = min(len(self.visited), self.visited_shown + VISITED_PER_FRAME)
        elif self.path_shown < len(self.path):
            self.path_shown = min(len(self.path), self.path_shown + PATH_PER_FRAME)

    def draw(self, screen, font):
        screen.fill(BACKGROUND)

        # 1. Grid: free cells and obstacles
        for row in range(ROWS):
            for col in range(COLS):
                color = OBSTACLE_COLOR if self.grid[row][col] else FREE_COLOR
                draw_cell(screen, (row, col), color)

        # 2. Animated visited cells, then animated path on top
        for cell in self.visited[:self.visited_shown]:
            draw_cell(screen, cell, VISITED_COLOR)
        for cell in self.path[:self.path_shown]:
            draw_cell(screen, cell, PATH_COLOR)

        # 3. Start and goal always on top
        draw_cell(screen, START, START_COLOR)
        draw_cell(screen, GOAL, GOAL_COLOR)

        # 4. Side panel: algorithm buttons
        for index, rect in enumerate(BUTTON_RECTS):
            selected = index == self.algorithm_index
            pygame.draw.rect(screen, SELECTED_BUTTON_COLOR if selected else BUTTON_COLOR,
                             rect, border_radius=6)
            label = font.render(f"{index + 1}. {ALGORITHMS[index][0]}", True, TEXT_COLOR)
            screen.blit(label, label.get_rect(center=rect.center))

        # 5. Side panel: results and controls
        path_text = f"{self.cost} moves" if self.path else "no path"
        lines = [
            f"Path length: {path_text}",
            f"Nodes explored: {len(self.visited)}",
            f"Execution time: {self.elapsed_ms:.3f} ms",
            "",
            "Space: replay",
            "R: new map",
            "Esc: quit",
        ]
        text_top = BUTTON_RECTS[-1].bottom + 30
        for line_number, line in enumerate(lines):
            surface = font.render(line, True, TEXT_COLOR)
            screen.blit(surface, (PANEL_LEFT + 20, text_top + line_number * 30))


def draw_cell(screen, cell, color):
    row, col = cell
    # -1 leaves a thin gap between cells so the grid lines are visible
    pygame.draw.rect(screen, color,
                     (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))


def main():
    pygame.init()
    screen = pygame.display.set_mode((COLS * CELL_SIZE + PANEL_WIDTH, ROWS * CELL_SIZE))
    pygame.display.set_caption("Pathfinding: Dijkstra / Greedy / A*")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    visualizer = Visualizer()

    running = True
    while running:
        for event in pygame.event.get():
            if not visualizer.handle_event(event):
                running = False
        visualizer.update()
        visualizer.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
