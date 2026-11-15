"""
Pygame visualizer for the DYNAMIC environment: watch an agent move step by
step among moving obstacles.

Run from the project root:
    python -m pathfinding.viz.dynamic_viz

Controls:
    1 / 2 / 3 / 4  Dijkstra / Greedy / A* / Q-learning
    Up / Down      more / fewer moving obstacles (0, 5, 10, 20)
    M              switch obstacle movement: random walk <-> patrol
    N              new obstacle layout (same map)
    R              new map
    Space          replay the episode from the start
    P              pause / resume
    Esc            quit

How it works: the episode is simulated in one go with run_episode() (the same
simulator the experiments use), then replayed frame by frame from the
recorded agent and obstacle positions. Q-learning must first be trained on
the current map and obstacle settings; that runs in a background thread
(about 10-20 seconds) and is remembered, so switching back is instant.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import pygame

from pathfinding.algorithms.astar import astar
from pathfinding.algorithms.dijkstra import dijkstra
from pathfinding.algorithms.greedy_best_first import greedy_best_first
from pathfinding.algorithms.qlearning import DynamicQLearningAgent
from pathfinding.env.maps import random_grid
from pathfinding.env.obstacles import create_obstacles
from pathfinding.sim.simulator import run_episode
from pathfinding.viz.pygame_viz import (
    BACKGROUND, FREE_COLOR, GOAL_COLOR, OBSTACLE_COLOR, SELECTED_BUTTON_COLOR, START_COLOR,
    TEXT_COLOR, VISITED_COLOR,
)

ALGORITHMS = ["Dijkstra", "Greedy", "A*", "Q-learning"]
SEARCHES = {"Dijkstra": dijkstra, "Greedy": greedy_best_first, "A*": astar}
OBSTACLE_COUNTS = [0, 5, 10, 20]
MODES = ["random_walk", "patrol"]

SIZE = 20                 # same 20x20 maps as the experiments
WALL_DENSITY = 0.2
START, GOAL = (0, 0), (SIZE - 1, SIZE - 1)
CELL_SIZE = 30
PANEL_WIDTH = 330
TRAINING_EPISODES = 3000  # same as the experiments
STEP_MS = 150             # time per simulation step in the replay
FPS = 60

AGENT_COLOR = (28, 92, 171)
MOVING_OBSTACLE_COLOR = (235, 104, 52)
COLLISION_COLOR = (208, 59, 59)
MUTED_TEXT = (160, 160, 170)


def train_q_agent(grid, num_obstacles, mode):
    """Train a Q-learning agent for one map and obstacle setting (runs in a thread)."""
    def make_obstacles(rng):
        if num_obstacles == 0:
            return []
        return create_obstacles(grid, num_obstacles, rng, mode, forbidden={START, GOAL})

    agent = DynamicQLearningAgent(seed=0)
    agent.train(grid, START, GOAL, make_obstacles, TRAINING_EPISODES, first_seed=1_000_000)
    return agent


class DynamicVisualizer:
    def __init__(self, map_seed=0):
        self.map_seed = map_seed
        self.obstacle_seed = 0
        self.algorithm_index = 2  # A*
        self.count_index = 2      # 10 obstacles
        self.mode_index = 0       # random walk
        self.paused = False
        self.trainer = ThreadPoolExecutor(max_workers=1)
        self.trained_agents = {}   # (map, count, mode) -> trained agent
        self.training_jobs = {}    # (map, count, mode) -> future still running
        self.new_map()

    # --- settings ------------------------------------------------------------

    @property
    def algorithm(self):
        return ALGORITHMS[self.algorithm_index]

    @property
    def num_obstacles(self):
        return OBSTACLE_COUNTS[self.count_index]

    @property
    def mode(self):
        return MODES[self.mode_index]

    def q_key(self):
        return self.map_seed, self.num_obstacles, self.mode

    def new_map(self):
        self.grid = random_grid(SIZE, SIZE, WALL_DENSITY, random.Random(self.map_seed), START, GOAL)
        self.new_episode()

    def make_obstacles(self):
        if self.num_obstacles == 0:
            return []
        return create_obstacles(self.grid, self.num_obstacles, random.Random(self.obstacle_seed),
                                self.mode, forbidden={START, GOAL})

    def new_episode(self):
        """Simulate the episode for the current settings (or start Q-learning training)."""
        self.result = None
        self.time_step = 0
        self.time_since_step = 0
        if self.algorithm == "Q-learning":
            key = self.q_key()
            if key not in self.trained_agents:
                if key not in self.training_jobs:
                    self.training_jobs[key] = self.trainer.submit(
                        train_q_agent, self.grid, self.num_obstacles, self.mode)
                return  # update() calls new_episode() again once training is done
            agent = self.trained_agents[key]
        else:
            agent = SEARCHES[self.algorithm]
        self.result = run_episode(self.grid, START, GOAL, self.make_obstacles(), agent,
                                  seed=self.obstacle_seed)

    def handle_event(self, event):
        """React to a key press. Returns False to quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type != pygame.KEYDOWN:
            return True
        key = event.key
        if key == pygame.K_ESCAPE:
            return False
        if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.algorithm_index = key - pygame.K_1
        elif key == pygame.K_UP:
            self.count_index = min(len(OBSTACLE_COUNTS) - 1, self.count_index + 1)
        elif key == pygame.K_DOWN:
            self.count_index = max(0, self.count_index - 1)
        elif key == pygame.K_m:
            self.mode_index = 1 - self.mode_index
        elif key == pygame.K_n:
            self.obstacle_seed += 1
        elif key == pygame.K_r:
            self.map_seed += 1
            self.new_map()
            return True
        elif key == pygame.K_SPACE:
            self.time_step = 0
            self.paused = False
            return True
        elif key == pygame.K_p:
            self.paused = not self.paused
            return True
        else:
            return True
        self.new_episode()
        return True

    def update(self, milliseconds):
        # Collect finished training jobs; if it is the one we wait for, simulate now.
        for key, job in list(self.training_jobs.items()):
            if job.done():
                self.trained_agents[key] = job.result()
                del self.training_jobs[key]
                if self.result is None and self.algorithm == "Q-learning" and key == self.q_key():
                    self.new_episode()

        if self.result is None or self.paused:
            return
        self.time_since_step += milliseconds
        last_step = len(self.result.trajectory) - 1
        while self.time_since_step >= STEP_MS and self.time_step < last_step:
            self.time_since_step -= STEP_MS
            self.time_step += 1

    # --- drawing -------------------------------------------------------------

    def draw(self, screen, font):
        screen.fill(BACKGROUND)
        for row in range(SIZE):
            for col in range(SIZE):
                draw_square(screen, (row, col), OBSTACLE_COLOR if self.grid[row][col] else FREE_COLOR)
        draw_square(screen, START, START_COLOR)
        draw_square(screen, GOAL, GOAL_COLOR)

        if self.result is not None:
            trail = self.result.trajectory[:self.time_step + 1]
            for cell in trail[1:-1]:
                if cell not in (START, GOAL):
                    draw_square(screen, cell, VISITED_COLOR)
            for cell in self.result.obstacle_history[self.time_step]:
                draw_circle(screen, cell, MOVING_OBSTACLE_COLOR, CELL_SIZE * 0.38)
            agent_cell = trail[-1]
            draw_circle(screen, agent_cell, AGENT_COLOR, CELL_SIZE * 0.32)
            if self.finished() and self.result.collided:
                draw_circle(screen, agent_cell, COLLISION_COLOR, CELL_SIZE * 0.48, width=3)

        self.draw_panel(screen, font)

    def finished(self):
        return self.result is not None and self.time_step == len(self.result.trajectory) - 1

    def draw_panel(self, screen, font):
        left, y = SIZE * CELL_SIZE + 20, 16

        def line(text, color=TEXT_COLOR, gap=26):
            nonlocal y
            screen.blit(font.render(text, True, color), (left, y))
            y += gap

        for index, name in enumerate(ALGORITHMS):
            selected = index == self.algorithm_index
            line(f"{'>' if selected else ' '} {index + 1}. {name}",
                 SELECTED_BUTTON_COLOR if selected else TEXT_COLOR)
        y += 8
        line(f"Moving obstacles: {self.num_obstacles}")
        line(f"Movement: {self.mode.replace('_', ' ')}", gap=36)

        if self.result is None:
            line("Training Q-learning...", SELECTED_BUTTON_COLOR)
            line(f"({TRAINING_EPISODES} episodes, ~10-20 s)", MUTED_TEXT)
        else:
            result = self.result
            if not self.finished():
                outcome = "paused" if self.paused else "running"
            elif result.success:
                outcome = "SUCCESS"
            elif result.collided:
                outcome = "COLLISION"
            else:
                outcome = "TIMEOUT"
            line(f"Step {self.time_step} / {len(result.trajectory) - 1}: {outcome}", gap=34)
            line("Whole episode:", MUTED_TEXT)
            line(f"Moves {result.moves}, waits {result.waits}, bumps {result.bumps}")
            if self.algorithm == "Q-learning":
                line("Learned policy (no search)")
            else:
                line(f"Replans: {result.replans}")
                line(f"Nodes expanded: {result.nodes_expanded}")
            line(f"Decision time: {result.planning_time_ms:.2f} ms")

        y = SIZE * CELL_SIZE - 6 * 22 - 10
        for text in ["1-4 algorithm   Up/Down obstacles", "M movement   N new obstacles",
                     "R new map   Space replay", "P pause   Esc quit",
                     "Blue trail: agent path", "Orange: moving obstacles"]:
            line(text, MUTED_TEXT, gap=22)


def cell_rect(cell):
    row, col = cell
    return pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)


def draw_square(screen, cell, color):
    pygame.draw.rect(screen, color, cell_rect(cell))


def draw_circle(screen, cell, color, radius, width=0):
    pygame.draw.circle(screen, color, cell_rect(cell).center, int(radius), width)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SIZE * CELL_SIZE + PANEL_WIDTH, SIZE * CELL_SIZE))
    pygame.display.set_caption("Pathfinding among moving obstacles")
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    visualizer = DynamicVisualizer()

    running = True
    while running:
        for event in pygame.event.get():
            if not visualizer.handle_event(event):
                running = False
        visualizer.update(clock.get_time())
        visualizer.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)

    visualizer.trainer.shutdown(wait=False, cancel_futures=True)
    pygame.quit()


if __name__ == "__main__":
    main()
