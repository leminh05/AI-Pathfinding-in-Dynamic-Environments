# AI Pathfinding in Dynamic Environments

Comparing **Dijkstra**, **Greedy Best-First Search**, **A\*** and **Q-learning** on
grid worlds with static walls and moving obstacles.

The project contains the four algorithms, a step-by-step simulator for moving
obstacles, an experiment runner (840 maps), a statistical analysis and two
Pygame visualizers. Everything is plain Python, written to be read by students.

| Algorithm | Picks the next cell by | Shortest path? | Needs training? |
|---|---|---|---|
| Dijkstra | distance travelled so far, g(n) | yes | no |
| Greedy Best-First | estimated distance to goal, h(n) | no | no |
| A\* | g(n) + h(n), Manhattan heuristic | yes | no |
| Q-learning | learned value of each action | learned | yes, per map |

## Installation

Python 3.10 or newer (tested on 3.11).

```bash
pip install -r requirements.txt
```

## Quick start

All commands are run from the project root.

| What | Command |
|---|---|
| Dijkstra / Greedy / A\* demos (text) | `python -m pathfinding.algorithms.astar` |
| Q-learning on a static maze (plots) | `python -m pathfinding.algorithms.qlearning` |
| Visualizer, static map | `python -m pathfinding.viz.pygame_viz` |
| Visualizer, moving obstacles | `python -m pathfinding.viz.dynamic_viz` |
| Small comparison table, all 4 algorithms | `python -m pathfinding.sim.simulator` |
| Full experiment (~20 min on 12 cores) | `python -m pathfinding.experiments.run` |
| Analysis of the experiment | `python -m pathfinding.experiments.analyze` |
| Tests | `python -m pytest` (after `pip install pytest`) |

Each test file also runs without pytest, for example `python -m tests.test_astar`.

### Visualizer controls

**Static map** (`pygame_viz`): `1` `2` `3` or click a button to choose the algorithm,
`Space` replay, `R` new map, `Esc` quit. Visited cells are animated in blue, then
the final path in yellow; the panel shows path length, nodes explored and
execution time.

**Moving obstacles** (`dynamic_viz`): `1`-`4` algorithm (4 = Q-learning),
`Up`/`Down` number of obstacles (0, 5, 10, 20), `M` random walk / patrol,
`N` new obstacle layout, `R` new map, `Space` replay, `P` pause, `Esc` quit.
Q-learning trains in the background first (about 10-20 s).

## Project structure

```
pathfinding/
  algorithms/   dijkstra.py, greedy_best_first.py, astar.py   (grid, start, goal) -> (path, visited, cost)
                qlearning.py   QLearningAgent (static), DynamicQLearningAgent (moving obstacles)
  env/          maps.py        random reachable maps
                obstacles.py   moving obstacles: random walk and patrol
  sim/          simulator.py   run_episode(): one episode, any agent, with replanning
  experiments/  run.py         experiment runner -> results/results.csv
                analyze.py     tables, statistical tests, figures -> results/analysis/
  viz/          pygame_viz.py  static visualizer
                dynamic_viz.py visualizer with moving obstacles
tests/          one test file per module
results/        generated output (git-ignored)
```

Grid format everywhere: a list of lists, `0` = free, `1` = wall; cells are
`(row, col)`; the agent moves up, down, left or right at cost 1.

## How the dynamic environment works

Each time step:

1. The agent sees where every moving obstacle is **now** (not where it will go).
2. The agent moves one cell or waits. Walking into a wall is a "bump" (it stays).
3. Every obstacle moves one cell. If an obstacle ends up on the agent, it is a
   **collision** and the episode fails.

An episode ends with success (goal), collision, or timeout (4 x number of cells
steps). Obstacles never enter walls, the goal or each other, and they ignore the
agent. With the same seed, every algorithm faces exactly the same obstacle
movement, so comparisons are fair.

**Dijkstra, Greedy and A\*** share one replanning rule (`PlannerAgent`): cells
holding an obstacle *or next to one* form a **danger zone**. The agent replans
whenever its next cell is in the danger zone, treating it as walls; if no path
avoids it, it replans around the obstacle cells only; if there is still no path,
it waits.

**Q-learning** (`DynamicQLearningAgent`):

- State: position + 5 bits saying whether its own cell and each neighbour are
  in the danger zone
- Actions: up, down, left, right, wait
- Rewards: goal +100, collision -100, bump -10, every step -1

Three choices were needed before it learned anything; each one is explained in the code:

- **Potential-based reward shaping** toward the goal (Ng, Harada & Russell,
  1999). Without it, a randomly exploring agent is hit long before it finds the
  goal.
- **Discount 0.99.** With 0.95, the shaping bonus for standing still outweighs
  the step cost, and the agent learns to wait forever.
- **Curriculum.** The first 20% of training episodes have no moving obstacles.

Q-learning is trained for 3000 episodes on each map, using obstacle layouts
other than the ones it is tested on, and is evaluated greedily (no exploration).

## Experiments

`python -m pathfinding.experiments.run` runs every combination of:

| Factor | Levels |
|---|---|
| Grid size | 15x15, 20x20 |
| Wall density | 10%, 20% |
| Moving obstacles | 0, 5, 10, 20 |
| Movement | random walk, patrol |
| Seeds | 30 per combination |

That is 840 maps x 4 algorithms = 3360 episodes. The map depends only on (size,
density, seed), so every scenario reuses the same maps and all comparisons are
paired. Finished maps are written immediately, so an interrupted run can be
restarted and continues where it stopped. The `--seeds`, `--episodes`,
`--workers` and `--out` options change the setup, for example
`--seeds 2 --episodes 300` for a quick test.

Metrics per episode: success, collision, timeout, moves, waits, bumps, replans,
nodes expanded, decision time, path ratio (moves / shortest static path) and
Q-learning training time.

`python -m pathfinding.experiments.analyze` writes `report.md`, `summary.csv`,
`stats.csv` and four figures to `results/analysis/`. All tests are paired:

- **Success:** Cochran's Q across the four algorithms, then exact McNemar tests between pairs.
- **Path ratio and nodes expanded:** Friedman, then Wilcoxon signed-rank tests between pairs.
- **Pairwise p-values** are Holm-corrected.
- **Intervals:** 95% Wilson for rates, t for means.

## Results

Success rate (120 maps per cell, pooled over grid size and wall density):

| Moving obstacles | Dijkstra | Greedy | A\* | Q-learning |
|---|---|---|---|---|
| 0 | 100% | 100% | 100% | 100% |
| 5 random walk | 87% | 88% | 87% | 84% |
| 10 random walk | 77% | 76% | 77% | 69% |
| 20 random walk | 42% | 34% | 40% | 14% |
| 5 patrol | 89% | 90% | 89% | 35% |
| 10 patrol | 76% | 71% | 75% | 13% |
| 20 patrol | 42% | 36% | 39% | 1% |

Main findings:

- **Without moving obstacles**, Dijkstra, A\* and Q-learning all find shortest
  paths. Greedy's paths are about 10% longer, but it expands the fewest nodes
  (41 on average, against 55 for A\* and 265 for Dijkstra).
- **With moving obstacles**, the three search algorithms do not differ
  significantly in success rate (e.g. Cochran's Q p = 0.24 at 10 random-walk
  obstacles). Success is decided by the shared replanning rule, not by the
  search. Greedy's paths stay significantly longer (p < 0.001), and A\* expands
  3-5x fewer nodes than Dijkstra (median).
- **Q-learning** is not significantly worse at 5 and 10 random-walk obstacles,
  but it is from 20 onwards and with patrolling obstacles at every level
  (p < 0.001). Its paths are 1.4-2.2x the optimum, and against patrols it
  mostly times out, stuck in loops that its local view cannot resolve. It also
  needs 13-48 s of training per map.

## Limitations

- Results are pooled over grid size and wall density.
- In crowded scenarios, few maps have all four algorithms succeed, so the
  path-ratio comparisons there rest on small n (reported per test in
  `report.md`).
- Q-learning is tabular and map-specific: it must be retrained for every map.
- Q-learning is evaluated without exploration. Allowing 5% random actions
  during evaluation raised patrol success sharply (4% -> 60% in a pilot run on
  one map) but lowered random-walk success.
- Planners know where obstacles are now but do not predict where they will go.
