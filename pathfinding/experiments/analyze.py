"""
Analysis of results/results.csv (made by pathfinding.experiments.run).

Run from the project root:
    python -m pathfinding.experiments.analyze

Writes to results/analysis/:
    summary.csv       one row per (obstacle count, mode, algorithm): rates with
                      95% confidence intervals, path ratio, nodes, time
    stats.csv         every statistical test with raw and Holm-corrected p-values
    report.md         the two tables above, formatted for reading
    success_rate.png  success rate vs number of moving obstacles
    outcomes.png      success / collision / timeout breakdown
    path_ratio.png    path length relative to the static optimum
    nodes_expanded.png  search effort of the three planners

A "scenario" is (number of moving obstacles, movement mode). Inside a scenario
the four algorithms ran on exactly the same maps (same size, density, seed), so
all tests are PAIRED tests:
    success (yes/no)        Cochran's Q over the 4 algorithms, then McNemar
                            (exact) for every pair
    path ratio, nodes       Friedman over the algorithms, then Wilcoxon
                            signed-rank for every pair
Pairwise p-values are Holm-corrected inside each scenario and metric, so the
chance of any false "significant" result per family stays at 5%.
"""

import argparse
import itertools
import math
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")  # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402

ALGORITHMS = ["Dijkstra", "Greedy", "A*", "Q-learning"]
PLANNERS = ["Dijkstra", "Greedy", "A*"]
MODES = ["random_walk", "patrol"]
PAIR_KEY = ["size", "density", "seed"]  # identifies the same map inside a scenario
ALPHA = 0.05

# Chart style: validated categorical palette (slots 1-4) plus a marker shape per
# algorithm, so identity never depends on color alone.
COLORS = {"Dijkstra": "#2a78d6", "Greedy": "#eb6834", "A*": "#1baf7a", "Q-learning": "#eda100"}
MARKERS = {"Dijkstra": "o", "Greedy": "s", "A*": "^", "Q-learning": "D"}
OUTCOME_COLORS = {"success": "#0ca30c", "collided": "#d03b3b", "timeout": "#fab219"}
SURFACE, INK, INK_SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# --- small statistics helpers ---------------------------------------------------

def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion (better than +-1.96*SE near 0 or 1)."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_ci(values):
    """Mean and the half-width of its 95% t-interval."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), math.nan
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(half)


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p_values = list(p_values)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [math.nan] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        if math.isnan(p_values[index]):
            continue
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def cochran_q(binary):
    """Cochran's Q test. `binary` is a (maps x algorithms) array of 0/1."""
    binary = np.asarray(binary)
    k = binary.shape[1]
    column_totals = binary.sum(axis=0)
    row_totals = binary.sum(axis=1)
    total = binary.sum()
    denominator = k * total - (row_totals ** 2).sum()
    if denominator == 0:
        return math.nan, 1.0  # every map had the same outcome for all algorithms
    q = (k - 1) * (k * (column_totals ** 2).sum() - total ** 2) / denominator
    return float(q), float(stats.chi2.sf(q, k - 1))


def mcnemar_exact(a, b):
    """Exact McNemar test for two paired 0/1 arrays."""
    a, b = np.asarray(a), np.asarray(b)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    if only_a + only_b == 0:
        return 0.0, 1.0
    return float(min(only_a, only_b)), float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)


def friedman(matrix):
    """Friedman test on a (maps x algorithms) array; p = 1 if all columns are identical."""
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) < 2 or np.all(matrix == matrix[:, :1]):
        return math.nan, 1.0
    result = stats.friedmanchisquare(*matrix.T)
    return float(result.statistic), float(result.pvalue)


def wilcoxon(a, b):
    """Wilcoxon signed-rank test; p = 1 if the two samples are identical."""
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(differences == 0):
        return math.nan, 1.0
    result = stats.wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)


# --- loading and tables ------------------------------------------------------

def load_results(path):
    df = pd.read_csv(path)
    df["outcome"] = np.select([df.success == 1, df.collided == 1], ["success", "collided"], "timeout")
    return df


def scenarios(df):
    """(num_obstacles, mode) pairs in a readable order."""
    keys = df[["num_obstacles", "mode"]].drop_duplicates()
    keys["mode_order"] = keys["mode"].map({"none": 0, "random_walk": 1, "patrol": 2})
    keys = keys.sort_values(["num_obstacles", "mode_order"])
    return list(keys[["num_obstacles", "mode"]].itertuples(index=False, name=None))


def summary_table(df):
    rows = []
    for (num_obstacles, mode) in scenarios(df):
        scenario = df[(df.num_obstacles == num_obstacles) & (df["mode"] == mode)]
        for algorithm in ALGORITHMS:
            group = scenario[scenario.algorithm == algorithm]
            n = len(group)
            if n == 0:
                continue
            successes = int(group.success.sum())
            low, high = wilson_interval(successes, n)
            ratio_mean, ratio_ci = mean_ci(group.path_ratio.dropna())
            nodes_mean, nodes_ci = mean_ci(group.nodes_expanded)
            nodes_median = float(group.nodes_expanded.median())
            time_mean, time_ci = mean_ci(group.planning_time_ms)
            rows.append({
                "num_obstacles": num_obstacles, "mode": mode, "algorithm": algorithm, "maps": n,
                "success_rate": successes / n, "success_ci_low": low, "success_ci_high": high,
                "collision_rate": group.collided.mean(), "timeout_rate": group.timeout.mean(),
                "path_ratio_mean": ratio_mean, "path_ratio_ci": ratio_ci,
                "nodes_mean": nodes_mean, "nodes_ci": nodes_ci, "nodes_median": nodes_median,
                "planning_time_ms_mean": time_mean, "planning_time_ms_ci": time_ci,
                "train_time_s_mean": group.train_time_s.mean() if algorithm == "Q-learning" else math.nan,
            })
    return pd.DataFrame(rows)


def statistical_tests(df):
    """All omnibus and pairwise tests, one row each."""
    rows = []

    def add(num_obstacles, mode, metric, test, comparison, n, statistic, p):
        rows.append({"num_obstacles": num_obstacles, "mode": mode, "metric": metric, "test": test,
                     "comparison": comparison, "n_maps": n, "statistic": statistic, "p": p})

    for (num_obstacles, mode) in scenarios(df):
        scenario = df[(df.num_obstacles == num_obstacles) & (df["mode"] == mode)]

        def wide(column, algorithms):
            table = scenario.pivot_table(index=PAIR_KEY, columns="algorithm", values=column)
            # reindex: an algorithm with no values at all (e.g. never succeeded) still gets a
            # column, so its maps are dropped instead of crashing
            return table.reindex(columns=algorithms).dropna()

        # Success: binary, all four algorithms.
        success = wide("success", ALGORITHMS)
        add(num_obstacles, mode, "success", "Cochran Q", "all", len(success), *cochran_q(success.values))
        for a, b in itertools.combinations(ALGORITHMS, 2):
            add(num_obstacles, mode, "success", "McNemar", f"{a} vs {b}", len(success),
                *mcnemar_exact(success[a], success[b]))

        # Path ratio: only maps where every algorithm reached the goal.
        # Nodes expanded: planners only (Q-learning does not search).
        for metric, algorithms in (("path_ratio", ALGORITHMS), ("nodes_expanded", PLANNERS)):
            values = wide(metric, algorithms)
            add(num_obstacles, mode, metric, "Friedman", "all", len(values), *friedman(values.values))
            for a, b in itertools.combinations(algorithms, 2):
                if len(values) >= 2:
                    add(num_obstacles, mode, metric, "Wilcoxon", f"{a} vs {b}", len(values),
                        *wilcoxon(values[a], values[b]))

    tests = pd.DataFrame(rows)
    tests["p_holm"] = tests["p"]
    pairwise = tests.test.isin(["McNemar", "Wilcoxon"])
    for _, family in tests[pairwise].groupby(["num_obstacles", "mode", "metric"]):
        tests.loc[family.index, "p_holm"] = holm(family.p)
    tests["significant"] = tests.p_holm < ALPHA
    return tests


# --- report --------------------------------------------------------------------

def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def scenario_name(num_obstacles, mode):
    return "no moving obstacles" if num_obstacles == 0 else f"{num_obstacles} {mode.replace('_', ' ')}"


def format_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_report(summary, tests, path):
    def pct(x):
        return f"{x:.0%}"

    def mean_text(mean, half_width, digits):
        if math.isnan(mean):
            return "-"
        if math.isnan(half_width):  # a single value has no interval
            return f"{mean:.{digits}f} (n=1)"
        return f"{mean:.{digits}f} ± {half_width:.{digits}f}"

    lines = ["# Experiment results", "",
             "Means over all grid sizes and wall densities. Brackets are 95% confidence",
             "intervals (Wilson for rates, t for means). Path ratio = moves / static optimal",
             "length, successful episodes only. Q-learning does not search, so it has no nodes.", ""]
    for (num_obstacles, mode), group in summary.groupby(["num_obstacles", "mode"], sort=False):
        lines += [f"## {scenario_name(num_obstacles, mode)}", ""]
        table_rows = []
        for row in group.itertuples():
            table_rows.append([
                row.algorithm,
                f"{pct(row.success_rate)} [{pct(row.success_ci_low)}, {pct(row.success_ci_high)}]",
                pct(row.collision_rate), pct(row.timeout_rate),
                mean_text(row.path_ratio_mean, row.path_ratio_ci, 2),
                "-" if row.algorithm == "Q-learning" else
                f"{mean_text(row.nodes_mean, row.nodes_ci, 0)} (median {row.nodes_median:.0f})",
                f"{row.planning_time_ms_mean:.2f}",
                "-" if math.isnan(row.train_time_s_mean) else f"{row.train_time_s_mean:.1f}",
            ])
        lines += [markdown_table(["Algorithm", "Success", "Collision", "Timeout", "Path ratio",
                                  "Nodes", "Decision time (ms)", "Training (s)"], table_rows), ""]

        scenario_tests = tests[(tests.num_obstacles == num_obstacles) & (tests["mode"] == mode)]
        omnibus = scenario_tests[scenario_tests.comparison == "all"]
        lines += ["Omnibus tests: " + "; ".join(
            f"{r.metric} ({r.test}, n={r.n_maps}): p={format_p(r.p)}" for r in omnibus.itertuples()), ""]
        significant = scenario_tests[scenario_tests.significant & (scenario_tests.comparison != "all")]
        if len(significant):
            lines += ["Significant pairwise differences (Holm-corrected p < 0.05): " + "; ".join(
                f"{r.metric} {r.comparison} (p={format_p(r.p_holm)})" for r in significant.itertuples()), ""]
        else:
            lines += ["No significant pairwise differences.", ""]

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# --- figures ---------------------------------------------------------------------

def style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
    axis.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    axis.title.set_color(INK)
    axis.xaxis.label.set_color(INK_SECONDARY)
    axis.yaxis.label.set_color(INK_SECONDARY)


def new_figure(columns, rows=1, width=6.0, height=4.2):
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows),
                                squeeze=False, facecolor=SURFACE)
    for axis in axes.flat:
        style_axis(axis)
    return figure, axes


def with_baseline(summary, mode):
    """Rows for one mode plus the shared 0-obstacle baseline."""
    return summary[(summary["mode"] == mode) | (summary.num_obstacles == 0)].sort_values("num_obstacles")


def plot_success_rate(summary, path):
    figure, axes = new_figure(len(MODES))
    for axis, mode in zip(axes[0], MODES):
        data = with_baseline(summary, mode)
        for algorithm in ALGORITHMS:
            rows = data[data.algorithm == algorithm]
            rate = rows.success_rate * 100
            error = [rate - rows.success_ci_low * 100, rows.success_ci_high * 100 - rate]
            axis.errorbar(rows.num_obstacles, rate, yerr=error, color=COLORS[algorithm],
                          marker=MARKERS[algorithm], markersize=8, linewidth=2, capsize=3,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, label=algorithm)
        axis.set(title=f"{mode.replace('_', ' ').capitalize()} obstacles", xlabel="Moving obstacles",
                 ylabel="Episodes reaching the goal (%)", ylim=(-3, 105),
                 xticks=sorted(data.num_obstacles.unique()))
    axes[0][0].legend(frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Success rate (95% CI)", color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_outcomes(df, path):
    counts = sorted(n for n in df.num_obstacles.unique() if n > 0)
    figure, axes = new_figure(len(counts), rows=len(MODES), width=4.2, height=3.4)
    for row_index, mode in enumerate(MODES):
        for column_index, num_obstacles in enumerate(counts):
            axis = axes[row_index][column_index]
            scenario = df[(df["mode"] == mode) & (df.num_obstacles == num_obstacles)]
            bottom = np.zeros(len(ALGORITHMS))
            for outcome, color in OUTCOME_COLORS.items():
                share = np.array([(scenario[scenario.algorithm == a].outcome == outcome).mean() * 100
                                  for a in ALGORITHMS])
                axis.bar(ALGORITHMS, share, bottom=bottom, color=color, edgecolor=SURFACE,
                         linewidth=1.5, width=0.7, label=outcome)
                bottom += share
            axis.set(title=f"{num_obstacles} {mode.replace('_', ' ')}", ylim=(0, 100))
            axis.tick_params(axis="x", labelsize=8)
            if column_index == 0:
                axis.set_ylabel("Episodes (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, labelcolor=INK_SECONDARY, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.5, 0.955))
    figure.suptitle("How episodes end: success, collision or timeout", color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_path_ratio(df, path):
    figure, axes = new_figure(len(MODES))
    for axis, mode in zip(axes[0], MODES):
        data = df[((df["mode"] == mode) | (df.num_obstacles == 0)) & (df.success == 1)]
        counts = sorted(data.num_obstacles.unique())
        width = 0.8 / len(ALGORITHMS)
        for index, algorithm in enumerate(ALGORITHMS):
            positions = [c_index + (index - 1.5) * width for c_index in range(len(counts))]
            samples = [data[(data.algorithm == algorithm) & (data.num_obstacles == c)].path_ratio
                       for c in counts]
            boxes = axis.boxplot(samples, positions=positions, widths=width * 0.85, patch_artist=True,
                                 showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
                                 whiskerprops={"color": MUTED}, capprops={"color": MUTED})
            for box in boxes["boxes"]:
                box.set(facecolor=COLORS[algorithm], edgecolor=SURFACE, linewidth=1.5)
            axis.plot([], [], marker="s", linestyle="", color=COLORS[algorithm], markersize=9,
                      label=algorithm)
        axis.set(title=f"{mode.replace('_', ' ').capitalize()} obstacles", xlabel="Moving obstacles",
                 ylabel="Moves / shortest static path", xticks=range(len(counts)), xticklabels=counts)
        axis.axhline(1, color="#c3c2b7", linewidth=1)
    axes[0][0].legend(frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Path length relative to the optimum (successful episodes, outliers hidden)", color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_nodes(df, path):
    # Node counts are very skewed (a few long waits dominate the mean), so this
    # shows the median with the interquartile range instead of mean +- CI.
    figure, axes = new_figure(len(MODES))
    for axis, mode in zip(axes[0], MODES):
        data = df[(df["mode"] == mode) | (df.num_obstacles == 0)]
        counts = sorted(data.num_obstacles.unique())
        for algorithm in PLANNERS:
            groups = [data[(data.algorithm == algorithm) & (data.num_obstacles == c)].nodes_expanded
                      for c in counts]
            median = np.array([g.median() for g in groups])
            error = [median - [g.quantile(0.25) for g in groups], [g.quantile(0.75) for g in groups] - median]
            axis.errorbar(counts, median, yerr=error, color=COLORS[algorithm],
                          marker=MARKERS[algorithm], markersize=8, linewidth=2, capsize=3,
                          markeredgecolor=SURFACE, markeredgewidth=1.5, label=algorithm)
        axis.set(title=f"{mode.replace('_', ' ').capitalize()} obstacles", xlabel="Moving obstacles",
                 ylabel="Nodes expanded per episode (log scale)", yscale="log", xticks=counts)
    axes[0][0].legend(frameon=False, labelcolor=INK_SECONDARY)
    figure.suptitle("Search effort, summed over all replans (median and interquartile range)", color=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Analyse experiment results")
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()

    df = load_results(args.results)
    os.makedirs(args.out, exist_ok=True)
    summary = summary_table(df)
    tests = statistical_tests(df)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    tests.to_csv(os.path.join(args.out, "stats.csv"), index=False)
    write_report(summary, tests, os.path.join(args.out, "report.md"))
    plot_success_rate(summary, os.path.join(args.out, "success_rate.png"))
    plot_outcomes(df, os.path.join(args.out, "outcomes.png"))
    plot_path_ratio(df, os.path.join(args.out, "path_ratio.png"))
    plot_nodes(df, os.path.join(args.out, "nodes_expanded.png"))
    print(f"Analysed {len(df)} episodes; wrote summary, stats, report and 4 figures to {args.out}")


if __name__ == "__main__":
    main()
