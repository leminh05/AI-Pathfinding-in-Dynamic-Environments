"""Tests for the analysis helpers. Run with `pytest` or `python -m tests.test_analyze`."""

import math
import os
import tempfile

import pandas as pd

from pathfinding.experiments import analyze, run


def close(a, b, tolerance=1e-3):
    return abs(a - b) < tolerance


def test_wilson_interval():
    low, high = analyze.wilson_interval(8, 10)
    assert close(low, 0.4902) and close(high, 0.9433)  # textbook values
    assert analyze.wilson_interval(0, 10)[0] == 0.0
    assert analyze.wilson_interval(10, 10)[1] == 1.0


def test_holm():
    adjusted = analyze.holm([0.01, 0.04, 0.03])
    assert [round(p, 3) for p in adjusted] == [0.03, 0.06, 0.06]


def test_cochran_q_and_mcnemar():
    same = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    assert analyze.cochran_q(same) == (math.nan, 1.0) or analyze.cochran_q(same)[1] == 1.0
    # algorithm 0 always succeeds where algorithm 1 fails: clearly different
    different = [[1, 0]] * 12 + [[1, 1]] * 5
    q, p = analyze.cochran_q(different)
    assert close(q, 12.0) and p < 0.001
    assert analyze.mcnemar_exact([r[0] for r in different], [r[1] for r in different])[1] < 0.001
    assert analyze.mcnemar_exact([1, 0, 1], [1, 0, 1]) == (0.0, 1.0)


def test_identical_samples_give_p_one():
    assert analyze.wilcoxon([1, 2, 3], [1, 2, 3])[1] == 1.0
    assert analyze.friedman([[1, 1, 1], [2, 2, 2]])[1] == 1.0


def test_full_pipeline_on_a_small_run():
    rows = []
    for seed in range(3):
        rows += run.run_task((8, 0.1, 0, "none", seed), training_episodes=30)
        for mode in analyze.MODES:
            rows += run.run_task((8, 0.1, 3, mode, seed), training_episodes=30)
    with tempfile.TemporaryDirectory() as folder:
        results = os.path.join(folder, "results.csv")
        pd.DataFrame(rows, columns=run.FIELDS).to_csv(results, index=False)
        df = analyze.load_results(results)
        summary = analyze.summary_table(df)
        tests = analyze.statistical_tests(df)
        assert len(summary) == 3 * len(analyze.ALGORITHMS)  # 3 scenarios x 4 algorithms
        assert summary.success_rate.between(0, 1).all()
        assert tests.p_holm.dropna().between(0, 1).all()

        out = os.path.join(folder, "analysis")
        os.makedirs(out)
        analyze.write_report(summary, tests, os.path.join(out, "report.md"))
        analyze.plot_success_rate(summary, os.path.join(out, "a.png"))
        analyze.plot_outcomes(df, os.path.join(out, "b.png"))
        analyze.plot_path_ratio(df, os.path.join(out, "c.png"))
        analyze.plot_nodes(df, os.path.join(out, "d.png"))
        assert len(os.listdir(out)) == 5


if __name__ == "__main__":
    for test in (test_wilson_interval, test_holm, test_cochran_q_and_mcnemar,
                 test_identical_samples_give_p_one, test_full_pipeline_on_a_small_run):
        test()
    print("all analysis tests passed")
