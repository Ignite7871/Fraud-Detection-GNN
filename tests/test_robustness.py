import pandas as pd

from src.robustness import (
    mean_std,
    paired_metric_differences,
    summarize_runs,
)


def test_mean_std():

    stats = mean_std([1.0, 2.0, 3.0])

    assert stats["mean"] == 2.0
    assert stats["std"] > 0.0


def make_results_df():

    rows = [
        {
            "Protocol": "Random",
            "Seed": 1,
            "Configuration": "A",
            "Accuracy": 0.9,
            "F1": 0.8,
            "ROC-AUC": 0.95,
        },
        {
            "Protocol": "Random",
            "Seed": 2,
            "Configuration": "A",
            "Accuracy": 0.8,
            "F1": 0.7,
            "ROC-AUC": 0.90,
        },
        {
            "Protocol": "Random",
            "Seed": 1,
            "Configuration": "B",
            "Accuracy": 0.7,
            "F1": 0.6,
            "ROC-AUC": 0.80,
        },
        {
            "Protocol": "Random",
            "Seed": 2,
            "Configuration": "B",
            "Accuracy": 0.6,
            "F1": 0.5,
            "ROC-AUC": 0.75,
        },
    ]

    return pd.DataFrame(rows)


def test_summarize_runs_shape():

    results = make_results_df()

    summary = summarize_runs(results)

    # one row per (Protocol, Configuration) combination
    assert len(summary) == 2

    top_level_columns = set(
        summary.columns.get_level_values(0)
    )

    assert "Protocol" in top_level_columns
    assert "Configuration" in top_level_columns
    assert "Accuracy" in top_level_columns
    assert "F1" in top_level_columns
    assert "ROC-AUC" in top_level_columns


def test_paired_metric_differences():

    results = make_results_df()

    differences = paired_metric_differences(
        results,
        protocol="Random",
        metric="ROC-AUC",
        configuration_a="A",
        configuration_b="B",
    )

    assert len(differences) == 2

    assert differences.loc[1] == 0.95 - 0.80
    assert differences.loc[2] == 0.90 - 0.75


def test_paired_metric_differences_filters_by_protocol():

    results = make_results_df()

    extra = pd.DataFrame(
        [
            {
                "Protocol": "Temporal",
                "Seed": 1,
                "Configuration": "A",
                "Accuracy": 0.5,
                "F1": 0.4,
                "ROC-AUC": 0.6,
            },
            {
                "Protocol": "Temporal",
                "Seed": 1,
                "Configuration": "B",
                "Accuracy": 0.4,
                "F1": 0.3,
                "ROC-AUC": 0.5,
            },
        ]
    )

    combined = pd.concat(
        [results, extra],
        ignore_index=True,
    )

    differences = paired_metric_differences(
        combined,
        protocol="Random",
        metric="ROC-AUC",
        configuration_a="A",
        configuration_b="B",
    )

    # only the two Random-protocol seeds should be present
    assert len(differences) == 2
    assert set(differences.index) == {1, 2}
