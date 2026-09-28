import numpy as np
import pandas as pd


def summarize_runs(
    results: pd.DataFrame,
    group_columns=("Protocol", "Configuration"),
):
    """
    Aggregate repeated-run results using mean and standard deviation.
    """

    metrics = ["Accuracy", "F1", "ROC-AUC"]

    summary = (
        results
        .groupby(list(group_columns))[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary


def paired_metric_differences(
    results: pd.DataFrame,
    protocol: str,
    metric: str,
    configuration_a: str,
    configuration_b: str,
):
    """
    Compute seed-matched differences:

        configuration_a - configuration_b

    for each seed.
    """

    subset = results[
        results["Protocol"].eq(protocol)
        & results["Configuration"].isin(
            [configuration_a, configuration_b]
        )
    ]

    pivot = subset.pivot(
        index="Seed",
        columns="Configuration",
        values=metric,
    )

    differences = (
        pivot[configuration_a]
        - pivot[configuration_b]
    )

    return differences.dropna()


def mean_std(values):
    """
    Return mean and sample standard deviation.
    """

    values = np.asarray(values, dtype=float)

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }
