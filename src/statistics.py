import itertools

import numpy as np


def bootstrap_mean_ci(
    differences,
    confidence=0.95,
    n_resamples=10000,
    seed=0,
):
    """
    Bootstrap confidence interval for the mean paired difference.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if differences.size == 0:
        raise ValueError("differences cannot be empty.")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        differences,
        size=(
            n_resamples,
            differences.size,
        ),
        replace=True,
    )

    means = samples.mean(axis=1)

    alpha = 1.0 - confidence

    lower = float(
        np.quantile(means, alpha / 2)
    )

    upper = float(
        np.quantile(means, 1.0 - alpha / 2)
    )

    return lower, upper


def exact_sign_flip_pvalue(
    differences,
):
    """
    Exact two-sided paired permutation test.

    Under the null, each paired difference may independently
    have either sign. For n <= 20 this enumerates every
    possible sign assignment.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        differences != 0
    ]

    if differences.size == 0:
        return 1.0

    observed = abs(
        differences.mean()
    )

    values = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=differences.size,
    ):
        signs = np.asarray(signs)
        statistic = abs(
            np.mean(
                differences * signs
            )
        )
        values.append(statistic)

    values = np.asarray(values)

    p_value = np.mean(
        values >= observed
    )

    return float(p_value)


def paired_effect_size(
    differences,
):
    """
    Standardized paired effect size (Cohen's dz):

        mean(difference) / std(difference)

    using the sample standard deviation.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if differences.size < 2:
        raise ValueError(
            "At least two paired observations are required."
        )

    std = differences.std(
        ddof=1
    )

    if std == 0:
        return np.inf if differences.mean() != 0 else 0.0

    return float(
        differences.mean() / std
    )


def paired_statistics(
    differences,
    confidence=0.95,
    n_resamples=10000,
    seed=0,
):
    """
    Summarize a set of seed-matched paired differences.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if differences.size == 0:
        raise ValueError(
            "differences cannot be empty."
        )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            differences,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
        )
    )

    return {
        "n": int(differences.size),
        "mean": float(differences.mean()),
        "std": float(
            differences.std(ddof=1)
        ),
        "median": float(
            np.median(differences)
        ),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": exact_sign_flip_pvalue(
            differences
        ),
        "effect_size_dz": paired_effect_size(
            differences
        ),
        "fraction_positive": float(
            np.mean(differences > 0)
        ),
        "fraction_negative": float(
            np.mean(differences < 0)
        ),
    }


def holm_correction(p_values):
    """
    Holm-Bonferroni multiple-comparison correction.

    Returns adjusted p-values in the original order.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted_value = (
            (m - rank)
            * p_values[index]
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted[index] = min(
            running_max,
            1.0,
        )

    return adjusted
