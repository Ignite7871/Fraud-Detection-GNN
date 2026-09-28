import numpy as np

from src.statistics import (
    bootstrap_mean_ci,
    exact_sign_flip_pvalue,
    holm_correction,
    paired_effect_size,
    paired_statistics,
)


def test_bootstrap_mean_ci():
    differences = np.array(
        [1.0, 2.0, 3.0]
    )

    low, high = bootstrap_mean_ci(
        differences,
        n_resamples=1000,
        seed=42,
    )

    assert low <= high
    assert low <= 2.0 <= high


def test_exact_sign_flip_pvalue():
    differences = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )

    p_value = exact_sign_flip_pvalue(
        differences
    )

    assert 0.0 < p_value <= 1.0


def test_paired_effect_size():
    differences = np.array(
        [1.0, 2.0, 3.0]
    )

    effect = paired_effect_size(
        differences
    )

    assert effect > 0.0


def test_paired_statistics():
    differences = np.array(
        [
            0.10,
            0.12,
            0.11,
            0.09,
            0.13,
        ]
    )

    stats = paired_statistics(
        differences,
        n_resamples=1000,
        seed=42,
    )

    assert stats["n"] == 5
    assert stats["mean"] > 0
    assert stats["median"] > 0
    assert stats["fraction_positive"] == 1.0


def test_holm_correction():
    p_values = np.array(
        [
            0.01,
            0.04,
            0.20,
        ]
    )

    corrected = holm_correction(
        p_values
    )

    assert len(corrected) == 3
    assert np.all(
        corrected >= p_values
    )
