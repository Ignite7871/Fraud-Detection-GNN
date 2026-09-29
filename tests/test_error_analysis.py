import numpy as np

from src.error_analysis import (
    brier_score,
    compute_error_report,
    expected_calibration_error,
    full_error_report,
    probability_distribution_summary,
    reliability_curve,
)


def test_compute_error_report_counts_and_rates():

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.4])

    report = compute_error_report(y_true, y_pred, y_prob)

    assert report["tn"] == 2
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert report["tp"] == 2

    assert report["n_negative"] == 3
    assert report["n_positive"] == 3

    assert report["false_positive_rate"] == 1 / 3
    assert report["false_negative_rate"] == 1 / 3

    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0
    assert 0.0 <= report["f1"] <= 1.0
    assert 0.0 <= report["roc_auc"] <= 1.0


def test_probability_distribution_summary_separates_classes():

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    summary = probability_distribution_summary(y_true, y_prob)

    assert summary["licit"]["count"] == 2
    assert summary["illicit"]["count"] == 2

    assert summary["licit"]["mean"] < summary["illicit"]["mean"]


def test_probability_distribution_summary_handles_empty_class():

    y_true = np.array([0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3])

    summary = probability_distribution_summary(y_true, y_prob)

    assert summary["illicit"]["count"] == 0
    assert np.isnan(summary["illicit"]["mean"])


def test_brier_score_perfect_predictions_is_zero():

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])

    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_matches_manual_computation():

    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.7])

    expected = np.mean(
        (y_prob - y_true) ** 2
    )

    assert brier_score(y_true, y_prob) == expected


def test_reliability_curve_bins_and_counts():

    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.85, 0.9, 0.95, 0.8])

    curve = reliability_curve(y_true, y_prob, n_bins=10)

    assert curve["bin_counts"].sum() == 6
    assert len(curve["bin_edges"]) == 11

    # low-probability bin should have low observed accuracy
    assert curve["bin_accuracy"][0] == 0.0

    # high-probability bin should have high observed accuracy
    assert curve["bin_accuracy"][8] == 1.0


def test_expected_calibration_error_perfect_calibration_is_zero():

    # In each bin, the mean predicted probability exactly matches the
    # observed positive fraction.
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    assert ece == 0.0


def test_expected_calibration_error_detects_miscalibration():

    # All predictions are confidently wrong: predicted ~1.0 but the
    # true label is always 0.
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.95, 0.97, 0.96, 0.98])

    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    assert ece > 0.9


def test_full_error_report_combines_all_pieces():

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.2, 0.6, 0.7, 0.9])

    report = full_error_report(y_true, y_pred, y_prob, n_bins=5)

    assert "confusion_matrix" in report
    assert "probability_distribution" in report
    assert "brier_score" in report
    assert "expected_calibration_error" in report

    assert report["probability_distribution"]["licit"]["count"] == 2
    assert report["probability_distribution"]["illicit"]["count"] == 2
