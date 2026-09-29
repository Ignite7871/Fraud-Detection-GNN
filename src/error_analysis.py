import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_error_report(y_true, y_pred, y_prob):
    """
    Confusion-matrix-based error report for a binary classifier.

    ROC-AUC is threshold-independent (computed from y_prob); every other
    metric here is threshold-dependent (computed from y_pred, the
    already-thresholded predictions the caller supplies).
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_negative = int(tn + fp)
    n_positive = int(tp + fn)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": auc,
        "false_positive_rate": (
            float(fp / n_negative) if n_negative > 0 else float("nan")
        ),
        "false_negative_rate": (
            float(fn / n_positive) if n_positive > 0 else float("nan")
        ),
        "n_negative": n_negative,
        "n_positive": n_positive,
    }


def probability_distribution_summary(y_true, y_prob):
    """
    Summary statistics of predicted probabilities, split by true class
    (licit = 0, illicit = 1).
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    licit_scores = y_prob[y_true == 0]
    illicit_scores = y_prob[y_true == 1]

    def _summary(values):
        if values.size == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "p25": float("nan"),
                "p75": float("nan"),
            }

        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": (
                float(values.std(ddof=1))
                if values.size > 1
                else 0.0
            ),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }

    return {
        "licit": _summary(licit_scores),
        "illicit": _summary(illicit_scores),
    }


def brier_score(y_true, y_prob):
    """Brier score (mean squared error between probability and outcome)."""

    return float(
        brier_score_loss(
            np.asarray(y_true),
            np.asarray(y_prob, dtype=float),
        )
    )


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """
    Bin predictions into n_bins equal-width bins over [0, 1] and compute,
    for each non-empty bin, the mean predicted probability (confidence)
    and the observed positive fraction (accuracy), for a
    calibration/reliability plot.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)

    bin_confidence = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_counts[b] = count

        if count > 0:
            bin_confidence[b] = float(y_prob[mask].mean())
            bin_accuracy[b] = float(y_true[mask].mean())

    return {
        "bin_edges": bin_edges,
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    """
    Expected Calibration Error (ECE): the bin-count-weighted average of
    |accuracy - confidence| across the reliability-curve bins. Empty
    bins are skipped rather than treated as zero error.
    """

    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    total = int(curve["bin_counts"].sum())

    if total == 0:
        return float("nan")

    weighted_error = 0.0

    for count, confidence, accuracy in zip(
        curve["bin_counts"],
        curve["bin_confidence"],
        curve["bin_accuracy"],
    ):
        if count == 0:
            continue

        weighted_error += (count / total) * abs(
            accuracy - confidence
        )

    return float(weighted_error)


def full_error_report(y_true, y_pred, y_prob, n_bins: int = 10):
    """
    Combine compute_error_report, probability_distribution_summary,
    brier_score, and expected_calibration_error into a single report,
    to avoid re-deriving the same pieces separately for every model in
    the notebook.
    """

    report = compute_error_report(y_true, y_pred, y_prob)
    report["probability_distribution"] = probability_distribution_summary(
        y_true, y_prob
    )
    report["brier_score"] = brier_score(y_true, y_prob)
    report["expected_calibration_error"] = expected_calibration_error(
        y_true, y_prob, n_bins=n_bins
    )

    return report
