import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def run_logistic_regression(
    X: np.ndarray,
    y_full: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Train and evaluate the Logistic Regression baseline."""

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = {
        "val": {
            "accuracy": accuracy_score(y_val, val_pred),
            "f1": f1_score(y_val, val_pred),
            "auc": roc_auc_score(y_val, val_prob),
        },
        "test": {
            "accuracy": accuracy_score(y_test, test_pred),
            "f1": f1_score(y_test, test_pred),
            "auc": roc_auc_score(y_test, test_prob),
        },
    }

    return model, metrics
